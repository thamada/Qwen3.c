/*
 * Flash-Attention decode diagnostics (tensor stats + CPU replay check).
 */

#include "fa_debug.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fa_debug = 0;
static char g_fa_backend[32] = "GPU";

static const int g_watch_pos[] = { 27, 28, 29, 30, 31, 32, 33, 34 };
static const int g_watch_pos_n =
    (int)(sizeof(g_watch_pos) / sizeof(g_watch_pos[0]));

static const int g_watch_layers[] = { 0, 17, 35 };
static const int g_watch_layers_n = 3;

void fa_debug_set_enabled(int on) { g_fa_debug = on ? 1 : 0; }
int fa_debug_enabled(void) { return g_fa_debug; }

void fa_debug_set_backend_label(const char *label)
{
    if (!label) {
        g_fa_backend[0] = '\0';
        return;
    }
    snprintf(g_fa_backend, sizeof(g_fa_backend), "%s", label);
}

static int pos_watched(int pos)
{
    for (int i = 0; i < g_watch_pos_n; i++)
        if (g_watch_pos[i] == pos) return 1;
    return 0;
}

static int layer_watched(int layer, int n_layers)
{
    for (int i = 0; i < g_watch_layers_n; i++) {
        if (g_watch_layers[i] >= 0 && g_watch_layers[i] == layer) return 1;
        if (g_watch_layers[i] < 0 && layer == n_layers + g_watch_layers[i])
            return 1;
    }
    return 0;
}

static void stats_host_f32(const float *h, int n, FaTensorStats *s)
{
    s->max_abs = 0.f;
    s->min_val = 0.f;
    s->mean_abs = 0.f;
    s->nan_count = 0;
    s->inf_count = 0;
    if (!h || n <= 0) return;

    int first = 1;
    double sum_abs = 0.0;
    for (int i = 0; i < n; i++) {
        float v = h[i];
        if (isnan(v)) { s->nan_count++; continue; }
        if (isinf(v)) { s->inf_count++; continue; }
        float a = fabsf(v);
        sum_abs += (double)a;
        if (first) { s->min_val = v; first = 0; }
        else if (v < s->min_val) s->min_val = v;
        if (a > s->max_abs) s->max_abs = a;
    }
    int valid = n - s->nan_count - s->inf_count;
    if (valid > 0) s->mean_abs = (float)(sum_abs / (double)valid);
}

static void stats_device_f32(const float *d, int n, const char *tag)
{
    FaTensorStats st;
    float *h = (float *)malloc((size_t)n * sizeof(float));
    if (!h) {
        fprintf(stderr, "fa_debug: OOM stats %s n=%d\n", tag, n);
        return;
    }
    cudaError_t err = cudaMemcpy(h, d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "fa_debug: memcpy %s: %s\n", tag, cudaGetErrorString(err));
        free(h);
        return;
    }
    stats_host_f32(h, n, &st);
    printf("    %-22s n=%6d max=%.4e mean=%.4e min=%.4e nan=%d inf=%d\n",
           tag, n, st.max_abs, st.mean_abs, st.min_val, st.nan_count, st.inf_count);
    free(h);
}

#ifndef FA_BR
#define FA_BR 64
#endif
#ifndef FA_HD
#define FA_HD 128
#endif

static void fa_cpu_attn_head0(const float *q, const float *kc, const float *vc,
                              int npos, int hd, int kv_dim, int kv_mul, float scale,
                              float *out)
{
    (void)kv_mul;
    const float *qh = q;
    const float *kbase = kc;
    const float *vbase = vc;

    float m = -1e30f;
    float l = 0.f;
    float o_sh[FA_HD];
    for (int d = 0; d < hd; d++) o_sh[d] = 0.f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        float scores[FA_BR];
        for (int j = 0; j < tc; j++) {
            float s = 0.f;
            for (int d = 0; d < hd; d++)
                s += qh[d] * kbase[(size_t)(t0 + j) * kv_dim + d];
            scores[j] = s * scale;
        }

        float m_tile = -1e30f;
        for (int j = 0; j < tc; j++)
            if (scores[j] > m_tile) m_tile = scores[j];

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.f;
        for (int d = 0; d < hd; d++) o_sh[d] *= alpha;

        for (int j = 0; j < tc; j++)
            scores[j] = expf(scores[j] - m_new);

        float l_tile = 0.f;
        for (int j = 0; j < tc; j++) l_tile += scores[j];

        for (int d = 0; d < hd; d++) {
            float acc = 0.f;
            for (int j = 0; j < tc; j++)
                acc += scores[j] * vbase[(size_t)(t0 + j) * kv_dim + d];
            o_sh[d] += acc;
        }

        l = l * alpha + l_tile;
        m = m_new;
    }

    for (int d = 0; d < hd; d++)
        out[d] = (l > 0.f) ? (o_sh[d] / l) : 0.f;
}

static void fa_debug_replay_head0(const float *d_q, const float *d_xb,
                                  const float *d_kc, const float *d_vc,
                                  int npos, int hd, int kv_dim, int kv_mul,
                                  float scale)
{
    float *h_q = (float *)malloc((size_t)hd * sizeof(float));
    float *h_xb = (float *)malloc((size_t)hd * sizeof(float));
    float *h_kc = (float *)malloc((size_t)npos * kv_dim * sizeof(float));
    float *h_vc = (float *)malloc((size_t)npos * kv_dim * sizeof(float));
    float *h_ref = (float *)malloc((size_t)hd * sizeof(float));
    if (!h_q || !h_xb || !h_kc || !h_vc || !h_ref) {
        free(h_q); free(h_xb); free(h_kc); free(h_vc); free(h_ref);
        return;
    }

    cudaMemcpy(h_q, d_q, (size_t)hd * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xb, d_xb, (size_t)hd * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_kc, d_kc, (size_t)npos * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vc, d_vc, (size_t)npos * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    fa_cpu_attn_head0(h_q, h_kc, h_vc, npos, hd, kv_dim, kv_mul, scale, h_ref);

    float max_diff = 0.f;
    for (int d = 0; d < hd; d++) {
        float diff = fabsf(h_ref[d] - h_xb[d]);
        if (diff > max_diff) max_diff = diff;
    }
    float ref_norm = 0.f;
    for (int d = 0; d < hd; d++) ref_norm = fmaxf(ref_norm, fabsf(h_ref[d]));

    int last_t0 = (npos > 0) ? ((npos - 1) / FA_BR) * FA_BR : 0;
    int last_tc = npos - last_t0;
    printf("    FA_replay head0: npos=%d FA_BR=%d tiles=%d last_t0=%d last_tc=%d "
           "max|cpu-gpu|=%.4e ratio=%.4e\n",
           npos, FA_BR, (npos + FA_BR - 1) / FA_BR, last_t0, last_tc,
           max_diff, ref_norm > 0.f ? max_diff / ref_norm : max_diff);

    free(h_q); free(h_xb); free(h_kc); free(h_vc); free(h_ref);
}

void fa_debug_decode_hook(int prefill_len, int pos, int layer, int n_layers,
                          int npos, int kv_dim, int hd, int n_heads, int n_kv,
                          int kv_mul, float scale,
                          const float *d_q, const float *d_k, const float *d_v,
                          const float *d_xb, const float *d_kc, const float *d_vc)
{
    if (!g_fa_debug) return;
    if (!pos_watched(pos)) return;
    if (!layer_watched(layer, n_layers)) return;

    printf("FA_DEBUG [%s] decode pos=%d npos=%d prefill_len=%d layer=%d/%d\n",
           g_fa_backend, pos, npos, prefill_len, layer, n_layers);

    stats_device_f32(d_q, n_heads * hd, "q_rope");
    stats_device_f32(d_k, n_kv * hd, "k_rope");
    stats_device_f32(d_v, n_kv * hd, "v_cur");

    const int kv_slots[] = { 0, prefill_len - 1, 28, 29, 30, 31, pos };
    for (size_t si = 0; si < sizeof(kv_slots) / sizeof(kv_slots[0]); si++) {
        int t = kv_slots[si];
        if (t < 0 || t >= pos) continue;
        char tag[32];
        snprintf(tag, sizeof(tag), "kc[t=%d]", t);
        stats_device_f32(d_kc + (size_t)t * kv_dim, kv_dim, tag);
        snprintf(tag, sizeof(tag), "vc[t=%d]", t);
        stats_device_f32(d_vc + (size_t)t * kv_dim, kv_dim, tag);
    }

    stats_device_f32(d_xb, n_heads * hd, "xb_attn");

    if (layer == 0)
        fa_debug_replay_head0(d_q, d_xb, d_kc, d_vc, npos, hd, kv_dim, kv_mul, scale);

    fflush(stdout);
}

static void stats_kv_slots(const float *d_kc, const float *d_vc,
                           int kv_dim, int hd, int n_tokens, const char *prefix)
{
    const int slots[] = { 0, n_tokens / 2, n_tokens - 1 };
    for (size_t si = 0; si < sizeof(slots) / sizeof(slots[0]); si++) {
        int t = slots[si];
        if (t < 0 || t >= n_tokens) continue;
        char tag[48];
        snprintf(tag, sizeof(tag), "%s kc[t=%d]", prefix, t);
        stats_device_f32(d_kc + (size_t)t * kv_dim, kv_dim, tag);
        snprintf(tag, sizeof(tag), "%s vc[t=%d]", prefix, t);
        stats_device_f32(d_vc + (size_t)t * kv_dim, kv_dim, tag);
        snprintf(tag, sizeof(tag), "%s k_h0[t=%d]", prefix, t);
        stats_device_f32(d_kc + (size_t)t * kv_dim, hd, tag);
    }
}

void fa_debug_prefill_layer_hook(int layer, const char *phase,
                                 int n_tokens, int dim, int kv_dim,
                                 const float *d_xb_norm, const float *d_k_wk)
{
    if (!g_fa_debug) return;

    printf("FA_DEBUG [%s] prefill L%d %s\n", g_fa_backend, layer, phase ? phase : "");
    const int slots[] = { 0, n_tokens / 2, n_tokens - 1 };
    for (size_t si = 0; si < sizeof(slots) / sizeof(slots[0]); si++) {
        int t = slots[si];
        if (t < 0 || t >= n_tokens) continue;
        char tag[40];
        if (d_xb_norm) {
            snprintf(tag, sizeof(tag), "xb[t=%d]", t);
            stats_device_f32(d_xb_norm + (size_t)t * dim, dim, tag);
        }
        if (d_k_wk) {
            snprintf(tag, sizeof(tag), "k_wk[t=%d]", t);
            stats_device_f32(d_k_wk + (size_t)t * kv_dim, kv_dim, tag);
        }
    }
    fflush(stdout);
}

void fa_debug_prefill_x_row0_tag(int layer, const char *sub, int dim,
                               const float *d_x_batch)
{
    if (!g_fa_debug) return;
    char tag[40];
    if (sub && sub[0])
        snprintf(tag, sizeof(tag), "x[t=0] L%d %s", layer, sub);
    else
        snprintf(tag, sizeof(tag), "x[t=0] L%d end", layer);
    stats_device_f32(d_x_batch, dim, tag);
    fflush(stdout);
}

void fa_debug_prefill_hb_row0_tag(int layer, const char *sub, int hidden,
                                  const float *d_hb_batch)
{
    if (!g_fa_debug) return;
    const int slots[] = { 0, 10, 19 };
    for (size_t si = 0; si < sizeof(slots) / sizeof(slots[0]); si++) {
        int t = slots[si];
        char tag[40];
        snprintf(tag, sizeof(tag), "hb[t=%d] L%d %s", t, layer, sub ? sub : "");
        stats_device_f32(d_hb_batch + (size_t)t * hidden, hidden, tag);
    }
    fflush(stdout);
}

void fa_debug_prefill_x_row0(int layer, int dim, const float *d_x_batch)
{
    fa_debug_prefill_x_row0_tag(layer, "end", dim, d_x_batch);
}

void fa_debug_prefill_kv_layer0_hook(int n_tokens, int kv_dim, int hd,
                                       const float *d_kc_l0, const float *d_vc_l0)
{
    if (!g_fa_debug) return;

    printf("FA_DEBUG [%s] prefill layer0 after kv_write\n", g_fa_backend);
    stats_kv_slots(d_kc_l0, d_vc_l0, kv_dim, hd, n_tokens, "L0");
    fflush(stdout);
}

void fa_debug_prefill_hook(int n_tokens, int kv_dim, int hd, int max_seq,
                           int n_layers, const float *d_kc_base,
                           const float *d_vc_base)
{
    if (!g_fa_debug) return;

    printf("FA_DEBUG [%s] prefill done n_tokens=%d FA_BR=%d\n",
           g_fa_backend, n_tokens, FA_BR);
    stats_kv_slots(d_kc_base, d_vc_base, kv_dim, hd, n_tokens, "L0");

    const int layers[] = { 17, n_layers - 1 };
    for (size_t li = 0; li < sizeof(layers) / sizeof(layers[0]); li++) {
        int l = layers[li];
        if (l <= 0 || l >= n_layers) continue;
        size_t loff = (size_t)l * (size_t)max_seq * (size_t)kv_dim;
        char prefix[16];
        snprintf(prefix, sizeof(prefix), "L%d", l);
        stats_kv_slots(d_kc_base + loff, d_vc_base + loff, kv_dim, hd, n_tokens, prefix);
    }
    fflush(stdout);
}
