/*
 * Qwen3-VL GPU forward — Vulkan compute backend.
 * FP16 GEMV linear layers + Flash Attention (online softmax, GQA).
 */

#include "gpu.h"
#include "vk_alloc.h"
#include "vk_context.h"
#include "vk_pipeline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DT_F32 0
#define DT_F16 1

typedef struct { void *ptr; } DevBuf;
typedef struct { void **layer; int n_layers; } DevLayerBuf;

struct GpuModel {
    GpuConfig cfg;
    DevBuf embd;
    int embd_t;
    DevLayerBuf norm_att, q_norm, k_norm, norm_ffn;
    DevLayerBuf wq, wk, wv, wo, gate, up, down;
    DevBuf norm_out;
    DevBuf out;
    int out_t;
    float *x, *xb, *xb2, *hb, *hb2;
    float *q, *k, *v, *logits;
    float *kc, *vc;
    float *x_batch, *xb_batch, *xb2_batch;
    float *q_batch, *k_batch, *v_batch;
    float *hb_batch, *hb2_batch;
    int *tokens_dev;
    int batch_cap;
    int prefill_len;
};

static VkCtx *g_ctx;
static char   g_shader_dir[512];

static DevLayerBuf dev_adopt_layers(int n_layers, void **ptrs)
{
    DevLayerBuf lb = { NULL, n_layers };
    lb.layer = (void **)malloc((size_t)n_layers * sizeof(void *));
    if (ptrs)
        memcpy(lb.layer, ptrs, (size_t)n_layers * sizeof(void *));
    return lb;
}

static VkBuffer vkb(const void *p) { return vk_ptr_buffer(p); }

static void disp1(VkPipeKind k, VkBuffer b0, const void *pc, uint32_t pcs, uint32_t gx)
{
    VkBuffer bufs[1] = { b0 };
    vk_dispatch(g_ctx, k, bufs, 1, pc, pcs, gx, 1, 1);
}

static void disp2(VkPipeKind k, VkBuffer b0, VkBuffer b1, const void *pc, uint32_t pcs, uint32_t gx)
{
    VkBuffer bufs[2] = { b0, b1 };
    vk_dispatch(g_ctx, k, bufs, 2, pc, pcs, gx, 1, 1);
}

static void disp3(VkPipeKind k, VkBuffer b0, VkBuffer b1, VkBuffer b2,
                  const void *pc, uint32_t pcs, uint32_t gx, uint32_t gy)
{
    VkBuffer bufs[3] = { b0, b1, b2 };
    vk_dispatch(g_ctx, k, bufs, 3, pc, pcs, gx, gy ? gy : 1, 1);
}

static void disp4(VkPipeKind k, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3,
                  const void *pc, uint32_t pcs, uint32_t gx)
{
    VkBuffer bufs[4] = { b0, b1, b2, b3 };
    vk_dispatch(g_ctx, k, bufs, 4, pc, pcs, gx, 1, 1);
}

static void launch_mm_f16(float *o, const float *x, const uint16_t *w, int n, int d)
{
    struct { int n; int d; } pc = { n, d };
    disp3(VKP_MM_F16_GEMV, vkb(o), vkb(x), vkb(w), &pc, sizeof(pc),
          (uint32_t)((d + 255) / 256), 1);
}

static void launch_mm_f16_batch(float *o, const float *x, const uint16_t *w,
                                int n, int d, int n_tokens)
{
    struct { int n; int d; int n_tokens; } pc = { n, d, n_tokens };
    disp3(VKP_MM_F16_GEMV_BATCH, vkb(o), vkb(x), vkb(w), &pc, sizeof(pc),
          (uint32_t)((n_tokens * d + 255) / 256), 1);
}

static void gpu_mm(float *o, const float *x, const DevLayerBuf *W, int wl,
                   int n, int d, int type)
{
    if (type == DT_F16)
        launch_mm_f16(o, x, (const uint16_t *)W->layer[wl], n, d);
    else {
        fprintf(stderr, "gpu_mm: unsupported weight type %d\n", type);
        exit(1);
    }
}

static void gpu_mm_batch(float *o, const float *x, const DevLayerBuf *W, int wl,
                         int n, int d, int type, int n_tokens)
{
    if (type == DT_F16)
        launch_mm_f16_batch(o, x, (const uint16_t *)W->layer[wl], n, d, n_tokens);
    else {
        fprintf(stderr, "gpu_mm_batch: unsupported weight type %d\n", type);
        exit(1);
    }
}

static void gpu_emb_lookup(GpuModel *gm, int token)
{
    struct { int id; int dim; } pc = { token, gm->cfg.dim };
    disp2(VKP_EMB_F16, vkb(gm->x), vkb(gm->embd.ptr), &pc, sizeof(pc),
          (uint32_t)((gm->cfg.dim + 255) / 256));
}

static void gpu_mm_out(float *o, const float *x, GpuModel *gm, int n, int d)
{
    if (gm->out_t == DT_F16)
        launch_mm_f16(o, x, (const uint16_t *)gm->out.ptr, n, d);
    else {
        fprintf(stderr, "gpu_mm_out: unsupported weight type %d\n", gm->out_t);
        exit(1);
    }
}

static void resolve_shader_dir(void)
{
    if (g_shader_dir[0]) return;
    const char *env = getenv("QWEN3_VK_SHADER_DIR");
    if (env && env[0]) {
        snprintf(g_shader_dir, sizeof g_shader_dir, "%s", env);
        return;
    }
    snprintf(g_shader_dir, sizeof g_shader_dir, "shaders");
}

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host)
{
    resolve_shader_dir();
    vk_device_init();
    g_ctx = vk_device_ctx();
    vk_pipelines_init(g_ctx, g_shader_dir);

    GpuModel *gm = (GpuModel *)calloc(1, sizeof(GpuModel));
    gm->cfg = *cfg;

    const int L = cfg->n_layers;
    const int dim = cfg->dim;
    const int hidden = cfg->hidden_dim;
    const int kv_dim = cfg->kv_dim;
    const int vocab = cfg->vocab_size;
    const int max_seq = cfg->max_seq;
    const int qdim = cfg->n_heads * cfg->head_dim;

    gm->norm_att = dev_adopt_layers(L, host->norm_att);
    gm->q_norm   = dev_adopt_layers(L, host->q_norm);
    gm->k_norm   = dev_adopt_layers(L, host->k_norm);
    gm->norm_ffn = dev_adopt_layers(L, host->norm_ffn);
    gm->wq       = dev_adopt_layers(L, host->wq);
    gm->wk       = dev_adopt_layers(L, host->wk);
    gm->wv       = dev_adopt_layers(L, host->wv);
    gm->wo       = dev_adopt_layers(L, host->wo);
    gm->gate     = dev_adopt_layers(L, host->gate);
    gm->up       = dev_adopt_layers(L, host->up);
    gm->down     = dev_adopt_layers(L, host->down);

    gm->embd.ptr = host->embd;
    gm->embd_t = host->embd_t;
    gm->out.ptr = host->out;
    gm->out_t = host->out_t;
    gm->norm_out.ptr = host->norm_out;

    gm->x      = (float *)vk_malloc((size_t)dim * sizeof(float));
    gm->xb     = (float *)vk_malloc((size_t)dim * sizeof(float));
    gm->xb2    = (float *)vk_malloc((size_t)dim * sizeof(float));
    gm->hb     = (float *)vk_malloc((size_t)hidden * sizeof(float));
    gm->hb2    = (float *)vk_malloc((size_t)hidden * sizeof(float));
    gm->q      = (float *)vk_malloc((size_t)qdim * sizeof(float));
    gm->k      = (float *)vk_malloc((size_t)kv_dim * sizeof(float));
    gm->v      = (float *)vk_malloc((size_t)kv_dim * sizeof(float));
    gm->logits = (float *)vk_malloc((size_t)vocab * sizeof(float));
    gm->kc     = (float *)vk_malloc((size_t)L * max_seq * kv_dim * sizeof(float));
    gm->vc     = (float *)vk_malloc((size_t)L * max_seq * kv_dim * sizeof(float));

    gm->batch_cap = max_seq;
    {
        size_t bc = (size_t)max_seq;
        gm->x_batch   = (float *)vk_malloc(bc * (size_t)dim * sizeof(float));
        gm->xb_batch  = (float *)vk_malloc(bc * (size_t)dim * sizeof(float));
        gm->xb2_batch = (float *)vk_malloc(bc * (size_t)dim * sizeof(float));
        gm->q_batch   = (float *)vk_malloc(bc * (size_t)qdim * sizeof(float));
        gm->k_batch   = (float *)vk_malloc(bc * (size_t)kv_dim * sizeof(float));
        gm->v_batch   = (float *)vk_malloc(bc * (size_t)kv_dim * sizeof(float));
        gm->hb_batch  = (float *)vk_malloc(bc * (size_t)hidden * sizeof(float));
        gm->hb2_batch = (float *)vk_malloc(bc * (size_t)hidden * sizeof(float));
        gm->tokens_dev = (int *)vk_malloc(bc * sizeof(int));
    }

    vk_device_sync();
    printf("Vulkan GPU: model ready (%d layers, dim=%d, vocab=%d)\n", L, dim, vocab);
    return gm;
}

void gpu_model_destroy(GpuModel *gm)
{
    if (!gm) return;
    /* weights freed by main.c via vk_free on host pointers */
    free(gm->norm_att.layer);
    free(gm->q_norm.layer);
    free(gm->k_norm.layer);
    free(gm->norm_ffn.layer);
    free(gm->wq.layer);
    free(gm->wk.layer);
    free(gm->wv.layer);
    free(gm->wo.layer);
    free(gm->gate.layer);
    free(gm->up.layer);
    free(gm->down.layer);

    vk_free(gm->x);
    vk_free(gm->xb);
    vk_free(gm->xb2);
    vk_free(gm->hb);
    vk_free(gm->hb2);
    vk_free(gm->q);
    vk_free(gm->k);
    vk_free(gm->v);
    vk_free(gm->logits);
    vk_free(gm->kc);
    vk_free(gm->vc);
    vk_free(gm->x_batch);
    vk_free(gm->xb_batch);
    vk_free(gm->xb2_batch);
    vk_free(gm->q_batch);
    vk_free(gm->k_batch);
    vk_free(gm->v_batch);
    vk_free(gm->hb_batch);
    vk_free(gm->hb2_batch);
    vk_free(gm->tokens_dev);

    if (g_ctx) {
        vk_pipelines_shutdown(g_ctx);
        g_ctx = NULL;
    }
    free(gm);
}

void gpu_forward(GpuModel *gm, int token, int pos)
{
    GpuConfig *c = &gm->cfg;
    int dim = c->dim, hd = c->head_dim, kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul, n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int max_seq = c->max_seq, hidden = c->hidden_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    const int wt = DT_F16;

    gpu_emb_lookup(gm, token);

    for (int l = 0; l < c->n_layers; l++) {
        struct { int n; float eps; int x_off; } pc_rn = { dim, c->norm_eps, 0 };
        disp3(VKP_RMSNORM, vkb(gm->xb), vkb(gm->x), vkb(gm->norm_att.layer[l]),
              &pc_rn, sizeof(pc_rn), 1, 1);

        gpu_mm(gm->q, gm->xb, &gm->wq, l, dim, n_heads * hd, wt);
        gpu_mm(gm->k, gm->xb, &gm->wk, l, dim, kv_dim, wt);
        gpu_mm(gm->v, gm->xb, &gm->wv, l, dim, kv_dim, wt);

        struct { int n_heads; int hd; float eps; } pc_hn = { n_heads, hd, c->norm_eps };
        disp2(VKP_RMSNORM_HEAD, vkb(gm->q), vkb(gm->q_norm.layer[l]),
              &pc_hn, sizeof(pc_hn), (uint32_t)n_heads);
        pc_hn.n_heads = n_kv;
        disp2(VKP_RMSNORM_HEAD, vkb(gm->k), vkb(gm->k_norm.layer[l]),
              &pc_hn, sizeof(pc_hn), (uint32_t)n_kv);

        {
            struct { int n_heads; int head_dim; int pos; float theta_base; } pc_rope;
            pc_rope.n_heads = n_heads;
            pc_rope.head_dim = hd;
            pc_rope.pos = pos;
            pc_rope.theta_base = c->rope_theta;
            disp1(VKP_ROPE, vkb(gm->q), &pc_rope, sizeof(pc_rope),
                  (uint32_t)((n_heads * (hd / 2) + 255) / 256));
            pc_rope.n_heads = n_kv;
            disp1(VKP_ROPE, vkb(gm->k), &pc_rope, sizeof(pc_rope),
                  (uint32_t)((n_kv * (hd / 2) + 255) / 256));
        }

        int loff = (int)((size_t)l * max_seq * kv_dim);
        struct { int kv_dim; int pos; int loff; } pc_kv = { kv_dim, pos, loff };
        disp2(VKP_KV_WRITE, vkb(gm->kc), vkb(gm->k), &pc_kv, sizeof(pc_kv),
              (uint32_t)((kv_dim + 255) / 256));
        disp2(VKP_KV_WRITE, vkb(gm->vc), vkb(gm->v), &pc_kv, sizeof(pc_kv),
              (uint32_t)((kv_dim + 255) / 256));

        struct {
            int loff; int pos; int n_heads; int kv_mul; int kv_dim; float scale;
        } pc_fa = { loff, pos, n_heads, kv_mul, kv_dim, scale };
        disp4(VKP_FLASH_ATTN_DECODE, vkb(gm->xb), vkb(gm->q),
              vkb(gm->kc), vkb(gm->vc), &pc_fa, sizeof(pc_fa), (uint32_t)n_heads);

        gpu_mm(gm->xb2, gm->xb, &gm->wo, l, dim, dim, wt);

        struct { int n; } pc_add = { dim };
        disp2(VKP_VEC_ADD, vkb(gm->x), vkb(gm->xb2), &pc_add, sizeof(pc_add),
              (uint32_t)((dim + 255) / 256));

        pc_rn.n = dim;
        disp3(VKP_RMSNORM, vkb(gm->xb), vkb(gm->x), vkb(gm->norm_ffn.layer[l]),
              &pc_rn, sizeof(pc_rn), 1, 1);

        gpu_mm(gm->hb,  gm->xb, &gm->gate, l, dim, hidden, wt);
        gpu_mm(gm->hb2, gm->xb, &gm->up,   l, dim, hidden, wt);

        pc_add.n = hidden;
        disp2(VKP_SILU_MUL, vkb(gm->hb), vkb(gm->hb2), &pc_add, sizeof(pc_add),
              (uint32_t)((hidden + 255) / 256));

        gpu_mm(gm->xb, gm->hb, &gm->down, l, hidden, dim, wt);
        pc_add.n = dim;
        disp2(VKP_VEC_ADD, vkb(gm->x), vkb(gm->xb), &pc_add, sizeof(pc_add),
              (uint32_t)((dim + 255) / 256));
    }

    struct { int n; float eps; int x_off; } pc_out = { dim, c->norm_eps, 0 };
    disp3(VKP_RMSNORM, vkb(gm->xb), vkb(gm->x), vkb(gm->norm_out.ptr),
          &pc_out, sizeof(pc_out), 1, 1);
    gpu_mm_out(gm->logits, gm->xb, gm, dim, c->vocab_size);
    vk_device_sync();
}

void gpu_forward_prefill(GpuModel *gm, const int *tokens, int n_tokens)
{
    if (n_tokens <= 0) return;
    if (n_tokens > gm->batch_cap) {
        fprintf(stderr, "gpu_forward_prefill: n_tokens=%d exceeds batch_cap=%d\n",
                n_tokens, gm->batch_cap);
        exit(1);
    }

    GpuConfig *c = &gm->cfg;
    int dim = c->dim, hd = c->head_dim, kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul, n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int max_seq = c->max_seq, hidden = c->hidden_dim;
    int qdim = n_heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    const int wt = DT_F16;

    vk_memcpy_h2d(gm->tokens_dev, tokens, (size_t)n_tokens * sizeof(int));

    struct { int dim; int n_tokens; } pc_emb = { dim, n_tokens };
    disp3(VKP_EMB_F16_BATCH, vkb(gm->x_batch), vkb(gm->embd.ptr), vkb(gm->tokens_dev),
          &pc_emb, sizeof(pc_emb), (uint32_t)((dim + 255) / 256), (uint32_t)n_tokens);

    for (int l = 0; l < c->n_layers; l++) {
        struct { int n; int n_tokens; float eps; } pc_rnb = { dim, n_tokens, c->norm_eps };
        disp3(VKP_RMSNORM_BATCH, vkb(gm->xb_batch), vkb(gm->x_batch),
              vkb(gm->norm_att.layer[l]), &pc_rnb, sizeof(pc_rnb), (uint32_t)n_tokens, 1);

        gpu_mm_batch(gm->q_batch, gm->xb_batch, &gm->wq, l, dim, qdim, wt, n_tokens);
        gpu_mm_batch(gm->k_batch, gm->xb_batch, &gm->wk, l, dim, kv_dim, wt, n_tokens);
        gpu_mm_batch(gm->v_batch, gm->xb_batch, &gm->wv, l, dim, kv_dim, wt, n_tokens);

        struct { int n_heads; int hd; int n_tokens; float eps; } pc_hnb =
            { n_heads, hd, n_tokens, c->norm_eps };
        disp2(VKP_RMSNORM_HEAD_BATCH, vkb(gm->q_batch), vkb(gm->q_norm.layer[l]),
              &pc_hnb, sizeof(pc_hnb), (uint32_t)(n_tokens * n_heads));
        pc_hnb.n_heads = n_kv;
        disp2(VKP_RMSNORM_HEAD_BATCH, vkb(gm->k_batch), vkb(gm->k_norm.layer[l]),
              &pc_hnb, sizeof(pc_hnb), (uint32_t)(n_tokens * n_kv));

        {
            struct { int n_heads; int head_dim; int n_tokens; float theta_base; } pc_rp;
            pc_rp.n_heads = n_heads;
            pc_rp.head_dim = hd;
            pc_rp.n_tokens = n_tokens;
            pc_rp.theta_base = c->rope_theta;
            disp1(VKP_ROPE_PREFILL_BATCH, vkb(gm->q_batch), &pc_rp, sizeof(pc_rp),
                  (uint32_t)((n_tokens * n_heads * (hd / 2) + 255) / 256));
            pc_rp.n_heads = n_kv;
            disp1(VKP_ROPE_PREFILL_BATCH, vkb(gm->k_batch), &pc_rp, sizeof(pc_rp),
                  (uint32_t)((n_tokens * n_kv * (hd / 2) + 255) / 256));
        }

        int loff = (int)((size_t)l * max_seq * kv_dim);
        struct { int kv_dim; int n_tokens; int loff; } pc_kvb = { kv_dim, n_tokens, loff };
        disp2(VKP_KV_WRITE_BATCH, vkb(gm->kc), vkb(gm->k_batch),
              &pc_kvb, sizeof(pc_kvb), (uint32_t)n_tokens);
        disp2(VKP_KV_WRITE_BATCH, vkb(gm->vc), vkb(gm->v_batch),
              &pc_kvb, sizeof(pc_kvb), (uint32_t)n_tokens);

        struct {
            int loff; int n_tokens; int n_heads; int kv_mul; int kv_dim; float scale;
        } pc_fa = { loff, n_tokens, n_heads, kv_mul, kv_dim, scale };
        disp4(VKP_FLASH_ATTN_PREFILL, vkb(gm->xb_batch), vkb(gm->q_batch),
              vkb(gm->kc), vkb(gm->vc), &pc_fa, sizeof(pc_fa),
              (uint32_t)(n_tokens * n_heads));

        gpu_mm_batch(gm->xb2_batch, gm->xb_batch, &gm->wo, l, qdim, dim, wt, n_tokens);

        struct { int n; int n_tokens; } pc_ab = { dim, n_tokens };
        disp2(VKP_VEC_ADD_BATCH, vkb(gm->x_batch), vkb(gm->xb2_batch),
              &pc_ab, sizeof(pc_ab), (uint32_t)((n_tokens * dim + 255) / 256));

        pc_rnb.n = dim;
        disp3(VKP_RMSNORM_BATCH, vkb(gm->xb_batch), vkb(gm->x_batch),
              vkb(gm->norm_ffn.layer[l]), &pc_rnb, sizeof(pc_rnb), (uint32_t)n_tokens, 1);

        gpu_mm_batch(gm->hb_batch,  gm->xb_batch, &gm->gate, l, dim, hidden, wt, n_tokens);
        gpu_mm_batch(gm->hb2_batch, gm->xb_batch, &gm->up,   l, dim, hidden, wt, n_tokens);

        pc_ab.n = hidden;
        disp2(VKP_SILU_MUL_BATCH, vkb(gm->hb_batch), vkb(gm->hb2_batch),
              &pc_ab, sizeof(pc_ab), (uint32_t)((n_tokens * hidden + 255) / 256));

        gpu_mm_batch(gm->xb_batch, gm->hb_batch, &gm->down, l, hidden, dim, wt, n_tokens);
        pc_ab.n = dim;
        disp2(VKP_VEC_ADD_BATCH, vkb(gm->x_batch), vkb(gm->xb_batch),
              &pc_ab, sizeof(pc_ab), (uint32_t)((n_tokens * dim + 255) / 256));
    }

    gm->prefill_len = n_tokens;

    struct { int n; float eps; int x_off; } pc_out =
        { dim, c->norm_eps, (n_tokens - 1) * dim };
    disp3(VKP_RMSNORM, vkb(gm->x), vkb(gm->x_batch), vkb(gm->norm_out.ptr),
          &pc_out, sizeof(pc_out), 1, 1);
    gpu_mm_out(gm->logits, gm->x, gm, dim, c->vocab_size);
    vk_device_sync();
}

void gpu_copy_logits(GpuModel *gm, float *host_logits)
{
    vk_memcpy_d2h(host_logits, gm->logits,
                  (size_t)gm->cfg.vocab_size * sizeof(float));
}

void gpu_set_prefill_len(GpuModel *gm, int n_tokens)
{
    if (gm) gm->prefill_len = n_tokens;
}

static size_t fp16_mat_bytes(int n_out, int n_in)
{
    return (size_t)n_out * (size_t)n_in * sizeof(uint16_t);
}

void gpu_model_vram_profile(const GpuModel *gm, GpuVramProfile *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!gm) return;

    const int L = gm->cfg.n_layers;
    const int dim = gm->cfg.dim;
    const int hidden = gm->cfg.hidden_dim;
    const int kv_dim = gm->cfg.kv_dim;
    const int vocab = gm->cfg.vocab_size;
    const int max_seq = gm->cfg.max_seq;
    const int qdim = gm->cfg.n_heads * gm->cfg.head_dim;

    out->weights_embd_bytes =
        (size_t)vocab * (size_t)dim * sizeof(uint16_t);
    out->weights_f32_norm_bytes =
        (size_t)L * (size_t)dim * sizeof(float) * 2 +
        (size_t)L * (size_t)gm->cfg.head_dim * sizeof(float) * 2 +
        (size_t)dim * sizeof(float);

    size_t per_layer =
        fp16_mat_bytes(dim, dim) +
        fp16_mat_bytes(kv_dim, dim) * 2 +
        fp16_mat_bytes(dim, dim) +
        fp16_mat_bytes(hidden, dim) * 2 +
        fp16_mat_bytes(dim, hidden);
    out->weights_linear_bytes = (size_t)L * per_layer;
    if (gm->out.ptr && gm->out.ptr != gm->embd.ptr)
        out->weights_linear_bytes += fp16_mat_bytes(vocab, dim);

    out->kv_cache_bytes =
        (size_t)L * (size_t)max_seq * (size_t)kv_dim * sizeof(float) * 2;

    out->decode_activations_bytes =
        (size_t)dim * sizeof(float) * 3 +
        (size_t)hidden * sizeof(float) * 2 +
        (size_t)qdim * sizeof(float) +
        (size_t)kv_dim * sizeof(float) * 2 +
        (size_t)vocab * sizeof(float);

    {
        size_t bc = (size_t)max_seq;
        out->prefill_batch_bytes =
            bc * (size_t)dim * sizeof(float) * 3 +
            bc * (size_t)qdim * sizeof(float) +
            bc * (size_t)kv_dim * sizeof(float) * 2 +
            bc * (size_t)hidden * sizeof(float) * 2 +
            bc * sizeof(int);
    }

    out->total_bytes =
        out->weights_embd_bytes +
        out->weights_f32_norm_bytes +
        out->weights_linear_bytes +
        out->kv_cache_bytes +
        out->decode_activations_bytes +
        out->prefill_batch_bytes;

    vk_ctx_get_memory_info(g_ctx, &out->device_total_bytes, &out->device_used_bytes);
}

void gpu_get_device_desc(char *buf, size_t cap)
{
    if (!buf || cap == 0) return;
    buf[0] = '\0';
    VkCtx *ctx = vk_device_ctx();
    char name[256];
    vk_ctx_get_device_name(ctx, name, sizeof name);
    size_t total = 0, used = 0;
    vk_ctx_get_memory_info(ctx, &total, &used);
    snprintf(buf, cap, "%s (Vulkan, %.1f GB)", name,
             (double)total / (1024.0 * 1024.0 * 1024.0));
}

void gpu_print_device_info(void)
{
    char desc[256];
    gpu_get_device_desc(desc, sizeof desc);
    printf("GPU: %s\n", desc);
}
