#ifndef QWEN3_FA_DEBUG_H
#define QWEN3_FA_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float max_abs;
    float min_val;
    float mean_abs;
    int   nan_count;
    int   inf_count;
} FaTensorStats;

void fa_debug_set_enabled(int on);
int  fa_debug_enabled(void);
void fa_debug_set_backend_label(const char *label);

void fa_debug_decode_hook(int prefill_len, int pos, int layer, int n_layers,
                          int npos, int kv_dim, int hd, int n_heads, int n_kv,
                          int kv_mul, float scale,
                          const float *d_q, const float *d_k, const float *d_v,
                          const float *d_xb, const float *d_kc, const float *d_vc);

void fa_debug_prefill_hook(int n_tokens, int kv_dim, int hd, int max_seq,
                           int n_layers, const float *d_kc_base,
                           const float *d_vc_base);

/* prefill: 指定層 wk 前 (xb) / 後 (k) */
void fa_debug_prefill_layer_hook(int layer, const char *phase,
                                 int n_tokens, int dim, int kv_dim,
                                 const float *d_xb_norm, const float *d_k_wk);

void fa_debug_prefill_x_row0(int layer, int dim, const float *d_x_batch);
void fa_debug_prefill_x_row0_tag(int layer, const char *tag, int dim,
                                 const float *d_x_batch);
void fa_debug_prefill_hb_row0_tag(int layer, const char *tag, int hidden,
                                  const float *d_hb_batch);

/* prefill 層0: kv_write 直後の KV キャッシュ */
void fa_debug_prefill_kv_layer0_hook(int n_tokens, int kv_dim, int hd,
                                     const float *d_kc_l0, const float *d_vc_l0);

#ifdef __cplusplus
}
#endif

#endif
