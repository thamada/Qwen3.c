#ifndef QWEN3_GPU_H
#define QWEN3_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    int head_dim, kv_dim, kv_mul;
    float norm_eps, rope_theta;
} GpuConfig;

typedef struct {
    void  *embd;    int embd_t;
    void **norm_att;
    void **wq, **wk, **wv, **wo;
    void **q_norm, **k_norm, **norm_ffn;
    void **gate, **up, **down;
    void  *norm_out;
    void  *out;     int out_t;
} GpuWeightsHost;

typedef struct GpuModel GpuModel;

typedef struct {
    size_t total_bytes;
    size_t weights_embd_bytes;
    size_t weights_f32_norm_bytes;
    size_t weights_linear_bytes;
    size_t kv_cache_bytes;
    size_t decode_activations_bytes;
    size_t prefill_batch_bytes;
    size_t device_used_bytes;
    size_t device_total_bytes;
} GpuVramProfile;

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host);
void      gpu_model_destroy(GpuModel *gm);
void      gpu_model_vram_profile(const GpuModel *gm, GpuVramProfile *out);
void      gpu_forward(GpuModel *gm, int token, int pos);
void      gpu_forward_prefill(GpuModel *gm, const int *tokens, int n_tokens);
void      gpu_copy_logits(GpuModel *gm, float *host_logits);
void      gpu_set_prefill_len(GpuModel *gm, int n_tokens);
void      gpu_print_device_info(void);
void      gpu_get_device_desc(char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
