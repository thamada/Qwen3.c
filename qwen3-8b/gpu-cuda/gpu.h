#ifndef QWEN3_GPU_H
#define QWEN3_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    int head_dim, kv_dim, kv_mul;
    float norm_eps, rope_theta;
} GpuConfig;

/* Device-resident weights (uploaded in main.c before gpu_model_create). */
typedef struct {
    void  *embd;    int embd_t;
    void **norm_att;
    void **wq, **wk, **wv, **wo;
    void **q_norm, **k_norm, **norm_ffn;
    void **gate, **up, **down;
    void  *norm_out;
    void  *out;     int out_t;
    /* NVFP4 caches (BONSAI_FP4): prefill Tensor Core path; wq etc. stay FP16 for decode. */
    void **wq_fp4, **wk_fp4, **wv_fp4, **wo_fp4;
    void **gate_fp4, **up_fp4, **down_fp4;
    void  *out_fp4;
} GpuWeightsHost;

typedef struct GpuModel GpuModel;

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host);
void      gpu_model_destroy(GpuModel *gm);
void      gpu_forward(GpuModel *gm, int token, int pos);
void      gpu_forward_prefill(GpuModel *gm, const int *tokens, int n_tokens);
void      gpu_copy_logits(GpuModel *gm, float *host_logits);
void      gpu_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif
