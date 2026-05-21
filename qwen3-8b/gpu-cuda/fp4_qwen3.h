#ifndef FP4_QWEN3_H
#define FP4_QWEN3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  fp4_qwen3_init(int max_M, int max_N, int max_K);
void fp4_qwen3_shutdown(void);

/* FP16 host weight [N,K] row-major -> NVFP4 cache (upload-time quantize). */
void *fp4_qwen3_weight_from_f16_host(const uint16_t *host_f16, int N, int K);
void  fp4_qwen3_free_weight(void *cache);

/* y[M*d] = W[d,n] @ x[M*n]  (W is cached NVFP4, x/y are F32). */
void fp4_qwen3_mm(const void *weight_cache,
                  const float *x, float *y,
                  int M, int n, int d);

#ifdef __cplusplus
}
#endif

#endif
