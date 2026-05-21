#ifndef BONSAI_FP4_BONSAI_H
#define BONSAI_FP4_BONSAI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int  fp4_bonsai_init(int max_M, int max_N, int max_K);
void fp4_bonsai_shutdown(void);

/* Q1_0 重み [N,K] を NVFP4 Tensor Core 用キャッシュへ変換（K/N は 128 倍数へパディング） */
void *fp4_bonsai_weight_from_q1_host(const void *host_q1, int N, int K,
                                       size_t row_stride);

void fp4_bonsai_free_weight(void *cache);

/* y[M*d] = W[d,n] @ x[M*n]  (W は事前量子化キャッシュ) */
void fp4_bonsai_mm(const void *weight_cache,
                   const float *x, float *y,
                   int M, int n, int d);

#ifdef __cplusplus
}
#endif

#endif
