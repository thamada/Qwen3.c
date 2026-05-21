#ifndef BONSAI_FP4_GEMM_H
#define BONSAI_FP4_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

int  fp4_gemm_sf_vec_size(void);
int  fp4_gemm_init(int M, int N, int K);
int  fp4_gemm_prealloc(int max_M, int max_N, int max_K);
int  fp4_gemm_run(const void *A_bf16, const void *B_bf16,
                  const void *C_bf16, void *D_bf16,
                  int M, int N, int K, float alpha, float beta);
void fp4_gemm_cleanup(void);
void fp4_gemm_sync(void);

void *fp4_quantize_weights(const void *weight_bf16, int N, int K);
const void *fp4_weight_cache_fp4_ptr(const void *cache);
const void *fp4_weight_cache_sf_ptr(const void *cache);
int   fp4_weight_cache_N(const void *cache);
int   fp4_weight_cache_K(const void *cache);
void  fp4_weight_cache_free(void *cache);

int  fp4_gemm_run_cached(const void *A_bf16, const void *cache_handle,
                          const void *C_bf16, void *D_bf16,
                          int M, float alpha, float beta);

#ifdef __cplusplus
}
#endif

#endif
