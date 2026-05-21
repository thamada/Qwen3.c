/*
 * Minimal NVFP4 GEMM layout verification (128^3).
 * Build: make fp4-test
 */

#include "fp4_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

static void cpu_gemm(const float *A, const float *B, float *D,
                     int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += A[(size_t)m * K + k] * B[(size_t)n * K + k];
            D[(size_t)m * N + n] = sum;
        }
    }
}

static float max_abs_diff(const float *a, const float *b, int n)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static int run_case(int M, int N, int K, const char *label)
{
    const int nA = M * K, nB = N * K, nD = M * N;

    float *h_Af = (float *)malloc((size_t)nA * sizeof(float));
    float *h_Bf = (float *)malloc((size_t)nB * sizeof(float));
    float *h_ref = (float *)malloc((size_t)nD * sizeof(float));
    float *h_out = (float *)malloc((size_t)nD * sizeof(float));
    __nv_bfloat16 *h_Ab = (__nv_bfloat16 *)malloc((size_t)nA * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Bb = (__nv_bfloat16 *)malloc((size_t)nB * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Db = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    if (!h_Af || !h_Bf || !h_ref || !h_out || !h_Ab || !h_Bb || !h_Db) return 1;

    for (int i = 0; i < nA; i++) h_Af[i] = 0.002f * (float)((i * 17 + 3) % 97 - 48);
    for (int i = 0; i < nB; i++) h_Bf[i] = 0.002f * (float)((i * 31 + 7) % 89 - 44);
    for (int i = 0; i < nA; i++) h_Ab[i] = __float2bfloat16_rn(h_Af[i]);
    for (int i = 0; i < nB; i++) h_Bb[i] = __float2bfloat16_rn(h_Bf[i]);
    cpu_gemm(h_Af, h_Bf, h_ref, M, N, K);

    __nv_bfloat16 *d_A = NULL, *d_B = NULL, *d_D = NULL;
    cudaMalloc(&d_A, (size_t)nA * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, (size_t)nB * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, (size_t)nD * sizeof(__nv_bfloat16));
    cudaMemcpy(d_A, h_Ab, (size_t)nA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_Bb, (size_t)nB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    fp4_gemm_cleanup();
    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) { printf("%s: quantize failed\n", label); return 1; }

    int rc = fp4_gemm_run_cached(d_A, cache, NULL, d_D, M, 1.f, 0.f);
    if (rc != 0) { printf("%s: gemm failed rc=%d\n", label, rc); return 1; }

    cudaMemcpy(h_Db, d_D, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nD; i++) h_out[i] = bf16_to_f32(h_Db[i]);

    float mx = max_abs_diff(h_ref, h_out, nD);
    float ref_norm = 0.f;
    for (int i = 0; i < nD; i++) ref_norm = fmaxf(ref_norm, fabsf(h_ref[i]));

    float ratio = ref_norm > 0 ? mx / ref_norm : mx;
    int pass = ratio < 0.45f;
    printf("%s (%dx%d @ M=%d): max_err=%.4f ratio=%.4f %s\n",
           label, N, K, M, mx, ratio, pass ? "PASS" : "FAIL");

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    free(h_Af); free(h_Bf); free(h_ref); free(h_out); free(h_Ab); free(h_Bb); free(h_Db);
    return pass ? 0 : 1;
}

int main(void)
{
    int fail = 0;
    fail |= run_case(128, 128, 128, "square");
    fail |= run_case(128, 1024, 4096, "wk");
    fail |= run_case(256, 4096, 4096, "wq_M256");
    return fail ? 1 : 0;
}

#if 0
int main_old(void)
{
    const int M = 128, N = 128, K = 128;
    const int nA = M * K, nB = N * K, nD = M * N;

    float *h_Af = (float *)malloc((size_t)nA * sizeof(float));
    float *h_Bf = (float *)malloc((size_t)nB * sizeof(float));
    float *h_ref = (float *)malloc((size_t)nD * sizeof(float));
    float *h_out = (float *)malloc((size_t)nD * sizeof(float));
    __nv_bfloat16 *h_Ab = (__nv_bfloat16 *)malloc((size_t)nA * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Bb = (__nv_bfloat16 *)malloc((size_t)nB * sizeof(__nv_bfloat16));
    if (!h_Af || !h_Bf || !h_ref || !h_out || !h_Ab || !h_Bb) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    for (int i = 0; i < nA; i++) h_Af[i] = 0.01f * (float)((i * 17 + 3) % 97 - 48);
    for (int i = 0; i < nB; i++) h_Bf[i] = 0.01f * (float)((i * 31 + 7) % 89 - 44);
    for (int i = 0; i < nA; i++) h_Ab[i] = __float2bfloat16_rn(h_Af[i]);
    for (int i = 0; i < nB; i++) h_Bb[i] = __float2bfloat16_rn(h_Bf[i]);

    cpu_gemm(h_Af, h_Bf, h_ref, M, N, K);

    __nv_bfloat16 *d_A = NULL, *d_B = NULL, *d_D = NULL;
    cudaMalloc(&d_A, (size_t)nA * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, (size_t)nB * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, (size_t)nD * sizeof(__nv_bfloat16));
    cudaMemcpy(d_A, h_Ab, (size_t)nA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_Bb, (size_t)nB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    int rc = fp4_gemm_run(d_A, d_B, NULL, d_D, M, N, K, 1.f, 0.f);
    if (rc != 0) {
        fprintf(stderr, "fp4_gemm_run failed rc=%d\n", rc);
        return 1;
    }

    __nv_bfloat16 *h_Db = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    cudaMemcpy(h_Db, d_D, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nD; i++) h_out[i] = bf16_to_f32(h_Db[i]);

    float mx = max_abs_diff(h_ref, h_out, nD);
    float ref_norm = 0.f;
    for (int i = 0; i < nD; i++) ref_norm = fmaxf(ref_norm, fabsf(h_ref[i]));

    printf("fp4_gemm_run 128^3: max_abs_err=%.6f ref_max=%.6f ratio=%.4f\n",
           mx, ref_norm, ref_norm > 0 ? mx / ref_norm : mx);
    printf("sample ref[0]=%.4f out[0]=%.4f ref[1]=%.4f out[1]=%.4f\n",
           h_ref[0], h_out[0], h_ref[1], h_out[1]);

    /* cached path */
    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) {
        fprintf(stderr, "fp4_quantize_weights failed\n");
        return 1;
    }
    rc = fp4_gemm_run_cached(d_A, cache, NULL, d_D, M, 1.f, 0.f);
    if (rc != 0) {
        fprintf(stderr, "fp4_gemm_run_cached failed rc=%d\n", rc);
        return 1;
    }
    cudaMemcpy(h_Db, d_D, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nD; i++) h_out[i] = bf16_to_f32(h_Db[i]);
    mx = max_abs_diff(h_ref, h_out, nD);
    printf("fp4_gemm_run_cached: max_abs_err=%.6f ratio=%.4f\n",
           mx, ref_norm > 0 ? mx / ref_norm : mx);
    printf("sample ref[0]=%.4f out[0]=%.4f\n", h_ref[0], h_out[0]);

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();

    int pass = (mx / ref_norm) < 0.15f;
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
#endif
