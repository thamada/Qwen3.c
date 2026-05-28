/*
 * NVFP4 GEMM verification: path consistency + optional FP32 reference.
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

static float max_abs_diff_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, int n)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(bf16_to_f32(a[i]) - bf16_to_f32(b[i]));
        if (d > mx) mx = d;
    }
    return mx;
}

static float max_abs_diff_f32(const float *a, const float *b, int n)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Compare cached / GPU-kernel-quant / host-layout-quant GEMM paths. */
static int run_path_compare(int M, int N, int K, const char *label)
{
    const int nA = M * K, nB = N * K, nD = M * N;

    __nv_bfloat16 *h_Ab = (__nv_bfloat16 *)malloc((size_t)nA * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Bb = (__nv_bfloat16 *)malloc((size_t)nB * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Dc = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Dg = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Dh = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    if (!h_Ab || !h_Bb || !h_Dc || !h_Dg || !h_Dh) return 1;

    for (int i = 0; i < nA; i++)
        h_Ab[i] = __float2bfloat16_rn(0.002f * (float)((i * 17 + 3) % 97 - 48));
    for (int i = 0; i < nB; i++)
        h_Bb[i] = __float2bfloat16_rn(0.002f * (float)((i * 31 + 7) % 89 - 44));

    __nv_bfloat16 *d_A = NULL, *d_B = NULL;
    __nv_bfloat16 *d_Dc = NULL, *d_Dg = NULL, *d_Dh = NULL;
    cudaMalloc(&d_A, (size_t)nA * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, (size_t)nB * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dc, (size_t)nD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dg, (size_t)nD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dh, (size_t)nD * sizeof(__nv_bfloat16));
    cudaMemcpy(d_A, h_Ab, (size_t)nA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_Bb, (size_t)nB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    fp4_gemm_cleanup();
    if (fp4_gemm_prealloc(M, N, K) != 0) {
        printf("%s paths: prealloc failed\n", label);
        return 1;
    }

    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) {
        printf("%s paths: weight quantize failed\n", label);
        return 1;
    }

    int rc_c = fp4_gemm_run_cached(d_A, cache, NULL, d_Dc, M, 1.f, 0.f);
    int rc_g = fp4_gemm_run(d_A, d_B, NULL, d_Dg, M, N, K, 1.f, 0.f);
    int rc_h = fp4_gemm_run_host(d_A, d_B, NULL, d_Dh, M, N, K, 1.f, 0.f);
    if (rc_c || rc_g || rc_h) {
        printf("%s paths: gemm rc cached=%d gpu=%d host=%d\n", label, rc_c, rc_g, rc_h);
        fp4_weight_cache_free(cache);
        return 1;
    }

    cudaMemcpy(h_Dc, d_Dc, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Dg, d_Dg, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Dh, d_Dh, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float mx_cg = max_abs_diff_bf16(h_Dc, h_Dg, nD);
    float mx_ch = max_abs_diff_bf16(h_Dc, h_Dh, nD);
    float mx_gh = max_abs_diff_bf16(h_Dg, h_Dh, nD);

    float norm = 0.f;
    for (int i = 0; i < nD; i++)
        norm = fmaxf(norm, fabsf(bf16_to_f32(h_Dc[i])));

    float tol = fmaxf(0.05f, norm * 0.02f);
    int pass = (mx_cg <= tol) && (mx_ch <= tol) && (mx_gh <= tol);
    printf("%s paths (%dx%d M=%d): |cached-gpu|=%.4f |cached-host|=%.4f |gpu-host|=%.4f tol=%.4f %s\n",
           label, N, K, M, mx_cg, mx_ch, mx_gh, tol, pass ? "PASS" : "FAIL");

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_Dc); cudaFree(d_Dg); cudaFree(d_Dh);
    free(h_Ab); free(h_Bb); free(h_Dc); free(h_Dg); free(h_Dh);
    return pass ? 0 : 1;
}

/* FP32 reference (informational; large K has high quant error vs full matmul). */
static int run_fp32_ref(int M, int N, int K, const char *label)
{
    const int nA = M * K, nB = N * K, nD = M * N;

    float *h_Af = (float *)malloc((size_t)nA * sizeof(float));
    float *h_Bf = (float *)malloc((size_t)nB * sizeof(float));
    float *h_ref = (float *)malloc((size_t)nD * sizeof(float));
    float *h_out = (float *)malloc((size_t)nD * sizeof(float));
    __nv_bfloat16 *h_Ab = (__nv_bfloat16 *)malloc((size_t)nA * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Bb = (__nv_bfloat16 *)malloc((size_t)nB * sizeof(__nv_bfloat16));
    if (!h_Af || !h_Bf || !h_ref || !h_out || !h_Ab || !h_Bb) return 1;

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
    if (fp4_gemm_prealloc(M, N, K) != 0) return 1;
    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) return 1;
    if (fp4_gemm_run_cached(d_A, cache, NULL, d_D, M, 1.f, 0.f) != 0) return 1;

    __nv_bfloat16 *h_Db = (__nv_bfloat16 *)malloc((size_t)nD * sizeof(__nv_bfloat16));
    cudaMemcpy(h_Db, d_D, (size_t)nD * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nD; i++) h_out[i] = bf16_to_f32(h_Db[i]);

    float mx = max_abs_diff_f32(h_ref, h_out, nD);
    float ref_norm = 0.f;
    for (int i = 0; i < nD; i++) ref_norm = fmaxf(ref_norm, fabsf(h_ref[i]));
    float ratio = ref_norm > 0 ? mx / ref_norm : mx;
    /* Large-K NVFP4 vs FP32 matmul: ratio ~0.7 is typical for this test pattern. */
    int pass = (K <= 128) ? (ratio < 0.45f) : (ratio < 0.85f);
    printf("%s fp32ref (%dx%d M=%d): max_err=%.4f ratio=%.4f %s\n",
           label, N, K, M, mx, ratio, pass ? "PASS" : "FAIL");

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    free(h_Af); free(h_Bf); free(h_ref); free(h_out); free(h_Ab); free(h_Bb); free(h_Db);
    return pass ? 0 : 1;
}

/* M_act rows in a padded M=128 batch vs M=1 with the same row in slot 0. */
static int run_batch_row_parity(int M_act, int N, int K, const char *label)
{
    const int M_pad = 128;
    const int K_pad = ((K + 127) / 128) * 128;
    const int N_pad = ((N + 127) / 128) * 128;

    __nv_bfloat16 *h_Abatch = (__nv_bfloat16 *)calloc((size_t)M_pad * K_pad, sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Asingle = (__nv_bfloat16 *)calloc((size_t)M_pad * K_pad, sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc((size_t)N * K * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Dbatch = (__nv_bfloat16 *)malloc((size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Dsingle = (__nv_bfloat16 *)malloc((size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    if (!h_Abatch || !h_Asingle || !h_B || !h_Dbatch || !h_Dsingle) return 1;

    for (int m = 0; m < M_act; m++) {
        for (int k = 0; k < K; k++) {
            float v = 0.01f * (float)((m * 131 + k * 17 + 3) % 211 - 105);
            h_Abatch[(size_t)m * K_pad + k] = __float2bfloat16_rn(v);
        }
    }
    for (int k = 0; k < K; k++) {
        float v = 0.01f * (float)((k * 17 + 3) % 211 - 105);
        h_Asingle[k] = __float2bfloat16_rn(v);
    }
    for (int i = 0; i < N * K; i++)
        h_B[i] = __float2bfloat16_rn(0.002f * (float)((i * 31 + 7) % 89 - 44));

    __nv_bfloat16 *d_Abatch = NULL, *d_Asingle = NULL, *d_B = NULL;
    __nv_bfloat16 *d_Dbatch = NULL, *d_Dsingle = NULL;
    cudaMalloc(&d_Abatch, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Asingle, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, (size_t)N * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dbatch, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dsingle, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    cudaMemcpy(d_Abatch, h_Abatch, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Asingle, h_Asingle, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    fp4_gemm_cleanup();
    if (fp4_gemm_prealloc(M_pad, N_pad, K_pad) != 0) {
        printf("%s: prealloc failed\n", label);
        return 1;
    }
    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) return 1;

    if (fp4_gemm_run_cached(d_Abatch, cache, NULL, d_Dbatch, M_pad, 1.f, 0.f) != 0 ||
        fp4_gemm_run_cached(d_Asingle, cache, NULL, d_Dsingle, M_pad, 1.f, 0.f) != 0) {
        printf("%s: gemm failed\n", label);
        fp4_weight_cache_free(cache);
        return 1;
    }
    fp4_gemm_sync();
    cudaMemcpy(h_Dbatch, d_Dbatch, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Dsingle, d_Dsingle, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int fail_rows = 0;
    float worst = 0.f;
    for (int m = 0; m < M_act; m++) {
        /* single-row: put batch row m into slot 0 and compare */
        for (int k = 0; k < K; k++)
            h_Asingle[k] = h_Abatch[(size_t)m * K_pad + k];
        cudaMemset((char *)d_Asingle + K * sizeof(__nv_bfloat16), 0,
            (size_t)(M_pad * K_pad - K) * sizeof(__nv_bfloat16));
        cudaMemcpy(d_Asingle, h_Asingle, (size_t)K_pad * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        if (fp4_gemm_run_cached(d_Asingle, cache, NULL, d_Dsingle, M_pad, 1.f, 0.f) != 0) {
            fail_rows++;
            continue;
        }
        fp4_gemm_sync();
        cudaMemcpy(h_Dsingle, d_Dsingle, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        float mx = 0.f;
        for (int n = 0; n < N; n++) {
            float a = bf16_to_f32(h_Dbatch[(size_t)m * N_pad + n]);
            float b = bf16_to_f32(h_Dsingle[n]);
            float d = fabsf(a - b);
            if (isnan(a) || isnan(b)) { mx = 1e30f; break; }
            if (d > mx) mx = d;
        }
        if (mx > worst) worst = mx;
        float ref = 0.f;
        for (int n = 0; n < N; n++)
            ref = fmaxf(ref, fabsf(bf16_to_f32(h_Dbatch[(size_t)m * N_pad + n])));
        if (mx > fmaxf(0.05f, ref * 0.05f)) fail_rows++;
    }

    int pass = (fail_rows == 0);
    printf("%s batch-vs-row0 (%dx%d M_act=%d M_pad=%d): fail_rows=%d worst=%.4f %s\n",
           label, N, K, M_act, M_pad, fail_rows, worst, pass ? "PASS" : "FAIL");

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();
    cudaFree(d_Abatch); cudaFree(d_Asingle); cudaFree(d_B);
    cudaFree(d_Dbatch); cudaFree(d_Dsingle);
    free(h_Abatch); free(h_Asingle); free(h_B); free(h_Dbatch); free(h_Dsingle);
    return pass ? 0 : 1;
}

static int has_nan_bf16(const __nv_bfloat16 *h, int n)
{
    for (int i = 0; i < n; i++) {
        float v = bf16_to_f32(h[i]);
        if (isnan(v) || isinf(v)) return 1;
    }
    return 0;
}

/* Reproduce inference-like large activations (hb ~5e3) through down GEMM. */
static int run_extreme_act(int N, int K, float peak, const char *label)
{
    const int M_pad = 128;
    const int K_pad = ((K + 127) / 128) * 128;
    const int N_pad = ((N + 127) / 128) * 128;

    __nv_bfloat16 *h_A = (__nv_bfloat16 *)calloc((size_t)M_pad * K_pad, sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc((size_t)N * K * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_D = (__nv_bfloat16 *)malloc((size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    if (!h_A || !h_B || !h_D) return 1;

    for (int k = 0; k < K; k++) {
        float v = (k & 1) ? peak : -0.25f * peak;
        h_A[k] = __float2bfloat16_rn(v);
    }
    for (int i = 0; i < N * K; i++)
        h_B[i] = __float2bfloat16_rn(0.002f * (float)((i * 31 + 7) % 89 - 44));

    __nv_bfloat16 *d_A = NULL, *d_B = NULL, *d_D = NULL;
    cudaMalloc(&d_A, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, (size_t)N * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, (size_t)M_pad * N_pad * sizeof(__nv_bfloat16));
    cudaMemcpy(d_A, h_A, (size_t)M_pad * K_pad * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    fp4_gemm_cleanup();
    if (fp4_gemm_prealloc(M_pad, N_pad, K_pad) != 0) return 1;
    void *cache = fp4_quantize_weights(d_B, N, K);
    if (!cache) return 1;
    if (fp4_gemm_run_cached(d_A, cache, NULL, d_D, M_pad, 1.f, 0.f) != 0) {
        printf("%s: gemm failed\n", label);
        return 1;
    }
    fp4_gemm_sync();
    cudaMemcpy(h_D, d_D, (size_t)N_pad * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int nan = has_nan_bf16(h_D, N);
    float mx = 0.f;
    for (int n = 0; n < N; n++)
        mx = fmaxf(mx, fabsf(bf16_to_f32(h_D[n])));
    printf("%s extreme act (N=%d K=%d): nan=%d max_out=%.4e %s\n",
           label, N, K, nan, mx, nan ? "FAIL" : "PASS");

    fp4_weight_cache_free(cache);
    fp4_gemm_cleanup();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    free(h_A); free(h_B); free(h_D);
    return nan ? 1 : 0;
}

int main(void)
{
    int fail = 0;
    printf("=== GEMM path consistency (cached vs gpu-quant vs host-layout-quant) ===\n");
    fail |= run_path_compare(128, 128, 128, "square");
    fail |= run_path_compare(128, 1024, 4096, "wk");
    fail |= run_path_compare(256, 4096, 4096, "wq_M256");

    printf("=== Batch row parity (M_act in M_pad=128) ===\n");
    fail |= run_batch_row_parity(20, 1024, 4096, "wk_M20");
    fail |= run_batch_row_parity(20, 4096, 12288, "down_M20");

    printf("=== Extreme activation (hb-scale) ===\n");
    fail |= run_extreme_act(4096, 12288, 4857.f, "down_extreme");

    printf("=== FP32 reference (informational) ===\n");
    fail |= run_fp32_ref(128, 128, 128, "square");
    fail |= run_fp32_ref(128, 1024, 4096, "wk");
    fail |= run_fp32_ref(256, 4096, 4096, "wq_M256");

    return fail ? 1 : 0;
}
