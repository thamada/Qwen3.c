/*
 * Qwen3 FP4 bridge: NVFP4 weight cache, F32 activations in/out.
 * Inference uses FP4 GEMV/GEMV-batch only. CUTLASS NVFP4 GEMM (M>=128) quantizes
 * activations to FP4; GEMV uses F32 activations — mixing them breaks generation
 * when prefill length >= 128 (GEMM prefill + GEMV decode mismatch).
 */

#include "fp4_qwen3.h"
#include "fp4_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static int g_max_M = 0, g_max_N = 0, g_max_K = 0;
static __nv_bfloat16 *g_act_bf16 = NULL;
static __nv_bfloat16 *g_out_bf16 = NULL;
static size_t g_act_cap = 0, g_out_cap = 0;

static int align128(int x) { return (x + 127) & ~127; }

int fp4_qwen3_init(int max_M, int max_N, int max_K)
{
    int M = align128(max_M);
    int N = align128(max_N);
    int K = align128(max_K);
    if (fp4_gemm_prealloc(M, N, K) != 0)
        return -1;

    size_t need_act = (size_t)M * K;
    size_t need_out = (size_t)M * N;
    if (need_act > g_act_cap) {
        if (g_act_bf16) cudaFree(g_act_bf16);
        CUDA_CHECK(cudaMalloc(&g_act_bf16, need_act * sizeof(__nv_bfloat16)));
        g_act_cap = need_act;
    }
    if (need_out > g_out_cap) {
        if (g_out_bf16) cudaFree(g_out_bf16);
        CUDA_CHECK(cudaMalloc(&g_out_bf16, need_out * sizeof(__nv_bfloat16)));
        g_out_cap = need_out;
    }
    g_max_M = M;
    g_max_N = N;
    g_max_K = K;

    /* 128³ smoke test — catches CUTLASS/workspace misconfig before full model load */
    {
        const int tM = 128, tN = 128, tK = 128;
        __nv_bfloat16 *dev_w = NULL, *dev_act = NULL, *dev_out = NULL;
        cudaMalloc(&dev_w, (size_t)tN * tK * sizeof(__nv_bfloat16));
        cudaMalloc(&dev_act, (size_t)tM * tK * sizeof(__nv_bfloat16));
        cudaMalloc(&dev_out, (size_t)tM * tN * sizeof(__nv_bfloat16));
        cudaMemset(dev_w, 0, (size_t)tN * tK * sizeof(__nv_bfloat16));
        cudaMemset(dev_act, 0, (size_t)tM * tK * sizeof(__nv_bfloat16));
        void *wc = fp4_quantize_weights(dev_w, tN, tK);
        int rc = wc ? fp4_gemm_run_cached(dev_act, wc, NULL, dev_out, tM, 1.0f, 0.0f) : -1;
        if (wc) fp4_weight_cache_free(wc);
        cudaFree(dev_w);
        cudaFree(dev_act);
        cudaFree(dev_out);
        if (rc != 0) {
            fprintf(stderr, "fp4_qwen3_init: NVFP4 sanity GEMM failed (rc=%d)\n", rc);
            return -1;
        }
    }
    return 0;
}

void fp4_qwen3_shutdown(void)
{
    fp4_gemm_cleanup();
    if (g_act_bf16) cudaFree(g_act_bf16);
    if (g_out_bf16) cudaFree(g_out_bf16);
    g_act_bf16 = NULL;
    g_out_bf16 = NULL;
    g_act_cap = g_out_cap = 0;
    g_max_M = g_max_N = g_max_K = 0;
}

void *fp4_qwen3_weight_from_f16_host(const uint16_t *host_f16, int N, int K)
{
    void *cache = fp4_quantize_weights_host_f16(host_f16, N, K);
    if (!cache)
        fprintf(stderr, "fp4_qwen3_weight_from_f16_host: quantize failed N=%d K=%d\n", N, K);
    return cache;
}

void *fp4_qwen3_weight_from_rows(int N, int K,
                                 fp4_dequant_row_fn get_row, void *ctx)
{
    FP4HostWeight *host = fp4_host_weight_build(N, K, get_row, ctx);
    if (!host) return NULL;
    void *cache = fp4_weight_cache_upload(host);
    fp4_host_weight_free(host);
    return cache;
}

FP4HostWeight *fp4_qwen3_host_weight_from_rows(int N, int K,
                                               fp4_dequant_row_fn get_row,
                                               void *ctx)
{
    return fp4_host_weight_build(N, K, get_row, ctx);
}

void *fp4_qwen3_weight_from_host(const FP4HostWeight *host)
{
    return fp4_weight_cache_upload(host);
}

void fp4_qwen3_free_weight(void *cache)
{
    fp4_weight_cache_free(cache);
}

size_t fp4_qwen3_vram_bytes(void)
{
    return g_act_cap * sizeof(__nv_bfloat16) + g_out_cap * sizeof(__nv_bfloat16);
}

void fp4_qwen3_set_gemm_row(int row)
{
    (void)row;
}

void fp4_qwen3_mm(const void *weight_cache,
                  const float *x, float *y,
                  int M, int n, int d)
{
    if (!weight_cache) {
        fprintf(stderr, "fp4_qwen3_mm: null weight cache\n");
        exit(1);
    }
    if (M <= 0) return;

    if (M == 1)
        fp4_gemv_cached(weight_cache, x, y, n, d);
    else
        fp4_gemv_batch_cached(weight_cache, x, y, M, n, d);
}
