/*
 * Qwen3 FP4 bridge: NVFP4 weight cache, F32 activations in/out.
 * Prefill (M>=128): CUTLASS block-scaled NVFP4 GEMM.
 * Decode  (M<128):  dedicated FP4 GEMV (no M=128 padding).
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

#define FP4_GEMM_MIN_M 128

static int g_max_M = 0, g_max_N = 0, g_max_K = 0;
static __nv_bfloat16 *g_act_bf16 = NULL;
static __nv_bfloat16 *g_out_bf16 = NULL;
static size_t g_act_cap = 0, g_out_cap = 0;

static int align128(int x) { return (x + 127) & ~127; }

static __global__ void f32_to_bf16_pad_kernel(
    const float *src, __nv_bfloat16 *dst, int M, int n, int K_pad, int M_act)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * K_pad;
    if (idx >= total) return;
    int m = idx / K_pad;
    int k = idx % K_pad;
    float v = 0.f;
    if (m < M_act && k < n)
        v = src[(size_t)m * n + k];
    dst[idx] = __float2bfloat16_rn(v);
}

static __global__ void bf16_to_f32_trunc_kernel(
    const __nv_bfloat16 *src, float *dst, int M, int d, int N_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;
    if (idx >= total) return;
    int m = idx / d;
    int n = idx - m * d;
    dst[idx] = __bfloat162float(src[(size_t)m * N_pad + n]);
}

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

void fp4_qwen3_mm(const void *weight_cache,
                  const float *x, float *y,
                  int M, int n, int d)
{
    if (!weight_cache) {
        fprintf(stderr, "fp4_qwen3_mm: null weight cache\n");
        exit(1);
    }

    if (M < FP4_GEMM_MIN_M) {
        if (M == 1)
            fp4_gemv_cached(weight_cache, x, y, n, d);
        else
            fp4_gemv_batch_cached(weight_cache, x, y, M, n, d);
        return;
    }

    int K_pad = fp4_weight_cache_K(weight_cache);
    int N_pad = fp4_weight_cache_N(weight_cache);
    int M_pad = align128(M);

    if (M_pad > g_max_M || N_pad > g_max_N || K_pad > g_max_K) {
        if (fp4_qwen3_init(M_pad > g_max_M ? M_pad : g_max_M,
                           N_pad > g_max_N ? N_pad : g_max_N,
                           K_pad > g_max_K ? K_pad : g_max_K) != 0) {
            fprintf(stderr, "fp4_qwen3_mm: init failed\n");
            exit(1);
        }
    }

    f32_to_bf16_pad_kernel<<<(M_pad * K_pad + 255) / 256, 256>>>(
        x, g_act_bf16, M_pad, n, K_pad, M);

    if (fp4_gemm_run_cached(g_act_bf16, weight_cache, NULL, g_out_bf16,
                            M_pad, 1.0f, 0.0f) != 0) {
        fprintf(stderr, "fp4_qwen3_mm: gemm failed M=%d n=%d d=%d\n", M, n, d);
        exit(1);
    }
    fp4_gemm_sync();

    bf16_to_f32_trunc_kernel<<<(M * d + 255) / 256, 256>>>(
        g_out_bf16, y, M, d, N_pad);
}
