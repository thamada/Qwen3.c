/*
 * Bonsai 向け FP4 ブリッジ: Q1_0 重み変換、128 アライン、F32 入出力。
 */

#include "fp4_bonsai.h"
#include "fp4_gemm.h"
#include "gpu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static __device__ float dev_f16f32(uint16_t h)
{
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

static __global__ void dequant_q1_0_to_bf16_kernel(
    const GpuBlockQ1_0 *W, __nv_bfloat16 *out,
    int N, int K, int K_pad, size_t row_stride)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const GpuBlockQ1_0 *wrow =
        (const GpuBlockQ1_0 *)((const char *)W + row * row_stride);
    int nb = K / GPU_QK1_0;
    __nv_bfloat16 *orow = out + (size_t)row * K_pad;

    for (int ib = 0; ib < nb; ib++) {
        float d = dev_f16f32(wrow[ib].d);
        for (int k = 0; k < GPU_QK1_0; k++) {
            int bit = (wrow[ib].qs[k / 8] >> (k % 8)) & 1;
            float val = bit ? d : -d;
            orow[ib * GPU_QK1_0 + k] = __float2bfloat16_rn(val);
        }
    }
    for (int k = K; k < K_pad; k++)
        orow[k] = __float2bfloat16_rn(0.0f);
}

static __global__ void f32_to_bf16_pad_kernel(
    const float *src, __nv_bfloat16 *dst, int M, int n, int K_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * K_pad;
    if (idx >= total) return;
    int m = idx / K_pad;
    int k = idx - m * K_pad;
    float v = (k < n) ? src[(size_t)m * n + k] : 0.0f;
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

int fp4_bonsai_init(int max_M, int max_N, int max_K)
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

void fp4_bonsai_shutdown(void)
{
    fp4_gemm_cleanup();
    if (g_act_bf16) cudaFree(g_act_bf16);
    if (g_out_bf16) cudaFree(g_out_bf16);
    g_act_bf16 = NULL;
    g_out_bf16 = NULL;
    g_act_cap = g_out_cap = 0;
    g_max_M = g_max_N = g_max_K = 0;
}

void *fp4_bonsai_weight_from_q1_host(const void *host_q1, int N, int K,
                                      size_t row_stride)
{
    int N_pad = align128(N);
    int K_pad = align128(K);

    GpuBlockQ1_0 *dev_q1 = NULL;
    __nv_bfloat16 *dev_bf16 = NULL;
    size_t q1_bytes = (size_t)N * row_stride;
    CUDA_CHECK(cudaMalloc(&dev_q1, q1_bytes));
    CUDA_CHECK(cudaMemcpy(dev_q1, host_q1, q1_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev_bf16, (size_t)N_pad * K_pad * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dev_bf16, 0, (size_t)N_pad * K_pad * sizeof(__nv_bfloat16)));

    dequant_q1_0_to_bf16_kernel<<<(N + 255) / 256, 256>>>(
        dev_q1, dev_bf16, N, K, K_pad, row_stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    void *cache = fp4_quantize_weights(dev_bf16, N_pad, K_pad);

    cudaFree(dev_q1);
    cudaFree(dev_bf16);

    if (!cache) {
        fprintf(stderr, "fp4_bonsai_weight_from_q1_host: quantize failed N=%d K=%d\n",
                N_pad, K_pad);
    }
    return cache;
}

void fp4_bonsai_free_weight(void *cache)
{
    fp4_weight_cache_free(cache);
}

void fp4_bonsai_mm(const void *weight_cache,
                   const float *x, float *y,
                   int M, int n, int d)
{
    if (!weight_cache) {
        fprintf(stderr, "fp4_bonsai_mm: null weight cache\n");
        exit(1);
    }

    int K_pad = fp4_weight_cache_K(weight_cache);
    int N_pad = fp4_weight_cache_N(weight_cache);
    int M_pad = align128(M);

    if (M_pad > g_max_M || N_pad > g_max_N || K_pad > g_max_K) {
        if (fp4_bonsai_init(M_pad > g_max_M ? M_pad : g_max_M,
                            N_pad > g_max_N ? N_pad : g_max_N,
                            K_pad > g_max_K ? K_pad : g_max_K) != 0) {
            fprintf(stderr, "fp4_bonsai_mm: init failed\n");
            exit(1);
        }
    }

    int act_elems = M_pad * K_pad;
    int out_elems = M_pad * N_pad;
    f32_to_bf16_pad_kernel<<<(act_elems + 255) / 256, 256>>>(
        x, g_act_bf16, M_pad, n, K_pad);

    if (fp4_gemm_run_cached(g_act_bf16, weight_cache, NULL, g_out_bf16,
                            M_pad, 1.0f, 0.0f) != 0) {
        fprintf(stderr, "fp4_bonsai_mm: gemm failed M=%d n=%d d=%d\n", M, n, d);
        exit(1);
    }
    fp4_gemm_sync();

    bf16_to_f32_trunc_kernel<<<(M * d + 255) / 256, 256>>>(
        g_out_bf16, y, M, d, N_pad);
    CUDA_CHECK(cudaDeviceSynchronize());
}
