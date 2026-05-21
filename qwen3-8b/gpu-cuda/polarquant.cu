/*
 * PolarQuant KV cache compression (arxiv:2502.02617).
 * PolarQuant-R: random Hadamard preconditioning + L=4 recursive polar quant.
 */

#include "polarquant.h"
#include "polarquant_kernels.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while (0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static PQState *g_pq_dev = NULL;

static uint64_t pq_xorshift64(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static double pq_pdf_level(int level, double psi)
{
    if (psi < 0.0 || psi > M_PI * 0.5) return 0.0;
    int exp = (1 << (level - 1)) - 1;
    double s = sin(2.0 * psi);
    if (s < 0.0) s = 0.0;
    double p = 1.0;
    for (int i = 0; i < exp; i++) p *= s;
    return p * sin(2.0 * psi);
}

static void pq_lloyd_max_level(float *centroids, int k, int level)
{
    const int grid = 4096;
    double cdf[4097];
    cdf[0] = 0.0;
    double total = 0.0;
    for (int i = 1; i <= grid; i++) {
        double psi = (M_PI * 0.5 * i) / grid;
        total += pq_pdf_level(level, psi);
        cdf[i] = total;
    }
    if (total <= 0.0) {
        for (int j = 0; j < k; j++)
            centroids[j] = (float)((j + 0.5) * M_PI * 0.5 / k);
        return;
    }
    for (int i = 0; i <= grid; i++)
        cdf[i] /= total;

    double edges[k + 1];
    edges[0] = 0.0;
    edges[k] = M_PI * 0.5;
    for (int j = 1; j < k; j++) {
        double target = (double)j / k;
        int lo = 0, hi = grid;
        while (lo + 1 < hi) {
            int mid = (lo + hi) >> 1;
            if (cdf[mid] < target) lo = mid;
            else hi = mid;
        }
        edges[j] = (M_PI * 0.5 * hi) / grid;
    }

    for (int j = 0; j < k; j++) {
        double num = 0.0, den = 0.0;
        int i0 = (int)(edges[j] / (M_PI * 0.5) * grid);
        int i1 = (int)(edges[j + 1] / (M_PI * 0.5) * grid);
        if (i1 <= i0) i1 = i0 + 1;
        for (int i = i0; i < i1; i++) {
            double psi = (M_PI * 0.5 * (i + 0.5)) / grid;
            double w = pq_pdf_level(level, psi);
            num += psi * w;
            den += w;
        }
        centroids[j] = (float)(den > 0.0 ? num / den : 0.5 * (edges[j] + edges[j + 1]));
    }
}

static void pq_build_codebook(PQCodebook *cb)
{
    for (int k = 0; k < 16; k++)
        cb->centroids_l1[k] = (float)((k + 0.5) * 2.0 * M_PI / 16.0);
    pq_lloyd_max_level(cb->centroids_l2, 4, 2);
    pq_lloyd_max_level(cb->centroids_l3, 4, 3);
    pq_lloyd_max_level(cb->centroids_l4, 4, 4);
}

static void pq_build_signs(float *sign, int n, uint64_t seed)
{
    uint64_t s = seed ? seed : 0x504F4C415251ULL;
    for (int i = 0; i < n; i++)
        sign[i] = (pq_xorshift64(&s) & 1) ? 1.0f : -1.0f;
}

__global__ void pq_kv_encode_kernel(PQBlock *dst, const float *src,
    int kv_dim, int n_tokens, const PQState *st)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_kv = kv_dim / PQ_HD;
    int total = n_tokens * n_kv;
    if (tid >= total) return;

    int t = tid / n_kv;
    int h = tid - t * n_kv;
    const float *vec = src + (size_t)t * kv_dim + (size_t)h * PQ_HD;
    PQBlock *out = dst + (size_t)tid * PQ_NBLK;
    pq_encode_head(st, vec, out);
}

__global__ void pq_kv_encode_one_kernel(PQBlock *dst, const float *src,
    int kv_dim, const PQState *st)
{
    int h = blockIdx.x;
    int n_kv = kv_dim / PQ_HD;
    if (h >= n_kv) return;
    const float *vec = src + (size_t)h * PQ_HD;
    PQBlock *out = dst + (size_t)h * PQ_NBLK;
    pq_encode_head(st, vec, out);
}

extern "C" int polarquant_init(int head_dim)
{
    if (head_dim != PQ_HD) {
        fprintf(stderr, "polarquant_init: head_dim=%d unsupported (need %d)\n",
                head_dim, PQ_HD);
        return -1;
    }
    if (g_pq_dev) return 0;

    PQState host;
    memset(&host, 0, sizeof(host));
    host.inv_sqrt_n = 1.0f / sqrtf((float)PQ_HD);
    pq_build_codebook(&host.cb);
    pq_build_signs(host.sign, PQ_HD, 0x504F4C415251ULL);

    CUDA_CHECK(cudaMalloc(&g_pq_dev, sizeof(PQState)));
    CUDA_CHECK(cudaMemcpy(g_pq_dev, &host, sizeof(PQState), cudaMemcpyHostToDevice));

    printf("PolarQuant-R: KV cache enabled (head_dim=%d, %d bytes/head, ~%.2fx vs F32)\n",
           PQ_HD, PQ_BYTES_HEAD, (float)(PQ_HD * sizeof(float)) / (float)PQ_BYTES_HEAD);
    return 0;
}

extern "C" void polarquant_shutdown(void)
{
    if (g_pq_dev) {
        cudaFree(g_pq_dev);
        g_pq_dev = NULL;
    }
}

extern "C" int polarquant_bytes_per_token(int n_kv_heads)
{
    return n_kv_heads * PQ_BYTES_HEAD;
}

extern "C" int polarquant_head_dim(void)
{
    return PQ_HD;
}

extern "C" void *polarquant_kv_cache_alloc(int n_layers, int max_seq, int n_kv_heads)
{
    size_t bytes = (size_t)n_layers * max_seq * n_kv_heads * PQ_BYTES_HEAD;
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "polarquant_kv_cache_alloc: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    err = cudaMemset(ptr, 0, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "polarquant_kv_cache_alloc memset: %s\n", cudaGetErrorString(err));
        cudaFree(ptr);
        return NULL;
    }
    return ptr;
}

extern "C" void polarquant_kv_cache_free(void *cache)
{
    if (cache) cudaFree(cache);
}

/* Called from kernels.cu */
extern "C" const void *polarquant_device_state(void)
{
    return g_pq_dev;
}

extern "C" void polarquant_kv_write_one(void *dst_blocks, const float *src, int kv_dim)
{
    if (!g_pq_dev) return;
    int n_kv = kv_dim / PQ_HD;
    pq_kv_encode_one_kernel<<<n_kv, 1>>>(
        (PQBlock *)dst_blocks, src, kv_dim, g_pq_dev);
}

extern "C" void polarquant_kv_write_batch(void *dst, const float *src,
    int kv_dim, int n_tokens)
{
    if (!g_pq_dev) return;
    int n_kv = kv_dim / PQ_HD;
    int total = n_tokens * n_kv;
    pq_kv_encode_kernel<<<(total + 63) / 64, 64>>>(
        (PQBlock *)dst, src, kv_dim, n_tokens, g_pq_dev);
}
