/*
 * Qwen3-VL GPU forward — CUDA kernels。
 * FP16 GEMV 線形層 + Flash Attention（online softmax、GQA 対応）。
 * RoPE: 標準 RoPE（隣接ペア回転）。
 */

#include "gpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef BONSAI_FP4
#include "fp4_qwen3.h"
#include "fp4_gemm.h"
/* CUTLASS NVFP4 GEMM requires M>=128 for prefill; M<128 uses FP4 GEMV (decode). */
#endif

#ifdef BONSAI_POLARQUANT
#include "polarquant.h"
#include "polarquant_kernels.cuh"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DT_F32   0
#define DT_F16   1

/* Flash Attention: K/V タイルを shared memory に staging（head_dim=128 固定）。
 * shared ≈ 2×FA_BR×FA_HD + q/o/scores/red。
 * FA_BR=64 → ≈65 KB（Ampere/Ada は carveout で 48 KB 超可）。
 * FA_BR=32 → ≈34 KB（Blackwell sm_120 の静的 shared 上限 48 KB 以内）。 */
#ifndef FA_BR
#define FA_BR  64
#endif
#define FA_HD  128   /* Qwen3-8B head_dim */

static __device__ float fa_sh_reduce_max(float val, float *red_sh)
{
    red_sh[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red_sh[threadIdx.x] = fmaxf(red_sh[threadIdx.x], red_sh[threadIdx.x + s]);
        __syncthreads();
    }
    return red_sh[0];
}

static __device__ float fa_sh_reduce_sum(float val, float *red_sh)
{
    red_sh[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red_sh[threadIdx.x] += red_sh[threadIdx.x + s];
        __syncthreads();
    }
    return red_sh[0];
}

/*
 * GQA デコード Attention（1 クエリ位置 × 全ヘッド）。
 * Flash Attention の online softmax。K/V タイルを shared に staging してから QK^T / PV。
 * grid: n_heads blocks, block: FA_HD threads。
 */
#ifdef BONSAI_POLARQUANT
static __global__ void flash_attn_gqa_pq_kernel(float *xb, const float *q,
    const PQBlock *kc_pq, const PQBlock *vc_pq, const PQState *pq_st,
    int npos, int n_heads, int hd, int n_kv, int kv_mul, float scale)
{
    int h = blockIdx.x;
    if (h >= n_heads || hd > FA_HD || !pq_st) return;

    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * hd;
    const PQBlock *kbase = kc_pq + (size_t)kvh * PQ_NBLK;
    const PQBlock *vbase = vc_pq + (size_t)kvh * PQ_NBLK;
    float *oh = xb + (size_t)h * hd;

    __shared__ float k_tile[FA_BR][FA_HD];
    __shared__ float v_tile[FA_BR][FA_HD];
    __shared__ float q_sh[FA_HD];
    __shared__ float o_sh[FA_HD];
    __shared__ float scores[FA_BR];
    __shared__ float red_sh[FA_HD];

    if (threadIdx.x < hd) {
        q_sh[threadIdx.x] = qh[threadIdx.x];
        o_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        for (int j = 0; j < tc; j++) {
            if (threadIdx.x == 0) {
                pq_decode_head(pq_st,
                    kbase + (size_t)(t0 + j) * n_kv * PQ_NBLK,
                    k_tile[j]);
                pq_decode_head(pq_st,
                    vbase + (size_t)(t0 + j) * n_kv * PQ_NBLK,
                    v_tile[j]);
            }
            __syncthreads();
        }

        if (threadIdx.x < tc) {
            float s = 0.0f;
            for (int d = 0; d < hd; d++)
                s += q_sh[d] * k_tile[threadIdx.x][d];
            scores[threadIdx.x] = s * scale;
        }
        __syncthreads();

        float m_tile = fa_sh_reduce_max(
            (threadIdx.x < tc) ? scores[threadIdx.x] : -1e30f, red_sh);
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;

        if (threadIdx.x < hd)
            o_sh[threadIdx.x] *= alpha;

        if (threadIdx.x < tc)
            scores[threadIdx.x] = expf(scores[threadIdx.x] - m_new);
        __syncthreads();

        float l_tile = fa_sh_reduce_sum(
            (threadIdx.x < tc) ? scores[threadIdx.x] : 0.0f, red_sh);
        __syncthreads();

        if (threadIdx.x < hd) {
            float acc = 0.0f;
            for (int j = 0; j < tc; j++)
                acc += scores[j] * v_tile[j][threadIdx.x];
            o_sh[threadIdx.x] += acc;
        }
        __syncthreads();

        l = l * alpha + l_tile;
        m = m_new;
    }

    if (threadIdx.x < hd)
        oh[threadIdx.x] = o_sh[threadIdx.x] / l;
}
#endif

static __global__ void flash_attn_gqa_kernel(float *xb, const float *q,
    const float *kc, const float *vc, int npos, int n_heads, int hd,
    int kv_dim, int kv_mul, float scale)
{
    int h = blockIdx.x;
    if (h >= n_heads || hd > FA_HD) return;

    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * hd;
    const float *kbase = kc + (size_t)kvh * hd;
    const float *vbase = vc + (size_t)kvh * hd;
    float *oh = xb + (size_t)h * hd;

    __shared__ float k_tile[FA_BR][FA_HD];
    __shared__ float v_tile[FA_BR][FA_HD];
    __shared__ float q_sh[FA_HD];
    __shared__ float o_sh[FA_HD];
    __shared__ float scores[FA_BR];
    __shared__ float red_sh[FA_HD];

    if (threadIdx.x < hd) {
        q_sh[threadIdx.x] = qh[threadIdx.x];
        o_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        /* K/V タイルを協調ロード（global はここだけ。以降は shared 参照） */
        for (int idx = threadIdx.x; idx < tc * hd; idx += blockDim.x) {
            int j = idx / hd;
            int d = idx - j * hd;
            k_tile[j][d] = kbase[(size_t)(t0 + j) * kv_dim + d];
            v_tile[j][d] = vbase[(size_t)(t0 + j) * kv_dim + d];
        }
        __syncthreads();

        if (threadIdx.x < tc) {
            float s = 0.0f;
            for (int d = 0; d < hd; d++)
                s += q_sh[d] * k_tile[threadIdx.x][d];
            scores[threadIdx.x] = s * scale;
        }
        __syncthreads();

        float m_tile = fa_sh_reduce_max(
            (threadIdx.x < tc) ? scores[threadIdx.x] : -1e30f, red_sh);
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;

        if (threadIdx.x < hd)
            o_sh[threadIdx.x] *= alpha;

        if (threadIdx.x < tc)
            scores[threadIdx.x] = expf(scores[threadIdx.x] - m_new);
        __syncthreads();

        float l_tile = fa_sh_reduce_sum(
            (threadIdx.x < tc) ? scores[threadIdx.x] : 0.0f, red_sh);
        __syncthreads();

        if (threadIdx.x < hd) {
            float acc = 0.0f;
            for (int j = 0; j < tc; j++)
                acc += scores[j] * v_tile[j][threadIdx.x];
            o_sh[threadIdx.x] += acc;
        }
        __syncthreads();

        l = l * alpha + l_tile;
        m = m_new;
    }

    if (threadIdx.x < hd)
        oh[threadIdx.x] = o_sh[threadIdx.x] / l;
}

/*
 * Prefill Attention — 各プロンプト位置 t を block (t, head) で並列。
 * 因果マスク: 位置 t は K/V の 0..t のみ参照（npos = t + 1）。
 */
#ifdef BONSAI_POLARQUANT
static __global__ void flash_attn_prefill_gqa_pq_kernel(float *xb, const float *q,
    const PQBlock *kc_pq, const PQBlock *vc_pq, const PQState *pq_st,
    int n_tokens, int n_heads, int hd, int n_kv, int kv_mul, float scale)
{
    int bt = blockIdx.x;
    int t = bt / n_heads;
    int h = bt % n_heads;
    if (t >= n_tokens || h >= n_heads || hd > FA_HD || !pq_st) return;

    const int npos = t + 1;
    int kvh = h / kv_mul;
    const float *qh = q + ((size_t)t * n_heads + h) * hd;
    const PQBlock *kbase = kc_pq + (size_t)kvh * PQ_NBLK;
    const PQBlock *vbase = vc_pq + (size_t)kvh * PQ_NBLK;
    float *oh = xb + ((size_t)t * n_heads + h) * hd;

    __shared__ float k_tile[FA_BR][FA_HD];
    __shared__ float v_tile[FA_BR][FA_HD];
    __shared__ float q_sh[FA_HD];
    __shared__ float o_sh[FA_HD];
    __shared__ float scores[FA_BR];
    __shared__ float red_sh[FA_HD];

    if (threadIdx.x < hd) {
        q_sh[threadIdx.x] = qh[threadIdx.x];
        o_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        for (int j = 0; j < tc; j++) {
            if (threadIdx.x == 0) {
                pq_decode_head(pq_st,
                    kbase + (size_t)(t0 + j) * n_kv * PQ_NBLK,
                    k_tile[j]);
                pq_decode_head(pq_st,
                    vbase + (size_t)(t0 + j) * n_kv * PQ_NBLK,
                    v_tile[j]);
            }
            __syncthreads();
        }

        if (threadIdx.x < tc) {
            float s = 0.0f;
            for (int d = 0; d < hd; d++)
                s += q_sh[d] * k_tile[threadIdx.x][d];
            scores[threadIdx.x] = s * scale;
        }
        __syncthreads();

        float m_tile = fa_sh_reduce_max(
            (threadIdx.x < tc) ? scores[threadIdx.x] : -1e30f, red_sh);
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;

        if (threadIdx.x < hd)
            o_sh[threadIdx.x] *= alpha;

        if (threadIdx.x < tc)
            scores[threadIdx.x] = expf(scores[threadIdx.x] - m_new);
        __syncthreads();

        float l_tile = fa_sh_reduce_sum(
            (threadIdx.x < tc) ? scores[threadIdx.x] : 0.0f, red_sh);
        __syncthreads();

        if (threadIdx.x < hd) {
            float acc = 0.0f;
            for (int j = 0; j < tc; j++)
                acc += scores[j] * v_tile[j][threadIdx.x];
            o_sh[threadIdx.x] += acc;
        }
        __syncthreads();

        l = l * alpha + l_tile;
        m = m_new;
    }

    if (threadIdx.x < hd)
        oh[threadIdx.x] = o_sh[threadIdx.x] / l;
}
#endif

static __global__ void flash_attn_prefill_gqa_kernel(float *xb, const float *q,
    const float *kc, const float *vc, int n_tokens, int n_heads, int hd,
    int kv_dim, int kv_mul, float scale)
{
    int bt = blockIdx.x;
    int t = bt / n_heads;
    int h = bt % n_heads;
    if (t >= n_tokens || h >= n_heads || hd > FA_HD) return;

    const int npos = t + 1;
    int kvh = h / kv_mul;
    const float *qh = q + ((size_t)t * n_heads + h) * hd;
    const float *kbase = kc + (size_t)kvh * hd;
    const float *vbase = vc + (size_t)kvh * hd;
    float *oh = xb + ((size_t)t * n_heads + h) * hd;

    __shared__ float k_tile[FA_BR][FA_HD];
    __shared__ float v_tile[FA_BR][FA_HD];
    __shared__ float q_sh[FA_HD];
    __shared__ float o_sh[FA_HD];
    __shared__ float scores[FA_BR];
    __shared__ float red_sh[FA_HD];

    if (threadIdx.x < hd) {
        q_sh[threadIdx.x] = qh[threadIdx.x];
        o_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < npos; t0 += FA_BR) {
        int tc = npos - t0;
        if (tc > FA_BR) tc = FA_BR;

        for (int idx = threadIdx.x; idx < tc * hd; idx += blockDim.x) {
            int j = idx / hd;
            int d = idx - j * hd;
            k_tile[j][d] = kbase[(size_t)(t0 + j) * kv_dim + d];
            v_tile[j][d] = vbase[(size_t)(t0 + j) * kv_dim + d];
        }
        __syncthreads();

        if (threadIdx.x < tc) {
            float s = 0.0f;
            for (int d = 0; d < hd; d++)
                s += q_sh[d] * k_tile[threadIdx.x][d];
            scores[threadIdx.x] = s * scale;
        }
        __syncthreads();

        float m_tile = fa_sh_reduce_max(
            (threadIdx.x < tc) ? scores[threadIdx.x] : -1e30f, red_sh);
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;

        if (threadIdx.x < hd)
            o_sh[threadIdx.x] *= alpha;

        if (threadIdx.x < tc)
            scores[threadIdx.x] = expf(scores[threadIdx.x] - m_new);
        __syncthreads();

        float l_tile = fa_sh_reduce_sum(
            (threadIdx.x < tc) ? scores[threadIdx.x] : 0.0f, red_sh);
        __syncthreads();

        if (threadIdx.x < hd) {
            float acc = 0.0f;
            for (int j = 0; j < tc; j++)
                acc += scores[j] * v_tile[j][threadIdx.x];
            o_sh[threadIdx.x] += acc;
        }
        __syncthreads();

        l = l * alpha + l_tile;
        m = m_new;
    }

    if (threadIdx.x < hd)
        oh[threadIdx.x] = o_sh[threadIdx.x] / l;
}

static void flash_attn_init_once(void)
{
    static int done = 0;
    if (done) return;
    cudaError_t err;
    err = cudaFuncSetAttribute(
        flash_attn_gqa_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (err != cudaSuccess)
        fprintf(stderr, "Warning: flash_attn shared carveout: %s\n", cudaGetErrorString(err));
    err = cudaFuncSetAttribute(
        flash_attn_prefill_gqa_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (err != cudaSuccess)
        fprintf(stderr, "Warning: flash_attn_prefill shared carveout: %s\n", cudaGetErrorString(err));
    done = 1;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static __device__ float dev_f16f32(uint16_t h)
{
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

#define GEMV_WARP 32
#define GEMV_ROWS_PER_BLOCK 8
#define GEMV_THREADS (GEMV_WARP * GEMV_ROWS_PER_BLOCK)

__launch_bounds__(GEMV_THREADS)
static __global__ void mm_f16_gemv_kernel(float *o, const float *x, const uint16_t *w, int n, int d)
{
    int local_row = threadIdx.x / GEMV_WARP;
    int lane      = threadIdx.x % GEMV_WARP;
    int row       = blockIdx.x * GEMV_ROWS_PER_BLOCK + local_row;
    if (row >= d) return;

    const uint16_t *roww = w + (size_t)row * (size_t)n;
    float val = 0.f;
    int n4 = n >> 2;
    for (int b = lane; b < n4; b += GEMV_WARP) {
        const uint16_t *wp = roww + b * 4;
        const float    *xp = x    + b * 4;
        val += xp[0] * dev_f16f32(wp[0]);
        val += xp[1] * dev_f16f32(wp[1]);
        val += xp[2] * dev_f16f32(wp[2]);
        val += xp[3] * dev_f16f32(wp[3]);
    }
    for (int j = n4 * 4 + lane; j < n; j += GEMV_WARP)
        val += x[j] * dev_f16f32(roww[j]);
    for (int offset = GEMV_WARP / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset, GEMV_WARP);
    if (lane == 0)
        o[row] = val;
}

static __global__ void mm_f32_gemv_kernel(float *o, const float *x, const float *w, int n, int d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d) return;
    const float *roww = w + (size_t)row * (size_t)n;
    float val = 0.f;
    for (int j = 0; j < n; j++) val += x[j] * roww[j];
    o[row] = val;
}

static __global__ void mm_f16_gemv_batch_kernel(float *o, const float *x, const uint16_t *w,
    int n, int d, int n_tokens)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int t = flat / d;
    int row = flat % d;
    if (t >= n_tokens) return;

    const uint16_t *roww = w + (size_t)row * (size_t)n;
    const float *xin = x + (size_t)t * (size_t)n;
    float val = 0.f;
    for (int j = 0; j < n; j++)
        val += xin[j] * dev_f16f32(roww[j]);
    o[(size_t)t * d + row] = val;
}

static __global__ void emb_f16_kernel(float *o, const uint16_t *w, int id, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t *row = w + (size_t)id * (size_t)dim;
    if (i < dim) o[i] = dev_f16f32(row[i]);
}

static __global__ void emb_f32_kernel(float *o, const float *w, int id, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float *row = w + (size_t)id * (size_t)dim;
    if (i < dim) o[i] = row[i];
}

static __global__ void emb_f16_batch_kernel(float *o, const uint16_t *w, const int *tokens,
    int dim, int n_tokens)
{
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens || i >= dim) return;
    o[(size_t)t * dim + i] = dev_f16f32(w[(size_t)tokens[t] * dim + i]);
}

static __global__ void emb_f32_batch_kernel(float *o, const float *w, const int *tokens,
    int dim, int n_tokens)
{
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens || i >= dim) return;
    o[(size_t)t * dim + i] = w[(size_t)tokens[t] * dim + i];
}

static __global__ void rope_kernel(float *vec, int n_heads, int head_dim, int pos, float theta_base)
{
    int pairs = n_heads * (head_dim / 2);
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    int h  = p / (head_dim / 2);
    int pr = p % (head_dim / 2);
    int i = pr * 2;
    float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
    float angle = (float)pos * freq;
    float cr, ci;
    __sincosf(angle, &ci, &cr);
    int idx = h * head_dim + i;
    float v0 = vec[idx], v1 = vec[idx + 1];
    vec[idx]     = v0 * cr - v1 * ci;
    vec[idx + 1] = v0 * ci + v1 * cr;
}

static __global__ void rope_prefill_batch_kernel(float *vec, int n_heads, int head_dim,
    float theta_base, int n_tokens)
{
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_token = n_heads * (head_dim / 2);
    int total = n_tokens * pairs_per_token;
    if (bt >= total) return;

    int t = bt / pairs_per_token;
    int rem = bt % pairs_per_token;
    int h = rem / (head_dim / 2);
    int pr = rem % (head_dim / 2);
    int pos = t;
    int i = pr * 2;
    float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
    float angle = (float)pos * freq;
    float cr, ci;
    __sincosf(angle, &ci, &cr);
    int idx = ((size_t)t * n_heads + h) * head_dim + i;
    float v0 = vec[idx], v1 = vec[idx + 1];
    vec[idx]     = v0 * cr - v1 * ci;
    vec[idx + 1] = v0 * ci + v1 * cr;
}

static __global__ void rmsnorm_kernel(float *o, const float *x, const float *w, int n, float eps)
{
    __shared__ float shmem[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += x[i] * x[i];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        o[i] = x[i] * inv * w[i];
}

static __global__ void rmsnorm_head_kernel(float *vec, const float *w,
    int n_heads, int hd, float eps)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float *seg = vec + h * hd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        ss += seg[i] * seg[i];
    __shared__ float shmem[256];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        seg[i] *= inv * w[i];
}

static __global__ void rmsnorm_batch_kernel(float *o, const float *x, const float *w,
    int n, int n_tokens, float eps)
{
    int t = blockIdx.x;
    if (t >= n_tokens) return;

    const float *xin = x + (size_t)t * n;
    float *xout = o + (size_t)t * n;

    __shared__ float shmem[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += xin[i] * xin[i];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        xout[i] = xin[i] * inv * w[i];
}

static __global__ void rmsnorm_head_batch_kernel(float *vec, const float *w,
    int n_heads, int hd, int n_tokens, float eps)
{
    int bt = blockIdx.x;
    int t = bt / n_heads;
    int h = bt % n_heads;
    if (t >= n_tokens || h >= n_heads) return;

    float *seg = vec + ((size_t)t * n_heads + h) * hd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        ss += seg[i] * seg[i];
    __shared__ float shmem[256];
    shmem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(shmem[0] / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        seg[i] *= inv * w[i];
}

static __global__ void swiglu_kernel(float *hb, const float *hb2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = hb[i];
    val = val / (1.0f + expf(-val));
    hb[i] = val * hb2[i];
}

static __global__ void swiglu_batch_kernel(float *hb, const float *hb2, int n, int n_tokens)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_tokens;
    if (i >= total) return;
    float val = hb[i];
    val = val / (1.0f + expf(-val));
    hb[i] = val * hb2[i];
}

static __global__ void add_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

static __global__ void add_batch_kernel(float *a, const float *b, int n, int n_tokens)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_tokens;
    if (i >= total) return;
    a[i] += b[i];
}

static __global__ void kv_write_batch_kernel(float *kc, const float *src,
    int kv_dim, int n_tokens)
{
    int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float *srow = src + (size_t)t * kv_dim;
    float *drow = kc + (size_t)t * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x)
        drow[i] = srow[i];
}

typedef struct {
    void *ptr;
} DevBuf;

typedef struct {
    void  **layer;
    int    n_layers;
} DevLayerBuf;

struct GpuModel {
    GpuConfig cfg;

    DevBuf embd;
    int embd_t;

    DevLayerBuf norm_att, q_norm, k_norm, norm_ffn;
    DevLayerBuf wq, wk, wv, wo, gate, up, down;

    DevBuf norm_out;
    DevBuf out;
    int out_t;

    float *x, *xb, *xb2, *hb, *hb2;
    float *q, *k, *v, *logits;
    float *kc, *vc;
#ifdef BONSAI_POLARQUANT
    PQBlock *kc_pq, *vc_pq;
    int use_polarquant;
#endif

    float *x_batch, *xb_batch, *xb2_batch;
    float *q_batch, *k_batch, *v_batch;
    float *hb_batch, *hb2_batch;
    int *tokens_dev;
    int batch_cap;
#ifdef BONSAI_FP4
    int use_fp4;
#endif
};

static DevLayerBuf dev_adopt_layers(int n_layers, void **ptrs)
{
    DevLayerBuf lb = { NULL, n_layers };
    lb.layer = (void **)malloc((size_t)n_layers * sizeof(void *));
    if (ptrs)
        memcpy(lb.layer, ptrs, (size_t)n_layers * sizeof(void *));
    return lb;
}

static void dev_free(DevBuf *b)
{
    if (b && b->ptr) cudaFree(b->ptr);
    if (b) b->ptr = NULL;
}

static void dev_free_layers(DevLayerBuf *lb)
{
    if (!lb) return;
    for (int l = 0; l < lb->n_layers; l++)
        if (lb->layer && lb->layer[l]) cudaFree(lb->layer[l]);
    free(lb->layer);
    lb->layer = NULL;
    lb->n_layers = 0;
}

#ifdef BONSAI_FP4
static void dev_free_fp4_layers(DevLayerBuf *lb)
{
    if (!lb) return;
    for (int l = 0; l < lb->n_layers; l++)
        if (lb->layer && lb->layer[l])
            fp4_qwen3_free_weight(lb->layer[l]);
    free(lb->layer);
    lb->layer = NULL;
    lb->n_layers = 0;
}

static DevLayerBuf dev_adopt_layers_fp4(void **fp4_layers, int n_layers)
{
    DevLayerBuf lb = { NULL, n_layers };
    lb.layer = (void **)calloc((size_t)n_layers, sizeof(void *));
    for (int l = 0; l < n_layers; l++) {
        lb.layer[l] = fp4_layers[l];
        if (!lb.layer[l]) {
            fprintf(stderr, "FP4 weight missing at layer %d\n", l);
            exit(1);
        }
    }
    return lb;
}
#endif

static void launch_mm_f16(float *o, const float *x, const uint16_t *w, int n, int d)
{
    int blocks = (d + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    mm_f16_gemv_kernel<<<blocks, GEMV_THREADS>>>(o, x, w, n, d);
}

static void launch_mm_f32(float *o, const float *x, const float *w, int n, int d)
{
    mm_f32_gemv_kernel<<<(d + 255) / 256, 256>>>(o, x, w, n, d);
}

static void gpu_mm(float *o, const float *x, const DevLayerBuf *W, int wl,
    int n, int d, int type, int M)
{
#ifdef BONSAI_FP4
    if (type == DT_F16) {
        fp4_qwen3_mm(W->layer[wl], x, o, M, n, d);
        return;
    }
#endif
    if (type == DT_F16)
        launch_mm_f16(o, x, (const uint16_t *)W->layer[wl], n, d);
    else if (type == DT_F32)
        launch_mm_f32(o, x, (const float *)W->layer[wl], n, d);
    else {
        fprintf(stderr, "gpu_mm: unsupported weight type %d\n", type);
        exit(1);
    }
}

static void gpu_mm_batch(float *o, const float *x, const DevLayerBuf *W, int wl,
    int n, int d, int type, int n_tokens)
{
#ifdef BONSAI_FP4
    if (type == DT_F16) {
        fp4_qwen3_mm(W->layer[wl], x, o, n_tokens, n, d);
        return;
    }
#endif
    if (type == DT_F16)
        mm_f16_gemv_batch_kernel<<<(n_tokens * d + 255) / 256, 256>>>(
            o, x, (const uint16_t *)W->layer[wl], n, d, n_tokens);
    else if (type == DT_F32) {
        for (int t = 0; t < n_tokens; t++)
            launch_mm_f32(o + (size_t)t * d, x + (size_t)t * n,
                (const float *)W->layer[wl], n, d);
    } else {
        fprintf(stderr, "gpu_mm_batch: unsupported weight type %d\n", type);
        exit(1);
    }
}

GpuModel *gpu_model_create(const GpuConfig *cfg, const GpuWeightsHost *host)
{
    flash_attn_init_once();

    GpuModel *gm = (GpuModel *)calloc(1, sizeof(GpuModel));
    gm->cfg = *cfg;

    const int L = cfg->n_layers;
    const int dim = cfg->dim;
    const int hidden = cfg->hidden_dim;
    const int kv_dim = cfg->kv_dim;
    const int vocab = cfg->vocab_size;
    const int max_seq = cfg->max_seq;
    const int qdim = cfg->n_heads * cfg->head_dim;

    gm->norm_att = dev_adopt_layers(L, host->norm_att);
    gm->q_norm   = dev_adopt_layers(L, host->q_norm);
    gm->k_norm   = dev_adopt_layers(L, host->k_norm);
    gm->norm_ffn = dev_adopt_layers(L, host->norm_ffn);

#ifdef BONSAI_FP4
    {
        int max_M = ((max_seq + 127) / 128) * 128;
        if (max_M < 128) max_M = 128;
        int max_K = ((dim + 127) / 128) * 128;
        if (hidden > max_K) max_K = ((hidden + 127) / 128) * 128;
        int max_N = ((vocab + 127) / 128) * 128;
        if (hidden > max_N) max_N = ((hidden + 127) / 128) * 128;

        if (fp4_qwen3_init(max_M, max_N, max_K) != 0) {
            fprintf(stderr, "FP4 init failed (max_M=%d max_N=%d max_K=%d)\n",
                    max_M, max_N, max_K);
            exit(1);
        }

        gm->wq   = dev_adopt_layers_fp4(host->wq_fp4,   L);
        gm->wk   = dev_adopt_layers_fp4(host->wk_fp4,   L);
        gm->wv   = dev_adopt_layers_fp4(host->wv_fp4,   L);
        gm->wo   = dev_adopt_layers_fp4(host->wo_fp4,   L);
        gm->gate = dev_adopt_layers_fp4(host->gate_fp4, L);
        gm->up   = dev_adopt_layers_fp4(host->up_fp4,   L);
        gm->down = dev_adopt_layers_fp4(host->down_fp4, L);

        gm->out.ptr = host->out_fp4;
        if (!gm->out.ptr) exit(1);
        gm->out_t = host->out_t;
        gm->use_fp4 = 1;
        printf("GPU: FP4 Tensor Core path enabled (GEMM M>=%d, GEMV decode)\n",
               128);
    }
#else
    gm->wq       = dev_adopt_layers(L, host->wq);
    gm->wk       = dev_adopt_layers(L, host->wk);
    gm->wv       = dev_adopt_layers(L, host->wv);
    gm->wo       = dev_adopt_layers(L, host->wo);
    gm->gate     = dev_adopt_layers(L, host->gate);
    gm->up       = dev_adopt_layers(L, host->up);
    gm->down     = dev_adopt_layers(L, host->down);
#endif

    gm->embd.ptr = host->embd;
    gm->embd_t = host->embd_t;
#ifndef BONSAI_FP4
    gm->out.ptr = host->out;
    gm->out_t = host->out_t;
#endif
    gm->norm_out.ptr = host->norm_out;

    CUDA_CHECK(cudaMalloc(&gm->x,      (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->xb,     (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->xb2,    (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->hb,     (size_t)hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->hb2,    (size_t)hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->q,      (size_t)qdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->k,      (size_t)kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->v,      (size_t)kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->logits, (size_t)vocab * sizeof(float)));
#ifdef BONSAI_POLARQUANT
    if (polarquant_init(cfg->head_dim) != 0) {
        fprintf(stderr, "PolarQuant init failed\n");
        exit(1);
    }
    gm->kc_pq = (PQBlock *)polarquant_kv_cache_alloc(L, max_seq, cfg->n_kv_heads);
    gm->vc_pq = (PQBlock *)polarquant_kv_cache_alloc(L, max_seq, cfg->n_kv_heads);
    gm->kc = NULL;
    gm->vc = NULL;
    gm->use_polarquant = 1;
#else
    CUDA_CHECK(cudaMalloc(&gm->kc,     (size_t)L * max_seq * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm->vc,     (size_t)L * max_seq * kv_dim * sizeof(float)));
#endif

    gm->batch_cap = max_seq;
    {
        size_t bc = (size_t)max_seq;
        CUDA_CHECK(cudaMalloc(&gm->x_batch,   bc * (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->xb_batch,  bc * (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->xb2_batch, bc * (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->q_batch,   bc * (size_t)qdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->k_batch,   bc * (size_t)kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->v_batch,   bc * (size_t)kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->hb_batch,  bc * (size_t)hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->hb2_batch, bc * (size_t)hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gm->tokens_dev, bc * sizeof(int)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("GPU: model ready (%d layers, dim=%d, vocab=%d)\n", L, dim, vocab);
    return gm;
}

void gpu_model_destroy(GpuModel *gm)
{
    if (!gm) return;
    dev_free(&gm->embd);
    dev_free_layers(&gm->norm_att);
    dev_free_layers(&gm->q_norm);
    dev_free_layers(&gm->k_norm);
    dev_free_layers(&gm->norm_ffn);
#ifdef BONSAI_FP4
    if (gm->use_fp4) {
        dev_free_fp4_layers(&gm->wq);
        dev_free_fp4_layers(&gm->wk);
        dev_free_fp4_layers(&gm->wv);
        dev_free_fp4_layers(&gm->wo);
        dev_free_fp4_layers(&gm->gate);
        dev_free_fp4_layers(&gm->up);
        dev_free_fp4_layers(&gm->down);
        if (gm->out.ptr) fp4_qwen3_free_weight(gm->out.ptr);
        fp4_qwen3_shutdown();
    } else
#endif
    {
        dev_free_layers(&gm->wq);
        dev_free_layers(&gm->wk);
        dev_free_layers(&gm->wv);
        dev_free_layers(&gm->wo);
        dev_free_layers(&gm->gate);
        dev_free_layers(&gm->up);
        dev_free_layers(&gm->down);
        if (gm->out.ptr && gm->out.ptr != gm->embd.ptr)
            dev_free(&gm->out);
    }
    dev_free(&gm->norm_out);
    cudaFree(gm->x); cudaFree(gm->xb); cudaFree(gm->xb2);
    cudaFree(gm->hb); cudaFree(gm->hb2);
    cudaFree(gm->q); cudaFree(gm->k); cudaFree(gm->v);
    cudaFree(gm->logits);
#ifdef BONSAI_POLARQUANT
    if (gm->use_polarquant) {
        polarquant_kv_cache_free(gm->kc_pq);
        polarquant_kv_cache_free(gm->vc_pq);
        polarquant_shutdown();
    } else
#endif
    {
        cudaFree(gm->kc);
        cudaFree(gm->vc);
    }
    cudaFree(gm->x_batch);
    cudaFree(gm->xb_batch);
    cudaFree(gm->xb2_batch);
    cudaFree(gm->q_batch);
    cudaFree(gm->k_batch);
    cudaFree(gm->v_batch);
    cudaFree(gm->hb_batch);
    cudaFree(gm->hb2_batch);
    cudaFree(gm->tokens_dev);
    free(gm);
}

static void gpu_emb_lookup(GpuModel *gm, int token)
{
    const int dim = gm->cfg.dim;
    if (gm->embd_t == DT_F32)
        emb_f32_kernel<<<(dim + 255) / 256, 256>>>(gm->x, (const float *)gm->embd.ptr, token, dim);
    else
        emb_f16_kernel<<<(dim + 255) / 256, 256>>>(gm->x, (const uint16_t *)gm->embd.ptr, token, dim);
}

static void gpu_mm_out(float *o, const float *x, GpuModel *gm, int n, int d, int M)
{
#ifdef BONSAI_FP4
    if (gm->use_fp4) {
        fp4_qwen3_mm(gm->out.ptr, x, o, M, n, d);
        return;
    }
#endif
    if (gm->out_t == DT_F16)
        launch_mm_f16(o, x, (const uint16_t *)gm->out.ptr, n, d);
    else if (gm->out_t == DT_F32)
        launch_mm_f32(o, x, (const float *)gm->out.ptr, n, d);
    else {
        fprintf(stderr, "gpu_mm_out: unsupported weight type %d\n", gm->out_t);
        exit(1);
    }
}

void gpu_forward(GpuModel *gm, int token, int pos)
{
    GpuConfig *c = &gm->cfg;
    int dim = c->dim, hd = c->head_dim, kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul, n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int max_seq = c->max_seq, hidden = c->hidden_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    const int npos = pos + 1;
    const int wt = DT_F16;

    gpu_emb_lookup(gm, token);

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm_kernel<<<1, 256>>>(gm->xb, gm->x, (float *)gm->norm_att.layer[l], dim, c->norm_eps);

        gpu_mm(gm->q, gm->xb, &gm->wq, l, dim, dim, wt, 1);
        gpu_mm(gm->k, gm->xb, &gm->wk, l, dim, kv_dim, wt, 1);
        gpu_mm(gm->v, gm->xb, &gm->wv, l, dim, kv_dim, wt, 1);

        rmsnorm_head_kernel<<<n_heads, 256>>>(gm->q, (float *)gm->q_norm.layer[l], n_heads, hd, c->norm_eps);
        rmsnorm_head_kernel<<<n_kv, 256>>>(gm->k, (float *)gm->k_norm.layer[l], n_kv, hd, c->norm_eps);

        {
            int pairs = n_heads * (hd / 2);
            rope_kernel<<<(pairs + 255) / 256, 256>>>(gm->q, n_heads, hd, pos, c->rope_theta);
            pairs = n_kv * (hd / 2);
            rope_kernel<<<(pairs + 255) / 256, 256>>>(gm->k, n_kv, hd, pos, c->rope_theta);
        }

#ifdef BONSAI_POLARQUANT
        size_t loff = (size_t)l * max_seq * c->n_kv_heads * PQ_NBLK;
        PQBlock *kc_pos = gm->kc_pq + loff + (size_t)pos * c->n_kv_heads * PQ_NBLK;
        PQBlock *vc_pos = gm->vc_pq + loff + (size_t)pos * c->n_kv_heads * PQ_NBLK;
        polarquant_kv_write_one(kc_pos, gm->k, kv_dim);
        polarquant_kv_write_one(vc_pos, gm->v, kv_dim);

        flash_attn_gqa_pq_kernel<<<n_heads, FA_HD>>>(
            gm->xb, gm->q, gm->kc_pq + loff, gm->vc_pq + loff,
            (const PQState *)polarquant_device_state(),
            npos, n_heads, hd, c->n_kv_heads, kv_mul, scale);
#else
        size_t loff = (size_t)l * max_seq * kv_dim;
        float *kc_pos = gm->kc + loff + (size_t)pos * kv_dim;
        float *vc_pos = gm->vc + loff + (size_t)pos * kv_dim;
        CUDA_CHECK(cudaMemcpy(kc_pos, gm->k, (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(vc_pos, gm->v, (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        flash_attn_gqa_kernel<<<n_heads, FA_HD>>>(
            gm->xb, gm->q, gm->kc + loff, gm->vc + loff,
            npos, n_heads, hd, kv_dim, kv_mul, scale);
#endif

        gpu_mm(gm->xb2, gm->xb, &gm->wo, l, dim, dim, wt, 1);
        add_kernel<<<(dim + 255) / 256, 256>>>(gm->x, gm->xb2, dim);

        rmsnorm_kernel<<<1, 256>>>(gm->xb, gm->x, (float *)gm->norm_ffn.layer[l], dim, c->norm_eps);

        gpu_mm(gm->hb,  gm->xb, &gm->gate, l, dim, hidden, wt, 1);
        gpu_mm(gm->hb2, gm->xb, &gm->up,   l, dim, hidden, wt, 1);

        swiglu_kernel<<<(hidden + 255) / 256, 256>>>(gm->hb, gm->hb2, hidden);

        gpu_mm(gm->xb, gm->hb, &gm->down, l, hidden, dim, wt, 1);
        add_kernel<<<(dim + 255) / 256, 256>>>(gm->x, gm->xb, dim);
    }

    rmsnorm_kernel<<<1, 256>>>(gm->x, gm->x, (float *)gm->norm_out.ptr, dim, c->norm_eps);
    gpu_mm_out(gm->logits, gm->x, gm, dim, c->vocab_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_forward_prefill(GpuModel *gm, const int *tokens, int n_tokens)
{
    if (n_tokens <= 0) return;
    if (n_tokens > gm->batch_cap) {
        fprintf(stderr, "gpu_forward_prefill: n_tokens=%d exceeds batch_cap=%d\n",
            n_tokens, gm->batch_cap);
        exit(1);
    }

    GpuConfig *c = &gm->cfg;
    int dim = c->dim, hd = c->head_dim, kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul, n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int max_seq = c->max_seq, hidden = c->hidden_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    const int wt = DT_F16;

    CUDA_CHECK(cudaMemcpy(gm->tokens_dev, tokens, (size_t)n_tokens * sizeof(int),
        cudaMemcpyHostToDevice));

    if (gm->embd_t == DT_F32)
        emb_f32_batch_kernel<<<dim3((dim + 255) / 256, n_tokens, 1), 256>>>(
            gm->x_batch, (const float *)gm->embd.ptr, gm->tokens_dev, dim, n_tokens);
    else
        emb_f16_batch_kernel<<<dim3((dim + 255) / 256, n_tokens, 1), 256>>>(
            gm->x_batch, (const uint16_t *)gm->embd.ptr, gm->tokens_dev, dim, n_tokens);

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm_batch_kernel<<<n_tokens, 256>>>(
            gm->xb_batch, gm->x_batch, (float *)gm->norm_att.layer[l], dim, n_tokens, c->norm_eps);

        gpu_mm_batch(gm->q_batch, gm->xb_batch, &gm->wq, l, dim, dim, wt, n_tokens);
        gpu_mm_batch(gm->k_batch, gm->xb_batch, &gm->wk, l, dim, kv_dim, wt, n_tokens);
        gpu_mm_batch(gm->v_batch, gm->xb_batch, &gm->wv, l, dim, kv_dim, wt, n_tokens);

        rmsnorm_head_batch_kernel<<<n_tokens * n_heads, 256>>>(
            gm->q_batch, (float *)gm->q_norm.layer[l], n_heads, hd, n_tokens, c->norm_eps);
        rmsnorm_head_batch_kernel<<<n_tokens * n_kv, 256>>>(
            gm->k_batch, (float *)gm->k_norm.layer[l], n_kv, hd, n_tokens, c->norm_eps);

        {
            int total_q = n_tokens * n_heads * (hd / 2);
            rope_prefill_batch_kernel<<<(total_q + 255) / 256, 256>>>(
                gm->q_batch, n_heads, hd, c->rope_theta, n_tokens);
            int total_k = n_tokens * n_kv * (hd / 2);
            rope_prefill_batch_kernel<<<(total_k + 255) / 256, 256>>>(
                gm->k_batch, n_kv, hd, c->rope_theta, n_tokens);
        }

#ifdef BONSAI_POLARQUANT
        size_t loff = (size_t)l * max_seq * c->n_kv_heads * PQ_NBLK;
        PQBlock *kc_pos = gm->kc_pq + loff;
        PQBlock *vc_pos = gm->vc_pq + loff;
        polarquant_kv_write_batch(kc_pos, gm->k_batch, kv_dim, n_tokens);
        polarquant_kv_write_batch(vc_pos, gm->v_batch, kv_dim, n_tokens);

        flash_attn_prefill_gqa_pq_kernel<<<n_tokens * n_heads, FA_HD>>>(
            gm->xb_batch, gm->q_batch, kc_pos, vc_pos,
            (const PQState *)polarquant_device_state(),
            n_tokens, n_heads, hd, c->n_kv_heads, kv_mul, scale);
#else
        size_t loff = (size_t)l * max_seq * kv_dim;
        kv_write_batch_kernel<<<n_tokens, 256>>>(
            gm->kc + loff, gm->k_batch, kv_dim, n_tokens);
        kv_write_batch_kernel<<<n_tokens, 256>>>(
            gm->vc + loff, gm->v_batch, kv_dim, n_tokens);

        flash_attn_prefill_gqa_kernel<<<n_tokens * n_heads, FA_HD>>>(
            gm->xb_batch, gm->q_batch, gm->kc + loff, gm->vc + loff,
            n_tokens, n_heads, hd, kv_dim, kv_mul, scale);
#endif

        gpu_mm_batch(gm->xb2_batch, gm->xb_batch, &gm->wo, l, dim, dim, wt, n_tokens);
        add_batch_kernel<<<(n_tokens * dim + 255) / 256, 256>>>(
            gm->x_batch, gm->xb2_batch, dim, n_tokens);

        rmsnorm_batch_kernel<<<n_tokens, 256>>>(
            gm->xb_batch, gm->x_batch, (float *)gm->norm_ffn.layer[l], dim, n_tokens, c->norm_eps);

        gpu_mm_batch(gm->hb_batch,  gm->xb_batch, &gm->gate, l, dim, hidden, wt, n_tokens);
        gpu_mm_batch(gm->hb2_batch, gm->xb_batch, &gm->up,   l, dim, hidden, wt, n_tokens);

        swiglu_batch_kernel<<<(n_tokens * hidden + 255) / 256, 256>>>(
            gm->hb_batch, gm->hb2_batch, hidden, n_tokens);

        gpu_mm_batch(gm->xb_batch, gm->hb_batch, &gm->down, l, hidden, dim, wt, n_tokens);
        add_batch_kernel<<<(n_tokens * dim + 255) / 256, 256>>>(
            gm->x_batch, gm->xb_batch, dim, n_tokens);
    }

    const float *x_last = gm->x_batch + (size_t)(n_tokens - 1) * dim;
    rmsnorm_kernel<<<1, 256>>>(gm->x, x_last, (float *)gm->norm_out.ptr, dim, c->norm_eps);
    gpu_mm_out(gm->logits, gm->x, gm, dim, c->vocab_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_copy_logits(GpuModel *gm, float *host_logits)
{
    CUDA_CHECK(cudaMemcpy(host_logits, gm->logits,
        (size_t)gm->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
}

static size_t fp16_mat_bytes(int n_out, int n_in)
{
    return (size_t)n_out * (size_t)n_in * sizeof(uint16_t);
}

static size_t weights_embd_bytes(const GpuModel *gm)
{
    size_t nel = (size_t)gm->cfg.vocab_size * (size_t)gm->cfg.dim;
    if (gm->embd_t == DT_F32)
        return nel * sizeof(float);
    return nel * sizeof(uint16_t);
}

static size_t weights_f32_norm_bytes(const GpuModel *gm)
{
    const int L = gm->cfg.n_layers;
    const int dim = gm->cfg.dim;
    const int hd = gm->cfg.head_dim;
    return (size_t)L * (size_t)dim * sizeof(float) * 2 +
           (size_t)L * (size_t)hd * sizeof(float) * 2 +
           (size_t)dim * sizeof(float);
}

#ifdef BONSAI_FP4
static size_t dev_layer_sum_fp4_bytes(const DevLayerBuf *lb)
{
    if (!lb || !lb->layer) return 0;
    size_t sum = 0;
    for (int l = 0; l < lb->n_layers; l++)
        sum += fp4_weight_cache_device_bytes(lb->layer[l]);
    return sum;
}

static size_t weights_fp4_linear_bytes(const GpuModel *gm)
{
    size_t b =
        dev_layer_sum_fp4_bytes(&gm->wq) +
        dev_layer_sum_fp4_bytes(&gm->wk) +
        dev_layer_sum_fp4_bytes(&gm->wv) +
        dev_layer_sum_fp4_bytes(&gm->wo) +
        dev_layer_sum_fp4_bytes(&gm->gate) +
        dev_layer_sum_fp4_bytes(&gm->up) +
        dev_layer_sum_fp4_bytes(&gm->down);
    if (gm->out.ptr)
        b += fp4_weight_cache_device_bytes(gm->out.ptr);
    return b;
}
#endif

static size_t weights_fp16_linear_bytes(const GpuModel *gm)
{
    const int L = gm->cfg.n_layers;
    const int dim = gm->cfg.dim;
    const int hidden = gm->cfg.hidden_dim;
    const int kv_dim = gm->cfg.kv_dim;
    const int vocab = gm->cfg.vocab_size;

    size_t per_layer =
        fp16_mat_bytes(dim, dim) +
        fp16_mat_bytes(kv_dim, dim) +
        fp16_mat_bytes(kv_dim, dim) +
        fp16_mat_bytes(dim, dim) +
        fp16_mat_bytes(hidden, dim) +
        fp16_mat_bytes(hidden, dim) +
        fp16_mat_bytes(dim, hidden);

    size_t b = (size_t)L * per_layer;
    if (gm->out.ptr && gm->out.ptr != gm->embd.ptr)
        b += fp16_mat_bytes(vocab, dim);
    return b;
}

void gpu_model_vram_profile(const GpuModel *gm, GpuVramProfile *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!gm) return;

    const int L = gm->cfg.n_layers;
    const int dim = gm->cfg.dim;
    const int hidden = gm->cfg.hidden_dim;
    const int kv_dim = gm->cfg.kv_dim;
    const int vocab = gm->cfg.vocab_size;
    const int max_seq = gm->cfg.max_seq;
    const int qdim = gm->cfg.n_heads * gm->cfg.head_dim;

    out->weights_embd_bytes = weights_embd_bytes(gm);
    out->weights_f32_norm_bytes = weights_f32_norm_bytes(gm);
#ifdef BONSAI_FP4
    if (gm->use_fp4) {
        out->weights_linear_bytes = weights_fp4_linear_bytes(gm);
        out->fp4_gemm_scratch_bytes =
            fp4_qwen3_vram_bytes() + fp4_gemm_vram_bytes();
    } else
#endif
        out->weights_linear_bytes = weights_fp16_linear_bytes(gm);

#ifdef BONSAI_POLARQUANT
    if (gm->use_polarquant) {
        int bpt = polarquant_bytes_per_token(gm->cfg.n_kv_heads);
        out->kv_cache_bytes =
            (size_t)L * (size_t)max_seq * (size_t)bpt * 2;
    } else
#endif
    {
        out->kv_cache_bytes =
            (size_t)L * (size_t)max_seq * (size_t)kv_dim * sizeof(float) * 2;
    }

    out->decode_activations_bytes =
        (size_t)dim * sizeof(float) * 3 +
        (size_t)hidden * sizeof(float) * 2 +
        (size_t)qdim * sizeof(float) +
        (size_t)kv_dim * sizeof(float) * 2 +
        (size_t)vocab * sizeof(float);

    {
        size_t bc = (size_t)max_seq;
        out->prefill_batch_bytes =
            bc * (size_t)dim * sizeof(float) * 3 +
            bc * (size_t)qdim * sizeof(float) +
            bc * (size_t)kv_dim * sizeof(float) * 2 +
            bc * (size_t)hidden * sizeof(float) * 2 +
            bc * sizeof(int);
    }

    out->total_bytes =
        out->weights_embd_bytes +
        out->weights_f32_norm_bytes +
        out->weights_linear_bytes +
        out->kv_cache_bytes +
        out->decode_activations_bytes +
        out->prefill_batch_bytes +
        out->fp4_gemm_scratch_bytes;

    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        out->device_total_bytes = total_bytes;
        if (total_bytes >= free_bytes)
            out->device_used_bytes = total_bytes - free_bytes;
    }
}

void gpu_get_device_desc(char *buf, size_t cap)
{
    if (!buf || cap == 0) return;
    buf[0] = '\0';
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        snprintf(buf, cap, "unknown");
        return;
    }
    snprintf(buf, cap, "%s (compute %d.%d, %.1f GB)",
             prop.name, prop.major, prop.minor,
             (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

void gpu_print_device_info(void)
{
    char desc[256];
    gpu_get_device_desc(desc, sizeof desc);
    printf("GPU: %s\n", desc);
}
