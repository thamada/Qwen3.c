#ifndef POLARQUANT_KERNELS_CUH
#define POLARQUANT_KERNELS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PQ_HD       128
#define PQ_BLK      16
#define PQ_LV       4
#define PQ_NBLK     8

typedef struct {
    uint16_t radius;
    uint32_t l1;
    uint8_t  l2;
    uint8_t  l3;
    uint8_t  l4;
    uint8_t  pad;
} PQBlock;

typedef struct {
    float centroids_l1[16];
    float centroids_l2[4];
    float centroids_l3[4];
    float centroids_l4[4];
} PQCodebook;

typedef struct {
    float sign[PQ_HD];
    float inv_sqrt_n;
    PQCodebook cb;
} PQState;

static __device__ __forceinline__ float pq_fp16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)h & 0x8000u) << 16 |
          ((((uint32_t)h & 0x7c00u) != 0 ? 0x38000000u : 0u) + ((uint32_t)h & 0x7fffu)) << 13;
    return v.f;
}

static __device__ __forceinline__ uint16_t pq_f32_to_fp16(float f)
{
    union { float f32; uint32_t u32; } v;
    v.f32 = f;
    uint32_t sign = (v.u32 >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u32 >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u32 & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static __device__ __forceinline__ int pq_quant_nearest(float psi, const float *centroids, int n)
{
    int best = 0;
    float best_d = fabsf(psi - centroids[0]);
    for (int k = 1; k < n; k++) {
        float d = fabsf(psi - centroids[k]);
        if (d < best_d) {
            best_d = d;
            best = k;
        }
    }
    return best;
}

static __device__ __forceinline__ void pq_fwht128(float *a)
{
    for (int len = 1; len < PQ_HD; len <<= 1) {
        for (int i = 0; i < PQ_HD; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = a[i + j];
                float v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
    }
}

static __device__ __forceinline__ void pq_precond_apply(const PQState *st, const float *x, float *y)
{
    float tmp[PQ_HD];
    for (int i = 0; i < PQ_HD; i++)
        tmp[i] = x[i] * st->sign[i];
    pq_fwht128(tmp);
    for (int i = 0; i < PQ_HD; i++)
        y[i] = tmp[i] * st->inv_sqrt_n;
}

static __device__ __forceinline__ void pq_precond_apply_transpose_from(const PQState *st, const float *x, float *y)
{
    float tmp[PQ_HD];
    for (int i = 0; i < PQ_HD; i++)
        tmp[i] = x[i];
    pq_fwht128(tmp);
    for (int i = 0; i < PQ_HD; i++)
        y[i] = tmp[i] * st->sign[i] * st->inv_sqrt_n;
}

static __device__ __forceinline__ void pq_polar_encode_block(const float *in16, PQBlock *out, const PQCodebook *cb)
{
    float r[16];
    for (int i = 0; i < 16; i++) r[i] = in16[i];

    int idx1[8], idx2[4], idx3[2], idx4;
    float psi1[8], psi2[4], psi3[2], psi4;
    float r1[8], r2[4], r3[2], r4;

    for (int j = 0; j < 8; j++) {
        psi1[j] = atan2f(r[2 * j + 1], r[2 * j]);
        if (psi1[j] < 0.0f) psi1[j] += 2.0f * (float)M_PI;
        r1[j] = sqrtf(r[2 * j] * r[2 * j] + r[2 * j + 1] * r[2 * j + 1]);
        idx1[j] = pq_quant_nearest(psi1[j], cb->centroids_l1, 16);
    }
    for (int j = 0; j < 4; j++) {
        psi2[j] = atan2f(r1[2 * j + 1], r1[2 * j]);
        if (psi2[j] < 0.0f) psi2[j] = 0.0f;
        if (psi2[j] > (float)M_PI * 0.5f) psi2[j] = (float)M_PI * 0.5f;
        r2[j] = sqrtf(r1[2 * j] * r1[2 * j] + r1[2 * j + 1] * r1[2 * j + 1]);
        idx2[j] = pq_quant_nearest(psi2[j], cb->centroids_l2, 4);
    }
    for (int j = 0; j < 2; j++) {
        psi3[j] = atan2f(r2[2 * j + 1], r2[2 * j]);
        if (psi3[j] < 0.0f) psi3[j] = 0.0f;
        if (psi3[j] > (float)M_PI * 0.5f) psi3[j] = (float)M_PI * 0.5f;
        r3[j] = sqrtf(r2[2 * j] * r2[2 * j] + r2[2 * j + 1] * r2[2 * j + 1]);
        idx3[j] = pq_quant_nearest(psi3[j], cb->centroids_l3, 4);
    }
    psi4 = atan2f(r3[1], r3[0]);
    if (psi4 < 0.0f) psi4 = 0.0f;
    if (psi4 > (float)M_PI * 0.5f) psi4 = (float)M_PI * 0.5f;
    r4 = sqrtf(r3[0] * r3[0] + r3[1] * r3[1]);
    idx4 = pq_quant_nearest(psi4, cb->centroids_l4, 4);

    out->radius = pq_f32_to_fp16(r4);
    out->l1 = 0;
    for (int j = 0; j < 8; j++)
        out->l1 |= (uint32_t)(idx1[j] & 0xF) << (j * 4);
    out->l2 = 0;
    for (int j = 0; j < 4; j++)
        out->l2 |= (uint8_t)((idx2[j] & 3) << (j * 2));
    out->l3 = 0;
    for (int j = 0; j < 2; j++)
        out->l3 |= (uint8_t)((idx3[j] & 3) << (j * 2));
    out->l4 = (uint8_t)(idx4 & 3);
    out->pad = 0;
}

static __device__ __forceinline__ void pq_polar_decode_block(const PQBlock *in, float *out16, const PQCodebook *cb)
{
    int idx1[8], idx2[4], idx3[2], idx4;
    for (int j = 0; j < 8; j++)
        idx1[j] = (int)((in->l1 >> (j * 4)) & 0xF);
    for (int j = 0; j < 4; j++)
        idx2[j] = (int)((in->l2 >> (j * 2)) & 3);
    for (int j = 0; j < 2; j++)
        idx3[j] = (int)((in->l3 >> (j * 2)) & 3);
    idx4 = (int)(in->l4 & 3);

    float r4 = pq_fp16_to_f32(in->radius);
    float th4 = cb->centroids_l4[idx4];
    float r3[2];
    r3[0] = r4 * cosf(th4);
    r3[1] = r4 * sinf(th4);

    float r2[4];
    for (int j = 0; j < 2; j++) {
        float th = cb->centroids_l3[idx3[j]];
        r2[2 * j]     = r3[j] * cosf(th);
        r2[2 * j + 1] = r3[j] * sinf(th);
    }

    float r1[8];
    for (int j = 0; j < 4; j++) {
        float th = cb->centroids_l2[idx2[j]];
        r1[2 * j]     = r2[j] * cosf(th);
        r1[2 * j + 1] = r2[j] * sinf(th);
    }

    for (int j = 0; j < 8; j++) {
        float th = cb->centroids_l1[idx1[j]];
        out16[2 * j]     = r1[j] * cosf(th);
        out16[2 * j + 1] = r1[j] * sinf(th);
    }
}

static __device__ __forceinline__ void pq_encode_head(const PQState *st, const float *x, PQBlock *blocks)
{
    float pre[PQ_HD];
    pq_precond_apply(st, x, pre);
    for (int b = 0; b < PQ_NBLK; b++)
        pq_polar_encode_block(pre + b * PQ_BLK, blocks + b, &st->cb);
}

static __device__ __forceinline__ void pq_decode_head(const PQState *st, const PQBlock *blocks, float *x)
{
    float pre[PQ_HD];
    for (int b = 0; b < PQ_NBLK; b++)
        pq_polar_decode_block(blocks + b, pre + b * PQ_BLK, &st->cb);
    pq_precond_apply_transpose_from(st, pre, x);
}

#endif
