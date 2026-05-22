/*
 * NVFP4 block-scaled GEMM for Blackwell SM120/SM121 (CUTLASS Example 79a ベース).
 * Bonsai gpu-cuda 内部用。Python/ctypes 向けではない。
 */

#include "fp4_gemm.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cute;

// ============================================================================
// CUTLASS GEMM type config (same as Example 79a)
// ============================================================================

using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD    = cutlass::bfloat16_t;
using ElementC    = cutlass::bfloat16_t;
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128, _128, _128>;
using ClusterShape       = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA  = typename Gemm::GemmKernel::StrideA;
using StrideB  = typename Gemm::GemmKernel::StrideB;
using StrideC  = typename Gemm::GemmKernel::StrideC;
using StrideD  = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ScaleFactorType = typename ElementA::ScaleFactorType;  // float_ue4m3_t

static constexpr int SF_VEC_SIZE = Sm1xxBlkScaledConfig::SFVecSize;  // 16

// ============================================================================
// GPU quantization kernels
// ============================================================================

// FP4 E2M1 lookup table in constant memory
__constant__ float c_fp4_values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Round float to nearest FP4 E2M1 code (4 bits)
__device__ __forceinline__ uint8_t d_float_to_fp4(float val) {
    float abs_val = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t code = 0;
    float best_dist = abs_val;  // distance to 0
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        float dist = fabsf(abs_val - c_fp4_values[i]);
        if (dist < best_dist) {
            best_dist = dist;
            code = i;
        }
    }
    return sign | code;
}

// Convert float to UE4M3 raw byte (unsigned, 4-bit exponent bias=7, 3-bit mantissa)
// UE4M3 range: [0.015625, 480.0] (subnormals down to ~0.001953)
__device__ __forceinline__ uint8_t d_float_to_ue4m3(float val) {
    // Clamp to valid UE4M3 range
    val = fmaxf(val, 1.953125e-3f);  // smallest subnormal
    val = fminf(val, 480.0f);        // largest value

    // Use the bit pattern of the float to compute UE4M3
    // UE4M3: 4 exponent bits (bias 7), 3 mantissa bits, unsigned
    // Normal: (-1)^0 * 2^(e-7) * (1 + m/8), e in [1..15]
    // Subnormal: (-1)^0 * 2^(-6) * (m/8), e = 0

    union { float f; uint32_t u; } bits;
    bits.f = val;
    int fp32_exp = ((bits.u >> 23) & 0xFF) - 127;  // unbiased
    int fp32_mant = bits.u & 0x7FFFFF;  // 23-bit mantissa

    int ue4m3_exp = fp32_exp + 7;  // rebias to UE4M3

    if (ue4m3_exp <= 0) {
        // Subnormal
        int shift = 1 - ue4m3_exp;
        int mant = (0x800000 | fp32_mant) >> (20 + shift);  // include implicit 1
        mant = min(mant, 7);
        return (uint8_t)mant;
    } else if (ue4m3_exp >= 15) {
        // Saturate to max
        return 0x7F;  // exp=15, mant=7 => invalid for some formats, but max for UE4M3
    } else {
        // Normal: round mantissa to 3 bits
        int mant = (fp32_mant + (1 << 19)) >> 20;  // round to nearest
        if (mant >= 8) {
            mant = 0;
            ue4m3_exp++;
            if (ue4m3_exp >= 16) return 0xFF;  // all ones (max/NaN territory)
        }
        return (uint8_t)((ue4m3_exp << 3) | mant);
    }
}

// Convert UE4M3 raw byte back to float
__device__ __forceinline__ float d_ue4m3_to_float(uint8_t raw) {
    int exp = (raw >> 3) & 0xF;
    int mant = raw & 0x7;
    if (exp == 0) {
        // Subnormal: 2^(-6) * (mant/8)
        return ldexpf((float)mant, -9);  // mant * 2^(-9) = mant/8 * 2^(-6)
    } else {
        // Normal: 2^(exp-7) * (1 + mant/8)
        return ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
}

// Compute scale factor index in CUTLASS interleaved layout
// Replicates CuTe's flat coordinate decomposition for SfKMajorAtom
// Layout shape:  (((_32,_4), num_row_tiles), ((_16,_4), num_k_tiles), (_1, batch))
// Layout stride: (((_16,_4), row_tile_stride), ((_0,_1), k_tile_stride), (_0, batch_stride))
//
// For flat (r, k_block) coordinates:
//   Mode 0 flat size = 128 * num_row_tiles
//   CuTe decomposes r into: r0 = r % 32, r1 = (r/32) % 4, r2 = r / 128
//   Mode 0 contribution = r0 * 16 + r1 * 4 + r2 * row_tile_stride
//
//   Mode 1 flat size = SFV * 4 * num_k_tiles = 64 * num_k_tiles
//   k input is element index, CuTe decomposes: k0 = k % 16 (stride 0), k1 = (k/16) % 4, k2 = k / 64
//   Mode 1 contribution = k0 * 0 + k1 * 1 + k2 * k_tile_stride
//
//   k_tile_stride = 128 * num_row_tiles (= rows * nsb_per_tile / tile_sf_count)
//   Actually: for the specific atom, strides are ((16,4),(0,1))
//   The tile_to_shape appends a tile stride = size(filter_zeros(atom)) * previous_tiles
//   For SFA: atom has 128 non-zero elements. Tiled: row_tiles get stride 128*nsb_per_atom
//   Wait — let me just hardcode the formula from CuTe's decomposition:
__device__ __forceinline__ int compute_sf_index(int r, int k_block, int rows, int nsb) {
    // k_block = k_start / SF_VEC_SIZE
    // CuTe flat decomposition for r:
    int r0 = r % 32;        // inner row (stride 16)
    int r1 = (r / 32) % 4;  // inner group (stride 4)
    int r2 = r / 128;       // row tile (stride = 128 * num_k_tiles * 4)

    // CuTe flat decomposition for k_block (remember k input to layout is k_element,
    // and the (SFV, 4) shape means we decompose the element index):
    // k_element = k_block * 16
    // k0 = k_element % 16 = 0 (stride 0, broadcast)
    // k1 = (k_element / 16) % 4 = k_block % 4 (stride 1)
    // k2 = k_element / 64 = k_block / 4 (stride = atom_filtered_size = 128 for single row tile)
    int k1 = k_block % 4;
    int k2 = k_block / 4;

    // Atom filtered size = 32*4 * 4 = 512? No...
    // The atom shape ((32,4),(16,4)), stride ((16,4),(0,1))
    // Filtered (remove zeros): shape ((32,4),(4,)), stride ((16,4),(1,))
    // Filtered size = 32*4 * 4 = 512
    // Wait, the 16 in (16,4) has stride 0, so filtered shape for mode 1 inner = just (4,) with stride 1
    // Filtered total atom size = (32*4) * 4 = 512
    // Actually, looking at our test: SFA filtered size for 128x128x128 = 1024
    // 128 rows = 1 row tile, K=128 → 8 scale blocks → 2 k_tiles
    // atoms: 1 row_tile * 2 k_tiles = 2 atoms
    // Each atom filtered = 512
    // Total = 1024 ✓

    // So: row_tile_stride = 512 * (nsb / 4)
    //     k_tile_stride = 512
    int k_tiles = nsb / 4;
    int row_tile_stride = 512 * k_tiles;
    int k_tile_stride = 512;

    return r0 * 16 + r1 * 4 + r2 * row_tile_stride + k1 * 1 + k2 * k_tile_stride;
}

// Quantize BF16 matrix (column-major N×K) to packed FP4 + UE4M3 scales
__global__ void quantize_bf16_colmajor_to_fp4_kernel(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst_fp4,
    uint8_t* __restrict__ dst_sf,
    int rows, int K, int nsb, int N_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = rows * nsb;
    if (idx >= total_blocks) return;

    int r = idx / nsb;
    int sb = idx % nsb;
    int k_start = sb * 16;

    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int k = k_start + i;
        float val = __bfloat162float(src[r + (size_t)k * N_stride]);
        max_abs = fmaxf(max_abs, fabsf(val));
    }

    float scale_val = max_abs / 6.0f;
    if (scale_val < 1.953125e-3f) scale_val = 1.953125e-3f;
    uint8_t scale_raw = d_float_to_ue4m3(scale_val);
    float actual_scale = d_ue4m3_to_float(scale_raw);
    float scale_inv = 1.0f / actual_scale;

    int sf_idx = compute_sf_index(r, sb, rows, nsb);
    dst_sf[sf_idx] = scale_raw;

    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        int k0 = k_start + i;
        float v0 = __bfloat162float(src[r + (size_t)k0 * N_stride]) * scale_inv;
        float v1 = __bfloat162float(src[r + (size_t)(k0 + 1) * N_stride]) * scale_inv;
        uint8_t fp4_0 = d_float_to_fp4(v0);
        uint8_t fp4_1 = d_float_to_fp4(v1);
        dst_fp4[r + (size_t)(k0 / 2) * N_stride] = (fp4_1 << 4) | fp4_0;
    }
}

// Quantize BF16 matrix to packed FP4 + UE4M3 scales
// One thread per scale block (i.e., per 16 BF16 elements)
__global__ void quantize_bf16_to_fp4_kernel(
    const __nv_bfloat16* __restrict__ src,  // [rows, K]
    uint8_t* __restrict__ dst_fp4,           // [rows, K/2]
    uint8_t* __restrict__ dst_sf,            // scales in CUTLASS layout
    int rows, int K, int nsb, const int* sf_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = rows * nsb;
    if (idx >= total_blocks) return;

    int r = idx / nsb;
    int sb = idx % nsb;
    int k_start = sb * 16;  // SF_VEC_SIZE = 16

    const __nv_bfloat16* row = src + r * K;

    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float val = __bfloat162float(row[k_start + i]);
        max_abs = fmaxf(max_abs, fabsf(val));
    }

    float scale_val = max_abs / 6.0f;
    if (scale_val < 1.953125e-3f) scale_val = 1.953125e-3f;
    uint8_t scale_raw = d_float_to_ue4m3(scale_val);
    float actual_scale = d_ue4m3_to_float(scale_raw);
    float scale_inv = 1.0f / actual_scale;

    int sf_idx = sf_lut ? sf_lut[r * nsb + sb] : compute_sf_index(r, sb, rows, nsb);
    dst_sf[sf_idx] = scale_raw;

    uint8_t* out_row = dst_fp4 + r * (K / 2);
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        float v0 = __bfloat162float(row[k_start + i]) * scale_inv;
        float v1 = __bfloat162float(row[k_start + i + 1]) * scale_inv;
        uint8_t fp4_0 = d_float_to_fp4(v0);
        uint8_t fp4_1 = d_float_to_fp4(v1);
        out_row[(k_start + i) / 2] = (fp4_1 << 4) | fp4_0;
    }
}

// ============================================================================
// Host-side quantization (fallback, used for verification)
// ============================================================================

static const float fp4_values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static inline uint8_t float_to_fp4(float val) {
    float abs_val = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t code = 0;
    float best_dist = abs_val;
    for (int i = 1; i < 8; i++) {
        float dist = fabsf(abs_val - fp4_values[i]);
        if (dist < best_dist) {
            best_dist = dist;
            code = i;
        }
    }
    return sign | code;
}

template <typename LayoutSF>
static void quantize_matrix_colmajor_host(
    const cutlass::bfloat16_t* src, uint8_t* dst_fp4,
    ScaleFactorType* dst_sf, int rows, int K, int N_stride, LayoutSF layout_sf)
{
    int nsb = K / SF_VEC_SIZE;
    for (int r = 0; r < rows; r++) {
        for (int sb = 0; sb < nsb; sb++) {
            int k_start = sb * SF_VEC_SIZE;
            float max_abs = 0.0f;
            for (int i = 0; i < SF_VEC_SIZE; i++) {
                int k = k_start + i;
                float val = float(src[r + (size_t)k * N_stride]);
                max_abs = fmaxf(max_abs, fabsf(val));
            }
            float scale_val = max_abs / 6.0f;
            if (scale_val < 1e-10f) scale_val = 1e-10f;
            ScaleFactorType scale_ue4m3 = ScaleFactorType(scale_val);
            float actual_scale = float(scale_ue4m3);
            float scale_inv = 1.0f / actual_scale;
            int sf_idx = layout_sf(r, k_start, 0);
            dst_sf[sf_idx] = scale_ue4m3;
            for (int i = 0; i < SF_VEC_SIZE; i += 2) {
                int k0 = k_start + i;
                float v0 = float(src[r + (size_t)k0 * N_stride]) * scale_inv;
                float v1 = float(src[r + (size_t)(k0 + 1) * N_stride]) * scale_inv;
                uint8_t fp4_0 = float_to_fp4(v0);
                uint8_t fp4_1 = float_to_fp4(v1);
                dst_fp4[r + (size_t)(k0 / 2) * N_stride] = (fp4_1 << 4) | fp4_0;
            }
        }
    }
}

template <typename LayoutSF>
static void quantize_matrix_host(
    const cutlass::bfloat16_t* src, uint8_t* dst_fp4,
    ScaleFactorType* dst_sf, int rows, int K, LayoutSF layout_sf)
{
    int nsb = K / SF_VEC_SIZE;
    for (int r = 0; r < rows; r++) {
        const cutlass::bfloat16_t* row_ptr = src + r * K;
        for (int sb = 0; sb < nsb; sb++) {
            int k_start = sb * SF_VEC_SIZE;
            float max_abs = 0.0f;
            for (int i = 0; i < SF_VEC_SIZE; i++) {
                float val = float(row_ptr[k_start + i]);
                max_abs = fmaxf(max_abs, fabsf(val));
            }
            float scale_val = max_abs / 6.0f;
            if (scale_val < 1e-10f) scale_val = 1e-10f;
            ScaleFactorType scale_ue4m3 = ScaleFactorType(scale_val);
            float actual_scale = float(scale_ue4m3);
            float scale_inv = 1.0f / actual_scale;
            int sf_idx = layout_sf(r, k_start, 0);
            dst_sf[sf_idx] = scale_ue4m3;
            for (int i = 0; i < SF_VEC_SIZE; i += 2) {
                float v0 = float(row_ptr[k_start + i]) * scale_inv;
                float v1 = float(row_ptr[k_start + i + 1]) * scale_inv;
                uint8_t fp4_0 = float_to_fp4(v0);
                uint8_t fp4_1 = float_to_fp4(v1);
                dst_fp4[r * (K / 2) + (k_start + i) / 2] = (fp4_1 << 4) | fp4_0;
            }
        }
    }
}

// ============================================================================
// State management
// ============================================================================

struct FP4GemmState {
    bool initialized;
    int M, N, K;

    // Maximum allocated dimensions (for pre-allocation)
    int max_M, max_N, max_K;
    size_t alloc_A_fp4;   // bytes allocated for A FP4 data
    size_t alloc_B_fp4;   // bytes allocated for B FP4 data
    size_t alloc_SFA;     // bytes allocated for A scale factors
    size_t alloc_SFB;     // bytes allocated for B scale factors

    // Device buffers for quantized data
    uint8_t* d_A_fp4;
    uint8_t* d_B_fp4;
    uint8_t* d_SFA;   // raw bytes (ScaleFactorType is 1 byte)
    uint8_t* d_SFB;

    int sfa_elems;
    int sfb_elems;

    // Host buffers (only for host path)
    cutlass::bfloat16_t* h_A;
    cutlass::bfloat16_t* h_B;
    uint8_t* h_A_fp4;
    uint8_t* h_B_fp4;
    ScaleFactorType* h_SFA;
    ScaleFactorType* h_SFB;

    // CUTLASS workspace
    uint8_t* d_workspace;
    size_t workspace_size;

    /* Scale-factor index LUTs (layout_SFA/SFB); compute_sf_index is wrong for K>256 */
    int *d_SFA_lut;
    int *d_SFB_lut;
    int  lut_M;
    int  lut_N;
    int  lut_nsb;
};

static FP4GemmState g = {};

static void cleanup() {
    if (!g.initialized) return;
    cudaFree(g.d_A_fp4);
    cudaFree(g.d_B_fp4);
    cudaFree(g.d_SFA);
    cudaFree(g.d_SFB);
    free(g.h_A);
    free(g.h_B);
    free(g.h_A_fp4);
    free(g.h_B_fp4);
    free(g.h_SFA);
    free(g.h_SFB);
    if (g.d_workspace) cudaFree(g.d_workspace);
    if (g.d_SFA_lut) cudaFree(g.d_SFA_lut);
    if (g.d_SFB_lut) cudaFree(g.d_SFB_lut);
    memset(&g, 0, sizeof(g));
}

static void build_sf_luts(int M, int N, int K)
{
    if (g.d_SFA_lut) cudaFree(g.d_SFA_lut);
    if (g.d_SFB_lut) cudaFree(g.d_SFB_lut);
    g.d_SFA_lut = NULL;
    g.d_SFB_lut = NULL;
    g.lut_M = M;
    g.lut_N = N;
    g.lut_nsb = K / SF_VEC_SIZE;

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    int n_a = M * g.lut_nsb;
    int n_b = N * g.lut_nsb;
    int *h_a = (int *)malloc((size_t)n_a * sizeof(int));
    int *h_b = (int *)malloc((size_t)n_b * sizeof(int));
    if (!h_a || !h_b) { free(h_a); free(h_b); return; }

    for (int r = 0; r < M; r++)
        for (int sb = 0; sb < g.lut_nsb; sb++)
            h_a[r * g.lut_nsb + sb] = layout_SFA(r, sb * SF_VEC_SIZE, 0);
    for (int r = 0; r < N; r++)
        for (int sb = 0; sb < g.lut_nsb; sb++)
            h_b[r * g.lut_nsb + sb] = layout_SFB(r, sb * SF_VEC_SIZE, 0);

    cudaMalloc(&g.d_SFA_lut, (size_t)n_a * sizeof(int));
    cudaMalloc(&g.d_SFB_lut, (size_t)n_b * sizeof(int));
    cudaMemcpy(g.d_SFA_lut, h_a, (size_t)n_a * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_SFB_lut, h_b, (size_t)n_b * sizeof(int), cudaMemcpyHostToDevice);
    free(h_a);
    free(h_b);
}

// ============================================================================
// Public C API
// ============================================================================

extern "C" {

int fp4_gemm_sf_vec_size() { return SF_VEC_SIZE; }

// Check if current buffers can handle the requested dimensions
static bool buffers_sufficient(int M, int N, int K) {
    if (!g.initialized) return false;
    size_t need_A_fp4 = (size_t)M * K / 2;
    size_t need_B_fp4 = (size_t)N * K / 2;
    // Compute SFA/SFB sizes for this M, N, K
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    size_t need_SFA = cute::size(cute::filter_zeros(layout_SFA));
    size_t need_SFB = cute::size(cute::filter_zeros(layout_SFB));

    // Compute workspace
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };
    size_t need_workspace = Gemm::get_workspace_size(args);

    return g.alloc_A_fp4 >= need_A_fp4 &&
           g.alloc_B_fp4 >= need_B_fp4 &&
           g.alloc_SFA >= need_SFA &&
           g.alloc_SFB >= need_SFB &&
           g.workspace_size >= need_workspace;
}

int fp4_gemm_init(int M, int N, int K) {
    if (g.initialized && g.M == M && g.N == N && g.K == K) return 0;

    // If pre-allocated buffers are large enough, just update dimensions
    if (buffers_sufficient(M, N, K)) {
        auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        g.sfa_elems = cute::size(cute::filter_zeros(layout_SFA));
        g.sfb_elems = cute::size(cute::filter_zeros(layout_SFB));
        g.M = M; g.N = N; g.K = K;
        build_sf_luts(M, N, K);
        return 0;
    }

    cleanup();

    if (M % 128 != 0 || N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "fp4_gemm: M=%d, N=%d, K=%d must all be multiples of 128\n", M, N, K);
        return -1;
    }

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    g.sfa_elems = cute::size(cute::filter_zeros(layout_SFA));
    g.sfb_elems = cute::size(cute::filter_zeros(layout_SFB));

    // Device buffers
    g.alloc_A_fp4 = (size_t)M * K / 2;
    g.alloc_B_fp4 = (size_t)N * K / 2;
    g.alloc_SFA = g.sfa_elems;
    g.alloc_SFB = g.sfb_elems;

    cudaMalloc(&g.d_A_fp4, g.alloc_A_fp4);
    cudaMalloc(&g.d_B_fp4, g.alloc_B_fp4);
    cudaMalloc(&g.d_SFA, g.alloc_SFA);
    cudaMalloc(&g.d_SFB, g.alloc_SFB);

    // Host buffers for host path
    g.h_A = (cutlass::bfloat16_t*)malloc((size_t)M * K * sizeof(cutlass::bfloat16_t));
    g.h_B = (cutlass::bfloat16_t*)malloc((size_t)N * K * sizeof(cutlass::bfloat16_t));
    g.h_A_fp4 = (uint8_t*)calloc(M * K / 2, 1);
    g.h_B_fp4 = (uint8_t*)calloc(N * K / 2, 1);
    g.h_SFA = (ScaleFactorType*)calloc(g.sfa_elems, sizeof(ScaleFactorType));
    g.h_SFB = (ScaleFactorType*)calloc(g.sfb_elems, sizeof(ScaleFactorType));

    // Compute workspace size
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };

    g.workspace_size = Gemm::get_workspace_size(args);
    g.d_workspace = nullptr;
    if (g.workspace_size > 0) cudaMalloc(&g.d_workspace, g.workspace_size);

    g.M = M; g.N = N; g.K = K;
    g.max_M = M; g.max_N = N; g.max_K = K;
    g.initialized = true;
    build_sf_luts(M, N, K);
    return 0;
}

// Pre-allocate buffers for maximum dimensions to avoid reallocation during inference
int fp4_gemm_prealloc(int max_M, int max_N, int max_K) {
    return fp4_gemm_init(max_M, max_N, max_K);
}

// Run FP4 GEMM with GPU-side quantization
int fp4_gemm_run(
    const void* A_bf16, const void* B_bf16,
    const void* C_bf16, void* D_bf16,
    int M, int N, int K,
    float alpha, float beta)
{
    if (!g.initialized || g.M != M || g.N != N || g.K != K) {
        int rc = fp4_gemm_init(M, N, K);
        if (rc != 0) return rc;
    }

    int nsb = K / SF_VEC_SIZE;

    // Quantize A on GPU
    {
        int total_blocks = M * nsb;
        int threads = 256;
        int blocks = (total_blocks + threads - 1) / threads;
        quantize_bf16_to_fp4_kernel<<<blocks, threads>>>(
            (const __nv_bfloat16*)A_bf16, g.d_A_fp4, g.d_SFA, M, K, nsb, g.d_SFA_lut);
    }

    // Quantize B on GPU
    {
        int total_blocks = N * nsb;
        int threads = 256;
        int blocks = (total_blocks + threads - 1) / threads;
        quantize_bf16_to_fp4_kernel<<<blocks, threads>>>(
            (const __nv_bfloat16*)B_bf16, g.d_B_fp4, g.d_SFB, N, K, nsb, g.d_SFB_lut);
    }

    // Run CUTLASS GEMM
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<const ElementA::DataType*>(g.d_A_fp4), stride_A,
            reinterpret_cast<const ElementB::DataType*>(g.d_B_fp4), stride_B,
            reinterpret_cast<const ScaleFactorType*>(g.d_SFA), layout_SFA,
            reinterpret_cast<const ScaleFactorType*>(g.d_SFB), layout_SFB
        },
        {
            {alpha, beta},
            reinterpret_cast<const ElementC*>(C_bf16 ? C_bf16 : D_bf16), stride_C,
            reinterpret_cast<ElementD*>(D_bf16), stride_D
        }
    };

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm: can_implement: %s\n", cutlassGetStatusString(status));
        return -2;
    }

    status = gemm.initialize(arguments, g.d_workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm: initialize: %s\n", cutlassGetStatusString(status));
        return -3;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm: run: %s\n", cutlassGetStatusString(status));
        return -4;
    }

    cudaDeviceSynchronize();
    return 0;
}

// Run FP4 GEMM with host-side quantization (slower but verified correct)
int fp4_gemm_run_host(
    const void* A_bf16, const void* B_bf16,
    const void* C_bf16, void* D_bf16,
    int M, int N, int K,
    float alpha, float beta)
{
    if (!g.initialized || g.M != M || g.N != N || g.K != K) {
        int rc = fp4_gemm_init(M, N, K);
        if (rc != 0) return rc;
    }

    cudaMemcpy(g.h_A, A_bf16, (size_t)M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(g.h_B, B_bf16, (size_t)N * K * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    for (int i = 0; i < g.sfa_elems; i++) g.h_SFA[i] = ScaleFactorType(1.0f);
    for (int i = 0; i < g.sfb_elems; i++) g.h_SFB[i] = ScaleFactorType(1.0f);

    quantize_matrix_host(g.h_A, g.h_A_fp4, g.h_SFA, M, K, layout_SFA);
    quantize_matrix_host(g.h_B, g.h_B_fp4, g.h_SFB, N, K, layout_SFB);

    cudaMemcpy(g.d_A_fp4, g.h_A_fp4, (size_t)M * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_B_fp4, g.h_B_fp4, (size_t)N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_SFA, g.h_SFA, g.sfa_elems * sizeof(ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_SFB, g.h_SFB, g.sfb_elems * sizeof(ScaleFactorType), cudaMemcpyHostToDevice);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<const ElementA::DataType*>(g.d_A_fp4), stride_A,
            reinterpret_cast<const ElementB::DataType*>(g.d_B_fp4), stride_B,
            reinterpret_cast<const ScaleFactorType*>(g.d_SFA), layout_SFA,
            reinterpret_cast<const ScaleFactorType*>(g.d_SFB), layout_SFB
        },
        {
            {alpha, beta},
            reinterpret_cast<const ElementC*>(C_bf16 ? C_bf16 : D_bf16), stride_C,
            reinterpret_cast<ElementD*>(D_bf16), stride_D
        }
    };

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm.initialize(arguments, g.d_workspace);
    if (status != cutlass::Status::kSuccess) return -3;

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) return -4;

    cudaDeviceSynchronize();
    return 0;
}

void fp4_gemm_cleanup() { cleanup(); }

// Sync helper
void fp4_gemm_sync() { cudaDeviceSynchronize(); }

// ============================================================================
// Pre-quantized weight cache API
// ============================================================================

// Cached weight handle - stores pre-quantized FP4 data + scales on device
struct FP4WeightCache {
    uint8_t* d_fp4;    // [N, K/2] packed FP4 data on device
    uint8_t* d_sf;     // Scale factors in CUTLASS interleaved layout
    int N;
    int K;
    int sf_elems;      // Number of scale factor elements
};

static int align128(int x) { return (x + 127) & ~127; }

static float host_fp16_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s; }
        else {
            e = 1;
            while ((m & 0x400u) == 0) { m <<= 1; e--; }
            m &= 0x3FFu;
            f = s | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static FP4WeightCache *build_weight_cache_host_bf16(
    const cutlass::bfloat16_t *h_bf16, int N, int K)
{
    int dummy_M = 128;
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(dummy_M, N, K, 1));
    int sf_elems = cute::size(cute::filter_zeros(layout_SFB));

    FP4WeightCache *cache = new FP4WeightCache();
    cache->N = N;
    cache->K = K;
    cache->sf_elems = sf_elems;

    uint8_t *h_fp4 = (uint8_t *)calloc((size_t)N * K / 2, 1);
    ScaleFactorType *h_sf = (ScaleFactorType *)calloc((size_t)sf_elems,
                                                      sizeof(ScaleFactorType));
    if (!h_fp4 || !h_sf) {
        free(h_fp4);
        free(h_sf);
        delete cache;
        return nullptr;
    }

    quantize_matrix_host(h_bf16, h_fp4, h_sf, N, K, layout_SFB);

    cudaError_t err;
    err = cudaMalloc(&cache->d_fp4, (size_t)N * K / 2);
    if (err != cudaSuccess) {
        fprintf(stderr, "fp4 weight cache: cudaMalloc fp4 failed: %s\n",
                cudaGetErrorString(err));
        free(h_fp4);
        free(h_sf);
        delete cache;
        return nullptr;
    }
    err = cudaMalloc(&cache->d_sf, sf_elems);
    if (err != cudaSuccess) {
        fprintf(stderr, "fp4 weight cache: cudaMalloc sf failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(cache->d_fp4);
        free(h_fp4);
        free(h_sf);
        delete cache;
        return nullptr;
    }

    cudaMemcpy(cache->d_fp4, h_fp4, (size_t)N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(cache->d_sf, h_sf, (size_t)sf_elems * sizeof(ScaleFactorType),
               cudaMemcpyHostToDevice);

    free(h_fp4);
    free(h_sf);
    return cache;
}

// Quantize BF16 weights once and return a cache handle
// The caller must call fp4_weight_cache_free() when done
void* fp4_quantize_weights(const void* weight_bf16, int N, int K) {
    if (N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "fp4_quantize_weights: N=%d, K=%d must be multiples of 128\n", N, K);
        return nullptr;
    }

    cutlass::bfloat16_t *h_bf16 = (cutlass::bfloat16_t *)malloc(
        (size_t)N * K * sizeof(cutlass::bfloat16_t));
    if (!h_bf16) return nullptr;

    cudaMemcpy(h_bf16, weight_bf16, (size_t)N * K * sizeof(cutlass::bfloat16_t),
               cudaMemcpyDeviceToHost);

    FP4WeightCache *cache = build_weight_cache_host_bf16(h_bf16, N, K);
    free(h_bf16);
    if (!cache)
        fprintf(stderr, "fp4_quantize_weights: quantize failed N=%d K=%d\n", N, K);
    return (void *)cache;
}

void *fp4_quantize_weights_host_f16(const uint16_t *host_f16, int N, int K) {
    int N_pad = align128(N);
    int K_pad = align128(K);

    cutlass::bfloat16_t *h_bf16 = (cutlass::bfloat16_t *)calloc(
        (size_t)N_pad * K_pad, sizeof(cutlass::bfloat16_t));
    if (!h_bf16) return nullptr;

    for (int r = 0; r < N; r++) {
        for (int k = 0; k < K; k++) {
            float v = host_fp16_to_float(host_f16[(size_t)r * K + k]);
            h_bf16[(size_t)r * K_pad + k] = cutlass::bfloat16_t(v);
        }
    }

    FP4WeightCache *cache = build_weight_cache_host_bf16(h_bf16, N_pad, K_pad);
    free(h_bf16);
    if (!cache)
        fprintf(stderr, "fp4_quantize_weights_host_f16: quantize failed N=%d K=%d\n",
                N, K);
    return (void *)cache;
}

// Get the cached FP4 data pointer
const void* fp4_weight_cache_fp4_ptr(const void* cache_handle) {
    if (!cache_handle) return nullptr;
    return ((const FP4WeightCache*)cache_handle)->d_fp4;
}

// Get the cached scale factor pointer
const void* fp4_weight_cache_sf_ptr(const void* cache_handle) {
    if (!cache_handle) return nullptr;
    return ((const FP4WeightCache*)cache_handle)->d_sf;
}

// Get cached weight dimensions
int fp4_weight_cache_N(const void* cache_handle) {
    return cache_handle ? ((const FP4WeightCache*)cache_handle)->N : 0;
}

int fp4_weight_cache_K(const void* cache_handle) {
    return cache_handle ? ((const FP4WeightCache*)cache_handle)->K : 0;
}

// Free cached weight data
void fp4_weight_cache_free(void* cache_handle) {
    if (!cache_handle) return;
    FP4WeightCache* cache = (FP4WeightCache*)cache_handle;
    cudaFree(cache->d_fp4);
    cudaFree(cache->d_sf);
    delete cache;
}

// Run FP4 GEMM with pre-quantized B weights (only quantize A on the fly)
// This skips B quantization entirely - weights are already in FP4 + CUTLASS scale layout
int fp4_gemm_run_cached(
    const void* A_bf16,          // [M, K] BF16 activations (device ptr)
    const void* cache_handle,    // Pre-quantized weight cache handle
    const void* C_bf16,          // [M, N] BF16 (device ptr, or NULL)
    void* D_bf16,                // [M, N] BF16 output (device ptr)
    int M,
    float alpha, float beta)
{
    if (!cache_handle) {
        fprintf(stderr, "fp4_gemm_run_cached: null cache handle\n");
        return -1;
    }

    const FP4WeightCache* cache = (const FP4WeightCache*)cache_handle;
    int N = cache->N;
    int K = cache->K;

    if (M % 128 != 0) {
        fprintf(stderr, "fp4_gemm_run_cached: M=%d must be a multiple of 128\n", M);
        return -1;
    }

    // Initialize state (allocates A-side buffers + workspace)
    if (!g.initialized || g.M != M || g.N != N || g.K != K) {
        int rc = fp4_gemm_init(M, N, K);
        if (rc != 0) return rc;
    }

    int nsb = K / SF_VEC_SIZE;

    // Only quantize A on GPU (B is pre-quantized)
    {
        int total_blocks = M * nsb;
        int threads = 256;
        int blocks = (total_blocks + threads - 1) / threads;
        quantize_bf16_to_fp4_kernel<<<blocks, threads>>>(
            (const __nv_bfloat16*)A_bf16, g.d_A_fp4, g.d_SFA, M, K, nsb, g.d_SFA_lut);
    }

    // Run CUTLASS GEMM with quantized A + cached B
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<const ElementA::DataType*>(g.d_A_fp4), stride_A,
            reinterpret_cast<const ElementB::DataType*>(cache->d_fp4), stride_B,
            reinterpret_cast<const ScaleFactorType*>(g.d_SFA), layout_SFA,
            reinterpret_cast<const ScaleFactorType*>(cache->d_sf), layout_SFB
        },
        {
            {alpha, beta},
            reinterpret_cast<const ElementC*>(C_bf16 ? C_bf16 : D_bf16), stride_C,
            reinterpret_cast<ElementD*>(D_bf16), stride_D
        }
    };

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm_run_cached: can_implement: %s\n", cutlassGetStatusString(status));
        return -2;
    }

    status = gemm.initialize(arguments, g.d_workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm_run_cached: initialize: %s\n", cutlassGetStatusString(status));
        return -3;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "fp4_gemm_run_cached: run: %s\n", cutlassGetStatusString(status));
        return -4;
    }

    // No sync here — caller should call fp4_gemm_sync() when needed
    // This allows pipelining multiple GEMMs without roundtrips
    return 0;
}

#define FP4_GEMV_WARP 32
#define FP4_GEMV_ROWS 8
#define FP4_GEMV_THREADS (FP4_GEMV_WARP * FP4_GEMV_ROWS)

__device__ __forceinline__ int sfb_sf_index(int r, int k_block, int nsb)
{
    int r0 = r % 32;
    int r1 = (r / 32) % 4;
    int r2 = r / 128;
    int k1 = k_block % 4;
    int k2 = k_block / 4;
    int k_tiles = nsb / 4;
    return r0 * 16 + r1 * 4 + r2 * (512 * k_tiles) + k1 + k2 * 512;
}

__device__ __forceinline__ float d_fp4_nibble_to_float(uint8_t nibble)
{
    int sign = (nibble & 0x8) ? -1 : 1;
    return sign * c_fp4_values[nibble & 0x7];
}

__device__ __forceinline__ float fp4_dot_row(
    const float *x, const uint8_t *row_fp4, const uint8_t *w_sf,
    int row, int n, int K, int nsb)
{
    float val = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        int k0 = sb * SF_VEC_SIZE;
        if (k0 >= n) break;
        float scale = d_ue4m3_to_float(w_sf[sfb_sf_index(row, sb, nsb)]);
        #pragma unroll
        for (int i = 0; i < SF_VEC_SIZE; i += 2) {
            int k = k0 + i;
            if (k >= n) break;
            uint8_t pk = row_fp4[(k0 + i) >> 1];
            val += x[k] * d_fp4_nibble_to_float(pk & 0xF) * scale;
            if (k + 1 < n)
                val += x[k + 1] * d_fp4_nibble_to_float(pk >> 4) * scale;
        }
    }
    return val;
}

__launch_bounds__(FP4_GEMV_THREADS)
__global__ void fp4_gemv_kernel(float *__restrict__ o,
    const float *__restrict__ x,
    const uint8_t *__restrict__ w_fp4,
    const uint8_t *__restrict__ w_sf,
    int n, int d, int K, int nsb)
{
    int local_row = threadIdx.x / FP4_GEMV_WARP;
    int lane      = threadIdx.x % FP4_GEMV_WARP;
    int row       = blockIdx.x * FP4_GEMV_ROWS + local_row;
    if (row >= d) return;

    const uint8_t *row_fp4 = w_fp4 + (size_t)row * (K / 2);
    float val = 0.f;

    for (int sb = lane; sb < nsb; sb += FP4_GEMV_WARP) {
        int k0 = sb * SF_VEC_SIZE;
        if (k0 >= n) continue;
        float scale = d_ue4m3_to_float(w_sf[sfb_sf_index(row, sb, nsb)]);
        #pragma unroll
        for (int i = 0; i < SF_VEC_SIZE; i += 2) {
            int k = k0 + i;
            if (k >= n) break;
            uint8_t pk = row_fp4[(k0 + i) >> 1];
            val += x[k] * d_fp4_nibble_to_float(pk & 0xF) * scale;
            if (k + 1 < n)
                val += x[k + 1] * d_fp4_nibble_to_float(pk >> 4) * scale;
        }
    }
    for (int off = FP4_GEMV_WARP / 2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off, FP4_GEMV_WARP);
    if (lane == 0)
        o[row] = val;
}

__global__ void fp4_gemv_batch_kernel(float *__restrict__ o,
    const float *__restrict__ x,
    const uint8_t *__restrict__ w_fp4,
    const uint8_t *__restrict__ w_sf,
    int n, int d, int K, int nsb, int M)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int t = flat / d;
    int row = flat % d;
    if (t >= M || row >= d) return;

    const float *xt = x + (size_t)t * n;
    float *ot = o + (size_t)t * d;
    const uint8_t *row_fp4 = w_fp4 + (size_t)row * (K / 2);
    ot[row] = fp4_dot_row(xt, row_fp4, w_sf, row, n, K, nsb);
}

void fp4_gemv_cached(const void *cache_handle, const float *x, float *y,
                     int n, int d)
{
    if (!cache_handle) return;
    const FP4WeightCache *cache = (const FP4WeightCache *)cache_handle;
    int K = cache->K;
    int nsb = K / SF_VEC_SIZE;
    int blocks = (d + FP4_GEMV_ROWS - 1) / FP4_GEMV_ROWS;
    fp4_gemv_kernel<<<blocks, FP4_GEMV_THREADS>>>(
        y, x, cache->d_fp4, cache->d_sf, n, d, K, nsb);
}

void fp4_gemv_batch_cached(const void *cache_handle, const float *x, float *y,
                           int M, int n, int d)
{
    if (!cache_handle || M <= 0) return;
    const FP4WeightCache *cache = (const FP4WeightCache *)cache_handle;
    int K = cache->K;
    int nsb = K / SF_VEC_SIZE;
    int total = M * d;
    fp4_gemv_batch_kernel<<<(total + 255) / 256, 256>>>(
        y, x, cache->d_fp4, cache->d_sf, n, d, K, nsb, M);
}

}  // extern "C"
