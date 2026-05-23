/* Minimal WMMA probe for gfx11 — used by `make wmma` to calibrate ISA detection. */
#include <hip/hip_runtime.h>
#include <stdio.h>

typedef float f32x8 __attribute__((ext_vector_type(8)));

#if defined(WMMA_PROBE_HAS_GFX11) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx11__)
#undef WMMA_PROBE_HAS_GFX11
#define WMMA_PROBE_HAS_GFX11 1
#endif

#if WMMA_PROBE_HAS_GFX11
__global__ void wmma_probe_kernel(float *out) {
    f32x8 zero = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    f32x8 a = (f32x8){1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f};
    f32x8 b = (f32x8){1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f};
    f32x8 r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, zero);
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *out = r[0];
}
#endif

int main(void) {
#if !WMMA_PROBE_HAS_GFX11
    fprintf(stderr, "wmma_probe: WMMA builtins require gfx11 (compile with --offload-arch=gfx11xx)\n");
    return 2;
#else
    float *d_out = NULL;
    float h_out = 0.f;
    if (hipMalloc(&d_out, sizeof(float)) != hipSuccess)
        return 1;
    hipLaunchKernelGGL(wmma_probe_kernel, dim3(1), dim3(32), 0, 0, d_out);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(d_out);
    printf("wmma_probe: ok (result=%g)\n", (double)h_out);
    return 0;
#endif
}
