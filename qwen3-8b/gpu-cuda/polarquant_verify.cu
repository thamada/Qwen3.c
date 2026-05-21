/*
 * PolarQuant round-trip verify: encode + decode vs original vector.
 */

#include "polarquant.h"
#include "polarquant_kernels.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void pq_roundtrip_kernel(const float *src, float *dst, const PQState *st)
{
    PQBlock blocks[PQ_NBLK];
    pq_encode_head(st, src, blocks);
    pq_decode_head(st, blocks, dst);
}

static int run_roundtrip(void)
{
    if (polarquant_init(PQ_HD) != 0) return 1;

    float h_in[PQ_HD], h_out[PQ_HD];
    for (int i = 0; i < PQ_HD; i++)
        h_in[i] = sinf(0.17f * (float)i) * 0.3f + cosf(0.09f * (float)i) * 0.1f;

    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in, PQ_HD * sizeof(float));
    cudaMalloc(&d_out, PQ_HD * sizeof(float));
    cudaMemcpy(d_in, h_in, PQ_HD * sizeof(float), cudaMemcpyHostToDevice);

    pq_roundtrip_kernel<<<1, 1>>>(d_in, d_out, (const PQState *)polarquant_device_state());
    cudaMemcpy(h_out, d_out, PQ_HD * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float max_err = 0.0f, rel = 0.0f, nrm = 0.0f;
    for (int i = 0; i < PQ_HD; i++) {
        float e = fabsf(h_in[i] - h_out[i]);
        if (e > max_err) max_err = e;
        nrm += h_in[i] * h_in[i];
    }
    nrm = sqrtf(nrm);
    rel = nrm > 0.0f ? max_err / nrm : max_err;

    cudaFree(d_in);
    cudaFree(d_out);
    polarquant_shutdown();

    printf("polarquant roundtrip: max_abs_err=%.6f rel=%.4f\n", max_err, rel);
    return (rel > 0.35f) ? 1 : 0;
}

int main(void)
{
    int rc = run_roundtrip();
    if (rc != 0)
        fprintf(stderr, "polarquant_verify FAILED\n");
    else
        printf("polarquant_verify OK\n");
    return rc;
}
