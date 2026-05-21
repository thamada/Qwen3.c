#ifndef POLARQUANT_H
#define POLARQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PolarQuant (arxiv:2502.02617) for KV cache compression.
 * L=4 recursive polar levels on 16-coord blocks, head_dim=128.
 * PolarQuant-R: random Hadamard preconditioning (shared across layers/heads). */

#define PQ_HEAD_DIM   128
#define PQ_BLOCK      16
#define PQ_LEVELS     4
#define PQ_NBLOCKS    (PQ_HEAD_DIM / PQ_BLOCK)   /* 8 */
#define PQ_BYTES_HEAD (PQ_NBLOCKS * 8)           /* 64 bytes / head vector */

int  polarquant_init(int head_dim);
void polarquant_shutdown(void);

int  polarquant_bytes_per_token(int n_kv_heads);
int  polarquant_head_dim(void);

/* Device-side KV cache allocation: [n_layers * max_seq * n_kv_heads * PQ_BYTES_HEAD] */
void *polarquant_kv_cache_alloc(int n_layers, int max_seq, int n_kv_heads);
void  polarquant_kv_cache_free(void *cache);

const void *polarquant_device_state(void);

void polarquant_kv_write_one(void *dst_blocks, const float *src, int kv_dim);
void polarquant_kv_write_batch(void *dst, const float *src, int kv_dim, int n_tokens);

#ifdef __cplusplus
}
#endif

#endif
