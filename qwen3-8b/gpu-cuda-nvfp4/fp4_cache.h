#ifndef FP4_CACHE_H
#define FP4_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP4_CACHE_MAGIC   0x5141464eu  /* "NFAQ" little-endian */
#define FP4_CACHE_VERSION 1

typedef struct FP4HostWeight {
    int N, K;
    int N_act, K_act;
    int sf_elems;
    uint8_t *h_fp4;
    uint8_t *h_sf;
} FP4HostWeight;

typedef void (*fp4_dequant_row_fn)(void *ctx, int row, float *row_out, int K_act);

void fp4_host_weight_free(FP4HostWeight *w);

int  fp4_host_weight_save(const FP4HostWeight *w, const char *path);
FP4HostWeight *fp4_host_weight_load(const char *path);

void fp4_cache_dir_path(const char *gguf_path, char *out, size_t out_sz);
void fp4_cache_tensor_path(const char *cache_dir, const char *tensor_name,
                           char *out, size_t out_sz);

int fp4_cache_manifest_valid(const char *cache_dir, const char *gguf_path);
int fp4_cache_write_manifest(const char *cache_dir, const char *gguf_path);

#ifdef __cplusplus
}
#endif

#endif
