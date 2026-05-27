#ifndef FP16_CACHE_H
#define FP16_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP16_CACHE_MAGIC   0x31485046u  /* "FPH1" little-endian */
#define FP16_CACHE_VERSION 1

typedef struct FP16HostWeight {
    int n_rows;
    int n_cols;
    uint16_t *h_f16;
} FP16HostWeight;

void fp16_host_weight_free(FP16HostWeight *w);

int  fp16_host_weight_save(const FP16HostWeight *w, const char *path);
FP16HostWeight *fp16_host_weight_load(const char *path);

void fp16_cache_dir_path(const char *gguf_path, char *out, size_t out_sz);
void fp16_cache_tensor_path(const char *cache_dir, const char *tensor_name,
                            char *out, size_t out_sz);

int fp16_cache_manifest_valid(const char *cache_dir, const char *gguf_path);
int fp16_cache_write_manifest(const char *cache_dir, const char *gguf_path);

#ifdef __cplusplus
}
#endif

#endif
