#include "fp16_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  n_rows;
    int32_t  n_cols;
    uint32_t reserved;
} Fp16CacheFileHeader;

void fp16_cache_dir_path(const char *gguf_path, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s.fp16", gguf_path);
}

void fp16_cache_tensor_path(const char *cache_dir, const char *tensor_name,
                            char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/%s.fp16bin", cache_dir, tensor_name);
}

static int mkdir_p(const char *path) {
    char tmp[1024];
    size_t len = strlen(path);
    if (len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) != 0) {
            struct stat st;
            if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode))
                return -1;
        }
        *p = '/';
    }
    if (mkdir(tmp, 0755) != 0) {
        struct stat st;
        if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode))
            return -1;
    }
    return 0;
}

void fp16_host_weight_free(FP16HostWeight *w) {
    if (!w) return;
    free(w->h_f16);
    free(w);
}

int fp16_host_weight_save(const FP16HostWeight *w, const char *path) {
    if (!w || !path || !w->h_f16 || w->n_rows <= 0 || w->n_cols <= 0)
        return -1;

    const char *slash = strrchr(path, '/');
    if (slash) {
        char dir[1024];
        size_t dlen = (size_t)(slash - path);
        if (dlen >= sizeof(dir)) return -1;
        memcpy(dir, path, dlen);
        dir[dlen] = '\0';
        if (mkdir_p(dir) != 0) return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    Fp16CacheFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = FP16_CACHE_MAGIC;
    hdr.version = FP16_CACHE_VERSION;
    hdr.n_rows = w->n_rows;
    hdr.n_cols = w->n_cols;

    size_t payload = (size_t)w->n_rows * (size_t)w->n_cols * sizeof(uint16_t);
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
        fwrite(w->h_f16, 1, payload, f) != payload) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

FP16HostWeight *fp16_host_weight_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    Fp16CacheFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic != FP16_CACHE_MAGIC ||
        hdr.version != FP16_CACHE_VERSION ||
        hdr.n_rows <= 0 || hdr.n_cols <= 0) {
        fclose(f);
        return NULL;
    }

    FP16HostWeight *w = (FP16HostWeight *)calloc(1, sizeof(*w));
    if (!w) { fclose(f); return NULL; }

    w->n_rows = hdr.n_rows;
    w->n_cols = hdr.n_cols;
    size_t payload = (size_t)w->n_rows * (size_t)w->n_cols * sizeof(uint16_t);
    w->h_f16 = (uint16_t *)malloc(payload);
    if (!w->h_f16) {
        fp16_host_weight_free(w);
        fclose(f);
        return NULL;
    }
    if (fread(w->h_f16, 1, payload, f) != payload) {
        fp16_host_weight_free(w);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return w;
}

static void manifest_path(const char *cache_dir, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/manifest", cache_dir);
}

int fp16_cache_manifest_valid(const char *cache_dir, const char *gguf_path) {
    char mpath[1024];
    manifest_path(cache_dir, mpath, sizeof(mpath));

    struct stat gst, mst;
    if (stat(gguf_path, &gst) != 0 || stat(mpath, &mst) != 0)
        return 0;

    FILE *f = fopen(mpath, "r");
    if (!f) return 0;

    int version = 0;
    unsigned long long gguf_size = 0;
    long long gguf_mtime = 0;
    if (fscanf(f, "%d\n", &version) != 1 || version != FP16_CACHE_VERSION) {
        fclose(f);
        return 0;
    }
    if (fscanf(f, "gguf_size=%llu\n", &gguf_size) != 1) {
        fclose(f);
        return 0;
    }
    if (fscanf(f, "gguf_mtime=%lld\n", &gguf_mtime) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);

    return (unsigned long long)gst.st_size == gguf_size &&
           (long long)gst.st_mtime == gguf_mtime;
}

int fp16_cache_write_manifest(const char *cache_dir, const char *gguf_path) {
    struct stat gst;
    if (stat(gguf_path, &gst) != 0) return -1;
    if (mkdir_p(cache_dir) != 0) return -1;

    char mpath[1024];
    manifest_path(cache_dir, mpath, sizeof(mpath));
    FILE *f = fopen(mpath, "w");
    if (!f) return -1;

    fprintf(f, "%d\n", FP16_CACHE_VERSION);
    fprintf(f, "gguf_size=%llu\n", (unsigned long long)gst.st_size);
    fprintf(f, "gguf_mtime=%lld\n", (long long)gst.st_mtime);
    fclose(f);
    return 0;
}
