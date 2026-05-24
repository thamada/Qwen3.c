#include "fp4_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  N, K;
    int32_t  N_act, K_act;
    int32_t  sf_elems;
    uint32_t reserved;
} Fp4CacheFileHeader;

void fp4_cache_dir_path(const char *gguf_path, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s.nvfp4", gguf_path);
}

void fp4_cache_tensor_path(const char *cache_dir, const char *tensor_name,
                           char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/%s.fp4bin", cache_dir, tensor_name);
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

void fp4_host_weight_free(FP4HostWeight *w) {
    if (!w) return;
    free(w->h_fp4);
    free(w->h_sf);
    free(w);
}

int fp4_host_weight_save(const FP4HostWeight *w, const char *path) {
    if (!w || !path) return -1;
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

    Fp4CacheFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = FP4_CACHE_MAGIC;
    hdr.version = FP4_CACHE_VERSION;
    hdr.N = w->N;
    hdr.K = w->K;
    hdr.N_act = w->N_act;
    hdr.K_act = w->K_act;
    hdr.sf_elems = w->sf_elems;

    size_t fp4_bytes = (size_t)w->N * (size_t)w->K / 2;
    size_t sf_bytes = (size_t)w->sf_elems;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
        fwrite(w->h_fp4, 1, fp4_bytes, f) != fp4_bytes ||
        fwrite(w->h_sf, 1, sf_bytes, f) != sf_bytes) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

FP4HostWeight *fp4_host_weight_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    Fp4CacheFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic != FP4_CACHE_MAGIC ||
        hdr.version != FP4_CACHE_VERSION ||
        hdr.N <= 0 || hdr.K <= 0 || hdr.sf_elems <= 0) {
        fclose(f);
        return NULL;
    }

    FP4HostWeight *w = (FP4HostWeight *)calloc(1, sizeof(*w));
    if (!w) { fclose(f); return NULL; }

    w->N = hdr.N;
    w->K = hdr.K;
    w->N_act = hdr.N_act;
    w->K_act = hdr.K_act;
    w->sf_elems = hdr.sf_elems;

    size_t fp4_bytes = (size_t)w->N * (size_t)w->K / 2;
    size_t sf_bytes = (size_t)w->sf_elems;

    w->h_fp4 = (uint8_t *)malloc(fp4_bytes);
    w->h_sf = (uint8_t *)malloc(sf_bytes);
    if (!w->h_fp4 || !w->h_sf) {
        fp4_host_weight_free(w);
        fclose(f);
        return NULL;
    }

    if (fread(w->h_fp4, 1, fp4_bytes, f) != fp4_bytes ||
        fread(w->h_sf, 1, sf_bytes, f) != sf_bytes) {
        fp4_host_weight_free(w);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return w;
}

static void manifest_path(const char *cache_dir, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/manifest", cache_dir);
}

int fp4_cache_manifest_valid(const char *cache_dir, const char *gguf_path) {
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
    if (fscanf(f, "%d\n", &version) != 1 || version != FP4_CACHE_VERSION) {
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

int fp4_cache_write_manifest(const char *cache_dir, const char *gguf_path) {
    struct stat gst;
    if (stat(gguf_path, &gst) != 0) return -1;
    if (mkdir_p(cache_dir) != 0) return -1;

    char mpath[1024];
    manifest_path(cache_dir, mpath, sizeof(mpath));
    FILE *f = fopen(mpath, "w");
    if (!f) return -1;

    fprintf(f, "%d\n", FP4_CACHE_VERSION);
    fprintf(f, "gguf_size=%llu\n", (unsigned long long)gst.st_size);
    fprintf(f, "gguf_mtime=%lld\n", (long long)gst.st_mtime);
    fclose(f);
    return 0;
}
