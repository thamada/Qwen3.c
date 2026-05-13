CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm
TARGET_CPU                     = qwen3-cpu
TARGET_CPU_OMP                 = qwen3-cpu-omp
TARGET_ROCM                    = qwen3-rocm
TARGET_XDNA2                   = qwen3-xdna2
TARGET_XDNA2_BFPX              = qwen3-xdna2-bfpx

MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

ROCM    ?= /opt/rocm
HIPCC   ?= $(ROCM)/bin/hipcc
# gfx1201 = AMD Radeon AI PRO R9700 (matches the llama3-8b sibling target).
GPU_ARCH ?= gfx1201

# Linux UAPI headers shipping <drm/drm.h>; on most distros installed under
# /usr/include. Override e.g. XDNA_INCS=-I/path/to/linux-headers if you build
# against an out-of-tree kernel.
XDNA_INCS ?=

# CPU reference inference (llama3-8b/main.c 相当)
build:
	$(CC) $(CFLAGS) -o $(TARGET_CPU) main.c $(LDFLAGS)

run: build
	./$(TARGET_CPU) $(MODEL) -p "$(PROMPT)"

# OpenMP 並列（main-omp.c）
build.omp:
	$(CC) $(CFLAGS) -fopenmp -o $(TARGET_CPU_OMP) main-omp.c $(LDFLAGS)

run.omp: build.omp
	./$(TARGET_CPU_OMP) $(MODEL) -p "$(PROMPT)"

# Qwen3-VL-8B (IQ2_M) — text-only decoder, layer-fused, FlashAttention pipeline.
# 既定エントリ: main-rocm.c（llama3-8b の main-rocm.c に相当する Make ターゲット名）
build.rocm:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o $(TARGET_ROCM) -x hip main-rocm.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run.rocm: build.rocm
	./$(TARGET_ROCM) $(MODEL) -p "$(PROMPT)"

# AMD Ryzen AI XDNA2 NPU backend — direct amdxdna DRM ioctl interface.
# Requires Linux kernel >= 6.10 with the in-tree amdxdna driver and a
# /dev/accel/accel0 node accessible to the running user. Without an NPU the
# binary still runs, transparently falling back to an OpenMP CPU GEMV path.
build.xdna2:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o $(TARGET_XDNA2) main-xdna2.c $(LDFLAGS)

run.xdna2: build.xdna2
	./$(TARGET_XDNA2) $(MODEL) -p "$(PROMPT)"

# Low-host-RAM XDNA2 path: convert linear weights to block FP16 (BF16 scale + int8)
# on load; mmap GGUF only during conversion, then munmap. GEMV uses chunked BF16 staging.
build.xdna2.bfpx:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o $(TARGET_XDNA2_BFPX) main-xdna2-bfpx.c $(LDFLAGS)

run.xdna2.bfpx: build.xdna2.bfpx
	./$(TARGET_XDNA2_BFPX) $(MODEL) -p "$(PROMPT)"

clean:
	rm -f $(TARGET_CPU) $(TARGET_CPU_OMP) $(TARGET_ROCM) $(TARGET_XDNA2) $(TARGET_XDNA2_BFPX)

.PHONY: build run build.omp run.omp build.rocm run.rocm build.xdna2 run.xdna2 build.xdna2.bfpx run.xdna2.bfpx clean
