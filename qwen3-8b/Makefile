# CPU / OpenMP / XDNA2: CC・CFLAGS は make CC=clang CFLAGS=-O2 などで上書き可
CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

# 実行例で使う既定（runtime）
MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# ROCM（HIP）
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201 # gfx1201 = AMD Radeon AI PRO R9700（llama3-8b 兄弟ターゲットと同じ）

# XDNA2: Linux UAPI（<drm/drm.h>）。通常は追加不要。
# カスタムカーネルヘッダに向けるときは例: XDNA_INCS=-I/path/to/linux-headers/usr/include
XDNA_INCS ?=

# -----------------------------------------------------------------------------
# CPU 参照実装（llama3-8b/main.c 相当）

build:
	$(CC) $(CFLAGS) -o qwen3-cpu main.c $(LDFLAGS)

run: build
	./qwen3-cpu $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# OpenMP 並列（main-omp.c）

build.omp:
	$(CC) $(CFLAGS) -fopenmp -o qwen3-cpu-omp main-omp.c $(LDFLAGS)

run.omp: build.omp
	./qwen3-cpu-omp $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Radeon GPU（ROCm / HIP）。hipcc が GPU にオフロードするバックエンド。
# ソースは main-rocm.c。Qwen3-VL-8B（IQ2_M）向けの text-only / fused layer / FA パイプライン。

build.rocm:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o qwen3-rocm -x hip main-rocm.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run.rocm: build.rocm
	./qwen3-rocm $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU（amdxdna DRM ioctl）。
# Kernel >= 6.10、/dev/accel/accel0。NPU なしでも OpenMP CPU GEMV にフォールバック。

build.xdna2:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) -o qwen3-xdna2 main-xdna2.c $(LDFLAGS)

run.xdna2: build.xdna2
	./qwen3-xdna2 $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# 省ホスト RAM の XDNA2: 線形重み→ブロック FP16（BF16 スケール+int8）変換読み込み

build.xdna2.bfpx:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) -o qwen3-xdna2-bfpx main-xdna2-bfpx.c $(LDFLAGS)

run.xdna2.bfpx: build.xdna2.bfpx
	./qwen3-xdna2-bfpx $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------

clean:
	rm -f qwen3-cpu qwen3-cpu-omp qwen3-rocm qwen3-xdna2 qwen3-xdna2-bfpx

.PHONY: build run build.omp run.omp build.rocm run.rocm build.xdna2 run.xdna2 build.xdna2.bfpx run.xdna2.bfpx clean
