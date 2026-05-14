CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# AMD Radeon GPU (ROCm / HIP)
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201

# XDNA2: <drm/drm.h> が標準外にあるときだけ `-I…` を渡す。
XDNA_INCS ?=

.PHONY: \
	build run \
	build.omp run.omp \
	build.rocm run.rocm \
	build.xdna2 run.xdna2 \
	build.xdna2.bfpx run.xdna2.bfpx \
	gen-xdna-kernels \
	clean

# -----------------------------------------------------------------------------
# CPU reference（ソースは cpu/main.c）
build:
	$(MAKE) -C cpu build CC="$(CC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

run: build
	./cpu/qwen3-cpu "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# CPU + OpenMP（cpu-multicore/main.c）
build.omp:
	$(MAKE) -C cpu-multicore build CC="$(CC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

run.omp: build.omp
	./cpu-multicore/qwen3-cpu-omp "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Radeon GPU（gpu/main.c）
build.rocm:
	$(MAKE) -C gpu build ROCM="$(ROCM)" HIPCC="$(HIPCC)" GPU_ARCH="$(GPU_ARCH)"

run.rocm: build.rocm
	./gpu/qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU（xdna2/main.c）
build.xdna2:
	$(MAKE) -C xdna2 build CC="$(CC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" \
		XDNA_INCS="$(XDNA_INCS)"

run.xdna2: build.xdna2
	./xdna2/qwen3-xdna2 "$(MODEL)" -p "$(PROMPT)"

# XDNA: スタブ再生成（リポジトリルート基準）。
gen-xdna-kernels:
	cd .. && python3 xdna-gemv/gen-xdna-gemv-stubs.py xdna-gemv/kernels

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 + BFPX（xdna2-bfp16/main.c）
build.xdna2.bfpx:
	$(MAKE) -C xdna2-bfp16 build CC="$(CC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" \
		XDNA_INCS="$(XDNA_INCS)"

run.xdna2.bfpx: build.xdna2.bfpx
	./xdna2-bfp16/qwen3-xdna2-bfpx "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
clean:
	for d in cpu cpu-multicore gpu xdna2 xdna2-bfp16; do $(MAKE) -C $$d clean; done
