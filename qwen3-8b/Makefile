MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS / ROCM 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang（コマンドラインは子 make に伝播）

.PHONY: \
	all \
	build.cpu run.cpu \
	build.cpu-multicore run.cpu-multicore \
	build.gpu run.gpu \
	build.xdna2 run.xdna2 \
	build.xdna2-bfp16 run.xdna2-bfp16 \
	gen-xdna-kernels \
	clean

.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# CPU reference（ソースは cpu/main.c）
build.cpu:
	cd cpu && make build

run.cpu: build.cpu
	./cpu/qwen3-cpu "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# CPU + OpenMP（cpu-multicore/main.c）
build.cpu-multicore:
	cd cpu-multicore && make build

run.cpu-multicore: build.cpu-multicore
	./cpu-multicore/qwen3-cpu-omp "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Radeon GPU（gpu/main.c）
# HIPCC が無い環境ではビルドを省略（gpu/Makefile の既定パスと一致させること）
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201

build.gpu:
	@if [ -x "$(HIPCC)" ]; then \
		cd gpu && make build ROCM="$(ROCM)" HIPCC="$(HIPCC)" GPU_ARCH="$(GPU_ARCH)"; \
	else \
		echo "Skipping build.gpu: HIPCC ($(HIPCC)) not found"; \
	fi

run.gpu: build.gpu
	./gpu/qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU（xdna2/main.c）
build.xdna2:
	cd xdna2 && make build

run.xdna2: build.xdna2
	./xdna2/qwen3-xdna2 "$(MODEL)" -p "$(PROMPT)"

# XDNA: スタブ再生成（リポジトリルート基準）。
gen-xdna-kernels:
	cd .. && python3 xdna-gemv/gen-xdna-gemv-stubs.py xdna-gemv/kernels

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 + BFPX（xdna2-bfp16/main.c）
build.xdna2-bfp16:
	cd xdna2-bfp16 && make build

run.xdna2-bfp16: build.xdna2-bfp16
	./xdna2-bfp16/qwen3-xdna2-bfpx "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# 全バリアントをビルド
all: build.cpu build.cpu-multicore build.gpu build.xdna2 build.xdna2-bfp16

# -----------------------------------------------------------------------------
clean:
	for d in cpu cpu-multicore gpu xdna2 xdna2-bfp16; do \
		(cd $$d && make clean); \
	done
