MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS / ROCM 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang（コマンドラインは子 make に伝播）

.PHONY: \
	all \
	build.cpu run.cpu \
	build.cpu-multicore run.cpu-multicore \
	build.gpu-rocm run.gpu-rocm \
	build.xdna2 run.xdna2 \
	build.xdna2-bfp16 run.xdna2-bfp16 \
	gen-xdna-kernels \
	clean

.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# GGUF モデル（gguf.txt の URL からダウンロード）
model:
	@if [ ! -f "$(MODEL).sha256sum" ]; then \
		echo "Error: checksum file '$(MODEL).sha256sum' not found."; \
		echo "Update the repository to the latest version, or verify the file exists."; \
		exit 1; \
	fi; \
	url=$$(sed 's|/blob/main/|/resolve/main/|' gguf.txt); \
	wget -O "$(MODEL)" "$$url" && \
	if sha256sum --check "$(MODEL).sha256sum"; then \
		echo ""; \
		echo "Checksum verification succeeded."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		echo "Model download and verification complete."; \
		echo "  make run.cpu"; \
		echo "or similar to run inference."; \
	else \
		echo ""; \
		echo "Error: checksum verification failed for the downloaded model file."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		echo "The file may have been corrupted during download."; \
		echo "Removing the corrupted file. Check your network connection, then run"; \
		echo "  make model"; \
		echo "again."; \
		rm -f "$(MODEL)"; \
		exit 1; \
	fi

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
# AMD Radeon GPU（gpu-rocm/main.c）
# HIPCC が無い環境ではビルドを省略（gpu-rocm/Makefile の既定パスと一致させること）
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201

build.gpu-rocm:
	@if [ -x "$(HIPCC)" ]; then \
		cd gpu-rocm && make build ROCM="$(ROCM)" HIPCC="$(HIPCC)" GPU_ARCH="$(GPU_ARCH)"; \
	else \
		echo "Skipping build.gpu-rocm: HIPCC ($(HIPCC)) not found"; \
	fi

run.gpu-rocm: build.gpu-rocm
	./gpu-rocm/qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU（xdna2/main.c）
build.xdna2:
	cd xdna2 && make build

run.xdna2: build.xdna2
	./xdna2/qwen3-xdna2 "$(MODEL)" -p "$(PROMPT)"

# XDNA: スタブ再生成（リポジトリルート基準）。
gen-xdna-kernels:
	cd .. && python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 + BFPX（xdna2-bfp16/main.c）
build.xdna2-bfp16:
	cd xdna2-bfp16 && make build

run.xdna2-bfp16: build.xdna2-bfp16
	./xdna2-bfp16/qwen3-xdna2-bfpx "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# 全バリアントをビルド
all: build.cpu build.cpu-multicore build.gpu-rocm build.xdna2 build.xdna2-bfp16

# -----------------------------------------------------------------------------
clean:
	for d in cpu cpu-multicore gpu-rocm xdna2 xdna2-bfp16; do \
		(cd $$d && make clean); \
	done
