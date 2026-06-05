# NVIDIA CUDA + NVFP4 (Blackwell Tensor Core / CUTLASS)。
# 共有ソースは ../gpu-cuda/ を参照。
# FP16 のみ版は ../gpu-cuda/ を参照。
#
# 要 CUDA 13 + sm_120a 対応 GPU（RTX 50 系等）。
# 初回: make cutlass または make blackwell（CUDA 13 導入込み）

SRC_DIR := ../gpu-cuda

CUDA_GENCODE ?= arch=compute_120a,code=sm_120a

CUDA_HOME ?= /usr/local/cuda
CUDA13_NVCC ?= /usr/local/cuda/bin/nvcc

NVCC    ?= $(CUDA_HOME)/bin/nvcc
CC      ?= cc

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter \
           -I. -I$(CUDA_HOME)/include
LDFLAGS ?= -L$(CUDA_HOME)/lib64 -lcudart -lm

ifeq ($(findstring /,$(NVCC)),/)
CUDA_BIN := $(dir $(shell readlink -f $(NVCC)))
else
CUDA_BIN := $(dir $(shell readlink -f $$(command -v $(NVCC) 2>/dev/null || echo $(CUDA13_NVCC))))
endif
export PATH := $(CUDA_BIN):$(PATH)

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
             -use_fast_math --expt-relaxed-constexpr \
             -I. \
             -gencode $(CUDA_GENCODE)

BLACKWELL_GENCODE ?= arch=compute_120a,code=sm_120a
BLACKWELL_FA_BR     ?= 32

CUTLASS_DIR ?= third_party/cutlass
CUTLASS_TAG ?= v4.5.0

FP4_CXXFLAGS = -I$(CUTLASS_DIR)/include \
               -I$(CUTLASS_DIR)/tools/util/include \
               -DCUTLASS_ARCH_MMA_SM120_SUPPORTED

# NVFP4 GEMM オブジェクトは sm_120a 固定 gencode（CUTLASS v4.5.0 + C++17 必須）
#
# CUTLASS v4.5.0 は CUDA 13 で非推奨になった long4 等のベクトル型を使わないため、
# 旧版 (v3.9.0) で必要だった -Wno-deprecated-declarations は不要。
#
# ただし v4.5.0 には別の nvcc 警告 (#20012) がある:
#   third_party/cutlass/.../sm100_static_tile_scheduler.hpp が
#     CUTLASS_HOST_DEVICE StaticPersistentTileScheduler100() = default;
#   のように「= default コンストラクタに __device__/__host__ を付ける」と
#   nvcc が「その属性は無視される」と警告する（当リポジトリのコードではなく CUTLASS 側）。
# 当リポジトリでは third_party/cutlass を改変しない。
# CUTLASS 公式ビルド (python/cutlass_cppgen/backend/compiler.py) も同じ nvcc オプションで
# この1件だけ警告を抑止しているため、それに合わせる:
#   -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
#
# 参考:
#   CUTLASS v4.5.0 リリース
#     https://github.com/NVIDIA/cutlass/releases/tag/v4.5.0
#   CUDA 13 非推奨ベクトル型への CUTLASS 側対応 (platform.h)
#     https://github.com/NVIDIA/cutlass/blob/v4.5.0/include/cutlass/platform/platform.h
#   警告 #20012 の発生箇所 (sm100_static_tile_scheduler.hpp)
#     https://github.com/NVIDIA/cutlass/blob/v4.5.0/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp#L53-L54
#   CUTLASS 公式 nvcc ビルドでの同オプション使用 (compiler.py L159)
#     https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/cutlass_cppgen/backend/compiler.py#L156-L160
#   nvcc --diag-suppress / -Xcudafe (NVIDIA NVCC マニュアル)
#     https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#diag-suppress-errnum-diag-suppress
#   = default コンストラクタへの __device__/__host__ 警告の背景 (Kokkos #1470)
#     https://github.com/kokkos/kokkos/issues/1470#issuecomment-378918573
BLACKWELL_NVCCFLAGS = -O3 -std=c++17 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
                      -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored \
                      -use_fast_math --expt-relaxed-constexpr \
                      -gencode $(BLACKWELL_GENCODE)

FP4_FLAG  := -DBONSAI_FP4=1
FP4_GEMM_OBJ  := fp4_gemm.sm120a.o
FP4_QWEN3_OBJ := fp4_qwen3.sm120a.o
FP4_OBJS      := $(FP4_GEMM_OBJ) $(FP4_QWEN3_OBJ) fp4_cache_io.o
FA_DEBUG_OBJ  := ../gpu-cuda/fa_debug.o

# PolarQuant KV cache: BONSAI_POLARQUANT=1（make build.polarquant）
BONSAI_POLARQUANT ?= 0

PQ_FLAG :=
PQ_OBJS :=
ifneq ($(BONSAI_POLARQUANT),0)
PQ_FLAG   := -DBONSAI_POLARQUANT=1
PQ_OBJS   := polarquant.o
endif

FA_BR_FLAG :=
FA_BR_VAL  := $(if $(FA_BR),$(FA_BR),$(BLACKWELL_FA_BR))
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

KERNELS_OBJ := kernels.fabr$(FA_BR_VAL).pq$(BONSAI_POLARQUANT).o
MAIN_OBJ    := main.pq$(BONSAI_POLARQUANT).o
BIN         := qwen3-gpu-cuda-nvfp4

FP4_CACHE_MANIFEST := $(MODEL).nvfp4/manifest

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# Benchmark defaults for log.push (override: make log.push BENCH_N=64)
# BENCH_PROMPT is ~128 tokens (ChatML-encoded prompt length).
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42
BENCH_TEMP   ?= 0 
BENCH_LOG_FILE ?= /tmp/benchmark.log

# GPU identifier for BENCH_LOG (Blackwell sm_120a target)
GPU_SM ?= sm_120a

.PHONY: build build.polarquant run run.polarquant clean blackwell cutlass fp4-test pq-test log log.push pack-cache fa-debug ensure-model

.DEFAULT_GOAL := run

build: cutlass $(MAIN_OBJ) $(KERNELS_OBJ) $(FP4_OBJS) $(FA_DEBUG_OBJ) $(PQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BIN) $(MAIN_OBJ) $(KERNELS_OBJ) $(FP4_OBJS) $(FA_DEBUG_OBJ) $(PQ_OBJS) $(LDFLAGS)

$(FA_DEBUG_OBJ): ../gpu-cuda/fa_debug.c ../gpu-cuda/fa_debug.h
	$(CC) $(CFLAGS) $(FA_BR_FLAG) -c ../gpu-cuda/fa_debug.c -o $@

fa-debug: build
	@echo "======== FA debug compare (Hello, gen=12) ========"
	@echo "--- FP16 ---"
	$(MAKE) -C ../gpu-cuda build CUDA_GENCODE=$(CUDA_GENCODE) FA_BR=$(BLACKWELL_FA_BR)
	../gpu-cuda/qwen3-gpu-cuda "$(MODEL)" -p "Hello" -n 12 -t 0 -s 42 --fa-debug 2>&1 | grep -E '^FA_DEBUG|^    ' || true
	@echo "--- NVFP4 ---"
	./$(BIN) "$(MODEL)" -p "Hello" -n 12 -t 0 -s 42 --fa-debug 2>&1 | grep -E '^FA_DEBUG|^    ' || true

fp4_cache_io.o: fp4_cache_io.c fp4_cache.h
	$(CC) $(CFLAGS) -c fp4_cache_io.c -o $@

build.polarquant:
	$(MAKE) build BONSAI_POLARQUANT=1

$(MAIN_OBJ): $(SRC_DIR)/main.c $(SRC_DIR)/gpu.h $(SRC_DIR)/fa_debug.h fp4_qwen3.h fp4_cache.h $(SRC_DIR)/polarquant.h
	$(CC) $(CFLAGS) $(FP4_FLAG) $(PQ_FLAG) -c $(SRC_DIR)/main.c -o $@

$(KERNELS_OBJ): $(SRC_DIR)/kernels.cu $(SRC_DIR)/gpu.h $(SRC_DIR)/fa_debug.h fp4_qwen3.h $(SRC_DIR)/polarquant.h
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) $(FP4_FLAG) $(PQ_FLAG) -c $(SRC_DIR)/kernels.cu -o $@

$(FP4_GEMM_OBJ): fp4_gemm.cu fp4_gemm.h | cutlass
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_gemm.cu -o $@

$(FP4_QWEN3_OBJ): fp4_qwen3.cu fp4_qwen3.h fp4_gemm.h | cutlass
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_qwen3.cu -o $@

polarquant.o: $(SRC_DIR)/polarquant.cu $(SRC_DIR)/polarquant.h $(SRC_DIR)/polarquant_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/polarquant.cu -o polarquant.o

fp4-test: cutlass $(FP4_GEMM_OBJ) fp4_cache_io.o
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -o fp4_verify fp4_verify.cu $(FP4_GEMM_OBJ) fp4_cache_io.o $(LDFLAGS)
	./fp4_verify

sfa-verify: cutlass
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) --expt-relaxed-constexpr \
		-x cu -o sfa_index_verify sfa_index_verify.cpp
	./sfa_index_verify

pq-test: polarquant.o
	$(NVCC) $(NVCCFLAGS) -DBONSAI_POLARQUANT=1 -o polarquant_verify $(SRC_DIR)/polarquant_verify.cu polarquant.o $(LDFLAGS)
	./polarquant_verify

cutlass:
	@if [ ! -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" ]; then \
		echo "=== Cloning CUTLASS $(CUTLASS_TAG) ==="; \
		mkdir -p third_party; \
		git clone --depth 1 --branch $(CUTLASS_TAG) \
			https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	elif [ -d "$(CUTLASS_DIR)/.git" ] && \
	     [ "$$(git -C $(CUTLASS_DIR) describe --tags --exact-match 2>/dev/null)" != "$(CUTLASS_TAG)" ]; then \
		echo "=== Upgrading CUTLASS to $(CUTLASS_TAG) ==="; \
		rm -rf $(CUTLASS_DIR); \
		mkdir -p third_party; \
		git clone --depth 1 --branch $(CUTLASS_TAG) \
			https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	fi

ensure-model:
	@if [ ! -f "$(MODEL)" ]; then \
		echo ""; \
		echo "Model not found: $(MODEL)"; \
		echo "Running: make -C .. model"; \
		echo ""; \
		$(MAKE) -C .. model; \
	fi

# run / build: binary + NVFP4 cache (pack-cache runs when manifest is missing or stale).
$(FP4_CACHE_MANIFEST): $(MODEL) build
	@test -f "$(MODEL)" || (echo "error: model not found: $(MODEL)" >&2; echo "  run: make -C .. model" >&2; exit 1)
	./$(BIN) "$(MODEL)" --pack-nvfp4-cache

pack-cache: $(FP4_CACHE_MANIFEST)

run: ensure-model pack-cache
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

run.polarquant: ensure-model build.polarquant pack-cache
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f $(BIN) fp4_verify polarquant_verify \
	      main.pq*.o kernels.*.o fp4_gemm.sm120a.o fp4_qwen3.sm120a.o fp4_cache_io.o polarquant.o

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model weight H2D is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-24T15:03:39|sm_120a|24365a8c2057|132|128|622.60|66.71|122.03
BENCH_LOG += 2026-05-29T02:02:20|sm_120a|de9a61c86ea0|132|128|619.62|66.81|122.13
BENCH_LOG += 2026-05-29T07:28:24|sm_120a|de9a61c86ea0|132|13|4623.10|66.64|648.35
# BENCH_LOG_END

log:
	@echo ""
	@echo "==============================================="
	@echo "  gpu-cuda-nvfp4 benchmark history"
	@echo "==============================================="
	@echo ""
	@count=$$(sed -n '/^BENCH_LOG += /p' Makefile | wc -l); \
	if [ "$$count" -eq 0 ]; then \
		echo "  (no entries yet — run: make log.push)"; \
	else \
		printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n" \
			"Date" "GPU" "Host" "Prompt" "Gen" "Prefill/s" "Decode/s" "Total/s"; \
		printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n" \
			"-------------------" "--------" "--------------" "--------" "------" "----------" "----------" "----------"; \
		sed -n '/^BENCH_LOG += /s/^BENCH_LOG += //p' Makefile | \
		awk -F'|' '{ \
			date = $$1; sub(/[+].*$$/, "", date); \
			printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n", \
				date, $$2, $$3, $$4, $$5, $$6, $$7, $$8 }'; \
	fi
	@echo ""

log.push: build
	@if [ ! -f "$(MODEL)" ]; then \
		echo "error: model not found: $(MODEL)" >&2; \
		echo "  run: make -C .. model" >&2; \
		exit 1; \
	fi
	@echo "Running benchmark (prompt=\"$(BENCH_PROMPT)\", -n $(BENCH_N), -t $(BENCH_TEMP), -s $(BENCH_SEED)) ..."
	@BENCH_LOG_FILE="$(BENCH_LOG_FILE)" ./$(BIN) "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t $(BENCH_TEMP) -s $(BENCH_SEED); \
	if [ ! -f "$(BENCH_LOG_FILE)" ]; then \
		echo "error: benchmark log not found: $(BENCH_LOG_FILE)" >&2; \
		exit 1; \
	fi; \
	np=$$(sed -n 's/^prompt_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	ng=$$(sed -n 's/^gen_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	pf=$$(sed -n 's/^prefill_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	dc=$$(sed -n 's/^decode_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	tt=$$(sed -n 's/^total_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	if [ -z "$$np" ] || [ -z "$$pf" ] || [ -z "$$dc" ] || [ -z "$$tt" ]; then \
		echo "error: failed to parse $(BENCH_LOG_FILE)" >&2; \
		cat "$(BENCH_LOG_FILE)" >&2; \
		exit 1; \
	fi; \
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_SM)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(GPU_SM)"; \
	echo "  run: make log"

# apt CUDA 11 を削除 → NVIDIA 公式 CUDA 13 をインストール → NVFP4 ビルド
blackwell:
	@if [ "$$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi; \
	. /etc/os-release; \
	case "$${ID}-$${VERSION_ID}" in \
		ubuntu-22.04) CUDA_REPO=ubuntu2204 ;; \
		ubuntu-24.04) CUDA_REPO=ubuntu2404 ;; \
		ubuntu-20.04) CUDA_REPO=ubuntu2004 ;; \
		debian-12)    CUDA_REPO=debian12 ;; \
		debian-13)    CUDA_REPO=debian13 ;; \
		*) echo "Error: unsupported OS for CUDA 13 install ($${PRETTY_NAME:-$$ID $$VERSION_ID})."; \
		   echo "See https://developer.nvidia.com/cuda-downloads"; exit 1 ;; \
	esac; \
	echo "=== Removing apt CUDA Toolkit 11.x packages ==="; \
	CUDA11_PKGS="nvidia-cuda-toolkit nvidia-cuda-toolkit-doc nvidia-cuda-dev nvidia-cuda-gdb nvidia-profiler nvidia-visual-profiler libcupti-dev libcupti-doc libcupti11.5 libnvrtc11.2 libnvrtc-builtins11.5 libcudart11.0"; \
	to_remove=""; \
	for pkg in $$CUDA11_PKGS; do \
		if dpkg-query -W -f='$${Status}' "$$pkg" 2>/dev/null | grep -q "install ok installed"; then \
			to_remove="$$to_remove $$pkg"; \
		fi; \
	done; \
	if [ -n "$$to_remove" ]; then \
		$$SUDO apt-get remove --purge -y $$to_remove; \
		$$SUDO apt-get autoremove -y; \
	else \
		echo "No apt CUDA 11.x packages installed."; \
	fi; \
	echo "=== Installing CUDA Toolkit 13 from NVIDIA ($$CUDA_REPO) ==="; \
	KEYRING_DEB="cuda-keyring_1.1-1_all.deb"; \
	KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/$$CUDA_REPO/x86_64/$$KEYRING_DEB"; \
	if ! dpkg-query -W -f='$${Status}' cuda-keyring 2>/dev/null | grep -q "install ok installed"; then \
		wget -O "/tmp/$$KEYRING_DEB" "$$KEYRING_URL"; \
		$$SUDO dpkg -i "/tmp/$$KEYRING_DEB"; \
		rm -f "/tmp/$$KEYRING_DEB"; \
	fi; \
	$$SUDO apt-get update; \
	$$SUDO apt-get install -y cuda-toolkit-13 git; \
	if [ ! -x "$(CUDA13_NVCC)" ]; then \
		echo "Error: $(CUDA13_NVCC) not found after cuda-toolkit-13 install."; \
		exit 1; \
	fi; \
	nvcc_ver=$$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([0-9][0-9]*\)\(\.[0-9]*\)*.*/\1/p' | head -1); \
	if [ "$$nvcc_ver" -lt 13 ] 2>/dev/null; then \
		echo "Error: expected CUDA Toolkit 13+, got release $$nvcc_ver"; \
		exit 1; \
	fi; \
	echo "=== CUDA $$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -1) ready ==="; \
	echo "=== Configuring CUDA PATH ==="; \
	CUDA_PROFILE="/etc/profile.d/cuda.sh"; \
	CUDA_BIN="/usr/local/cuda/bin"; \
	if [ ! -f "$$CUDA_PROFILE" ] || ! grep -q 'cuda/bin' "$$CUDA_PROFILE" 2>/dev/null; then \
		printf '%s\n' 'export PATH="/usr/local/cuda/bin$${PATH:+:$${PATH}}"' | $$SUDO tee "$$CUDA_PROFILE" > /dev/null; \
		$$SUDO chmod 644 "$$CUDA_PROFILE"; \
	fi; \
	$$SUDO ln -sf "$$CUDA_BIN/nvcc" /usr/local/bin/nvcc; \
	$$SUDO ln -sf "$$CUDA_BIN/nvlink" /usr/local/bin/nvlink; \
	if ! command -v nvcc >/dev/null 2>&1; then export PATH="$$CUDA_BIN:$$PATH"; fi; \
	echo "nvcc: $$(command -v nvcc)"; \
	$(MAKE) cutlass; \
	echo "=== Building NVFP4 ($(BLACKWELL_GENCODE), FA_BR=$(BLACKWELL_FA_BR)) ==="; \
	$(MAKE) build
