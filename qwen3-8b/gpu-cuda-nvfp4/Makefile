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
CUTLASS_TAG ?= v3.9.0

FP4_CXXFLAGS = -std=c++17 --expt-relaxed-constexpr \
               -I$(CUTLASS_DIR)/include \
               -I$(CUTLASS_DIR)/tools/util/include \
               -DCUTLASS_ARCH_MMA_SM120_SUPPORTED

# NVFP4 GEMM オブジェクトは sm_120a 固定 gencode
BLACKWELL_NVCCFLAGS = -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
                      -use_fast_math --expt-relaxed-constexpr \
                      -gencode $(BLACKWELL_GENCODE)

FP4_FLAG  := -DBONSAI_FP4=1
FP4_GEMM_OBJ  := fp4_gemm.sm120a.o
FP4_QWEN3_OBJ := fp4_qwen3.sm120a.o
FP4_OBJS      := $(FP4_GEMM_OBJ) $(FP4_QWEN3_OBJ)

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

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build build.polarquant run run.polarquant clean blackwell cutlass fp4-test pq-test

.DEFAULT_GOAL := run

build: cutlass $(MAIN_OBJ) $(KERNELS_OBJ) $(FP4_OBJS) $(PQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BIN) $(MAIN_OBJ) $(KERNELS_OBJ) $(FP4_OBJS) $(PQ_OBJS) $(LDFLAGS)

build.polarquant:
	$(MAKE) build BONSAI_POLARQUANT=1

$(MAIN_OBJ): $(SRC_DIR)/main.c $(SRC_DIR)/gpu.h fp4_qwen3.h $(SRC_DIR)/polarquant.h
	$(CC) $(CFLAGS) $(FP4_FLAG) $(PQ_FLAG) -c $(SRC_DIR)/main.c -o $@

$(KERNELS_OBJ): $(SRC_DIR)/kernels.cu $(SRC_DIR)/gpu.h fp4_qwen3.h $(SRC_DIR)/polarquant.h
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) $(FP4_FLAG) $(PQ_FLAG) -c $(SRC_DIR)/kernels.cu -o $@

$(FP4_GEMM_OBJ): fp4_gemm.cu fp4_gemm.h | cutlass
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_gemm.cu -o $@

$(FP4_QWEN3_OBJ): fp4_qwen3.cu fp4_qwen3.h fp4_gemm.h | cutlass
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_qwen3.cu -o $@

polarquant.o: $(SRC_DIR)/polarquant.cu $(SRC_DIR)/polarquant.h $(SRC_DIR)/polarquant_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/polarquant.cu -o polarquant.o

fp4-test: cutlass $(FP4_GEMM_OBJ)
	$(NVCC) $(BLACKWELL_NVCCFLAGS) $(FP4_CXXFLAGS) -o fp4_verify fp4_verify.cu $(FP4_GEMM_OBJ) $(LDFLAGS)
	./fp4_verify

pq-test: polarquant.o
	$(NVCC) $(NVCCFLAGS) -DBONSAI_POLARQUANT=1 -o polarquant_verify $(SRC_DIR)/polarquant_verify.cu polarquant.o $(LDFLAGS)
	./polarquant_verify

cutlass:
	@if [ ! -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" ]; then \
		echo "=== Cloning CUTLASS $(CUTLASS_TAG) ==="; \
		mkdir -p third_party; \
		git clone --depth 1 --branch $(CUTLASS_TAG) \
			https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	fi

run: build
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

run.polarquant: build.polarquant
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f $(BIN) fp4_verify polarquant_verify \
	      main.pq*.o kernels.*.o fp4_gemm.sm120a.o fp4_qwen3.sm120a.o polarquant.o

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
