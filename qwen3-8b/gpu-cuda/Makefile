# NVIDIA CUDA。CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# GPU アーキテクチャ（CUDA 11.x は -arch=native 非対応）。
# 既定は PTX (compute_86) を生成し、ドライバ JIT で新 GPU にも対応。
# 例: make build CUDA_GENCODE=arch=compute_90,code=sm_90
#
# Blackwell GPU（RTX 50 系等）: make blackwell
#   apt CUDA 11 削除 → CUDA 13 インストール → NVFP4 Tensor Core (CUTLASS) + sm_120 ビルド

CUDA_GENCODE ?= arch=compute_86,code=compute_86

CUDA_HOME ?= /usr/local/cuda

NVCC    ?= $(CUDA_HOME)/bin/nvcc
CC      ?= cc

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter \
           -I$(CUDA_HOME)/include
LDFLAGS ?= -L$(CUDA_HOME)/lib64 -lcudart -lm

# /usr/local/bin/nvcc のみ PATH にあるとリンク時の nvlink が見つからない
ifeq ($(findstring /,$(NVCC)),/)
CUDA_BIN := $(dir $(shell readlink -f $(NVCC)))
else
CUDA_BIN := $(dir $(shell readlink -f $$(command -v $(NVCC) 2>/dev/null || echo $(CUDA13_NVCC))))
endif
export PATH := $(CUDA_BIN):$(PATH)

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
             -use_fast_math --expt-relaxed-constexpr \
             -gencode $(CUDA_GENCODE)

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

BLACKWELL_GENCODE ?= arch=compute_120a,code=sm_120a
BLACKWELL_FA_BR     ?= 32
CUDA13_NVCC         ?= /usr/local/cuda/bin/nvcc

CUTLASS_DIR ?= third_party/cutlass
CUTLASS_TAG ?= v3.9.0

# Blackwell FP4 ビルド: BONSAI_FP4=1（make run / make build.fp4 が設定）
BONSAI_FP4 ?= 0

FP4_CXXFLAGS = -std=c++17 --expt-relaxed-constexpr \
               -I$(CUTLASS_DIR)/include \
               -I$(CUTLASS_DIR)/tools/util/include \
               -DCUTLASS_ARCH_MMA_SM120_SUPPORTED

FP4_FLAG :=
FP4_OBJS :=
ifneq ($(BONSAI_FP4),0)
FP4_FLAG  := -DBONSAI_FP4=1
FP4_OBJS  := fp4_gemm.o fp4_qwen3.o
endif

# Blackwell (sm_120) は静的 shared 上限 48 KB。FA_BR=32 で Flash Attention が収まる。
FA_BR_FLAG :=
FA_BR_VAL  := $(if $(FA_BR),$(FA_BR),64)
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

# BONSAI_FP4 / FA_BR が変わったら kernels を必ず再コンパイル（古い .o 再利用防止）
KERNELS_OBJ := kernels.bfp4$(BONSAI_FP4).fabr$(FA_BR_VAL).o

.PHONY: build build.fp4 run run.no-fp4 clean blackwell cutlass fp4-test

fp4-test: fp4_gemm.o | cutlass
	$(NVCC) $(NVCCFLAGS) $(FP4_CXXFLAGS) -o fp4_verify fp4_verify.cu fp4_gemm.o $(LDFLAGS)
	./fp4_verify

.DEFAULT_GOAL := run

build: main.o $(KERNELS_OBJ) $(FP4_OBJS)
	$(NVCC) $(NVCCFLAGS) -o qwen3-gpu-cuda main.o $(KERNELS_OBJ) $(FP4_OBJS) $(LDFLAGS)

build.fp4: cutlass
	$(MAKE) build CUDA_GENCODE="$(BLACKWELL_GENCODE)" FA_BR="$(BLACKWELL_FA_BR)" BONSAI_FP4=1

build.no-fp4:
	$(MAKE) build CUDA_GENCODE="$(CUDA_GENCODE)" BONSAI_FP4=0

main.o: main.c gpu.h fp4_qwen3.h
	$(CC) $(CFLAGS) $(FP4_FLAG) -c main.c -o main.o

$(KERNELS_OBJ): kernels.cu gpu.h fp4_qwen3.h
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) $(FP4_FLAG) -c kernels.cu -o $@

fp4_gemm.o: fp4_gemm.cu fp4_gemm.h | cutlass
	$(NVCC) $(NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_gemm.cu -o fp4_gemm.o

fp4_qwen3.o: fp4_qwen3.cu fp4_qwen3.h fp4_gemm.h | cutlass
	$(NVCC) $(NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_qwen3.cu -o fp4_qwen3.o

cutlass:
	@if [ ! -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" ]; then \
		echo "=== Cloning CUTLASS $(CUTLASS_TAG) ==="; \
		mkdir -p third_party; \
		git clone --depth 1 --branch $(CUTLASS_TAG) \
			https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	fi

run.no-fp4: build.no-fp4
	./qwen3-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-gpu-cuda fp4_verify main.o kernels.*.o fp4_gemm.o fp4_qwen3.o

# apt CUDA 11 を削除 → NVIDIA 公式 CUDA 13 をインストール → Blackwell FP4 ビルド
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
	echo "=== Building Blackwell FP4 ($(BLACKWELL_GENCODE), FA_BR=$(BLACKWELL_FA_BR)) ==="; \
	$(MAKE) build.fp4

run: build.fp4
	./qwen3-gpu-cuda "$(MODEL)" -p "$(PROMPT)"
