# NVIDIA CUDA (FP16 線形層 — NVFP4 なし)。
# NVFP4 版は ../gpu-cuda-nvfp4/ を参照。
#
# CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# 既定は PTX (compute_86) を生成し、ドライバ JIT で新 GPU にも対応。
# 例: make build CUDA_GENCODE=arch=compute_90,code=sm_90

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
CUDA_BIN := $(dir $(shell readlink -f $$(command -v $(NVCC) 2>/dev/null || echo $(NVCC))))
endif
export PATH := $(CUDA_BIN):$(PATH)

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
             -use_fast_math --expt-relaxed-constexpr \
             -gencode $(CUDA_GENCODE)

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# PolarQuant KV cache: BONSAI_POLARQUANT=1（make build.polarquant）
BONSAI_POLARQUANT ?= 0

PQ_FLAG :=
PQ_OBJS :=
ifneq ($(BONSAI_POLARQUANT),0)
PQ_FLAG   := -DBONSAI_POLARQUANT=1
PQ_OBJS   := polarquant.o
endif

FA_BR_FLAG :=
FA_BR_VAL  := $(if $(FA_BR),$(FA_BR),64)
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

KERNELS_OBJ := kernels.fabr$(FA_BR_VAL).pq$(BONSAI_POLARQUANT).o
MAIN_OBJ    := main.pq$(BONSAI_POLARQUANT).o
BIN         := qwen3-gpu-cuda

.PHONY: build build.polarquant run run.polarquant clean pq-test

.DEFAULT_GOAL := run

build: $(MAIN_OBJ) $(KERNELS_OBJ) $(PQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BIN) $(MAIN_OBJ) $(KERNELS_OBJ) $(PQ_OBJS) $(LDFLAGS)

build.polarquant:
	$(MAKE) build BONSAI_POLARQUANT=1

$(MAIN_OBJ): main.c gpu.h polarquant.h
	$(CC) $(CFLAGS) $(PQ_FLAG) -c main.c -o $@

$(KERNELS_OBJ): kernels.cu gpu.h polarquant.h
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) $(PQ_FLAG) -c kernels.cu -o $@

polarquant.o: polarquant.cu polarquant.h polarquant_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c polarquant.cu -o polarquant.o

pq-test: polarquant.o
	$(NVCC) $(NVCCFLAGS) -DBONSAI_POLARQUANT=1 -o polarquant_verify polarquant_verify.cu polarquant.o $(LDFLAGS)
	./polarquant_verify

run: build
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

run.polarquant: build.polarquant
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f $(BIN) polarquant_verify \
	      main.o main.pq*.o kernels.*.o polarquant.o
