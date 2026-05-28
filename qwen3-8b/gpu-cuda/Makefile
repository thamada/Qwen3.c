# NVIDIA CUDA (FP16 線形層 — NVFP4 なし)。
# NVFP4 版は ../gpu-cuda-nvfp4/ を参照。
#
# CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# 既定: 先頭 GPU の compute capability から -gencode を自動選択（nvidia-smi）。
#   RTX 50 / Blackwell (12.x) → sm_120 + FA_BR=32（PTX JIT の compute_86 は推論が壊れる）
#   上記以外のフォールバック → compute_86 PTX
# 手動例: make build CUDA_GENCODE=arch=compute_90,code=sm_90

CUDA13_NVCC ?= /usr/local/cuda/bin/nvcc

# nvidia-smi の "12.0" → 120
GPU_CCAP_NUM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{print $$1$$2}')
ifeq ($(GPU_CCAP_NUM),120)
DEFAULT_GENCODE := arch=compute_120,code=sm_120
DEFAULT_FA_BR   := 32
else ifeq ($(GPU_CCAP_NUM),121)
DEFAULT_GENCODE := arch=compute_120a,code=sm_120a
DEFAULT_FA_BR   := 32
else ifeq ($(GPU_CCAP_NUM),90)
DEFAULT_GENCODE := arch=compute_90,code=sm_90
else ifeq ($(GPU_CCAP_NUM),89)
DEFAULT_GENCODE := arch=compute_89,code=sm_89
else ifeq ($(GPU_CCAP_NUM),86)
DEFAULT_GENCODE := arch=compute_86,code=sm_86
else
DEFAULT_GENCODE := arch=compute_86,code=compute_86
endif

CUDA_GENCODE ?= $(DEFAULT_GENCODE)
FA_BR        ?= $(or $(DEFAULT_FA_BR),64)

CUDA_HOME ?= /usr/local/cuda

# Blackwell 向けには CUDA 13 の nvcc を優先（存在すれば）
ifneq ($(filter 120 121,$(GPU_CCAP_NUM)),)
ifneq ($(wildcard $(CUDA13_NVCC)),)
NVCC ?= $(CUDA13_NVCC)
endif
endif
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

# Benchmark defaults for log.push (override: make log.push BENCH_N=64)
# BENCH_PROMPT is ~128 tokens (ChatML-encoded prompt length).
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42
BENCH_LOG_FILE ?= /tmp/benchmark.log

# sm_* label for benchmark log (nvidia-smi compute cap, e.g. 12.0 → sm_120)
GPU_SM := sm_$(GPU_CCAP_NUM)

# PolarQuant KV cache: BONSAI_POLARQUANT=1（make build.polarquant）
BONSAI_POLARQUANT ?= 0

PQ_FLAG :=
PQ_OBJS :=
ifneq ($(BONSAI_POLARQUANT),0)
PQ_FLAG   := -DBONSAI_POLARQUANT=1
PQ_OBJS   := polarquant.o
endif

FA_BR_FLAG :=
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

FA_BR_VAL := $(FA_BR)

# CUDA_GENCODE / FA_BR が変わったら kernels を必ず再コンパイル
BUILD_STAMP  := .build_config.stamp
BUILD_CONFIG := $(CUDA_GENCODE)|fa=$(FA_BR_VAL)|pq=$(BONSAI_POLARQUANT)

KERNELS_OBJ := kernels.fabr$(FA_BR_VAL).pq$(BONSAI_POLARQUANT).o
MAIN_OBJ    := main.pq$(BONSAI_POLARQUANT).o
BIN         := qwen3-gpu-cuda

.PHONY: build build.polarquant run run.polarquant clean pq-test log log.push pack-cache FORCE

.DEFAULT_GOAL := run

$(BUILD_STAMP): FORCE
	@printf '%s\n' '$(BUILD_CONFIG)' > $@.new
	@if ! cmp -s $@.new $@ 2>/dev/null; then mv $@.new $@; else rm -f $@.new; fi

FORCE:

build: $(MAIN_OBJ) $(KERNELS_OBJ) $(PQ_OBJS) fp16_cache_io.o
	@echo "=== build: GPU_CCAP=$(GPU_CCAP_NUM) CUDA_GENCODE=$(CUDA_GENCODE) FA_BR=$(FA_BR_VAL) NVCC=$(NVCC) ==="
	$(NVCC) $(NVCCFLAGS) -o $(BIN) $(MAIN_OBJ) $(KERNELS_OBJ) $(PQ_OBJS) fp16_cache_io.o $(LDFLAGS)

build.polarquant:
	$(MAKE) build BONSAI_POLARQUANT=1

$(MAIN_OBJ): main.c gpu.h polarquant.h fp16_cache.h
	$(CC) $(CFLAGS) $(PQ_FLAG) -c main.c -o $@

fp16_cache_io.o: fp16_cache_io.c fp16_cache.h
	$(CC) $(CFLAGS) -c fp16_cache_io.c -o fp16_cache_io.o

$(KERNELS_OBJ): kernels.cu gpu.h polarquant.h $(BUILD_STAMP)
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

pack-cache: build
	./$(BIN) "$(MODEL)" --pack-fp16-cache

clean:
	rm -f $(BIN) polarquant_verify .build_config.stamp \
	      main.o main.pq*.o kernels.*.o polarquant.o fp16_cache_io.o

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model weight H2D is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-24T15:10:24|compute_86|24365a8c2057|132|16|62.83|75.27|63.98
BENCH_LOG += 2026-05-28T22:36:23|sm_120|de9a61c86ea0|132|16|62.80|75.91|64.00
# BENCH_LOG_END

log:
	@echo ""
	@echo "==============================================="
	@echo "  gpu-cuda benchmark history"
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
	@echo "Running benchmark (prompt=\"$(BENCH_PROMPT)\", -n $(BENCH_N), -t 0, -s $(BENCH_SEED)) ..."
	@BENCH_LOG_FILE="$(BENCH_LOG_FILE)" ./$(BIN) "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED); \
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
