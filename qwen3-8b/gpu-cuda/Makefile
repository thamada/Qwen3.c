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

# Benchmark defaults for log.push (override: make log.push BENCH_N=64)
# BENCH_PROMPT is ~128 tokens (ChatML-encoded prompt length).
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42

# GPU identifier for BENCH_LOG (CUDA_GENCODE の code=、例: compute_86 / sm_90)
GPU_SM := $(shell echo '$(CUDA_GENCODE)' | sed -n 's/.*code=\([^,]*\).*/\1/p')

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

.PHONY: build build.polarquant run run.polarquant clean pq-test log log.push

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

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model weight H2D is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-24T15:10:24|compute_86|24365a8c2057|132|16|62.83|75.27|63.98
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
	@out=$$(./$(BIN) "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED) 2>&1); \
	np=$$(echo "$$out" | sed -n 's/^--- \([0-9][0-9]*\) prompt tokens + \([0-9][0-9]*\) generated tokens ---/\1/p' | tail -1); \
	ng=$$(echo "$$out" | sed -n 's/^--- \([0-9][0-9]*\) prompt tokens + \([0-9][0-9]*\) generated tokens ---/\2/p' | tail -1); \
	pf=$$(echo "$$out" | sed -n 's/^prefill_tps: \([0-9.][0-9.]*\)/\1/p' | tail -1); \
	dc=$$(echo "$$out" | sed -n 's/^decode_tps: \([0-9.][0-9.]*\)/\1/p' | tail -1); \
	tt=$$(echo "$$out" | sed -n 's/^total_tps: \([0-9.][0-9.]*\)/\1/p' | tail -1); \
	if [ -z "$$np" ] || [ -z "$$pf" ] || [ -z "$$dc" ] || [ -z "$$tt" ]; then \
		echo "error: failed to parse benchmark output" >&2; \
		echo "$$out" >&2; \
		exit 1; \
	fi; \
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_SM)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(GPU_SM)"; \
	echo "  run: make log"
