# Vulkan compute (FP16 linear layers + Flash Attention)
# Requires: libvulkan-dev, glslang-tools, vulkan-loader

CC       ?= cc
GLSLANG  ?= glslangValidator

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lvulkan -lm

SHADERS := $(wildcard shaders/*.comp)
SPV     := $(SHADERS:.comp=.spv)

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

FP16_CACHE_MANIFEST := $(MODEL).fp16/manifest

.PHONY: build run clean pack-cache log log.push shaders

.DEFAULT_GOAL := build

shaders: $(SPV)

shaders/%.spv: shaders/%.comp shaders/common.inc
	$(GLSLANG) -V $< -o $@ -Ishaders

qwen3-vulkan: main.o vk_kernels.o vk_pipeline.o vk_alloc.o vk_context.o fp16_cache_io.o $(SPV)
	$(CC) $(CFLAGS) -o qwen3-vulkan main.o vk_kernels.o vk_pipeline.o vk_alloc.o vk_context.o fp16_cache_io.o $(LDFLAGS)

build: qwen3-vulkan

main.o: main.c gpu.h fp16_cache.h vk_alloc.h
	$(CC) $(CFLAGS) -c main.c -o main.o

vk_kernels.o: vk_kernels.c gpu.h vk_alloc.h vk_context.h vk_pipeline.h
	$(CC) $(CFLAGS) -c vk_kernels.c -o vk_kernels.o

vk_pipeline.o: vk_pipeline.c vk_pipeline.h vk_context.h
	$(CC) $(CFLAGS) -c vk_pipeline.c -o vk_pipeline.o

vk_alloc.o: vk_alloc.c vk_alloc.h vk_context.h
	$(CC) $(CFLAGS) -c vk_alloc.c -o vk_alloc.o

vk_context.o: vk_context.c vk_context.h
	$(CC) $(CFLAGS) -c vk_context.c -o vk_context.o

fp16_cache_io.o: fp16_cache_io.c fp16_cache.h
	$(CC) $(CFLAGS) -c fp16_cache_io.c -o fp16_cache_io.o

ifneq ($(wildcard $(MODEL)),)
BUILD_PREREQS := qwen3-vulkan $(FP16_CACHE_MANIFEST)
else
BUILD_PREREQS := qwen3-vulkan
endif

$(FP16_CACHE_MANIFEST): $(MODEL) qwen3-vulkan
	@test -f "$(MODEL)" || (echo "error: model not found: $(MODEL)" >&2; exit 1)
	QWEN3_VK_SHADER_DIR=$(CURDIR)/shaders ./qwen3-vulkan "$(MODEL)" --pack-fp16-cache

pack-cache: $(FP16_CACHE_MANIFEST)

run: qwen3-vulkan
	QWEN3_VK_SHADER_DIR=$(CURDIR)/shaders ./qwen3-vulkan "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-vulkan *.o shaders/*.spv

# Benchmark log (make log / make log.push)
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42
BENCH_LOG_FILE ?= /tmp/benchmark.log

GPU_VK := $(shell QWEN3_VK_SHADER_DIR=. vulkaninfo --summary 2>/dev/null | awk '/deviceName/ {print $$2; exit}' || echo vulkan)

log:
	@echo ""
	@echo "==============================================="
	@echo "  gpu-vulkan benchmark history"
	@echo "==============================================="
	@echo ""
	@count=$$(sed -n '/^BENCH_LOG += /p' Makefile | wc -l); \
	if [ "$$count" -eq 0 ]; then \
		echo "  (no entries yet — run: make log.push)"; \
	else \
		printf "  %-19s %14s %14s %8s %6s %10s %10s %10s\n" \
			"Date" "GPU" "Host" "Prompt" "Gen" "Prefill/s" "Decode/s" "Total/s"; \
		sed -n '/^BENCH_LOG += /s/^BENCH_LOG += //p' Makefile | \
		awk -F'|' '{ printf "  %-19s %14s %14s %8s %6s %10s %10s %10s\n", $$1, $$2, $$3, $$4, $$5, $$6, $$7, $$8 }'; \
	fi
	@echo ""

# BENCH_LOG_BEGIN
BENCH_LOG += 2026-05-29T09:21:52|=|a0ea39c6eb9d|132|16|2.89|2.27|2.81
# BENCH_LOG_END

log.push: build
	@if [ ! -f "$(MODEL)" ]; then echo "error: model not found: $(MODEL)" >&2; exit 1; fi
	@QWEN3_VK_SHADER_DIR=$(CURDIR)/shaders BENCH_LOG_FILE="$(BENCH_LOG_FILE)" \
		./qwen3-vulkan "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED)
	@np=$$(sed -n 's/^prompt_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	ng=$$(sed -n 's/^gen_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	pf=$$(sed -n 's/^prefill_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	dc=$$(sed -n 's/^decode_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	tt=$$(sed -n 's/^total_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_VK)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo "Logged on $(GPU_VK)"
