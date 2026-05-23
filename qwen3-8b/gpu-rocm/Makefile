# AMD Radeon GPU (ROCm / HIP)
ROCM       ?= /opt/rocm
HIPCC      ?= $(ROCM)/bin/hipcc
ROCMININFO ?= $(ROCM)/bin/rocminfo

# Auto-detect GPU ISA from rocminfo (first GPU agent Name: gfx*). Override: make GPU_ARCH=gfx1100
GPU_ARCH   ?= $(shell $(ROCMININFO) 2>/dev/null | awk '/^  Name:/ { n=$$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }')

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# Benchmark defaults for log.push (override: make log.push BENCH_N=64)
# BENCH_PROMPT is ~128 tokens (ChatML-encoded prompt length).
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42

.PHONY: build run clean detect-gpu-arch log log.push

# -----------------------------------------------------------------------------
# GPU architecture (auto-detect via rocminfo, or override on command line)
# -----------------------------------------------------------------------------

detect-gpu-arch:
	@if [ -z "$(GPU_ARCH)" ]; then \
		echo "error: GPU_ARCH not detected. Ensure $(ROCMININFO) reports a GPU, or set GPU_ARCH manually (e.g. make GPU_ARCH=gfx1100)." >&2; \
		exit 1; \
	fi
	@echo ""
	@echo "==============================================="
	@echo "  Detected GPU arch: $(GPU_ARCH)"
	@echo "==============================================="
	@echo ""
# -----------------------------------------------------------------------------
# Build / run
# -----------------------------------------------------------------------------

build: detect-gpu-arch
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o qwen3-rocm -x hip main.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run: build
	./qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-rocm

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model weight H2D is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-23T15:54:16|gfx1100|7d630c88a170|132|8|28.74|29.35|28.77
BENCH_LOG += 2026-05-23T16:22:01|gfx1100|7d630c88a170|132|16|53.95|25.49|48.14
# BENCH_LOG_END

log:
	@echo ""
	@echo "==============================================="
	@echo "  gpu-rocm benchmark history"
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
	@out=$$(./qwen3-rocm "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED) 2>&1); \
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
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_ARCH)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(GPU_ARCH)"; \
	echo "  run: make log"
