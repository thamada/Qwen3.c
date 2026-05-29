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
BENCH_LOG_FILE ?= /tmp/benchmark.log

FP16_CACHE_MANIFEST := $(MODEL).fp16/manifest

.PHONY: run clean detect-gpu-arch log log.push pack-cache wmma wmma-probe

LLVM_OBJDUMP ?= $(ROCM)/llvm/bin/llvm-objdump

# WMMA check (make wmma): static + rocBLAS lib ISA + optional runtime (needs MODEL)
# Static-only without model: make wmma WMMA_SKIP_RUN=1
# Enable rocprof runtime ISA: make wmma WMMA_SKIP_ROCPROF=0
WMMA_PROMPT      ?= Hello
WMMA_N           ?= 0
WMMA_SKIP_RUN    ?= 0
WMMA_SKIP_ROCPROF ?= 1

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

CXX ?= g++

# hipcc (ROCm clang) does not search GCC libstdc++ headers/libs by default.
CXX_INC_DIRS := $(shell $(CXX) -E -x c++ - -v </dev/null 2>&1 | \
	awk '/#include <\.\.\.> search starts here:/,/^End of search list\./{gsub(/^ /,""); if ($$0 ~ /^\//) print $$0}')
ifeq ($(CXX_INC_DIRS),)
  GCC_VER := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
  CXX_INC_DIRS := /usr/include/c++/$(GCC_VER) /usr/include/x86_64-linux-gnu/c++/$(GCC_VER)
endif
HIP_CXXFLAGS := $(foreach d,$(CXX_INC_DIRS),-isystem$(d))

# libstdc++.so symlink lives under gcc's lib dir (not only libstdc++.so.6 in /usr/lib/...-gnu).
STDCXX_LIBDIR ?= $(shell dirname "$$($(CXX) -print-file-name=libstdc++.a 2>/dev/null)" 2>/dev/null)
LDFLAGS_STDCXX := $(if $(STDCXX_LIBDIR),-L$(STDCXX_LIBDIR),-L/usr/lib/gcc/x86_64-linux-gnu/$(shell $(CXX) -dumpversion | cut -d. -f1)) -lstdc++

HIP_CFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
	--offload-arch=$(GPU_ARCH) -I$(ROCM)/include $(HIP_CXXFLAGS)
HIP_LDFLAGS := -L$(ROCM)/lib -lamdhip64 -lhipblas -lrocblas -Wl,-rpath,$(ROCM)/lib $(LDFLAGS_STDCXX)

ifeq ($(GPU_ARCH),)
$(error GPU_ARCH not detected. Ensure $(ROCMININFO) reports a GPU, or: make GPU_ARCH=gfx1100 build)
endif

# run: binary only (skip pack-cache). build: binary + FP16 cache when MODEL exists.
ifneq ($(wildcard $(MODEL)),)
BUILD_PREREQS := qwen3-rocm $(FP16_CACHE_MANIFEST)
else
BUILD_PREREQS := qwen3-rocm
endif

qwen3-rocm: main.o fp16_cache_io.o
	@test -n "$(CXX_INC_DIRS)" || (echo "error: C++ headers not found. Install: apt install g++ libstdc++-dev" >&2; exit 1)
	@echo ""
	@echo "==============================================="
	@echo "  Detected GPU arch: $(GPU_ARCH)"
	@echo "==============================================="
	@echo ""
	$(HIPCC) $(HIP_CFLAGS) -o qwen3-rocm main.o fp16_cache_io.o $(HIP_LDFLAGS)

# Offline FP16 cache (<model>.gguf.fp16/). Rebuilt when MODEL or binary is newer than manifest.
$(FP16_CACHE_MANIFEST): $(MODEL) qwen3-rocm
	@test -f "$(MODEL)" || (echo "error: model not found: $(MODEL)" >&2; echo "  run: make -C .. model" >&2; exit 1)
	./qwen3-rocm "$(MODEL)" --pack-fp16-cache

pack-cache: $(FP16_CACHE_MANIFEST)

build: $(BUILD_PREREQS)

main.o: main.c fp16_cache.h
	@test -n "$(CXX_INC_DIRS)" || (echo "error: C++ headers not found. Install: apt install g++ libstdc++-dev" >&2; exit 1)
	$(HIPCC) $(HIP_CFLAGS) -c -x hip main.c -o main.o

fp16_cache_io.o: fp16_cache_io.c fp16_cache.h
	$(HIPCC) -O3 -Wall -Wextra -x c -c fp16_cache_io.c -o fp16_cache_io.o

run: qwen3-rocm
	./qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-rocm wmma-probe main.o fp16_cache_io.o

# -----------------------------------------------------------------------------
# WMMA usage check (make wmma)
# -----------------------------------------------------------------------------

wmma-probe: detect-gpu-arch
	$(HIPCC) -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		$(if $(findstring gfx11,$(GPU_ARCH)),-DWMMA_PROBE_HAS_GFX11=1,) \
		-o wmma-probe -x hip wmma_probe.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

wmma: qwen3-rocm wmma-probe
	@GPU_ARCH=$(GPU_ARCH) MODEL="$(MODEL)" ROCM=$(ROCM) \
		LLVM_OBJDUMP="$(LLVM_OBJDUMP)" \
		WMMA_PROMPT="$(WMMA_PROMPT)" WMMA_N=$(WMMA_N) \
		WMMA_SKIP_RUN=$(WMMA_SKIP_RUN) WMMA_SKIP_ROCPROF=$(WMMA_SKIP_ROCPROF) \
		bash scripts/check_wmma.sh

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model weight H2D is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-23T15:54:16|gfx1100|7d630c88a170|132|8|28.74|29.35|28.77
BENCH_LOG += 2026-05-23T16:22:01|gfx1100|7d630c88a170|132|16|53.95|25.49|48.14
BENCH_LOG += 2026-05-24T04:56:58|gfx1100|7d630c88a170|132|16|556.77|25.78|172.56
BENCH_LOG += 2026-05-27T18:42:13|gfx1201|5ae102a9321e|132|16|101.94|28.84|80.01
BENCH_LOG += 2026-05-29T08:12:06|gfx1201|a0ea39c6eb9d|132|16|124.95|28.88|91.89
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
	@BENCH_LOG_FILE="$(BENCH_LOG_FILE)" ./qwen3-rocm "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED); \
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
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_ARCH)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(GPU_ARCH)"; \
	echo "  run: make log"
