# AMD Radeon GPU (ROCm / HIP)
ROCM       ?= /opt/rocm
HIPCC      ?= $(ROCM)/bin/hipcc
ROCMININFO ?= $(ROCM)/bin/rocminfo

# Auto-detect GPU ISA from rocminfo (first GPU agent Name: gfx*). Override: make GPU_ARCH=gfx1100
GPU_ARCH   ?= $(shell $(ROCMININFO) 2>/dev/null | awk '/^  Name:/ { n=$$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }')

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean detect-gpu-arch

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
