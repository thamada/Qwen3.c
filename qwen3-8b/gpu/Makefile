# AMD Radeon GPU (ROCm / HIP)
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean

build:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o qwen3-rocm -x hip main.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run: build
	./qwen3-rocm "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-rocm
