CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm
TARGET_CPU                     = qwen3-cpu
TARGET_CPU_OMP                 = qwen3-cpu-omp
TARGET_ROCM                    = qwen3-rocm
TARGET_ROCM_FULLGPU_FLASH_OPT2 = qwen3-rocm-fullgpu-flash-opt2

MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

ROCM    ?= /opt/rocm
HIPCC   ?= $(ROCM)/bin/hipcc
# gfx1201 = AMD Radeon AI PRO R9700 (matches the llama3-8b sibling target).
GPU_ARCH ?= gfx1201

# CPU reference inference (llama3-8b/main.c 相当)
build:
	$(CC) $(CFLAGS) -o $(TARGET_CPU) main.c $(LDFLAGS)

run: build
	./$(TARGET_CPU) $(MODEL) -p "$(PROMPT)"

# OpenMP 並列（main-omp.c）
build.omp:
	$(CC) $(CFLAGS) -fopenmp -o $(TARGET_CPU_OMP) main-omp.c $(LDFLAGS)

run.omp: build.omp
	./$(TARGET_CPU_OMP) $(MODEL) -p "$(PROMPT)"

# Qwen3-VL-8B (IQ2_M) — text-only decoder, layer-fused, FlashAttention pipeline.
# 既定エントリ: main-rocm.c（llama3-8b の main-rocm.c に相当する Make ターゲット名）
build.rocm:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o $(TARGET_ROCM) -x hip main-rocm.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run.rocm: build.rocm
	./$(TARGET_ROCM) $(MODEL) -p "$(PROMPT)"

build.rocm.fullgpu.flash.opt2:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o $(TARGET_ROCM_FULLGPU_FLASH_OPT2) -x hip main-rocm-fullgpu-flash-opt2.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run.rocm.fullgpu.flash.opt2: build.rocm.fullgpu.flash.opt2
	./$(TARGET_ROCM_FULLGPU_FLASH_OPT2) $(MODEL) -p "$(PROMPT)"

clean:
	rm -f $(TARGET_CPU) $(TARGET_CPU_OMP) $(TARGET_ROCM) $(TARGET_ROCM_FULLGPU_FLASH_OPT2)

.PHONY: build run build.omp run.omp build.rocm run.rocm build.rocm.fullgpu.flash.opt2 run.rocm.fullgpu.flash.opt2 clean
