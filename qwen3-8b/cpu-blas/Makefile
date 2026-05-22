# OpenBLAS + OpenMP。ヘッダが見つからない場合は CPPFLAGS で追加してください。
# 例（Debian/Ubuntu の pthread ビルド）:
#   make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
# OpenBLAS は main 内で openblas_set_num_threads(1) 固定。並列度は OMP_NUM_THREADS で調整。

CC      ?= cc
# -ffast-math は IQ2_S/IQ3_S 量子化内積・RMSNorm で数値が崩れ推論が壊れるため使わない
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp
LDFLAGS ?= -fopenmp -lopenblas -lm

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null)
ifneq ($(strip $(OPENBLAS_LIBS)),)
  CPPFLAGS += $(OPENBLAS_CFLAGS)
  LDFLAGS := $(filter-out -lopenblas,$(LDFLAGS))
  LDFLAGS += $(OPENBLAS_LIBS)
endif

.PHONY: build run clean

build:
	$(CC) $(CPPFLAGS) $(CFLAGS) -o qwen3-cpu-blas main.c $(LDFLAGS)

run: build
	./qwen3-cpu-blas "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-cpu-blas
