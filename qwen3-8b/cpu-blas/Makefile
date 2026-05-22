# OpenBLAS + OpenMP。ヘッダが見つからない場合は CPPFLAGS で追加してください。
# 例（Debian/Ubuntu の pthread ビルド）:
#   make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
# OpenBLAS は main 内で openblas_set_num_threads(1) 固定。並列度は OMP_NUM_THREADS で調整。

CC      ?= cc
# -ffast-math は IQ2_S/IQ3_S 量子化内積・RMSNorm で数値が崩れ推論が壊れるため使わない
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp -march=native
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

.PHONY: build run clean openblas

build:
	@echo "$(CC) $(CPPFLAGS) $(CFLAGS) -o qwen3-cpu-blas main.c $(LDFLAGS)"
	@out=$$($(CC) $(CPPFLAGS) $(CFLAGS) -o qwen3-cpu-blas main.c $(LDFLAGS) 2>&1); rc=$$?; \
	if [ $$rc -ne 0 ]; then \
		echo "$$out"; \
		if echo "$$out" | grep -qE 'fatal error:.*cblas\.h|fatal error:.*openblas/cblas\.h|cblas\.h: No such file|openblas/cblas\.h: No such file'; then \
			echo ""; \
			echo "Error: OpenBLAS development headers not found (cblas.h)."; \
			echo ""; \
			echo "Install OpenBLAS with:"; \
			echo "  make openblas"; \
			echo ""; \
			echo "If headers are already installed but in a non-standard path (Debian/Ubuntu pthread build), try:"; \
			echo "  make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread"; \
			echo ""; \
		fi; \
		exit $$rc; \
	fi

run: build
	./qwen3-cpu-blas "$(MODEL)" -p "$(PROMPT)"

# Debian/Ubuntu 等: libopenblas-dev + OpenMP ランタイム
openblas:
	@if [ "$$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi; \
	echo "=== Installing OpenBLAS (libopenblas-dev) and OpenMP runtime (libgomp1) ==="; \
	$$SUDO apt-get update; \
	$$SUDO apt-get install -y libopenblas-dev libgomp1

clean:
	rm -f qwen3-cpu-blas
