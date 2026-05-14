CC        ?= cc
CFLAGS    ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS   ?= -lm

XDNA_INCS ?=

MODEL     ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT    ?= Hello, how are you?

# XDNA2 + BFPX ホスト重み（mmap 早期解放）。
.PHONY: build run clean

build:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o qwen3-xdna2-bfpx main.c $(LDFLAGS)

run: build
	./qwen3-xdna2-bfpx "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-xdna2-bfpx
