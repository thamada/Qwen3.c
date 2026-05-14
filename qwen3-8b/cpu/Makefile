CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean

build:
	$(CC) $(CFLAGS) -o qwen3-cpu main.c $(LDFLAGS)

run: build
	./qwen3-cpu "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-cpu
