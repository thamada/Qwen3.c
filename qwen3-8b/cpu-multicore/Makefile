CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

.DEFAULT_GOAL := run

.PHONY: build run clean ensure-model

build:
	$(CC) $(CFLAGS) -fopenmp -o qwen3-cpu-omp main.c $(LDFLAGS)

ensure-model:
	@if [ ! -f "$(MODEL)" ]; then \
		echo ""; \
		echo "Model not found: $(MODEL)"; \
		echo "Running: make -C .. model"; \
		echo ""; \
		$(MAKE) -C .. model; \
	fi

run: build ensure-model
	./qwen3-cpu-omp "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f qwen3-cpu-omp
