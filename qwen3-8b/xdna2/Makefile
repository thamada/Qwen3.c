CC        ?= cc
CFLAGS    ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS   ?= -lm

# `<drm/drm.h>` が標準外にあるときだけ `-I…` を渡す。
XDNA_INCS ?=

MODEL     ?= ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT    ?= Hello, how are you?

# AMD Ryzen AI XDNA2 NPU — amdxdna DRM ioctl。Kernel >= 6.10、`/dev/accel/accel0`。
.PHONY: build run gen-xdna-kernels clean

build:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o qwen3-xdna2 main.c $(LDFLAGS)

run: build
	./qwen3-xdna2 "$(MODEL)" -p "$(PROMPT)"

# スタブのみ（実 NPU ctrlcode ではない）— リポジトリルート基準。
gen-xdna-kernels:
	cd ../.. && python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels

clean:
	rm -f qwen3-xdna2
