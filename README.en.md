# Qwen3.c

This repository is an **inference implementation** that runs **Qwen3-family models** directly from **a single C source**, **without relying on external libraries**.

**This project does not link userland ML libraries or runtimes such as PyTorch, TensorFlow, JAX, or ONNX Runtime.**  
The reference implementation uses **standard C and `libm` only**: build the **single-thread CPU** binary (`cpu/qwen3-cpu`) from `qwen3-8b/cpu/main.c`.  
For faster trials on the same GGUF, build **`qwen3-cpu-omp`** from `qwen3-8b/cpu-multicore/main.c` (**OpenMP**; runtime: **standard C + `libm` + OpenMP**).  
**`qwen3-8b/cpu-blas/`** adds **OpenMP + OpenBLAS** and **Q8_K activations + AVX2 integer dots for all types** to produce **`qwen3-cpu-blas`** (**standard C + `libm` + OpenMP + OpenBLAS**).  
**ROCm/HIP** (AMD GPU), **Vulkan compute** (vendor-neutral GPU), **CUDA** (NVIDIA GPU), and **XDNA2 NPU** (`amdxdna` ioctl) builds are **appendix** material at the **end of this README** (the main focus is the **three CPU variants**).

### Why avoid ML libraries?

Typical LLM inference using PyTorch or similar stacks is short to write and fast to run. In that setup, though, low-level details—**execution order, memory layout, alignment, quantization packing**—often hide inside the framework or runtime.

This repository deliberately skips that layer and **makes the full path visible in C**: reading GGUF, restoring weights, linear algebra, Transformer forward, and sampling. The goal is not to replace existing frameworks but to **inspect, validate, and change** the inference path when needed.

That choice helps with:

- **Understandability**: You can follow what is read from the model file, which buffers hold it, and in what order computation runs, straight from the sources and `doc/design.md`.
- **Simpler dependencies**: You do not need a Python stack or a large ML tree—just a C toolchain and a minimal environment to exercise the path.
- **Experimentation**: Quantization formats, memory layouts (e.g. BFPX), splitting work across CPU/GPU/NPU, and direct access to `/dev/accel` are easier to try without framework abstractions in the way.
- **Reference value**: It shows how Qwen3-style decoder inference can work with minimal scaffolding and can serve as a baseline for comparison or validation.

So this is **not** aimed at maximum performance or full feature parity. The focus is **not** treating LLM inference as a black box, but letting developers see and modify the implementation.

The scope is the **text decoder of Qwen3-VL-8B-Instruct**. Image input and the vision encoder are **out of scope**; use cases are prompt-in, text-out generation.

日本語版は [README.md](README.md) を参照してください。

## What you can run

There are **three CPU variants**: **single-thread**, **OpenMP**, and **OpenMP + OpenBLAS**.

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | Learning the flow, minimal setup. **Prefill progress bar** and throughput summary on stderr |
| CPU OpenMP | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | Faster CPU trials |
| CPU OpenMP + OpenBLAS | `qwen3-8b/cpu-blas/main.c` | `cpu-blas/qwen3-cpu-blas` | BLAS for F32 GEMV and attention; quantized GEMV uses **Q8_K activations + AVX2 integer dots for all types** (layer-shared Q8). **RoPE cache**, prefill **LM head skip**, greedy **`mm_argmax_row`**. **F16 embedding via F16C**. **Prefill progress bar** on stderr |

An 8B model on CPU is **very slow**. CPU is fine for a first smoke test; for usable throughput, prefer **`cpu-blas`**. Faster paths for AMD GPU, cross-vendor Vulkan GPU, NVIDIA GPU, and XDNA2 NPU are in the **appendix** at the end of this README.

## Repository layout

```text
.
├── README.md
├── README.en.md
├── doc/
│   ├── ChangeLog.md
│   └── design.md
└── qwen3-8b/
    ├── Makefile
    ├── gguf.txt
    ├── cpu/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-multicore/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-blas/
    │   ├── Makefile
    │   └── main.c
    ├── gpu-rocm/          # appendix (end of README)
    │   ├── Makefile
    │   ├── main.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   ├── wmma_probe.c          (`make wmma-probe` — WMMA detector calibration)
    │   └── scripts/
    │       └── check_wmma.sh     (`make wmma`)
    ├── gpu-vulkan/        # appendix (end of README · Vulkan compute)
    │   ├── Makefile
    │   ├── main.c
    │   ├── gpu.h
    │   ├── vk_context.c/h / vk_alloc.c/h / vk_pipeline.c/h / vk_kernels.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   └── shaders/              (GLSL compute → `.spv` via `make`)
    ├── gpu-cuda/          # appendix (end of README · FP16)
    │   ├── Makefile
    │   ├── main.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   ├── kernels.cu
    │   ├── gpu.h
    │   └── polarquant.cu / polarquant_kernels.cuh / polarquant_verify.cu  (PolarQuant-R KV)
    ├── gpu-cuda-nvfp4/    # appendix (end of README · NVFP4)
    │   ├── Makefile
    │   ├── fp4_cache.h / fp4_cache_io.c / fp4_gemm.cu / fp4_qwen3.cu / fp4_verify.cu
    │   └── third_party/cutlass/  (fetched via make cutlass)
    │   (main.c / kernels.cu / gpu.h / polarquant.* reference ../gpu-cuda/)
    ├── xdna2/             # appendix (end of README)
    │   ├── Makefile
    │   ├── main.c
    │   └── xdna-gemv/
    │       ├── README.md
    │       ├── gen-xdna-gemv-stubs.py
    │       ├── kernels/
    │       └── toolchain/
    ├── xdna2-bfp16/       # appendix (end of README · BFPX)
    │   ├── Makefile
    │   └── main.c
    └── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

The reference inference code is **`qwen3-8b/cpu/`** (single-thread). Parallel builds live in **`qwen3-8b/cpu-multicore/`**; the optimized CPU build is **`qwen3-8b/cpu-blas/`**. GPU builds **`gpu-rocm`** (AMD · ROCm/HIP), **`gpu-vulkan`** (AMD/NVIDIA/Intel etc. · Vulkan compute), **`gpu-cuda`** (NVIDIA · FP16), and **`gpu-cuda-nvfp4`** (NVIDIA · NVFP4) are **appendix** material. XDNA2 NPU builds **`xdna2`** / **`xdna2-bfp16`** are also appendix (end of this README). Use **`make model`** under `qwen3-8b/` to fetch the GGUF; **build and run** from each subdirectory Makefile.

## Beginners: what happens during LLM inference?

Roughly:

1. **Read the GGUF file**  
   A large file with weights, vocabulary, and hyperparameters.

2. **Tokenize the prompt**  
   Turn a string like `"Hello"` into a sequence of integer token IDs.

3. **Run the Transformer one token at a time**  
   The model predicts likely next tokens.

4. **Sample**  
   Pick the next token from the prediction. Adjust behavior with `-t` (temperature) and `-k` (top-p).

5. **Decode tokens to text**  
   Print the chosen tokens as human-readable text.

Here, that pipeline is **not** hidden inside PyTorch: you can follow it **in the C sources**.

## Requirements

### Common

- Linux
- `make`
- A C compiler (e.g. `gcc`, `clang`, `cc`)
- `libm` (usually provided by the system)
- A GGUF file for Qwen3-VL-8B-Instruct

On Ubuntu-like systems, CPU builds often need only:

```bash
sudo apt update
sudo apt install -y build-essential make
```

### OpenMP build

GCC typically builds with `-fopenmp`. Some setups need the OpenMP runtime:

```bash
sudo apt install -y libgomp1
```

### OpenBLAS build (`cpu-blas`)

You need **OpenBLAS** (e.g. `libopenblas-dev`) and the OpenMP runtime. When `pkg-config openblas` works, the Makefile picks up include/link flags automatically. You can also install packages via **`make openblas`** in **`cpu-blas/`** (`libopenblas-dev` / `libgomp1` via apt).

```bash
sudo apt install -y libopenblas-dev libgomp1
# or
cd qwen3-8b/cpu-blas && make openblas
```

If headers are not on the default path (e.g. Debian/Ubuntu pthread build), set `CPPFLAGS` at build time. When `cblas.h` is missing, the Makefile prints **`make openblas`** and a **`CPPFLAGS`** example.

```bash
cd qwen3-8b/cpu-blas
make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
```

At run time, set **`OMP_NUM_THREADS`** for CPU parallelism. OpenBLAS is fixed to **one thread** via **`openblas_set_num_threads(1)`** to avoid nested parallelism with OpenMP. **Do not use `-ffast-math`** for this build—it breaks IQ / Q8_K quantized dot products (disabled in the bundled Makefile). Default **`CFLAGS`** include **`-march=native`** (optimize for the build CPU over portability).


## Obtain the model file

The default name in `qwen3-8b/Makefile` is:

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

The GGUF is **not** shipped in the repo (license + size). Place it under `qwen3-8b/`. Recommended: **`make model`** (`wget` + bundled `.sha256sum` verification; skips download when the file is already present and the checksum passes).

```bash
cd qwen3-8b
make model
```

On success, the terminal prints a checksum verification banner.

Manual download:

```bash
cd qwen3-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf "$url"
sha256sum -c Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

You should then have:

```text
qwen3-8b/
├── Makefile
├── cpu/ … (`main.c` → `cpu/qwen3-cpu`)
├── cpu-multicore/ …
├── cpu-blas/ …
├── gpu-rocm/ …
├── gpu-vulkan/ …
├── gpu-cuda/ …
├── gpu-cuda-nvfp4/ …
├── xdna2/ …
├── xdna2-bfp16/ …
└── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

Verify SHA256:

```bash
cd qwen3-8b
sha256sum -c Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

`OK` means the file name and hash match what this repo expects.

## Quick start

Build the CPU binary first. An 8B model is slow on CPU; use a small `-n` (e.g. `-n 1`) for a quick check.

```bash
cd qwen3-8b
make model          # if missing: download and verify GGUF (skip if already verified)
cd cpu && make build
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

On success, text should appear gradually after load.

## CPU single-thread

### Build

```bash
cd qwen3-8b/cpu
make build
```

Produces **`qwen3-cpu`** (inside `cpu/`).

```bash
ls -lh qwen3-cpu
```

### Run

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries.

```bash
cd qwen3-8b/cpu
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Give a one-sentence introduction of yourself." \
  -n 16
```

Using the subdirectory `Makefile` `run` target:

```bash
cd qwen3-8b/cpu
make run PROMPT="Give a one-sentence introduction of yourself."
```

Model elsewhere:

```bash
cd qwen3-8b/cpu
make run MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP

Uses multiple CPU cores; same model file as single-thread.

### Build

```bash
cd qwen3-8b/cpu-multicore
make build
```

Produces **`qwen3-cpu-omp`**.

### Run

```bash
cd qwen3-8b/cpu-multicore
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain in bullet points what quantization is." \
  -n 32
```

`OMP_NUM_THREADS` sets thread count; try 4 or 8 first.

```bash
cd qwen3-8b/cpu-multicore
OMP_NUM_THREADS=4 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

Speedup depends on core count, memory bandwidth, and quantization.

## CPU OpenMP + OpenBLAS

Same decoder and GGUF as **`cpu-multicore`**, but **F32 matmul** (`cblas_sgemv`) and **attention K/V combine** go through OpenBLAS. For IQ2_S / IQ3_S / Q4_K / Q5_K quantized GEMV, activations are quantized to **Q8_K** (`quantize_row_q8_K`), then **`vec_dot_*_q8_K`** integer dot products (aligned with **ggml-cpu/quants.c**) are used—without full float[256] row dequantization like **`cpu-multicore`**. **GEMVs that share the same activation vector within a layer reuse one Q8 quantization** (layer-shared Q8). With **`__AVX2__`**, IQ2_S / IQ3_S / Q4_K / Q5_K dots and Q8 quantization are SIMD-accelerated (**`-march=native`** by default).

Additional CPU optimizations: **RoPE cos/sin cache** (precomputed at startup), **LM head skip** during prefill (except the last prompt token; **`FWD_NO_LM`**), **`mm_argmax_row`** for greedy sampling (**`-t 0`**, no full vocab logits buffer), and **F16 embedding** lookup via **F16C+AVX2**. During prefill, stderr shows a **Prefill progress bar** (`Prefill [====...]`, width 40) plus prefill / decode / total throughput summaries.

### Build

```bash
cd qwen3-8b/cpu-blas
make build
```

Produces **`qwen3-cpu-blas`** (inside `cpu-blas/`).

### Run

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries.

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Hello, how are you?" \
  -n 32
```

Via the subdirectory Makefile:

```bash
cd qwen3-8b/cpu-blas
make run PROMPT="Hello, how are you?"
```

## Common CLI options

| Option | Example | Meaning |
|---|---|---|
| `-p` | `-p "Hello"` | Input prompt |
| `-n` | `-n 64` | Max new tokens |
| `-t` | `-t 0.7` | Temperature (lower = sharper) |
| `-k` | `-k 0.9` | Top-p |
| `-s` | `-s 1234` | RNG seed |
| `-l` | `-l 512` | Max sequence length |

Start small:

```bash
cd qwen3-8b/cpu
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

Then increase `-n`:

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Write a short poem." \
  -n 128
```

## More deterministic output

Lower temperature and fix the seed when comparing runs:

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "One sentence: what is GGUF?" \
  -n 32 \
  -t 0.2 \
  -s 42
```

Byte-identical output across CPU vs GPU is not guaranteed; compare with the **same binary**, **same model**, and **same flags**.

## Clean

Remove build artifacts from each variant subdirectory:

```bash
cd qwen3-8b/cpu && make clean
cd qwen3-8b/cpu-multicore && make clean
cd qwen3-8b/cpu-blas && make clean
# appendix GPU / XDNA:
cd qwen3-8b/gpu-rocm && make clean
cd qwen3-8b/gpu-vulkan && make clean
cd qwen3-8b/gpu-cuda && make clean
cd qwen3-8b/gpu-cuda-nvfp4 && make clean
cd qwen3-8b/xdna2 && make clean
cd qwen3-8b/xdna2-bfp16 && make clean
```

Typical files removed:

- `cpu/qwen3-cpu`
- `cpu-multicore/qwen3-cpu-omp`
- `cpu-blas/qwen3-cpu-blas`
- `gpu-rocm/qwen3-rocm`
- `gpu-vulkan/qwen3-vulkan`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`
- `gpu-cuda/qwen3-gpu-cuda`
- `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4`

`make clean` does **not** delete the GGUF model.

## Troubleshooting

### `No such file or directory`

Wrong model path.

```bash
ls -lh qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

Put the model under `qwen3-8b/` or pass an absolute path:

```bash
cd qwen3-8b/cpu
./qwen3-cpu /data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

### CPU is slow

Expected for 8B on CPU alone. Try `-n 1` or `-n 4`:

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

On CPU only, **`cpu-blas/qwen3-cpu-blas`** (OpenBLAS + Q8_K quantized GEMV + AVX2 dots + layer-shared Q8 + RoPE cache + prefill LM skip + greedy argmax) often outperforms **`cpu-multicore`**. **Greedy (`-t 0`)** makes decode LM head even lighter. For AMD GPU, NVIDIA GPU, or XDNA2 NPU paths, see the **appendix** at the end of this README.

### `cpu-blas` build fails / `cblas.h` not found

Install OpenBLAS dev packages and set `CPPFLAGS` if needed (see **OpenBLAS build** under Requirements).

### `cpu-blas` output is garbage (repeated characters, etc.)

Building with **`-ffast-math`** breaks IQ / Q8_K quantized dot products. The repo Makefile disables it—remove it if you override `CFLAGS`.

### `sha256sum -c` fails

File name or contents differ from this repo’s expectations:

- Is the file named `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`?
- Complete download?
- Wrong quantization variant?

Using another model is fine for hashing but the implementation must match supported GGUF metadata and tensor layout.

## Reading the codebase

Suggested order:

1. `README.en.md` (or `README.md`) — build and run successfully first.
2. `doc/design.md` — design, quantization, Qwen3 specifics.
3. `qwen3-8b/cpu/main.c` — GGUF load through one-token generation on CPU.
4. `qwen3-8b/cpu-multicore/main.c` — OpenMP parallelization.
5. `qwen3-8b/cpu-blas/main.c` — OpenBLAS (`cblas_sgemv`) for F32 GEMV and batched attention; Q8_K activations + AVX2 integer dots for all quant types (layer-shared Q8). RoPE cache, **`lm_mode`** (prefill LM skip / greedy **`mm_argmax_row`**), F16 emb F16C. See **`doc/design.md`**, section **“`cpu-blas`: Q8_K activation GEMV”**.

## Advanced features (multi-turn chat and Thinking mode)

The Qwen3 family (including reasoning lines such as QwQ) assumes **ChatML templates** and **Thinking mode** beyond a single `-p "..."` prompt. This repository implements only **decoder forward + sampling** in C; the advanced features below are **not implemented**. Keep the gap from official behavior in mind when using or extending the code.

General template background: [Qwen3 official blog](https://qwenlm.github.io/blog/qwen3/), [The 4 Things Qwen-3’s Chat Template Teaches Us (Hugging Face Blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive), [Qwen/Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B).

### Multi-turn dialogue

**Official behavior** — Append system / user / assistant turns in ChatML order and carry context via KV cache or re-prefill. In function calling, past assistant turns, tool results, and (when needed) reasoning regions are passed to the next turn.

**This repo today** — Each `main.c` implements **`chat_encode` as a fixed single turn only**:

```text
<|im_start|>system … <|im_end|>
<|im_start|>user\n{ string passed via -p }<|im_end|>
<|im_start|>assistant\n
```

There is no CLI for prior turns and **no conversation state across process invocations** (every run prefills from scratch). To approximate multi-turn behavior:

1. Hand-build ChatML history and pass it via `-p`
2. Extend `chat_encode` to accept a turn list
3. Reuse KV from a previous run (not supported today)

History longer than `-l` (`max_seq`) needs truncation or summarization.

**References (technical details)**

- [Transformers — Chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating) — `messages` to ChatML, `apply_chat_template`
- [Function Calling (Qwen docs)](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — Hermes-style format, chaining assistant / tool roles
- [Core concepts — Tool Calling (Qwen)](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html) — multi-turn / multi-step tool calling template example

### Thinking mode (reasoning before the final answer)

**Official behavior** — Qwen3 **hybrid thinking** (DeepSeek-R1 / QwQ-style “think then answer”) is toggled via **hard switch** (`apply_chat_template(..., enable_thinking=True/False)` or API `enable_thinking`) and **soft switch** (`/think` / `/no_think` appended to user messages; latest instruction wins in multi-turn). When enabled, a **thinking block** (a reasoning region inserted by the template) precedes the final answer. When disabled, an **empty thinking block** steers the model toward direct answers. These markers are tokenized as normal text, unlike ChatML special tokens such as `<|im_start|>`.

**This repo today** — It does **not**:

- Control generation prompts equivalent to **`enable_thinking`** (e.g. empty thinking block before assistant generation)
- **Separate or hide** thinking vs final answer (`print_tok` suppresses only ChatML special IDs; reasoning text goes to stdout as-is)
- Implement API-style extras such as **`thinking_budget`** or dedicated reasoning streams

Running thinking-capable GGUF weights as-is may **mix reasoning text into the terminal** or **degrade quality** on template mismatch. Correct support requires **`chat_encode` / generation-loop changes** aligned with GGUF `tokenizer.chat_template` metadata and **parsing of thinking regions** on output.

**References (technical details)**

- [Quickstart — Thinking & Non-Thinking Mode (Qwen)](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html) — hard / soft switch, `thinking_budget`, recommended sampling
- [Transformers inference guide (Qwen)](https://qwen.readthedocs.io/en/latest/inference/transformers.html) — toggling thinking, parsing `reasoning_content`
- [Thinking (Qwen Cloud)](https://docs.qwencloud.com/developer-guides/text-generation/thinking) — API `enable_thinking` / `thinking_budget` / `reasoning_content`
- [vLLM deployment (Qwen)](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) — `chat_template_kwargs.enable_thinking`, reasoning parser

### Where this repo stands (summary)

| Feature | Qwen3 family (official) | This repo (today) |
|---|---|---|
| ChatML single turn (system + user + assistant start) | Yes | Yes (fixed system string + `-p`) |
| Multi-turn history | Yes | No (manual ChatML in `-p` only) |
| KV / session persistence | Yes (framework) | No (single run only) |
| Thinking on/off | Yes (`enable_thinking`, etc.) | No |
| `/think` / `/no_think` | Yes (hybrid models) | No (not interpreted) |
| Filtered thinking display | Yes (API / UI) | No |

This repo is sufficient as a **one-shot text generation** reference. For **ChatGPT / Qwen API–class multi-turn chat or Thinking UI**, extend the code or use existing runtimes such as vLLM, llama.cpp, or Transformers.

## Out of scope

- Training / fine-tuning
- **AMD NPU (XDNA2, etc.)** and **ROCm / Vulkan / CUDA GPU** code (**`gpu-rocm`**, **`gpu-vulkan`**, **`gpu-cuda`**, **`gpu-cuda-nvfp4`**, **`xdna2`**, **`xdna2-bfp16`** are optional appendix builds; the main focus is the **three CPU variants**)
- Batch inference tuning (GPU appendix prefill batching is for faster decode, not server batching)
- Image input
- **Built-in multi-turn CLI** (history management, KV reuse, full official chat template)
- **Thinking mode control and filtered reasoning display** (`enable_thinking`, `/think`, `/no_think`, etc.)
- Server or Web API packaging
- Universal support for every GGUF quantization
- Guaranteed numerical match with official implementations

The goal is to **understand, experiment with, and adapt** Qwen3-family GGUF text inference in **C**.

## More documentation

- Design: `doc/design.md`
- Changelog: `doc/ChangeLog.md`

When stuck, confirm you ran **`make model`** under `qwen3-8b/`, built the binary in the correct variant subdirectory, and pass a model path that matches at runtime.

---

## AMD ROCm / HIP implementation (`gpu-rocm`)

**`qwen3-8b/gpu-rocm/` is an appendix outside this repo's goals** (single C source, minimal dependencies). It splits into `main.c` plus HIP kernels and requires **ROCm (`hipcc`), an AMD GPU driver, and physical hardware**. ROCm/HIP are **GPU compilers and runtimes**, not high-level neural network frameworks (the Transformer is built from custom HIP kernels and host code). The reference implementation to read first is **`cpu/main.c`**.

It is bundled only because the author **wanted to see how fast AMD GPUs could go**. It does not complement the project's purpose and is not an official feature for readers. It is easy to misread as part of the main project, so **`gpu-rocm/` is planned to move to a separate repository**. First-time readers can **ignore it**.

What follows is a technical note for anyone curious about GPU speed comparisons.

### Requirements (ROCm)

You need an **AMD GPU** and **ROCm**. The Makefile assumes ROCm at **`/opt/rocm`**. **`rocminfo`** reports **`Name: gfx*`** as **`GPU_ARCH_DETECTED`**; **`hipcc --offload-arch`** uses **`HIP_OFFLOAD_ARCH`**. On **`gfx1152`** / **`gfx1153`** (e.g. **Ryzen AI 5 340 + Radeon 840M**), official rocBLAS often ships **no `gfx1152` Tensile libraries** — the Makefile applies **`gfx1151` build + `HSA_OVERRIDE_GFX_VERSION=11.5.1`** automatically (**upgrading to ROCm 7.2.1 alone may not fix this** — see **“Ryzen AI / gfx1152 and rocBLAS”** below). Linking **`fp16_cache_io.o`** requires **g++ / libstdc++-dev**.

```bash
sudo apt install -y g++ libstdc++-dev
```

Check:

```bash
/opt/rocm/bin/hipcc --version
make -C gpu-rocm detect-gpu-arch   # e.g. Detected GPU arch: gfx1100
```

If `rocminfo` reports no GPU, set **`GPU_ARCH=gfx1100`** (or your ISA) at build time.

### `GPU_ARCH` (auto-detect)

`gpu-rocm/Makefile` reads the first GPU agent name (`gfx*`) from **`$(ROCM)/bin/rocminfo`** as **`GPU_ARCH_DETECTED`** (override via **`GPU_ARCH`**). On success you will see:

```text
===============================================
  Detected GPU arch: gfx1100
  Build offload arch: gfx1100
===============================================
```

To override manually:

```bash
cd qwen3-8b/gpu-rocm
make build GPU_ARCH=gfx1100
```

### Ryzen AI / gfx1152 and rocBLAS (environment-specific)

On **AMD Ryzen AI APUs** with integrated Radeon (**RDNA 3.5**), **`gfx1152`** (e.g. **Ryzen AI 5 340 + Radeon 840M**) often fails at Prefill linear layers because **hipBLAS → rocBLAS** cannot load Tensile kernels for that arch.

| Topic | Detail |
|-------|--------|
| Typical error | **`rocBLAS error: Cannot read … TensileLibrary.dat … for GPU arch : gfx1152`** (only **`gfx1150` / `gfx1151`** listed as available) |
| Root cause | No **`TensileLibrary_lazy_gfx1152.dat`** under official **`/opt/rocm/lib/rocblas/library/`** (as of **ROCm 7.1.x / 7.2.x** apt packages) |
| Is ROCm 7.2.1 required? | **Not for this GPU specifically.** Upgrading may still leave **`gfx1152`** Tensile unpackaged |
| This repo | **`HIP_OFFLOAD_ARCH=gfx1151`** + **`HSA_OVERRIDE_GFX_VERSION=11.5.1`** (auto on **`make build` / `make run`**) |

**Build log when `gfx1152` / `gfx1153` is detected:**

```text
  Detected GPU arch: gfx1152
  Build offload arch: gfx1151
  HSA_OVERRIDE_GFX_VERSION: 11.5.1
```

**Successful startup (override active; `gcnArchName` may show **`gfx1151`**):**

```text
ROCm HIP device 0: AMD Radeon 840M Graphics (gcnArchName: gfx1151)
Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)
```

**Other Ryzen AI SKUs:** e.g. Ryzen AI 9 HX 370 often reports **`gfx1150`** (**`TensileLibrary_lazy_gfx1150.dat`** bundled) — override usually not needed.

**Check:**

```bash
ls /opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx115*.dat
cd qwen3-8b/gpu-rocm && make detect-gpu-arch
```

**Avoid:** **symlink/copy `gfx1151` libs to `gfx1152` names only** → **`hipBLAS error: 6`**. **`HSA_OVERRIDE` only** with binary still built for **`gfx1152` offload** → **segmentation fault**.

See [`doc/design.md`](doc/design.md), section **“Environment: gfx1152 (Ryzen AI / Radeon 840M) and rocBLAS”** (Japanese heading in source).

### Build

```bash
cd qwen3-8b/gpu-rocm
make build
```

**`make build`**: builds **`qwen3-rocm`** and, when **`MODEL`** exists, also creates **`<model>.gguf.fp16/manifest`** (offline FP16 cache). **`make run`**: binary only (skips pack-cache).

If ROCm is not under `/opt/rocm`:

```bash
cd qwen3-8b/gpu-rocm
make build ROCM=/path/to/rocm
```

Produces **`qwen3-rocm`**. Links **`-lhipblas -lrocblas -lstdc++`** (Prefill linear GEMM + **`fp16_cache_io.o`**).

Startup logs **`Loading FP16 cache from …`** or **`Uploading weights (row dequant -> FP16)...`** confirm the FP16 load path is active.

### Offline FP16 cache (`make pack-cache`)

Pre-pack FP16 weights to skip GGUF dequant on every run. Default output: **`<model>.gguf.fp16`** (per-tensor **`.fp16bin`** files + **`manifest`**). Same format as **`gpu-vulkan`** and **`gpu-cuda`**.

```bash
cd qwen3-8b/gpu-rocm
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# When MODEL already exists, make build auto-packs as well
```

From the binary directly:

```bash
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-rocm ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-rocm ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

Re-run **`make pack-cache`** after updating the GGUF (**`manifest`** checks GGUF size and mtime).

### Prefill acceleration (overview)

The ROCm build **separates Prefill** (prompt processing) and **Decode** (token generation).

| Phase | Function | Linear layers | Description |
|-------|----------|---------------|-------------|
| **Prefill** | `forward_prefill_gpu` | **hipBLAS `GemmEx`** | All **S prompt tokens** in **one forward** |
| **Decode** | `forward_gpu` | Custom **GEMV** | **One token at a time** |

**Why Prefill can be much faster** (details in [`doc/design.md`](doc/design.md), section **“ROCm Prefill acceleration (3 stages)”**):

1. During **Prefill**, all prompt tokens are known, so linear layers become **S×d GEMM** (matrix×matrix). Weight HBM reads are **amortized over S tokens**.
2. **Decode** stays **GEMV** (matrix×vector) one token at a time — this dominates inter-token latency (ITL).

Improvements were done in **3 stages** (132 prompt tokens, RX 7900 XTX / gfx1100, `make log.push` equivalent):

| Stage | Method | prefill tok/s | Speedup vs stage 0 |
|-------|--------|---------------|-------------------|
| 0 (before) | One `forward_gpu` per token (GEMV × S) | 28.7 | 1.0× |
| 1 | Batched `forward_prefill_gpu` + custom `mm_f16_gemv_batch_kernel` | 54 | 1.9× |
| 2 | Above + **hipBLAS GemmEx** ([llama.cpp](https://github.com/ggml-org/llama.cpp/) `cublasGemmEx` equivalent) | **~550** | **~19×** |
| 2 (re-measured) | Stage 2 path (after `threadIdx` cast in `attn_flash_prefill_kernel`) | **~557** | **~19×** |

Stage 2 highlights:

- **Matmul**: `O[S,d] = X[S,n] @ W[d,n]^T` via **`hipblasGemmEx(OP_T, OP_N, ...)`** (FP16 weights, FP16 activations, FP32 output).
- **Activation conversion**: `f32_to_f16_batch_kernel` → **`d_scratch_f16`**. Shared inputs (q/k/v, gate/up) convert **once**.
- **Attention / RoPE / Norm / FFN activations** remain custom HIP batch kernels (`attn_flash_prefill_kernel`, etc.). Prefill Flash Attention casts **`threadIdx.x`** to **`(int)threadIdx.x`** when comparing with **`hd`** / **`tc`** (avoids **`-Wsign-compare`**; behavior unchanged).
- Startup log **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** confirms the hipBLAS path is active.

Stage 2 re-measurement (**`BENCH_LOG` 2026-05-24**, 132 prompt tokens): prefill **556.77** / **556.95** tok/s — similar to the first run (549.94; within measurement variance).

Decode throughput (~26 tok/s) is largely unchanged. Long prompts: **TTFT (Time To First Token)** tracks Prefill speed.

### Run

```bash
cd qwen3-8b/gpu-rocm
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what ROCm is for beginners." \
  -n 64
```

Using the subdirectory `run` target:

```bash
cd qwen3-8b/gpu-rocm
make run PROMPT="Short explanation in English."
```

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries (same format as **`cpu-blas`**). After inference, **`qwen3-rocm`** / **`qwen3-vulkan`** / **`gpu-cuda/`** / **`gpu-cuda-nvfp4/`** write a structured benchmark log to **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**) as key=value lines (model, GPU, token counts, tok/s, full prompt, etc.; **inference only**; model weight H2D excluded). tok/s plus **VRAM breakdown** (**`GpuVramProfile`** / **`model_vram_profile`**) appear under **`[vram_breakdown]`** (**`gpu-vulkan`**: **`vram_total`** plus a simplified **`[vram_breakdown]`**; linear weights are included in **`vram_total`** only). During weight upload, stdout prints **`layer N/L uploaded: X.XX sec, X.XX GB/sec`** every 8 layers. tok/s metrics appear only in the stderr **`--- throughput ---`** summary and **`BENCH_LOG_FILE`** (no stdout benchmark lines).

### Benchmark history (`gpu-rocm/Makefile`)

From **`gpu-rocm/`**, **`make log.push`** runs a benchmark with the default long prompt (~128 tokens), **`-n 128`**, and **`-t 0`**, reads metrics from **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**), then appends one line to **`BENCH_LOG`** in the Makefile. **`make log`** prints the history as a table.

```bash
cd qwen3-8b/gpu-rocm
make log.push                    # default BENCH_N=128, BENCH_SEED=42
make log                         # show history
make log.push BENCH_N=64         # override generation length, etc.
# Custom log path example:
make log.push BENCH_LOG_FILE=/tmp/my-bench.log
```

**Makefile append line** (pipe-separated): **`timestamp|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

For manual runs: **`BENCH_LOG_FILE=/path/to/log ./qwen3-rocm model.gguf -p "…" -n 64`**, then inspect **`prefill_tps=`** etc. in that file.

**Benchmark log file** (overwritten after inference): main keys **`timestamp`**, **`hostname`**, **`model`**, **`gpu`**, **`prompt_tokens`**, **`gen_tokens`**, **`prefill_tps`**, **`decode_tps`**, **`total_tps`**, **`vram_total`**, **`[vram_breakdown]`** (**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`**), plus the full prompt under **`--- prompt ---`**. **`make log.push`** (**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`**) reads **`prompt_tokens=`** etc. from this file and appends one line to **`BENCH_LOG`** in the Makefile. Makefile **`BENCH_LOG`** history records tok/s only; see **`BENCH_LOG_FILE`** for VRAM breakdown.

### WMMA usage check (`make wmma`)

Prefill linear layers go through **hipBLAS / rocBLAS**; **`main.c` does not embed WMMA / rocWMMA / MFMA directly** (Prefill GEMM is delegated to hipBLAS). **`make wmma`** verifies:

- **`main.c` / `qwen3-rocm` binary** has **no WMMA instructions** (expected)
- **`wmma-probe`** (gfx11 calibration binary) **does** contain WMMA (calibrates `llvm-objdump` detection)
- **rocBLAS** bundled ISA may contain WMMA (0 is OK — FMAC path WARN possible)
- Optional: runtime log **`Prefill linear: hipBLAS GemmEx`**, **`rocprofv3`** kernel trace

```bash
cd qwen3-8b/gpu-rocm
make wmma                           # build + wmma-probe + checks (needs MODEL)
make wmma WMMA_SKIP_RUN=1           # static checks only (no MODEL)
make wmma WMMA_SKIP_ROCPROF=0       # also try rocprofv3 kernel ISA
make wmma-probe                     # build calibration binary only
```

See [`doc/design.md`](doc/design.md) ROCm build section and **`scripts/check_wmma.sh`**.

### Source reading

6. `qwen3-8b/gpu-rocm/fp16_cache_io.c` / `main.c` — AMD GPU. **`<model>.gguf.fp16`** offline cache, GGUF fused row-wise dequant. **Prefill**: **`forward_prefill_gpu`** (**hipBLAS GemmEx**). **Decode**: **`forward_gpu`**. Prefill details in **`doc/design.md`** section on ROCm prefill optimization.

## Vulkan compute implementation (`gpu-vulkan`)

**`qwen3-8b/gpu-vulkan/`** is an **appendix** that runs GPU inference with **Vulkan 1.1 compute shaders only** — **no ROCm, no CUDA**. It targets **Vulkan-capable GPUs** (AMD via RADV/Mesa, NVIDIA proprietary, Intel, etc.) but **does not use vendor-specific libraries** (hipBLAS, cuBLAS, Tensor Cores, etc.). Read **`cpu/main.c`** first as the reference.

### Positioning (`gpu-rocm` / `gpu-cuda`)

| Aspect | `gpu-rocm` | `gpu-vulkan` | `gpu-cuda` |
|--------|------------|--------------|------------|
| Runtime | ROCm / HIP | Vulkan loader + compute | CUDA |
| Target GPU | AMD (ROCm required) | Any Vulkan GPU (vendor-neutral) | NVIDIA |
| Linear Prefill | hipBLAS GemmEx | FP16 GEMV batch (compute) | FP16 GEMV batch |
| Linear Decode | Custom GEMV (HIP) | FP16 GEMV (compute) | FP16 GEMV (CUDA) |
| Attention | Flash Attention (HIP) | Flash Attention (GLSL) | Flash Attention (CUDA) |
| FP16 cache | **`<model>.gguf.fp16`** (shared) | same | same |

Use this when you **cannot or do not want to install ROCm on AMD**, or when you want a **cross-vendor GPU path**. For **maximum throughput**, prefer **`gpu-rocm`** (AMD) or **`gpu-cuda`** (NVIDIA).

### Requirements

- **Vulkan 1.1+** GPU and driver (Linux examples: Mesa RADV, NVIDIA proprietary)
- Dev packages: **`libvulkan-dev`**
- Shader compile: **`glslang-tools`** (`glslangValidator`)
- Runtime: **`vulkan-loader`** (often **`libvulkan1`** on distros)

Check:

```bash
vulkaninfo --summary
glslangValidator --version
```

### Build and run

```bash
cd qwen3-8b/gpu-vulkan
make build          # shaders/*.comp → shaders/*.spv, then link qwen3-vulkan
make run            # default PROMPT="Hello, how are you?"
make pack-cache     # <model>.gguf.fp16 (same format as gpu-rocm / gpu-cuda)
```

**`make run`** sets **`QWEN3_VK_SHADER_DIR=$(pwd)/shaders`**. When running the binary directly, point **`QWEN3_VK_SHADER_DIR`** at the directory containing **SPIR-V (`.spv`)** files:

```bash
cd qwen3-8b/gpu-vulkan
QWEN3_VK_SHADER_DIR=$(pwd)/shaders ./qwen3-vulkan ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 64
```

CLI matches **`gpu-rocm` / `gpu-cuda`**: **`--pack-fp16-cache`**, **`--no-fp16-cache`**, **`-p` / `-n` / `-t` / `-k` / `-s` / `-l`**.

### Implementation layout

- **`main.c`** — GGUF load, tokenizer, FP16 weight H2D (based on `gpu-cuda`)
- **`vk_context.c`** — Vulkan instance, device, queue, command pool
- **`vk_alloc.c`** — device buffers (`vk_malloc` / H2D / D2H)
- **`vk_pipeline.c`** — compute pipelines and dispatch
- **`vk_kernels.c`** — `gpu_forward` / `gpu_forward_prefill` (`gpu.h` API)
- **`shaders/*.comp`** — RMSNorm, RoPE, FP16 GEMV, Flash Attention, etc. (18 shaders)

Weight loading matches **`gpu-rocm` / `gpu-cuda` (FP16)**: if **`<model>.gguf.fp16`** exists, H2D from **`.fp16bin`**; otherwise fused row-wise GGUF dequant → FP16 → VRAM. KV cache is **F32**.

### Known limitations and future work

**`gpu-vulkan`** is an **initial implementation verified to produce correct output**. Throughput is **typically much lower than `gpu-rocm`** (both kernel quality and host-side overhead).

Main factors:

1. **Kernel launch overhead** — many **`vkCmdDispatch`** calls per layer; each dispatch allocates/updates descriptor sets and waits on a fence (synchronous).
2. **Prefill linear layers** — no **batched GEMM** equivalent to **`gpu-rocm`** hipBLAS GemmEx; prefill and decode both use **FP16 GEMV compute shaders**.
3. **Portable shaders** — no WMMA / cooperative matrix intrinsics (GLSL portability first).
4. **Transfer path** — weight H2D via **staging buffers** (simpler than CUDA/HIP pinned memory).

Possible improvements (not implemented):

- **Reuse** descriptor sets and command buffers (reduce CPU overhead even if dispatch count stays the same)
- Prefill **batched GEMM** compute shaders (or **`VK_KHR_cooperative_matrix`**, etc.)
- **Mega-kernel** fusion (one dispatch per layer)
- Long-prompt bench improvements (currently **~3 tok/s** class; **~10–40×** slower than ROCm on the same GPU)

### Benchmark history (`gpu-vulkan/Makefile`)

Same pattern as **`gpu-rocm`**: **`make log.push`** / **`make log`** read **`BENCH_LOG_FILE`** and append to **`BENCH_LOG`** in the Makefile. Column 2 is the **GPU name from `vulkaninfo`**.

```bash
cd qwen3-8b/gpu-vulkan
make log.push    # long prompt ~132 tokens + -n 128 -t 0
make log
```

#### Vulkan GPU long prompt (`gpu-vulkan` · `make log.push`)

| Item | Value |
|---|---|
| GPU | AMD Radeon Graphics (**RADV GFX1201**, 32 GiB VRAM) |
| OS | Linux |
| Model | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` (H2D from **`<model>.gguf.fp16`** cache) |
| Command | **`make log.push`** |
| Workload | Long prompt (**132** tokens after ChatML) + decode **up to 128** (**`-n 128 -t 0 -s 42`**) |
| Table metrics | **`prefill_tps` / `decode_tps` / `total_tps`** in **`/tmp/benchmark.log`** (or **`BENCH_LOG_FILE`**) — inference interval only |
| Reproduce | `qwen3-8b/gpu-vulkan/` → **`make log.push`** → **`make log`** |

| Measured | GPU | prefill tok/s | decode tok/s | total tok/s | Notes |
|---|---|---:|---:|---:|---|
| 2026-05-29 09:39 | **RADV GFX1201** | **2.97** | **2.17** | **2.86** | 132+16 tokens (**`make log.push`**; **`-t 0`** → EOS stopped at 16 generated) |

**VRAM breakdown** (**2026-05-29 09:39** run; **`BENCH_LOG_FILE`** **`[vram_breakdown]`**; default **`-l`** → **`max_seq=512`**):

| Item | bytes | MiB | Notes |
|---|---:|---:|---|
| **`vram_total`** (theoretical sum) | 16,621,944,320 | **15851.92** | categories below + FP16 linear weights (**~14435 MiB**) |
| **`vram_device_total`** | 34,208,743,424 | **32624.00** | total GPU VRAM (**`vram_device_used`** not reported on Vulkan) |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm weights |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`** (depends on **`-l`**) |
| `vram_decode_activations` | 779,776 | **0.74** | single-token decode buffers |
| `vram_prefill_batch` | 88,082,432 | **84.00** | prefill batch (**`batch_cap = max_seq`**) |

On the same GPU, **`gpu-rocm`** (**`make log.push`** history) measured prefill **124.95** / decode **28.88** / total **91.89** tok/s. **This Vulkan run was ~42× slower on prefill and ~13× slower on decode** than ROCm (kernel cost plus large **`vkCmdDispatch`** sync overhead). Numbers are **environment-dependent**.

### Source reading

7. `qwen3-8b/gpu-vulkan/fp16_cache_io.c` / `main.c` / `vk_kernels.c` / `vk_pipeline.c` / `shaders/*.comp` — Vulkan compute. Forward split via **`gpu.h`**. **`QWEN3_VK_SHADER_DIR`** locates `.spv` files.

### Troubleshooting (`gpu-vulkan`)

**`Cannot open shader: …/xxx.spv`**

Confirm **`shaders/*.spv`** exist after **`make build`**. When running the binary directly, set **`QWEN3_VK_SHADER_DIR`** to the **`shaders/`** directory.

**`vkAllocateDescriptorSets` / OUT_OF_POOL_MEMORY**

One forward issues many dispatches and can exhaust the descriptor pool. Recent builds enlarge the pool and call **`vkFreeDescriptorSets`**. Try **`make clean && make build`**.

**RADV `not a conformant Vulkan implementation` warning**

Mesa RADV may print this during development. This appendix is for **validation**; it does not assume a fully conformant certified Vulkan implementation.

**Extremely slow vs ROCm / CUDA**

See **Known limitations** above. For performance, use **`gpu-rocm`** or **`gpu-cuda`**.

## NVIDIA CUDA implementation (`gpu-cuda` / `gpu-cuda-nvfp4`)

**`qwen3-8b/gpu-cuda/`** and **`qwen3-8b/gpu-cuda-nvfp4/`** are also **appendix** builds. They use `main.c` + `kernels.cu` (plus NVFP4 `fp4_*.cu`) and require **CUDA Toolkit (`nvcc`), an NVIDIA driver, and a physical GPU**. The NVFP4 build needs **CUDA 13 + Blackwell (sm_120 class)** and CUTLASS. First-time readers can **ignore them**.

What follows is a technical note for NVIDIA GPU builds.



### Requirements (CUDA)

You need an **NVIDIA GPU** and **CUDA Toolkit** (`nvcc`, `libcudart`). **`qwen3-8b/Makefile`** only fetches the model; CUDA builds live in:

- **`qwen3-8b/gpu-cuda/`** — FP16 linear layers (general NVIDIA GPUs)
- **`qwen3-8b/gpu-cuda-nvfp4/`** — NVFP4 linear layers (Blackwell / RTX 50 series, **CUDA 13** + CUTLASS). First run: **`make cutlass`** for **`third_party/cutlass`**. **`fp4_*` objects need C++17** (CUTLASS). Do not mix apt **`nvidia-cuda-toolkit` (CUDA 11)** with CUDA 13 (**`make blackwell`** removes 11.x and installs 13).

Check:

```bash
nvcc --version
nvidia-smi
```

Put CUDA **`bin`** on **`PATH`** (`/usr/local/bin/nvcc` alone may fail at link time).

| Use case | Command |
|------|----------|
| **ROCm FP16 offline cache** | `cd qwen3-8b/gpu-rocm` → `make pack-cache` (**`make build`** auto-packs when MODEL exists) |
| **Vulkan compute FP16** (no ROCm/CUDA · cross-vendor) | `cd qwen3-8b/gpu-vulkan` → `make build` / `make run` (**`QWEN3_VK_SHADER_DIR`** for `.spv`) |
| **Vulkan FP16 offline cache** | `cd qwen3-8b/gpu-vulkan` → `make pack-cache` |
| **FP16 only** (Ampere/Ada, PTX OK) | `cd qwen3-8b/gpu-cuda` → `make build` / `make run` |
| **FP16 offline cache** | `cd qwen3-8b/gpu-cuda` → `make pack-cache` |
| **PolarQuant-R KV** (FP16 linear, any GPU) | `cd qwen3-8b/gpu-cuda` → `make build.polarquant` / `make run.polarquant` |
| **Blackwell NVFP4** | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build` / `make run` |
| **NVFP4 offline cache** | `cd qwen3-8b/gpu-cuda-nvfp4` → `make pack-cache` |
| **NVFP4 + PolarQuant** | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build.polarquant` / `make run.polarquant` |
| CUDA 13 + NVFP4 full setup | `cd qwen3-8b/gpu-cuda-nvfp4` → `make blackwell` |
| PolarQuant round-trip test | `make pq-test` in each directory |
| CUTLASS NVFP4 GEMM unit test | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fp4-test` (**Blackwell / sm_120a required**) |
| SFA/SFB index verification | `cd qwen3-8b/gpu-cuda-nvfp4` → `make sfa-verify` |
| Flash Attention debug (Hello) | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fa-debug` (FP16 vs NVFP4) |
| Benchmark history (ROCm) | `cd qwen3-8b/gpu-rocm` → `make log.push` / `make log` (**`BENCH_LOG_FILE`**, default **`/tmp/benchmark.log`**) |
| Benchmark history (Vulkan) | `cd qwen3-8b/gpu-vulkan` → `make log.push` / `make log` (**`BENCH_LOG_FILE`**, default **`/tmp/benchmark.log`**) |
| Benchmark history (CUDA FP16) | `cd qwen3-8b/gpu-cuda` → `make log.push` / `make log` (**`BENCH_LOG_FILE`**, default **`/tmp/benchmark.log`**) |
| Benchmark history (CUDA NVFP4) | `cd qwen3-8b/gpu-cuda-nvfp4` → `make log.push` / `make log` (**`BENCH_LOG_FILE`**, default **`/tmp/benchmark.log`**) |

Default FP16 build (`gpu-cuda`) uses **`nvidia-smi` auto-detection** (Blackwell **12.x** → **`sm_120`** + **`FA_BR=32`**). **PTX `compute_86` JIT** can corrupt output on RTX 5090; avoid it. NVFP4 build defaults to **`sm_120a`**. **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** use **`BLACKWELL_NVCCFLAGS`** (**`-std=c++17`** + fixed **`sm_120a`**). CUTLASS **`v4.5.0`** is fetched via **`make cutlass`**.


| Directory | Weights at load | Linear / KV at runtime |
|-----------|-----------------|------------------------|
| **`gpu-rocm`** | If offline cache (**`<model>.gguf.fp16`**) exists, H2D from **`.fp16bin`**; otherwise **fused row-wise GGUF dequant** → FP16 | hipBLAS GemmEx (Prefill) + custom GEMV (Decode); KV in **F32** |
| **`gpu-vulkan`** | Same FP16 cache format as above | FP16 GEMV compute shaders (Prefill / Decode); Flash Attention (GLSL); KV in **F32**. No hipBLAS / cuBLAS-class **GEMM** yet |
| **`gpu-cuda`** | If offline cache (**`<model>.gguf.fp16`**) exists, H2D from **`.fp16bin`**; otherwise **fused row-wise GGUF dequant** → FP16 | FP16 GEMV kernels; KV in **F32** (default) |
| **`gpu-cuda`** + **`build.polarquant`** | Linear weights stay FP16 (same as above) | KV in **PolarQuant-R** (64 B/head); tile-wise F32 decode during attention |
| **`gpu-cuda-nvfp4`** | If offline cache (**`<model>.gguf.nvfp4`**, **`FP4_CACHE_VERSION=2`**) exists, H2D from **`.fp4bin`**; otherwise **fused row-wise GGUF dequant** → NVFP4. **`token_embd`** via row-wise FP16 H2D | **`fp4_qwen3_mm`** — prefill / decode both use **CUTLASS NVFP4 GEMM** (**`fp4_gemm_run_cached`**, **`M` padded to 128**). Activations clamped with **`FP4_QUANT_MAX_ABS=1024`** |
| **`gpu-cuda-nvfp4`** + **`build.polarquant`** | Same as above (NVFP4 only) | Same GEMM linear path; PolarQuant-R KV |

**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** (FP16) also spend time on first-run GGUF row dequant. Pre-generate **`<model>.gguf.fp16`** with **`make pack-cache`** (or **`--pack-fp16-cache`**) so later runs show **`Loading FP16 cache from …`**. Use **`--no-fp16-cache`** to force re-dequantization from GGUF every time. On **`gpu-rocm`**, **`make build`** (when **`MODEL`** exists) auto-packs as well.

**`gpu-cuda-nvfp4`** uses CUTLASS **NVFP4** and targets **CUDA 13 + sm_120-class GPUs** (Blackwell / RTX 50). First startup quantizes from GGUF and can take a while. Pre-generate **`<model>.gguf.nvfp4`** with **`make pack-cache`** (or **`--pack-nvfp4-cache`**) so later runs show **`Loading NVFP4 cache from …`**. Use **`--no-nvfp4-cache`** to force re-quantization from GGUF every time.

### Build and run (FP16, general GPUs)

```bash
cd qwen3-8b/gpu-cuda
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

Produces **`qwen3-gpu-cuda`**. At startup you should see **`Loading FP16 cache from …`** or **`Uploading weights (fused dequant -> FP16)...`** when the FP16 load path is active. Example with a specific architecture:

```bash
make build CUDA_GENCODE=arch=compute_89,code=sm_89
```

### Offline FP16 cache (`make pack-cache`)

Pre-pack FP16 weights to skip GGUF dequant on every run. Default output: **`<model>.gguf.fp16`** (per-tensor **`.fp16bin`** files + **`manifest`**). Same format for **`gpu-rocm`**, **`gpu-vulkan`**, and **`gpu-cuda`**.

**`gpu-cuda`**:

```bash
cd qwen3-8b/gpu-cuda
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

**`gpu-rocm`**:

```bash
cd qwen3-8b/gpu-rocm
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

**`gpu-vulkan`**:

```bash
cd qwen3-8b/gpu-vulkan
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

Direct binary (**`gpu-cuda`** / **`gpu-rocm`** / **`gpu-vulkan`** share CLI):

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-gpu-cuda ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-gpu-cuda ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

Re-run **`make pack-cache`** after updating the GGUF (**`manifest`** checks GGUF size and mtime).

### Build and run (Blackwell + NVFP4)

If CUDA 13 and CUTLASS are not set up yet (needs root-like privileges):

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make blackwell
```

If CUDA 13 is already installed:

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make cutlass          # clone third_party/cutlass (first time)
make build            # sm_120a + BONSAI_FP4=1 + FA_BR=32
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

Produces **`qwen3-gpu-cuda-nvfp4`**. At startup you should see **`Loading NVFP4 cache from …`** or **`Uploading weights (fused dequant -> NVFP4 linear layers)...`** and **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** when the FP4 path is active.

If a short prompt produces **`?,` repetition** or garbage tokens, confirm a recent build where **`make fp4-test`** passes (investigation log: **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`**, baseline commit **`433319eb`**).

### Offline NVFP4 cache (`make pack-cache`)

Pre-pack NVFP4 weights to skip GGUF dequant + quantization on every run. Default output: **`<model>.gguf.nvfp4`** (per-tensor **`.fp4bin`** files + **`manifest`**).

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

Direct binary:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-nvfp4-cache
./qwen3-gpu-cuda-nvfp4 ../model.gguf --pack-nvfp4-cache /path/to/cache
./qwen3-gpu-cuda-nvfp4 ../model.gguf --no-nvfp4-cache -p "Hello" -n 64
```

Re-run **`make pack-cache`** after updating the GGUF (**`manifest`** checks GGUF size and mtime).

**`fp4_gemm.cu`** / **`fp4_qwen3.cu`** require **C++17** (CUTLASS). **PolarQuant-R** ([arxiv:2502.02617](https://arxiv.org/abs/2502.02617)) compresses the KV cache; **`head_dim=128` required** (Qwen3-VL-8B). Roughly **~8×** less KV VRAM vs F32 (~144 MiB → ~18 MiB for 36 layers × 512 seq). NVFP4 + PolarQuant combined saves both linear FP16 (~15 GiB class for 8B) and F32 KV.

### Build (PolarQuant-R KV cache)

Linear weights remain FP16; only the KV cache is compressed with PolarQuant-R (no Blackwell required).

```bash
cd qwen3-8b/gpu-cuda
make build.polarquant
make pq-test    # encode→decode round-trip verify
```

At startup you should see **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**.

### Build and run (Blackwell + NVFP4 + PolarQuant-R)

Compress linear weights with NVFP4 and the KV cache with PolarQuant-R. Requires **CUDA 13 + CUTLASS + sm_120-class GPU**.

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make cutlass                 # first time only
make build.polarquant
make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

At startup you should see **all** of the following:

- **`Loading NVFP4 cache from …`** or **`Uploading weights (fused dequant -> NVFP4 linear layers)...`**
- **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`**
- **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**

### Run (binary directly)

FP16 build:

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what CUDA is for beginners." \
  -n 64
./qwen3-gpu-cuda ../model.gguf --pack-fp16-cache
./qwen3-gpu-cuda ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

NVFP4 build:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what CUDA is for beginners." \
  -n 64
```

Optional verification and debug (**Blackwell / sm_120a** required):

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make fp4-test      # CUTLASS path parity, batch rows, extreme activation (down_extreme)
make sfa-verify    # compute_sf_index vs CUTLASS layout
make fa-debug      # FP16 vs NVFP4, Hello + --fa-debug
./qwen3-gpu-cuda-nvfp4 ../model.gguf -p "Hello" -n 12 --fa-debug 2>&1 | grep FA_DEBUG
```

See **`doc/design.md`** (CUDA section) and **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`** for details.

### Benchmark history (`gpu-cuda/Makefile` / `gpu-cuda-nvfp4/Makefile`)

**`gpu-cuda/`** / **`gpu-cuda-nvfp4/`** / **`gpu-vulkan/`** follow the same pattern as **`gpu-rocm/`**: **`make log.push`** runs inference, then reads **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**) and appends one line to **`BENCH_LOG`** in that directory's Makefile. **`make log`** prints the table. Column 2 is **`GPU_SM`** from **`nvidia-smi`** (e.g. **`sm_120`**) for CUDA builds; **`gpu-vulkan`** uses the **GPU name from `vulkaninfo`**.

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPT` | long English prompt (in Makefile) | benchmark prompt |
| `BENCH_N` | `128` | max generated tokens (`-n`) |
| `BENCH_SEED` | `42` | RNG seed (`-s`); **ignored when `-t 0`** |
| `BENCH_TEMP` | `0` | temperature (`-t`); **passed on the `log.push` command line** |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value metrics file |

```bash
cd qwen3-8b/gpu-cuda
make log.push
make log
make log.push BENCH_N=64 BENCH_TEMP=0.8 BENCH_SEED=77

cd qwen3-8b/gpu-cuda-nvfp4
make log.push
make log
```

**Note:** **`make log.push` rewrites the `Makefile` in each GPU variant** (`gpu-rocm` / `gpu-vulkan` / `gpu-cuda` / `gpu-cuda-nvfp4`). Check **`git diff`** before committing. Table **`total_tps`** is **inference only** (VRAM weight upload excluded).

**`BENCH_LOG_FILE` VRAM fields** (after inference): **`vram_total`**, **`vram_device_used`** / **`vram_device_total`** (from **`cudaMemGetInfo`** / **`hipMemGetInfo`**, bytes and **`_mib`**), **`[vram_breakdown]`** section.

- **`gpu-rocm` (FP16)**: FP16 embedding / F32 norm / FP16 linear weights (**`vram_weights_linear`**) / KV / decode activations / prefill batch (includes **`d_scratch_f16`**)
- **`gpu-vulkan` (FP16)**: FP16 embedding / F32 norm / FP16 linear weights (in **`vram_total`** only; no separate key) / KV / decode activations / prefill batch
- **`gpu-cuda` (FP16)**: FP16 embedding / F32 norm / FP16 linear weights (**`vram_weights_linear`**) / KV / decode activations / prefill batch
- **`gpu-cuda-nvfp4` (NVFP4)**: linear weights in **`vram_weights_fp4`** plus **`vram_fp4_gemm_scratch`** (BF16 activations/output + CUTLASS workspace, etc.)

Makefile **`BENCH_LOG`** history records tok/s only; see **`BENCH_LOG_FILE`** for VRAM breakdown.

One line per entry (pipe-separated): **`timestamp|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

- **`gpu-vulkan`**: column 2 from **`vulkaninfo`** (GPU name)
- **`gpu-cuda`**: column 2 from **`nvidia-smi`** (e.g. **`sm_120`**)
- **`gpu-cuda-nvfp4`**: column 2 defaults to **`sm_120a`**

**`gpu-vulkan`** long-prompt measured table and VRAM breakdown: see **Vulkan compute implementation → Vulkan GPU long prompt** above (**2026-05-29 09:39**: prefill **2.97** / decode **2.17** / total **2.86** tok/s).

Example NVFP4 entries (132 prompt tokens, RTX 5090): **622.60** / **66.71** / **122.03** tok/s (**`2026-05-24`**, **`gen=128`**); **4623.10** / **66.64** / **648.35** (**`2026-05-29`**, **`gen=13`** early EOS). VRAM breakdown is in **`BENCH_LOG_FILE`** (`[vram_breakdown]`), not in **`BENCH_LOG`**.

### Source reading

8. `qwen3-8b/gpu-cuda/fp16_cache_io.c` / `main.c` / `kernels.cu` / `polarquant.cu` / `fa_debug.c` — CUDA FP16 (**`--fa-debug`**, VRAM bench log).  
9. `qwen3-8b/gpu-cuda-nvfp4/fp4_cache_io.c` / `fp4_qwen3.cu` / `fp4_gemm.cu` / `fp4_verify.cu` — Blackwell NVFP4 (**`FP4_QUANT_MAX_ABS`**, shared **`kernels.cu`**).  
10. `qwen3-8b/gpu-cuda-nvfp4/DEBUG.md` — NVFP4 anomaly investigation log (baseline **`433319eb31c3c992536afb5c9a3717084ea5d137`**).


### Troubleshooting (GPU / XDNA appendix)

### `nvcc` not found / `nvlink` errors

Ensure CUDA `bin` is on `PATH` or set `CUDA_HOME` in each directory’s `Makefile`.

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

If a PTX-only build is very slow, rebuild with `CUDA_GENCODE=arch=compute_XX,code=sm_XX` for your GPU.

### FP16 first startup is slow / cache miss (`gpu-rocm` / `gpu-vulkan` / `gpu-cuda`)

First run dequantizes from GGUF row-by-row. Pre-generate **`<model>.gguf.fp16`** with **`cd gpu-rocm && make pack-cache`**, **`cd gpu-vulkan && make pack-cache`**, or **`cd gpu-cuda && make pack-cache`** (**`gpu-rocm`**: **`make build`** auto-packs when **`MODEL`** exists). If you see **`Warning: FP16 cache miss for …`**, a **`.fp16bin`** is missing, shapes mismatch, or **`manifest`** is invalid — re-run **`make pack-cache`** or use **`--no-fp16-cache`** to force re-dequantization.

### NVFP4 first startup is slow / cache miss

First run quantizes from GGUF row-by-row. Pre-generate **`<model>.gguf.nvfp4`** with **`make pack-cache`**. If you see **`Warning: NVFP4 cache miss for …`**, a **`.fp4bin`** is missing, shapes mismatch, or **`manifest`** is invalid — re-run **`make pack-cache`** or use **`--no-nvfp4-cache`** to force re-quantization.

### NVFP4 build fails / `NVFP4 quantize failed`

In **`gpu-cuda-nvfp4`**, confirm **`sm_120a`** build, CUDA 13, and **`make cutlass`**. On general GPUs use **`gpu-cuda`** with **`make build`** (FP16). If apt **CUDA 11** and **CUDA 13** are mixed, run **`make blackwell`** or remove 11.x manually.

### `make fp4-test` fails / `Arch conditional MMA instruction... Aborting`

You may be using a stale **`fp4_gemm.o`** built for **`compute_86` PTX** only. In **`gpu-cuda-nvfp4`**, run **`make clean`** → **`make fp4-test`** again (**`fp4_gemm.sm120a.o`** is fixed to **`sm_120a` + C++17**). Requires a **Blackwell GPU**.

### NVFP4 garbage output after short prefill (`?,` repetition)

Extreme activations at layer 6 **`down`** can saturate NVFP4 activation scales → GEMM **NaN** → corrupted KV (Flash Attention itself is fine). Use a build with **`FP4_QUANT_MAX_ABS=1024`** where **`make fp4-test`** passes (**`down_extreme`** included). See **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`**.

### NVFP4 repetition / “LLM” spam

May be an old **GEMM prefill vs GEMV decode** mismatch or the NaN path above. Re-run **`make pack-cache`** with **`FP4_CACHE_VERSION=2`** and confirm **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** at startup.

### NVFP4 `make log.push` stops early or repeats every run

Long bench prompts often hit **EOS early** (e.g. **`gen_tokens=13`**). **`-t 0`** ignores seed. Try **`make log.push BENCH_TEMP=0.8 BENCH_SEED=77`**. Ensure **`BENCH_TEMP`** is on the **`log.push`** recipe in **`Makefile`**.

### `gpu-cuda-nvfp4` PolarQuant build is slow / `Killed` (OOM)

Compiling **`kernels.cu`** with **`sm_120a` + FP4 + PolarQuant** uses a lot of RAM. You may see **`Error 137`**. Try adding swap, disabling parallel builds, or running **`make cutlass`** then **`make build.polarquant`** again.

### PolarQuant not active

If **`PolarQuant-R: KV cache enabled`** is missing at startup, confirm you built with **`build.polarquant`** (`gpu-cuda` or `gpu-cuda-nvfp4`). PolarQuant is disabled when **`head_dim ≠ 128`**.

### `gpu-rocm` build fails (C++ headers not found)

**g++ / libstdc++-dev** may be missing.

```bash
sudo apt install -y g++ libstdc++-dev
cd qwen3-8b/gpu-rocm && make clean build
```

### `hipcc` not found

Check ROCm location:

```bash
ls /opt/rocm/bin/hipcc
```

If elsewhere:

```bash
cd qwen3-8b/gpu-rocm
make build ROCM=/path/to/rocm
```

### `rocBLAS error: … for GPU arch : gfx1152` (Ryzen AI 5 340, etc.)

rocBLAS has **no `gfx1152` Tensile bundle**. **A ROCm minor upgrade alone** may not fix this.

```bash
cd qwen3-8b/gpu-rocm
make clean build && make run
```

Confirm build log shows **`Build offload arch: gfx1151`** and **`HSA_OVERRIDE_GFX_VERSION: 11.5.1`**. See **“Ryzen AI / gfx1152 and rocBLAS”** above and [`doc/design.md`](doc/design.md).

### `hipBLAS error: 6` (Prefill at 0%)

**`HIPBLAS_STATUS_INTERNAL_ERROR`**. Often caused by **symlinking `gfx1151` rocBLAS files to `gfx1152` names only**, or **HSA override without matching `--offload-arch`**. Use **`make clean build`** then **`make run`** (Makefile auto settings).

### Segmentation fault (right after Prefill starts; log shows `gcnArchName: gfx1151`)

Common when **`HSA_OVERRIDE_GFX_VERSION=11.5.1`** is set but the binary was built with **`gfx1152` offload**. **`make clean build`** — verify **`Build offload arch: gfx1151`** — then **`make run`**.

### Wrong `GPU_ARCH` / detection failed

**`GPU_ARCH_DETECTED`** should match hardware. Normally auto-detected from `rocminfo`. On **`gfx1152`** hardware the Makefile maps to **`gfx1151` offload** internally. Override manually if needed:

```bash
rocminfo | awk '/^  Name:/ { n=$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }'
cd qwen3-8b/gpu-rocm
make build GPU_ARCH=gfx1100
make detect-gpu-arch
```

### ROCm Prefill is slow (~30 tok/s)

Check startup log for **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`**. If missing or prefill is very slow, **`make -C gpu-rocm clean build`** and confirm **`-lhipblas -lrocblas`** are linked. See **“Prefill acceleration (overview)”** above and **`doc/design.md`**, section **“ROCm Prefill acceleration (3 stages)”**.

### **`undefined reference to hipblas*`** / hipBLAS link failure

Verify **`libhipblas.so`** / **`librocblas.so`** under **`$(ROCM)/lib`**. Reinstall ROCm or fix **`ROCM=`** path if needed.

### **`make wmma` fails**

Possible causes: **WMMA instructions in `qwen3-rocm` binary** (unexpected), **hipBLAS path not reported**, **`wmma-probe` calibration failure**. Try **`make wmma WMMA_SKIP_RUN=1`** for static checks only. Set **`LLVM_OBJDUMP=$(ROCM)/llvm/bin/llvm-objdump`**. Use **`WMMA_SKIP_RUN=1`** when MODEL is not available.

### `/dev/accel/accel0` opens but `CREATE_HWCTX` returns `EINVAL`

The driver may reject column/tile settings. Try `XDNA_NUM_COL=1` and check `dmesg` for `amdxdna` (see `doc/design.md`).


---

## AMD Ryzen AI XDNA2 NPU implementation (`xdna2` / `xdna2-bfp16`)

**`qwen3-8b/xdna2/`** and **`qwen3-8b/xdna2-bfp16/`** are also **appendix** builds. They talk to the in-kernel **`amdxdna`** module directly (no XRT userland), but require **Ryzen AI hardware and GEMV control-code binaries**. First-time readers can **ignore them**.

For an **overview of AMD XDNA** (design goals, tile-level architecture, generational changes, dtypes and accuracy, software stack, comparison with other NPUs, etc.), see [thamada/xdna-overview](https://github.com/thamada/xdna-overview) (`main.md` plus a PDF).

What follows is a technical note for XDNA2 NPU builds.

Uses the XDNA2 NPU on AMD Ryzen AI APUs (e.g. Phoenix / Hawk Point / Strix Point).

### Prerequisites

1. Linux kernel **6.10+** with `drivers/accel/amdxdna` enabled. Check: `lsmod | grep amdxdna`.
2. `/dev/accel/accel0` exists and your user is in the `render` group.

```bash
ls -l /dev/accel/accel0
sudo usermod -aG render "$USER"   # re-login to apply
```

3. `<drm/drm.h>` UAPI headers installed (often via `linux-libc-dev`).

### Build

```bash
cd qwen3-8b/xdna2
make build
```

Produces **`qwen3-xdna2`**.

### Run

For fast BF16 GEMV on the NPU you need **MLIR-AIE / IRON**-generated control microcode bundles, named like `bf16-gemv-<n>x<d>.bin`, under `XDNA_GEMV_DIR`. If missing, the code falls back to OpenMP BF16 GEMV on CPU (**bit-identical** with the NPU path).

The repo ships **`xdna2/xdna-gemv/kernels/`** (paths relative to **`qwen3-8b/`**) with **64-byte placeholders** (magic `GQF3XDNA`). They are **not** executed on the device (`--xdna-status` shows `[STUB]`). Regenerate with `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels` from the repo root. Replace with real MLIR-AIE outputs for hardware GEMV.

Useful env vars: `XDNA_GEMV_DIR` (search path for control blobs), `XDNA_FORCE_CPU=1` (force CPU), `XDNA_NUM_COL` (column count; try `XDNA_NUM_COL=1` if `CREATE_HWCTX` returns `EINVAL`).

```bash
cd qwen3-8b/xdna2
# Force CPU fallback
XDNA_FORCE_CPU=1 ./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# Repo stub placeholders (relative to xdna2/): not real NPU ctrlcode — `--xdna-status` shows [STUB]
XDNA_GEMV_DIR=xdna-gemv/kernels ./qwen3-xdna2 \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status

# With real MLIR-AIE blobs under XDNA_GEMV_DIR: NPU path
XDNA_GEMV_DIR=xdna-gemv/kernels ./qwen3-xdna2 \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

Or use the subdirectory Makefile:

```bash
cd qwen3-8b/xdna2
make run PROMPT="Short explanation in English."
```

### XDNA2 + BFPX host weights (`xdna2-bfp16/qwen3-xdna2-bfpx`)

`xdna2-bfp16/main.c` shares the **same DRM ioctl and chunked BF16 GEMV** as `xdna2/main.c`, but converts linear weights at load time to **BFPX (per-block BF16 scale + int8)** on the host and releases the GGUF mmap afterward. CPU fallback uses **`mm_bfpx`** (float activations × BFPX weights) and is **not numerically aligned** with `xdna2/qwen3-xdna2`. Block approximation means **behavior differs** from **`xdna2/qwen3-xdna2`**, which decodes quantized mmap weights into BF16 **on each GEMV**; neither quality nor speed dominates in all cases.

```bash
cd qwen3-8b/xdna2-bfp16
make build
XDNA_FORCE_CPU=1 ./qwen3-xdna2-bfpx ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=../xdna2/xdna-gemv/kernels ./qwen3-xdna2-bfpx \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
cd qwen3-8b/xdna2-bfp16
make run PROMPT="Short explanation in English."
```

### Notes

- **`xdna2/qwen3-xdna2`**: Linear weights stay **mmap'd** (**CPU OpenMP** build–like). One **BF16 scratch** sized for the **largest text-path GEMV** (often **LM head / embedding scale**) may still require substantial **DRAM**. There is **no** persistent duplicate BF16 copy of **all** layers. Insufficient RAM can still kill the process or fail mmap/allocs.
- **`xdna2-bfp16/qwen3-xdna2-bfpx`**: Inference residency is often dominated by **BFPX + norm buffers** with mmap released early, but **conversion** can **spike memory** (GGUF mmap plus temporary full-tensor staging).
- On NPU runs you reserve AIE columns; other NPU workloads (e.g. Windows Studio Effects) may contend.

### Source reading

9. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c` — `amdxdna` ioctl, CPU fallback, BFPX weight layout.


