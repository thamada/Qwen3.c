# Qwen3.c

This repository is an **inference implementation** that runs **Qwen3-family models** directly from **a single C source**, **without relying on external libraries**.

**This project does not link userland ML libraries or runtimes such as PyTorch, TensorFlow, JAX, or ONNX Runtime.** Inference is built around **standard C and `libm`**, with one or a few sources under `qwen3-8b/`. The AMD GPU build uses **ROCm/HIP** (`hipcc`), the NVIDIA GPU build uses the **CUDA Toolkit** (`nvcc`), CPU parallelism uses **OpenMP** (`cpu-multicore`) or **OpenMP + OpenBLAS** (`cpu-blas`), and the XDNA2 NPU build talks to the **Linux kernel `amdxdna` DRM ioctl (UAPI)** directly—there is no dependency on a Python runtime or `torch`.

ROCm/HIP and CUDA are **GPU compilers and runtimes**, not high-level neural network frameworks (this repo builds the Transformer from custom HIP / CUDA kernels and host code).

### Why avoid ML libraries?

Typical LLM inference using PyTorch or similar stacks is short to write and fast to run. In that setup, though, low-level details—**execution order, memory layout, alignment, quantization packing**—often hide inside the framework or runtime.

This repository deliberately skips that layer and **makes the full path visible in C**: reading GGUF, restoring weights, linear algebra, Transformer forward, and sampling. The goal is not to replace existing frameworks but to **inspect, validate, and change** the inference path when needed.

That choice helps with:

- **Understandability**: You can follow what is read from the model file, which buffers hold it, and in what order computation runs, straight from the sources and `doc/design.md`.
- **Simpler dependencies**: You do not need a Python stack or a large ML tree—just a C toolchain and a minimal environment to exercise the path.
- **Experimentation**: Quantization formats, memory layouts (e.g. BFPX), splitting work across CPU/GPU/NPU, and direct access to `/dev/accel` are easier to try without framework abstractions in the way.
- **Reference value**: It shows how Qwen3-style decoder inference can work with minimal scaffolding and can serve as a baseline for comparison or validation.

So this is **not** aimed at maximum performance or full feature parity. The focus is **not** treating LLM inference as a black box, but letting developers see and modify the implementation.

---

A small inference implementation that runs Qwen3-family GGUF models from **straightforward C sources** under `qwen3-8b/`. Paths include **CPU**, **OpenMP**, **OpenMP + OpenBLAS**, **ROCm/HIP on AMD GPUs**, **CUDA on NVIDIA GPUs**, and **AMD Ryzen AI XDNA2 NPU** (direct **`amdxdna` DRM ioctl** usage).

The scope is the **text decoder of Qwen3-VL-8B-Instruct**. Image input and the vision encoder are **out of scope**; use cases are prompt-in, text-out generation.

日本語版は [README.md](README.md) を参照してください。

## What you can run

Build the C sources under `qwen3-8b/` and try the following targets:

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | Learning the flow, minimal setup. **Prefill progress bar** and throughput summary on stderr |
| CPU OpenMP | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | Faster CPU trials |
| CPU OpenMP + OpenBLAS | `qwen3-8b/cpu-blas/main.c` | `cpu-blas/qwen3-cpu-blas` | BLAS for F32 GEMV and attention; quantized GEMV uses **Q8_K activations + AVX2 integer dots for all types** (layer-shared Q8). **RoPE cache**, prefill **LM head skip**, greedy **`mm_argmax_row`**. **F16 embedding via F16C**. **Prefill progress bar** on stderr |
| ROCm/HIP GPU | `qwen3-8b/gpu-rocm/main.c` | `gpu-rocm/qwen3-rocm` | AMD GPU. **Prefill**: one batched forward + **hipBLAS GemmEx** ([llama.cpp](https://github.com/ggml-org/llama.cpp/) cublas-style path). **Decode**: one-token GEMV. **Prefill progress bar** and prefill / decode / total throughput summaries. Benchmark history via **`make log` / `make log.push`**. Verify hipBLAS path / no embedded WMMA with **`make wmma`** |
| CUDA GPU (FP16) | `qwen3-8b/gpu-cuda/main.c` + `kernels.cu` | `gpu-cuda/qwen3-gpu-cuda` | NVIDIA GPUs; prefill batch + Flash Attention. All linear layers in **FP16 VRAM**. Optional **`build.polarquant`**: **PolarQuant-R** KV (64 B/head). **Prefill progress bar** and throughput summaries. Benchmark history via **`make log` / `make log.push`**. Not in aggregate `Makefile` |
| CUDA GPU (NVFP4) | `qwen3-8b/gpu-cuda-nvfp4/` + shared `gpu-cuda/` | `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4` | Blackwell (e.g. RTX 50). Linear weights **NVFP4 only** at H2D (CUTLASS). Embedding only in FP16 VRAM. Optional **`build.polarquant`**: NVFP4 + PolarQuant-R combined (max VRAM savings). Benchmark history via **`make log` / `make log.push`**. Not in aggregate `Makefile` |
| AMD Ryzen AI XDNA2 NPU (mmap + per-GEMV BF16 scratch) | `qwen3-8b/xdna2/main.c` | `xdna2/qwen3-xdna2` | NPU via direct `amdxdna` ioctl; weights **mmap'd** like **CPU OpenMP** build; single BF16 scratch BO filled **per GEMV** |
| AMD Ryzen AI XDNA2 NPU (BFPX host weights) | `qwen3-8b/xdna2-bfp16/main.c` | `xdna2-bfp16/qwen3-xdna2-bfpx` | Same ioctl/GEMV path; linear weights held on host as block FP (BF16 scale + int8); GGUF mmap released after conversion |

For an **overview of AMD XDNA** (design goals, tile-level architecture, generational changes, dtypes and accuracy, software stack, comparison with other NPUs, etc.), see the companion write-up: [thamada/xdna-overview](https://github.com/thamada/xdna-overview) (`main.md` plus a PDF).

An 8B model on CPU is **very slow**. CPU is fine for a first smoke test; for usable token throughput, use **ROCm/HIP** (AMD GPU), **CUDA** (NVIDIA GPU), or the XDNA2 NPU builds. **`xdna2/qwen3-xdna2`** avoids **persistent BF16 copies of every layer**—it keeps quantized weights **mmap'd** (**CPU OpenMP** build–style residency) and decodes **one GEMV matrix at a time** into a BF16 scratch for the NPU, so latency per matmul rises. For tighter residency or another host layout, **`xdna2-bfp16/qwen3-xdna2-bfpx`** converts to **BFPX** and drops the GGUF mmap after conversion (**large load-time peak possible**); output is **not** bit-aligned with **`xdna2/qwen3-xdna2`**.

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
    ├── gpu-rocm/
    │   ├── Makefile
    │   ├── main.c
    │   ├── wmma_probe.c          (`make wmma-probe` — WMMA detector calibration)
    │   └── scripts/
    │       └── check_wmma.sh     (`make wmma`)
    ├── gpu-cuda/                    (FP16 linear layers, no NVFP4)
    │   ├── Makefile
    │   ├── main.c
    │   ├── kernels.cu
    │   ├── gpu.h
    │   └── polarquant.cu / polarquant_kernels.cuh / polarquant_verify.cu  (PolarQuant-R KV)
    ├── gpu-cuda-nvfp4/              (Blackwell NVFP4)
    │   ├── Makefile
    │   ├── fp4_gemm.cu / fp4_qwen3.cu / fp4_verify.cu
    │   └── third_party/cutlass/  (fetched via make cutlass)
    │   (main.c / kernels.cu / gpu.h / polarquant.* reference ../gpu-cuda/)
    ├── xdna2/
    │   ├── Makefile
    │   ├── main.c
    │   └── xdna-gemv/
    │       ├── README.md
    │       ├── gen-xdna-gemv-stubs.py
    │       ├── kernels/
    │       └── toolchain/
    ├── xdna2-bfp16/
    │   ├── Makefile
    │   └── main.c
    └── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

You normally work inside `qwen3-8b/` for builds and runs.

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

### ROCm/HIP build

You need an AMD GPU and ROCm. The `Makefile` assumes ROCm under `/opt/rocm` by default. **`GPU_ARCH`** (for `hipcc --offload-arch`) is **auto-detected from `rocminfo`**.

Check:

```bash
/opt/rocm/bin/hipcc --version
make -C gpu-rocm detect-gpu-arch   # e.g. Detected GPU arch: gfx1100
```

If `rocminfo` does not report a GPU, pass **`GPU_ARCH=gfx1100`** (or your ISA) manually at build time.

### CUDA build

You need an NVIDIA GPU and the **CUDA Toolkit** (`nvcc`, `libcudart`). There is **no** CUDA target in the aggregate `qwen3-8b/Makefile`; build in one of these directories:

- **`qwen3-8b/gpu-cuda/`** — FP16 linear layers (general NVIDIA GPUs)
- **`qwen3-8b/gpu-cuda-nvfp4/`** — NVFP4 linear layers (Blackwell / RTX 50, **CUDA 13** + CUTLASS). First run **`make cutlass`** to fetch **`third_party/cutlass`**. **`fp4_*` objects are built with C++17** (CUTLASS requirement). Do **not** mix apt **`nvidia-cuda-toolkit` (CUDA 11)** with CUDA 13 (**`make blackwell`** removes 11.x and installs 13).

Check:

```bash
nvcc --version
nvidia-smi
```

Put CUDA’s **`bin`** directory on **`PATH`** (linking can fail if only `/usr/local/bin/nvcc` is visible).

| Use case | Command |
|----------|---------|
| **FP16 only** (Ampere/Ada, PTX OK) | `cd qwen3-8b/gpu-cuda` → `make build` / `make run` |
| **PolarQuant-R KV cache** (FP16 linear, any GPU) | `cd qwen3-8b/gpu-cuda` → `make build.polarquant` / `make run.polarquant` |
| **Blackwell NVFP4** (e.g. RTX 50) | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build` / `make run` |
| **NVFP4 + PolarQuant combined** (Blackwell, max VRAM savings) | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build.polarquant` / `make run.polarquant` |
| Install CUDA 13 + full NVFP4 build | `cd qwen3-8b/gpu-cuda-nvfp4` → `make blackwell` |
| PolarQuant round-trip verify | `make pq-test` in either directory |
| CUTLASS NVFP4 GEMM unit verify | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fp4-test` (**Blackwell / sm_120a required**) |
| Benchmark history (FP16) | `cd qwen3-8b/gpu-cuda` → `make log.push` / `make log` |
| Benchmark history (NVFP4) | `cd qwen3-8b/gpu-cuda-nvfp4` → `make log.push` / `make log` |

FP16 builds (`gpu-cuda`) default to PTX (`compute_86`). For native SASS, set `CUDA_GENCODE=arch=compute_XX,code=sm_XX`. NVFP4 builds (`gpu-cuda-nvfp4`) default to **`sm_120a`**. **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** are compiled with **`BLACKWELL_NVCCFLAGS`** (**`-std=c++17`** + fixed **`sm_120a`**). CUTLASS **`v4.5.0`** is fetched via **`make cutlass`** into **`third_party/cutlass`**; CUDA 13 deprecated vector-type warnings are resolved upstream in CUTLASS.

## Obtain the model file

The default name in `qwen3-8b/Makefile` is:

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

The GGUF is **not** shipped in the repo (license + size). Place it under `qwen3-8b/`. Recommended: aggregate **`make model`** (`wget` + bundled `.sha256sum` verification).

```bash
cd qwen3-8b
make model
```

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
make model          # if missing: download and verify GGUF
make build.cpu
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

On success, text should appear gradually after load.

## CPU single-thread

### Build

```bash
cd qwen3-8b
make build.cpu
```

Produces **`cpu/qwen3-cpu`** (or `make build` inside `cpu/`).

```bash
ls -lh cpu/qwen3-cpu
```

### Run

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries.

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Give a one-sentence introduction of yourself." \
  -n 16
```

Using aggregate `run.cpu`:

```bash
make run.cpu PROMPT="Give a one-sentence introduction of yourself."
```

Model elsewhere:

```bash
make run.cpu MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP

Uses multiple CPU cores; same model file as single-thread.

### Build

```bash
cd qwen3-8b
make build.cpu-multicore
```

Produces **`cpu-multicore/qwen3-cpu-omp`**.

### Run

```bash
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain in bullet points what quantization is." \
  -n 32
```

`OMP_NUM_THREADS` sets thread count; try 4 or 8 first.

```bash
OMP_NUM_THREADS=4 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

Speedup depends on core count, memory bandwidth, and quantization.

## CPU OpenMP + OpenBLAS

Same decoder and GGUF as **`cpu-multicore`**, but **F32 matmul** (`cblas_sgemv`) and **attention K/V combine** go through OpenBLAS. For IQ2_S / IQ3_S / Q4_K / Q5_K quantized GEMV, activations are quantized to **Q8_K** (`quantize_row_q8_K`), then **`vec_dot_*_q8_K`** integer dot products (aligned with **ggml-cpu/quants.c**) are used—without full float[256] row dequantization like **`cpu-multicore`**. **GEMVs that share the same activation vector within a layer reuse one Q8 quantization** (layer-shared Q8). With **`__AVX2__`**, IQ2_S / IQ3_S / Q4_K / Q5_K dots and Q8 quantization are SIMD-accelerated (**`-march=native`** by default).

Additional CPU optimizations: **RoPE cos/sin cache** (precomputed at startup), **LM head skip** during prefill (except the last prompt token; **`FWD_NO_LM`**), **`mm_argmax_row`** for greedy sampling (**`-t 0`**, no full vocab logits buffer), and **F16 embedding** lookup via **F16C+AVX2**. During prefill, stderr shows a **Prefill progress bar** (`Prefill [====...]`, width 40) plus prefill / decode / total throughput summaries.

### Build

```bash
cd qwen3-8b
make build.cpu-blas
```

Produces **`cpu-blas/qwen3-cpu-blas`** (or `make build` inside `cpu-blas/`).

### Run

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries.

```bash
OMP_NUM_THREADS=8 ./cpu-blas/qwen3-cpu-blas Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Hello, how are you?" \
  -n 32
```

Via aggregate Makefile:

```bash
make run.cpu-blas PROMPT="Hello, how are you?"
```

## ROCm/HIP GPU

The primary path when ROCm and an AMD GPU are available.

### `GPU_ARCH` (auto-detect)

`gpu-rocm/Makefile` **auto-detects** the first GPU agent name (`gfx*`) from **`$(ROCM)/bin/rocminfo`** as **`GPU_ARCH`** before building. On success you will see:

```text
===============================================
  Detected GPU arch: gfx1100
===============================================
```

To override manually:

```bash
make build GPU_ARCH=gfx1100          # inside gpu-rocm/
make build.gpu-rocm GPU_ARCH=gfx1100 # from qwen3-8b/
```

### Build

```bash
cd qwen3-8b
make build.gpu-rocm
```

If ROCm is not under `/opt/rocm`:

```bash
make build.gpu-rocm ROCM=/path/to/rocm
```

Produces **`gpu-rocm/qwen3-rocm`**. Links **`-lhipblas -lrocblas`** (Prefill linear GEMM).

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
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what ROCm is for beginners." \
  -n 64
```

Using `run.gpu-rocm`:

```bash
make run.gpu-rocm PROMPT="Short explanation in English."
```

During prefill, stderr shows a **Prefill progress bar** plus prefill / decode / total throughput summaries (same format as **`cpu-blas`**). On exit, stdout also prints **`prefill_tps:` / `decode_tps:` / `total_tps:`** for **`make log.push`** to parse (**inference only**; model weight H2D is excluded). **`gpu-cuda/`** and **`gpu-cuda-nvfp4/`** use the same format via shared **`main.c`**.

### Benchmark history (`gpu-rocm/Makefile`)

From **`gpu-rocm/`**, **`make log.push`** runs a benchmark with the default long prompt (~128 tokens), **`-n 128`**, and **`-t 0`**, then appends the result to **`BENCH_LOG`** in the Makefile. **`make log`** prints the history as a table.

```bash
cd qwen3-8b/gpu-rocm
make log.push                    # default BENCH_N=128, BENCH_SEED=42
make log                         # show history
make log.push BENCH_N=64         # override generation length, etc.
```

One line per entry (pipe-separated): **`timestamp|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

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

## CUDA GPU (NVIDIA)

For NVIDIA GPUs with CUDA. There is **no** `build.gpu-cuda` in the aggregate `qwen3-8b/Makefile`; build under **`gpu-cuda/`** (FP16) or **`gpu-cuda-nvfp4/`** (NVFP4) depending on your GPU. Prompts run as a **prefill batch**; generation is **one-token decode**; attention uses **Flash Attention** (GQA).

| Directory | Weights at load | Linear / KV at runtime |
|-----------|-----------------|------------------------|
| **`gpu-cuda`** | CPU dequant → **FP16** → VRAM (ROCm-like) | FP16 GEMV kernels; KV in **F32** (default) |
| **`gpu-cuda`** + **`build.polarquant`** | Linear weights stay FP16 (same as above) | KV in **PolarQuant-R** (64 B/head); tile-wise F32 decode during attention |
| **`gpu-cuda-nvfp4`** | H2D: linear tensors → **NVFP4 cache only**; **`token_embd`** only in FP16 | **`fp4_qwen3_mm`** — decode / short prefill via **FP4 GEMV**; long prefill via CUTLASS GEMM |
| **`gpu-cuda-nvfp4`** + **`build.polarquant`** | Same as above (NVFP4 only) | FP4 GEMV/GEMM for linear; PolarQuant-R KV |

**`gpu-cuda-nvfp4`** uses CUTLASS **NVFP4** and targets **CUDA 13 + sm_120-class GPUs** (Blackwell / RTX 50). **`fp4_gemm.cu`** / **`fp4_qwen3.cu`** require **C++17** (CUTLASS). **PolarQuant-R** ([arxiv:2502.02617](https://arxiv.org/abs/2502.02617)) compresses the KV cache; **`head_dim=128` required** (Qwen3-VL-8B). Roughly **~8×** less KV VRAM vs F32 (~144 MiB → ~18 MiB for 36 layers × 512 seq). NVFP4 + PolarQuant combined saves both linear FP16 (~15 GiB class for 8B) and F32 KV.

### Build and run (FP16, general GPUs)

```bash
cd qwen3-8b/gpu-cuda
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

Produces **`qwen3-gpu-cuda`**. Example with a specific architecture:

```bash
make build CUDA_GENCODE=arch=compute_89,code=sm_89
```

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

Produces **`qwen3-gpu-cuda-nvfp4`**. At startup you should see **`Uploading weights to device (dequant -> NVFP4 linear layers)...`** and **`GPU: FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`** when the FP4 path is active.

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

- **`Uploading weights to device (dequant -> NVFP4 linear layers)...`**
- **`GPU: FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`**
- **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**

### Run (binary directly)

FP16 build:

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what CUDA is for beginners." \
  -n 64
```

NVFP4 build:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what CUDA is for beginners." \
  -n 64
```

Optional CUTLASS NVFP4 GEMM smoke test: `cd qwen3-8b/gpu-cuda-nvfp4 && make fp4-test` (builds and runs **`fp4_verify.cu`**, **Blackwell / sm_120a**, e.g. RTX 50. **`fp4_gemm.sm120a.o`** uses **C++17 + fixed `sm_120a`**). See `doc/design.md` (CUDA section).

### Benchmark history (`gpu-cuda/Makefile` / `gpu-cuda-nvfp4/Makefile`)

Same format as **`gpu-rocm/`**. From **`gpu-cuda/`** or **`gpu-cuda-nvfp4/`**, **`make log.push`** runs a benchmark with the default long prompt (~128 tokens), **`-n 128`**, and **`-t 0`**, then appends the result to **`BENCH_LOG`** in that directory's Makefile. **`make log`** prints the history as a table.

```bash
cd qwen3-8b/gpu-cuda
make log.push                    # default BENCH_N=128, BENCH_SEED=42
make log                         # show history
make log.push BENCH_N=64         # override generation length, etc.

cd qwen3-8b/gpu-cuda-nvfp4
make log.push
make log
```

One line per entry (pipe-separated): **`timestamp|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

- **`gpu-cuda`**: column 2 **`GPU_SM`** is taken from **`CUDA_GENCODE`** **`code=`** (default **`compute_86`**). Override example: **`make log.push CUDA_GENCODE=arch=compute_90,code=sm_90`**
- **`gpu-cuda-nvfp4`**: column 2 defaults to **`sm_120a`**

Example **`BENCH_LOG`** entry (NVFP4, 132 prompt tokens, RTX 5090 / Blackwell): prefill **622.60** / decode **66.71** / total **122.03** tok/s (**`2026-05-24T15:03:39|sm_120a|…`**).

## AMD Ryzen AI XDNA2 NPU

Uses the XDNA2 NPU on AMD Ryzen AI APUs (e.g. Phoenix / Hawk Point / Strix Point). Implementation talks to the in-tree **`amdxdna` kernel module**; no extra userland like XRT is required.

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
cd qwen3-8b
make build.xdna2
```

Produces **`xdna2/qwen3-xdna2`**.

### Run

For fast BF16 GEMV on the NPU you need **MLIR-AIE / IRON**-generated control microcode bundles, named like `bf16-gemv-<n>x<d>.bin`, under `XDNA_GEMV_DIR`. If missing, the code falls back to OpenMP BF16 GEMV on CPU (**bit-identical** with the NPU path).

The repo ships **`xdna2/xdna-gemv/kernels/`** (paths relative to **`qwen3-8b/`**) with **64-byte placeholders** (magic `GQF3XDNA`). They are **not** executed on the device (`--xdna-status` shows `[STUB]`). Regenerate with `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels` from the repo root, or `make gen-xdna-kernels` from `qwen3-8b/`. Replace with real MLIR-AIE outputs for hardware GEMV.

Useful env vars: `XDNA_GEMV_DIR` (search path for control blobs), `XDNA_FORCE_CPU=1` (force CPU), `XDNA_NUM_COL` (column count; try `XDNA_NUM_COL=1` if `CREATE_HWCTX` returns `EINVAL`).

```bash
# Force CPU fallback
XDNA_FORCE_CPU=1 ./xdna2/qwen3-xdna2 Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# Repo stub placeholders (from qwen3-8b/): not real NPU ctrlcode — `--xdna-status` shows [STUB]
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status

# With real MLIR-AIE blobs under XDNA_GEMV_DIR: NPU path
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

Or:

```bash
make run.xdna2 PROMPT="Short explanation in English."
```

### XDNA2 + BFPX host weights (`xdna2-bfp16/qwen3-xdna2-bfpx`)

`xdna2-bfp16/main.c` shares the **same DRM ioctl and chunked BF16 GEMV** as `xdna2/main.c`, but converts linear weights at load time to **BFPX (per-block BF16 scale + int8)** on the host and releases the GGUF mmap afterward. CPU fallback uses **`mm_bfpx`** (float activations × BFPX weights) and is **not numerically aligned** with `xdna2/qwen3-xdna2`. Block approximation means **behavior differs** from **`xdna2/qwen3-xdna2`**, which decodes quantized mmap weights into BF16 **on each GEMV**; neither quality nor speed dominates in all cases.

```bash
cd qwen3-8b
make build.xdna2-bfp16
XDNA_FORCE_CPU=1 ./xdna2-bfp16/qwen3-xdna2-bfpx Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2-bfp16/qwen3-xdna2-bfpx \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
make run.xdna2-bfp16 PROMPT="Short explanation in English."
```

### Notes

- **`xdna2/qwen3-xdna2`**: Linear weights stay **mmap'd** (**CPU OpenMP** build–like). One **BF16 scratch** sized for the **largest text-path GEMV** (often **LM head / embedding scale**) may still require substantial **DRAM**. There is **no** persistent duplicate BF16 copy of **all** layers. Insufficient RAM can still kill the process or fail mmap/allocs.
- **`xdna2-bfp16/qwen3-xdna2-bfpx`**: Inference residency is often dominated by **BFPX + norm buffers** with mmap released early, but **conversion** can **spike memory** (GGUF mmap plus temporary full-tensor staging).
- On NPU runs you reserve AIE columns; other NPU workloads (e.g. Windows Studio Effects) may contend.

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
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

Then increase `-n`:

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Write a short poem." \
  -n 128
```

## More deterministic output

Lower temperature and fix the seed when comparing runs:

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "One sentence: what is GGUF?" \
  -n 32 \
  -t 0.2 \
  -s 42
```

Byte-identical output across CPU vs GPU is not guaranteed; compare with the **same binary**, **same model**, and **same flags**.

## Clean

Remove build artifacts:

```bash
cd qwen3-8b
make clean
```

Typical files removed:

- `cpu/qwen3-cpu`
- `cpu-multicore/qwen3-cpu-omp`
- `cpu-blas/qwen3-cpu-blas`
- `gpu-rocm/qwen3-rocm`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`

**CUDA** artifacts (`gpu-cuda/qwen3-gpu-cuda`, `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4`, etc.) are **not** removed by aggregate `make clean`. To clean them:

```bash
cd qwen3-8b/gpu-cuda && make clean
cd qwen3-8b/gpu-cuda-nvfp4 && make clean
```

`make clean` does **not** delete the GGUF model.

## Troubleshooting

### `No such file or directory`

Wrong model path.

```bash
ls -lh qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

Put the model under `qwen3-8b/` or pass an absolute path:

```bash
./cpu/qwen3-cpu /data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

### CPU is slow

Expected for 8B on CPU alone. Try `-n 1` or `-n 4`:

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

For speed, use `./gpu-rocm/qwen3-rocm` on AMD GPUs, `gpu-cuda/qwen3-gpu-cuda` (FP16) or `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4` (Blackwell NVFP4) on NVIDIA GPUs. On CPU only, **`cpu-blas/qwen3-cpu-blas`** (OpenBLAS + Q8_K quantized GEMV + AVX2 dots + layer-shared Q8 + RoPE cache + prefill LM skip + greedy argmax) often outperforms **`cpu-multicore`**. **Greedy (`-t 0`)** makes decode LM head even lighter.

### `cpu-blas` build fails / `cblas.h` not found

Install OpenBLAS dev packages and set `CPPFLAGS` if needed (see **OpenBLAS build** under Requirements).

### `cpu-blas` output is garbage (repeated characters, etc.)

Building with **`-ffast-math`** breaks IQ / Q8_K quantized dot products. The repo Makefile disables it—remove it if you override `CFLAGS`.

### `nvcc` not found / `nvlink` errors

Ensure CUDA `bin` is on `PATH` or set `CUDA_HOME` in each directory’s `Makefile`.

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

If a PTX-only build is very slow, rebuild with `CUDA_GENCODE=arch=compute_XX,code=sm_XX` for your GPU.

### NVFP4 build fails / `NVFP4 quantize failed`

In **`gpu-cuda-nvfp4`**, confirm **`sm_120a`** build, CUDA 13, and **`make cutlass`**. On general GPUs use **`gpu-cuda`** with **`make build`** (FP16). If apt **CUDA 11** and **CUDA 13** are mixed, run **`make blackwell`** or remove 11.x manually.

### `make fp4-test` fails / `Arch conditional MMA instruction... Aborting`

You may be using a stale **`fp4_gemm.o`** built for **`compute_86` PTX** only. In **`gpu-cuda-nvfp4`**, run **`make clean`** → **`make fp4-test`** again (**`fp4_gemm.sm120a.o`** is fixed to **`sm_120a` + C++17**). Requires a **Blackwell GPU**.

### NVFP4 decode is extremely slow

An older binary may route M=1 through CUTLASS with M=128 padding. Confirm a recent **`gpu-cuda-nvfp4`** build with **`fp4_gemv_cached`** and check startup logs for **`GEMM M>=128, GEMV decode`**.

### `gpu-cuda-nvfp4` PolarQuant build is slow / `Killed` (OOM)

Compiling **`kernels.cu`** with **`sm_120a` + FP4 + PolarQuant** uses a lot of RAM. You may see **`Error 137`**. Try adding swap, disabling parallel builds, or running **`make cutlass`** then **`make build.polarquant`** again.

### PolarQuant not active

If **`PolarQuant-R: KV cache enabled`** is missing at startup, confirm you built with **`build.polarquant`** (`gpu-cuda` or `gpu-cuda-nvfp4`). PolarQuant is disabled when **`head_dim ≠ 128`**.

### `hipcc` not found

Check ROCm location:

```bash
ls /opt/rocm/bin/hipcc
```

If elsewhere:

```bash
make build.gpu-rocm ROCM=/path/to/rocm
```

### Wrong `GPU_ARCH` / detection failed

Must match the GPU ISA. Normally **`GPU_ARCH`** is auto-detected from `rocminfo`; if detection fails or you need a different target, set it manually:

```bash
rocminfo | awk '/^  Name:/ { n=$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }'
make build.gpu-rocm GPU_ARCH=gfx1100
```

### ROCm Prefill is slow (~30 tok/s)

Check startup log for **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`**. If missing or prefill is very slow, **`make -C gpu-rocm clean build`** and confirm **`-lhipblas -lrocblas`** are linked. See **“Prefill acceleration (overview)”** above and **`doc/design.md`**, section **“ROCm Prefill acceleration (3 stages)”**.

### **`undefined reference to hipblas*`** / hipBLAS link failure

Verify **`libhipblas.so`** / **`librocblas.so`** under **`$(ROCM)/lib`**. Reinstall ROCm or fix **`ROCM=`** path if needed.

### **`make wmma` fails**

Possible causes: **WMMA instructions in `qwen3-rocm` binary** (unexpected), **hipBLAS path not reported**, **`wmma-probe` calibration failure**. Try **`make wmma WMMA_SKIP_RUN=1`** for static checks only. Set **`LLVM_OBJDUMP=$(ROCM)/llvm/bin/llvm-objdump`**. Use **`WMMA_SKIP_RUN=1`** when MODEL is not available.

### `/dev/accel/accel0` opens but `CREATE_HWCTX` returns `EINVAL`

The driver may reject column/tile settings. Try `XDNA_NUM_COL=1` and check `dmesg` for `amdxdna` (see `doc/design.md`).

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
6. `qwen3-8b/gpu-rocm/main.c` — GPU memory, HIP kernels, GPU sampling. **Prefill**: **`forward_prefill_gpu`** (**hipBLAS GemmEx** + batch Attention/Norm, etc.; **`attn_flash_prefill_kernel`** uses **`(int)threadIdx.x`** for signed comparisons). **Decode**: **`forward_gpu`**. **Prefill progress bar** and throughput summaries. Benchmark history via **`make log` / `make log.push`**. **`make wmma`** (**`wmma_probe.c`** / **`scripts/check_wmma.sh`**) to verify hipBLAS path. Prefill details in **`doc/design.md`**, section **“ROCm Prefill acceleration (3 stages)”**.
7. `qwen3-8b/gpu-cuda/main.c` / `kernels.cu` / `polarquant.cu`  
   CUDA FP16 prefill/decode, Flash Attention. Optional **`build.polarquant`**: PolarQuant-R KV (**`pq_decode_head`** tile decode). **Prefill progress bar** and throughput summaries. Benchmark history via **`gpu-cuda/Makefile`** **`make log` / `make log.push`**.

8. `qwen3-8b/gpu-cuda-nvfp4/fp4_qwen3.cu` / `fp4_gemm.cu`  
   Blackwell NVFP4 build. H2D NVFP4 load, **`fp4_gemv_cached`** (decode), **`fp4_qwen3_mm`** (GEMM/GEMV routing). **`fp4_*` uses C++17 + fixed `sm_120a`** (CUTLASS). Shared sources live under **`../gpu-cuda/`**. Benchmark history via **`gpu-cuda-nvfp4/Makefile`** **`make log` / `make log.push`**.

9. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c` — `amdxdna` ioctl, `ERT_START_NPU`, `launch_mm_bf16`, CPU fallback. **Mmap scratch build**: `load_weights_xdna` / `weight_prepare_bf16` / single `w_scratch_bo`. **BFPX**: `bfpx_convert_weight_2d` and the mmap release path.

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
- Batch inference tuning
- Image input
- **Built-in multi-turn CLI** (history management, KV reuse, full official chat template)
- **Thinking mode control and filtered reasoning display** (`enable_thinking`, `/think`, `/no_think`, etc.)
- Server or Web API packaging
- Universal support for every GGUF quantization
- Guaranteed numerical match with official implementations

The goal is to **understand, experiment with, and adapt** Qwen3-family GGUF text inference in C, HIP, and CUDA.

## More documentation

- Design: `doc/design.md`
- Changelog: `doc/ChangeLog.md`

When stuck, check `qwen3-8b/Makefile` target names and the model path you pass at runtime—most build/run issues come from those two drifting apart.
