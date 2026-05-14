# Qwen3.c

This repository is an **inference implementation** that runs **Qwen3-family models** directly from **a single C source**, **without relying on external libraries**.

**This project does not link userland ML libraries or runtimes such as PyTorch, TensorFlow, JAX, or ONNX Runtime.** Inference is built around **standard C and `libm`**, with one or a few sources under `qwen3-8b/`. The GPU build uses **ROCm/HIP** (`hipcc`), CPU parallelism uses **OpenMP**, and the XDNA2 NPU build talks to the **Linux kernel `amdxdna` DRM ioctl (UAPI)** directly—there is no dependency on a Python runtime or `torch`.

Among the above, ROCm/HIP is AMD’s GPU compiler and runtime; it is **not a high-level neural network framework** (this repo builds the Transformer from custom HIP kernels and host code).

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

A small inference implementation that runs Qwen3-family GGUF models from **straightforward C sources** under `qwen3-8b/`. Paths include **CPU**, **OpenMP**, **ROCm/HIP on AMD GPUs**, and **AMD Ryzen AI XDNA2 NPU** (direct **`amdxdna` DRM ioctl** usage).

The scope is the **text decoder of Qwen3-VL-8B-Instruct**. Image input and the vision encoder are **out of scope**; use cases are prompt-in, text-out generation.

日本語版は [README.md](README.md) を参照してください。

## What you can run

Build the C sources under `qwen3-8b/` and try the following targets:

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | Learning the flow, minimal setup |
| CPU OpenMP | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | Faster CPU trials |
| ROCm/HIP GPU | `qwen3-8b/gpu/main.c` | `gpu/qwen3-rocm` | Practical speed on AMD GPUs |
| AMD Ryzen AI XDNA2 NPU (mmap + per-GEMV BF16 scratch) | `qwen3-8b/xdna2/main.c` | `xdna2/qwen3-xdna2` | NPU via direct `amdxdna` ioctl; weights **mmap'd** like **CPU OpenMP** build; single BF16 scratch BO filled **per GEMV** |
| AMD Ryzen AI XDNA2 NPU (BFPX host weights) | `qwen3-8b/xdna2-bfp16/main.c` | `xdna2-bfp16/qwen3-xdna2-bfpx` | Same ioctl/GEMV path; linear weights held on host as block FP (BF16 scale + int8); GGUF mmap released after conversion |

For an **overview of AMD XDNA** (design goals, tile-level architecture, generational changes, dtypes and accuracy, software stack, comparison with other NPUs, etc.), see the companion write-up: [thamada/xdna-overview](https://github.com/thamada/xdna-overview) (`main.md` plus a PDF).

An 8B model on CPU is **very slow**. CPU is fine for a first smoke test; for usable token throughput, use ROCm/HIP or the XDNA2 NPU builds. **`xdna2/qwen3-xdna2`** avoids **persistent BF16 copies of every layer**—it keeps quantized weights **mmap'd** (**CPU OpenMP** build–style residency) and decodes **one GEMV matrix at a time** into a BF16 scratch for the NPU, so latency per matmul rises. For tighter residency or another host layout, **`xdna2-bfp16/qwen3-xdna2-bfpx`** converts to **BFPX** and drops the GGUF mmap after conversion (**large load-time peak possible**); output is **not** bit-aligned with **`xdna2/qwen3-xdna2`**.

## Repository layout

```text
.
├── README.md
├── README.en.md
├── doc/
│   ├── ChangeLog
│   └── design.md
├── xdna-gemv/
│   ├── README.md
│   ├── gen-xdna-gemv-stubs.py
│   ├── kernels/
│   └── toolchain/
└── qwen3-8b/
    ├── Makefile
    ├── gguf.txt
    ├── cpu/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-multicore/
    │   ├── Makefile
    │   └── main.c
    ├── gpu/
    │   ├── Makefile
    │   └── main.c
    ├── xdna2/
    │   ├── Makefile
    │   └── main.c
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

### ROCm/HIP build

You need an AMD GPU and ROCm. The `Makefile` assumes ROCm under `/opt/rocm` by default.

Check:

```bash
/opt/rocm/bin/hipcc --version
rocminfo | grep -m 1 gfx
```

Pass the `gfx…` value from `rocminfo` as `GPU_ARCH` at build time.

## Obtain the model file

The default name in `qwen3-8b/Makefile` is:

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

The GGUF is **not** shipped in the repo (license + size). Download it from the URL in `qwen3-8b/gguf.txt` and place it under `qwen3-8b/`.

```bash
cd qwen3-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf "$url"
```

You should then have:

```text
qwen3-8b/
├── Makefile
├── cpu/ … (`main.c` → `cpu/qwen3-cpu`)
├── cpu-multicore/ …
├── gpu/ …
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
make build
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

On success, text should appear gradually after load.

## CPU single-thread

### Build

```bash
cd qwen3-8b
make build
```

Produces **`cpu/qwen3-cpu`**.

```bash
ls -lh cpu/qwen3-cpu
```

### Run

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Give a one-sentence introduction of yourself." \
  -n 16
```

Using the `Makefile` `run` target:

```bash
make run PROMPT="Give a one-sentence introduction of yourself."
```

Model elsewhere:

```bash
make run MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP

Uses multiple CPU cores; same model file as single-thread.

### Build

```bash
cd qwen3-8b
make build.omp
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

## ROCm/HIP GPU

The primary path when ROCm and an AMD GPU are available.

### Find `GPU_ARCH`

```bash
rocminfo | grep -m 1 gfx
```

If you see e.g. `gfx1201`, build with `GPU_ARCH=gfx1201`.

### Build

```bash
cd qwen3-8b
make build.rocm GPU_ARCH=gfx1201
```

If ROCm is not under `/opt/rocm`:

```bash
make build.rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

Produces **`gpu/qwen3-rocm`**.

### Run

```bash
./gpu/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Explain what ROCm is for beginners." \
  -n 64
```

Using `run.rocm`:

```bash
make run.rocm GPU_ARCH=gfx1201 PROMPT="Short explanation in English."
```

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

The repo ships **`xdna-gemv/kernels/`** with **64-byte placeholders** (magic `GQF3XDNA`). They are **not** executed on the device (`--xdna-status` shows `[STUB]`). Regenerate with `python3 xdna-gemv/gen-xdna-gemv-stubs.py xdna-gemv/kernels` from the repo root, or `make gen-xdna-kernels` from `qwen3-8b/`. Replace with real MLIR-AIE outputs for hardware GEMV.

Useful env vars: `XDNA_GEMV_DIR` (search path for control blobs), `XDNA_FORCE_CPU=1` (force CPU), `XDNA_NUM_COL` (column count; try `XDNA_NUM_COL=1` if `CREATE_HWCTX` returns `EINVAL`).

```bash
# Force CPU fallback
XDNA_FORCE_CPU=1 ./xdna2/qwen3-xdna2 Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# Repo stub placeholders (from qwen3-8b/): not real NPU ctrlcode — `--xdna-status` shows [STUB]
XDNA_GEMV_DIR=../xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status

# With real MLIR-AIE blobs under XDNA_GEMV_DIR: NPU path
XDNA_GEMV_DIR=../xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
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
make build.xdna2.bfpx
XDNA_FORCE_CPU=1 ./xdna2-bfp16/qwen3-xdna2-bfpx Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=../xdna-gemv/kernels ./xdna2-bfp16/qwen3-xdna2-bfpx \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
make run.xdna2.bfpx PROMPT="Short explanation in English."
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
./gpu/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Write a short poem." \
  -n 128
```

## More deterministic output

Lower temperature and fix the seed when comparing runs:

```bash
./gpu/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
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
- `gpu/qwen3-rocm`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`

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

For speed, use `./gpu/qwen3-rocm`.

### `hipcc` not found

Check ROCm location:

```bash
ls /opt/rocm/bin/hipcc
```

If elsewhere:

```bash
make build.rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

### Wrong `GPU_ARCH`

Must match the GPU ISA:

```bash
rocminfo | grep -m 1 gfx
```

```bash
make build.rocm GPU_ARCH=gfx1100
```

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
5. `qwen3-8b/gpu/main.c` — GPU memory, HIP kernels, GPU sampling.
6. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c` — `amdxdna` ioctl, `ERT_START_NPU`, `launch_mm_bf16`, CPU fallback. **Mmap scratch build**: `load_weights_xdna` / `weight_prepare_bf16` / single `w_scratch_bo`. **BFPX**: `bfpx_convert_weight_2d` and the mmap release path.

## Out of scope

- Training / fine-tuning
- Batch inference tuning
- Image input
- Server or Web API packaging
- Universal support for every GGUF quantization
- Guaranteed numerical match with official implementations

The goal is to **understand, experiment with, and adapt** Qwen3-family GGUF text inference in C/HIP.

## More documentation

- Design: `doc/design.md`
- Changelog: `doc/ChangeLog`

When stuck, check `qwen3-8b/Makefile` target names and the model path you pass at runtime—most build/run issues come from those two drifting apart.
