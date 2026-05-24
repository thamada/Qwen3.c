# Qwen3.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、**ライブラリに依存せず、単一の C言語ソースから Qwen3系モデルを直接動かす推論実装**です。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。** 推論は **標準Cと `libm`** を中心に、`qwen3-8b/` 内の単一ソースで完結します。AMD GPU 版は **ROCm/HIP**（`hipcc`）、NVIDIA GPU 版は **CUDA Toolkit**（`nvcc`）、CPU 並列は **OpenMP**（`cpu-multicore`）または **OpenMP + OpenBLAS**（`cpu-blas`）、XDNA2 NPU 版は **Linux カーネルの `amdxdna` DRM ioctl（UAPI）** を直接叩く構成であり、Pythonランタイムや `torch` に依存するレイヤはありません。

ROCm/HIP および CUDA はいずれも **GPU 向けのコンパイラ・ランタイム** であり、**ニューラルネット用の高レベルフレームワークではありません**（ここからさらに自作の HIP / CUDA カーネルとホストコードで Transformer を組み立てています）。

### なぜライブラリ非依存なのか

一般的な LLM推論は PyTorch などの高レベルな機械学習フレームワークを利用することで、短いコードで高速に実行できます。一方で、その構成では **計算手順、メモリ配置、アライメント、量子化レイアウト**といった低レベルの詳細が、フレームワークやランタイムの内部に隠れがちです。

本リポジトリでは、あえてその層に依存せず、**GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを Cのコードパスとして明示する**ことを重視しています。これは既存フレームワークを置き換えるためではなく、推論処理の実体を観察し、検証し、必要に応じて変更できる形で保持するためです。

この方針には、次の意義があります。

- **理解可能性**: モデルファイルから何を読み、どのバッファに置き、どの順序で計算しているかを、ソースコードと `doc/design.md` から直接追跡できる。
- **依存関係の単純化**: Python環境や大規模な機械学習スタックを前提にせず、基本的な Cコンパイラと必要最小限の実行環境で動作経路を確認できる。
- **実験の自由度**: 量子化形式、メモリ表現（例: BFPX）、CPU/GPU/NPU への処理分担、`/dev/accel` への直接アクセスなど、フレームワークの抽象化に制約されやすい領域を個別に試せる。
- **参照実装としての価値**: 「最小限の構成で Qwen3系デコーダ推論がどのように成立するか」を示し、既存スタックとの比較や実装検証の基準にできる。

したがって、この実装は最高性能や機能網羅を第一目的とするものではありません。主眼は、LLM推論の仕組みをブラックボックスにせず、開発者が実装の細部を把握しながら改造できる状態に置くことです。

---

Qwen3系GGUFモデルを、**Cの単一ソース群**から直接動かす小さな推論実装です。実行経路は **CPU／OpenMP／OpenMP+OpenBLAS／ROCm HIP（AMD GPU）／CUDA（NVIDIA GPU）／AMD Ryzen AI XDNA2 NPU（`amdxdna` DRM ioctl の直叩き）**と選べます。

このリポジトリは **Qwen3-VL-8B-Instruct のテキストデコーダ**を対象にしています。画像入力や Vision エンコーダは扱わず、プロンプト文字列を入力してテキストを生成する用途に絞っています。

## まず何ができるのか

このリポジトリでは、`qwen3-8b/` の中にある Cソースをビルドして、次の実行方法を試せます。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | 仕組みを追う、最小構成で動かす。**Prefill progress bar** とスループット要約を stderr に出力 |
| CPU OpenMP 並列 | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | CPU で少しでも速く試す |
| CPU OpenMP + OpenBLAS | `qwen3-8b/cpu-blas/main.c` | `cpu-blas/qwen3-cpu-blas` | F32 GEMV と Attention を BLAS 化。量子化 GEMV は **Q8_K 活性化 + 全型 AVX2 整数内積**（層内 Q8 共有）。**RoPE キャッシュ**、prefill 中 **LM head スキップ**、greedy 時 **`mm_argmax_row`**。**F16 埋め込み F16C**。stderr に **Prefill progress bar** |
| ROCm/HIP GPU | `qwen3-8b/gpu-rocm/main.c` | `gpu-rocm/qwen3-rocm` | AMD GPU。**Prefill** は全プロンプトを 1 回 forward + **hipBLAS GemmEx**（[llama.cpp](https://github.com/ggml-org/llama.cpp/) の cublas 経路同趣旨）。**Decode** は 1 トークン GEMV。**Prefill progress bar** と prefill / decode / total スループット要約。**`make log` / `make log.push`** ベンチ履歴。**`make wmma`** で hipBLAS 経路・WMMA 非埋め込みを確認 |
| CUDA GPU（FP16） | `qwen3-8b/gpu-cuda/main.c` + `kernels.cu` | `gpu-cuda/qwen3-gpu-cuda` | NVIDIA GPU。Prefill バッチ + Flash Attention。全線形層 **FP16 VRAM**。任意で **`build.polarquant`**: KV **PolarQuant-R**（64 B/head）。集約 `Makefile` 外 |
| CUDA GPU（NVFP4） | `qwen3-8b/gpu-cuda-nvfp4/` + 共有 `gpu-cuda/` | `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4` | Blackwell（RTX 50 系等）。線形層は H2D 時 **NVFP4 のみ**（CUTLASS）。埋め込みのみ FP16 VRAM。任意で **`build.polarquant`**: NVFP4 + PolarQuant-R 同時（最大 VRAM 節約）。集約 `Makefile` 外 |
| AMD Ryzen AI XDNA2 NPU（mmap＋GEMV単一BF16スクラッチ） | `qwen3-8b/xdna2/main.c` | `xdna2/qwen3-xdna2` | `amdxdna` ioctl 直通。ウェイトは **GGUF mmap**（CPU OpenMP 版と同様）。各 GEMV 直前のみ **単一 BF16 SHMEM** に復号展開して NPU へ載せる |
| AMD Ryzen AI XDNA2 NPU（BFPXホスト重み） | `qwen3-8b/xdna2-bfp16/main.c` | `xdna2-bfp16/qwen3-xdna2-bfpx` | 同上の IOCTL・GEMV パイプラインだが、線形重みをブロック FP（BF16スケール + int8）でホスト保持。GGUF mmap は変換後に解放 |

**AMD XDNA の概要**（設計思想、アーキテクチャの基本構造とタイル、世代別の進化、データ型と精度、ソフトウェアスタック、他社 NPU との比較など）については、別リポジトリに解説記事としてまとめてあります：[thamada/xdna-overview](https://github.com/thamada/xdna-overview)（本文は `main.md`、PDF 付き）。

8B 級モデルの CPU 実行は非常に重いです。最初の動作確認としては CPU でも構いませんが、実用的な生成速度が必要な場合は **ROCm/HIP 版**（AMD GPU）、**CUDA 版**（NVIDIA GPU）、または **XDNA2 NPU 版**を使う想定です。**`xdna2/qwen3-xdna2`** は **全重みを恒久に BF16 へ複製しない**ため、推論中にメモリへ置く主なデータは **GGUF の mmap** と **最大 GEMV 向けスクラッチ**になり、CPU OpenMP 版に近い構成です。一方で、**GEMV のたびに行列全体を復号する**ためレイテンシは増えやすいです。**推論中のメモリ使用量をさらに減らしたい**ときや別の重み表現が必要なときは **`xdna2-bfp16/qwen3-xdna2-bfpx`** を検討してください（BFPX 変換後に mmap を解放。**ロード時のメモリピークは大きくなり得る**。出力は **`xdna2/qwen3-xdna2`** とビット単位では一致しません）。

## ディレクトリ構成

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
    │   ├── wmma_probe.c          （`make wmma-probe` — WMMA 検出器校正）
    │   └── scripts/
    │       └── check_wmma.sh     （`make wmma`）
    ├── gpu-cuda/                    （FP16 線形層・NVFP4 なし）
    │   ├── Makefile
    │   ├── main.c
    │   ├── kernels.cu
    │   ├── gpu.h
    │   └── polarquant.cu / polarquant_kernels.cuh / polarquant_verify.cu  （PolarQuant-R KV 時）
    ├── gpu-cuda-nvfp4/              （Blackwell NVFP4）
    │   ├── Makefile
    │   ├── fp4_gemm.cu / fp4_qwen3.cu / fp4_verify.cu
    │   └── third_party/cutlass/  （make cutlass で取得）
    │   （main.c / kernels.cu / gpu.h / polarquant.* は ../gpu-cuda/ を参照）
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

主に触る場所は `qwen3-8b/` です。ビルドも推論実行も、基本的にはこのディレクトリに移動してから行います。

## 初心者向け: LLM推論で何が起きるか

LLM推論は、大まかには次の流れです。

1. **GGUF ファイルを読む**  
   モデルの重み、語彙、設定値が入った大きなファイルを読みます。

2. **プロンプトをトークンに分解する**  
   `"こんにちは"` のような文字列を、モデルが扱える整数 ID の列に変換します。

3. **Transformer を 1 トークンずつ実行する**  
   モデルは「次に来そうなトークン」を予測します。

4. **サンプリングする**  
   予測結果から次のトークンを選びます。`-t` や `-k` で選び方を調整できます。

5. **トークンを文字列に戻して表示する**  
   選ばれたトークンをテキストとして端末に出します。

このリポジトリの特徴は、この流れを **PyTorch 等の機械学習スタックに載せず**、巨大なフレームワークに隠さず **Cソースの中で追える**ことです。

## 必要なもの

### 共通

- Linux
- `make`
- Cコンパイラ（例: `gcc`, `clang`, `cc`）
- `libm`（通常は標準で入っています）
- Qwen3-VL-8B-Instruct の GGUF モデルファイル

Ubuntu系なら、CPU版に必要な基本ツールは次で入ることが多いです。

```bash
sudo apt update
sudo apt install -y build-essential make
```

### OpenMP 版を使う場合

GCC なら通常 `-fopenmp` でビルドできます。環境によっては OpenMP ランタイムが必要です。

```bash
sudo apt install -y libgomp1
```

### OpenBLAS 版（`cpu-blas`）を使う場合

**OpenBLAS**（`libopenblas-dev` 等）と OpenMP ランタイムが必要です。`pkg-config openblas` が使える環境では Makefile が自動で include / link フラグを拾います。パッケージ導入は **`cpu-blas/Makefile`** の **`make openblas`** でも行えます（`libopenblas-dev` / `libgomp1` を apt 導入）。

```bash
sudo apt install -y libopenblas-dev libgomp1
# または
cd qwen3-8b/cpu-blas && make openblas
```

ヘッダが標準パスに無い場合（Debian/Ubuntu の pthread ビルド等）は、ビルド時に `CPPFLAGS` で指定します。`cblas.h` 未検出時は Makefile が **`make openblas`** と **`CPPFLAGS`** 例を表示します。

```bash
cd qwen3-8b/cpu-blas
make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
```

実行時は **`OMP_NUM_THREADS`** で CPU 並列度を調整します。OpenBLAS 側は **`openblas_set_num_threads(1)`** で 1 スレッド固定（OpenMP との二重並列化を避ける）です。**`-ffast-math` は IQ / Q8_K 量子化で数値が崩れるため Makefile では無効**にしています。既定 **`CFLAGS`** には **`-march=native`** が含まれます（移植性より当該 CPU 向け最適化を優先）。

### ROCm/HIP 版を使う場合

AMD GPU と ROCm が必要です。`Makefile` は既定で ROCm を `/opt/rocm` にあるものとして扱います。`GPU_ARCH`（`hipcc --offload-arch`）は **`rocminfo` から自動検出**されます。

確認例:

```bash
/opt/rocm/bin/hipcc --version
make -C gpu-rocm detect-gpu-arch   # 例: Detected GPU arch: gfx1100
```

`rocminfo` が GPU を報告しない環境では、ビルド時に `GPU_ARCH=gfx1100` のように手動指定してください。

### CUDA 版を使う場合

NVIDIA GPU と **CUDA Toolkit**（`nvcc`・`libcudart`）が必要です。集約 `qwen3-8b/Makefile` には CUDA ターゲットは無く、次の2ディレクトリで単体ビルドします。

- **`qwen3-8b/gpu-cuda/`** — FP16 線形層（汎用 NVIDIA GPU）
- **`qwen3-8b/gpu-cuda-nvfp4/`** — NVFP4 線形層（Blackwell / RTX 50 系、**CUDA 13** + CUTLASS）。初回は **`make cutlass`** で **`third_party/cutlass`** を取得。**`fp4_*` オブジェクトは C++17 でビルド**（CUTLASS 要件）。apt **`nvidia-cuda-toolkit`（CUDA 11）** と CUDA 13 は併存させない（**`make blackwell`** が 11.x を除去して 13 を入れる）。

確認例:

```bash
nvcc --version
nvidia-smi
```

`nvcc` / `nvlink` は **CUDA の `bin` ディレクトリ**を `PATH` に通してください（`/usr/local/bin/nvcc` のみだとリンクに失敗することがあります）。

| 用途 | コマンド |
|------|----------|
| **FP16 のみ**（Ampere/Ada 等・PTX 可） | `cd qwen3-8b/gpu-cuda` → `make build` / `make run` |
| **PolarQuant-R KV キャッシュ**（FP16 線形・任意 GPU） | `cd qwen3-8b/gpu-cuda` → `make build.polarquant` / `make run.polarquant` |
| **Blackwell NVFP4**（RTX 50 系等） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build` / `make run` |
| **NVFP4 + PolarQuant 同時**（Blackwell・最大 VRAM 節約） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build.polarquant` / `make run.polarquant` |
| CUDA 13 の導入から NVFP4 一式 | `cd qwen3-8b/gpu-cuda-nvfp4` → `make blackwell` |
| PolarQuant ラウンドトリップ検証 | 各ディレクトリで `make pq-test` |
| CUTLASS NVFP4 GEMM 単体検証 | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fp4-test`（**Blackwell / sm_120a 必須**） |

FP16 ビルド（`gpu-cuda`）の既定は PTX（`compute_86`）。実 GPU 向けには `CUDA_GENCODE=arch=compute_XX,code=sm_XX` を指定します。NVFP4 ビルド（`gpu-cuda-nvfp4`）の既定は **`sm_120a`** です。**`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** は **`BLACKWELL_NVCCFLAGS`**（**`-std=c++17`** + **`sm_120a` 固定**）でコンパイルします。CUTLASS は **`v4.5.0`**（**`make cutlass`** で **`third_party/cutlass`** を取得）を使用し、CUDA 13 非推奨ベクトル型警告は CUTLASS 側で解消済みです。

## モデルファイルを置く

`qwen3-8b/Makefile` の既定モデル名は次です。

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

モデルファイルは著作権とファイルサイズの都合により、リポジトリには含めません。`qwen3-8b/gguf.txt` の URL から取得し、`qwen3-8b/` の直下に置きます。推奨は集約 Makefile の **`make model`**（`wget` + 同梱 `.sha256sum` で検証）です。

```bash
cd qwen3-8b
make model
```

手動で取得する場合:

```bash
cd qwen3-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf "$url"
sha256sum -c Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

配置後、次のようになっていれば準備完了です。

```text
qwen3-8b/
├── Makefile
├── cpu/ … （`main.c` → `cpu/qwen3-cpu`）
├── cpu-multicore/ …
├── cpu-blas/ …
├── gpu-rocm/ …
├── gpu-cuda/ …
├── gpu-cuda-nvfp4/ …
├── xdna2/ …
├── xdna2-bfp16/ …
└── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

SHA256 を確認したい場合:

```bash
cd qwen3-8b
sha256sum -c Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

`OK` と出れば、少なくともこのリポジトリが想定しているファイル名とハッシュに一致しています。

## いちばん簡単な実行手順

まず CPU 版で「ビルドできるか」を確認します。8B モデルなので生成は遅くても問題ありません。`-n 1` のように生成トークン数を少なくすると、初回確認が楽です。

```bash
cd qwen3-8b
make model          # 未取得なら GGUF を取得・検証
make build.cpu
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

うまくいくと、モデル読み込み後に少しずつテキストが表示されます。

## CPU 単スレッド版

### ビルド

```bash
cd qwen3-8b
make build.cpu
```

成功すると **`cpu/qwen3-cpu`** ができます（`cpu/` 直下で `make build` でも可）。

```bash
ls -lh cpu/qwen3-cpu
```

### 実行

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます。

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で短く自己紹介してください。" -n 16
```

集約 `Makefile` の `run.cpu` を使う場合:

```bash
make run.cpu PROMPT="日本語で短く自己紹介してください。"
```

別の場所にあるモデルを使う場合:

```bash
make run.cpu MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP 版

CPU コアを複数使う版です。単スレッド版と同じモデルを読みます。

### ビルド

```bash
cd qwen3-8b
make build.cpu-multicore
```

成功すると **`cpu-multicore/qwen3-cpu-omp`** ができます。

### 実行

```bash
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "箇条書きで、量子化とは何かを説明してください。" \
  -n 32
```

`OMP_NUM_THREADS` は使う CPU スレッド数です。迷ったら、まずは 4 や 8 から試してください。

```bash
OMP_NUM_THREADS=4 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

速くなるかどうかは CPU のコア数、メモリ帯域、モデルの量子化形式に依存します。

## CPU OpenMP + OpenBLAS 版

`cpu-multicore` と同じデコーダ・同じ GGUF を読み、**F32 行列積（`cblas_sgemv`）** と **Attention の K/V 合成**を OpenBLAS に任せます。IQ2_S / IQ3_S / Q4_K / Q5_K の量子化 GEMV は、入力を **Q8_K** に量子化（**`quantize_row_q8_K`**）したうえで **ggml-cpu/quants.c** 準拠の **`vec_dot_*_q8_K`** 整数内積を使います（`cpu-multicore` のような per-row float[256] dequantization は行わない）。**Attention / FFN 内で同一活性ベクトルを共有する GEMV では Q8 量子化を 1 回にまとめる**（層内 Q8 共有）。**`__AVX2__`** 時は IQ2_S / IQ3_S / Q4_K / Q5_K の dot と Q8 量子化を SIMD 化（**`-march=native`** 既定）。

そのほか CPU 向けの最適化として、**RoPE cos/sin キャッシュ**（起動時に `[max_seq × head_dim/2]` を一括計算）、prefill 中（最終プロンプト token 以外）の **LM head スキップ**（**`FWD_NO_LM`**）、**`-t 0`（greedy）** 時の **`mm_argmax_row`**（全 vocab logits を確保しない）、**F16 埋め込み**の **F16C+AVX2** SIMD 変換があります。プロンプト区間は stderr に **Prefill progress bar**（`Prefill [====...]`、幅 40）と prefill / decode / total のスループット要約を出力します。

### ビルド

```bash
cd qwen3-8b
make build.cpu-blas
```

成功すると **`cpu-blas/qwen3-cpu-blas`** ができます（`cpu-blas/` 直下で `make build` でも可）。

### 実行

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます。

```bash
OMP_NUM_THREADS=8 ./cpu-blas/qwen3-cpu-blas Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Hello, how are you?" \
  -n 32
```

集約 `Makefile` の `run.cpu-blas` を使う場合:

```bash
make run.cpu-blas PROMPT="Hello, how are you?"
```

## ROCm/HIP GPU 版

AMD GPU と ROCm が使える環境では、こちらが本命です。

### GPU_ARCH（自動検出）

`gpu-rocm/Makefile` は、ビルド前に `$(ROCM)/bin/rocminfo` から最初の GPU エージェント名（`gfx*`）を **`GPU_ARCH` として自動検出**します。成功すると次のように表示されます:

```text
===============================================
  Detected GPU arch: gfx1100
===============================================
```

手動で上書きする場合:

```bash
make build GPU_ARCH=gfx1100          # gpu-rocm/ 内
make build.gpu-rocm GPU_ARCH=gfx1100 # qwen3-8b/ から
```

### ビルド

```bash
cd qwen3-8b
make build.gpu-rocm
```

ROCm が `/opt/rocm` 以外にある場合:

```bash
make build.gpu-rocm ROCM=/path/to/rocm
```

成功すると **`gpu-rocm/qwen3-rocm`** ができます。リンクには **`-lhipblas -lrocblas`** が含まれます（Prefill 線形層の GEMM 用）。

### Prefill 高速化（概要）

ROCm 版は Prefill（プロンプト処理）と Decode（トークン生成）を **分離** しています。

| フェーズ | 関数 | 線形層 | 説明 |
|----------|------|--------|------|
| **Prefill** | `forward_prefill_gpu` | **hipBLAS `GemmEx`** | プロンプト **S トークン** を **1 回の forward** で並列処理 |
| **Decode** | `forward_gpu` | カスタム **GEMV** | 生成トークンを **1 つずつ** 処理 |

**なぜ Prefill だけ速くできるか**（詳細は [`doc/design.md`](doc/design.md) の **「ROCm Prefill 高速化の詳細（3 段階）」**）:

1. **Prefill** ではプロンプト全トークンが既知なので、線形層を **S×d の GEMM**（行列×行列）としてまとめられる。重み HBM 読み出しを **S トークン分で割れる**。
2. **Decode** は 1 トークンずつ **GEMV**（行列×ベクトル）のまま。これが ITL（トークン間レイテンシ）の主体。

改善は **3 段階** で行った（132 prompt tokens、RX 7900 XTX / gfx1100、`make log.push` 相当）:

| 段階 | 方式 | prefill tok/s | 倍率（対段階 0） |
|------|------|---------------|------------------|
| 0（改善前） | 1 トークンずつ `forward_gpu`（GEMV × S 回） | 28.7 | 1.0× |
| 1 | バッチ `forward_prefill_gpu` + カスタム `mm_f16_gemv_batch_kernel` | 54 | 1.9× |
| 2 | 上記 + **hipBLAS GemmEx**（[llama.cpp](https://github.com/ggml-org/llama.cpp/) `cublasGemmEx` 同型） | **~550** | **~19×** |
| 2（再計測） | 段階 2 経路（`attn_flash_prefill_kernel` の `threadIdx` キャスト後） | **~557** | **~19×** |

段階 2 の要点:

- **行列積**: `O[S,d] = X[S,n] @ W[d,n]^T` を **`hipblasGemmEx(OP_T, OP_N, ...)`** で実行（FP16 重み・FP16 活性・FP32 出力）。
- **活性化変換**: `f32_to_f16_batch_kernel` で **`d_scratch_f16`** に変換。q/k/v や gate/up など **同一入力** は変換 **1 回** で使い回し。
- **Attention / RoPE / Norm / FFN 活性化** は従来の HIP バッチカーネルのまま（`attn_flash_prefill_kernel` 等）。Prefill Flash Attention では **`threadIdx.x`** を **`(int)threadIdx.x`** にキャストして **`hd`** / **`tc`** と比較（**`-Wsign-compare`** 回避。挙動は同一）。
- 起動時に **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** と表示されれば hipBLAS 経路が有効。

段階 2 再計測（**`BENCH_LOG` 2026-05-24**、132 prompt tokens）: prefill **556.77** / **556.95** tok/s — 初回（549.94）と同程度（計測ばらつき範囲）。

Decode スループット（~26 tok/s）は Prefill 改善の影響を受けない。長文プロンプトの **TTFT（Time To First Token）** が Prefill 速度に直結する。

### 実行

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、ROCmとは何かを初心者向けに説明してください。" \
  -n 64
```

`Makefile` の `run.gpu-rocm` を使う場合:

```bash
make run.gpu-rocm PROMPT="日本語で短く説明してください。"
```

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます（**`cpu-blas`** と同形式）。終了時 stdout には **`prefill_tps:` / `decode_tps:` / `total_tps:`** も出力され、**`make log.push`** がパースします（**推論区間のみ**。モデル重み H2D は計測外）。

### ベンチマーク履歴（`gpu-rocm/Makefile`）

`gpu-rocm/` で **`make log.push`** を実行すると、既定の長文プロンプト（~128 token）・**`-n 128`**・**`-t 0`** でベンチを走らせ、結果を **`Makefile` 内の `BENCH_LOG`** に追記します。**`make log`** で履歴を表表示できます。

```bash
cd qwen3-8b/gpu-rocm
make log.push                    # 既定 BENCH_N=128, BENCH_SEED=42
make log                         # 履歴一覧
make log.push BENCH_N=64         # 生成トークン数など上書き可
```

1 行形式（パイ区切り）: **`日時|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

### WMMA 利用状況の確認（`make wmma`）

Prefill 線形層は **hipBLAS / rocBLAS** ライブラリ経由であり、**`main.c` に WMMA / rocWMMA / MFMA を直接書いていません**（Prefill GEMM は hipBLAS に委譲）。**`make wmma`** で次を確認できます。

- **`main.c` / `qwen3-rocm` バイナリ**に WMMA 命令が無いこと（正常）
- **`wmma-probe`**（gfx11 向け校正用バイナリ）に WMMA があること（`llvm-objdump` 検出器の校正）
- **rocBLAS** バンドル ISA に WMMA があるか（0 件でも FMAC 経路の WARN がありうる）
- 任意: 実行時ログ **`Prefill linear: hipBLAS GemmEx`**、**`rocprofv3`** カーネル trace

```bash
cd qwen3-8b/gpu-rocm
make wmma                           # build + wmma-probe + チェック（MODEL 要）
make wmma WMMA_SKIP_RUN=1           # 静的チェックのみ（MODEL 不要）
make wmma WMMA_SKIP_ROCPROF=0       # rocprofv3 カーネル ISA も試行
make wmma-probe                     # 校正用 wmma-probe のみビルド
```

詳細は [`doc/design.md`](doc/design.md) の ROCm ビルド節と **`scripts/check_wmma.sh`** を参照。

## CUDA GPU 版（NVIDIA）

NVIDIA GPU と CUDA が使える環境向けです。**`qwen3-8b/Makefile` には `build.gpu-cuda` は無い**ため、用途に応じて **`gpu-cuda/`**（FP16）または **`gpu-cuda-nvfp4/`**（NVFP4）で単体ビルドします。プロンプトは **Prefill バッチ**、生成は **1 トークン Decode**、Attention は **Flash Attention**（GQA）です。

| ディレクトリ | ロード時の重み | 線形層 / KV キャッシュの実行 |
|--------------|----------------|------------------------------|
| **`gpu-cuda`** | CPU 逆量子化 → **FP16** → VRAM（ROCm 版と同趣旨） | FP16 GEMV カーネル。KV は **F32**（既定） |
| **`gpu-cuda`** + **`build.polarquant`** | 線形は FP16（上と同じ） | KV は **PolarQuant-R**（64 B/head）。Attention タイル読み出し時に F32 復号 |
| **`gpu-cuda-nvfp4`** | H2D 時に線形層を **NVFP4 キャッシュのみ** VRAM へ。**`token_embd`** のみ FP16 | **`fp4_qwen3_mm`** — decode / 短 Prefill は **FP4 GEMV**、長 Prefill は CUTLASS GEMM |
| **`gpu-cuda-nvfp4`** + **`build.polarquant`** | 線形は NVFP4 のみ（上と同じ） | 線形は FP4 GEMV/GEMM。KV は PolarQuant-R |

**`gpu-cuda-nvfp4`** は CUTLASS **NVFP4** を使い、**CUDA 13 + sm_120 系 GPU**（Blackwell / RTX 50 系等）向けです。**`fp4_gemm.cu`** / **`fp4_qwen3.cu`** は **C++17** 必須（CUTLASS）。**PolarQuant-R** は [arxiv:2502.02617](https://arxiv.org/abs/2502.02617) に基づく KV 圧縮で、**`head_dim=128` 固定**（Qwen3-VL-8B 向け）。F32 KV 比 **約 8×** の VRAM 削減（36 layer × 512 seq で ~144 MiB → ~18 MiB 概算）。NVFP4 + PolarQuant 同時では線形 FP16 分（8B 級で約 15 GiB 相当）と KV F32 分の両方を削減できます。

### ビルド・実行（FP16・汎用 GPU）

```bash
cd qwen3-8b/gpu-cuda
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

成功すると **`qwen3-gpu-cuda`** ができます。実 GPU アーキテクチャを直接指定する例:

```bash
make build CUDA_GENCODE=arch=compute_89,code=sm_89
```

### ビルド・実行（Blackwell + NVFP4）

CUDA 13 と CUTLASS が未導入なら、まず環境構築（要 root 相当）:

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make blackwell
```

既に CUDA 13 がある場合:

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make cutlass          # third_party/cutlass を clone（初回のみ）
make build            # sm_120a + BONSAI_FP4=1 + FA_BR=32
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

成功すると **`qwen3-gpu-cuda-nvfp4`** ができます。起動ログに **`Uploading weights to device (dequant -> NVFP4 linear layers)...`** と **`GPU: FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`** が出れば FP4 経路が有効です。

### ビルド（PolarQuant-R KV キャッシュ）

線形重みは FP16 のまま、KV キャッシュだけ PolarQuant-R で圧縮します（Blackwell 不要）。

```bash
cd qwen3-8b/gpu-cuda
make build.polarquant
make pq-test    # エンコード→復号のラウンドトリップ検証
```

起動ログに **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`** が出れば有効です。

### ビルド・実行（Blackwell + NVFP4 + PolarQuant-R）

線形重みを NVFP4、KV キャッシュを PolarQuant-R の両方で圧縮します。**CUDA 13 + CUTLASS + sm_120 系 GPU** が必要です。

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make cutlass                 # 初回のみ
make build.polarquant
make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

起動ログに次の **両方** が出れば有効です。

- **`Uploading weights to device (dequant -> NVFP4 linear layers)...`**
- **`GPU: FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`**
- **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**

### 実行（バイナリを直接）

FP16 版:

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、CUDAとは何かを初心者向けに説明してください。" \
  -n 64
```

NVFP4 版:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、CUDAとは何かを初心者向けに説明してください。" \
  -n 64
```

CUTLASS NVFP4 GEMM の単体確認（任意）: `cd qwen3-8b/gpu-cuda-nvfp4 && make fp4-test`（**`fp4_verify.cu`**。**Blackwell / sm_120a 向け**、RTX 50 系等。**`fp4_gemm.sm120a.o`** は **C++17 + `sm_120a` 固定**）。詳細は `doc/design.md` の CUDA 節を参照してください。

## AMD Ryzen AI XDNA2 NPU 版

AMD Ryzen AI（Phoenix / Hawk Point / Strix Point など）に内蔵されている XDNA2 NPU を使う版です。Linux カーネル付属の **`amdxdna` カーネルモジュール**を直叩きする実装で、XRT などの追加ユーザランドは不要です。

### 前提

1. Linux カーネル 6.10 以降（in-tree の `drivers/accel/amdxdna` が有効）。`lsmod | grep amdxdna` で確認。
2. `/dev/accel/accel0` が存在し、自分のユーザが `render` グループに所属していること。

```bash
ls -l /dev/accel/accel0
sudo usermod -aG render "$USER"   # 反映には再ログイン要
```

3. `<drm/drm.h>` UAPI ヘッダがインストールされていること（多くのディストロでは `linux-libc-dev` パッケージで入る）。

### ビルド

```bash
cd qwen3-8b
make build.xdna2
```

成功すると **`xdna2/qwen3-xdna2`** ができます。

### 実行

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON ツールチェイン**で生成した BF16 GEMV 制御コードバイナリ一式が必要です。`bf16-gemv-<n>x<d>.bin` という命名で `XDNA_GEMV_DIR` 配下に配置します。未配置の場合は OpenMP BF16 GEMV にフォールバックします（**NPU 経路とこの CPU フォールバックは bit-identical**）。

**`xdna2/xdna-gemv/kernels/`** には、名前とレイアウト用の **64 バイト・プレースホルダ**（マジック `GQF3XDNA`）が置いてあります。**実機の ERT には渡されません**（`--xdna-status` では `[STUB]`）。再生成はリポジトリルートで `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels`、または `cd qwen3-8b && make gen-xdna-kernels`。本番の NPU 用には MLIR-AIE 等で生成したバイナリに差し替えてください。

環境変数の例: `XDNA_GEMV_DIR`（制御コード検索ディレクトリ）、`XDNA_FORCE_CPU=1`（CPU 強制）、`XDNA_NUM_COL`（列数。`CREATE_HWCTX` が EINVAL になる環境では `XDNA_NUM_COL=1` を試す）。

```bash
# 強制的に CPU フォールバックで動かす場合
XDNA_FORCE_CPU=1 ./xdna2/qwen3-xdna2 Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# リポジトリ同梱プレースホルダ（qwen3-8b からの相対パス）。実 NPU ctrlcode ではない。
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status

# 制御コードが揃っているときは NPU 経路で実行（本物の .bin に差し替え後）
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

`make run.xdna2` も使えます。

```bash
make run.xdna2 PROMPT="日本語で短く説明してください。"
```

### XDNA2 + BFPX ホスト重み版（`xdna2-bfp16/qwen3-xdna2-bfpx`）

`xdna2-bfp16/main.c` は、**`xdna2/main.c` と同一の DRM ioctl** および **チャンク構成の BF16 GEMV（NPU 経路の枠組み）** を用います。一方で、密な行列レイアウトの重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 の係数）** へ変換し、ホストメモリ上にのみ保持します。GGUF への mmap は、この変換が終わってから解放します。NPU が使えないときの CPU 側のフォールバックでは **`mm_bfpx`** が用いられ、活性値は単精度浮動小数点数のまま、BFPX 形式の重みとの一般行列ベクトル積を計算します。**`xdna2/qwen3-xdna2` とビット単位で完全一致するとは限りません**。量子化に加えブロック近似の誤差があります。**GEMV で量子化 mmap から直接 BF16 へ展開する `xdna2/qwen3-xdna2`** とは経路も誤差の立ち方も異なるため、品質の優劣はケースによります。

```bash
cd qwen3-8b
make build.xdna2-bfp16
XDNA_FORCE_CPU=1 ./xdna2-bfp16/qwen3-xdna2-bfpx Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2-bfp16/qwen3-xdna2-bfpx \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
make run.xdna2-bfp16 PROMPT="日本語で短く説明してください。"
```

### 注意

- **`xdna2/qwen3-xdna2`**: 線形ウェイトは **mmap**（CPU OpenMP 版と同様）。GEMV に使う単一 BF16 スクラッチは **語彙×次元クラスの最大行列**サイズになり得るので、モデルサイズと **VRAM／DRAM に余裕**が必要になる場合があります。恒久の「全レイヤー BF16 二重複製」は行いません。RAM 不足では従来通りプロセスや mmap が失敗し得ます。
- **`xdna2-bfp16/qwen3-xdna2-bfpx`**: 推論中は BFPX とノルム用 F32 が中心で mmap を早めに離せる一方、**変換中**は GGUF mmap とフルテンソル用の一時バッファなどで **ピークメモリが大きくなります**。
- NPU 側で実行する場合は AIE 列を予約するため、同時に動いているほかの NPU ワークロード（Windows Studio Effects 等）と競合する可能性があります。

## よく使うオプション

| オプション | 例 | 意味 |
|---|---|---|
| `-p` | `-p "Hello"` | 入力プロンプト |
| `-n` | `-n 64` | 最大生成トークン数 |
| `-t` | `-t 0.7` | 温度。低いほど堅め、高いほどランダム |
| `-k` | `-k 0.9` | Top-p。候補を上位確率に絞る |
| `-s` | `-s 1234` | 乱数シード |
| `-l` | `-l 512` | 最大シーケンス長 |

まずは次のように短めに試すのがおすすめです。

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

慣れてきたら `-n` を増やします。

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で詩を書いてください。" -n 128
```

## 生成を安定させたいとき

同じ入力で結果を比較したい場合は、温度を下げたり seed を固定します。

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "1文で説明してください: GGUFとは？" \
  -n 32 \
  -t 0.2 \
  -s 42
```

完全に同じ結果になるかは、CPU 版と GPU 版、サンプリング経路、GPU の実行環境によって変わることがあります。比較するときは、同じ実行ファイル・同じモデル・同じオプションで試してください。

## 片付け

ビルド生成物を消すには:

```bash
cd qwen3-8b
make clean
```

削除される主なファイル:

- `cpu/qwen3-cpu`
- `cpu-multicore/qwen3-cpu-omp`
- `cpu-blas/qwen3-cpu-blas`
- `gpu-rocm/qwen3-rocm`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`

**CUDA 版**（`gpu-cuda/qwen3-gpu-cuda`、`gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4` 等）は集約 `make clean` の対象外です。消す場合:

```bash
cd qwen3-8b/gpu-cuda && make clean
cd qwen3-8b/gpu-cuda-nvfp4 && make clean
```

モデルファイルは `make clean` では削除されません。

## よくあるトラブル

### `No such file or directory` と出る

モデルファイルの場所が間違っている可能性があります。

```bash
ls -lh qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

見つからない場合は、モデルを `qwen3-8b/` に置くか、実行時に絶対パスを指定してください。

```bash
./cpu/qwen3-cpu /data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

### CPU 版が遅い

正常です。8B 級モデルは CPU だけで動かすには重いです。まずは `-n 1` や `-n 4` で確認してください。

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

速度が必要なら AMD GPU では `./gpu-rocm/qwen3-rocm`、NVIDIA GPU では `gpu-cuda/qwen3-gpu-cuda`（FP16）または `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4`（Blackwell NVFP4）を使ってください。CPU のみの場合は **`cpu-blas/qwen3-cpu-blas`**（OpenBLAS + Q8_K 量子化 GEMV + AVX2 dot + 層内 Q8 共有 + RoPE キャッシュ + prefill LM スキップ + greedy argmax）の方が **`cpu-multicore`** より速くなることが多いです。**greedy（`-t 0`）** では decode の LM head がさらに軽くなります。

### `cpu-blas` のビルドが失敗する／`cblas.h` が見つからない

OpenBLAS 開発パッケージを入れ、必要なら `CPPFLAGS` でヘッダパスを指定してください（上記「OpenBLAS 版を使う場合」参照）。

### `cpu-blas` の出力が意味不明（同じ文字の連打など）

**`-ffast-math`** を付けてビルドすると IQ / Q8_K 量子化内積で数値が崩れます。リポジトリ同梱の `cpu-blas/Makefile` では無効化済みです。手元で CFLAGS を上書きしている場合は外してください。

### `nvcc` が見つからない／`nvlink` エラー

CUDA Toolkit の `bin` を `PATH` に通すか、各ディレクトリの `Makefile` の `CUDA_HOME` を確認してください。

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

PTX のみのビルドで極端に遅い場合は、実機の `sm_XX` を `CUDA_GENCODE` で指定して再ビルドしてください。

### NVFP4 ビルドが失敗する／`NVFP4 quantize failed`

**`gpu-cuda-nvfp4`** で **`sm_120a`** 向けビルドか、CUDA 13 + **`make cutlass`** 済みかを確認してください。汎用 GPU では **`gpu-cuda`** で **`make build`**（FP16）を使います。apt **CUDA 11** と **CUDA 13** が混在している場合は **`make blackwell`** で整理するか、手動で 11.x を除去してください。

### `make fp4-test` が FAIL する／`Arch conditional MMA instruction... Aborting`

**`fp4_gemm`** を **`compute_86` PTX** のみでビルドした古い **`fp4_gemm.o`** を使っている可能性があります。**`gpu-cuda-nvfp4`** で **`make clean`** → **`make fp4-test`** を再実行してください（**`fp4_gemm.sm120a.o`** は **`sm_120a` + C++17** 固定）。**Blackwell GPU 必須**です。

### NVFP4 で decode が極端に遅い

古いバイナリが M=1 を CUTLASS M=128 パディング経路に落としている場合があります。最新 **`fp4_gemv_cached`** 入り **`gpu-cuda-nvfp4`** ビルドか確認し、起動ログに **`GEMM M>=128, GEMV decode`** が出ることを確認してください。

### `gpu-cuda-nvfp4` の PolarQuant ビルドが遅い／`Killed`（OOM）

**`kernels.cu`** を **`sm_120a` + FP4 + PolarQuant** でコンパイルするとメモリを多く使います。RAM が不足すると **`Error 137`** で落ちることがあります。対処: スワップを増やす、並列ビルドを止める、または段階的に **`make cutlass`** → **`make build.polarquant`** を再実行してください。

### PolarQuant が有効にならない

起動ログに **`PolarQuant-R: KV cache enabled`** が無い場合、**`build.polarquant`** でビルドしたバイナリか確認してください（`gpu-cuda` または `gpu-cuda-nvfp4`）。**`head_dim=128`** 以外のモデルでは PolarQuant は無効化されます。

### `hipcc` が見つからない

ROCm の場所を確認してください。

```bash
ls /opt/rocm/bin/hipcc
```

別の場所にある場合:

```bash
make build.gpu-rocm ROCM=/path/to/rocm
```

### GPU_ARCH が合わない / 検出に失敗する

`GPU_ARCH` は実機の GPU ISA に合わせる必要があります。通常は `rocminfo` から自動検出されますが、検出に失敗したり別の GPU 向けにビルドしたい場合は手動指定してください。

```bash
rocminfo | awk '/^  Name:/ { n=$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }'
make build.gpu-rocm GPU_ARCH=gfx1100
```

### ROCm Prefill が遅い（~30 tok/s 程度）

起動ログに **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** があるか確認してください。無い場合や prefill が極端に遅い場合は **`make -C gpu-rocm clean build`** で再ビルドし、**`-lhipblas -lrocblas`** がリンクされているか確認します。詳細は上記 **「Prefill 高速化（概要）」** と **`doc/design.md`** の **「ROCm Prefill 高速化の詳細（3 段階）」** を参照。

### **`undefined reference to hipblas*`** / hipBLAS リンク失敗

**`$(ROCM)/lib`** に **`libhipblas.so`** / **`librocblas.so`** があるか確認してください。ROCm の再インストールまたは **`ROCM=`** パスの修正が必要な場合があります。

### **`make wmma` が FAIL**

**`qwen3-rocm` バイナリに WMMA 命令が含まれる**、**hipBLAS 経路が報告されない**、**`wmma-probe` 校正失敗** 等が考えられます。まず **`make wmma WMMA_SKIP_RUN=1`** で静的チェックのみ実行。**`LLVM_OBJDUMP=$(ROCM)/llvm/bin/llvm-objdump`** を明示。MODEL 未配置時は **`WMMA_SKIP_RUN=1`** を使ってください。

### `/dev/accel/accel0` は開けるが `CREATE_HWCTX` が EINVAL

ドライバが列数・タイル数の組み合わせを拒否していることがあります。`XDNA_NUM_COL=1` を試し、`dmesg` の `amdxdna` メッセージを確認してください（詳細は `doc/design.md` のトラブルシュート）。

### `sha256sum -c` が失敗する

ファイル名または中身が、このリポジトリの想定と違います。次を確認してください。

- GGUF ファイル名が `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` になっているか
- ダウンロードが途中で壊れていないか
- 別量子化のモデルを置いていないか

別モデルを使う場合は、ハッシュ確認は一致しなくて当然です。その場合でも実装が対応するメタデータ・テンソル構造である必要があります。

## 実装を読みたい人へ

最初に読むなら、次の順番がおすすめです。

1. `README.md` / `README.en.md`  
   まずビルドと実行を成功させる。

2. `doc/design.md`  
   全体の設計、量子化、Qwen3固有処理を把握する。

3. `qwen3-8b/cpu/main.c`  
   CPU 版で、GGUF 読み込みから 1 トークン生成までを追う。

4. `qwen3-8b/cpu-multicore/main.c`  
   OpenMP による並列化箇所を見る。

5. `qwen3-8b/cpu-blas/main.c`  
   OpenBLAS（`cblas_sgemv`）による F32 GEMV と Attention 集約。量子化 GEMV は Q8_K 活性化 + 全型 AVX2 整数内積（層内 Q8 共有）。RoPE キャッシュ、**`lm_mode`**（prefill LM スキップ / greedy **`mm_argmax_row`**）、F16 emb F16C。詳細は **`doc/design.md`** の **「`cpu-blas`：Q8_K 活性化 GEMV」** 節。

6. `qwen3-8b/gpu-rocm/main.c`  
   GPU メモリ、HIP カーネル、GPU サンプリング。**Prefill** は **`forward_prefill_gpu`**（**hipBLAS GemmEx** + バッチ Attention/Norm 等。**`attn_flash_prefill_kernel`** は **`(int)threadIdx.x`** による符号付き比較）。**Decode** は **`forward_gpu`**。**Prefill progress bar** とスループット要約。**`gpu-rocm/Makefile`** の **`make log` / `make log.push`** ベンチ履歴、**`make wmma`**（**`wmma_probe.c`** / **`scripts/check_wmma.sh`**）で hipBLAS 経路確認。Prefill 高速化の詳細は **`doc/design.md`** の **「ROCm Prefill 高速化の詳細（3 段階）」**。

7. `qwen3-8b/gpu-cuda/main.c` / `kernels.cu` / `polarquant.cu`  
   CUDA FP16 版の Prefill／Decode、Flash Attention。任意で **`build.polarquant`**: PolarQuant-R KV（**`pq_decode_head`** でタイル復号）。

8. `qwen3-8b/gpu-cuda-nvfp4/fp4_qwen3.cu` / `fp4_gemm.cu`  
   Blackwell NVFP4 版。H2D 時 NVFP4 ロード、**`fp4_gemv_cached`**（decode）と **`fp4_qwen3_mm`**（GEMM/GEMV 分岐）。**`fp4_*` は C++17 + `sm_120a` 固定**（CUTLASS）。共有ソースは **`../gpu-cuda/`** を参照。

9. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c`  
   `amdxdna` ioctl、`ERT_START_NPU`、`launch_mm_bf16`、CPU フォールバック。mmap スクラッチ方式は **`load_weights_xdna`／`weight_prepare_bf16`／単一 `w_scratch_bo`**。BFPX 版は **`bfpx_convert_weight_2d`** と mmap 解放パス。

## 高度な機能（マルチターン対話・Thinking モード）について

Qwen3 ファミリー（QwQ など reasoning 系を含む）では、単発の `-p "..."` より **ChatML テンプレート**と **Thinking モード**が公式スタックの前提です。本リポジトリは **デコーダ forward とサンプリング**だけを C で再現しており、以下の高度機能は **未実装**です。利用・改造時は公式仕様との差分を意識してください。

テンプレート全般の背景: [Qwen3 公式ブログ](https://qwenlm.github.io/blog/qwen3/)、[The 4 Things Qwen-3’s Chat Template Teaches Us（Hugging Face Blog）](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)、[Qwen/Qwen3-8B モデルカード](https://huggingface.co/Qwen/Qwen3-8B)。

### マルチターン（複数回の会話）

**公式の想定** — ChatML で system / user / assistant をターン順に並べ、KV キャッシュまたは再 prefill で文脈を引き継ぎます。function calling では、過去の assistant 発話・tool 結果・（必要に応じた）reasoning 区間も次ターンへ渡します。

**本リポジトリの現状** — 各 `main.c` の `chat_encode` は **1 ターン固定**です。

```text
<|im_start|>system … <|im_end|>
<|im_start|>user\n{ -p で渡した文字列 }<|im_end|>
<|im_start|>assistant\n
```

CLI から過去ターンを渡す手段はなく、**プロセス間の会話状態も保持しません**（prefill は毎回ゼロから）。マルチターンに近づけるには、次のいずれかが必要です。

1. 履歴を手で ChatML に組み立て、`-p` に渡す
2. `chat_encode` を拡張し、ターン列を受け取る
3. 生成済み KV を次推論へ再利用する（現状未対応）

`-l`（最大シーケンス長）を超える履歴は切り詰めまたは要約が必要です。

**参考（技術詳細）**

- [Transformers — Chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating) — `messages` から ChatML 文字列への変換、`apply_chat_template`
- [Function Calling（Qwen 公式ドキュメント）](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — Hermes 形式、assistant / tool ロールの連結
- [Core concepts — Tool Calling（Qwen）](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html) — マルチターン／マルチステップ tool calling のテンプレート例

### Thinking モード（思考プロセスを伴う推論）

**公式の想定** — Qwen3 の **ハイブリッド thinking**（DeepSeek-R1 / QwQ 系の「考えてから答える」構成）では、**hard switch**（`apply_chat_template(..., enable_thinking=True/False)` や API の `enable_thinking`）と **soft switch**（user メッセージ末尾の `/think` / `/no_think`。マルチターンでは最新指示が優先）で切り替えます。有効時は assistant 出力先頭に **thinking ブロック**（テンプレートが挿入する reasoning 区間）が付き、その後に最終回答が続きます。無効時は **空の thinking ブロック**を挿入し、即答寄りに誘導します。これらのマーカーは ChatML 特殊トークン（`<|im_start|>` 等）とは別に、通常テキストとしてトークン化されます。

**本リポジトリの現状** — 次を **行っていません**。

- `enable_thinking` に相当する **生成プロンプト制御**（assistant 直前への空 thinking ブロック挿入等）
- 生成結果から **thinking 区間と最終回答の分離・非表示**（`print_tok` は ChatML 特殊 ID のみ抑制し、reasoning 文字列はそのまま stdout へ出る）
- **`thinking_budget`** や reasoning 専用ストリームなど、API 側の thinking 付帯パラメータ

thinking 対応 GGUF をそのまま動かすと、**reasoning テキストが端末に混ざる**、または **テンプレート不一致で品質が落ちる**ことがあります。正しく扱うには GGUF メタデータの `tokenizer.chat_template` に沿った **プロンプト組み立て**と、出力側の **thinking 区間パース**を `chat_encode` / 生成ループへ追加する必要があります。

**参考（技術詳細）**

- [Quickstart — Thinking & Non-Thinking Mode（Qwen）](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html) — hard / soft switch、`thinking_budget`、推奨サンプリング
- [Transformers 推論ガイド（Qwen）](https://qwen.readthedocs.io/en/latest/inference/transformers.html) — thinking 切替、`reasoning_content` のパース例
- [Thinking（Qwen Cloud）](https://docs.qwencloud.com/developer-guides/text-generation/thinking) — API の `enable_thinking` / `thinking_budget` / `reasoning_content`
- [vLLM デプロイ（Qwen）](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) — `chat_template_kwargs.enable_thinking`、reasoning parser

### 本リポジトリの位置づけ（まとめ）

| 機能 | Qwen3 ファミリー（公式想定） | 本リポジトリ（現状） |
|---|---|---|
| ChatML 1 ターン（system + user + assistant 開始） | ○ | ○（`-p` 固定 system 文付き） |
| マルチターン履歴 | ○ | ×（手動で `-p` に ChatML を埋め込む必要） |
| KV / 会話状態の保持 | ○（フレームワーク側） | ×（1 回の実行内のみ） |
| Thinking オン／オフ | ○（`enable_thinking` 等） | × |
| `/think`・`/no_think` | ○（ハイブリッドモデル） | ×（未解釈） |
| 思考ブロックのフィルタ表示 | ○（API / UI） | × |

**テキスト 1 ターン生成**の参照実装としては十分です。**ChatGPT / Qwen API 相当のマルチターン対話や Thinking UI** が必要なら、上記を拡張するか、vLLM・llama.cpp・Transformers など既存ランタイムの利用を検討してください。

## このリポジトリで扱わないもの

- 学習、ファインチューニング
- バッチ推論の最適化
- 画像入力
- **マルチターン対話の組み込み CLI**（履歴管理・KV 再利用・公式 chat template の完全再現）
- **Thinking モードの制御・思考ブロックの分離表示**（`enable_thinking` / `/think` / `/no_think` 等）
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- 公式実装との完全な数値一致保証

目的は、Qwen3系GGUFのテキスト推論を C／HIP／CUDA で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- 設計仕様: `doc/design.md`
- 変更履歴: `doc/ChangeLog.md`

困ったときは、まず `qwen3-8b/Makefile` のターゲット名と、実行時に渡しているモデルパスを確認してください。ビルドと実行の大半の問題は、この 2 つの不一致から起きます。
