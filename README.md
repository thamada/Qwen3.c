# Qwen3.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、**ライブラリに依存せず、単一の C言語ソースから Qwen3系モデルを直接動かす推論実装**です。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。** 推論は **標準Cと `libm`** を中心に、`qwen3-8b/` 内の単一ソースで完結します。AMD GPU 版は **ROCm/HIP**（`hipcc`）、NVIDIA GPU 版は **CUDA Toolkit**（`nvcc`）、CPU 並列は **OpenMP**、XDNA2 NPU 版は **Linux カーネルの `amdxdna` DRM ioctl（UAPI）** を直接叩く構成であり、Pythonランタイムや `torch` に依存するレイヤはありません。

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

Qwen3系GGUFモデルを、**Cの単一ソース群**から直接動かす小さな推論実装です。実行経路は **CPU／OpenMP／ROCm HIP（AMD GPU）／CUDA（NVIDIA GPU）／AMD Ryzen AI XDNA2 NPU（`amdxdna` DRM ioctl の直叩き）**と選べます。

このリポジトリは **Qwen3-VL-8B-Instruct のテキストデコーダ**を対象にしています。画像入力や Vision エンコーダは扱わず、プロンプト文字列を入力してテキストを生成する用途に絞っています。

## まず何ができるのか

このリポジトリでは、`qwen3-8b/` の中にある Cソースをビルドして、次の実行方法を試せます。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | 仕組みを追う、最小構成で動かす |
| CPU OpenMP 並列 | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | CPU で少しでも速く試す |
| ROCm/HIP GPU | `qwen3-8b/gpu-rocm/main.c` | `gpu-rocm/qwen3-rocm` | AMD GPU で実用的な速度を狙う |
| CUDA GPU | `qwen3-8b/gpu-cuda/main.c` + `kernels.cu` | `gpu-cuda/qwen3-gpu-cuda` | NVIDIA GPU。Prefill バッチ + Flash Attention。**`build.no-fp4`**: 全線形層 FP16 VRAM。**`build.fp4`**（Blackwell）: 線形層は H2D 時 **NVFP4 のみ**（`fp4_qwen3` + CUTLASS）、decode は **FP4 GEMV**、長 Prefill は Tensor Core GEMM。**`build.polarquant`**: KV キャッシュ **PolarQuant-R**（64 B/head、~8× 圧縮）。埋め込みのみ FP16 VRAM。集約 `Makefile` 外 |
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
    ├── gpu-rocm/
    │   ├── Makefile
    │   └── main.c
    ├── gpu-cuda/
    │   ├── Makefile
    │   ├── main.c
    │   ├── kernels.cu
    │   ├── gpu.h
    │   ├── fp4_gemm.cu / fp4_qwen3.cu / fp4_verify.cu  （Blackwell FP4 時）
    │   ├── polarquant.cu / polarquant_kernels.cuh / polarquant_verify.cu  （PolarQuant-R KV 時）
    │   └── third_party/cutlass/  （make cutlass で取得）
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

### ROCm/HIP 版を使う場合

AMD GPU と ROCm が必要です。`Makefile` は既定で ROCm を `/opt/rocm` にあるものとして扱います。

確認例:

```bash
/opt/rocm/bin/hipcc --version
rocminfo | grep -m 1 gfx
```

`rocminfo` で表示される `gfx1201` などの値を、ビルド時の `GPU_ARCH` に指定します。

### CUDA 版を使う場合

NVIDIA GPU と **CUDA Toolkit**（`nvcc`・`libcudart`）が必要です。集約 `qwen3-8b/Makefile` には CUDA ターゲットは無く、**`qwen3-8b/gpu-cuda/`** で単体ビルドします。

確認例:

```bash
nvcc --version
nvidia-smi
```

`nvcc` / `nvlink` は **CUDA の `bin` ディレクトリ**を `PATH` に通してください（`/usr/local/bin/nvcc` のみだとリンクに失敗することがあります）。

| 用途 | コマンド（`cd qwen3-8b/gpu-cuda`） |
|------|-----------------------------------|
| **FP16 のみ**（Ampere/Ada 等・PTX 可） | `make build.no-fp4` または `make run.no-fp4` |
| **Blackwell NVFP4**（RTX 50 系等） | `make build.fp4` / 既定の `make run`（内部で `build.fp4`） |
| **PolarQuant-R KV キャッシュ**（任意 GPU・FP16 線形） | `make build.polarquant` |
| **NVFP4 + PolarQuant 同時** | `make build BONSAI_FP4=1 BONSAI_POLARQUANT=1` |
| CUDA 13 の導入から一式 | `make blackwell`（apt CUDA 11 除去 → CUDA 13 → CUTLASS → `build.fp4`） |
| PolarQuant ラウンドトリップ検証 | `make pq-test` |

FP16 ビルドの既定は PTX（`compute_86`）。実 GPU 向けには `CUDA_GENCODE=arch=compute_XX,code=sm_XX` を指定します。**`gpu-cuda/Makefile` の既定ターゲット `run` は `build.fp4` を呼ぶ**ため、Blackwell 以外では **`make run.no-fp4`** を使ってください。

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
├── gpu-rocm/ …
├── gpu-cuda/ …
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

## ROCm/HIP GPU 版

AMD GPU と ROCm が使える環境では、こちらが本命です。

### GPU_ARCH を確認する

```bash
rocminfo | grep -m 1 gfx
```

例として `gfx1201` と表示されたら、ビルド時に `GPU_ARCH=gfx1201` を指定します。

### ビルド

```bash
cd qwen3-8b
make build.gpu-rocm GPU_ARCH=gfx1201
```

ROCm が `/opt/rocm` 以外にある場合:

```bash
make build.gpu-rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

成功すると **`gpu-rocm/qwen3-rocm`** ができます。

### 実行

```bash
./gpu-rocm/qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、ROCmとは何かを初心者向けに説明してください。" \
  -n 64
```

`Makefile` の `run.gpu-rocm` を使う場合:

```bash
make run.gpu-rocm GPU_ARCH=gfx1201 PROMPT="日本語で短く説明してください。"
```

## CUDA GPU 版（NVIDIA）

NVIDIA GPU と CUDA が使える環境向けです。**`qwen3-8b/Makefile` には `build.gpu-cuda` は無い**ため、`gpu-cuda/` で単体ビルドします。プロンプトは **Prefill バッチ**、生成は **1 トークン Decode**、Attention は **Flash Attention**（GQA）です。

| ビルド | ロード時の重み | 線形層 / KV キャッシュの実行 |
|--------|----------------|------------------------------|
| **`build.no-fp4`** | CPU 逆量子化 → **FP16** → VRAM（ROCm 版と同趣旨） | FP16 GEMV カーネル。KV は **F32** |
| **`build.fp4`**（Blackwell） | H2D 時に線形層を **NVFP4 キャッシュのみ** VRAM へ。**`token_embd`** のみ FP16 | **`fp4_qwen3_mm`** — decode / 短 Prefill は **FP4 GEMV**、長 Prefill は CUTLASS GEMM |
| **`build.polarquant`** | 線形は FP16（`build.no-fp4` 同趣旨） | KV は **PolarQuant-R**（64 B/head）。Attention タイル読み出し時に F32 復号 |

**Blackwell（sm_120 系）** 向け **`build.fp4`** は CUTLASS **NVFP4** を使います。Ampere/Ada 等では **`build.no-fp4` / `make run.no-fp4`** のみを想定してください。**PolarQuant-R** は [arxiv:2502.02617](https://arxiv.org/abs/2502.02617) に基づく KV 圧縮で、**`head_dim=128` 固定**（Qwen3-VL-8B 向け）。F32 KV 比 **約 8×** の VRAM 削減（36 layer × 512 seq で ~144 MiB → ~18 MiB 概算）。

### ビルド（FP16 のみ・汎用 GPU）

```bash
cd qwen3-8b/gpu-cuda
make build.no-fp4
```

成功すると **`qwen3-gpu-cuda`** ができます。実 GPU アーキテクチャを直接指定する例:

```bash
make build.no-fp4 CUDA_GENCODE=arch=compute_89,code=sm_89
```

### ビルド・実行（Blackwell + NVFP4）

CUDA 13 と CUTLASS が未導入なら、まず環境構築（要 root 相当）:

```bash
cd qwen3-8b/gpu-cuda
make blackwell
```

既に CUDA 13 がある場合:

```bash
make cutlass          # third_party/cutlass を clone（初回のみ）
make build.fp4        # sm_120a + BONSAI_FP4=1 + FA_BR=32
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

起動ログに **`Uploading weights to device (dequant -> NVFP4 linear layers)...`** と **`GPU: FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`** が出れば FP4 経路が有効です。

**注意**: `gpu-cuda/Makefile` の **既定ターゲットは `run` → `build.fp4`** です。Blackwell 以外の GPU では **`make run.no-fp4`** を使ってください。

### ビルド（PolarQuant-R KV キャッシュ）

線形重みは FP16 のまま、KV キャッシュだけ PolarQuant-R で圧縮します（Blackwell 不要）。

```bash
cd qwen3-8b/gpu-cuda
make build.polarquant
make pq-test    # エンコード→復号のラウンドトリップ検証
```

起動ログに **`PolarQuant-R: KV cache enabled (64 B/head, ~8x vs F32)`** が出れば有効です。NVFP4 と併用する場合:

```bash
make build BONSAI_FP4=1 BONSAI_POLARQUANT=1
```

### 実行（バイナリを直接）

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、CUDAとは何かを初心者向けに説明してください。" \
  -n 64
```

FP16 のみでビルド済みのとき:

```bash
make run.no-fp4 MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="日本語で短く説明してください。"
```

CUTLASS NVFP4 GEMM の単体確認（任意）: `make fp4-test`（**`fp4_verify.cu`** をビルドして実行）。詳細は `doc/design.md` の CUDA 節を参照してください。

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
- `gpu-rocm/qwen3-rocm`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`

**CUDA 版**（`gpu-cuda/qwen3-gpu-cuda` 等）は集約 `make clean` の対象外です。消す場合:

```bash
cd qwen3-8b/gpu-cuda && make clean
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

速度が必要なら AMD GPU では `./gpu-rocm/qwen3-rocm`、NVIDIA GPU では `gpu-cuda/qwen3-gpu-cuda` を使ってください。

### `nvcc` が見つからない／`nvlink` エラー

CUDA Toolkit の `bin` を `PATH` に通すか、`gpu-cuda/Makefile` の `CUDA_HOME` を確認してください。

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

PTX のみのビルドで極端に遅い場合は、実機の `sm_XX` を `CUDA_GENCODE` で指定して再ビルドしてください。

### `build.fp4` が失敗する／`NVFP4 quantize failed`

**`sm_120a`** 向けビルドか、CUDA 13 + **`make cutlass`** 済みかを確認してください。汎用 GPU では **`make build.no-fp4`** を使います。

### `hipcc` が見つからない

ROCm の場所を確認してください。

```bash
ls /opt/rocm/bin/hipcc
```

別の場所にある場合:

```bash
make build.gpu-rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

### GPU_ARCH が合わない

`GPU_ARCH` は実機の GPU ISA に合わせる必要があります。

```bash
rocminfo | grep -m 1 gfx
```

表示された値を使います。

```bash
make build.gpu-rocm GPU_ARCH=gfx1100
```

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

5. `qwen3-8b/gpu-rocm/main.c`  
   GPU メモリ、HIP カーネル、GPU サンプリングの流れを見る。

6. `qwen3-8b/gpu-cuda/main.c` / `kernels.cu` / `fp4_qwen3.cu` / `fp4_gemm.cu` / `polarquant.cu`  
   CUDA 版の Prefill／Decode、Flash Attention。**`build.no-fp4`**: FP16 GEMV。**`build.fp4`**: H2D 時 NVFP4 ロード、**`fp4_gemv_cached`**（decode）と **`fp4_qwen3_mm`**（GEMM/GEMV 分岐）。**`build.polarquant`**: PolarQuant-R KV（**`pq_decode_head`** でタイル復号）。

7. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c`  
   `amdxdna` ioctl、`ERT_START_NPU`、`launch_mm_bf16`、CPU フォールバック。mmap スクラッチ方式は **`load_weights_xdna`／`weight_prepare_bf16`／単一 `w_scratch_bo`**。BFPX 版は **`bfpx_convert_weight_2d`** と mmap 解放パス。

## このリポジトリで扱わないもの

- 学習、ファインチューニング
- バッチ推論の最適化
- 画像入力
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- 公式実装との完全な数値一致保証

目的は、Qwen3系GGUFのテキスト推論を C／HIP／CUDA で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- 設計仕様: `doc/design.md`
- 変更履歴: `doc/ChangeLog.md`

困ったときは、まず `qwen3-8b/Makefile` のターゲット名と、実行時に渡しているモデルパスを確認してください。ビルドと実行の大半の問題は、この 2 つの不一致から起きます。
