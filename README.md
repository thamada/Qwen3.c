# Qwen3.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、**ライブラリに依存せず、単一の C言語ソースから Qwen3系モデルを直接動かす推論実装**です。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。**  
推論の基準となる実装は **標準Cと `libm`** のみで、`qwen3-8b/cpu/main.c` から **CPU 単スレッド**の実行ファイル（`cpu/qwen3-cpu`）をビルドします。  
より速い検証向けに、同じ GGUF に対応した **OpenMP マルチスレッド**版を `qwen3-8b/cpu-multicore/main.c` から **`qwen3-cpu-omp`** として別ビルドできます（ランタイムは **標準C + `libm` + OpenMP ランタイム**）。  
さらに **`qwen3-8b/cpu-blas/`** では **OpenMP + OpenBLAS** と **Q8_K 活性化 + 全型 AVX2 整数内積**で **`qwen3-cpu-blas`** をビルドできます（**標準C + `libm` + OpenMP + OpenBLAS**）。  
**ROCm/HIP**（AMD GPU）・**Vulkan compute**（ベンダー非依存 GPU）・**CUDA**（NVIDIA GPU）・**XDNA2 NPU**（`amdxdna` ioctl）向けは **付録**として README 末尾にまとめています（本文の主眼は CPU 3 バリアント）。

### なぜライブラリ非依存なのか

一般的な LLM推論は PyTorch などの高レベルな機械学習フレームワークを利用することで、短いコードで高速に実行できます。一方で、その構成では **計算手順、メモリ配置、アライメント、量子化レイアウト**といった低レベルの詳細が、フレームワークやランタイムの内部に隠れがちです。

本リポジトリでは、あえてその層に依存せず、**GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを Cのコードパスとして明示する**ことを重視しています。これは既存フレームワークを置き換えるためではなく、推論処理の実体を観察し、検証し、必要に応じて変更できる形で保持するためです。

この方針には、次の意義があります。

- **理解可能性**: モデルファイルから何を読み、どのバッファに置き、どの順序で計算しているかを、ソースコードと `doc/design.md` から直接追跡できる。
- **依存関係の単純化**: Python環境や大規模な機械学習スタックを前提にせず、基本的な Cコンパイラと必要最小限の実行環境で動作経路を確認できる。
- **実験の自由度**: 量子化形式、メモリ表現（例: BFPX）、CPU/GPU/NPU への処理分担、`/dev/accel` への直接アクセスなど、フレームワークの抽象化に制約されやすい領域を個別に試せる。
- **参照実装としての価値**: 「最小限の構成で Qwen3系デコーダ推論がどのように成立するか」を示し、既存スタックとの比較や実装検証の基準にできる。

したがって、この実装は最高性能や機能網羅を第一目的とするものではありません。主眼は、LLM推論の仕組みをブラックボックスにせず、開発者が実装の細部を把握しながら改造できる状態に置くことです。

このリポジトリは **Qwen3-VL-8B-Instruct のテキストデコーダ**を対象にしています。画像入力や Vision エンコーダは扱わず、プロンプト文字列を入力してテキストを生成する用途に絞っています。

## まず何ができるのか

**CPU 単スレッド**版、その **OpenMP** 並列版、**OpenMP + OpenBLAS** 最適化版の 3 通りがあります。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `qwen3-8b/cpu/main.c` | `cpu/qwen3-cpu` | 仕組みを追う、最小構成で動かす。**Prefill progress bar** とスループット要約を stderr に出力 |
| CPU OpenMP 並列 | `qwen3-8b/cpu-multicore/main.c` | `cpu-multicore/qwen3-cpu-omp` | CPU で少しでも速く試す |
| CPU OpenMP + OpenBLAS | `qwen3-8b/cpu-blas/main.c` | `cpu-blas/qwen3-cpu-blas` | F32 GEMV と Attention を BLAS 化。量子化 GEMV は **Q8_K 活性化 + 全型 AVX2 整数内積**（層内 Q8 共有）。**RoPE キャッシュ**、prefill 中 **LM head スキップ**、greedy 時 **`mm_argmax_row`**。**F16 埋め込み F16C**。stderr に **Prefill progress bar** |

8B 級モデルの CPU 実行は非常に重いです。最初の動作確認としては CPU でも構いませんが、実用的な生成速度が必要な場合は **`cpu-blas`** を推奨します。AMD GPU・NVIDIA GPU・XDNA2 NPU 向けの高速経路は **付録**（README 末尾）にあります。

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
    ├── gpu-rocm/          # 付録（README 末尾）
    │   ├── Makefile
    │   ├── main.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   ├── wmma_probe.c          （`make wmma-probe` — WMMA 検出器校正）
    │   └── scripts/
    │       └── check_wmma.sh     （`make wmma`）
    ├── gpu-vulkan/        # 付録（README 末尾・Vulkan compute）
    │   ├── Makefile
    │   ├── main.c
    │   ├── gpu.h
    │   ├── vk_context.c/h / vk_alloc.c/h / vk_pipeline.c/h / vk_kernels.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   └── shaders/              （GLSL compute → `make` で `.spv` 生成）
    ├── gpu-cuda/          # 付録（README 末尾・FP16）
    │   ├── Makefile
    │   ├── main.c
    │   ├── fp16_cache.h / fp16_cache_io.c
    │   ├── kernels.cu
    │   ├── gpu.h
    │   └── polarquant.cu / polarquant_kernels.cuh / polarquant_verify.cu  （PolarQuant-R KV 時）
    ├── gpu-cuda-nvfp4/    # 付録（README 末尾・NVFP4）
    │   ├── Makefile
    │   ├── fp4_cache.h / fp4_cache_io.c / fp4_gemm.cu / fp4_qwen3.cu / fp4_verify.cu
    │   └── third_party/cutlass/  （make cutlass で取得）
    │   （main.c / kernels.cu / gpu.h / polarquant.* は ../gpu-cuda/ を参照）
    ├── xdna2/             # 付録（README 末尾）
    │   ├── Makefile
    │   ├── main.c
    │   └── xdna-gemv/
    │       ├── README.md
    │       ├── gen-xdna-gemv-stubs.py
    │       ├── kernels/
    │       └── toolchain/
    ├── xdna2-bfp16/       # 付録（README 末尾・BFPX）
    │   ├── Makefile
    │   └── main.c
    └── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

推論コードは **`qwen3-8b/cpu/`**（参照・単スレッド）が基準です。並列版は **`qwen3-8b/cpu-multicore/`**、CPU 最適化版は **`qwen3-8b/cpu-blas/`** です。GPU 向けは **`gpu-rocm`**（AMD・ROCm/HIP）、**`gpu-vulkan`**（AMD/NVIDIA/Intel 等・Vulkan compute）、**`gpu-cuda`**（NVIDIA・FP16）、**`gpu-cuda-nvfp4`**（NVIDIA・NVFP4）が付録としてあります。XDNA2 NPU 向けは **`xdna2`** / **`xdna2-bfp16`** が付録です（本文末尾）。**GGUF の取得**は `qwen3-8b/` の **`make model`**、**ビルドと実行**は各サブディレクトリの Makefile で行います。

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


## モデルファイルを置く

`qwen3-8b/Makefile` の既定モデル名は次です。

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

モデルファイルは著作権とファイルサイズの都合により、リポジトリには含めません。`qwen3-8b/gguf.txt` の URL から取得し、`qwen3-8b/` の直下に置きます。推奨は **`make model`**（`wget` + 同梱 `.sha256sum` で検証。既にファイルがありチェックサムが通ればダウンロードをスキップ）です。

```bash
cd qwen3-8b
make model
```

成功時はターミナルにチェックサム検証成功のメッセージが表示されます。

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
├── gpu-vulkan/ …
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
make model          # 未取得なら GGUF を取得・検証（取得済みならスキップ）
cd cpu && make build
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

うまくいくと、モデル読み込み後に少しずつテキストが表示されます。

## CPU 単スレッド版

### ビルド

```bash
cd qwen3-8b/cpu
make build
```

成功すると **`qwen3-cpu`** ができます（`cpu/` 直下）。

```bash
ls -lh qwen3-cpu
```

### 実行

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます。

```bash
cd qwen3-8b/cpu
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で短く自己紹介してください。" -n 16
```

`Makefile` の `run` を使う場合:

```bash
cd qwen3-8b/cpu
make run PROMPT="日本語で短く自己紹介してください。"
```

別の場所にあるモデルを使う場合:

```bash
cd qwen3-8b/cpu
make run MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP 版

CPU コアを複数使う版です。単スレッド版と同じモデルを読みます。

### ビルド

```bash
cd qwen3-8b/cpu-multicore
make build
```

成功すると **`qwen3-cpu-omp`** ができます。

### 実行

```bash
cd qwen3-8b/cpu-multicore
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "箇条書きで、量子化とは何かを説明してください。" \
  -n 32
```

`OMP_NUM_THREADS` は使う CPU スレッド数です。迷ったら、まずは 4 や 8 から試してください。

```bash
cd qwen3-8b/cpu-multicore
OMP_NUM_THREADS=4 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

速くなるかどうかは CPU のコア数、メモリ帯域、モデルの量子化形式に依存します。

## CPU OpenMP + OpenBLAS 版

`cpu-multicore` と同じデコーダ・同じ GGUF を読み、**F32 行列積（`cblas_sgemv`）** と **Attention の K/V 合成**を OpenBLAS に任せます。IQ2_S / IQ3_S / Q4_K / Q5_K の量子化 GEMV は、入力を **Q8_K** に量子化（**`quantize_row_q8_K`**）したうえで **ggml-cpu/quants.c** 準拠の **`vec_dot_*_q8_K`** 整数内積を使います（`cpu-multicore` のような per-row float[256] dequantization は行わない）。**Attention / FFN 内で同一活性ベクトルを共有する GEMV では Q8 量子化を 1 回にまとめる**（層内 Q8 共有）。**`__AVX2__`** 時は IQ2_S / IQ3_S / Q4_K / Q5_K の dot と Q8 量子化を SIMD 化（**`-march=native`** 既定）。

そのほか CPU 向けの最適化として、**RoPE cos/sin キャッシュ**（起動時に `[max_seq × head_dim/2]` を一括計算）、prefill 中（最終プロンプト token 以外）の **LM head スキップ**（**`FWD_NO_LM`**）、**`-t 0`（greedy）** 時の **`mm_argmax_row`**（全 vocab logits を確保しない）、**F16 埋め込み**の **F16C+AVX2** SIMD 変換があります。プロンプト区間は stderr に **Prefill progress bar**（`Prefill [====...]`、幅 40）と prefill / decode / total のスループット要約を出力します。

### ビルド

```bash
cd qwen3-8b/cpu-blas
make build
```

成功すると **`qwen3-cpu-blas`** ができます（`cpu-blas/` 直下）。

### 実行

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます。

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "Hello, how are you?" \
  -n 32
```

`Makefile` の `run` を使う場合:

```bash
cd qwen3-8b/cpu-blas
make run PROMPT="Hello, how are you?"
```

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
cd qwen3-8b/cpu
./qwen3-cpu ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

慣れてきたら `-n` を増やします。

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で詩を書いてください。" -n 128
```

## 生成を安定させたいとき

同じ入力で結果を比較したい場合は、温度を下げたり seed を固定します。

```bash
cd qwen3-8b/cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "1文で説明してください: GGUFとは？" \
  -n 32 \
  -t 0.2 \
  -s 42
```

完全に同じ結果になるかは、CPU 版と GPU 版、サンプリング経路、GPU の実行環境によって変わることがあります。比較するときは、同じ実行ファイル・同じモデル・同じオプションで試してください。

## 片付け

各バリアントのビルド生成物は、対応するサブディレクトリで `make clean` します。

```bash
cd qwen3-8b/cpu && make clean
cd qwen3-8b/cpu-multicore && make clean
cd qwen3-8b/cpu-blas && make clean
# 付録 GPU / XDNA:
cd qwen3-8b/gpu-rocm && make clean
cd qwen3-8b/gpu-vulkan && make clean
cd qwen3-8b/gpu-cuda && make clean
cd qwen3-8b/gpu-cuda-nvfp4 && make clean
cd qwen3-8b/xdna2 && make clean
cd qwen3-8b/xdna2-bfp16 && make clean
```

削除される主なファイル:

- `cpu/qwen3-cpu`
- `cpu-multicore/qwen3-cpu-omp`
- `cpu-blas/qwen3-cpu-blas`
- `gpu-rocm/qwen3-rocm`
- `gpu-vulkan/qwen3-vulkan`
- `xdna2/qwen3-xdna2`
- `xdna2-bfp16/qwen3-xdna2-bfpx`
- `gpu-cuda/qwen3-gpu-cuda`
- `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4`

`make clean` では GGUF モデルファイルは削除されません。

## よくあるトラブル

### `No such file or directory` と出る

モデルファイルの場所が間違っている可能性があります。

```bash
ls -lh qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

見つからない場合は、モデルを `qwen3-8b/` に置くか、実行時に絶対パスを指定してください。

```bash
cd qwen3-8b/cpu
./qwen3-cpu /data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

### CPU 版が遅い

正常です。8B 級モデルは CPU だけで動かすには重いです。まずは `-n 1` や `-n 4` で確認してください。

```bash
./cpu/qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

CPU のみの場合は **`cpu-blas/qwen3-cpu-blas`**（OpenBLAS + Q8_K 量子化 GEMV + AVX2 dot + 層内 Q8 共有 + RoPE キャッシュ + prefill LM スキップ + greedy argmax）の方が **`cpu-multicore`** より速くなることが多いです。**greedy（`-t 0`）** では decode の LM head がさらに軽くなります。AMD GPU・NVIDIA GPU・XDNA2 NPU 向けの高速経路は README 末尾の **付録**を参照してください。

### `cpu-blas` のビルドが失敗する／`cblas.h` が見つからない

OpenBLAS 開発パッケージを入れ、必要なら `CPPFLAGS` でヘッダパスを指定してください（上記「OpenBLAS 版を使う場合」参照）。

### `cpu-blas` の出力が意味不明（同じ文字の連打など）

**`-ffast-math`** を付けてビルドすると IQ / Q8_K 量子化内積で数値が崩れます。リポジトリ同梱の `cpu-blas/Makefile` では無効化済みです。手元で CFLAGS を上書きしている場合は外してください。

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
- **AMD NPU（XDNA2 等）** や **ROCm / Vulkan / CUDA GPU** 向けコード（**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`** / **`xdna2`** / **`xdna2-bfp16`** は付録の参考実装。本文の主眼は CPU 3 バリアント）
- バッチ推論の最適化（GPU 付録の prefill バッチは decode 高速化用）
- 画像入力
- **マルチターン対話の組み込み CLI**（履歴管理・KV 再利用・公式 chat template の完全再現）
- **Thinking モードの制御・思考ブロックの分離表示**（`enable_thinking` / `/think` / `/no_think` 等）
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- 公式実装との完全な数値一致保証

目的は、Qwen3系GGUFのテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- 設計仕様: `doc/design.md`
- 変更履歴: `doc/ChangeLog.md`

困ったときは、まず `qwen3-8b/` で **`make model`** 済みか、各サブディレクトリでビルドしたバイナリと、実行時に渡しているモデルパスが一致しているかを確認してください。

---

## AMD ROCm / HIP 実装（`gpu-rocm`）について

**`qwen3-8b/gpu-rocm/` は本リポジトリの目的（単一 C ソース・依存最小）から外れた付録**です。`main.c` + HIP カーネルに分かれ、**ROCm（`hipcc`）・AMD GPU ドライバ・実機**が必要です。ROCm/HIP は **GPU 向けのコンパイラ・ランタイム** であり、**ニューラルネット用の高レベルフレームワークではありません**（ここからさらに自作の HIP カーネルとホストコードで Transformer を組み立てています）。参照すべき最小実装は **`cpu/main.c`** です。

同梱している理由は、筆者が **AMD GPU 上でどこまで高速化できるか** を試したくなっただけです。本プロジェクトの目的を補うものでも、読者向けの正式な機能でもありません。混乱を招きやすいため、**将来的には別リポジトリへ移す予定**です。初めて読む方は無視して構いません。

以下は、GPU 上の速度比較に興味がある場合の技術メモです。

### 必要なもの（ROCm）

AMD GPU と ROCm が必要です。`Makefile` は既定で ROCm を `/opt/rocm` にあるものとして扱います。**`rocminfo` の `Name: gfx*`** を **`GPU_ARCH_DETECTED`** として取得し、**`hipcc --offload-arch`** には **`HIP_OFFLOAD_ARCH`** を使います。**Ryzen AI 5 340（Radeon 840M / `gfx1152`）** など **rocBLAS が `gfx1152` 用 Tensile を同梱していない GPU** では Makefile が **`gfx1151` ビルド + `HSA_OVERRIDE_GFX_VERSION=11.5.1`** を自動適用します（**ROCm を 7.2.1 に上げるだけでは直らない**場合がある — 下記 **「Ryzen AI / gfx1152 と rocBLAS」**）。**`fp16_cache_io.o`** リンクのため **g++ / libstdc++-dev** も必要です。

```bash
sudo apt install -y g++ libstdc++-dev   # C++ ヘッダ／libstdc++ リンク用
```

確認例:

```bash
/opt/rocm/bin/hipcc --version
make -C gpu-rocm detect-gpu-arch   # 例: Detected GPU arch: gfx1100
```

`rocminfo` が GPU を報告しない環境では、ビルド時に `GPU_ARCH=gfx1100` のように手動指定してください。

### GPU_ARCH（自動検出）

`gpu-rocm/Makefile` は、ビルド前に `$(ROCM)/bin/rocminfo` から最初の GPU エージェント名（`gfx*`）を **`GPU_ARCH_DETECTED`** として取得します（上書き変数は **`GPU_ARCH`**）。成功すると次のように表示されます:

```text
===============================================
  Detected GPU arch: gfx1100
  Build offload arch: gfx1100
===============================================
```

手動で上書きする場合:

```bash
cd qwen3-8b/gpu-rocm
make build GPU_ARCH=gfx1100
```

### Ryzen AI / gfx1152 と rocBLAS（環境依存）

**AMD Ryzen AI 系 APU** の内蔵 Radeon（RDNA 3.5）のうち、**`gfx1152`**（例: **Ryzen AI 5 340 + Radeon 840M**）では、Prefill 線形層の **hipBLAS → rocBLAS** が次の理由で失敗しやすいです。

| 項目 | 内容 |
|------|------|
| 典型エラー | **`rocBLAS error: Cannot read … TensileLibrary.dat … for GPU arch : gfx1152`**（利用可能リストに **`gfx1150` / `gfx1151` のみ**） |
| 本質 | 公式 **`/opt/rocm/lib/rocblas/library/`** に **`TensileLibrary_lazy_gfx1152.dat`** が無い（**ROCm 7.1.x / 7.2.x の apt 同梱時点**） |
| ROCm 7.2.1 へ上げる必要 | **この GPU 専用の必須条件ではない**。上げても **`gfx1152` Tensile が同梱されない**場合がある |
| 本リポジトリの対処 | **`HIP_OFFLOAD_ARCH=gfx1151`** + **`HSA_OVERRIDE_GFX_VERSION=11.5.1`**（**`make build` / `make run`** で自動） |

**`gfx1152` / `gfx1153` 検出時のビルドログ例:**

```text
  Detected GPU arch: gfx1152
  Build offload arch: gfx1151
  HSA_OVERRIDE_GFX_VERSION: 11.5.1
```

**成功時の起動ログ例**（オーバーライド有効時、`gcnArchName` は **`gfx1151`** と表示）:

```text
ROCm HIP device 0: AMD Radeon 840M Graphics (gcnArchName: gfx1151)
Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)
```

**参考（他の Ryzen AI 世代）:** Ryzen AI 9 HX 370 等は **`gfx1150`**（**`TensileLibrary_lazy_gfx1150.dat`** 同梱）のことが多く、通常はオーバーライド不要です。

**確認:**

```bash
ls /opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx115*.dat
cd qwen3-8b/gpu-rocm && make detect-gpu-arch
```

**避けるべき例:** **`gfx1151` の `.dat` を `gfx1152` 名でコピー／symlink だけ** → **`hipBLAS error: 6`**。**`HSA_OVERRIDE` のみ**でバイナリが **`gfx1152` offload** のまま → **Segmentation fault**。

詳細は [`doc/design.md`](doc/design.md) の **「環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS」** を参照。

### ビルド

```bash
cd qwen3-8b/gpu-rocm
make build
```

**`make build`**: **`qwen3-rocm`** をビルドし、**`MODEL`** が存在する場合は **`<model>.gguf.fp16/manifest`** まで生成（オフライン FP16 キャッシュ）。**`make run`**: バイナリのみ（pack-cache をスキップ）。

ROCm が `/opt/rocm` 以外にある場合:

```bash
cd qwen3-8b/gpu-rocm
make build ROCM=/path/to/rocm
```

成功すると **`qwen3-rocm`** ができます。リンクには **`-lhipblas -lrocblas -lstdc++`** が含まれます（Prefill 線形層の GEMM 用 + **`fp16_cache_io.o`**）。

起動ログに **`Loading FP16 cache from …`** または **`Uploading weights (row dequant -> FP16)...`** が出れば FP16 ロード経路が有効です。

### オフライン FP16 キャッシュ（`make pack-cache`）

初回起動の GGUF 逆量子化を省略するため、事前に FP16 キャッシュを生成します。出力先は既定で **`<model>.gguf.fp16`**（各 tensor の **`.fp16bin`** + **`manifest`**）。**`gpu-cuda`** と同形式です。

```bash
cd qwen3-8b/gpu-rocm
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# MODEL が既にある場合、make build でも自動 pack されます
```

バイナリから直接:

```bash
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-rocm ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-rocm ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

GGUF を更新した場合は **`make pack-cache`** を再実行してください（**`manifest`** が GGUF のサイズ・mtime と一致しないとキャッシュは無効化されます）。

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
cd qwen3-8b/gpu-rocm
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、ROCmとは何かを初心者向けに説明してください。" \
  -n 64
```

`Makefile` の `run` を使う場合:

```bash
cd qwen3-8b/gpu-rocm
make run PROMPT="日本語で短く説明してください。"
```

プロンプト区間では stderr に **Prefill progress bar** と prefill / decode / total のスループット要約が出ます（**`cpu-blas`** と同形式）。推論終了時、**`qwen3-rocm`** / **`qwen3-vulkan`** / **`gpu-cuda/`** / **`gpu-cuda-nvfp4/`** は **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value 形式のベンチログ（モデル・GPU・トークン数・tok/s・プロンプト全文など）を書き出します（**推論区間のみ**。モデル重み H2D は計測外）。tok/s に加え **VRAM 内訳**（**`GpuVramProfile`** / **`model_vram_profile`**）も **`[vram_breakdown]`** 節で出力します（**`gpu-vulkan`** は **`vram_total`** と簡易 **`[vram_breakdown]`**（線形重みは **`vram_total` に含む**））。重みアップロード中は 8 レイヤーごとに **`layer N/L uploaded: X.XX sec, X.XX GB/sec`** が stdout に出ます。tok/s メトリクスは stderr の **`--- throughput ---`** 要約と **`BENCH_LOG_FILE`** のみ（stdout ベンチ行なし）。

### ベンチマーク履歴（`gpu-rocm/Makefile`）

`gpu-rocm/` で **`make log.push`** を実行すると、既定の長文プロンプト（~128 token）・**`-n 128`**・**`-t 0`** でベンチを走らせ、**`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）からメトリクスを読み取り、結果を **`Makefile` 内の `BENCH_LOG`** に追記します。**`make log`** で履歴を表表示できます。

```bash
cd qwen3-8b/gpu-rocm
make log.push                    # 既定 BENCH_N=128, BENCH_SEED=42
make log                         # 履歴一覧
make log.push BENCH_N=64         # 生成トークン数など上書き可
# ログファイルのパスを変える例:
make log.push BENCH_LOG_FILE=/tmp/my-bench.log
```

**`Makefile` 追記行**（パイ区切り）: **`日時|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

手動実行時は **`BENCH_LOG_FILE=/path/to/log ./qwen3-rocm model.gguf -p "…" -n 64`** のあと、同ファイルの **`prefill_tps=`** 等を参照できます。

**ベンチログファイル**（推論終了時に上書き）の主なキー: **`timestamp`**, **`hostname`**, **`model`**, **`gpu`**, **`prompt_tokens`**, **`gen_tokens`**, **`prefill_tps`**, **`decode_tps`**, **`total_tps`**、**`vram_total`**、**`[vram_breakdown]`**（**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`**）。プロンプト全文は **`--- prompt ---`** 節。**`make log.push`**（**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`**）はこのファイルから **`prompt_tokens=`** 等を読み、**`Makefile` の `BENCH_LOG`** に 1 行追記します。Makefile の **`BENCH_LOG`** 履歴には tok/s のみ追記され、VRAM 内訳は **`BENCH_LOG_FILE`** を参照してください。

#### ROCm GPU 長プロンプト（`gpu-rocm`・`make log.push`）

| 項目 | 値 |
|---|---|
| GPU | AMD Radeon Graphics（**`gfx1201`**、32 GiB VRAM） |
| OS | Linux |
| モデル | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`（**`<model>.gguf.fp16`** キャッシュから H2D） |
| コマンド | **`make log.push`**（**`GPU_ARCH=gfx1201`**） |
| ワークロード | 長文プロンプト（ChatML 後 **132** トークン）+ decode **最大 128**（**`-n 128 -t 0 -s 42`**） |
| 表の指標 | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`prefill_tps` / `decode_tps` / `total_tps`** — 推論区間のみ |
| 再現 | `qwen3-8b/gpu-rocm/` で **`make log.push`** → **`make log`** |

| 計測日時 | GPU | prefill tok/s | decode tok/s | total tok/s | 備考 |
|---|---|---:|---:|---:|---|
| 2026-05-29 08:12 | **gfx1201** | **124.95** | **28.88** | **91.89** | 132+16 トークン（**`make log.push`**。`-t 0` で EOS により生成 16 で打切） |

**VRAM 内訳**（上記 **2026-05-29 08:12** 計測。**`BENCH_LOG_FILE`** の **`[vram_breakdown]`**。`-l` 既定 **`max_seq=512`** 条件）:

| 項目 | bytes | MiB | 備考 |
|---|---:|---:|---|
| **`vram_total`**（理論合計） | 16,634,527,232 | **15863.92** | 下記カテゴリの合計 |
| **`vram_device_used`** | 16,989,028,352 | **16202.00** | **`hipMemGetInfo`**（HIP ランタイム等を含む場合あり） |
| **`vram_device_total`** | 34,208,743,424 | **32624.00** | GPU 全体 VRAM |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm 重み |
| `vram_weights_linear` | 15,136,194,560 | **14435.00** | FP16 線形重み（**`wq`〜`down`** + LM head） |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`**（`-l` 依存） |
| `vram_decode_activations` | 779,776 | **0.74** | 単トークン decode 用バッファ |
| `vram_prefill_batch` | 100,665,344 | **96.00** | prefill バッチ（**`batch_cap = max_seq`**）+ **`d_scratch_f16`**（hipBLAS GemmEx 用 FP16 活性スクラッチ） |

線形重み（**~14435 MiB**）と embedding（**~1187 MiB**）が理論 VRAM の大半（8B 級 FP16 常駐）。KV（**~144 MiB**）は **`-l`** を増やすと比例して増えます。**`vram_prefill_batch`** は **`gpu-cuda`**（**~84 MiB**）より大きく、Prefill 用 **`d_scratch_f16`**（**`max_seq × hidden_dim`** FP16）を含みます。**`vram_device_used`** は **`vram_total`** より大きいことがあります（ドライバ／HIP 割当の差）。

### WMMA 利用状況の確認（`make wmma`）

**WMMA が実際に使われるかは rocBLAS のカーネル選択次第**であり、**`qwen3-rocm` 側で ON/OFF する手段はありません**。Prefill の GEMM は hipBLAS `GemmEx` に渡され、内部で rocBLAS が行列サイズ・GPU アーキテクチャ・精度などに応じてカーネルを選びます。その結果として WMMA 命令を含むカーネルが選ばれることもあれば、**`v_fmac_f32` など FMAC 系カーネルだけが使われることもあります**（`gfx1201` では `make wmma` の lib チェックで WMMA 0 件の WARN があり得ます）。WMMA 利用を「推論モード」として切り替えるのではなく、**`make wmma`**（静的 ISA 確認）や **`WMMA_SKIP_ROCPROF=0`**（実行時 `rocprofv3` trace）で、ライブラリ側が実際にどの命令を使ったかを事後確認する形になります。

Prefill 線形層は **hipBLAS / rocBLAS** ライブラリ経由であり、**`main.c` に WMMA / rocWMMA / MFMA を直接書いていません**（Prefill GEMM は hipBLAS に委譲）。**`make wmma`** で次を確認できます。

- **`main.c` / `qwen3-rocm` バイナリ**に WMMA 命令が無いこと（正常）
- **`wmma-probe`**（RDNA gfx11/gfx12 向け校正用バイナリ）に WMMA があること（`llvm-objdump` 検出器の校正）
- **rocBLAS** バンドル ISA に WMMA があるか（0 件でも FMAC 経路の WARN がありうる）
- 任意: 実行時ログ **`Prefill linear: hipBLAS GemmEx`**、**`rocprofv3`** カーネル trace

```bash
cd qwen3-8b/gpu-rocm
make wmma                           # build + wmma-probe + チェック（MODEL 要）
make wmma WMMA_SKIP_RUN=1           # 静的チェックのみ（MODEL 不要）
make wmma WMMA_SKIP_ROCPROF=0       # rocprofv3 カーネル ISA も試行
make wmma-probe                     # 校正用 wmma-probe のみビルド
```

#### 手動で `rocprofv3 --kernel-trace` から `v_wmma` を数える

Prefill 実行中に実際にロードされたカーネル（`.hsaco` / `.co`）をダンプし、WMMA 命令の有無を確認する手順です。**`make wmma WMMA_SKIP_ROCPROF=0`** が内部で行う処理と同趣旨です。

**前提**: `qwen3-rocm` をビルド済み、MODEL の GGUF が配置済み、`rocprofv3` が PATH にあること（ROCm 7 系。`/opt/rocm/bin` など）。

```bash
cd qwen3-8b/gpu-rocm
MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf   # 実際のパスに合わせる
TRACE_DIR=/tmp/qwen3-wmma-trace
LLVM_OBJDUMP=${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}

mkdir -p "$TRACE_DIR"
rocprofv3 --kernel-trace -d "$TRACE_DIR" -f csv -- \
  ./qwen3-rocm "$MODEL" -p "Hello" -n 0 -t 0 -s 42
```

**`-n 0`** は生成トークン 0（Prefill のみ短時間）の例です。プロンプト長や GPU によっては **`-n 1`** などに変えても構いません。

trace 出力ディレクトリ内のコードオブジェクトを列挙し、各ファイルの逆アセンブルから **`v_wmma`** 命令を数えます。

```bash
# ダンプされた .hsaco / .co を確認
find "$TRACE_DIR" -type f \( -name '*.hsaco' -o -name '*.co' \)

# ファイルごとの v_wmma 件数
find "$TRACE_DIR" -type f \( -name '*.hsaco' -o -name '*.co' \) -print0 | while IFS= read -r -d '' f; do
  n=$("$LLVM_OBJDUMP" -d "$f" 2>/dev/null | grep -ciE '\tv_wmma|\bv_wmma_' || true)
  printf '%4d  %s\n' "$n" "$f"
done

# 合計（check_wmma.sh の runtime チェックと同様）
total=0
while IFS= read -r -d '' f; do
  n=$("$LLVM_OBJDUMP" -d "$f" 2>/dev/null | grep -ciE '\tv_wmma|\bv_wmma_' || true)
  total=$((total + n))
done < <(find "$TRACE_DIR" -type f \( -name '*.hsaco' -o -name '*.co' \) -print0)
echo "total v_wmma instructions: $total"
```

**読み方**: 合計が **0** なら、今回の Prefill では rocBLAS 等が WMMA カーネルを選ばなかった可能性が高いです（FMAC 経路など）。**1 以上**なら、trace 中に WMMA を含むカーネルが実行されています。`.hsaco` が 1 件も出ない場合は ROCm プロファイラの設定（GPU アクセス、`libdw.so` 不足など）を確認してください。

詳細は [`doc/design.md`](doc/design.md) の ROCm ビルド節と **`scripts/check_wmma.sh`** を参照。

### ソースを読む場合

6. `qwen3-8b/gpu-rocm/fp16_cache_io.c` / `main.c` — AMD GPU。**`<model>.gguf.fp16`** オフラインキャッシュ、GGUF 行単位融合逆量子化。**Prefill** は **`forward_prefill_gpu`**（**hipBLAS GemmEx**）。**Decode** は **`forward_gpu`**。Prefill 高速化の詳細は **`doc/design.md`** の **「ROCm Prefill 高速化の詳細（3 段階）」**。

## Vulkan compute 実装（`gpu-vulkan`）について

**`qwen3-8b/gpu-vulkan/`** は **ROCm も CUDA も使わず**、**Vulkan 1.1 compute シェーダ**だけで GPU 推論を行う **付録**です。AMD（RADV/Mesa）、NVIDIA、Intel など **Vulkan 対応 GPU** で動作を想定していますが、**ベンダー固有の最適化（hipBLAS / cuBLAS / Tensor Core 等）は使いません**。参照すべき最小実装は **`cpu/main.c`** です。

### 位置づけ（`gpu-rocm` / `gpu-cuda` との関係）

| 観点 | `gpu-rocm` | `gpu-vulkan` | `gpu-cuda` |
|------|------------|--------------|------------|
| ランタイム | ROCm / HIP | Vulkan loader + compute | CUDA |
| 対象 GPU | AMD（ROCm 必須） | Vulkan 対応 GPU（ベンダー非依存） | NVIDIA |
| 線形層 Prefill | hipBLAS GemmEx | FP16 GEMV バッチ（compute） | FP16 GEMV バッチ |
| 線形層 Decode | カスタム GEMV（HIP） | FP16 GEMV（compute） | FP16 GEMV（CUDA） |
| Attention | Flash Attention（HIP） | Flash Attention（GLSL） | Flash Attention（CUDA） |
| FP16 キャッシュ | **`<model>.gguf.fp16`**（共通） | 同左 | 同左 |

**AMD GPU で ROCm を入れたくない**、**クロスベンダーで同じ GPU 経路を試したい**場合の代替経路です。**スループット最優先なら `gpu-rocm`（AMD）または `gpu-cuda`（NVIDIA）を推奨**します。

### 必要なもの

- **Vulkan 1.1 以降**対応 GPU とドライバ（Linux 例: Mesa RADV、NVIDIA プロプライエタリ）
- 開発パッケージ: **`libvulkan-dev`**
- シェーダコンパイル: **`glslang-tools`**（`glslangValidator`）
- 実行時: **`vulkan-loader`**（多くのディストリビューションでは `libvulkan1`）

確認例:

```bash
vulkaninfo --summary
glslangValidator --version
```

### ビルドと実行

```bash
cd qwen3-8b/gpu-vulkan
make build          # shaders/*.comp → shaders/*.spv を生成して qwen3-vulkan をリンク
make run            # 既定 PROMPT="Hello, how are you?"
make pack-cache     # <model>.gguf.fp16 を生成（gpu-rocm / gpu-cuda と同形式）
```

`make run` は内部で **`QWEN3_VK_SHADER_DIR=$(pwd)/shaders`** を渡します。バイナリを直接起動する場合も、**SPIR-V（`.spv`）のあるディレクトリ**を指定してください:

```bash
cd qwen3-8b/gpu-vulkan
QWEN3_VK_SHADER_DIR=$(pwd)/shaders ./qwen3-vulkan ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 64
```

CLI は **`gpu-rocm` / `gpu-cuda`** と同趣旨です（**`--pack-fp16-cache`**、**`--no-fp16-cache`**、**`-p` / `-n` / `-t` / `-k` / `-s` / `-l`**）。

### 実装構成

- **`main.c`** — GGUF 読み込み、トークナイザ、FP16 重み H2D（`gpu-cuda` ベース）
- **`vk_context.c`** — Vulkan インスタンス・デバイス・キュー・コマンドプール
- **`vk_alloc.c`** — デバイスバッファ（`vk_malloc` / H2D / D2H）
- **`vk_pipeline.c`** — compute パイプラインと dispatch
- **`vk_kernels.c`** — `gpu_forward` / `gpu_forward_prefill`（`gpu.h` API）
- **`shaders/*.comp`** — RMSNorm、RoPE、FP16 GEMV、Flash Attention 等（18 本）

重みロードは **`gpu-rocm` / `gpu-cuda`（FP16）** と同じです。**`<model>.gguf.fp16`** があれば **`.fp16bin`** から H2D、無ければ GGUF 行単位逆量子化 → FP16 → VRAM。KV キャッシュは **F32** です。

### 既知の制約と今後の改善余地

現状の **`gpu-vulkan`** は **正しく推論できることを確認した初期実装**です。`gpu-rocm` と比べ **スループットは大幅に低い**のが通常です（カーネル自体の差に加え、ホスト側オーバーヘッドが大きい）。

主な要因:

1. **カーネル起動オーバーヘッド** — レイヤーごとに多数の **`vkCmdDispatch`** を発行。各 dispatch でディスクリプタセット確保・更新・フェンス待ちを行う（同期実行）。
2. **Prefill 線形層** — **`gpu-rocm`** の hipBLAS GemmEx に相当する **GEMM バッチ最適化が未実装**。Prefill も decode も **FP16 GEMV 系 compute シェーダ**に依存。
3. **シェーダ汎用性優先** — WMMA / cooperative matrix 等の **ハードウェア固有 intrinsics は未使用**（GLSL の可搬性を優先）。
4. **転送経路** — 重み H2D は **ステージングバッファ経由の都度コピー**（CUDA/HIP の pinned memory 最適化より単純）。

改善の候補（未実装）:

- ディスクリプタセット・コマンドバッファの **再利用**（dispatch 回数は据え置きでも CPU オーバーヘッド削減）
- Prefill 用 **batched GEMM** compute シェーダ（または Vulkan **`VK_KHR_cooperative_matrix`** 等）
- 1 レイヤー分を **1 回の dispatch に融合**する mega-kernel 化
- 長プロンプトベンチ（**`make log.push`**）での性能改善（現状 **~3 tok/s** 級・ROCm 比 **10〜40×** 遅い）

### ベンチマーク履歴（`gpu-vulkan/Makefile`）

**`make log.push`** / **`make log`** の枠組みは **`gpu-rocm`** と同趣旨です（**`BENCH_LOG_FILE`** から読取、Makefile の **`BENCH_LOG`** に追記）。履歴行の第 2 列は **`vulkaninfo` 由来の GPU 名**です。

```bash
cd qwen3-8b/gpu-vulkan
make log.push    # 長プロンプト ~132 token + -n 128 -t 0
make log
```

#### Vulkan GPU 長プロンプト（`gpu-vulkan`・`make log.push`）

| 項目 | 値 |
|---|---|
| GPU | AMD Radeon Graphics（**RADV GFX1201**、32 GiB VRAM） |
| OS | Linux |
| モデル | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`（**`<model>.gguf.fp16`** キャッシュから H2D） |
| コマンド | **`make log.push`** |
| ワークロード | 長文プロンプト（ChatML 後 **132** トークン）+ decode **最大 128**（**`-n 128 -t 0 -s 42`**） |
| 表の指標 | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`prefill_tps` / `decode_tps` / `total_tps`** — 推論区間のみ |
| 再現 | `qwen3-8b/gpu-vulkan/` で **`make log.push`** → **`make log`** |

| 計測日時 | GPU | prefill tok/s | decode tok/s | total tok/s | 備考 |
|---|---|---:|---:|---:|---|
| 2026-05-29 09:39 | **RADV GFX1201** | **2.97** | **2.17** | **2.86** | 132+16 トークン（**`make log.push`**。`-t 0` で EOS により生成 16 で打切） |

**VRAM 内訳**（上記 **2026-05-29 09:39** 計測。**`BENCH_LOG_FILE`** の **`[vram_breakdown]`**。`-l` 既定 **`max_seq=512`** 条件）:

| 項目 | bytes | MiB | 備考 |
|---|---:|---:|---|
| **`vram_total`**（理論合計） | 16,621,944,320 | **15851.92** | 下記カテゴリ + FP16 線形重み（**~14435 MiB**） |
| **`vram_device_total`** | 34,208,743,424 | **32624.00** | GPU 全体 VRAM（Vulkan では **`vram_device_used`** 未取得） |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm 重み |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`**（`-l` 依存） |
| `vram_decode_activations` | 779,776 | **0.74** | 単トークン decode 用バッファ |
| `vram_prefill_batch` | 88,082,432 | **84.00** | prefill バッチ（**`batch_cap = max_seq`**） |

同一 GPU の **`gpu-rocm`**（**`make log.push`** 履歴）では prefill **124.95** / decode **28.88** / total **91.89** tok/s です。**Vulkan 経路は ROCm 比で prefill 約 42×・decode 約 13× 遅い**のが今回の実測です（カーネル性能に加え **`vkCmdDispatch` 同期オーバーヘッド**が大きい）。**数値は環境依存**です。

### ソースを読む場合

7. `qwen3-8b/gpu-vulkan/fp16_cache_io.c` / `main.c` / `vk_kernels.c` / `vk_pipeline.c` / `shaders/*.comp` — Vulkan compute。**`gpu.h`** 経由で forward を分離。**`QWEN3_VK_SHADER_DIR`** で `.spv` 探索。

### よくあるトラブル（`gpu-vulkan`）

**`Cannot open shader: …/xxx.spv`**

`make build` で **`shaders/*.spv`** が生成されているか確認してください。直接実行時は **`QWEN3_VK_SHADER_DIR`** を **`shaders/` ディレクトリ**に設定します。

**`vkAllocateDescriptorSets` / OUT_OF_POOL_MEMORY**

1 回の forward で大量の dispatch を行うため、ディスクリプタプールを使い切ることがあります。最新版ではプールサイズ拡大と **`vkFreeDescriptorSets`** による解放を行っています。**`make clean && make build`** で再ビルドしてください。

**RADV の `not a conformant Vulkan implementation` 警告**

Mesa RADV では開発中の警告が出ることがあります。本付録は **検証用**であり、Vulkan 適合性認証済み実装を前提としていません。

**ROCm / CUDA と比べて極端に遅い**

上記 **「既知の制約」** を参照してください。性能が目的なら **`gpu-rocm`** または **`gpu-cuda`** を使用してください。

## NVIDIA CUDA 実装（`gpu-cuda` / `gpu-cuda-nvfp4`）について

**`qwen3-8b/gpu-cuda/`** および **`qwen3-8b/gpu-cuda-nvfp4/`** も同様に **付録**です。`main.c` + `kernels.cu`（+ NVFP4 用 `fp4_*.cu`）に分かれ、**CUDA Toolkit（`nvcc`）・NVIDIA ドライバ・GPU 実機**が必要です。NVFP4 版は **CUDA 13 + Blackwell（sm_120 系）** + CUTLASS が必要です。初めて読む方は無視して構いません。

以下は、NVIDIA GPU 向けの技術メモです。

NVIDIA GPU と CUDA が使える環境向けです。**`qwen3-8b/Makefile` はモデル取得のみ**のため、用途に応じて **`gpu-cuda/`**（FP16）または **`gpu-cuda-nvfp4/`**（NVFP4）で単体ビルドします。

### 必要なもの（CUDA）

NVIDIA GPU と **CUDA Toolkit**（`nvcc`・`libcudart`）が必要です。`qwen3-8b/Makefile` はモデル取得（**`make model`**）のみで、CUDA ビルドは次の2ディレクトリで行います。

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
| **ROCm FP16 オフラインキャッシュ**（2 回目以降の起動を高速化） | `cd qwen3-8b/gpu-rocm` → `make pack-cache`（**`make build`** は MODEL 存在時に自動 pack） |
| **Vulkan compute FP16**（ROCm/CUDA 不要・クロスベンダー） | `cd qwen3-8b/gpu-vulkan` → `make build` / `make run`（**`QWEN3_VK_SHADER_DIR`** で `.spv` 指定） |
| **Vulkan FP16 オフラインキャッシュ** | `cd qwen3-8b/gpu-vulkan` → `make pack-cache` |
| **FP16 のみ**（Ampere/Ada 等・PTX 可） | `cd qwen3-8b/gpu-cuda` → `make build` / `make run` |
| **FP16 オフラインキャッシュ**（2 回目以降の起動を高速化） | `cd qwen3-8b/gpu-cuda` → `make pack-cache` |
| **PolarQuant-R KV キャッシュ**（FP16 線形・任意 GPU） | `cd qwen3-8b/gpu-cuda` → `make build.polarquant` / `make run.polarquant` |
| **Blackwell NVFP4**（RTX 50 系等） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build` / `make run` |
| **NVFP4 オフラインキャッシュ**（2 回目以降の起動を高速化） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make pack-cache` |
| **NVFP4 + PolarQuant 同時**（Blackwell・最大 VRAM 節約） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make build.polarquant` / `make run.polarquant` |
| CUDA 13 の導入から NVFP4 一式 | `cd qwen3-8b/gpu-cuda-nvfp4` → `make blackwell` |
| PolarQuant ラウンドトリップ検証 | 各ディレクトリで `make pq-test` |
| CUTLASS NVFP4 GEMM 単体検証 | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fp4-test`（**Blackwell / sm_120a 必須**） |
| SFA/SFB 索引検証 | `cd qwen3-8b/gpu-cuda-nvfp4` → `make sfa-verify` |
| Flash Attention 診断（Hello） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make fa-debug`（FP16 / NVFP4 比較） |
| ベンチマーク履歴（ROCm） | `cd qwen3-8b/gpu-rocm` → `make log.push` / `make log`（**`BENCH_LOG_FILE`** から読取。既定 **`/tmp/benchmark.log`**） |
| ベンチマーク履歴（Vulkan） | `cd qwen3-8b/gpu-vulkan` → `make log.push` / `make log`（**`BENCH_LOG_FILE`** から読取。既定 **`/tmp/benchmark.log`**） |
| ベンチマーク履歴（CUDA FP16） | `cd qwen3-8b/gpu-cuda` → `make log.push` / `make log`（**`BENCH_LOG_FILE`** から読取。既定 **`/tmp/benchmark.log`**） |
| ベンチマーク履歴（CUDA NVFP4） | `cd qwen3-8b/gpu-cuda-nvfp4` → `make log.push` / `make log`（**`BENCH_LOG_FILE`** から読取。既定 **`/tmp/benchmark.log`**） |

FP16 ビルド（`gpu-cuda`）の既定は **`nvidia-smi` による GPU 自動検出**（RTX 50 / Blackwell **12.x** → **`sm_120`** + **`FA_BR=32`**）。**PTX `compute_86` JIT** は RTX 5090 等で**推論が文字化け**するため非推奨。手動指定例: `make build CUDA_GENCODE=arch=compute_89,code=sm_89`。NVFP4 ビルド（`gpu-cuda-nvfp4`）の既定は **`sm_120a`** です。**`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** は **`BLACKWELL_NVCCFLAGS`**（**`-std=c++17`** + **`sm_120a` 固定**）でコンパイルします。CUTLASS は **`v4.5.0`**（**`make cutlass`** で **`third_party/cutlass`** を取得）を使用し、CUDA 13 非推奨ベクトル型警告は CUTLASS 側で解消済みです。

プロンプトは **Prefill バッチ**、生成は **1 トークン Decode**、Attention は **Flash Attention**（GQA）です。

| ディレクトリ | ロード時の重み | 線形層 / KV キャッシュの実行 |
|--------------|----------------|------------------------------|
| **`gpu-rocm`** | オフラインキャッシュ（**`<model>.gguf.fp16`**）があれば **`.fp16bin`** から H2D。無ければ GGUF **行単位融合逆量子化** → FP16 | hipBLAS GemmEx（Prefill）+ カスタム GEMV（Decode）。KV は **F32** |
| **`gpu-vulkan`** | 上記 FP16 キャッシュと同形式 | FP16 GEMV compute シェーダ（Prefill / Decode 共通）。Flash Attention（GLSL）。KV は **F32**。hipBLAS / cuBLAS 相当の GEMM **未実装** |
| **`gpu-cuda`** | オフラインキャッシュ（**`<model>.gguf.fp16`**）があれば **`.fp16bin`** から H2D。無ければ GGUF **行単位融合逆量子化** → FP16 | FP16 GEMV カーネル。KV は **F32**（既定） |
| **`gpu-cuda`** + **`build.polarquant`** | 線形は FP16（上と同じ） | KV は **PolarQuant-R**（64 B/head）。Attention タイル読み出し時に F32 復号 |
| **`gpu-cuda-nvfp4`** | オフラインキャッシュ（**`<model>.gguf.nvfp4`**、**`FP4_CACHE_VERSION=2`**）があれば **`.fp4bin`** から H2D。無ければ GGUF **行単位融合逆量子化** → NVFP4。**`token_embd`** は行単位 FP16 H2D | **`fp4_qwen3_mm`** — prefill / decode とも **CUTLASS NVFP4 GEMM**（**`fp4_gemm_run_cached`**、**`M` を 128 整列**）。活性量子化は **`FP4_QUANT_MAX_ABS=1024`** でクランプ |
| **`gpu-cuda-nvfp4`** + **`build.polarquant`** | 線形は NVFP4 のみ（上と同じ） | 線形は上記 GEMM。KV は PolarQuant-R |

**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`**（FP16）も初回起動は GGUF 行逆量子化に時間がかかります。**`make pack-cache`**（または **`--pack-fp16-cache`**）で **`<model>.gguf.fp16`** を事前生成すると、2 回目以降は **`Loading FP16 cache from …`** から H2D できます。**`--no-fp16-cache`** でキャッシュを無視して毎回再逆量子化します。**`gpu-rocm`** では **`make build`**（**`MODEL`** 存在時）でも自動 pack されます。

**`gpu-cuda-nvfp4`** は CUTLASS **NVFP4** を使い、**CUDA 13 + sm_120 系 GPU**（Blackwell / RTX 50 系等）向けです。初回起動は GGUF から NVFP4 へ量子化するため時間がかかります。**`make pack-cache`**（または **`--pack-nvfp4-cache`**）で **`<model>.gguf.nvfp4`** を事前生成すると、2 回目以降は **`Loading NVFP4 cache from …`** から H2D できます。**`--no-nvfp4-cache`** でキャッシュを無視して毎回再量子化します。

### ビルド・実行（FP16・汎用 GPU）

```bash
cd qwen3-8b/gpu-cuda
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

成功すると **`qwen3-gpu-cuda`** ができます。起動ログに **`Loading FP16 cache from …`** または **`Uploading weights (fused dequant -> FP16)...`** が出れば FP16 ロード経路が有効です。実 GPU アーキテクチャを直接指定する例:

```bash
make build CUDA_GENCODE=arch=compute_89,code=sm_89
```

### オフライン FP16 キャッシュ（`make pack-cache`）

初回起動の GGUF 逆量子化を省略するため、事前に FP16 キャッシュを生成します。出力先は既定で **`<model>.gguf.fp16`**（各 tensor の **`.fp16bin`** + **`manifest`**）。**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** で同形式です。

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

バイナリから直接（**`gpu-cuda`** / **`gpu-rocm`** / **`gpu-vulkan`** 共通 CLI）:

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-gpu-cuda ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-gpu-cuda ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

GGUF を更新した場合は **`make pack-cache`** を再実行してください（**`manifest`** が GGUF のサイズ・mtime と一致しないとキャッシュは無効化されます）。

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

成功すると **`qwen3-gpu-cuda-nvfp4`** ができます。起動ログに **`Loading NVFP4 cache from …`** または **`Uploading weights (fused dequant -> NVFP4 linear layers)...`** と **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** が出れば FP4 経路が有効です。

短いプロンプトで **`?,` 連打** 等が出る場合は **`make fp4-test`** が PASS する最新ビルドか確認してください（調査ログ: **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`**、ベースコミット **`433319eb`**）。

### オフライン NVFP4 キャッシュ（`make pack-cache`）

初回起動の GGUF 逆量子化 + NVFP4 量子化を省略するため、事前にキャッシュを生成します。出力先は既定で **`<model>.gguf.nvfp4`**（各 tensor の **`.fp4bin`** + **`manifest`**）。

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

バイナリから直接:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-nvfp4-cache
./qwen3-gpu-cuda-nvfp4 ../model.gguf --pack-nvfp4-cache /path/to/cache
./qwen3-gpu-cuda-nvfp4 ../model.gguf --no-nvfp4-cache -p "Hello" -n 64
```

GGUF を更新した場合は **`make pack-cache`** を再実行してください（**`manifest`** が GGUF のサイズ・mtime と一致しないとキャッシュは無効化されます）。

**`fp4_gemm.cu`** / **`fp4_qwen3.cu`** は **C++17** 必須（CUTLASS）。**PolarQuant-R** は [arxiv:2502.02617](https://arxiv.org/abs/2502.02617) に基づく KV 圧縮で、**`head_dim=128` 固定**（Qwen3-VL-8B 向け）。F32 KV 比 **約 8×** の VRAM 削減（36 layer × 512 seq で ~144 MiB → ~18 MiB 概算）。NVFP4 + PolarQuant 同時では線形 FP16 分（8B 級で約 15 GiB 相当）と KV F32 分の両方を削減できます。

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

- **`Loading NVFP4 cache from …`** または **`Uploading weights (fused dequant -> NVFP4 linear layers)...`**
- **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`**
- **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**

### 実行（バイナリを直接）

FP16 版:

```bash
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、CUDAとは何かを初心者向けに説明してください。" \
  -n 64
./qwen3-gpu-cuda ../model.gguf --pack-fp16-cache
./qwen3-gpu-cuda ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

NVFP4 版:

```bash
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、CUDAとは何かを初心者向けに説明してください。" \
  -n 64
```

検証・診断（任意、**Blackwell / sm_120a 必須**）:

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make fp4-test      # CUTLASS GEMM 経路一致・バッチ行・極大活性（down_extreme）
make sfa-verify    # compute_sf_index と CUTLASS layout
make fa-debug      # FP16 vs NVFP4、Hello + --fa-debug
./qwen3-gpu-cuda-nvfp4 ../model.gguf -p "Hello" -n 12 --fa-debug 2>&1 | grep FA_DEBUG
```

設計詳細は **`doc/design.md`** の CUDA 節、調査ログは **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`** を参照してください。

### ベンチマーク履歴（`gpu-cuda/Makefile` / `gpu-cuda-nvfp4/Makefile`）

**`gpu-cuda/`** / **`gpu-cuda-nvfp4/`** / **`gpu-vulkan/`** と **`gpu-rocm/`** 同趣旨。長プロンプト（ChatML 後 **約 132 トークン**）+ decode **最大 128** トークン（**`-n 128 -t 0 -s 42`**）のベンチを実行し、結果を各 **`Makefile` 内の `BENCH_LOG`** に追記できます。第 2 列は **`GPU_SM`**（**`nvidia-smi` の compute capability**、例: **`sm_120`**）。**`gpu-vulkan`** は **`vulkaninfo` 由来の GPU 名**です。

| 変数 | 既定 | 意味 |
|---|---|---|
| `BENCH_PROMPT` | 長文英文（Makefile 内） | ベンチ用プロンプト |
| `BENCH_N` | `128` | 最大生成トークン数（`-n`） |
| `BENCH_SEED` | `42` | 乱数シード（`-s`）。**`-t 0` では無効** |
| `BENCH_TEMP` | `0` | 温度（`-t`）。**`make log.push` の実行行に渡される** |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value ログ出力先 |

```bash
cd qwen3-8b/gpu-cuda
make log.push          # ビルド → ベンチ実行 → Makefile に 1 行追記
make log               # 追記済み BENCH_LOG を表形式で表示
make log.push BENCH_N=64
make log.push BENCH_LOG_FILE=/tmp/my-bench.log

cd qwen3-8b/gpu-cuda-nvfp4
make log.push
make log
```

**注意:** **`make log.push` は各 GPU バリアントの `Makefile`（`gpu-rocm` / `gpu-vulkan` / `gpu-cuda` / `gpu-cuda-nvfp4`）を書き換えます**。コミット前に `git diff` で差分を確認してください。表の **`total_tps`** は **推論区間のみ**（重みの VRAM アップロードは含みません）。

**`BENCH_LOG_FILE` の VRAM 項目**（推論終了時）: **`vram_total`**、**`vram_device_used`** / **`vram_device_total`**（**`cudaMemGetInfo`** / **`hipMemGetInfo`**、各 **bytes** と **`_mib`**）、**`[vram_breakdown]`** セクション。

- **`gpu-rocm`（FP16）**: FP16 embedding / F32 norm / FP16 線形重み（**`vram_weights_linear`**）/ KV / decode 活性 / prefill バッチ（**`d_scratch_f16`** 含む）
- **`gpu-vulkan`（FP16）**: FP16 embedding / F32 norm / FP16 線形重み（**`vram_total` に含む**・個別キー未出力）/ KV / decode 活性 / prefill バッチ
- **`gpu-cuda`（FP16）**: FP16 embedding / F32 norm / FP16 線形重み（**`vram_weights_linear`**）/ KV / decode 活性 / prefill バッチ
- **`gpu-cuda-nvfp4`（NVFP4）**: 上記のうち線形は **`vram_weights_fp4`**、加えて **`vram_fp4_gemm_scratch`**（BF16 活性／出力 + CUTLASS workspace 等）

Makefile の **`BENCH_LOG`** 履歴には tok/s のみ追記され、VRAM 内訳は **`BENCH_LOG_FILE`** を参照してください。

1 行形式（パイ区切り）: **`日時|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

- **`gpu-vulkan`**: 第 2 列は **`vulkaninfo`** 由来の GPU 名
- **`gpu-cuda`**: 第 2 列 **`sm_120`** 等（**`nvidia-smi` 自動検出**）
- **`gpu-cuda-nvfp4`**: 第 2 列は既定 **`sm_120a`**

**`gpu-vulkan`** の長プロンプト実測表・VRAM 内訳は上記 **「Vulkan compute 実装」→「Vulkan GPU 長プロンプト」** を参照（**2026-05-29 09:39** 計測: prefill **2.97** / decode **2.17** / total **2.86** tok/s）。

#### FP16 GPU 長プロンプト（`gpu-cuda`・`make log.push`）

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090（31 GiB VRAM） |
| OS | Linux |
| モデル | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`（**`<model>.gguf.fp16`** キャッシュから H2D） |
| コマンド | **`make log.push`**（**`sm_120`**・**`FA_BR=32`**） |
| ワークロード | 長文プロンプト（ChatML 後 **132** トークン）+ decode **最大 128**（**`-n 128 -t 0 -s 42`**） |
| 表の指標 | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`prefill_tps` / `decode_tps` / `total_tps`** — 推論区間のみ |
| 再現 | `qwen3-8b/gpu-cuda/` で **`make log.push`** → **`make log`** |

| 計測日時 | GPU | prefill tok/s | decode tok/s | total tok/s | 備考 |
|---|---|---:|---:|---:|---|
| 2026-05-28 22:36 | **sm_120** | **62.80** | **75.91** | **64.00** | 132+16 トークン（**`make log.push`**。`-t 0` で EOS により生成 16 で打切） |

**VRAM 内訳**（上記 **2026-05-28 22:36** 計測。**`BENCH_LOG_FILE`** の **`[vram_breakdown]`**。`-l` 既定 **`max_seq=512`** 条件）:

| 項目 | bytes | MiB | 備考 |
|---|---:|---:|---|
| **`vram_total`**（理論合計） | 16,621,944,320 | **15851.92** | 下記カテゴリの合計 |
| **`vram_device_used`** | 17,164,402,688 | **16369.25** | **`cudaMemGetInfo`**（CUDA ランタイム等を含む場合あり） |
| **`vram_device_total`** | 33,669,513,216 | **32109.75** | GPU 全体 VRAM |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm 重み |
| `vram_weights_linear` | 15,136,194,560 | **14435.00** | FP16 線形重み（**`wq`〜`down`** + LM head） |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`**（`-l` 依存） |
| `vram_decode_activations` | 779,776 | **0.74** | 単トークン decode 用バッファ |
| `vram_prefill_batch` | 88,082,432 | **84.00** | prefill バッチ（**`batch_cap = max_seq`**） |

線形重み（**~14435 MiB**）と embedding（**~1187 MiB**）が理論 VRAM の大半（8B 級 FP16 常駐）。KV（**~144 MiB**）は **`-l`** を増やすと比例して増えます。**`vram_device_used`** は **`vram_total`** より大きいことがあります（ドライバ／CUDA 割当の差）。

**PTX `compute_86` JIT** は RTX 5090 では**推論が文字化け**するため非推奨。**`make build`** は **`nvidia-smi` で `CUDA_GENCODE` を自動選択**します（Blackwell → **`sm_120`** + **`FA_BR=32`**）。

#### NVFP4 GPU 長プロンプト（`gpu-cuda-nvfp4`・`make log.push`）

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090（31 GiB VRAM） |
| OS | Linux |
| モデル | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`（起動時 GGUF 融合逆量子化 → NVFP4 H2D） |
| コマンド | **`make log.push`**（**`sm_120a`**・**`FA_BR=32`**） |
| ワークロード | 長文プロンプト（ChatML 後 **132** トークン）+ decode **128** トークン（**`-n 128 -t 0 -s 42`**） |
| 表の指標 | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`prefill_tps` / `decode_tps` / `total_tps`** — 推論区間のみ |
| 再現 | `qwen3-8b/gpu-cuda-nvfp4/` で **`make log.push`** → **`make log`** |

| 計測日時 | GPU | prefill tok/s | decode tok/s | total tok/s | 備考 |
|---|---|---:|---:|---:|---|
| 2026-05-24 15:03 | **sm_120a** | **622.60** | **66.71** | **122.03** | 132+128 トークン（**`make log.push`**・旧 stdout パース） |
| 2026-05-28 22:44 | **sm_120a** | **595.27** | **66.71** | **121.47** | 同上（**`BENCH_LOG_FILE`**） |
| 2026-05-29 02:02 | **sm_120a** | **619.62** | **66.81** | **122.13** | 同上（長 prefill・**`gen=128`**） |
| 2026-05-29 07:28 | **sm_120a** | **4623.10** | **66.64** | **648.35** | 同上（**`gen=13`**・EOS 早期終了。異常時の 128 連打ログとは別） |

**VRAM 内訳**（**最新実装**・**`BENCH_LOG_FILE`** の **`[vram_breakdown]`**。`-l` 既定 **`max_seq=512`** 条件。モデル **`Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**・RTX 5090）:

| 項目 | bytes | MiB | 備考 |
|---|---:|---:|---|
| **`vram_total`**（理論合計） | 8,856,722,944 | **8446.43** | 下記カテゴリの合計 |
| **`vram_device_used`** | 9,639,821,312 | **9193.25** | **`cudaMemGetInfo`**（CUDA ランタイム等を含む場合あり） |
| **`vram_device_total`** | 33,669,513,216 | **32109.75** | GPU 全体 VRAM |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm 重み |
| `vram_weights_fp4` | 6,149,079,040 | **5864.22** | NVFP4 線形重み（packed FP4 + スケール因子 + デバイス側 LUT） |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`**（`-l` 依存） |
| `vram_decode_activations` | 779,776 | **0.74** | 単トークン decode 用バッファ |
| `vram_prefill_batch` | 88,082,432 | **84.00** | prefill バッチ（**`batch_cap = max_seq`**） |
| `vram_fp4_gemm_scratch` | 1,221,894,144 | **1165.29** | BF16 活性／出力バッファ + CUTLASS workspace（**`fp4_qwen3_init`** 時確保） |

NVFP4 線形重み（**~5864 MiB**）と GEMM スクラッチ（**~1165 MiB**）が理論 VRAM の大半。**FP16 経路**（上記 **`gpu-cuda`** 長プロンプト表・理論 **~15852 MiB**）より小さい。embedding（**~1187 MiB**）・KV（**~144 MiB**）・prefill バッチ（**~84 MiB**）は FP16 版と同程度。**`vram_device_used`** は **`vram_total`** より大きいことがあります（ドライバ／CUDA 割当の差）。

**旧計測（中間実装）**: 理論 **`vram_total` ~6655 MiB**（**`vram_weights_fp4` ~4060 MiB**）。現行は上表の **~8446 MiB**（LUT 含む FP4 重み + GEMM スクラッチ）。

### ソースを読む場合

8. `qwen3-8b/gpu-cuda/fp16_cache_io.c` / `main.c` / `kernels.cu` / `polarquant.cu` / `fa_debug.c` — CUDA FP16 版（**`gpu_model_vram_profile`**・**VRAM ベンチログ**・**`--fa-debug`** 含む）。  
9. `qwen3-8b/gpu-cuda-nvfp4/fp4_cache_io.c` / `fp4_qwen3.cu` / `fp4_gemm.cu` / `fp4_verify.cu` — Blackwell NVFP4 版（**`FP4_QUANT_MAX_ABS`**・共有 **`kernels.cu`**）。  
10. `qwen3-8b/gpu-cuda-nvfp4/DEBUG.md` — NVFP4 異常出力の調査・修正ログ（ベース **`433319eb31c3c992536afb5c9a3717084ea5d137`**）。


### よくあるトラブル（GPU / XDNA 付録）

### `nvcc` が見つからない／`nvlink` エラー

CUDA Toolkit の `bin` を `PATH` に通すか、各ディレクトリの `Makefile` の `CUDA_HOME` を確認してください。

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

**RTX 50 系**は **CUDA 13** と **`sm_120` / `sm_120a` ネイティブ**が必要です（**PTX `compute_86` は使わない**）。**`gpu-cuda`** の **`make build`** は **`nvidia-smi` で `CUDA_GENCODE` を自動選択**します。PTX のみのビルドで極端に遅い／出力が壊れる場合は、実機の `sm_XX` を `CUDA_GENCODE` で指定して再ビルドしてください。

### FP16 初回起動が遅い／キャッシュ miss（`gpu-rocm` / `gpu-vulkan` / `gpu-cuda`）

初回は GGUF 行逆量子化が走ります。**`cd gpu-rocm && make pack-cache`**、**`cd gpu-vulkan && make pack-cache`**、または **`cd gpu-cuda && make pack-cache`** で **`<model>.gguf.fp16`** を事前生成してください（**`gpu-rocm`** では **`make build`**（**`MODEL`** 存在時）でも自動 pack）。起動時に **`Warning: FP16 cache miss for …`** が出る場合は **`.fp16bin`** 欠落・形状不一致・**`manifest`** 無効です。**`make pack-cache`** を再実行するか、**`--no-fp16-cache`** で強制再逆量子化します。

### NVFP4 初回起動が遅い／キャッシュ miss

初回は GGUF 行逆量子化 + NVFP4 量子化が走ります。**`make pack-cache`** で **`<model>.gguf.nvfp4`** を事前生成してください。起動時に **`Warning: NVFP4 cache miss for …`** が出る場合は **`.fp4bin`** 欠落・形状不一致・**`manifest`** 無効です。**`make pack-cache`** を再実行するか、**`--no-nvfp4-cache`** で強制再量子化します。

### NVFP4 ビルドが失敗する／`NVFP4 quantize failed`

**`gpu-cuda-nvfp4`** で **`sm_120a`** 向けビルドか、CUDA 13 + **`make cutlass`** 済みかを確認してください。汎用 GPU では **`gpu-cuda`** で **`make build`**（FP16）を使います。apt **CUDA 11** と **CUDA 13** が混在している場合は **`make blackwell`** で整理するか、手動で 11.x を除去してください。

### `make fp4-test` が FAIL する／`Arch conditional MMA instruction... Aborting`

**`fp4_gemm`** を **`compute_86` PTX** のみでビルドした古い **`fp4_gemm.o`** を使っている可能性があります。**`gpu-cuda-nvfp4`** で **`make clean`** → **`make fp4-test`** を再実行してください（**`fp4_gemm.sm120a.o`** は **`sm_120a` + C++17** 固定）。**Blackwell GPU 必須**です。

### NVFP4 で短 prefill 後に `?,` 連打・文字化け

L6 **`down`** 付近で活性が極大化し、NVFP4 量子化スケールが飽和 → GEMM **NaN** → KV 破壊、という経路があります（Flash Attention 単体は正常）。**`FP4_QUANT_MAX_ABS=1024`** 入りの最新ビルドで **`make fp4-test`**（**`down_extreme`** 含む）が PASS するか確認してください。詳細は **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`**。

### NVFP4 で同文繰り返し・「LLM」連発等

旧バイナリの **GEMM prefill と GEMV decode の不一致**、または上記の活性 NaN が原因のことがあります。**`FP4_CACHE_VERSION=2`** で **`make pack-cache`** を再実行し、起動ログ **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** を確認してください。

### NVFP4 で `make log.push` の生成が短い／毎回同じ

長いベンチプロンプトは **EOS で早期終了**しやすく（例: **`gen_tokens=13`**）、**`-t 0`** ではシードは効きません。**`-t 0.8`** かつ **`-s` を変える**と枝が変わる場合があります（**`make log.push BENCH_TEMP=0.8 BENCH_SEED=77`**）。**`BENCH_TEMP`** が実行に渡っているか **`Makefile`** の **`log.push`** 行を確認してください。

### `gpu-cuda-nvfp4` の PolarQuant ビルドが遅い／`Killed`（OOM）

**`kernels.cu`** を **`sm_120a` + FP4 + PolarQuant** でコンパイルするとメモリを多く使います。RAM が不足すると **`Error 137`** で落ちることがあります。対処: スワップを増やす、並列ビルドを止める、または段階的に **`make cutlass`** → **`make build.polarquant`** を再実行してください。

### PolarQuant が有効にならない

起動ログに **`PolarQuant-R: KV cache enabled`** が無い場合、**`build.polarquant`** でビルドしたバイナリか確認してください（`gpu-cuda` または `gpu-cuda-nvfp4`）。**`head_dim=128`** 以外のモデルでは PolarQuant は無効化されます。

### `gpu-rocm` ビルド失敗（C++ headers not found）

**g++ / libstdc++-dev** が未導入の可能性があります。

```bash
sudo apt install -y g++ libstdc++-dev
cd qwen3-8b/gpu-rocm && make clean build
```

### `hipcc` が見つからない

ROCm の場所を確認してください。

```bash
ls /opt/rocm/bin/hipcc
```

別の場所にある場合:

```bash
cd qwen3-8b/gpu-rocm
make build ROCM=/path/to/rocm
```

### `rocBLAS error: … for GPU arch : gfx1152`（Ryzen AI 5 340 等）

rocBLAS に **`gfx1152` 用 Tensile が無い**状態です。**ROCm のマイナーアップのみ**では直らないことがあります。

```bash
cd qwen3-8b/gpu-rocm
make clean build && make run
```

ビルドログで **`Build offload arch: gfx1151`** と **`HSA_OVERRIDE_GFX_VERSION: 11.5.1`** が出ているか確認してください。詳細は上記 **「Ryzen AI / gfx1152 と rocBLAS」** と [`doc/design.md`](doc/design.md)。

### `hipBLAS error: 6`（Prefill 0%）

**`HIPBLAS_STATUS_INTERNAL_ERROR`**。**`gfx1152` 名での rocBLAS ライブラリ symlink のみ**、または **HSA オーバーライドと `--offload-arch` の不一致**が多いです。**`make clean build`** 後 **`make run`**（Makefile の自動設定に任せる）。

### Segmentation fault（Prefill 直後・`gcnArchName: gfx1151` 表示）

**`HSA_OVERRIDE_GFX_VERSION=11.5.1` だけ**設定し、バイナリが **`gfx1152` offload** のままのときに起きやすいです。**`make clean build`** で **`Build offload arch: gfx1151`** を確認してから **`make run`**。

### GPU_ARCH が合わない / 検出に失敗する

**`GPU_ARCH_DETECTED`** は実機の GPU ISA です。通常は `rocminfo` から自動検出されます。**`gfx1152` 実機**では Makefile が内部で **`gfx1151` offload** にマップします。検出失敗時や別 GPU 向けビルドでは手動指定してください。

```bash
rocminfo | awk '/^  Name:/ { n=$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }'
cd qwen3-8b/gpu-rocm
make build GPU_ARCH=gfx1100
make detect-gpu-arch   # Detected / Build offload を表示
```

### ROCm Prefill が遅い（~30 tok/s 程度）

起動ログに **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** があるか確認してください。無い場合や prefill が極端に遅い場合は **`make -C gpu-rocm clean build`** で再ビルドし、**`-lhipblas -lrocblas`** がリンクされているか確認します。詳細は上記 **「Prefill 高速化（概要）」** と **`doc/design.md`** の **「ROCm Prefill 高速化の詳細（3 段階）」** を参照。

### **`undefined reference to hipblas*`** / hipBLAS リンク失敗

**`$(ROCM)/lib`** に **`libhipblas.so`** / **`librocblas.so`** があるか確認してください。ROCm の再インストールまたは **`ROCM=`** パスの修正が必要な場合があります。

### **`make wmma` が FAIL**

**`qwen3-rocm` バイナリに WMMA 命令が含まれる**、**hipBLAS 経路が報告されない**、**`wmma-probe` 校正失敗** 等が考えられます。まず **`make wmma WMMA_SKIP_RUN=1`** で静的チェックのみ実行。**`LLVM_OBJDUMP=$(ROCM)/llvm/bin/llvm-objdump`** を明示。MODEL 未配置時は **`WMMA_SKIP_RUN=1`** を使ってください。

### `/dev/accel/accel0` は開けるが `CREATE_HWCTX` が EINVAL

ドライバが列数・タイル数の組み合わせを拒否していることがあります。`XDNA_NUM_COL=1` を試し、`dmesg` の `amdxdna` メッセージを確認してください（詳細は `doc/design.md` のトラブルシュート）。


---

## AMD Ryzen AI XDNA2 NPU 実装（`xdna2` / `xdna2-bfp16`）について

**`qwen3-8b/xdna2/`** および **`qwen3-8b/xdna2-bfp16/`** も **付録**です。Linux カーネル付属の **`amdxdna` カーネルモジュール**を直叩きする実装で、XRT などの追加ユーザランドは不要ですが、**Ryzen AI 実機と専用 GEMV 制御コード**が前提になります。初めて読む方は無視して構いません。

**AMD XDNA の概要**（設計思想、アーキテクチャの基本構造とタイル、世代別の進化、データ型と精度、ソフトウェアスタック、他社 NPU との比較など）については、別リポジトリに解説記事としてまとめてあります：[thamada/xdna-overview](https://github.com/thamada/xdna-overview)（本文は `main.md`、PDF 付き）。

以下は、XDNA2 NPU 向けの技術メモです。

AMD Ryzen AI（Phoenix / Hawk Point / Strix Point など）に内蔵されている XDNA2 NPU を使う版です。

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
cd qwen3-8b/xdna2
make build
```

成功すると **`qwen3-xdna2`** ができます。

### 実行

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON ツールチェイン**で生成した BF16 GEMV 制御コードバイナリ一式が必要です。`bf16-gemv-<n>x<d>.bin` という命名で `XDNA_GEMV_DIR` 配下に配置します。未配置の場合は OpenMP BF16 GEMV にフォールバックします（**NPU 経路とこの CPU フォールバックは bit-identical**）。

**`xdna2/xdna-gemv/kernels/`** には、名前とレイアウト用の **64 バイト・プレースホルダ**（マジック `GQF3XDNA`）が置いてあります。**実機の ERT には渡されません**（`--xdna-status` では `[STUB]`）。再生成はリポジトリルートで `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels`。本番の NPU 用には MLIR-AIE 等で生成したバイナリに差し替えてください。

環境変数の例: `XDNA_GEMV_DIR`（制御コード検索ディレクトリ）、`XDNA_FORCE_CPU=1`（CPU 強制）、`XDNA_NUM_COL`（列数。`CREATE_HWCTX` が EINVAL になる環境では `XDNA_NUM_COL=1` を試す）。

```bash
cd qwen3-8b/xdna2
# 強制的に CPU フォールバックで動かす場合
XDNA_FORCE_CPU=1 ./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# リポジトリ同梱プレースホルダ（xdna2/ からの相対パス）。実 NPU ctrlcode ではない。
XDNA_GEMV_DIR=xdna-gemv/kernels ./qwen3-xdna2 \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status

# 制御コードが揃っているときは NPU 経路で実行（本物の .bin に差し替え後）
XDNA_GEMV_DIR=xdna-gemv/kernels ./qwen3-xdna2 \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

`Makefile` の `run` を使う場合:

```bash
cd qwen3-8b/xdna2
make run PROMPT="日本語で短く説明してください。"
```

### XDNA2 + BFPX ホスト重み版（`xdna2-bfp16/qwen3-xdna2-bfpx`）

`xdna2-bfp16/main.c` は、**`xdna2/main.c` と同一の DRM ioctl** および **チャンク構成の BF16 GEMV（NPU 経路の枠組み）** を用います。一方で、密な行列レイアウトの重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 の係数）** へ変換し、ホストメモリ上にのみ保持します。GGUF への mmap は、この変換が終わってから解放します。NPU が使えないときの CPU 側のフォールバックでは **`mm_bfpx`** が用いられ、活性値は単精度浮動小数点数のまま、BFPX 形式の重みとの一般行列ベクトル積を計算します。**`xdna2/qwen3-xdna2` とビット単位で完全一致するとは限りません**。量子化に加えブロック近似の誤差があります。**GEMV で量子化 mmap から直接 BF16 へ展開する `xdna2/qwen3-xdna2`** とは経路も誤差の立ち方も異なるため、品質の優劣はケースによります。

```bash
cd qwen3-8b/xdna2-bfp16
make build
XDNA_FORCE_CPU=1 ./qwen3-xdna2-bfpx ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=../xdna2/xdna-gemv/kernels ./qwen3-xdna2-bfpx \
  ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
cd qwen3-8b/xdna2-bfp16
make run PROMPT="日本語で短く説明してください。"
```

### 注意

- **`xdna2/qwen3-xdna2`**: 線形ウェイトは **mmap**（CPU OpenMP 版と同様）。GEMV に使う単一 BF16 スクラッチは **語彙×次元クラスの最大行列**サイズになり得るので、モデルサイズと **VRAM／DRAM に余裕**が必要になる場合があります。恒久の「全レイヤー BF16 二重複製」は行いません。RAM 不足では従来通りプロセスや mmap が失敗し得ます。
- **`xdna2-bfp16/qwen3-xdna2-bfpx`**: 推論中は BFPX とノルム用 F32 が中心で mmap を早めに離せる一方、**変換中**は GGUF mmap とフルテンソル用の一時バッファなどで **ピークメモリが大きくなります**。
- NPU 側で実行する場合は AIE 列を予約するため、同時に動いているほかの NPU ワークロード（Windows Studio Effects 等）と競合する可能性があります。

### ソースを読む場合

9. `qwen3-8b/xdna2/main.c` / `qwen3-8b/xdna2-bfp16/main.c` — `amdxdna` ioctl、CPU フォールバック、BFPX 重み表現。


