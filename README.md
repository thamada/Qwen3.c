# Qwen3.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、**ライブラリに依存せず、単一の C言語ソースから Qwen3系モデルを直接動かす推論実装**です。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。** 推論は **標準Cと `libm`** を中心に、`qwen3-8b/` 内の単一〜少数ソースで完結します。GPU 版は **ROCm/HIP**（`hipcc`）、CPU 並列は **OpenMP**、XDNA2 NPU 版は **Linux カーネルの `amdxdna` DRM ioctl（UAPI）** を直接叩く構成であり、Pythonランタイムや `torch` に依存するレイヤはありません。

上記のうち ROCm/HIP は AMD GPU 向けのコンパイラ・ランタイムであり、**ニューラルネット用の高レベルフレームワークではありません**（ここからさらに自作の HIP カーネルとホストコードで Transformer を組み立てています）。

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

Qwen3系GGUFモデルを、**Cの単一ソース群**から直接動かす小さな推論実装です。実行経路は **CPU／OpenMP／ROCm HIP（AMD GPU）／AMD Ryzen AI XDNA2 NPU（`amdxdna` DRM ioctl の直叩き）**と選べます。

このリポジトリは **Qwen3-VL-8B-Instruct のテキストデコーダ**を対象にしています。画像入力や Vision エンコーダは扱わず、プロンプト文字列を入力してテキストを生成する用途に絞っています。

## まず何ができるのか

このリポジトリでは、`qwen3-8b/` の中にある Cソースをビルドして、次の実行方法を試せます。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `qwen3-8b/main.c` | `qwen3-cpu` | 仕組みを追う、最小構成で動かす |
| CPU OpenMP 並列 | `qwen3-8b/main-omp.c` | `qwen3-cpu-omp` | CPU で少しでも速く試す |
| ROCm/HIP GPU | `qwen3-8b/main-rocm.c` | `qwen3-rocm` | AMD GPU で実用的な速度を狙う |
| AMD Ryzen AI XDNA2 NPU（BF16常駐） | `qwen3-8b/main-xdna2.c` | `qwen3-xdna2` | `amdxdna` カーネルモジュール直叩きで NPU を使う |
| AMD Ryzen AI XDNA2 NPU（BFPXホスト重み） | `qwen3-8b/main-xdna2-bfpx.c` | `qwen3-xdna2-bfpx` | 同上の IOCTL・GEMV パイプラインだが、線形重みをブロック FP（BF16スケール + int8）でホスト保持。GGUF mmap は変換後に解放 |

**AMD XDNA の概要**（設計思想、アーキテクチャの基本構造とタイル、世代別の進化、データ型と精度、ソフトウェアスタック、他社 NPU との比較など）については、別リポジトリに解説記事としてまとめてあります：[thamada/xdna-overview](https://github.com/thamada/xdna-overview)（本文は `main.md`、PDF 付き）。

8B 級モデルの CPU 実行は非常に重いです。最初の動作確認としては CPU でも試せますが、実用的な生成速度を期待する場合は ROCm/HIP 版または XDNA2 NPU 版を使う想定です。VRAM／常駐 DRAM が厳しい場合は **`qwen3-xdna2-bfpx`** が **`qwen3-xdna2`** より常駐が軽くなることがあります（ロード時ピークは別途発生します）。

## ディレクトリ構成

```text
.
├── README.md
├── README.en.md
├── doc/
│   ├── ChangeLog
│   └── design.md
└── qwen3-8b/
    ├── Makefile
    ├── main.c
    ├── main-omp.c
    ├── main-rocm.c
    ├── main-xdna2.c
    ├── main-xdna2-bfpx.c
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

## モデルファイルを置く

`qwen3-8b/Makefile` の既定モデル名は次です。

```text
Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

モデルファイルは著作権とファイルサイズの都合により、リポジトリには含めません。各自で `qwen3-8b/gguf.txt` に記載してある URL から GGUF ファイルをダウンロードし、`qwen3-8b/` の直下に置いてください。

```bash
cd qwen3-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf "$url"
```

配置後、次のようになっていれば準備完了です。

```text
qwen3-8b/
├── Makefile
├── main.c
├── main-omp.c
├── main-rocm.c
├── main-xdna2.c
├── main-xdna2-bfpx.c
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
make build
./qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

うまくいくと、モデル読み込み後に少しずつテキストが表示されます。

## CPU 単スレッド版

### ビルド

```bash
cd qwen3-8b
make build
```

成功すると `qwen3-cpu` ができます。

```bash
ls -lh qwen3-cpu
```

### 実行

```bash
./qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で短く自己紹介してください。" -n 16
```

`Makefile` の `run` ターゲットを使う場合:

```bash
make run PROMPT="日本語で短く自己紹介してください。"
```

別の場所にあるモデルを使う場合:

```bash
make run MODEL=/data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

## CPU OpenMP 版

CPU コアを複数使う版です。単スレッド版と同じモデルを読みます。

### ビルド

```bash
cd qwen3-8b
make build.omp
```

成功すると `qwen3-cpu-omp` ができます。

### 実行

```bash
OMP_NUM_THREADS=8 ./qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "箇条書きで、量子化とは何かを説明してください。" \
  -n 32
```

`OMP_NUM_THREADS` は使う CPU スレッド数です。迷ったら、まずは 4 や 8 から試してください。

```bash
OMP_NUM_THREADS=4 ./qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
OMP_NUM_THREADS=8 ./qwen3-cpu-omp Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
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
make build.rocm GPU_ARCH=gfx1201
```

ROCm が `/opt/rocm` 以外にある場合:

```bash
make build.rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

成功すると `qwen3-rocm` ができます。

### 実行

```bash
./qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
  -p "日本語で、ROCmとは何かを初心者向けに説明してください。" \
  -n 64
```

`Makefile` の `run.rocm` を使う場合:

```bash
make run.rocm GPU_ARCH=gfx1201 PROMPT="日本語で短く説明してください。"
```

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

成功すると `qwen3-xdna2` ができます。

### 実行

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON ツールチェイン**で生成した BF16 GEMV 制御コードバイナリ一式が必要です。`bf16-gemv-<n>x<d>.bin` という命名で `XDNA_GEMV_DIR` 配下に配置します。未配置の場合は OpenMP BF16 GEMV にフォールバックします（**NPU 経路とこの CPU フォールバックは bit-identical**）。

環境変数の例: `XDNA_GEMV_DIR`（制御コード検索ディレクトリ）、`XDNA_FORCE_CPU=1`（CPU 強制）、`XDNA_NUM_COL`（列数。`CREATE_HWCTX` が EINVAL になる環境では `XDNA_NUM_COL=1` を試す）。

```bash
# 強制的に CPU フォールバックで動かす場合
XDNA_FORCE_CPU=1 ./qwen3-xdna2 Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8

# 制御コードが揃っているときは NPU 経路で実行
XDNA_GEMV_DIR=./xdna-kernels ./qwen3-xdna2 \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

`make run.xdna2` も使えます。

```bash
make run.xdna2 PROMPT="日本語で短く説明してください。"
```

### XDNA2 + BFPX ホスト重み版（`qwen3-xdna2-bfpx`）

`main-xdna2-bfpx.c` は、**`main-xdna2.c` と同一の DRM ioctl** および **チャンク構成の BF16 GEMV（NPU 経路の枠組み）** を用います。一方で、密な行列レイアウトの重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 の係数）** へ変換し、ホストメモリ上にのみ保持します。GGUF への mmap は、この変換が終わってから解放します。NPU が使えないときの CPU 側のフォールバックでは **`mm_bfpx`** が用いられ、活性値は単精度浮動小数点数のまま、BFPX 形式の重みとの一般行列ベクトル積を計算します。**`qwen3-xdna2` とビット単位で完全一致するとは限りません**。量子化に加えブロック近似の誤差があるため、**`qwen3-xdna2` に比べ出力が劣化することがあります**。

```bash
cd qwen3-8b
make build.xdna2.bfpx
XDNA_FORCE_CPU=1 ./qwen3-xdna2-bfpx Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
XDNA_GEMV_DIR=./xdna-kernels ./qwen3-xdna2-bfpx \
  Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 8
```

```bash
make run.xdna2.bfpx PROMPT="日本語で短く説明してください。"
```

### 注意

- **`qwen3-xdna2`**: 量子化重みをロード時に **BF16** に一括展開して NPU からも見える DRM バッファに常駐させるため、8B モデルでは **常駐 DDR が ~16 GB** 必要です。RAM が足りない場合は OOM Killer に落とされます。
- **`qwen3-xdna2-bfpx`**: 推論中は BFPX とノルム用 F32 が中心で **常駐は軽め**になりやすい一方、**変換中**は GGUF mmap とフルテンソル用の一時バッファなどで **ピークメモリが大きくなります**。
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
./qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

慣れてきたら `-n` を増やします。

```bash
./qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "日本語で詩を書いてください。" -n 128
```

## 生成を安定させたいとき

同じ入力で結果を比較したい場合は、温度を下げたり seed を固定します。

```bash
./qwen3-rocm Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf \
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

- `qwen3-cpu`
- `qwen3-cpu-omp`
- `qwen3-rocm`
- `qwen3-xdna2`
- `qwen3-xdna2-bfpx`

モデルファイルは `make clean` では削除されません。

## よくあるトラブル

### `No such file or directory` と出る

モデルファイルの場所が間違っている可能性があります。

```bash
ls -lh qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
```

見つからない場合は、モデルを `qwen3-8b/` に置くか、実行時に絶対パスを指定してください。

```bash
./qwen3-cpu /data/models/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

### CPU 版が遅い

正常です。8B 級モデルは CPU だけで動かすには重いです。まずは `-n 1` や `-n 4` で確認してください。

```bash
./qwen3-cpu Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 1
```

速度が必要なら `qwen3-rocm` を使ってください。

### `hipcc` が見つからない

ROCm の場所を確認してください。

```bash
ls /opt/rocm/bin/hipcc
```

別の場所にある場合:

```bash
make build.rocm ROCM=/path/to/rocm GPU_ARCH=gfx1201
```

### GPU_ARCH が合わない

`GPU_ARCH` は実機の GPU ISA に合わせる必要があります。

```bash
rocminfo | grep -m 1 gfx
```

表示された値を使います。

```bash
make build.rocm GPU_ARCH=gfx1100
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

3. `qwen3-8b/main.c`  
   CPU 版で、GGUF 読み込みから 1 トークン生成までを追う。

4. `qwen3-8b/main-omp.c`  
   OpenMP による並列化箇所を見る。

5. `qwen3-8b/main-rocm.c`  
   GPU メモリ、HIP カーネル、GPU サンプリングの流れを見る。

6. `qwen3-8b/main-xdna2.c` / `qwen3-8b/main-xdna2-bfpx.c`  
   `amdxdna` ioctl、ERT/NPU GEMV、CPU フォールバック。BFPX 版は `bfpx_convert_weight_2d` と mmap 解放パスを読む。

## このリポジトリで扱わないもの

- 学習、ファインチューニング
- バッチ推論の最適化
- 画像入力
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- 公式実装との完全な数値一致保証

目的は、Qwen3系GGUFのテキスト推論を C/HIP で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- 設計仕様: `doc/design.md`
- 変更履歴: `doc/ChangeLog`

困ったときは、まず `qwen3-8b/Makefile` のターゲット名と、実行時に渡しているモデルパスを確認してください。ビルドと実行の大半の問題は、この 2 つの不一致から起きます。
