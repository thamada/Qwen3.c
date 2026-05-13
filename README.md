# Qwen3.c

Qwen3 系 GGUF モデルを、C 言語と ROCm/HIP で直接動かすための小さな推論実装です。

このリポジトリは **Qwen3-VL-8B-Instruct のテキストデコーダ**を対象にしています。画像入力や Vision エンコーダは扱わず、プロンプト文字列を入力してテキストを生成する用途に絞っています。

## まず何ができるのか

このリポジトリでは、`qwen3-8b/` の中にある C ソースをビルドして、次の 3 種類の実行方法を試せます。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `qwen3-8b/main.c` | `qwen3-cpu` | 仕組みを追う、最小構成で動かす |
| CPU OpenMP 並列 | `qwen3-8b/main-omp.c` | `qwen3-cpu-omp` | CPU で少しでも速く試す |
| ROCm/HIP GPU | `qwen3-8b/main-rocm.c` | `qwen3-rocm` | AMD GPU で実用的な速度を狙う |

8B 級モデルの CPU 実行は非常に重いです。最初の動作確認としては CPU でも試せますが、実用的な生成速度を期待する場合は ROCm/HIP 版を使う想定です。

## ディレクトリ構成

```text
.
├── README.md
├── doc/
│   ├── ChangeLog
│   └── design.md
└── qwen3-8b/
    ├── Makefile
    ├── main.c
    ├── main-omp.c
    ├── main-rocm.c
    └── Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum
```

主に触る場所は `qwen3-8b/` です。ビルドも推論実行も、基本的にはこのディレクトリに移動してから行います。

## 初心者向け: LLM 推論で何が起きるか

LLM 推論は、大まかには次の流れです。

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

このリポジトリの特徴は、この流れを巨大なフレームワークに隠さず、C ソースの中で追えることです。

## 必要なもの

### 共通

- Linux
- `make`
- C コンパイラ（例: `gcc`, `clang`, `cc`）
- `libm`（通常は標準で入っています）
- Qwen3-VL-8B-Instruct の GGUF モデルファイル

Ubuntu 系なら、CPU 版に必要な基本ツールは次で入ることが多いです。

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

### `sha256sum -c` が失敗する

ファイル名または中身が、このリポジトリの想定と違います。次を確認してください。

- GGUF ファイル名が `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` になっているか
- ダウンロードが途中で壊れていないか
- 別量子化のモデルを置いていないか

別モデルを使う場合は、ハッシュ確認は一致しなくて当然です。その場合でも実装が対応するメタデータ・テンソル構造である必要があります。

## 実装を読みたい人へ

最初に読むなら、次の順番がおすすめです。

1. `README.md`  
   まずビルドと実行を成功させる。

2. `doc/design.md`  
   全体の設計、量子化、Qwen3 固有処理を把握する。

3. `qwen3-8b/main.c`  
   CPU 版で、GGUF 読み込みから 1 トークン生成までを追う。

4. `qwen3-8b/main-omp.c`  
   OpenMP による並列化箇所を見る。

5. `qwen3-8b/main-rocm.c`  
   GPU メモリ、HIP カーネル、GPU サンプリングの流れを見る。

## このリポジトリで扱わないもの

- 学習、ファインチューニング
- バッチ推論の最適化
- 画像入力
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- 公式実装との完全な数値一致保証

目的は、Qwen3 系 GGUF のテキスト推論を C/HIP で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- 設計仕様: `doc/design.md`
- 変更履歴: `doc/ChangeLog`

困ったときは、まず `qwen3-8b/Makefile` のターゲット名と、実行時に渡しているモデルパスを確認してください。ビルドと実行の大半の問題は、この 2 つの不一致から起きます。
