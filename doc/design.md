# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog`を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

### リポジトリの目的とスコープ

本リポジトリは、**Qwen3 系（Qwen3-VL-8B-Instruct）** の **GGUF** 形式モデルを、**単一または少数の C/HIP ソースファイル**からビルド可能な形で **推論（テキスト生成）**するエンジンである。**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムにはリンクしない。** コアは **標準 C と `libm`**。GPU 版は **ROCm/HIP**（コンパイラ・ランタイムでありニューラルネット用の高レベルフレームワークではない）、CPU 並列は **OpenMP**、XDNA2 NPU 版は **`amdxdna` DRM ioctl（UAPI）** を直接利用する。Python ランタイムや `torch` に依存する層は置かない。**GGUF の読み取り・トークナイズ・Transformer フォワード・サンプリング**を一連のコードパスとして理解・改変しやすくすることを目的とする。学習・ファインチューニング・バッチ推論の最適化はスコープ外であり、主に **対話形式のインタラクティブ生成**（プロンプト＋続きの生成）を想定する。

#### ライブラリ非依存とその意義

高レベルフレームワークに載せた推論は実装が簡潔になり高速化もしやすいが、**計算手順・メモリ配置・アライメント・量子化レイアウト**等がランタイム内部に隠れやすい。本リポジトリはその抽象層に依存せず、推論の実体を **C の明示的なコードパス**として観察・検証・変更できる状態に置く。フレームワークの代替を第一目的とするものではない。

- **理解可能性**: モデルファイルからの読み取り、バッファ配置、演算順序をソースと本書で追跡できる。
- **依存関係の単純化**: Python 環境や大規模 ML スタックを前提とせず、コンパイラと必要最小限の実行環境で経路を確認できる。
- **実験の自由度**: 量子化・メモリ表現（例: BFPX）・CPU/GPU/NPU の分担・`/dev/accel` への直接アクセスなど、抽象化に縛られやすい領域を試しやすい。
- **参照実装としての価値**: 最小構成で Qwen3 系デコーダ推論が成立する見取り図として、他スタックとの比較・検証の基準になる。

**最高性能や機能網羅を第一目的とはしない。** 主眼は、LLM 推論をブラックボックスにせず、実装の細部を把握したうえで改造できることである。利用者向けの入口説明は **`README.md`**（日本語）および **`README.en.md`**（英語）に詳しい。

文中の「decoder-only」「GQA」「FlashAttention 系デコードカーネル」等は、**Transformer デコーダの一般的なパターン**を指す。**実装はすべて `qwen3-8b/` に置かれる。** 対象例は **Qwen3-VL-8B-Instruct** の **IQ2_S / IQ3_S 等が混在した GGUF**（例: `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`）。**Vision（画像エンコード・deepstack・画像トークン）は実装しない**。**テキスト用デコーダのみ**を実行する。

### 実装バリアント（本リポジトリに含まれるもの）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `main.c` | CPU、単スレッド | GGUF mmap、`qwen3vl.*` パース。線形層は **IQ2_S / IQ3_S / Q4_K / Q5_K** 等を **`QK_K=256` ブロック単位**にデ量子化しつつ GEMV（全重みの float 一括展開なし）。`libm` のみ。 |
| `main-omp.c` | CPU、**OpenMP** | 上記と同一アルゴリズム。**GEMV** は出力行並列、**Attention** はヘッド並列、`main-rocm.c` のカーネル粒度に相当する並列化（RoPE、RMSNorm、残差、SiLU 等）。 |
| `main-rocm.c` | **ROCm / HIP** | ロード時に量子化重みを CPU で **F16** に展開して VRAM に載せ、**フル GPU** パスで推論。**Flash 系デコード注意**・**KV カーネル書き込み**・**レイヤー間のホスト非介在**・GPU サンプリング（top-p 時は logits D2H フォールバック）等を含む。**`make build.rocm` の既定エントリ**。 |
| `main-xdna2.c` | **AMD Ryzen AI NPU (XDNA2)** | **`main-omp.c` と同様**に線形ウェイトは **GGUF mmap 上の量子化形式を参照**。埋め込みは行単位ブロック復号。各 **GEMV ごとに**当該重み行列を **`AMDXDNA_BO_SHMEM` の単一 BF16 スクラッチ**へ展開して NPU が DMA、`scratch_f32` でデ量子化～BF16 を兼用。rmsnorm などの小型 F32 も mmap 指す。`DRM ioctl` と **`ERT_START_NPU`** 経路、`/dev/accel/accelN` 不可／制御コード未配置時の **OpenMP BF16 CPU フォールバック（NPU と bit-identical）**は従来どおり。XRT 不要・UAPI inline 持ち運びは不変。**スクラッチサイズはテキスト経路 GEMV に必要な最大要素数のみ**（パーサ済み名前走査、`TensorInfo` は推論前に開放しうる）。**起動時レポートと `--xdna-status` / `-X`** で各形状の **`bf16-gemv-<n>x<d>.bin`** 可否・推論後の NPU/CPU GEMV カウンタを確認できる。 |
| `main-xdna2-bfpx.c` | **AMD Ryzen AI NPU (XDNA2) + BFPX ホスト重み** | **`main-xdna2.c` と同一の DRM ioctl** および **チャンク BF16 GEMV（NPU 経路の枠組み）** を共有する。**密行列レイアウト**の重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 係数、ブロック長 64）** に変換しホストのみ保持し、GGUF mmap は変換完了後に解放する。**論理形状は `main-omp.c` の `mm(..., n_in, n_out)` と一致**させ、`[n_in,n_out]` 型の GGUF 転置は **`bfpx_convert_weight_2d`** で吸収。量子化に加えブロック近似のため、**GEMV で逐次 BF16 に展開する `main-xdna2.c` と同一ビットでの一致は期待できず**、品質が劣ることがある。NPU 不可時の CPU は **`mm_bfpx`** が単精度浮動小数点数の活性と BFPX 形式の重みの積を計算する。 |

メタデータキーは **`qwen3vl.*`**。Qwen3 固有として、線形射影の直後に **`attn_q_norm` / `attn_k_norm`**（ヘッド長に対する RMSNorm）を挟み、その後 **RoPE** を適用する。チャットは **ChatML**（`<|im_start|>` / `<|im_end|>` 等）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` | ビルド・実行・方針の説明（日本語）。 |
| `README.en.md` | 同上（英語）。 |
| `qwen3-8b/main.c` | CPU 単スレッド推論。 |
| `qwen3-8b/main-omp.c` | CPU OpenMP 並列推論。 |
| `qwen3-8b/main-rocm.c` | ROCm 推論（既定の HIP ビルド対象）。 |
| `qwen3-8b/main-xdna2.c` | AMD Ryzen AI（XDNA2）NPU。**mmap ウェイト + GEMV 毎 BF16 スクラッチ**・`amdxdna` ioctl 直叩き。**`--xdna-status` / `-X`** で制御コード環境の軽量診断。 |
| `qwen3-8b/main-xdna2-bfpx.c` | **`main-xdna2.c` と同一の IOCTL／チャンク BF16 GEMV（枠組み）。密行列レイアウトの重みをロード時に BFPX 化しホストのみ保持、mmap は変換完了後に解放。** |
| `qwen3-8b/Makefile` | 各ソース向け **`build*` / `run*` / `clean`**。**`qwen3-*`** 出力名はレシピに直書き（**`TARGET_*` は使わない**）。上書き用 **`?=` 変数**は下記「共通」を参照（**`.PHONY` / `clean` は複数行で列挙**）。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog` | 変更履歴。 |
| `.gitignore` | バイナリ等の除外。 |
| `qwen3-8b/gguf.txt` | 既定 GGUF の取得元 URL 参照。Hugging Face の `blob/main` URL を `resolve/main` に置換して `wget` できる。 |
| `qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum` | 既定 GGUF の SHA256 参照。 |

### 生成バイナリと Make ターゲット（`qwen3-8b/`）

作業ディレクトリは **`qwen3-8b/`**。既定 `MODEL=Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`、`GPU_ARCH` 既定例は `gfx1201`（実機の `rocminfo` に合わせる）。

| Makefile ターゲット | 出力バイナリ | ソース |
|---------------------|--------------|--------|
| `build` / `run` | `qwen3-cpu` | `main.c` |
| `build.omp` / `run.omp` | `qwen3-cpu-omp` | `main-omp.c`（`-fopenmp`、`OMP_NUM_THREADS`） |
| `build.rocm` / `run.rocm` | `qwen3-rocm` | `main-rocm.c` |
| `build.xdna2` / `run.xdna2` | `qwen3-xdna2` | `main-xdna2.c`（`-fopenmp`。`amdxdna` カーネルモジュールが `/dev/accel/accelN` を提供） |
| **`build.xdna2.bfpx` / `run.xdna2.bfpx`** | **`qwen3-xdna2-bfpx`** | **`main-xdna2-bfpx.c`**（`-fopenmp`。NPU 経路・環境変数は `qwen3-xdna2` と同種。ホスト重みは BFPX） |

```bash
cd qwen3-8b
make build
make build.omp
make build.rocm              # hipcc・ROCm 必須
make build.xdna2             # Linux >= 6.10 + amdxdna カーネルモジュール（XRT 不要）
make build.xdna2.bfpx        # 同上 + BFPX ホスト重み版バイナリ
OMP_NUM_THREADS=8 ./qwen3-cpu-omp "$(MODEL)" -p "Hello" -n 4
```

**CPU（IQ 混在 8B）**はブロック単位デ量子化のため **非常に遅くなり得る**。実用スループットは **ROCm 版**を優先する想定である。

## ビルドと実行

### 共通（`qwen3-8b/Makefile`）

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `CC` | C コンパイラ（CPU / OpenMP / XDNA） | `cc` |
| `CFLAGS` | C コンパイルフラグ | `-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter` |
| `LDFLAGS` | リンクフラグ・ライブラリ | `-lm` |
| `ROCM` | ROCm ルート | `/opt/rocm` |
| `HIPCC` | HIP コンパイラ | `$(ROCM)/bin/hipcc` |
| `GPU_ARCH` | `--offload-arch=`（**行末に `# …` と同書きしない**。値末尾空白で HIP が失敗することがあるので、説明は別行コメントへ） | `gfx1201` |
| `XDNA_INCS` | `<drm/drm.h>` が標準外にあるときだけ付与する `-I…`（`build.xdna2` / `build.xdna2.bfpx`） | 未定義（空で可） |
| `MODEL` | GGUF パス | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |
| `PROMPT` | ユーザプロンプト文字列 | `Hello, how are you?` |

### CPU

```bash
cd qwen3-8b
make build
make run PROMPT="質問" MODEL=path/to/model.gguf
```

### CPU（OpenMP）

```bash
cd qwen3-8b
make build.omp
OMP_NUM_THREADS=8 ./qwen3-cpu-omp path/to/model.gguf -p "Hi" -n 8
```

### ROCm

```bash
cd qwen3-8b
make build.rocm GPU_ARCH=gfx1201
make run.rocm PROMPT="Hello"
```

### XDNA2（AMD Ryzen AI NPU）

```bash
cd qwen3-8b
make build.xdna2
# ユーザを render グループに追加して /dev/accel/accel0 を開けるようにしておく
sudo usermod -aG render "$USER"
# 既定では NPU 行列乗算用の制御コードバイナリは XDNA_GEMV_DIR から検索する。
# 未設定/未配置の場合は OpenMP CPU フォールバックに自動切り替え。
XDNA_GEMV_DIR=./xdna-kernels ./qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8
# 強制的に NPU を使わず CPU OpenMP で実行する場合:
XDNA_FORCE_CPU=1 ./qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8
# 上級者向け: ハードウェアコンテキストの試行値を直接上書き
#   XDNA_NUM_COL=<n>     CREATE_HWCTX で要求する列数の上限
#   XDNA_NUM_TILES=<n>   num_tiles を直接固定（core.row_count 整数倍が必要）
#   XDNA_HEAP_SIZE=<bytes>  DEV_HEAP のサイズ（既定 64 MiB、ファーム上限による）
XDNA_NUM_COL=1 XDNA_HEAP_SIZE=33554432 ./qwen3-xdna2 path/to/model.gguf --xdna-status
# NPU / XDNA_GEMV_DIR / 各形状の bf16-gemv-<n>x<d>.bin の可否だけ確認し終了（重みロード・推論なし）
./qwen3-xdna2 path/to/model.gguf --xdna-status
# 同上の短い別名
./qwen3-xdna2 path/to/model.gguf -X
# BFPX ホスト重み版（ビルド後バイナリは qwen3-xdna2-bfpx）
make build.xdna2.bfpx
./qwen3-xdna2-bfpx path/to/model.gguf -p "Hi" -n 8
```

## 実行時の挙動

**CPU（`qwen3-cpu` / `qwen3-cpu-omp`）**: 重みは mmap 上の GGUF を参照。量子化行は都度ブロックデ量子化してから内積。KV・活性は主に float32。サンプリングはホスト上の logits に対して実施。

**ROCm（`qwen3-rocm`）**: ロード時に F16 重みを VRAM に配置。各ステップは **埋め込み〜全レイヤー〜LM ヘッド**を GPU 上で実行。教師強制区間では LM ヘッドを省略可能。**`0 < top-p < 1`** の nucleus は実装上 **logits 全語彙を D2H** して CPU で処理する場合がある（実装コメント参照）。それ以外は GPU で argmax / softmax＋多項サンプル等。

**XDNA2（`qwen3-xdna2`）**: 線形ウェイトは **mmap された GGUF** を **`main-omp.c` と同様**に参照する（埋め込みは mmap 上行の量子化レイアウトからブロック単位復号）。各 **GEMV** のたび、その行列だけを **`AMDXDNA_BO_SHMEM` に確保した単一 BF16 スクラッチ**へ CPU で復号・BF16 化し、`SYNC_BO` でデバイス可視にしたうえで、入力 BF16・重み・出力への `xdna_addr` を **`ERT_START_NPU`** で `DRM_IOCTL_AMDXDNA_EXEC_CMD` に渡す構成は従来どおり。**レイヤー分の恒久 BF16 重み BO は保持しない**。RMSNorm／Qwen3 ヘッド RMSNorm／Attention 等も **CPU**。NPU が使えないときは BF16 GEMV が **OpenMP** にフォールバックする（実装どおり bit-identical）。

推論開始前に **`=== XDNA GEMV / NPU ctrlcode status ===`** ブロックを標準出力へ出し、`XDNA_FORCE_CPU`・DRM オープン可否・`XDNA_GEMV_DIR`・テキスト経路で使う **6 種類の GEMV 形状**それぞれについて `bf16-gemv-<n>x<d>.bin` が **`access(R_OK)` で読めるか**とフルパスを表示する。推論後は **NPU GEMV 回数／CPU GEMV 回数**に加え、**すべて NPU／すべて CPU／混在**を短文で表示する（実際に `EXEC_CMD` が成功したかはランタイムカウントが基準）。**`--xdna-status`** または **`-X`** は GGUF パースと `npu_open` のみ行い当該レポートを出力して **終了**する（重みロード・生成ループなし）。

**XDNA2 + BFPX（`qwen3-xdna2-bfpx`）**: IOCTL 系列および **チャンク BF16 GEMV（NPU 経路の枠組み）** は **`qwen3-xdna2`** と同様。ただし常駐重みは **ホストの BFPX バッファ**とし、各チャンクを BF16 に展開して SHMEM BO へステージングしてから NPU に載せる。NPU が使えないときの CPU 側は **`mm_bfpx`** が、単精度浮動小数点数の活性と BFPX 形式の重みとで一般行列ベクトル積を計算する（常に CPU のみになる場合もある）。GGUF mmap は線形～BFPX 変換の完了後に解放する。**ロード時ピーク**には GGUF 全体の mmap とフルテンソル換算の一時 F32 などが乗り、メモリを大きく使う。**`qwen3-xdna2` と出力がビット単位で完全一致するとは限らない**。量子化に加えブロック近似がある。**逐次 GEMV で BF16 へ復号する `qwen3-xdna2`** に較べてホスト側の恒久表現や誤差の立ち位置が異なるため、品質や速度の優劣はケースによる。

## コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `-p <prompt>` | ユーザープロンプト | `Hello` |
| `-n <tokens>` | 最大生成トークン数 | `256` |
| `-t <temp>` | Temperature | `0.6` |
| `-k <topp>` | Top-p サンプリング | `0.9` |
| `-s <seed>` | 乱数シード | `time(NULL)` |
| `-l <len>` | 最大シーケンス長 | `512` |

**補足（`qwen3-xdna2` のみ）**: コマンドラインの任意位置に **`--xdna-status`** または **`-X`** を付与すると、上記 **GEMV／ctrlcode 状態レポート**のみ出力して終了する（他オプションと併用可。モデルパスは第 1 引数のまま）。

## アーキテクチャ

### 実装のレイヤー構成

各実装ファイルは、外部ライブラリに分割せず、ほぼ同じ順序で機能を持つ。

1. **GGUF と量子化形式の定義**: GGUF の値型、GGML tensor dtype、`QK_K=256` の K-quant / IQ ブロック構造を定義する。`BlockQ4_K`、`BlockQ5_K`、`BlockIQ2_S`、`BlockIQ3_S` は GGML の packed layout に合わせて `#pragma pack(push, 1)` で定義する。
2. **IQ2_S / IQ3_S の復元テーブル**: `iq2s_grid`、`iq3s_grid` などを持ち、GGML 側の小さな格子表現を `float` に戻す。
3. **モデル構造体**: `Config` がモデル形状、`TensorInfo` が GGUF 内 tensor descriptor、`Tok` が tokenizer、`Weights` / `WeightsDev` が重み、`State` が実行時バッファ、`Model` がそれらをまとめる。
4. **ロード処理**: `mmap` した GGUF からメタデータと tensor descriptor を読み、CPU 版は tensor へのポインタを保持し、ROCm 版は重みを GPU にアップロードする。
5. **推論処理**: 1 トークン単位の forward を繰り返し、プロンプト区間は teacher forcing、生成区間は logits から次トークンを選ぶ。

この構成により、CPU 版は「GGUF の量子化重みをその場で読む参照実装」、ROCm 版は「同じモデル構造を GPU 常駐重みに変換して動かす実装」として対応づけられる。

### 主要データ構造

`Config` は `dim`、`hidden_dim`、`n_layers`、`n_heads`、`n_kv_heads`、`vocab_size`、`max_seq`、`rope_theta`、`norm_eps` を保持する。Qwen3-VL では `head_dim` を `qwen3vl.attention.key_length` から読む。値が無い場合のみ `dim / n_heads` にフォールバックし、`kv_dim = n_kv_heads * head_dim`、`kv_mul = n_heads / n_kv_heads` を派生させる。

`TensorInfo` は GGUF tensor の `name`、次元数、各次元長 `ne[4]`、dtype、data section 内 offset を持つ。実データ位置は `fdata + doff + offset` で求める。`doff` は `general.alignment`（既定 32）に基づいて tensor data section の開始位置へ丸めた値である。

`Tok` は語彙文字列、語彙長、BPE score、特殊トークン ID、ハッシュ表、byte fallback 用 token を持つ。`<|im_start|>` と `<|im_end|>` は ChatML 用に語彙から探索し、見つかった場合は `im_start` / `im_end` として保存する。

`State` は forward 中の一時バッファを持つ。主なものは hidden state `x`、RMSNorm 後や射影後に使う `xb` / `xb2`、FFN の `hb` / `hb2`、attention の `q` / `k` / `v`、logits、KV cache `kc` / `vc` である。KV cache は **`n_layers * max_seq * kv_dim`** 要素を Key と Value でそれぞれ確保する。

### GGUF パーサー

GGUF パーサーは v2 以上を対象とする。先頭の magic と version を検証し、metadata key-value、tensor descriptor、tensor data offset を順に読む。未知の metadata は `skip` で読み飛ばし、実装が必要とする key のみ `Config` / `Tok` に反映する。

モデル形状は **`qwen3vl.*`** の metadata から読む。主な key は次の通りである。

- `qwen3vl.embedding_length`: `dim`
- `qwen3vl.feed_forward_length`: `hidden_dim`
- `qwen3vl.block_count`: `n_layers`
- `qwen3vl.attention.head_count`: `n_heads`
- `qwen3vl.attention.head_count_kv`: `n_kv_heads`
- `qwen3vl.attention.key_length`: `head_dim`
- `qwen3vl.attention.layer_norm_rms_epsilon`: `norm_eps`
- `qwen3vl.rope.freq_base`: `rope_theta`

Tokenizer は `tokenizer.ggml.tokens`、`tokenizer.ggml.scores`、`tokenizer.ggml.merges`、`tokenizer.ggml.bos_token_id`、`tokenizer.ggml.eos_token_id` を読む。`tokenizer.ggml.merges` は後段の BPE score 初期化に使うため、metadata 読み込み中に一時的な文字列配列として保持する。

### 重みテンソルの対応

各レイヤー `L` は次の tensor 群を要求する。

- Attention norm: `blk.L.attn_norm.weight`
- Q/K/V/O: `blk.L.attn_q.weight`、`blk.L.attn_k.weight`、`blk.L.attn_v.weight`、`blk.L.attn_output.weight`
- Qwen3 固有の head norm: `blk.L.attn_q_norm.weight`、`blk.L.attn_k_norm.weight`
- FFN norm: `blk.L.ffn_norm.weight`
- SwiGLU FFN: `blk.L.ffn_gate.weight`、`blk.L.ffn_up.weight`、`blk.L.ffn_down.weight`

全体では `token_embd.weight`、`output_norm.weight`、`output.weight` を使う。CPU 実装は Qwen3-VL Instruct の前提として `output.weight` を必須にしている。ROCm 実装は `output.weight` が無い場合だけ `token_embd.weight` を LM head として再利用するが、既定モデルでは untied embedding のため通常は `output.weight` が存在する。

### 量子化と行列積

対象 GGUF では、norm 系は主に F32、埋め込み・射影・FFN・LM head は **IQ2_S / IQ3_S / Q4_K / Q5_K** などの混在になる。すべての K-quant / IQ block は `QK_K=256` 要素を単位に復元する。

CPU 版は重み全体を float に展開しない。`mm_quant_rows` が出力行ごとに量子化 row を走査し、row 内の各 256 要素 block を stack 上の `float blk[QK_K]` に復元して、入力ベクトルとの内積に足し込む。これによりメモリ使用量は抑えられるが、同じ重みを毎 token で復元するため速度は遅い。

ROCm 版はロード時に一度だけホスト上で量子化 tensor を F32 に復元し、さらに F16 staging buffer に変換して GPU にアップロードする。norm tensor は F32 のまま GPU に置く。実行時の GEMV は `mm_f16_gemv_kernel` で FP16 重みを読み、accumulator は float で計算する。この方針はロード時間と VRAM 使用量を増やす代わりに、推論中の量子化復元を避けて GPU カーネルの種類を単純にする。

### トークナイザーと ChatML

Tokenizer は GPT-2 系の byte-level BPE として実装する。まず入力文字列の各 byte を vocabulary 内の byte token に変換し、その後、隣接 token の連結が語彙に存在し、かつ BPE score が高いものを繰り返し merge する。`tokenizer.ggml.merges` がある場合は merge 順位から score を構成し、無い場合は GGUF 内の scores を使う。

プロンプトは `chat_encode` により固定の ChatML 形式へ変換される。

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

出力時は特殊トークンを表示せず、GPT-2 byte fallback の Unicode codepoint 表現を raw byte に戻して端末へ書き出す。

### CPU forward

CPU 版の forward は、すべて `float` の activation buffer 上で逐次実行する。重みは GGUF mmap 上の raw tensor を参照し、dtype に応じて `mm_f32`、`mm_f16`、`mm_quant_rows` に分岐する。

1 token の処理は次の順序である。

1. `token_embd.weight` から token ID の行を読み、`x` に展開する。
2. 各レイヤーで `attn_norm` による RMSNorm を `xb` に出す。
3. `xb` に対して Q/K/V の GEMV を行い、`q` / `k` / `v` を作る。
4. Qwen3 固有の `attn_q_norm` / `attn_k_norm` を head ごとに in-place 適用する。
5. Q/K に RoPE を適用し、現在位置 `pos` の K/V を KV cache に書く。
6. GQA に従い、query head `h` は `kvh = h / kv_mul` の KV head を参照する。過去 `0..pos` の score を softmax し、Value の重み付き和を作る。
7. attention 出力を `attn_output.weight` で射影し、残差として `x` に加える。
8. `ffn_norm`、`ffn_gate`、`ffn_up`、`SiLU(gate) * up`、`ffn_down` の順で FFN を実行し、再び残差を加える。
9. logits が必要な位置だけ `output_norm` と `output.weight` を実行する。

プロンプト消費中は次 token が既知なので、最後のプロンプト token 以外では LM head を省略できる。

### ROCm forward

ROCm 版の `forward_gpu` は embedding から LM head まで device buffer 上で実行する。各カーネル起動の依存は default stream の順序に任せ、forward の最後に `hipDeviceSynchronize()` する。

主なカーネルは次の通りである。

- `emb_f16_kernel`: token embedding の 1 行を FP16 から float activation に展開する。
- `rmsnorm_kernel`: block 内 reduction で二乗平均を求め、`float4` 単位も使って RMSNorm を適用する。
- `mm_f16_gemv_kernel`: 1 warp が 1 出力行を担当し、warp reduction で GEMV の和を作る。1 block は複数行を処理する。
- `rmsnorm_head_kernel`: Q/K の各 head を 1 block で処理し、Qwen3 の head RMSNorm を in-place で適用する。
- `rope_kernel`: head 内の偶数・奇数ペアに対して RoPE 回転を適用する。
- `kv_cache_write_kernel`: 現在 token の K/V を layer offset と position offset から求めた cache 位置へ書く。
- `attn_flash_decode_kernel_hd128` / `attn_flash_decode_kernel`: decode 用 attention。query head ごとに 1 block を使い、過去 token を tile 化しながら online softmax の形で max と分母を更新する。
- `attn_mha_kernel`: head dimension が Flash decode の上限を超える場合の fallback。
- `silu_mul_kernel`: `SiLU(gate) * up` を要素ごとに計算する。
- `vec_add_kernel`: attention / FFN の残差加算を行う。

Qwen3-VL-8B の代表形状では `head_dim=128` なので、専用の `attn_flash_decode_kernel_hd128` が使われる。これは K tile を shared memory に置き、Q と K の dot、online softmax、Value の重み付き和を 1 kernel 内で処理する。attention score 行列全体を global memory に持たないため、decode 時のメモリ転送を抑えられる。

### 生成ループとサンプリング

生成ループは `prompt[0]` から開始し、`pos` を 0 から進める。`pos < n_prompt - 1` の間は teacher forcing として `prompt[pos + 1]` を次 token に使う。`pos >= n_prompt - 1` になったら logits から次 token をサンプリングし、`eos` または `eot` なら停止する。`max_seq` を超える場合も停止する。

サンプリングは次の分岐を持つ。

- `temp <= 0`: greedy。ROCm 版は GPU argmax 経路を使える。
- `temp > 0` かつ `top-p` 無効相当: logits を temperature で割り、softmax 後に多項サンプルする。ROCm 版は GPU softmax / multinomial 経路を使える。
- `0 < top-p < 1`: nucleus sampling。ROCm 版でも語彙全体の logits を host に戻して CPU で sort / 累積確率処理を行う fallback がある。

乱数は xorshift 系の 64-bit state を使う。seed が 0 の場合は 1 に置き換え、CPU fallback と GPU sampling の間で host 側の state を同期する。

## モデル参照

利用する GGUF のファイル名は **`qwen3-8b/Makefile` の `MODEL`** を参照する。モデル本体は著作権とファイルサイズの都合でリポジトリに含めず、既定モデルの取得元は **`qwen3-8b/gguf.txt`** に URL として置く。ダウンロード時は Hugging Face の `blob/main` URL を `resolve/main` に置換して実体ファイルを取得する。ハッシュ確認は例として `qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum` がある。別量子化・別サイズに切り替える場合は **`MODEL`** と本書の前提（メタキー `qwen3vl.*`・テンソル名）が実装と一致するかを確認すること。

## 制約・既知の制限

- **CPU 版**: IQ 混在 8B は計算量が大きく、**実用的な速度は期待しにくい**。OpenMP はアルゴリズム忠実なまま並列化するが、帯域 bound のため環境次第では伸びが限定的な場合がある。
- **ROCm 版**: AMD GPU・ROCm・`hipcc`、`GPU_ARCH` と実機 ISA の一致が必要。
- **XDNA2 版（`qwen3-xdna2`）**: 恒久の全レイヤー **BF16 重み複製は行わない**。**mmap + 単一 GEMV 用 BF16 スクラッチ**（および `scratch_f32`）であり、代表的 8B 級 IQ 量子化モデルでも **`main-omp.c` に近い「GGUF を載せつつ増分バッファ」**になる（スクラッチの最大要素数は **`output.weight`** クラスの巨大行列にひもづき、VRAM／DRAM の余裕が依然必要になる場合がある）。変換済み GGUF でない限りロード済みモデルサイズより **桁違いの常駐 BF16 が乗らない**。NPU 本線には **MLIR-AIE / IRON** が生成した制御コード（`XDNA_GEMV_DIR`）。未配置時は OpenMP CPU フォールバック。`/dev/accel/accel0` は `render`。**推論レイテンシは GEMV のたびフル復号するため増えうる**。
- **テキストのみ**: Vision・マルチモーダル入力は未対応。
- **コンテキスト長**: `-l` 既定 512。長くすると KV メモリ（CPU ヒープまたは VRAM）が増加する。
- 実装は **参照・研究用**を想定し、商用 API や公式実装との **ビット一致・品質一致**は保証しない。

## 補足：Qwen3-VL-8B 級の形状イメージ

実値は **GGUF メタデータ**に従う。以下は **説明用の代表値**である。

| 記号 | 意味 | 代表値（8B 付近） |
|------|------|-------------------|
| `dim` | 隠れ状態幅 | 4096 |
| `hidden_dim` | SwiGLU 中間幅 | 12288 |
| `n_layers` | ブロック数 | 36 |
| `n_heads` | クエリヘッド数 | 32 |
| `n_kv_heads` | KV ヘッド数 | 8 |
| `head_dim` | `key_length`（128）または `dim/n_heads` | 128 |
| `kv_dim` | `n_kv_heads * head_dim` | 1024 |
| `vocab_size` | 語彙数 | 151936 前後 |

## 補足：サンプリング・GPU（ROCm）の典型分岐

1. **`temp <= 0`**: GPU argmax が可能な経路ではデバイス上で argmaxし、トークン ID を D2H。  
2. **`temp > 0` かつ top-p が無効相当**: GPU で softmax → 多項サンプル。  
3. **`0 < top-p < 1`**: 実装により **logits 全語彙 D2H** して CPU で nucleus する場合がある。

教師強制区間では LM ヘッドを省略してプロンプトトークンを消費する。

## 補足：テンソル名（Qwen3 デコーダ）

- `token_embd.weight`  
- `blk.L.attn_norm.weight`, `blk.L.attn_q_norm.weight`, `blk.L.attn_k_norm.weight`  
- `blk.L.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`  
- `blk.L.ffn_norm.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`  
- `output_norm.weight`, `output.weight`

## 補足：KV とメモリの目安

レイヤーあたり、`pos+1` 位置分の Key/Value で **`(pos+1) * kv_dim`** 要素（×4 バイト if float32）。全レイヤーで `n_layers` 倍。`-l` を大きくすると最悪ケースの割り当てが増える。

## 補足：トラブルシューティング早見表

| 現象 | 想定原因 | 確認・対処 |
|------|----------|------------|
| `hipcc` not found | `ROCM` 誤り | `make ROCM=/opt/rocm` 等 |
| ISA 不一致 | `GPU_ARCH` 誤り | `rocminfo` で確認 |
| mmap 失敗 | パス・権限 | `MODEL` を確認 |
| CPU が極端に遅い | IQ デ量子化コスト | ROCm 版の利用、`-n` を小さく |
| `/dev/accel/accel0` を開けない | `render` グループ未参加 / `amdxdna` 未ロード | `sudo usermod -aG render "$USER"`、`lsmod \| grep amdxdna` を確認 |
| XDNA2 で速度が出ない／NPU が効いていない | `XDNA_GEMV_DIR` 未設定、形状欠け、DRM 不可、`XDNA_FORCE_CPU` 等 | 起動時の **`=== XDNA GEMV / NPU ctrlcode status ===`** で各形状の `MISS` を確認。**`./qwen3-xdna2 model.gguf --xdna-status`** で軽量診断。推論後の **NPU GEMV / CPU GEMV** カウントが **CPU のみ**なら NPU 経路は実行されていない |
| **`CREATE_HWCTX` が EINVAL（`qwen3-xdna2` / `qwen3-xdna2-bfpx`）** | ドライバが列数・タイル数・QoS・DEV_HEAP・ファームを拒否 | 実装は **`vaddr=0`** で `CREATE_BO` し **必ず `mmap()`** して `userptr` を確立、**QoS は全 0**（`qos_meet` 抵触回避）、`num_tiles` は **`ncol×core.row_count`** を主軸に `core+mem+shim` 合算と `1` を順次フォールバック。なお解消しない場合は **`XDNA_NUM_COL`**・**`XDNA_NUM_TILES`**・**`XDNA_HEAP_SIZE`** 上書きと `dmesg` の `amdxdna` 行（`MAP_HOST_BUFFER status 0x4000003` などのファームエラー、ファーム `amdnpu/<vendor>_<rev>/npu.sbin` の有無）を確認 |
| XDNA2 ビルドで `drm/drm.h` not found | カーネル UAPI ヘッダ未インストール | `apt install linux-libc-dev` 等で `<drm/drm.h>` を導入 |

## 補足：ドキュメント間の役割

- **`README.md`**: ビルド・実行・バリアント選択の手順、およびライブラリ非依存の方針とその意義の説明（日本語）。
- **`README.en.md`**: 上記と同等の内容（英語）。
- **`doc/design.md`（本書）**: 現行の設計・仕様。
- **`doc/ChangeLog`**: 日付付き変更履歴。
- **外部（参考）**: **AMD XDNA** のアーキテクチャ概要・世代・ソフトウェアスタック等は、別リポジトリ **[thamada/xdna-overview](https://github.com/thamada/xdna-overview)** にまとめてある（本リポジトリの実装説明とは独立した背景資料）。

実装の詳細は **`qwen3-8b/*.c`** の先頭コメントとソースを参照する。

## 補足：`design.md` 更新時のチェックリスト

1. **`qwen3-8b/`** にソースまたはターゲットを増やしたら、**構成表**と **バイナリ表**を更新する。  
2. **`Makefile`** の変更と本書を同期する。  
3. 仕様変更は **`doc/ChangeLog`** にも記載する。  
4. 利用者向け手順や方針を **`README.md`** で変えたら、 **`README.en.md`** も同趣旨に揃える（またはその逆）。

## 補足：XDNA2 NPU 実装メモ

`main-xdna2.c` と **`main-xdna2-bfpx.c`** はいずれも **`amdxdna` カーネルモジュール**（`drivers/accel/amdxdna`）の DRM ioctl を直接呼ぶ単一ソース実装である。XRT (Xilinx Runtime) / xdna-driver ユーザランドや C++ shim 等の外部依存を持たない。**主な違いはホスト側の線形重みの持ち方**（**mmap を推論中も読む BF16/GEMV スクラッチ方式** と **変換済み BFPX＋チャンクステージング＋mmap 早期解放**）である。`XdnaDev` / `ERT_START_NPU` / GEMV 入力・出力の短命 SHMEM とフォールバックの枠組みは共通である。

主要コンポーネント:

- **UAPI 取り込み**: `<drm/amdxdna_accel.h>` の構造体・ioctl 番号を inline で持つ（外部ヘッダ未配備でもビルド可）。
- **`XdnaDev`**: `open("/dev/accel/accelN")` した fd、`hwctx_handle`、`syncobj_handle`、AIE 列数・行数・FW バージョン等を保持。
- **`XdnaBo`**: 1 つの DRM バッファオブジェクト（`AMDXDNA_BO_SHMEM` / `AMDXDNA_BO_DEV_HEAP` / `AMDXDNA_BO_CMD`）を `handle`、`mmap()` 後の userspace ポインタ、NPU 側仮想アドレス `xdna_addr` の組で持つ。
- **`npu_open`**: `DRM_IOCTL_AMDXDNA_GET_INFO` で AIE topology / FW を取得し、`DRM_IOCTL_AMDXDNA_CREATE_HWCTX` でハードウェアコンテキストを作成、64 MiB の `DEV_HEAP` 命令バッファと `CMD`/`SHMEM` 補助 BO を準備する。
- **`npu_submit_start_npu`**: `ert_packet` ヘッダ + `cu_mask` + `amdxdna_cmd_start_npu` 構造体（命令バッファアドレス・サイズ・引数）を `CMD` BO に書き込み、`DRM_IOCTL_AMDXDNA_EXEC_CMD` で投入。
- **`npu_wait`**: `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` でコマンド完了を待つ。
- **`load_gemv_kernel`**: `XDNA_GEMV_DIR/bf16-gemv-<n>x<d>.bin` から事前コンパイル済の control code を命令バッファアリーナにロードする。MLIR-AIE / IRON ツールチェイン側で生成する想定。
- **`launch_mm_bf16`**: NPU ディスパッチが可能な場合（`have_device && !force_cpu && XDNA_GEMV_DIR`）にのみ **`weight_prepare_bf16`** で単一 **`w_scratch_bo`** に復号転送し、NPU 経路（ctrlcode あり）か CPU OpenMP BF16 GEMV を呼ぶ。**最初から NPU 不可**な run では BF16 スクラッチを確保せず、**`main-omp.c` と同じブロック単位の量子化直 GEMV**（`mm_quant_rows_xdna` / `mm_f32_xdna` / `mm_f16_xdna`）へ自動分岐し、メモリと走行コストを削減する。
- **`print_xdna_gemv_ctrlcode_report`**: `npu_open` 後に、DRM・環境変数・6 形状分の **`bf16-gemv-<n>x<d>.bin`** 読み取り可否を一覧する。推論ループとは独立し **`--xdna-status` / `-X`** からも呼ばれる。
- **重みのレイアウト**: 線形層は **GGUF mmap** の量子化レイアウトを **`WeightsDev` が参照**。各 GEMV 前にだけ **単一 BF16 SHMEM (`w_scratch_bo`)** と **FP32 ステージング (`scratch_f32`)** で展開。小型の norm は mmap 指す。
- **CPU 上の処理**: RMSNorm、Qwen3 ヘッド RMSNorm、RoPE、KV cache 書き込み、FlashAttention 相当、SwiGLU、残差加算、softmax、サンプリングをすべて OpenMP で並列化（`main-omp.c` と同じ粒度）。

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON** で生成した形状 `(n, d)` 専用 BF16 GEMV 制御コードバイナリが必要（本リポジトリには同梱しない）。バイナリ未配置などで CPU に落ちる際、**`qwen3-xdna2`** は BF16 GEMV を OpenMP で実行する（実装では NPU 経路との **bit-identical** が謳われている。**`main-xdna2.c` の先頭コメント**参照）。一方 **`qwen3-xdna2-bfpx`** は **`mm_bfpx`** であり、出力が **`qwen3-xdna2` とビット単位では一致しない**。

### `main-xdna2-bfpx.c`（BFPX）メモ

- **`BfpxMat`**: 各行についてブロックごとに BF16 スケールと int8 係数を保持する。
- **`bfpx_convert_weight_2d`**: 期待する **`n_out × n_in`** と GGUF の **`ne[0]×ne[1]`** を突き合わせ、`main-omp.c` と同じ GEMV 向け論理形状に正規化する（転置 `[n_in,n_out]` はフルデ量子化後に行抽出）。
- **`model_drop_gguf_mmap`**: 変換完了後に tensor 名・mmap・モデル fd を解放し、推論中は BFPX とノルム用 F32 のみを参照する。

## 補足：本書の保守方針

設計書は **実装の下位互換の参照**である。実装と矛盾する場合は、実装か本書のいずれかを修正する。本リポジトリは **Qwen3 系のみ**を対象とするため、**他アーキテクチャ専用のファイル名・ターゲットを本書に書かない**。
