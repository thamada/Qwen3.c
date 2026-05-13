# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog`を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

### リポジトリの目的とスコープ

本リポジトリは、**Qwen3 系（Qwen3-VL-8B-Instruct）** の **GGUF** 形式モデルを、**単一または少数の C/HIP ソースファイル**からビルド可能な形で **推論（テキスト生成）**するエンジンである。外部の大規模フレームワーク（PyTorch 等）に依存せず、**GGUF の読み取り・トークナイズ・Transformer フォワード・サンプリング**を一連のコードパスとして理解・改変しやすくすることを目的とする。学習・ファインチューニング・バッチ推論の最適化はスコープ外であり、主に **対話形式のインタラクティブ生成**（プロンプト＋続きの生成）を想定する。

文中の「decoder-only」「GQA」「FlashAttention 系デコードカーネル」等は、**Transformer デコーダの一般的なパターン**を指す。**実装はすべて `qwen3-8b/` に置かれる。** 対象例は **Qwen3-VL-8B-Instruct** の **IQ2_S / IQ3_S 等が混在した GGUF**（例: `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`）。**Vision（画像エンコード・deepstack・画像トークン）は実装しない**。**テキスト用デコーダのみ**を実行する。

### 実装バリアント（本リポジトリに含まれるもの）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `main.c` | CPU、単スレッド | GGUF mmap、`qwen3vl.*` パース。線形層は **IQ2_S / IQ3_S / Q4_K / Q5_K** 等を **`QK_K=256` ブロック単位**にデ量子化しつつ GEMV（全重みの float 一括展開なし）。`libm` のみ。 |
| `main-omp.c` | CPU、**OpenMP** | 上記と同一アルゴリズム。**GEMV** は出力行並列、**Attention** はヘッド並列、`main-rocm.c` のカーネル粒度に相当する並列化（RoPE、RMSNorm、残差、SiLU 等）。 |
| `main-rocm.c` | **ROCm / HIP** | ロード時に量子化重みを CPU で **F16** に展開して VRAM に載せ、**フル GPU** パスで推論。**Flash 系デコード注意**・**KV カーネル書き込み**・**レイヤー間のホスト非介在**・GPU サンプリング（top-p 時は logits D2H フォールバック）等を含む。**`make build.rocm` の既定エントリ**。 |
| `main-xdna2.c` | **AMD Ryzen AI NPU (XDNA2)** | ロード時に量子化重みを CPU で **BF16** に展開し、`amdxdna` カーネルモジュールの DRM ioctl（`/dev/accel/accelN`）で直接 BO に載せる。GEMV を NPU で実行し、RMSNorm/Qwen3 ヘッド RMSNorm/RoPE/Attention/SwiGLU/サンプリング等は CPU（OpenMP）で実行する**ハイブリッド構成**。XRT 等のユーザランド SDK を必要とせず、UAPI ヘッダ相当を inline で持つ自己完結ビルド。`/dev/accel/accelN` が利用不可な場合・GEMV 制御コード未配置の場合は、bit-identical な OpenMP CPU フォールバックに透過的に切り替わる。 |

メタデータキーは **`qwen3vl.*`**。Qwen3 固有として、線形射影の直後に **`attn_q_norm` / `attn_k_norm`**（ヘッド長に対する RMSNorm）を挟み、その後 **RoPE** を適用する。チャットは **ChatML**（`<|im_start|>` / `<|im_end|>` 等）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `qwen3-8b/main.c` | CPU 単スレッド推論。 |
| `qwen3-8b/main-omp.c` | CPU OpenMP 並列推論。 |
| `qwen3-8b/main-rocm.c` | ROCm 推論（既定の HIP ビルド対象）。 |
| `qwen3-8b/main-xdna2.c` | AMD Ryzen AI（XDNA2）NPU 推論。`amdxdna` カーネルモジュール直叩き。 |
| `qwen3-8b/Makefile` | `build` / `build.omp` / `build.rocm` / `build.xdna2` および対応する `run.*`、`clean`。 |
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

```bash
cd qwen3-8b
make build
make build.omp
make build.rocm              # hipcc・ROCm 必須
make build.xdna2             # Linux >= 6.10 + amdxdna カーネルモジュール（XRT 不要）
OMP_NUM_THREADS=8 ./qwen3-cpu-omp "$(MODEL)" -p "Hello" -n 4
```

**CPU（IQ 混在 8B）**はブロック単位デ量子化のため **非常に遅くなり得る**。実用スループットは **ROCm 版**を優先する想定である。

## ビルドと実行

### 共通（`qwen3-8b/Makefile`）

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `ROCM` | ROCm ルート | `/opt/rocm` |
| `HIPCC` | HIP コンパイラ | `$(ROCM)/bin/hipcc` |
| `GPU_ARCH` | `--offload-arch=` | `gfx1201` |
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
```

## 実行時の挙動

**CPU（`qwen3-cpu` / `qwen3-cpu-omp`）**: 重みは mmap 上の GGUF を参照。量子化行は都度ブロックデ量子化してから内積。KV・活性は主に float32。サンプリングはホスト上の logits に対して実施。

**ROCm（`qwen3-rocm`）**: ロード時に F16 重みを VRAM に配置。各ステップは **埋め込み〜全レイヤー〜LM ヘッド**を GPU 上で実行。教師強制区間では LM ヘッドを省略可能。**`0 < top-p < 1`** の nucleus は実装上 **logits 全語彙を D2H** して CPU で処理する場合がある（実装コメント参照）。それ以外は GPU で argmax / softmax＋多項サンプル等。

**XDNA2（`qwen3-xdna2`）**: ロード時に量子化重みを CPU で **BF16** に展開し、`AMDXDNA_BO_SHMEM` 型の DRM バッファオブジェクトに格納する（`/dev/accel/accelN` 経由）。各 GEMV ステップごとに、入力 BF16 ベクトル・重み BO・出力バッファのアドレスを `ERT_START_NPU` パケットに詰めて `DRM_IOCTL_AMDXDNA_EXEC_CMD` で NPU マイクロコントローラ（MERT/ERT）に投げ、`drm_syncobj_timeline_wait` で完了を待つ。RMSNorm、Qwen3 ヘッド RMSNorm、RoPE、Attention、SwiGLU、サンプリングはホスト CPU 上で実行する。NPU が使えない場合（`/dev/accel` を開けない、`XDNA_FORCE_CPU=1`、GEMV 制御コードが未配置）には bit-identical な OpenMP BF16 GEMV にシームレスに切り替わる。

## コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `-p <prompt>` | ユーザープロンプト | `Hello` |
| `-n <tokens>` | 最大生成トークン数 | `256` |
| `-t <temp>` | Temperature | `0.6` |
| `-k <topp>` | Top-p サンプリング | `0.9` |
| `-s <seed>` | 乱数シード | `time(NULL)` |
| `-l <len>` | 最大シーケンス長 | `512` |

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
- **XDNA2 版**: 重みをロード時に BF16 へ一括展開するため、8B クラスでは **約 16 GB の DDR** が常駐する。NPU の本当の高速化を得るには **MLIR-AIE / IRON ツールチェイン**で生成した BF16 GEMV 制御コード一式が `XDNA_GEMV_DIR` 下に必要。未配置時は OpenMP CPU フォールバックが採用される。ユーザは `render` グループに所属している必要がある（`/dev/accel/accel0` は `crw-rw---- root:render`）。
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
| XDNA2 で速度が出ない（CPU fallback 表示） | `XDNA_GEMV_DIR` 未設定 or 制御コード未配置 | MLIR-AIE で生成した `bf16-gemv-<n>x<d>.bin` を所定パスに配置 |
| XDNA2 ビルドで `drm/drm.h` not found | カーネル UAPI ヘッダ未インストール | `apt install linux-libc-dev` 等で `<drm/drm.h>` を導入 |

## 補足：ドキュメント間の役割

- **`doc/design.md`（本書）**: 現行の設計・仕様。  
- **`doc/ChangeLog`**: 日付付き変更履歴。  

実装の詳細は **`qwen3-8b/*.c`** の先頭コメントとソースを参照する。

## 補足：`design.md` 更新時のチェックリスト

1. **`qwen3-8b/`** にソースまたはターゲットを増やしたら、**構成表**と **バイナリ表**を更新する。  
2. **`Makefile`** の変更と本書を同期する。  
3. 仕様変更は **`doc/ChangeLog`** にも記載する。

## 補足：XDNA2 NPU 実装メモ

`main-xdna2.c` は **`amdxdna` カーネルモジュール**（`drivers/accel/amdxdna`）の DRM ioctl を直接呼ぶ単一ソース実装である。XRT (Xilinx Runtime) / xdna-driver ユーザランドや C++ shim 等の外部依存を持たない。

主要コンポーネント:

- **UAPI 取り込み**: `<drm/amdxdna_accel.h>` の構造体・ioctl 番号を inline で持つ（外部ヘッダ未配備でもビルド可）。
- **`XdnaDev`**: `open("/dev/accel/accelN")` した fd、`hwctx_handle`、`syncobj_handle`、AIE 列数・行数・FW バージョン等を保持。
- **`XdnaBo`**: 1 つの DRM バッファオブジェクト（`AMDXDNA_BO_SHMEM` / `AMDXDNA_BO_DEV_HEAP` / `AMDXDNA_BO_CMD`）を `handle`、`mmap()` 後の userspace ポインタ、NPU 側仮想アドレス `xdna_addr` の組で持つ。
- **`npu_open`**: `DRM_IOCTL_AMDXDNA_GET_INFO` で AIE topology / FW を取得し、`DRM_IOCTL_AMDXDNA_CREATE_HWCTX` でハードウェアコンテキストを作成、64 MiB の `DEV_HEAP` 命令バッファと `CMD`/`SHMEM` 補助 BO を準備する。
- **`npu_submit_start_npu`**: `ert_packet` ヘッダ + `cu_mask` + `amdxdna_cmd_start_npu` 構造体（命令バッファアドレス・サイズ・引数）を `CMD` BO に書き込み、`DRM_IOCTL_AMDXDNA_EXEC_CMD` で投入。
- **`npu_wait`**: `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` でコマンド完了を待つ。
- **`load_gemv_kernel`**: `XDNA_GEMV_DIR/bf16-gemv-<n>x<d>.bin` から事前コンパイル済の control code を命令バッファアリーナにロードする。MLIR-AIE / IRON ツールチェイン側で生成する想定。
- **`launch_mm_bf16`**: NPU 経路（制御コードあり）と CPU OpenMP 経路（フォールバック）の両方を実装し、実行時に透過的にディスパッチ。
- **重みのレイアウト**: 量子化重みをロード時に CPU で BF16 に展開し、`AMDXDNA_BO_SHMEM` BO に常駐させる。norm 系は CPU 上の通常メモリに留める（小さいため）。
- **CPU 上の処理**: RMSNorm、Qwen3 ヘッド RMSNorm、RoPE、KV cache 書き込み、FlashAttention 相当、SwiGLU、残差加算、softmax、サンプリングをすべて OpenMP で並列化（`main-omp.c` と同じ粒度）。

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON** で生成した形状 `(n, d)` 専用 BF16 GEMV 制御コードバイナリが必要（本リポジトリには同梱しない）。バイナリ未配置時は OpenMP CPU 経路に bit-identical でフォールバックする。

## 補足：本書の保守方針

設計書は **実装の下位互換の参照**である。実装と矛盾する場合は、実装か本書のいずれかを修正する。本リポジトリは **Qwen3 系のみ**を対象とするため、**他アーキテクチャ専用のファイル名・ターゲットを本書に書かない**。
