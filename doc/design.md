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
| `main-rocm-fullgpu-flash-opt2.c` | 同上 | **`main-rocm.c` と同一ロジック**の別名ソース。`make build.rocm.fullgpu.flash.opt2` がこれをビルドする。 |

メタデータキーは **`qwen3vl.*`**。Qwen3 固有として、線形射影の直後に **`attn_q_norm` / `attn_k_norm`**（ヘッド長に対する RMSNorm）を挟み、その後 **RoPE** を適用する。チャットは **ChatML**（`<|im_start|>` / `<|im_end|>` 等）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `qwen3-8b/main.c` | CPU 単スレッド推論。 |
| `qwen3-8b/main-omp.c` | CPU OpenMP 並列推論。 |
| `qwen3-8b/main-rocm.c` | ROCm 推論（既定の HIP ビルド対象）。 |
| `qwen3-8b/main-rocm-fullgpu-flash-opt2.c` | `main-rocm.c` と同一内容の別ファイル。 |
| `qwen3-8b/Makefile` | `build` / `build.omp` / `build.rocm` / `build.rocm.fullgpu.flash.opt2` および対応する `run.*`、`clean`。 |
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
| `build.rocm.fullgpu.flash.opt2` / `run.rocm.fullgpu.flash.opt2` | `qwen3-rocm-fullgpu-flash-opt2` | `main-rocm-fullgpu-flash-opt2.c` |

```bash
cd qwen3-8b
make build
make build.omp
make build.rocm              # hipcc・ROCm 必須
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

## 実行時の挙動

**CPU（`qwen3-cpu` / `qwen3-cpu-omp`）**: 重みは mmap 上の GGUF を参照。量子化行は都度ブロックデ量子化してから内積。KV・活性は主に float32。サンプリングはホスト上の logits に対して実施。

**ROCm（`qwen3-rocm` / `qwen3-rocm-fullgpu-flash-opt2`）**: ロード時に F16 重みを VRAM に配置。各ステップは **埋め込み〜全レイヤー〜LM ヘッド**を GPU 上で実行。教師強制区間では LM ヘッドを省略可能。**`0 < top-p < 1`** の nucleus は実装上 **logits 全語彙を D2H** して CPU で処理する場合がある（実装コメント参照）。それ以外は GPU で argmax / softmax＋多項サンプル等。

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

### GGUF パーサー

- GGUF v2/v3 のマジック検証、メタデータ KV、テンソルディスクリプタを解析する。
- **Qwen3-VL** 用メタキー **`qwen3vl.*`**（例: `embedding_length`, `feed_forward_length`, `block_count`, `attention.head_count`, `attention.head_count_kv`, `attention.key_length`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`）から `dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`, `rope_theta`, `norm_eps` 等を取得する。`head_dim` は **`attention.key_length`** から読み取り、無い場合は `dim / n_heads` にフォールバックする。
- `kv_dim = n_kv_heads * head_dim`、`kv_mul = n_heads / n_kv_heads`。
- テンソル名は **`blk.L.attn_q.weight`** 等に加え **`blk.L.attn_q_norm.weight`**, **`blk.L.attn_k_norm.weight`** を参照する（`L` はレイヤー番号）。

### 量子化（本リポジトリで扱う形式）

- **ROCm パス**: ロード時に **IQ2_S / IQ3_S / Q4_K / Q5_K / F32 / F16** をホストで処理し、**F16** を GPU に載せる（詳細は `main-rocm.c` 先頭コメント）。
- **CPU パス**: 同上のブロック形式を **実行時にブロック単位でデ量子化**し、`float` で内積。Q8_0 専用実装ではない。

### トークナイザー

- GPT-2 スタイルのバイトレベル BPE。`tokenizer.ggml.*` を使用。
- チャットは **ChatML**: `<|im_start|>`, `<|im_end|>` 等を語彙から解決し、`chat_encode` が system / user / assistant ブロックを組み立てる。

### 推論エンジン（論理）

- **Transformer**: RMSNorm、**Q/K ヘッド RMSNorm**（Qwen3）、RoPE、GQA による注意、SiLU 付き SwiGLU FFN、残差。
- **KV キャッシュ**: レイヤー×位置×`kv_dim`（float）。
- **サンプリング**: greedy（temp=0）、温度付き、Top-p（ホストまたはデバイス、実装依存）。

### フォワード（1 トークン・概念）

1. **埋め込み**（トークン ID → `dim`）。  
2. 各レイヤー: 注意前 RMSNorm → Q/K/V 線形 → **Q/K ヘッド RMSNorm** → RoPE(Q,K) → KV 書き込み → 注意 → 出力射影 → 残差 → FFN 前 RMSNorm → gate/up/down → 残差。  
3. 最終 RMSNorm → **LM ヘッド**（`output_norm` + `output.weight`）→ サンプリング。CPU 実装は **`output.weight` 必須**（Qwen3-VL Instruct 用 GGUF は通常 **埋め込みと tie しない**別テンソル）。ROCm 実装は **`output.weight` が無い場合のみ** `token_embd` にフォールバックする。

## モデル参照

利用する GGUF のファイル名は **`qwen3-8b/Makefile` の `MODEL`** を参照する。モデル本体は著作権とファイルサイズの都合でリポジトリに含めず、既定モデルの取得元は **`qwen3-8b/gguf.txt`** に URL として置く。ダウンロード時は Hugging Face の `blob/main` URL を `resolve/main` に置換して実体ファイルを取得する。ハッシュ確認は例として `qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum` がある。別量子化・別サイズに切り替える場合は **`MODEL`** と本書の前提（メタキー `qwen3vl.*`・テンソル名）が実装と一致するかを確認すること。

## 制約・既知の制限

- **CPU 版**: IQ 混在 8B は計算量が大きく、**実用的な速度は期待しにくい**。OpenMP はアルゴリズム忠实なまま並列化するが、帯域 bound のため環境次第では伸びが限定的な場合がある。
- **ROCm 版**: AMD GPU・ROCm・`hipcc`、`GPU_ARCH` と実機 ISA の一致が必要。
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

## 補足：ドキュメント間の役割

- **`doc/design.md`（本書）**: 現行の設計・仕様。  
- **`doc/ChangeLog`**: 日付付き変更履歴。  

実装の詳細は **`qwen3-8b/*.c`** の先頭コメントとソースを参照する。

## 補足：`design.md` 更新時のチェックリスト

1. **`qwen3-8b/`** にソースまたはターゲットを増やしたら、**構成表**と **バイナリ表**を更新する。  
2. **`Makefile`** の変更と本書を同期する。  
3. 仕様変更は **`doc/ChangeLog`** にも記載する。

## 補足：本書の保守方針

設計書は **実装の下位互換の参照**である。実装と矛盾する場合は、実装か本書のいずれかを修正する。本リポジトリは **Qwen3 系のみ**を対象とするため、**他アーキテクチャ専用のファイル名・ターゲットを本書に書かない**。
