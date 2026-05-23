# 変更履歴

> **注意**:
>   本ドキュメントは変更履歴です。日付はdateコマンドで確認して2026-01-23 12:34:55のように年-月-日 時:分:秒のようにします。
>   最も最新のものから順に並べて記入します。

## 2026-05-23 17:02:22

**`qwen3-8b/gpu-rocm/`** — **WMMA 利用状況の確認**（**`make wmma`** / **`make wmma-probe`**）。

#### 背景

Prefill 線形層は **hipBLAS / rocBLAS** 経由であり、**`main.c` に WMMA / rocWMMA / MFMA を直接書いていない**。一方 rocBLAS 内部カーネルが gfx11 で **WMMA 命令**を使う場合がある。**`make wmma`** で「自前コードに WMMA が無いこと」「検出器が WMMA を拾えること」「ライブラリ側 ISA」をまとめて確認する。

#### `gpu-rocm/Makefile`

- **`wmma-probe`**: **`wmma_probe.c`** を **`hipcc`** でビルド（gfx11 時 **`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`** を含む校正用バイナリ）。
- **`wmma`**: **`build`** + **`wmma-probe`** の後 **`scripts/check_wmma.sh`** を実行。
- **`clean`**: **`wmma-probe`** も削除。
- 変数: **`WMMA_PROMPT`** / **`WMMA_N`** / **`WMMA_SKIP_RUN`**（既定 0）/ **`WMMA_SKIP_ROCPROF`**（既定 1）/ **`LLVM_OBJDUMP`**（既定 **`$(ROCM)/llvm/bin/llvm-objdump`**）。

#### `scripts/check_wmma.sh`

| 段階 | 内容 | 期待 |
|------|------|------|
| **static: source** | **`main.c`** に WMMA/MFMA/rocWMMA 参照が無い | OK |
| **static: binary** | **`qwen3-rocm`** の objdump に WMMA 命令が無い | OK |
| **probe** | **`wmma-probe`** に WMMA 命令あり（gfx11） | 検出器校正 |
| **lib** | **`rocBLAS`** の **`Kernels.so-000-$(GPU_ARCH).hsaco`** に WMMA 有無 | 0 でも WARN（FMAC 経路の可能性） |
| **path** | **`./qwen3-rocm`** 実行で **`Prefill linear: hipBLAS GemmEx`** | MODEL 要（**`WMMA_SKIP_RUN=1`** で省略可） |
| **runtime** | **`rocprofv3 --kernel-trace`** で Prefill カーネル ISA | **`WMMA_SKIP_ROCPROF=0`** で有効化 |

#### `wmma_probe.c`

- gfx11 向け最小 HIP カーネル。**`make wmma`** の **`llvm-objdump`** による WMMA 検出が機能するかの校正用。

#### ドキュメント

**`README.md`** / **`README.en.md`**: 実行経路表・ディレクトリツリー・ROCm 節（**`make wmma`**）・トラブルシュート・「実装を読む順序」を上記に同期。

**`doc/design.md`**: **`gpu-rocm`** ディレクトリ表・ROCm ビルド節（**`make wmma`**）・トラブルシュートを上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 16:40:33

**`qwen3-8b/gpu-rocm/`** — Prefill 線形層を **hipBLAS `GemmEx`** に切替（[llama.cpp](https://github.com/ggml-org/llama.cpp/) の `cublasGemmEx` 経路同趣旨）。Prefill スループット **~19×**（段階 0 比）。

#### `gpu-rocm/main.c`

- **`hipblasGemmEx`**（**`HIPBLAS_OP_T, HIPBLAS_OP_N`**）: **`O[S,d] = X[S,n] @ W[d,n]^T`**。FP16 重み・FP16 活性・FP32 出力（**`HIPBLAS_COMPUTE_32F`**）。
- **`Model`**: **`hipblasHandle_t`**、**`d_scratch_f16`**（**`max_seq × hidden_dim`** 要素）を追加。**`alloc_state_gpu`** / **`free_hipblas`**。
- **`f32_to_f16_batch_kernel`** / **`dev_f32f16`**: 活性 FP32 → FP16 変換。同一入力の q/k/v や gate/up は **変換 1 回**で使い回し。
- **`launch_mm_f16_batch`**: **`n_tokens >= 2`** かつ **`x_f16 != NULL`** なら hipBLAS。それ以外は **`mm_f16_gemv_batch_kernel`** フォールバック。
- 起動時 **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** を表示。

#### `gpu-rocm/Makefile`

- リンクに **`-lhipblas -lrocblas`** を追加。
- **`BENCH_LOG`** サンプル行を 1 件追加（**`2026-05-23T16:37:12|…|549.94|25.67|171.41`**）。

#### ベンチマーク（132 prompt tokens、RX 7900 XTX / gfx1100、`make log.push` 相当）

| 段階 | prefill tok/s | 備考 |
|------|---------------|------|
| 0: 1 トークンずつ GEMV | 28.74 | 改善前 |
| 1: バッチ + カスタム GEMV | 53.95 | 前コミット |
| 2: hipBLAS GemmEx | **549.94** | 本変更 |

Decode（~26 tok/s）はほぼ不変。詳細は **`doc/design.md`** **「ROCm Prefill 高速化の詳細（3 段階）」**。

#### ドキュメント

**`README.md`**: 実行経路表・ROCm 節（Prefill 高速化概要・3 段階ベンチ表）・「実装を読む順序」を上記に同期。

**`doc/design.md`**: **`gpu-rocm`** のバリアント表・ディレクトリ表・実行時挙動・ROCm forward 節・**「ROCm Prefill 高速化の詳細（3 段階）」** を上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 16:23:31

**`qwen3-8b/gpu-rocm/main.c`** — **Prefill バッチ forward**（CUDA **`gpu_forward_prefill`** と同趣旨）。Decode は従来どおり 1 トークン **`forward_gpu`**。

#### `gpu-rocm/main.c`

- **`forward_prefill_gpu`**: プロンプト全トークン（**`n_prompt > 1`**）を **1 回の GPU forward** で処理。teacher forcing ループを廃止。
- **バッチ GPU バッファ**（**`Model`**: **`d_x_batch` / `d_xb_batch` / `d_q_batch` 等**、**`batch_cap = max_seq`**）を **`alloc_state_gpu`** で確保。
- **Prefill 用カーネル**:
  - **`emb_f16_batch_kernel`** — 全トークン embedding 一括。
  - **`rmsnorm_batch_kernel`** / **`rmsnorm_head_batch_kernel`** — トークン次元バッチ RMSNorm。
  - **`mm_f16_gemv_batch_kernel`** — 線形層 **S×d GEMM 相当**（重み行を全トークンで再利用）。
  - **`rope_prefill_batch_kernel`** — 位置 **`t`** を token index として RoPE。
  - **`kv_write_batch_kernel`** — K/V を **`0..n_tokens-1`** に一括書込。
  - **`attn_flash_prefill_kernel`** — 因果マスク付き Flash Attention（位置 **`t`** は **`0..t`** のみ参照）。**`n_tokens × n_heads`** block 並列。
  - **`silu_mul_batch_kernel`** / **`vec_add_batch_kernel`** — FFN・残差のバッチ版。
- **Prefill 終了時**: 最終プロンプト token の hidden に **`output_norm` + LM head**（**`launch_mm_f16`**）で logits を 1 回だけ計算。
- **Decode**: **`forward_gpu`**（**`mm_f16_gemv_kernel`** + **`attn_flash_decode_kernel_hd128`**）を生成トークンごとに実行。
- **`generate`**: **`n_prompt > 1`** → **`forward_prefill_gpu`** → サンプル＋decode ループ。**`n_prompt == 1`** は **`forward_gpu(prompt[0], 0, 1)`** にフォールバック。
- **Prefill progress bar**: バッチ prefill 中は **0 → 完了** の 2 段表示（トークン単位の `\r` 更新はしない）。

#### `gpu-rocm/Makefile`

- **`BENCH_LOG`** サンプル行を 1 件追加（**`2026-05-23T16:22:01|gfx1100|…`**）。

#### ドキュメント

**`doc/design.md`**: **`gpu-rocm`** のバリアント表・ディレクトリ表・実行時挙動・ROCm forward 節・生成ループを上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 16:05:09

**`qwen3-8b/gpu-rocm/Makefile`** — **`make log`** 表表示の列幅調整と **`BENCH_LOG`** 日時形式の統一。

#### 変更内容

- **`log.push`**: 追記日時を **`date -Iseconds`**（**`+00:00` 付き**）から **`date +%Y-%m-%dT%H:%M:%S`**（ローカル、オフセットなし）に変更。
- **`log`**: 表ヘッダ・データ行の **`printf`** 列幅を調整（Date 19 / GPU 8 / Host 14 等）。**`awk`** で日時フィールド末尾の **`+TZ`** を除去してから表示（旧エントリ互換）。
- 既存 **`BENCH_LOG`** サンプル行の日時を **`2026-05-23T15:54:16`** 形式に更新。

#### ドキュメント

**`doc/design.md`**: ROCm ビルド節の **`BENCH_LOG`** 行形式・**`gpu-rocm/Makefile`** 一行説明を上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 15:55:32

**`qwen3-8b/gpu-rocm/`** — **`cpu-blas`** と同形式の **Prefill progress bar**・スループット要約、および **`make log` / `make log.push`** ベンチマーク履歴。

#### `gpu-rocm/main.c`

- **`prefill_progress_update`** / **`prefill_progress_done`** / **`decode_progress_done`** / **`throughput_summary`** を追加（**`cpu-blas/main.c`** と同形式。**`Prefill [====...]`** バー幅 40）。
- 終了時 stderr に **prefill / decode / total** の tok/s 要約（**`--- throughput ---`**）。
- **`make log.push`** 用に stdout へ **`--- benchmark ---`** と **`prefill_tps:` / `decode_tps:` / `total_tps:`** を出力（**推論区間のみ**。モデル重み H2D は計測外）。
- 既存の **`--- N prompt tokens + M generated tokens ---`** / **`--- X.Xs total ---`** は維持。

#### `gpu-rocm/Makefile`

- **`log`**: **`BENCH_LOG += …`** 行を表形式で表示（日時・**`GPU_ARCH`**・ホスト・prompt/gen トークン数・prefill/decode/total tok/s）。
- **`log.push`**: 既定 **`BENCH_PROMPT`**（~128 token ChatML）・**`-n $(BENCH_N)`**（既定 128）・**`-t 0 -s $(BENCH_SEED)`** でベンチ実行し、結果を **`# BENCH_LOG_END`** 直前に **`BENCH_LOG += YYYY-MM-DDTHH:MM:SS|GPU_ARCH|hostname|…`** として追記（**`date +%Y-%m-%dT%H:%M:%S`**）。
- 上書き例: **`make log.push BENCH_N=64 BENCH_SEED=42`**。

#### ドキュメント

**`README.md`** / **`README.en.md`**: 実行経路表の ROCm 行、ROCm 節（progress bar・**`make log` / `make log.push`**）、「実装を読む順序」を上記に同期。

**`doc/design.md`**: **`gpu-rocm`** のバリアント表・ディレクトリ表・実行時挙動・ROCm ビルド節を上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 06:33:02

**`qwen3-8b/gpu-rocm/`** — **`GPU_ARCH` の `rocminfo` 自動検出**と集約 Makefile の追随。

#### `gpu-rocm/Makefile`

- **`ROCMININFO`**（既定 **`$(ROCM)/bin/rocminfo`**）の awk で、最初の GPU エージェント **`Name: gfx*`** を **`GPU_ARCH`** に採用（**`GPU_ARCH ?= $(shell …)`**）。
- **`detect-gpu-arch`** ターゲット: 検出成功時に **`Detected GPU arch: …`** を表示。未検出時はエラー終了（手動 **`GPU_ARCH=gfx1100`** を案内）。
- **`build`** は **`detect-gpu-arch`** に依存。既定の固定 **`gfx1201`** は廃止。
- 手動上書き: **`make build GPU_ARCH=gfx1100`** / **`make detect-gpu-arch`**。

#### 集約 `qwen3-8b/Makefile`

- ルートの **`GPU_ARCH ?= gfx1201`** を削除。**`build.gpu-rocm`** は **`gpu-rocm`** 側の自動検出に任せ、コマンドラインで **`GPU_ARCH`** が渡されたときだけ **`$(if $(GPU_ARCH),GPU_ARCH="…")`** で子 Makefile に転送。

#### ドキュメント

**`README.md`** / **`README.en.md`**: ROCm 節を自動検出前提に更新（**`make -C gpu-rocm detect-gpu-arch`**、集約 **`make build.gpu-rocm`** から **`GPU_ARCH` 省略可**、トラブルシュートの awk 例）。

**`doc/design.md`**: 実装バリアント表・Make 変数表・ROCm ビルド例・制約・トラブルシュートを上記に同期。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 04:53:10

**`qwen3-8b/cpu-blas/`** — **IQ2_S / IQ3_S の AVX2 整数内積**、**RoPE キャッシュ**、**prefill LM head スキップ**、**greedy argmax 専用パス**、**F16 埋め込み F16C**。

#### IQ2_S / IQ3_S の AVX2 化

- 従来 **`vec_dot_iq2_s_q8_K` / `vec_dot_iq3_s_q8_K`** を **`_generic`** に改名。**`#if defined(__AVX2__)`** で ggml 準拠の SIMD 版を追加（Q4_K / Q5_K と同列に **全量子化型が AVX2 対応**）。
- **IQ2_S AVX2**:
  - **`iq2s_grid`** を **`_mm256_set_epi64x`** で 4 組まとめて load（**`qs`/`qh`** から index 合成）。
  - **`signs`** は **`k_mask1`/`k_mask2`** + **`shuffle_epi8`** で 8 lane に展開 → **`cmpeq` + `xor`/`sub`** で Q8 符号反転（**`q8s = sub(xor(s2, q8), s2)`**）。
  - **`maddubs_epi16(q2, q8s)`** → **`madd_epi16(scale, ·)`** → **`sumi1/sumi2`**。ブロックごと **`fmadd(d, sumi, accumf)`**。**`*out = 0.125f * hsum_float_8(accumf)`**。
  - **4-bit scale** は **`scales[8]`** を **`memcpy` + bit unpack** → **`cvtepi8_epi16`** → **`get_scale_shuffle_k4`**。
- **IQ3_S AVX2**:
  - grid index を **`sllv_epi32`** + **`iq3s_grid[ix]` gather**（16 index を **`storeu` → set_epi32`**）。
  - signs 処理は IQ2_S 同型。**scale** は **`2*ls+1`** を **`set1_epi16`**。
  - **`*out = hsum_float_8(accumf)`**（IQ2 の 0.125 係数なし）。

#### RoPE cos/sin キャッシュ

- **`Model.rope_cr` / `Model.rope_ci`**: **`[max_seq × head_dim/2]`** float（起動時 **`init_rope_cache`** で **`cosf/sinf(pos·freq)`** を一括計算。**`freq = 1/rope_theta^(2i/head_dim)`**）。
- **`apply_rope`** は **`powf/cosf/sinf` を毎ヘッド・毎ペアで呼ばず**、**`rope_cr[pos]` / `rope_ci[pos]`** を参照。Qwen3-VL-8B（**`max_seq=512`**, **`head_dim=128`**）で **~256 KiB**（cr+ci）。
- **`main`** で **`load_weights` 後に `init_rope_cache`**、終了時 **`free_rope_cache`**。

#### `forward` の LM head モード（`lm_mode`）

- **`enum { FWD_NO_LM, FWD_LM_FULL, FWD_LM_ARGMAX }`**。**`forward(m, token, pos, lm_mode)`**。
- **`generate`** が **`pos` と `temp` から選択:
  - **`pos < n_prompt - 1`**（prefill 中・最終プロンプト token 以外）: **`FWD_NO_LM`** — **`output_norm` + LM head をスキップ**（teacher forcing で次 token 既知のため logits 不要）。
  - **`pos >= n_prompt - 1` かつ `temp <= 0`**（greedy）: **`FWD_LM_ARGMAX`** — **`mm_argmax_row`** のみ。
  - **それ以外**（サンプリング）: **`FWD_LM_FULL`** — 従来どおり **`mm(logits, ...)`** + **`sample_token`**。

#### `mm_argmax_row`（greedy 専用）

- **全 vocab 行の logits ベクトル `s->logits[vocab_size]` を materialize しない**。OpenMP で行 **`i`** ごとに dot を計算し **thread-local max** → **`critical`** で global max 更新。
- **量子化 LM head**: **`quantize_row_q8_K(x, q8, dim)`** 1 回 → 各行 **`vec_dot_row_q8_K`**（AVX2 IQ2_S 等が効く）。
- **F32**: **`cblas_sdot`** 行ごと。**F16**: スカラー内積ループ。
- 結果は **`State.argmax_tok`**。**`sample_token` は呼ばず `next = argmax_tok`**。

#### F16 埋め込み lookup（`emb_lookup`）

- **`DT_F16`** かつ **`__AVX2__ && __F16C__`**: **`_mm_loadu_si128` + `_mm256_cvtph_ps`** で 8 要素ずつ FP16→F32（単スレッド。token 1 回あたり **`dim=4096` → 512 SIMD iter**）。
- 非 F16C または非 AVX2: 従来 **OpenMP 行並列 `host_f16f32`**。

#### ドキュメント

**`doc/design.md`**: 上記を **「`cpu-blas`：Q8_K 活性化 GEMV」** 節・**CPU forward**・**実行時挙動**・**State/Model**・バリアント表に反映。IQ2_M gain 表を **AVX2 IQ2/IQ3 dot あり**に更新。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 04:34:38

**`qwen3-8b/cpu-blas/`** — 量子化 GEMV の **AVX2 最適化**と **層内 Q8_K 量子化共有**。

#### 前提：Q8_K 活性化 GEMV とは

- **IQ2_S / IQ3_S / Q4_K / Q5_K** の重み行と float 活性 **`x[n]`** の GEMV は、llama.cpp / ggml と同様 **「活性を Q8_K に量子化 → 重みブロックと整数内積」** で計算する（per-row **`float[256]` 全復号**は行わない）。
- 活性 **`BlockQ8_K`**（ggml **`block_q8_K`** 準拠）:
  - **`float d`**: スーパーブロック scale（**`d = 1/iscale`**。**`iscale = -127/maxv`**。**`maxv`** はブロック内 **符号付き**最大値）。
  - **`int8_t qs[256]`**: QK_K=256 要素の量子化係数。
  - **`int16_t bsums[16]`**: **`qs`** を 16 要素ずつ足した partial sum（Q4_K / Q5_K の **dmin × mins** 補正で使用）。
- 重み側スーパーブロック（参考）: **`BlockQ4_K`** 144 B（**`d`/`dmin` FP16 + scales[12] + qs[128] nibble**）、**`BlockQ5_K`** 176 B（**+ qh[32]** 第 5 bit）、**`BlockIQ2_S`** 82 B、**`BlockIQ3_S`** 110 B。
- **`State.q8`**: **`calloc(hidden_dim / QK_K, sizeof(BlockQ8_K))`**（Qwen3-VL-8B なら **14336/256 = 56 ブロック**）。Attention / gate/up（**`n=dim=4096` → 16 ブロック**）と **down**（**`n=hidden_dim` → 56 ブロック**）の両方に足りるサイズ。

#### 背景（従来 `mm_quant_rows` の問題）

- 旧 **`mm_quant_rows`** は **GEMV 呼び出しのたび** 先頭で **`quantize_row_q8_K(x, q8, n)`**、続けて OpenMP で **`d` 行**の **`vec_dot_row_q8_K`**。
- **`forward` 1 層・1 token** の quantize 回数（量子化 GEMV 全テンソル想定）:

| 呼び出し | 入力 **`x`** | 旧 quantize |
|---|---|---|
| **`mm(wq/wk/wv)`** | attn RMSNorm 後 **`xb`** | **3**（同一ベクトルなのに重複） |
| **`mm(wo)`** | attn 出力 **`xb`** | 1 |
| **`mm(gate/up)`** | ffn RMSNorm 後 **`xb`** | **2**（重複） |
| **`mm(down)`** | SwiGLU 後 **`hb`** | 1 |

- **`dim=4096`** では 1 quantize = **16 スーパーブロック ×（256 回 abs-max + 256 int8 変換 + 16 bsums）**。Attention+FFN だけで **冗長 4 quantize/層** → **28 層で ~112 quantize/token 削減**の余地。

#### 層内 Q8_K 量子化共有

- **`mm(o, x, w, n, d, type, q8, q8_ready)`** — **`q8_ready=1`** なら quantize 省略、**`mm_quant_dot_rows`** のみ。
- **`forward` 制御**（層 **`l`**）:

```text
rmsnorm → xb
q8_att = is_q8_mm_type(wq_t[l])
if (q8_att) quantize_row_q8_K(xb → q8, n=dim)    // 層内 1 回
mm(q/k/v, xb, ..., q8, q8_att)                   // 3 GEMV で Q8 読み取り共有
... OpenBLAS attention ...
mm(xb2, xb, wo, ..., q8, 0)                      // attn 出力 xb → 都度 quantize

rmsnorm → xb
q8_ffn = is_q8_mm_type(gate_t[l])
if (q8_ffn) quantize_row_q8_K(xb → q8, n=dim)
mm(gate/up, xb, ..., q8, q8_ffn)
SwiGLU → hb
mm(xb, hb, down, n=hidden, ..., q8, 0)           // 入力・長さ変更 → 都度 quantize
mm(logits, x, out, ..., q8, 0)                   // LM head
```

- **判定キー**: Attention は **`wq_t[l]`**、FFN は **`gate_t[l]`** のみ参照。**wq が F16 で wk が Q4_K** 等の混在では **`q8_att=0`** → wk/wv が **個別 quantize**（旧挙動）。
- **OpenMP**: quantize は **単スレッド**、並列 **`mm_quant_dot_rows`** は **quantize 後に `q8` 読取のみ** → data race なし。
- **層あたり quantize（理想ケース）**: 旧 **7 → 新 4**（wq/wk/wv: 3→1、gate/up: 2→1。wo/down/output は不変）。

#### API 分離

- **`mm_quant_dot_rows`**: **`row = wb + i * row_bytes_quant(type, n)`** → **`o[i] = vec_dot_row_q8_K(...)`**。OpenMP **`schedule(static)`** で出力行並列。
- **`vec_dot_row_q8_K`**: IQ2_S / IQ3_S / Q4_K / Q5_K を **`switch`** ディスパッチ。

#### AVX2 共通ユーティリティ

- **`hsum_float_8`**: **`__m256`** 8 float の水平和。
- **`get_scale_shuffle_k4(i)`**: Q4/Q5 の 6-bit scale を **`_mm256_shuffle_epi8`** 用に 32 lane へ複製。**256 B 静的 `k_shuffle[]`**（インデックス **`2*j+0/1`**, **`j=0..3`**）。
- **`MM256_SET_M128I`**: 128-bit scale ベクトルの 256-bit 複製。

#### AVX2 — `quantize_row_q8_K`（非 AVX2: **`quantize_row_q8_K_ref`**）

1. **abs-max**: **`__m256` ×8 load** + **`andnot(signBit)`**。ただし **符号付き `maxv`** 保持のため max 更新は **スカラー**（ggml と同じ semantics）。
2. **`iscale=-127/maxv`**, **`d=1/iscale`**。ゼロブロックは **`d=0`**, zero fill。
3. **int8 化**: **32 要素/iter** — **`mul → round_ps(NEAREST) → cvtps_epi32 → min(127) → packs → permutevar8x32 → store`**。
4. **`bsums`**: 16 要素ずつ **スカラー sum**（AVX 化なし）。
5. ref は **`lrintf`**、AVX2 は **`round_ps`** — いずれも **127 キャップ**。

#### AVX2 — `vec_dot_q4_K_q8_K`（非 AVX2: **`_generic`**）

- **式（1 ブロック）**: **`dot += d·Σ(scale·q4·q8) − dmin·Σ(mins·bsums)`**。 **`d=y.d·f16(x.d)`**, **`dmin=−y.d·f16(x.dmin)`**。
- **generic のコスト**: **`aux8[256]`** へ nibble 全面展開 → 8 要素 **`aux16=q8·a`** ループ × 多段。**~300 B+ スタック/呼び出し**。
- **AVX2**: **`kmask1/2/3`** で **scales[12]** 復号 → **dmin 項を `acc_m` に fmadd** → **64 要素サブループ ×4** で **`maddubs_epi16(q4,q8)` + `madd_epi16(scale,·)` → `sumi`** → **`acc += d·sumi`**。**`aux8` 不要**。
- **返却**: **`hsum_float_8(acc) + hsum(acc_m)`**。

#### AVX2 — `vec_dot_q5_K_q8_K`（非 AVX2: **`_generic`**）

- **`qh[32]`** から **`hmask` 1 bit shift** で第 5 bit を取り **`q5 = q5l + (q5h<<4)`** 相当を **`add_epi8`**。
- 64 要素を **`q5_0/q5_1` × `q8_0/q8_1`** の 2 組で **`maddubs`**。**dmin は `summs` スカラー**。
- **返却**: **`hsum_float_8(acc) + summs`**。

#### IQ2_S / IQ3_S（スカラー据置）

- **1024 エントリ grid**・**signs ビット分岐**・IQ3 **grid1/grid2** 交互 — AVX2 化対象外。
- **IQ2_M モデル**では dot 本体は従来速度だが **層内 Q8 共有による quantize 削減**が主 gain。

#### ビルド・スコープ

- **`-march=native`** → 通常 **`__AVX2__`**。SIMD 対象は **quantize + Q4_K/Q5_K dot の 3 関数のみ**。
- **スコープ外**: IQ2/IQ3 dot、**bsums/abs-max の signed-max 部分**、OpenBLAS 経路、token **emb_lookup** の block dequant、**層跨ぎ Q8 再利用**、KV（float32 のまま）。
- **`-ffast-math` 無効** — AVX2 導入後も IQ/Q8 精度方針は不変。

**`doc/design.md`**: **`cpu-blas`** のバリアント表・ディレクトリ表・量子化と行列積・実行時挙動・制約・トラブルシュートを上記に追随。**`cpu-blas/Makefile`** の **`make openblas`** と **`cblas.h` 未検出時の案内**を追記。**「`cpu-blas`：Q8_K 活性化 GEMV（層内共有・AVX2）」** 節を新設し、ChangeLog 同等の深掘り（3 経路比較・数式・型別 dot・OpenBLAS 分担・IQ2_M gain 内訳）を設計仕様として記載。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 03:53:52

**`README.md`**・**`README.en.md`**: **高度な機能（マルチターン・Thinking）** 節を **公式の想定 / 本リポジトリの現状 / 参考 URL** の構成に整理。テンプレート背景・マルチターン・Thinking 向けの **技術参考リンク**を追記。thinking マーカーが ChatML 特殊トークンではなく通常テキストとしてトークン化される旨を明記。

**`doc/design.md`**: **「高度な機能（マルチターン・Thinking）」** を上記 README と同趣旨に更新（公式／本リポジトリの対比、回避策、(3) KV 再利用未対応、thinking マーカーのトークン化、外部参考は README へ誘導）。**`print_tok` / `is_special`** の説明を修正。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 02:21:36

**ドキュメント**: マルチターン対話・Thinking モードの **対応状況と配慮**を **`README.md`**・**`README.en.md`**・**`doc/design.md`** に追記（公式 Qwen3 の `enable_thinking` / `/think` / `/no_think`、1 ターン固定 `chat_encode`、thinking 生出力の注意、`is_special` の範囲）。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 01:50:43

**`qwen3-8b/cpu/`** — **`cpu-blas`** と同形式の **Prefill progress bar** と prefill / decode スループット要約を stderr に出力。

- **`main.c`**: **`prefill_progress_update`** / **`prefill_progress_done`** / **`decode_progress_done`** / **`throughput_summary`** を追加（**`cpu-blas/main.c`**・**`gpu-cuda/main.c`** と同形式）。プロンプト区間は **1 トークンずつ forward** しながら **`Prefill [====...]`** バー（幅 40）を `\r` で更新。終了時の tok/s 一行表示を **prefill / decode / total** の stderr 要約に置き換え。

**`doc/design.md`**: **`cpu`** のバリアント表・ディレクトリ表・実行時挙動を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 01:46:44

**`qwen3-8b/cpu-blas/`** — **Prefill progress bar** と prefill / decode スループット要約を stderr に出力。

- **`main.c`**: Bonsai.c（**`bonsai-8b/cpu-blas/main.c`**）および **`gpu-cuda/main.c`** と同形式の **`prefill_progress_update`** / **`prefill_progress_done`** / **`decode_progress_done`** / **`throughput_summary`** を追加。プロンプト区間は **1 トークンずつ forward** しながら **`Prefill [====...]`** バー（幅 40）を `\r` で更新。prefill 完了時・decode 完了時に tok/s を表示し、終了時に **prefill / decode / total** のスループット一覧を stderr に出す。

**`doc/design.md`**: **`cpu-blas`** のバリアント表・ディレクトリ表・実行時挙動を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-23 01:29:29

**`qwen3-8b/cpu-blas/`** — 量子化 GEMV を **Q8_K 活性化 + 整数内積**に変更。

- **`main.c`**: 量子化 GEMV（IQ2_S / IQ3_S / Q4_K / Q5_K）で、per-row **float[256] dequant** を廃止。**`quantize_row_q8_K`** で活性を Q8_K 化し、**`vec_dot_*_q8_K`**（**ggml-cpu/quants.c** / **llama.cpp** の **`ggml_vec_dot_*_q8_K`** 準拠）で重み行と整数内積。**`State.q8`**（**`hidden_dim / QK_K`** 分）を追加。出力行の OpenMP 並列（**`mm_quant_rows`**）は維持。
- **`Makefile`**: **`CFLAGS`** に **`-march=native`** を追加。

**`doc/design.md`**: **`cpu-blas`** のバリアント表・実行時挙動・量子化と行列積・制約を上記に追随。

**`README.md`**・**`README.en.md`**: 実行経路表・OpenBLAS 節・CPU OpenMP + OpenBLAS 節・トラブルシュート・実装を読む順序を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-22 23:48:05

**CPU OpenMP + OpenBLAS 版（`cpu-blas`）** を追加。

- **`qwen3-8b/cpu-blas/main.c`**: **`cpu-multicore`** と同一デコーダ。**F32 GEMV**（**`cblas_sgemv`**）と **Attention の K 内積・V 合成**を OpenBLAS に集約。IQ2_S / IQ3_S 等の量子化 GEMV は **`cpu-multicore`** 同等の OpenMP 行並列。**`openblas_set_num_threads(1)`** で OpenBLAS 側は 1 スレッド固定（並列度は **`OMP_NUM_THREADS`**）。**`-ffast-math`** は IQ 量子化で数値が崩れるため Makefile では無効。
- **`qwen3-8b/cpu-blas/Makefile`**: **`pkg-config openblas`** で include / link を自動取得。ヘッダが非標準パスの場合は **`CPPFLAGS`** で指定（Debian/Ubuntu の pthread ビルド例をコメント記載）。
- **`qwen3-8b/Makefile`**: **`build.cpu-blas` / `run.cpu-blas`** を追加。**`all`**・**`clean`** の対象ディレクトリに **`cpu-blas`** を含める。
- **`.gitignore`**: **`cpu-blas/qwen3-cpu-blas`** を追加。

**`README.md`**・**`README.en.md`**: 実行経路表・ディレクトリツリー・前提パッケージ・ビルド／実行・トラブルシュート・「実装を読みたい人へ」を更新。

**`doc/design.md`**: バリアント表・ディレクトリ表・Make ターゲット表・ビルド例・実行時挙動・量子化と行列積・トラブルシュートを上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-22 18:35:51

**CUTLASS v4.5.0 へ更新・ビルド警告整理**（**`gpu-cuda-nvfp4`**）:

- **`CUTLASS_TAG`**: **`v3.9.0`** → **`v4.5.0`**。CUDA 13 非推奨ベクトル型（**`long4`** 等）警告は CUTLASS 側で解消。**`-Wno-deprecated-declarations`** を削除。
- **`BLACKWELL_NVCCFLAGS`**: v4.5.0 の **`sm100_static_tile_scheduler.hpp`** 由来 nvcc #20012 用に **`-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored`** を追加（CUTLASS 公式 **`compiler.py`** と同オプション）。Makefile コメントに背景・参考 URL を記載。
- **`cutlass` ターゲット**: **`third_party/cutlass`** の clone タグが **`CUTLASS_TAG`** と不一致なら再 clone（旧 v3.9.0 からの自動アップグレード）。
- **`fp4_gemm.cu`**: 未使用変数 **`num_row_tiles`** を削除。

**`README.md`**・**`README.en.md`**: CUTLASS **v4.5.0** 利用・非推奨型警告解消を明記。**`-Wno-deprecated-declarations`** の記述を削除。

**`doc/design.md`**: **`CUTLASS_TAG`** 変数表・**`BLACKWELL_NVCCFLAGS`** 説明・NVFP4 ビルド注意・**`third_party/cutlass`** 行を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-22 18:09:40

**`qwen3-8b/gpu-cuda-nvfp4/Makefile`**:

- **`BLACKWELL_NVCCFLAGS`**: **`-std=c++14`** → **`-std=c++17`**（CUTLASS NVFP4 は C++17 必須）。**`FP4_CXXFLAGS`** から重複していた **`-std=c++17`** を除去。
- **`-Wno-deprecated-declarations`**: CUDA 13 で CUTLASS **`platform.h`** が **`long4`** 等の非推奨ベクトル型により **`-Wdeprecated-declarations`** を発するため、当リポジトリ側で修正できない警告を抑制。

**`doc/design.md`**: **`BLACKWELL_NVCCFLAGS`** 変数表・NVFP4 実装メモのビルド注意を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-22 17:55:46

**CUDA ディレクトリ分割** — NVFP4 なし **`qwen3-8b/gpu-cuda/`** と NVFP4 専用 **`qwen3-8b/gpu-cuda-nvfp4/`** に分離。

- **`gpu-cuda/`**: FP16 線形層のみ。**`make build` / `make run`** が既定。旧 **`build.no-fp4` / `run.no-fp4`** を廃止。**`fp4_*`**・**`third_party/cutlass`** を **`gpu-cuda-nvfp4/`** へ移動。
- **`gpu-cuda-nvfp4/`**: **`BONSAI_FP4=1`** 固定。**`make build` / `make run`** が既定。共有ソース（**`main.c` / `kernels.cu` / `gpu.h` / `polarquant.*`**）は **`../gpu-cuda/`** を参照。出力 **`qwen3-gpu-cuda-nvfp4`**。旧 **`build.fp4` / `build.fp4.polarquant` / `run.fp4.polarquant`** は **`build` / `build.polarquant` / `run.polarquant`** に統合。
- **`.gitignore`**: **`gpu-cuda-nvfp4/`** のビルド成果物を追加。

**`README.md`**・**`README.en.md`**: 2 ディレクトリ構成・コマンド表・ビルド手順・トラブルシュートを更新。

**`doc/design.md`**: バリアント表・ディレクトリ表・Make ターゲット表・CUDA 節・実行時挙動・NVFP4 / PolarQuant 実装メモ・トラブルシュートを上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 22:30:51

**`qwen3-8b/gpu-cuda/`**:

- **PolarQuant-R** KV キャッシュ圧縮を実装（**`polarquant.cu`** / **`polarquant_kernels.cuh`** / **`polarquant.h`**）。**`BONSAI_POLARQUANT=1`** 時は **`kc_pq`/`vc_pq`**（**`PQBlock`** 64 B/head）と **`flash_attn_*_pq_kernel`**。
- **`build.polarquant`** / **`pq-test`** / **`polarquant_verify.cu`** を追加。
- **`build.fp4.polarquant`** / **`run.fp4.polarquant`** — NVFP4 線形 + PolarQuant-R KV の組み合わせビルド。
- **`fp4-test` 修正**: **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** を **`BLACKWELL_NVCCFLAGS`**（**`BLACKWELL_GENCODE`** 固定）でコンパイル。旧 **`compute_86` PTX** ビルドでは sm_120 MMA が **`Arch conditional MMA instruction... Aborting`** で失敗していた問題を解消。
- **`MAIN_OBJ`**（**`main.bfp4*.pq*.o`**）を **`KERNELS_OBJ`** と同様にフラグ別名化。

**`README.md`**: **`build.fp4.polarquant`** / **`run.fp4.polarquant`**、PolarQuant 専用節、**`fp4-test`** の Blackwell 必須注記、トラブルシュート（OOM / MMA abort）を追記。

**`doc/design.md`**: 上記に追随（バリアント表・Make ターゲット表・CUDA 変数表・実行時挙動・PolarQuant / NVFP4 実装メモ・トラブルシュート）。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 21:19:33

- **doc**: PolarQuant-R（KV キャッシュ圧縮）の実装説明を追加 — `doc/design.md` にアルゴリズム・VRAM 削減・Flash Attention 統合・ビルド手順、`README.md` / `README.en.md` に `build.polarquant` / `pq-test`、ファイル表に `polarquant.*` を追記。

## 2026-05-21 21:16:24

**`doc/design.md`**: **`build.fp4`** のロード／実行フロー（H2D 時 **NVFP4** 化、線形 **FP16 VRAM なし**、**`fp4_gemv_cached`** による decode、M≥128 の CUTLASS GEMM）を **「実行時の挙動」** と **「CUDA NVFP4 実装メモ」** に整理。VRAM 目安・decode 低速のトラブルシュート行を追記。

**`README.md`**・**`README.en.md`**: 起動ログ文言（**`GEMM M>=128, GEMV decode`**）、実行経路表・読み方ガイドを現行実装に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 21:04:42

**`doc/ChangeLog`** を **`doc/ChangeLog.md`** にリネーム。**`README.md`**・**`README.en.md`**・**`doc/design.md`** のパス表記を追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 20:53:55

**`qwen3-8b/gpu-cuda/`**（**`db8ce6f`**）: **`BONSAI_FP4`** 時に線形層の **FP16 VRAM 複製を廃止**し **NVFP4 キャッシュのみ**（**`upload_linear_fp4`** → **`fp4_qwen3_weight_from_f16_host`**）。**`fp4_gemm`** に **FP4 GEMV**（**`fp4_gemv_cached`** / **`fp4_gemv_batch_cached`**）を追加し、**`fp4_qwen3_mm`** が M=1 デコードと Prefill バッチを経路分岐。

**`fdd860d`**（同ブランチ・前コミット）: **`main.c` / `kernels.cu`** で Blackwell **NVFP4** ロードと **`gpu_model_create`** の FP4 採用、**`fp4_gemm`** 拡張。

**`README.md`**・**`README.en.md`**: **`build.no-fp4` / `build.fp4`** のロード・実行対照表、起動ログ、**`NVFP4 quantize failed`** のトラブルシュート。

**`doc/design.md`**: 上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 20:18:22

**`qwen3-8b/gpu-cuda/`**: **`fp4_bonsai`** を **`fp4_qwen3`**（CUTLASS NVFP4 ブリッジ）に置換。**`Makefile`** に **`build.fp4`**・**`build.no-fp4`**・**`run.no-fp4`** を追加。既定 **`run` → `build.fp4`**（Blackwell）。**`KERNELS_OBJ`** で **`BONSAI_FP4`/`FA_BR` 別オブジェクト**。**`fp4_verify.cu`** と **`fp4-test`** ターゲット。

**`README.md`**・**`README.en.md`**: FP16 汎用ビルドと Blackwell NVFP4 の手順を分離。既定 **`make run` が `build.fp4` である**注意を追記。

**`doc/design.md`**: 上記に追随（バリアント表・ディレクトリ表・Make ターゲット・CUDA 節・実行時挙動・制約・トラブルシュート）。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 17:44:31

**`README.md`**・**`README.en.md`**: **`gpu-cuda/`**（CUDA ビルド・実行・要件）を追記。実行経路表・ディレクトリツリー・モデル取得（**`make model`**）を更新。集約 Makefile のターゲット名に追随（**`build.cpu`**・**`build.cpu-multicore`**・**`build.xdna2-bfp16`** 等）。CUDA のトラブルシュート・**`make clean`** 注記を追加。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 17:42:28

**`qwen3-8b/gpu-cuda/`**: NVIDIA **CUDA** 向けテキストデコーダ推論を追加（**`main.c`**・**`kernels.cu`**・**`gpu.h`**）。ロード時に CPU で逆量子化 → **FP16** → VRAM、**Prefill** は **`gpu_forward_prefill`**（プロンプト全トークン並列）、**Decode** は 1 トークン **`gpu_forward`**。線形層は **FP16 GEMV**、Attention は **Flash Attention**（GQA・`head_dim=128`）。サンプリングは logits の **D2H** 後に CPU。出力バイナリは **`qwen3-gpu-cuda`**（当ディレクトリの **`Makefile`** で **`make build` / `make run`**）。既定 **`CUDA_GENCODE`** は PTX **`compute_86`**（JIT 対応）。**Blackwell**（RTX 50 系等）向けに **`make blackwell`**（apt CUDA 11 除去 → CUDA 13 導入 → **CUTLASS** `third_party/cutlass` 取得 → **`BONSAI_FP4=1`**・**`sm_120a`**・**`FA_BR=32`**）を用意。集約 **`qwen3-8b/Makefile`** には未統合（**`build.gpu-cuda`** なし）。

**`doc/design.md`**: 概要・実装バリアント表・ディレクトリ表・ビルド／実行（CUDA 節）・実行時挙動・トラブルシューティングに **`gpu-cuda/`** を追記。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 16:50:16

**`qwen3-8b/gpu/`** を **`qwen3-8b/gpu-rocm/`** にリネーム（**`main.c`**・**`Makefile`** を移動）。集約 **`qwen3-8b/Makefile`** の **`build.gpu` / `run.gpu`** を **`build.gpu-rocm` / `run.gpu-rocm`** に変更し、実行パスを **`gpu-rocm/qwen3-rocm`** に更新。**`clean`** の対象ディレクトリも **`gpu-rocm`** に追随。

**`.gitignore`**・**`README.md`**・**`README.en.md`**・**`qwen3-8b/cpu-multicore/main.c`**・**`qwen3-8b/gpu-rocm/main.c`**（ビルド注記）のパス・Make ターゲット表記を上記に追随。

**`doc/design.md`**: 概要・実装バリアント表・ディレクトリ表・Make ターゲット表・ビルド例の **`gpu/`** / **`build.gpu`** 表記を **`gpu-rocm/`** / **`build.gpu-rocm`** に更新。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 16:46:11

**`qwen3-8b/hf-model.py`**: 削除。**`make model`**（**`wget`** + **`.sha256sum`**）に GGUF 取得を一本化。

**`doc/design.md`**: **`hf-model.py`** 関連の記述を削除し、**「モデル参照」** を **`make model`** と手動取得の 2 通りに整理。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-21 16:45:20

**`qwen3-8b/Makefile`**: **`model`** ターゲットを追加。**`gguf.txt`** の URL（`blob/main` → `resolve/main`）から **`wget`** で **`$(MODEL)`** を取得し、**`$(MODEL).sha256sum`** が存在するとき **`sha256sum --check`** で検証。チェックサムファイル欠如・検証失敗時はエラー終了し、破損ファイルは削除。

**`doc/design.md`**: ディレクトリ表の **`qwen3-8b/Makefile`** 一行、Make ターゲット表に **`model`**、ビルド例、**「モデル参照」**（**`make model`**・手動 **`wget`**）を反映。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-18 20:38:59

**`qwen3-8b/hf-model.py`**: **`qwen3-8b/gguf.txt`** の Hugging Face **`blob` / `resolve` URL** を解釈し、**`hf download`** で既定配置先（スクリプトと同じディレクトリ、**`--local-dir`** で変更可）に取得。開始時に **`hf auth login`** を実行（**`HF_TOKEN`** または **`--token`** で非対話可）。Hub の tree API から当該ファイルの **LFS `oid`（SHA256）** を取得し、ダウンロード後のファイルと照合する。旧ファイル名 **`download_gguf_hf.py`** は **`hf-model.py`** に変更。

**`.gitignore`**: **`__pycache__/`** と **`*.py[cod]`** を追加（**`qwen3-8b/__pycache__`** 等を Git 対象外に）。

**`doc/design.md`**: ディレクトリ表の **`.gitignore`**・**`gguf.txt`**・**`hf-model.py`** の一行、**「モデル参照」** に上記を反映。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-18 20:18:49

**`xdna-gemv/`** を **`qwen3-8b/xdna2/xdna-gemv/`** に移動。**`gen-xdna-gemv-stubs.py`** はスクリプト隣接の **`kernels/`** を既定出力とするよう変更。**`qwen3-8b/Makefile`**・**`qwen3-8b/xdna2/Makefile`** の **`gen-xdna-kernels`**、`README.md` / **`README.en.md`** / **`doc/design.md`** / 同梱ドキュメント・**`qwen3-8b/xdna2/main.c`** のパス表記を追随（**`qwen3-8b` からの `XDNA_GEMV_DIR`** 例は **`xdna2/xdna-gemv/kernels`**）。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-16 03:03:47

**`qwen3-8b/Makefile`**: 集約 **`build.*` / `run.*`** を **サブディレクトリ名**に揃えた（**`build`/`run` → `build.cpu`/`run.cpu`**、**`build.omp`/`run.omp` → `build.cpu-multicore`/`run.cpu-multicore`**、**`build.rocm`/`run.rocm` → `build.gpu`/`run.gpu`**、**`build.xdna2.bfpx`/`run.xdna2.bfpx` → `build.xdna2-bfp16`/`run.xdna2-bfp16`**）。**`build.xdna2` / `run.xdna2`** は不変。

**`doc/design.md`**: 実装バリアント表・ディレクトリ表・Make ターゲット表・ビルド例・**`XDNA_INCS`** 説明を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-16 02:38:57

**`qwen3-8b/cpu-multicore/main.c`**: ファイル先頭コメントを **`qwen3-8b/`** 配下のパス（本ファイル・**`qwen3-8b/gpu/main.c`**）とルート **`Makefile`** の **`make build.cpu-multicore`**／**`cpu-multicore/`** 単体 **`make build`**（出力 **`qwen3-cpu-omp`**）に合わせて更新。

**`doc/design.md`**: ディレクトリ表の **`qwen3-8b/cpu-multicore/main.c`** 一行を上記に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-16 02:22:26

**`xdna-gemv/toolchain/README.md`**: §0「最初にお読みください」の**日本語の推敲**（`qwen3-xdna2`／公式サンプル／Linux **AMD NPU** 文書／ACRi チュートリアルの読みやすさ）。**東京科学大学（2026年現在の名称。旧・東京工業大学）** およびマニュアル URL が **`titech.ac.jp` のまま**であることの注記を反映。

**`doc/design.md`**: ディレクトリ表および**補足：ドキュメント間の役割**における **`xdna-gemv/toolchain/README.md`** の説明を上記の内容に追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 14:12:50

**`qwen3-8b/` の実行経路ごとにディレクトリを分割**し、いずれも **`main.c`** に統一。**`cpu/`**・**`cpu-multicore/`**・**`gpu/`**・**`xdna2/`**・**`xdna2-bfp16/`** にそれぞれ **`Makefile`** を配置。ルートの **`qwen3-8b/Makefile`** は **`make -C`** で各サブディレクトリに委譲し、実行ファイルパスも **`cpu/qwen3-cpu`** 等に変更。**`.gitignore`**・**`README.md`**・**`README.en.md`**・**`doc/design.md`**・**`doc/ChangeLog`**（コード例の実行パスのみ一部更新）・**`xdna-gemv/`** 配下の参照を追随。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 13:40:39

**`xdna-gemv/`** に **NPU GEMV 用 ctrlcode 準備**（**`bf16-gemv-*.bin`**）関連を集約。旧 **`tools/gen-xdna-gemv-stubs.py`** → **`xdna-gemv/gen-xdna-gemv-stubs.py`**（既定出力 **`xdna-gemv/kernels/`**）、旧 **`xdna-kernels/`** → **`xdna-gemv/kernels/`**（**`README.md`**・**`Makefile`**・スタブ **`.bin`**）、旧 **`xdna-gemv-toolchain/README.md`** → **`xdna-gemv/toolchain/README.md`**。入口として **`xdna-gemv/README.md`** を追加。**`qwen3-8b/Makefile`** の **`gen-xdna-kernels`**、**`README.md`** / **`README.en.md`** / **`doc/design.md`** / **`qwen3-8b/xdna2/main.c`** / **`xdna-gemv/toolchain/README.md`** / **`xdna-gemv/kernels/README.md`** のパス・記述を追随。

**`doc/ChangeLog`**: 本エントリ（あわせて過去ブロック内の **`XDNA_GEMV_DIR`** 例を **`../xdna-gemv/kernels`** に更新）。

## 2026-05-14 13:32:53

**`xdna-gemv-toolchain/README.md`**: 用語・文意の整理、です・ます調への統一、`EXEC_CMD` 表記修正など**日本語校正**。

**`doc/design.md`**: ディレクトリ表の **`xdna-gemv-toolchain/README.md`** 一行を手引きの位置づけに合わせて更新。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 13:25:04

**`xdna-gemv-toolchain/README.md`**: **Xilinx mlir-aie / AMD IRON / Peano / `aiecc`** の公式 README に沿った **Ubuntu／XRT／wheel／環境セットアップの具体コマンド**、IRON **`GEMV(M,K)` と `bf16-gemv-<n>x<d>.bin`** の対応、**`aiecc --aie-generate-npu-insts`** の説明、**ioctl 単体ランタイム（本リポ）と XRT 検証パスの差**、公式リンク一覧を日本語で記載。

**`xdna-kernels/README.md`**: §8 に **`xdna-gemv-toolchain`** への導線（§8.2）。

**`doc/design.md`**: ディレクトリ表 **`xdna-gemv-toolchain/README.md`** と **ドキュメント間の役割**、「詳細（入門）」、**XDNA2 NPU 実装メモ** に同文書参照を追加。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 13:13:06

**`xdna-kernels/Makefile`**: **`XDNA_GEMV_BIN_URL_BASE`** の既定値を Makefile 内に記述（**`make` のみで取得を試行**）。**`help`** と **`xdna-kernels/README.md` §8.1** を追随。**`doc/design.md`** の一行説明を更新。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 04:51:57

**`xdna-kernels/Makefile`**: **`XDNA_GEMV_BIN_URL_BASE`** を指定したとき **`make`** で **`bf16-gemv-*.bin`** を **5 本まとめて取得**（`curl` / `wget`）。URL 未設定時は明示エラー。**`make help`**。

**`xdna-kernels/README.md`**: **§8.1** で上記の使い方を記載（公開ミラーはリポジトリでは保証しない旨）。

**`doc/design.md`**: ディレクトリ表に **`xdna-kernels/Makefile`** を追加。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 04:41:49

**`xdna-kernels/README.md`**: **§0 で ctrlcode を先に定義**。**§3 で ROCm/HIP の GPU カーネルと対比**し、**オーバーレイ＋ctrlcode** が HIP の 1 カーネルと 1:1 にならないこと、**HIP と同じ意味でのユーザー主導プログラミングとは典型的には異なるがツールチェイン経由では間接的に生成可能**であることを **§3.3 で明示**。用語ミニ辞典・処理フロー図・見出し番号の整理（§4〜§11）を維持。

**`doc/design.md`**: ディレクトリ表の **`xdna-kernels/README.md`** 一行と XDNA2 節の **「詳細（入門）」** を上記内容に合わせて拡張。**XDNA2 NPU 実装メモ**末尾に **GPU カーネル対比・プログラム可能性**は同 README（§3）参照を追記。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 04:28:45

**`xdna-kernels/README.md`**: **GEMV／**`n`**×**`d`**／NPU と ctrlcode／スタブとマジック **`GQF3XDNA`**／**`XDNA_GEMV_DIR`** と **`--xdna-status`**／8B ファイル対応／実 ctrlcode への差し替え・再生成・実行例**など、初心者向けに構成を拡充した。

**`doc/design.md`**: リポジトリ構成表に **`xdna-kernels/README.md`** と **`tools/gen-xdna-gemv-stubs.py`** を追加。XDNA2 実行例ブロックの直後に **`xdna-kernels/README.md`** への参照を記載。実装メモの「制御コードバイナリ」説明を **スタブ同梱・ランタイムでは非スタブのみロード**に合わせて修正。**`load_gemv_kernel`** の箇条書きに **スタブ拒否**を明記。

## 2026-05-14 04:18:35

**`xdna-kernels/`**: Qwen3-VL-8B テキスト経路向け **`bf16-gemv-<n>x<d>.bin`** を **5 ファイル（64 B プレースホルダ）**として追加。先頭マジック **`GQF3XDNA`** で識別し、**ERT に載せない**（誤実行防止）。

**`tools/gen-xdna-gemv-stubs.py`**: 上記スタブの生成スクリプト（出力先ディレクトリを引数で変更可）。

**`qwen3-8b/Makefile`**: **`gen-xdna-kernels`** ターゲット（ルートの `xdna-kernels/` を更新）。

**`qwen3-8b/main-xdna2.c` / `main-xdna2-bfpx.c`**: `load_gemv_kernel` がスタブ `.bin` を拒否。**`print_xdna_gemv_ctrlcode_report`** で **`[STUB]` / `[ OK ]` / `MISS`** を表示。解説メッセージをスタブ・部分欠けに対応。

**`xdna-kernels/README.md`**, **`README.md`**, **`README.en.md`**, **`doc/design.md`**: プレースホルダと MLIR-AIE 差し替えの説明を追記。

## 2026-05-14 04:03:53

**`qwen3-8b/main-xdna2.c`**: NPU 経路の `amdxdna` UAPI 適合を全面修正。

- **`CREATE_BO`**：`amdxdna_drm_create_bo_ioctl` は `args->vaddr != 0` を **無条件で `-EINVAL`** にするため、**常に `vaddr=0`** で発行。**`DEV_HEAP` の `userptr`** は **ユーザー側 `mmap()` 時**に `amdxdna_hmm_register` が `vma->vm_start` を記録する仕様なので、`GET_BO_INFO` の戻り `vaddr` 値（`AMDXDNA_INVALID_ADDR` = `~0UL` を返しうる）に依らず **`SHMEM` / `DEV_HEAP` / `CMD` は必ず `mmap()`** するよう経路を整理。`DEV_HEAP` は `MAP_HOST_BUFFER` 直前にページフォールトを起こすため**全頁タッチ＋`mlock()`**（失敗時は警告）。
- **`CREATE_HWCTX`**：`aie2_solver.c::sanity_check` の `qos_meet` が **`gops=30000, fps=1`** で必ず失敗していた（`request_gops` が NPU の `max_opc × cu_clk` から導く `cgops ≈ 3686` を超過）ため、**`amdxdna_qos_info` を全フィールド 0** に変更（`is_valid_qos_dpm_params == false` で最大 DPM へフォールバック）。`num_tiles` は `aie2_hwctx_col_list` の契約に従い **`ncol × core.row_count` を最優先**、`shim + mem + core` 合算と `1` 行をフォールバックとして順次試行。**`XDNA_NUM_COL`** に加え **`XDNA_NUM_TILES`**（強制値）・**`XDNA_HEAP_SIZE`**（DEV_HEAP バイト数）の上書きを追加。失敗時のエラーログに `core/mem/shim` 行数と試行ヒントを表示。
- **`launch_mm_bf16` の CPU フォールバック高速路**：NPU ディスパッチが**最初から不可**な条件（`have_device == 0`、`XDNA_FORCE_CPU`、`XDNA_GEMV_DIR` 未設定）では、GEMV ごとの **フル行列 BF16 展開・`SYNC_BO`・スクラッチ往復**を**省略**し、`main-omp.c` と同じ **ブロック単位の量子化直 GEMV**（`mm_quant_rows_xdna`／`mm_f32_xdna`／`mm_f16_xdna`）で出力するよう分岐。
- **`alloc_weight_scratch` の条件化**：上記 NPU 不可条件では **`w_scratch_bo`（GEMV 最大の BF16 SHMEM、典型 1.2 GiB）と `scratch_f32`（典型 2.4 GiB）を確保せず**スキップ。
- **`weight_prepare_bf16`**：BF16 量子化変換ループを `#pragma omp parallel for` で並列化。

**`doc/design.md`** / **`doc/ChangeLog`**: 上記の追加環境変数と挙動を反映。

## 2026-05-14 03:28:55

**`qwen3-8b/main-xdna2.c`**: **`print_xdna_gemv_ctrlcode_report`** を追加。`npu_open` 後に **`XDNA_FORCE_CPU`**・DRM オープン可否・**`XDNA_GEMV_DIR`**・テキスト経路 **6 形状**の **`bf16-gemv-<n>x<d>.bin`** の読み取り可否（フルパス付き）を標準出力へ一覧する。推論終了後は **NPU GEMV / CPU GEMV** 件数に加え、**すべて NPU／すべて CPU／混在**を短文で表示。コマンドライン **`--xdna-status`** / **`-X`** は GGUF パース・トークナイザ初期化・`npu_open` のみ行い当該レポート出力後に終了する（重みロード・生成なし）。

**`doc/design.md`**: XDNA2 実行例（**`--xdna-status` / `-X`**）、**実行時の挙動**、**コマンドライン補足**、トラブルシューティング、**XDNA2 NPU 実装メモ**、バリアント／ファイル一覧の **`main-xdna2.c` 一行**を上記に合わせて更新。

**`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 03:16:38

**`qwen3-8b/main-xdna2.c`**: **ホスト常駐メモリ削減**を目的に、線形ウェイトをロード時に全件 **BF16 の `AMDXDNA_BO_SHMEM` に複製しない**構成へ変更。**`main-omp.c` と同様**に GGUF **mmap を参照**。埋め込みは **`emb_lookup` 相当の行単位復号**。各 **GEMV 直前**に当該行列だけを **`w_scratch_bo`（単一 BF16 SHMEM）** と **`scratch_f32`** に展開し NPU／CPU GEMV が従来どおり読む。**テキスト経路 GEMV で使うテンソル**のみから **`gemv_max_elems`** を求め VL 側の肥大テンソルはサイズ算出から除外。**`TensorInfo` 名リストは **`free_tensor_index()`** で推論開始前に解放。

**ドキュメント整合**: **`doc/design.md`**（バリアント表・ファイル一覧の一行・実行時挙動・制約の **~16 GB** 記述・**XDNA2 NPU 実装メモ**および BFPX 節での **`qwen3-xdna2` との比較文言**）、**`README.md` / `README.en.md`**（表記「BF16 常駐」・メモリ目安・BFPX 比較文）を現行実装に合わせ更新。

- **`doc/ChangeLog.md`**: 本エントリ。


**`README.md`**: 概要まわりで **欧文と和文が接する箇所**の半角スペースを読みやすさの方針に合わせて整理。**`LLM推論`**・**`Python環境` / `Pythonランタイム`**・**`C` 周り**などの表記を調整。**XDNA2 + BFPX** 節を、ioctl／チャンク GEMV、mmap の解放タイミング、CPU 側 **`mm_bfpx`** と単精度活性、**ビット完全一致の非保証**、品質説明になるよう**自然な文語**へ書き直し。表内の **`BF16` / `BFPX`** 複合語のスペースも合わせて整理。

**`doc/design.md`**: 実装バリアント表・ファイル一覧の **`main-xdna2-bfpx`** 説明、**実行時の挙動（XDNA2 + BFPX）**、フォールバック注記を **`README.md` と同趣旨の用語・語順**に同期。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 01:52:46

**`qwen3-8b/Makefile`**（ドキュメントと整合させるために内容を確定させた整理）:**`TARGET_*` を維持しない**運用へ合わせ、出力バイナリ名 **`qwen3-*`** を各レシピに直書きする方針を維持。セクション見出しの整理、**.PHONY** と **`clean`** の縦並び列挙。ROCm／XDNA の短い説明コメント。**`GPU_ARCH ?= …` と同じ行に `# …` と書くと GNU Make が値末尾に空白を残し `--offload-arch=` が破損することがある**ため、説明だけを次行コメントへ分離。

- **`doc/design.md`**: ファイル構成の **`Makefile` 一行** と **共通変数表** を現状に同期（直書きの出力名、`CC`/`CFLAGS`/`LDFLAGS`、`XDNA_INCS`、`GPU_ARCH` と行末 `#` に関する注意）。
- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 01:23:23

**`README.md`**: 「まず何ができるのか」の表直後に **AMD XDNA** の一般的な解説として外部リポジトリ **[thamada/xdna-overview](https://github.com/thamada/xdna-overview)**（本文 `main.md`・PDF 同梱）へのリンクを追加。タイトル直後に **`README.en.md`** への案内を追加。ディレクトリ構成ツリーに **`README.en.md`** を追記。「実装を読みたい人へ」の最初の項目で **`README.md` / `README.en.md`** を併記。

**`README.en.md`**: 日本語 **`README.md`** と同等構成の **英語版 README** を新規追加。相互に **`README.md`** へリンク。

**`doc/design.md`**: リポジトリの目的節で利用者向け入口を **`README.md` / `README.en.md`** に更新。**`補足：ドキュメント間の役割`** に **`README.en.md`** と、XDNA 背景の外部文書（**xdna-overview**）を追記。**`design.md` 更新時のチェックリスト** に README 2 言語の整合を 1 項目追加。**`ディレクトリとファイル構成`** 表にルートの **`README.md` / `README.en.md`** を追記。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 01:18:09

**`README.md`**: 冒頭で **PyTorch・TensorFlow・JAX・ONNX Runtime** 等への非依存と、**標準 C / ROCm-HIP / OpenMP / `amdxdna` ioctl** の役割を明示。ROCm/HIP が NN フレームワークではない旨を補足。**「なぜライブラリ非依存なのか」** をフォーマルに整理（理解可能性・依存の単純化・実験の自由度・参照実装としての価値。主眼は性能網羅ではなく実装の観察・改造可能性）。

**`doc/design.md`**: **`概要` → `リポジトリの目的とスコープ`** に README と整合するライブラリ非依存の範囲と **`ライブラリ非依存とその意義`** 小節を追加。**`補足：ドキュメント間の役割`** に **`README.md`** の位置づけを追記。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-14 01:09:19

**`qwen3-8b/main-xdna2-bfpx.c`** を追加（出力バイナリ **`qwen3-xdna2-bfpx`**、`make build.xdna2.bfpx`）。`main-xdna2.c` と同じ **amdxdna DRM ioctl + チャンク BF16 GEMV（`ERT_START_NPU`）** 経路を使うが、線形重みは GGUF mmap を長時間載せず、ロード時に **ブロック浮動小数点（各ブロックに BF16 スケール + int8 係数、`BFPU_BLK=64`）** に変換してホストに保持する。RMSNorm 等の小さいテンソルは従来どおり F32 をヒーブに複製。論理形状は **`main-omp.c` の `mm(o,x,w,n_in,n_out)`** と整合させ、`GGUF` が **`[n_in,n_out]`** と並んでいるテンソル（例: **`ffn_down`**）は転置レイアウトとして full dequant してから行単位でエンコードする。**`token_embd.weight` / `output.weight`** も同じ **`bfpx_convert_weight_2d`** で `[dim,vocab]` / IQ 時のストライド布局を吸収。

### 実装ハイライト

- **`BfpxMat` / `bfpx_convert_weight_2d`**: GEMV の `n_out × n_in` と GGUF の `(ne[0], ne[1])` を対応づけ。正方行列では転置 memcpy 分岐に誤入しないよう **`n_in != n_out`** でゲート。
- **mmap の寿命**: `parse_gguf` → `load_weights`（変換）後 **`model_drop_gguf_mmap`** で tensor メタデータ・mmap・モデル fd を解放。
- **堅牢性**: **`bfpx_mat_destroy`** は内側バッファのみ開放（構造体は **`free_weight_ptrs`** が開放）。**`sample_token`** の top-p で **`np==0`** のとき **`pi[-1]`** を読まないガード。**`emb_lookup`** のトークン ID 範囲チェック。
- **NPU**: **`CREATE_HWCTX`** が失敗した環境でも **`npu_close` が fd=-1 で早期 return** する既存動作と整合（GEMV 未確保時の二重解放を避ける）。CREATE_HWCTX 失敗時は CPU **`mm_bfpx`** のみ。

### 周辺更新

- **`qwen3-8b/Makefile`**: **`build.xdna2.bfpx` / `run.xdna2.bfpx`**、`TARGET_XDNA2_BFPX`、`clean`。旧 **`build.xdna2.mmap`** は削除済み。
- **`.gitignore`**: **`qwen3-xdna2-bfpx`**（旧 mmap バイナリ名は削除）。
- **`doc/design.md`**: バリアント表・ファイル構成・Make ターゲット表・XDNA2 実行例・実行時挙動・XDNA2 メモに **`main-xdna2-bfpx.c`** を追記。
- **`doc/ChangeLog.md`**: 本エントリ。

### ビルド・実行

```bash
cd qwen3-8b
make build.xdna2.bfpx
./xdna2-bfp16/qwen3-xdna2-bfpx path/to/model.gguf -p "Hi" -n 8
# NPU 制御コード・環境変数は xdna2/main.c と同様（XDNA_GEMV_DIR / XDNA_FORCE_CPU / XDNA_NUM_COL 等）
```

## 2026-05-13 23:57:46

**`qwen3-8b/main-xdna2.c`** を新規追加。AMD Ryzen AI 内蔵 **XDNA2 NPU** 用推論バックエンドを、`amdxdna` カーネルモジュール（`drivers/accel/amdxdna`）の DRM ioctl を直接叩く形で実装した。XRT (Xilinx Runtime) や xdna-driver の C++ shim 等の外部ユーザランドに依存しない単一 C ソース構成。

### 実装ハイライト

- **UAPI セルフコンテイン**: `<drm/amdxdna_accel.h>` 相当（`amdxdna_drm_create_hwctx` / `amdxdna_drm_create_bo` / `amdxdna_drm_get_bo_info` / `amdxdna_drm_sync_bo` / `amdxdna_drm_exec_cmd` / `amdxdna_drm_get_info` / `DRM_IOCTL_AMDXDNA_*` 等）を inline で持つ。
- **ERT パケット ABI**: `ert_packet_header` / `amdxdna_cmd_start_npu` / `amdxdna_cmd_chain` を実装し、`ERT_START_NPU` 経由でハードウェアコンテキストにコマンドを投げる。
- **デバイス抽象層**: `npu_open` で `/dev/accel/accelN` を順に開き、`DRM_AMDXDNA_QUERY_AIE_METADATA` / `_VERSION` / `_FIRMWARE_VERSION` / `_CLOCK_METADATA` でトポロジ取得。`DRM_IOCTL_AMDXDNA_CREATE_HWCTX` でハードウェアコンテキストを作成。`DRM_IOCTL_AMDXDNA_CREATE_BO` で 64 MiB の `AMDXDNA_BO_DEV_HEAP` 命令アリーナを確保し、`AMDXDNA_BO_CMD` / `_SHMEM`（umq・ログ）を補助的に確保する。
- **コマンド投入**: `npu_submit_start_npu` が `ert_packet_header` + `cu_mask` + `amdxdna_cmd_start_npu`（命令バッファアドレス・サイズ・引数）を `CMD` BO に書き込み、`DRM_IOCTL_AMDXDNA_EXEC_CMD` で hwctx に投げる。完了は `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` で待機。
- **重みのレイアウト**: 量子化重みをロード時に CPU で **BF16** に展開し、`AMDXDNA_BO_SHMEM` 型 DRM バッファオブジェクトに格納（NPU 列 DMA からも CPU からも同じバイト列を見られる）。norm 系は CPU 通常メモリに留める。
- **GEMV ハイブリッド**: NPU 上での実 GEMV は事前コンパイル済み制御コード `$XDNA_GEMV_DIR/bf16-gemv-<n>x<d>.bin`（MLIR-AIE / IRON ツールチェイン生成）を命令アリーナに読み込んで `ERT_START_NPU` でディスパッチする。制御コードが未配置・`/dev/accel/accelN` 不可・`XDNA_FORCE_CPU=1` のいずれかで透過的に OpenMP BF16 CPU GEMV にフォールバックする（bit-identical な結果）。
- **CPU 側カーネル**: RMSNorm（全体／ヘッド）、RoPE、Attention（per-head パラレル）、SwiGLU、残差加算、softmax、サンプリングを OpenMP で `main-rocm.c` の GPU カーネル粒度に合わせて並列化。

### 周辺更新

- **`qwen3-8b/Makefile`**: `build.xdna2` / `run.xdna2` ターゲットを追加。`TARGET_XDNA2 = qwen3-xdna2`、`XDNA_INCS` 変数で UAPI ヘッダのパスを上書き可能。`clean` も更新。
- **`doc/design.md`**: 実装バリアント表、ファイル構成表、Make ターゲット表、ビルド・実行方法、実行時挙動、制約事項、トラブルシューティング、`XDNA2 NPU 実装メモ` の節を追加・更新。
- **`doc/ChangeLog.md`**: 本エントリ。

### ビルド・実行

```bash
cd qwen3-8b
make build.xdna2
sudo usermod -aG render "$USER"   # /dev/accel/accel0 アクセスのため
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8
XDNA_FORCE_CPU=1 ./xdna2/qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8  # CPU 強制
```

## 2026-05-13 23:25:34

**`doc/design.md`** のアーキテクチャ説明を拡充。主要データ構造、GGUF パース、重みテンソル対応、量子化と行列積、Tokenizer / ChatML、CPU forward、ROCm forward、生成ループとサンプリングについて、現行実装に沿った詳細説明を追加した。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-13 23:21:20

**`qwen3-8b/main-rocm-fullgpu-flash-opt2.c`** を削除。`make build.rocm.fullgpu.flash.opt2` と同一ロジックの別名ソース・バイナリは不要となったため。**`qwen3-8b/Makefile`** から対応ターゲットを削除し、**`README.md`**・**`doc/design.md`**・**`.gitignore`**・**`qwen3-8b/main-rocm.c`** のビルド注記を **`build.rocm` / `qwen3-rocm` に一本化**。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-13 23:12:27

**`README.md`** のモデル配置手順を更新。プロジェクト見出しを **Qwen3.c** に整理し、GGUF 本体をリポジトリに含めない方針を明記したうえで、**`qwen3-8b/gguf.txt`** の Hugging Face URL から `blob/main` を `resolve/main` に置換して `wget` する手順を追加した。

- **`doc/design.md`**: ファイル構成に **`qwen3-8b/gguf.txt`** を追加し、モデル参照節に取得元 URL と SHA256 確認の役割を反映。
- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-13 22:57:14

**`README.md`** を新規作成。Qwen3 系 GGUF テキスト推論エンジンとしての概要、必要環境、モデル配置、SHA256 確認、CPU / OpenMP / ROCm のビルド方法、推論実行方法、主要オプション、トラブルシューティング、実装を読む順序を初心者向けに整理した。

- **`doc/ChangeLog.md`**: 本エントリ。

## 2026-05-13 22:41:41

**`qwen3-8b/`** の **CPU OpenMP** 実装（`main-omp.c`、`make build.omp` → `qwen3-cpu-omp`）および **`doc/design.md`** への **Qwen3-8B 専用節**の追記、`ChangeLog` の更新。

### ソース・ビルド（既存の整理を文書化）

- **`qwen3-8b/main-omp.c`**: `main.c` をベースに OpenMP を付与。**GEMV** は出力行並列、**Attention** はヘッド並列、**RoPE**・**RMSNorm**（全体／ヘッド）・残差・SiLU は `main-rocm.c` の並列単位に相当する粒度で並列化。
- **`qwen3-8b/Makefile`**: `build.omp` / `run.omp`、**`clean`** に `qwen3-cpu-omp` を含む（既存）。

### ドキュメント

- **`doc/design.md`**: 概要に **`qwen3-8b/`** を現行ツリーとして明記。**[Qwen3-8B（`qwen3-8b/`）](#qwen3-8bqwen3-8b)** 節を追加（スコープ、`qwen3vl.*`、IQ デ量子化方針、Q/K norm、ChatML、`main.c` / `main-omp.c` / `main-rocm*.c`、Make ターゲットとバイナリ、ビルド例）。**ビルドと実行**に `qwen3-8b` 向け CPU/OpenMP/ROCm のサブ節を追加。
- **`doc/ChangeLog.md`**: 本エントリ。

