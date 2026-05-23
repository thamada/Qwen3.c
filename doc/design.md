# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog.md` を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

### リポジトリの目的とスコープ

本リポジトリは、**Qwen3 系（Qwen3-VL-8B-Instruct）** の **GGUF** 形式モデルを、**単一または少数の C／HIP／CUDA ソースファイル**からビルド可能な形で **推論（テキスト生成）**するエンジンである。**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムにはリンクしない。** コアは **標準 C と `libm`**。AMD GPU 版は **ROCm/HIP**（コンパイラ・ランタイムでありニューラルネット用の高レベルフレームワークではない）、NVIDIA GPU 版は **CUDA Toolkit / `nvcc`**（同様に低レベル）、CPU 並列は **OpenMP**（**`cpu-multicore`**）または **OpenMP + OpenBLAS**（**`cpu-blas`** — BLAS は F32 GEMV / Attention 集約。量子化 GEMV は **Q8_K 活性化 + ggml 準拠の整数内積**）、XDNA2 NPU 版は **`amdxdna` DRM ioctl（UAPI）** を直接利用する。Python ランタイムや `torch` に依存する層は置かない。**GGUF の読み取り・トークナイズ・Transformer フォワード・サンプリング**を一連のコードパスとして理解・改変しやすくすることを目的とする。学習・ファインチューニング・バッチ推論の最適化はスコープ外であり、主に **単発プロンプトからのテキスト生成**（`-p` + ChatML 1 ターン）を想定する。**マルチターン対話・Thinking モード**は Qwen3 ファミリーで重要だが本リポジトリでは未対応（**「高度な機能（マルチターン・Thinking）」** 参照）。

#### ライブラリ非依存とその意義

高レベルフレームワークに載せた推論は実装が簡潔になり高速化もしやすいが、**計算手順・メモリ配置・アライメント・量子化レイアウト**等がランタイム内部に隠れやすい。本リポジトリはその抽象層に依存せず、推論の実体を **C の明示的なコードパス**として観察・検証・変更できる状態に置く。フレームワークの代替を第一目的とするものではない。

- **理解可能性**: モデルファイルからの読み取り、バッファ配置、演算順序をソースと本書で追跡できる。
- **依存関係の単純化**: Python 環境や大規模 ML スタックを前提とせず、コンパイラと必要最小限の実行環境で経路を確認できる。
- **実験の自由度**: 量子化・メモリ表現（例: BFPX）・CPU/GPU/NPU の分担・`/dev/accel` への直接アクセスなど、抽象化に縛られやすい領域を試しやすい。
- **参照実装としての価値**: 最小構成で Qwen3 系デコーダ推論が成立する見取り図として、他スタックとの比較・検証の基準になる。

**最高性能や機能網羅を第一目的とはしない。** 主眼は、LLM 推論をブラックボックスにせず、実装の細部を把握したうえで改造できることである。利用者向けの入口説明は **`README.md`**（日本語）および **`README.en.md`**（英語）に詳しい。

文中の「decoder-only」「GQA」「FlashAttention 系デコードカーネル」等は、**Transformer デコーダの一般的なパターン**を指す。**推論ソースはすべて `qwen3-8b/` 配下であり、実行経路ごとに `cpu/`・`cpu-multicore/`・`cpu-blas/`・`gpu-rocm/`・`gpu-cuda/`・`gpu-cuda-nvfp4/`・`xdna2/`・`xdna2-bfp16/` の各ディレクトリに **`main.c`（CUDA 版は **`kernels.cu`** 併用）を置いた単一ソース構成**とする。** 対象例は **Qwen3-VL-8B-Instruct** の **IQ2_S / IQ3_S 等が混在した GGUF**（例: `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`）。**Vision（画像エンコード・deepstack・画像トークン）は実装しない**。**テキスト用デコーダのみ**を実行する。

### 実装バリアント（本リポジトリに含まれるもの）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `qwen3-8b/cpu/main.c` | CPU、単スレッド | GGUF mmap、`qwen3vl.*` パース。線形層は **IQ2_S / IQ3_S / Q4_K / Q5_K** 等を **`QK_K=256` ブロック単位**にデ量子化しつつ GEMV（全重みの float 一括展開なし）。`libm` のみ。**Prefill** は 1 トークンずつ forward し stderr に **progress bar**（**`Prefill [====...]`**、幅 40）と prefill / decode / total の **スループット要約**を出力。 |
| `qwen3-8b/cpu-multicore/main.c` | CPU、**OpenMP** | 上記と同一アルゴリズム。**GEMV** は出力行並列、**Attention** はヘッド並列、`qwen3-8b/gpu-rocm/main.c`（ROCm 版）のカーネル粒度に相当する並列化（RoPE、RMSNorm、残差、SiLU 等）。 |
| `qwen3-8b/cpu-blas/main.c` | CPU、**OpenMP + OpenBLAS** | **`cpu-multicore`** と同一デコーダ・同一 GGUF。**F32 行列積**（**`cblas_sgemv`**）と **Attention の K 内積・V 合成**を OpenBLAS に委譲。量子化 GEMV は **Q8_K 活性化 + 全型 `vec_dot_*_q8_K` 整数内積**（**`__AVX2__`** で IQ2_S/IQ3_S/Q4_K/Q5_K）。**層内 Q8 共有**（**`mm(..., q8_ready)`**）。**RoPE cos/sin キャッシュ**、**prefill 中 LM head スキップ**（**`FWD_NO_LM`**）、**greedy 時 `mm_argmax_row`**（**`FWD_LM_ARGMAX`**）。**F16 埋め込み**は **F16C+AVX2** で 8 要素 SIMD 変換。**Prefill progress bar** とスループット要約を stderr に出力。**`-march=native`** 既定。 |
| `qwen3-8b/gpu-rocm/main.c` | **ROCm / HIP** | ロード時に量子化重みを CPU で **F16** に展開して VRAM に載せ、**フル GPU** パスで推論。**Flash 系デコード注意**・**KV カーネル書き込み**・**レイヤー間のホスト非介在**・GPU サンプリング（top-p 時は logits D2H フォールバック）等を含む。**Prefill progress bar**（**`Prefill [====...]`**、幅 40）と prefill / decode / total の **スループット要約**を stderr に出力（**`cpu-blas`** 同形式）。**`make build.gpu-rocm` の既定 AMD GPU エントリ**。ビルド時 **`GPU_ARCH`** は **`gpu-rocm/Makefile`** が **`rocminfo`** から自動検出（**`detect-gpu-arch`**）。 |
| `qwen3-8b/gpu-cuda/main.c` + `kernels.cu` | **NVIDIA CUDA（FP16）** | **Prefill バッチ** + **Decode 1 トークン**、**Flash Attention**（GQA）。全線形 **FP16 VRAM**（ROCm 同趣旨）。任意で **`build.polarquant`**: KV キャッシュ **PolarQuant-R**（64 B/head、F32 比 ~8×）。サンプリング **logits D2H**。集約 Makefile 外。 |
| `qwen3-8b/gpu-cuda-nvfp4/` + 共有 `gpu-cuda/` | **NVIDIA CUDA（NVFP4）** | 上記と同じ Prefill / Decode / Flash Attention。線形層はロード時 **NVFP4 のみ**（**`fp4_qwen3`**、**`BONSAI_FP4=1`** 固定）、**`token_embd`** は **FP16**、norm は **F32**。線形は **`fp4_qwen3_mm`**（M=1→**FP4 GEMV**、M≥128→CUTLASS GEMM）。任意で **`build.polarquant`**: NVFP4 線形 + PolarQuant-R KV（Blackwell 向け最大 VRAM 節約）。要 **CUDA 13 + CUTLASS + sm_120 系 GPU**。集約 Makefile 外。 |
| `qwen3-8b/xdna2/main.c` | **AMD Ryzen AI NPU (XDNA2)** | **CPU OpenMP 版と同様**に線形ウェイトは **GGUF mmap 上の量子化形式を参照**。埋め込みは行単位ブロック復号。各 **GEMV ごとに**当該重み行列を **`AMDXDNA_BO_SHMEM` の単一 BF16 スクラッチ**へ展開して NPU が DMA、`scratch_f32` でデ量子化～BF16 を兼用。rmsnorm などの小型 F32 も mmap 指す。`DRM ioctl` と **`ERT_START_NPU`** 経路、`/dev/accel/accelN` 不可／制御コード未配置時の **OpenMP BF16 CPU フォールバック（NPU と bit-identical）**は従来どおり。XRT 不要・UAPI inline 持ち運びは不変。**スクラッチサイズはテキスト経路 GEMV に必要な最大要素数のみ**（パーサ済み名前走査、`TensorInfo` は推論前に開放しうる）。**起動時レポートと `--xdna-status` / `-X`** で各形状の **`bf16-gemv-<n>x<d>.bin`** 可否・推論後の NPU/CPU GEMV カウンタを確認できる。 |
| `qwen3-8b/xdna2-bfp16/main.c` | **AMD Ryzen AI NPU (XDNA2) + BFPX ホスト重み** | **`qwen3-8b/xdna2/main.c` と同一の DRM ioctl** および **チャンク BF16 GEMV（NPU 経路の枠組み）** を共有する。**密行列レイアウト**の重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 係数、ブロック長 64）** に変換しホストのみ保持し、GGUF mmap は変換完了後に解放する。**論理形状は OpenMP CPU 版（`cpu-multicore/main.c`）の `mm(..., n_in, n_out)` と一致**させ、`[n_in,n_out]` 型の GGUF 転置は **`bfpx_convert_weight_2d`** で吸収。量子化に加えブロック近似のため、**GEMV で逐次 BF16 に展開する mmap スクラッチ方式（`xdna2/main.c`）と同一ビットでの一致は期待できず**、品質が劣ることがある。NPU 不可時の CPU は **`mm_bfpx`** が単精度浮動小数点数の活性と BFPX 形式の重みの積を計算する。 |

メタデータキーは **`qwen3vl.*`**。Qwen3 固有として、線形射影の直後に **`attn_q_norm` / `attn_k_norm`**（ヘッド長に対する RMSNorm）を挟み、その後 **RoPE** を適用する。チャットは **ChatML**（`<|im_start|>` / `<|im_end|>` 等）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` | ビルド・実行・方針の説明（日本語）。 |
| `README.en.md` | 同上（英語）。 |
| `qwen3-8b/cpu/main.c` | CPU 単スレッド推論。**Prefill progress bar**（**`prefill_progress_*`**）と prefill / decode スループット要約を stderr に出力。 |
| `qwen3-8b/cpu-multicore/main.c` | CPU OpenMP 並列推論。**ソース先頭**に **`qwen3-8b/gpu-rocm/main.c`**（ROCm/HIP）との並列粒度対応、`qwen3-8b/Makefile` の **`make build.cpu-multicore`** と当ディレクトリ単体 **`make build`**（**`qwen3-cpu-omp`**）を記載。 |
| `qwen3-8b/cpu-blas/main.c` | CPU OpenMP + OpenBLAS。**Q8_K GEMV**（全型 AVX2 整数内積・層内 Q8 共有）、**RoPE キャッシュ**、**lm_mode**（prefill LM スキップ / greedy argmax）、**F16 emb F16C**。詳細は **「量子化と行列積」→「`cpu-blas`：Q8_K 活性化 GEMV」**。**Prefill progress bar** を stderr に出力。 |
| `qwen3-8b/cpu-blas/Makefile` | **`qwen3-cpu-blas`** をビルド。**`-ffast-math` 無効**（IQ 量子化の精度維持）。**`-march=native`** 既定。**`openblas_set_num_threads(1)`** は **`main.c`** 実行時。**`make openblas`** で **`libopenblas-dev`** / **`libgomp1`** を apt 導入。**`cblas.h` 未検出時**はエラーメッセージで **`make openblas`** と **`CPPFLAGS`** 例を案内。 |
| `qwen3-8b/gpu-rocm/main.c` | ROCm 推論（集約 Makefile の HIP ビルド対象）。**Prefill progress bar** と prefill / decode / total スループット要約を stderr に出力。**`make log.push`** 用に stdout へ **`prefill_tps:` / `decode_tps:` / `total_tps:`**（推論区間のみ）。 |
| `qwen3-8b/gpu-rocm/Makefile` | **`qwen3-rocm`** を **`hipcc`** でビルド。**`GPU_ARCH`** は **`$(ROCM)/bin/rocminfo`** の最初の **`gfx*`** を自動検出（**`detect-gpu-arch`**）。未検出時はエラー。上書きは **`make GPU_ARCH=…`**。**`make log`** / **`make log.push`** でベンチマーク履歴（**`BENCH_LOG += …`** を Makefile 内に追記・表示）。 |
| `qwen3-8b/gpu-cuda/main.c` | NVIDIA CUDA 推論ホスト（FP16 線形層）。**`kernels.cu`** がデバイス forward。**`gpu.h`** が C/CUDA 境界。 |
| `qwen3-8b/gpu-cuda/Makefile` | **`nvcc`** で **`qwen3-gpu-cuda`** をビルド。既定 **`make build` / `make run`**（FP16・PTX 既定）。**`build.polarquant`** / **`run.polarquant`**（**`BONSAI_POLARQUANT=1`**）、**`pq-test`**。**`KERNELS_OBJ`** / **`MAIN_OBJ`** は **`BONSAI_POLARQUANT`/`FA_BR` 別名で stale `.o` 回避。NVFP4 関連は含まない。 |
| `qwen3-8b/gpu-cuda/kernels.cu` | FP16 GEMV・Flash Attention（decode / prefill）・RoPE 等。PolarQuant 時は KV を **`PQBlock`** 経路に切替。 |
| `qwen3-8b/gpu-cuda/polarquant.cu` / `polarquant.h` | **PolarQuant-R** KV キャッシュ圧縮。ホスト側コードブック初期化（Lloyd-Max L=2〜4）、デバイス **`PQState`**、KV 書き込み API。 |
| `qwen3-8b/gpu-cuda/polarquant_kernels.cuh` | デバイス側 encode/decode（ランダム符号 + FWHT-128、L=4 再帰 polar 量子化、**`PQBlock`** 8 bytes × 8 blocks）。 |
| `qwen3-8b/gpu-cuda/polarquant_verify.cu` | **`make pq-test`** 用のラウンドトリップ検証（推論バイナリには未リンク）。 |
| `qwen3-8b/gpu-cuda-nvfp4/Makefile` | **`nvcc`** で **`qwen3-gpu-cuda-nvfp4`** をビルド。既定 **`make build` / `make run`**（**`sm_120a`** + **`BONSAI_FP4=1`** + **`FA_BR=32`**）。**`build.polarquant`** / **`run.polarquant`**、**`blackwell`**（CUDA 13 導入 → **`build`**）、**`cutlass`**、**`fp4-test`**、**`pq-test`**。**`main.c` / `kernels.cu` / `gpu.h` / `polarquant.*`** は **`../gpu-cuda/`** を参照。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_gemm.cu` / `fp4_gemm.h` | CUTLASS **NVFP4** GEMM（M≥128）とデバイス側 **FP4 GEMV**（**`fp4_gemv_cached`**、バッチ版）。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_qwen3.cu` / `fp4_qwen3.h` | ホスト FP16 → NVFP4 キャッシュ（**`fp4_qwen3_weight_from_f16_host`**）、F32 活性の **`fp4_qwen3_mm`**（M に応じて GEMV / GEMM）。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_verify.cu` | **`make fp4-test`** 用の CUTLASS GEMM 単体検証（推論バイナリには未リンク）。 |
| `qwen3-8b/gpu-cuda-nvfp4/third_party/cutlass/` | **`make cutlass` / `make blackwell`** で clone される CUTLASS **v4.5.0**（**`CUTLASS_TAG`**）。リポジトリ同梱ではない。タグ不一致時は **`make cutlass`** が再 clone する。 |
| `qwen3-8b/xdna2/main.c` | AMD Ryzen AI（XDNA2）NPU。**mmap ウェイト + GEMV 毎 BF16 スクラッチ**・`amdxdna` ioctl 直叩き。**`--xdna-status` / `-X`** で制御コード環境の軽量診断。 |
| `qwen3-8b/xdna2-bfp16/main.c` | **`xdna2/main.c` と同一の IOCTL／チャンク BF16 GEMV（枠組み）。密行列レイアウトの重みをロード時に BFPX 化しホストのみ保持、mmap は変換完了後に解放。** |
| `qwen3-8b/Makefile` | **`model`**（**`gguf.txt`** の URL を **`wget`** で取得し **`$(MODEL).sha256sum`** で検証。失敗時は破損ファイルを削除）、**`build.<サブディレクトリ名>` / `run.<サブディレクトリ名>`**（例: **`build.cpu`**・**`build.gpu-rocm`**）、**`clean` / `gen-xdna-kernels`** を **`cpu/`** ほか各サブディレクトリの **`Makefile`** に委譲。 |
| `qwen3-8b/cpu/Makefile` ほか（各経路直下） | 当該サブディレクトリのみの **`make build`** / **`make run`** / **`clean`**（単体開発用）。出力バイナリは **`cpu/qwen3-cpu`** のように経路直下に生成。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog.md` | 変更履歴。 |
| `qwen3-8b/xdna2/xdna-gemv/README.md` | **NPU GEMV 用 ctrlcode まわりの入口**。**`kernels/`**・**`toolchain/`**・スタブ再生成スクリプトへの導線。 |
| `qwen3-8b/xdna2/xdna-gemv/kernels/Makefile` | **`bf16-gemv-*.bin` を `curl`/`wget` で一括取得**。既定の **`XDNA_GEMV_BIN_URL_BASE`** を Makefile 内に記述（上書き可）。詳細は **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md` §8.1**。 |
| `qwen3-8b/xdna2/xdna-gemv/kernels/README.md` | **ctrlcode**（**`bf16-gemv-<n>x<d>.bin`**）と GEMV 形状、ホスト／NPU 分担、用語ミニ辞典、スタブ **`GQF3XDNA`** と **`--xdna-status`**、8B 対応表、差し替え・再生成。**§3 で ROCm/HIP の GPU カーネルとの対比と「ユーザーが HIP のように書けるか」** の整理を含む入門。 |
| `qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py` | **`qwen3-8b/xdna2/xdna-gemv/kernels/`** のスタブ `.bin` を生成（**`qwen3-8b`** の **`make gen-xdna-kernels`** がリポジトリルートから実行）。 |
| `qwen3-8b/xdna2/xdna-gemv/toolchain/README.md` | **NPU 用 `bf16-gemv-*.bin` を自前生成する手引き**（Xilinx **mlir-aie**／**AMD IRON**／**Peano**／**`aiecc`** の公式手順に沿ったコマンド、Qwen-VL-8B と **IRON `GEMV(M,K)`** の対応、`--aie-generate-npu-insts` と `qwen3-xdna2` 統合時の注意）。**冒頭**で Linux カーネル文書 **AMD NPU** における **`ctrlcode`** と本リポジトリ実装の対応、**`qwen3-xdna2`（XRT 非依存）と公式サンプル（XRT 経由）の違い**を整理。**東京科学大学（2026年現在の名称。旧・東京工業大学）ACRi** ルーム公開の日本語チュートリアル（外部リンク・vadd 題材）への導線あり。本文は日本語（です・ます調）。 |
| `.gitignore` | ビルド生成バイナリ・**`*.gguf`** 等に加え、**Python の `__pycache__/` と `*.py[cod]`** を除外。 |
| `qwen3-8b/gguf.txt` | 既定 GGUF の取得元 URL 参照。Hugging Face の `blob/main` URL を `resolve/main` に置換して **`wget`** する（**`make model`** が同処理を実行）。 |
| `qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum` | 既定 GGUF の SHA256 参照（**`make model`** および手動 **`sha256sum -c`** 用）。 |

### 生成バイナリと Make ターゲット（`qwen3-8b/`）

作業ディレクトリは **`qwen3-8b/`**（集約 **`Makefile`** が各サブディレクトリの **`Makefile`** を呼び出す）。既定 `MODEL=Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`。ROCm 版の **`GPU_ARCH`** は **`gpu-rocm/Makefile`** が **`rocminfo`** から自動検出（集約 Makefile は **`GPU_ARCH` を渡したときのみ子へ転送）。GGUF 未取得時は先に **`make model`**（詳細は **「モデル参照」**）。

| Makefile ターゲット | 出力バイナリ | ソース |
|---------------------|--------------|--------|
| **`model`** | **`$(MODEL)`**（既定 **`Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**） | **`gguf.txt`** の URL → **`wget`**。検証は同ディレクトリの **`$(MODEL).sha256sum`**（**`sha256sum --check`**）。チェックサムファイル欠如・検証失敗時はエラー終了 |
| `build.cpu` / `run.cpu` | `cpu/qwen3-cpu` | `cpu/main.c` |
| `build.cpu-multicore` / `run.cpu-multicore` | `cpu-multicore/qwen3-cpu-omp` | `cpu-multicore/main.c`（`-fopenmp`、`OMP_NUM_THREADS`） |
| `build.cpu-blas` / `run.cpu-blas` | `cpu-blas/qwen3-cpu-blas` | `cpu-blas/main.c`（`-fopenmp`、**`-march=native`**、**`-lopenblas`**、`OMP_NUM_THREADS`。OpenBLAS は実行時 1 スレッド固定） |
| `build.gpu-rocm` / `run.gpu-rocm` | `gpu-rocm/qwen3-rocm` | `gpu-rocm/main.c` |
| （集約 Makefile なし）**`gpu-cuda` の `build` / `run`** | `gpu-cuda/qwen3-gpu-cuda` | FP16 のみ（PTX 既定可）。**`build.polarquant`** で PolarQuant-R KV |
| （同上）**`gpu-cuda-nvfp4` の `build` / `run`** | `gpu-cuda-nvfp4/qwen3-gpu-cuda-nvfp4` | **`fp4_gemm.sm120a.o`** + **`fp4_qwen3.sm120a.o`** + **`kernels.fabr32.pq0.o`** 等（**`sm_120a`**、**`BONSAI_FP4=1`** 固定） |
| （同上）**`gpu-cuda-nvfp4` の `blackwell`** | 同上 | apt CUDA 11 除去 → CUDA 13 → **`cutlass`** → **`build`** |
| （同上）**`gpu-cuda-nvfp4` の `fp4-test`** | `fp4_verify`（一時） | **`fp4_verify.cu`** + **`fp4_gemm.sm120a.o`**（**`BLACKWELL_NVCCFLAGS`** 固定） |
| （同上）**`gpu-cuda-nvfp4` の `build.polarquant` / `run.polarquant`** | 同上 | **`fp4_gemm.sm120a.o`** + **`fp4_qwen3.sm120a.o`** + **`polarquant.o`** + **`kernels.fabr32.pq1.o`** |
| （同上）**`pq-test`**（各 CUDA ディレクトリ） | `polarquant_verify`（一時） | **`polarquant_verify.cu`** + **`polarquant.o`** |
| `build.xdna2` / `run.xdna2` | `xdna2/qwen3-xdna2` | `xdna2/main.c`（`-fopenmp`。`amdxdna` カーネルモジュールが `/dev/accel/accelN` を提供） |
| **`gen-xdna-kernels`** | （出力なし） | `qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py` で **`qwen3-8b/xdna2/xdna-gemv/kernels/bf16-gemv-*.bin`** プレースホルダを再生成 |
| **`build.xdna2-bfp16` / `run.xdna2-bfp16`** | **`xdna2-bfp16/qwen3-xdna2-bfpx`** | **`xdna2-bfp16/main.c`**（`-fopenmp`。NPU 経路・環境変数は `qwen3-xdna2` と同種。ホスト重みは BFPX） |

```bash
cd qwen3-8b
make model                   # 既定 GGUF を gguf.txt から取得し .sha256sum で検証
make build.cpu
make build.cpu-multicore
make build.cpu-blas               # libopenblas-dev 等が必要
make build.gpu-rocm               # hipcc・ROCm 必須（GPU_ARCH は gpu-rocm で rocminfo 自動検出）
cd gpu-cuda && make build              # 汎用 NVIDIA GPU（FP16・PTX 可）
cd gpu-cuda && make build.polarquant    # PolarQuant-R KV キャッシュ（FP16 線形重み・任意 GPU）
cd gpu-cuda-nvfp4 && make build         # Blackwell + NVFP4（要 CUDA 13 + CUTLASS）
cd gpu-cuda-nvfp4 && make build.polarquant   # NVFP4 線形 + PolarQuant-R KV（Blackwell）
cd gpu-cuda-nvfp4 && make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
make build.xdna2             # Linux >= 6.10 + amdxdna カーネルモジュール（XRT 不要）
make gen-xdna-kernels        # xdna2/xdna-gemv/kernels に bf16-gemv-* プレースホルダ生成（実 NPU ctrlcode ではない）
make build.xdna2-bfp16       # 同上 + BFPX ホスト重み版バイナリ
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp "$(MODEL)" -p "Hello" -n 4
OMP_NUM_THREADS=8 ./cpu-blas/qwen3-cpu-blas "$(MODEL)" -p "Hello" -n 4
```

**CPU（IQ 混在 8B）**はブロック単位デ量子化のため **非常に遅くなり得る**。**`cpu-blas`** は F32 経路の OpenBLAS 化に加え、量子化 GEMV を **Q8_K + 整数内積**に置き換えるため **`cpu-multicore` より速くなることが多い**が、実用スループットは **ROCm 版**（AMD GPU）または **`gpu-cuda` / `gpu-cuda-nvfp4`**（NVIDIA GPU）を優先する想定である。

## ビルドと実行

### 共通（`qwen3-8b/Makefile`）

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `CC` | C コンパイラ（CPU / OpenMP / XDNA） | `cc` |
| `CFLAGS` | C コンパイルフラグ | `-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter` |
| `LDFLAGS` | リンクフラグ・ライブラリ | `-lm` |
| `ROCM` | ROCm ルート | `/opt/rocm` |
| `HIPCC` | HIP コンパイラ | `$(ROCM)/bin/hipcc` |
| `GPU_ARCH` | **`hipcc --offload-arch=`**（**`gpu-rocm/Makefile`** で **`rocminfo` から自動検出**。集約 Makefile からは省略可。上書き例: **`gfx1100`**。**行末に `# …` と同書きしない** — 値末尾空白で HIP が失敗することがある） | （自動。未検出時はビルド失敗） |
| `XDNA_INCS` | `<drm/drm.h>` が標準外にあるときだけ付与する `-I…`（`build.xdna2` / `build.xdna2-bfp16`） | 未定義（空で可） |
| `MODEL` | GGUF パス | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |
| `PROMPT` | ユーザプロンプト文字列 | `Hello, how are you?` |

### CPU

```bash
cd qwen3-8b
make build.cpu
make run.cpu PROMPT="質問" MODEL=path/to/model.gguf
```

### CPU（OpenMP）

```bash
cd qwen3-8b
make build.cpu-multicore
OMP_NUM_THREADS=8 ./cpu-multicore/qwen3-cpu-omp path/to/model.gguf -p "Hi" -n 8
```

### CPU（OpenMP + OpenBLAS）

**OpenBLAS**（`libopenblas-dev` 等）と OpenMP ランタイムが必要。**`pkg-config openblas`** が使える環境では **`cpu-blas/Makefile`** が include / link フラグを自動取得する。ヘッダが標準パスに無い場合（Debian/Ubuntu の pthread ビルド等）は **`CPPFLAGS`** で指定する。

```bash
cd qwen3-8b/cpu-blas && make openblas   # libopenblas-dev / libgomp1 を apt 導入
cd qwen3-8b
make build.cpu-blas
OMP_NUM_THREADS=8 ./cpu-blas/qwen3-cpu-blas path/to/model.gguf -p "Hi" -n 8
# ヘッダパスが必要な例:
cd cpu-blas && make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
```

**`openblas_set_num_threads(1)`** で OpenBLAS 側は 1 スレッド固定（OpenMP との二重並列化を避ける）。**`-ffast-math`** は IQ2_S / IQ3_S 量子化内積で数値が崩れるため **`cpu-blas/Makefile`** では無効。

### ROCm

**`GPU_ARCH`** はビルド前に **`gpu-rocm/Makefile`** が **`$(ROCM)/bin/rocminfo`** から自動検出する（**`make -C gpu-rocm detect-gpu-arch`** で確認）。検出に失敗する環境や別 ISA 向けビルドでは **`GPU_ARCH=gfx1100`** 等を明示する。

```bash
cd qwen3-8b
make build.gpu-rocm
make run.gpu-rocm PROMPT="Hello"
# 手動上書きの例:
make build.gpu-rocm GPU_ARCH=gfx1100
make -C gpu-rocm detect-gpu-arch
# ベンチマーク履歴（gpu-rocm/ 単体 Makefile）:
cd gpu-rocm
make log.push              # 既定 BENCH_PROMPT (~128 tok) / BENCH_N=128 / -t 0
make log                   # Makefile 内 BENCH_LOG を表表示
make log.push BENCH_N=64   # 生成トークン数など上書き可
```

| 変数（`gpu-rocm/Makefile`） | 意味 | 既定例 |
|-----------------------------|------|--------|
| `BENCH_PROMPT` | **`log.push`** のプロンプト文字列（~128 token 想定） | 英語長文（Makefile 内） |
| `BENCH_N` | **`log.push`** の **`-n`**（生成トークン上限） | `128` |
| `BENCH_SEED` | **`log.push`** の **`-s`** | `42` |

**`log.push`** の 1 行形式（パイ区切り）: **`ISO8601|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**。スループットは推論のみ（prefill+decode）。モデル重み H2D は含まない。

### CUDA（NVIDIA GPU）

集約 **`qwen3-8b/Makefile`** には **`build.gpu-cuda`** が無い。**`qwen3-8b/gpu-cuda/`**（FP16）または **`qwen3-8b/gpu-cuda-nvfp4/`**（NVFP4）で単体ビルドする。

```bash
cd qwen3-8b/gpu-cuda
# 汎用 GPU（Ampere/Ada 等）— FP16 のみ
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# 実 GPU アーキテクチャを直接指定する例:
make build CUDA_GENCODE=arch=compute_89,code=sm_89

# PolarQuant-R KV キャッシュ（線形 FP16 のまま・任意 GPU）
make build.polarquant
make pq-test

cd ../gpu-cuda-nvfp4
# Blackwell（RTX 50 系等）— CUDA 13 + CUTLASS + NVFP4
make cutlass                            # 初回: third_party/cutlass を clone
make build                              # sm_120a, BONSAI_FP4=1, FA_BR=32
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# 環境構築から一式（apt CUDA 11 除去 → CUDA 13、要 root 相当）:
make blackwell

# Blackwell — NVFP4 線形 + PolarQuant-R KV（最大 VRAM 節約）
make build.polarquant
make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `CUDA_HOME` | CUDA Toolkit ルート | `/usr/local/cuda` |
| `NVCC` | CUDA コンパイラ | `$(CUDA_HOME)/bin/nvcc` |
| `CUDA_GENCODE` | `-gencode` 引数（**`gpu-cuda`** の **`kernels.cu`**、および **`gpu-cuda-nvfp4`** の PolarQuant 無し時） | **`gpu-cuda`**: `arch=compute_86,code=compute_86`。**`gpu-cuda-nvfp4`**: `arch=compute_120a,code=sm_120a` |
| `BLACKWELL_GENCODE` | **`gpu-cuda-nvfp4`** の **`fp4_*`** オブジェクト用 `-gencode` | `arch=compute_120a,code=sm_120a` |
| `CUTLASS_TAG` | **`third_party/cutlass`** の clone タグ（**`gpu-cuda-nvfp4/Makefile`**） | **`v4.5.0`** |
| `BLACKWELL_NVCCFLAGS` | **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** / **`fp4-test`** 用 | **`-std=c++17`**（CUTLASS 必須）+ **`BLACKWELL_GENCODE`** 固定（NVFP4 MMA 必須）。**`-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored`**（CUTLASS v4.5.0 の **`sm100_static_tile_scheduler.hpp`** 由来 nvcc #20012 用。CUTLASS 公式ビルドと同オプション。詳細は **`gpu-cuda-nvfp4/Makefile`** コメント参照） |
| `FP4_GEMM_OBJ` / `FP4_QWEN3_OBJ` | NVFP4 オブジェクト名（**`gpu-cuda-nvfp4`** のみ） | **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** |
| `SRC_DIR` | 共有ソース参照（**`gpu-cuda-nvfp4/Makefile`**） | `../gpu-cuda` |
| `MODEL` | GGUF パス（各 CUDA ディレクトリからの相対） | `../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |
| `BONSAI_FP4` | NVFP4 線形層（**`fp4_qwen3`**） | **`gpu-cuda-nvfp4`** で常に `1`（**`-DBONSAI_FP4=1`** 固定）。**`gpu-cuda`** では未使用 |
| `BONSAI_POLARQUANT` | PolarQuant-R KV キャッシュ | `0`（**`build.polarquant`** で `1`） |
| `FA_BR` | Flash Attention の K/V タイル幅 | **`gpu-cuda`**: 未指定時 64。**`gpu-cuda-nvfp4`**: 未指定時 32 |
| `KERNELS_OBJ` | **`kernels.cu` の出力名** | **`gpu-cuda`**: `kernels.fabr$(FA_BR).pq$(BONSAI_POLARQUANT).o`。**`gpu-cuda-nvfp4`**: 同上 |
| `MAIN_OBJ` | **`main.c` の出力名** | `main.pq$(BONSAI_POLARQUANT).o` |

**`gpu-cuda`** と **`gpu-cuda-nvfp4`** のいずれも **`.DEFAULT_GOAL` は `run` → `build`**。**`nvcc` / `nvlink` は `PATH` に CUDA の `bin` を通す**。apt **`nvidia-cuda-toolkit`（CUDA 11）** と **CUDA 13** は併存させない（**`gpu-cuda-nvfp4` の `make blackwell`** が 11.x を除去して 13 を入れる）。

### XDNA2（AMD Ryzen AI NPU）

```bash
cd qwen3-8b
make build.xdna2
# ユーザを render グループに追加して /dev/accel/accel0 を開けるようにしておく
sudo usermod -aG render "$USER"
# 既定では NPU 行列乗算用の制御コードバイナリは XDNA_GEMV_DIR から検索する。
# リポジトリ同梱は xdna2/xdna-gemv/kernels のスタブ（実機では使われない）— MLIR-AIE 生成物に差し替える。
# 未設定/未配置の場合は OpenMP CPU フォールバックに自動切り替え。
XDNA_GEMV_DIR=xdna2/xdna-gemv/kernels ./xdna2/qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8
# 強制的に NPU を使わず CPU OpenMP で実行する場合:
XDNA_FORCE_CPU=1 ./xdna2/qwen3-xdna2 path/to/model.gguf -p "Hi" -n 8
# 上級者向け: ハードウェアコンテキストの試行値を直接上書き
#   XDNA_NUM_COL=<n>     CREATE_HWCTX で要求する列数の上限
#   XDNA_NUM_TILES=<n>   num_tiles を直接固定（core.row_count 整数倍が必要）
#   XDNA_HEAP_SIZE=<bytes>  DEV_HEAP のサイズ（既定 64 MiB、ファーム上限による）
XDNA_NUM_COL=1 XDNA_HEAP_SIZE=33554432 ./xdna2/qwen3-xdna2 path/to/model.gguf --xdna-status
# NPU / XDNA_GEMV_DIR / 各形状の bf16-gemv-<n>x<d>.bin の可否だけ確認し終了（重みロード・推論なし）
./xdna2/qwen3-xdna2 path/to/model.gguf --xdna-status
# 同上の短い別名
./xdna2/qwen3-xdna2 path/to/model.gguf -X
# BFPX ホスト重み版（ビルド後バイナリは qwen3-xdna2-bfpx）
make build.xdna2-bfp16
./xdna2-bfp16/qwen3-xdna2-bfpx path/to/model.gguf -p "Hi" -n 8
```

**詳細（入門）**: **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md`** に **ctrlcode の定義**（ERT・Instruction Buffer・`ERT_START_NPU`）、**GEMV と** **`(n,d)`**、**GPU（ROCm/HIP）カーネルとの対比**（オーバーレイ＋ctrlcode、標準ワークフローでのユーザー主導性の違い）、公開ミラーがある場合の **`qwen3-8b/xdna2/xdna-gemv/kernels/Makefile` での一括ダウンロード（README §8.1）**、**MLIR-AIE / IRON での自前ビルド手順（`qwen3-8b/xdna2/xdna-gemv/toolchain/README.md`）**、各 **`bf16-gemv-<n>x<d>.bin`** と Qwen3-VL-8B の対応、スタブ **`GQF3XDNA`**、**`--xdna-status`**、プレースホルダ再生成と MLIR-AIE / IRON 生成物への差し替えを記す。

## 実行時の挙動

**CPU（`qwen3-cpu` / `qwen3-cpu-omp` / `qwen3-cpu-blas`）**: 重みは mmap 上の GGUF を参照。KV・活性は主に float32。サンプリングはホスト上の logits に対して実施。**`qwen3-cpu`** / **`qwen3-cpu-omp`** は量子化行を都度ブロックデ量子化してから内積。**`qwen3-cpu-blas`** は F32 行列積と Attention を **`cblas_sgemv`** に集約。量子化 GEMV は **Q8_K + 全型 AVX2 整数内積**（詳細は **「量子化と行列積」** 参照）。**prefill** 中（最終プロンプト token 以外）は **LM head をスキップ**。**`-t 0`（greedy）** では **全 vocab logits を確保せず `mm_argmax_row`**。**RoPE** は起動時 **cos/sin キャッシュ**参照。プロンプト区間は **1 トークンずつ teacher forcing**。**Prefill progress bar** と tok/s 要約を stderr に出力。

**ROCm（`qwen3-rocm`）**: ロード時に F16 重みを VRAM に配置。各ステップは **埋め込み〜全レイヤー〜LM ヘッド**を GPU 上で実行。教師強制区間では LM ヘッドを省略可能。**`0 < top-p < 1`** の nucleus は実装上 **logits 全語彙を D2H** して CPU で処理する場合がある（実装コメント参照）。それ以外は GPU で argmax / softmax＋多項サンプル等。プロンプト区間は **1 トークンずつ forward** し stderr に **Prefill progress bar** と prefill / decode / total の **スループット要約**を出力（**`cpu-blas`** 同形式）。終了時 stdout には **`--- N prompt tokens + M generated tokens ---`** に加え、**`make log.push`** 用の **`prefill_tps:` / `decode_tps:` / `total_tps:`**（**推論区間のみ**。重み H2D は計測外）を出す。

**CUDA（`qwen3-gpu-cuda` / `qwen3-gpu-cuda-nvfp4`）**: プロンプトは **`gpu_forward_prefill`**、生成は **`gpu_forward`**（1 トークン）。Attention・RoPE・残差は **`kernels.cu`**（**`../gpu-cuda/kernels.cu`** を **`gpu-cuda-nvfp4`** が参照）。**サンプリングはホスト**（logits D2H）。prefill/decode のスループット要約を stderr に出力。

- **`gpu-cuda`（FP16）**: ROCm 版と同様—CPU 逆量子化 → **全線形 FP16 VRAM** → **`mm_f16_gemv_kernel`**。
- **`gpu-cuda-nvfp4`（NVFP4）**: 線形（Q/K/V/O、gate/up/down、LM head）は **`upload_linear_fp4`** でホスト FP16 ステージング後 **NVFP4 キャッシュのみ**（線形の FP16 VRAM 複製なし）。**`token_embd`** は **FP16**、norm 系は **F32**。**`gpu_model_create`** が **`wq_fp4` 等**を採用し **`use_fp4=1`**。起動ログ例: **`dequant -> NVFP4 linear layers`**、**`FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`**。
- **`build.polarquant`**（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**）: KV キャッシュは F32 の代わりに **`PQBlock`** 配列（**64 B/head**）。K/V 書き込みは **`polarquant_kv_write_one`** / **`polarquant_kv_write_batch`** でエンコード。Attention は **`flash_attn_gqa_pq_kernel`** / **`flash_attn_prefill_gqa_pq_kernel`** がタイル単位で **`pq_decode_head`** により F32 復号。起動ログ例: **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**。**`head_dim=128` 固定**（Qwen3-VL-8B 向け）。
- **`gpu-cuda-nvfp4` + `build.polarquant`**: NVFP4 線形経路と PolarQuant-R KV 経路を同時有効化。起動ログに **NVFP4** と **PolarQuant-R** の両方が出る。
- **NVFP4 線形の実行**（**`gpu-cuda-nvfp4`** のみ）: **`fp4_qwen3_mm`** — **M&lt;128**（デコードおよび短い Prefill）は **`fp4_gemv_cached`** / **`fp4_gemv_batch_cached`**（**`fp4_gemm.cu`**。NVFP4 重みを直接参照し **M=128 パディングなし**）；**M≥128** は CUTLASS **`fp4_gemm_run_cached`**（長い Prefill バッチ）。

**XDNA2（`qwen3-xdna2`）**: 線形ウェイトは **mmap された GGUF** を **`main-omp.c` と同様**に参照する（埋め込みは mmap 上行の量子化レイアウトからブロック単位復号）。各 **GEMV** のたび、その行列だけを **`AMDXDNA_BO_SHMEM` に確保した単一 BF16 スクラッチ**へ CPU で復号・BF16 化し、`SYNC_BO` でデバイス可視にしたうえで、入力 BF16・重み・出力への `xdna_addr` を **`ERT_START_NPU`** で `DRM_IOCTL_AMDXDNA_EXEC_CMD` に渡す構成は従来どおり。**レイヤー分の恒久 BF16 重み BO は保持しない**。RMSNorm／Qwen3 ヘッド RMSNorm／Attention 等も **CPU**。NPU が使えないときは BF16 GEMV が **OpenMP** にフォールバックする（実装どおり bit-identical）。

推論開始前に **`=== XDNA GEMV / NPU ctrlcode status ===`** ブロックを標準出力へ出し、`XDNA_FORCE_CPU`・DRM オープン可否・`XDNA_GEMV_DIR`・テキスト経路で使う **6 種類の GEMV 形状**それぞれについて `bf16-gemv-<n>x<d>.bin` を **`[ OK ]`（実 ctrlcode）／`[STUB]`（リポジトリ同梱プレースホルダ・マジック `GQF3XDNA`）／`MISS`** で表示する（実機でロードされるのは非 STUB のみ）。推論後は **NPU GEMV 回数／CPU GEMV 回数**に加え、**すべて NPU／すべて CPU／混在**を短文で表示する（実際に `EXEC_CMD` が成功したかはランタイムカウントが基準）。**`--xdna-status`** または **`-X`** は GGUF パースと `npu_open` のみ行い当該レポートを出力して **終了**する（重みロード・生成ループなし）。**`qwen3-8b/xdna2/xdna-gemv/kernels/`** の `.bin` プレースホルダの再生成はリポジトリルートで `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels`、または `qwen3-8b` で `make gen-xdna-kernels`。

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

## アーキテクチャ

### 実装のレイヤー構成

各実装ファイルは、外部ライブラリに分割せず、ほぼ同じ順序で機能を持つ。

1. **GGUF と量子化形式の定義**: GGUF の値型、GGML tensor dtype、`QK_K=256` の K-quant / IQ ブロック構造を定義する。`BlockQ4_K`、`BlockQ5_K`、`BlockIQ2_S`、`BlockIQ3_S` は GGML の packed layout に合わせて `#pragma pack(push, 1)` で定義する。
2. **IQ2_S / IQ3_S の復元テーブル**: `iq2s_grid`、`iq3s_grid` などを持ち、GGML 側の小さな格子表現を `float` に戻す。
3. **モデル構造体**: `Config` がモデル形状、`TensorInfo` が GGUF 内 tensor descriptor、`Tok` が tokenizer、`Weights` / `WeightsDev` が重み、`State` が実行時バッファ、`Model` がそれらをまとめる。
4. **ロード処理**: `mmap` した GGUF からメタデータと tensor descriptor を読み、CPU 版は tensor へのポインタを保持し、ROCm / CUDA 版は重みを GPU にアップロードする。
5. **推論処理**: 1 トークン単位の forward を繰り返し、プロンプト区間は teacher forcing、生成区間は logits から次トークンを選ぶ（CUDA 版はプロンプトを **Prefill バッチ**で先に処理）。

この構成により、CPU 版は「GGUF の量子化重みをその場で読む参照実装」、ROCm / CUDA 版は「同じモデル構造を GPU 常駐重みに変換して動かす実装」として対応づけられる。

### 主要データ構造

`Config` は `dim`、`hidden_dim`、`n_layers`、`n_heads`、`n_kv_heads`、`vocab_size`、`max_seq`、`rope_theta`、`norm_eps` を保持する。Qwen3-VL では `head_dim` を `qwen3vl.attention.key_length` から読む。値が無い場合のみ `dim / n_heads` にフォールバックし、`kv_dim = n_kv_heads * head_dim`、`kv_mul = n_heads / n_kv_heads` を派生させる。

`TensorInfo` は GGUF tensor の `name`、次元数、各次元長 `ne[4]`、dtype、data section 内 offset を持つ。実データ位置は `fdata + doff + offset` で求める。`doff` は `general.alignment`（既定 32）に基づいて tensor data section の開始位置へ丸めた値である。

`Tok` は語彙文字列、語彙長、BPE score、特殊トークン ID、ハッシュ表、byte fallback 用 token を持つ。`<|im_start|>` と `<|im_end|>` は ChatML 用に語彙から探索し、見つかった場合は `im_start` / `im_end` として保存する。

`State` は forward 中の一時バッファを持つ。主なものは hidden state `x`、RMSNorm 後や射影後に使う `xb` / `xb2`、FFN の `hb` / `hb2`、attention の `q` / `k` / `v`、logits、KV cache である。CPU / ROCm / CUDA **`gpu-cuda`（FP16）** では **`kc` / `vc`**（float32、**`n_layers * max_seq * kv_dim`** 要素 × Key/Value）。**`cpu-blas`** では量子化 GEMV 用 **`q8`**（**`hidden_dim / QK_K`** ブロック）に加え、greedy 用 **`State.argmax_tok`**、**`Model.rope_cr` / `Model.rope_ci`**（**`max_seq × head_dim/2`** ずつ、RoPE キャッシュ）を保持する。層内 Q8 共有・全型 AVX2 dot・LM モード等の詳細は **「量子化と行列積」→「`cpu-blas`：Q8_K 活性化 GEMV」** を参照。CUDA **`build.polarquant`**（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**）では **`kc_pq` / `vc_pq`**（**`PQBlock`** 配列、**`n_layers * max_seq * n_kv_heads * PQ_BYTES_HEAD`**）に置き換える。

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

CPU 版（**`cpu`** / **`cpu-multicore`**）は重み全体を float に展開しない。`mm_quant_rows` が出力行ごとに量子化 row を走査し、row 内の各 256 要素 block を stack 上の `float blk[QK_K]` に復元して、入力ベクトルとの内積に足し込む。これによりメモリ使用量は抑えられるが、同じ重みを毎 token で復元するため速度は遅い。

**`cpu-blas`** は F32 テンソルに対する **`mm_f32`** を OpenMP 行帯分割 + **`cblas_sgemv`** に置き換え、Attention もヘッドごとに K/V 合成を BLAS 化する。量子化 GEMV は **Q8_K 活性化 + ggml 準拠整数内積**（詳細は次節 **「`cpu-blas`：Q8_K 活性化 GEMV」**）。weight row を float[256] に dequant しないため **`cpu-multicore`** より高速化しやすい。出力行の OpenMP 並列は維持する。

#### `cpu-blas`：Q8_K 活性化 GEMV（層内共有・AVX2）

**`cpu-blas/main.c`** の量子化 GEMV（IQ2_S / IQ3_S / Q4_K / Q5_K）は、llama.cpp / ggml と同様 **「活性 float ベクトルを Q8_K に量子化 → 重みスーパーブロックと整数内積」** で計算する。per-row **`float[256]` 全復号**は行わない。

##### 3 経路の比較（なぜ `cpu-blas` か）

| 経路 | 量子化 GEMV のやり方 | 主なボトルネック |
|------|---------------------|------------------|
| **`cpu` / `cpu-multicore`** | 重み行ごとに **256 要素 block を stack 上 `float blk[256]` に dequant** → float 内積 | 毎 GEMV・毎行で **dequant 再実行**（帯域 bound） |
| **`cpu-blas`（本節）** | 活性 **1 回 Q8_K 化**（層内共有で削減）→ **`vec_dot_*_q8_K` 整数内積** | IQ2_S dot はスカラー。quantize + OpenMP 行並列 |
| **`gpu-rocm` / `gpu-cuda`** | ロード時 **全線形 F16 VRAM 展開** → GPU GEMV | VRAM 使用量（8B 級で数 GiB 級） |

`cpu-blas` は **mmap 上の量子化 GGUF をそのまま参照**しつつ、F32 経路（norm 以外の F32 テンソル）と Attention を **OpenBLAS** に寄せ、量子化 GEMV だけを **ggml Q8_K 経路**に置き換える。**GPU ほどの速度は出ない**が、**`cpu-multicore` より dequant コストを大幅に削れる**中間解である。

##### 活性 `BlockQ8_K` と重みブロック

**活性 `BlockQ8_K`**（ggml **`block_q8_K`** 準拠、**`sizeof ≈ 272 B`**）:

| フィールド | 意味 |
|-----------|------|
| **`float d`** | スーパーブロック scale（**`d = 1/iscale`**。**`iscale = -127/maxv`**。**`maxv`** は 256 要素中の **符号付き**最大値） |
| **`int8_t qs[256]`** | QK_K=256 の量子化係数。**`qs[j] ≈ round(iscale · x[j])`**（**127 キャップ**） |
| **`int16_t bsums[16]`** | **`qs[k*16 .. k*16+15]`** の 16 要素和（Q4_K / Q5_K の **dmin × mins** 補正で使用） |

**量子化の数式**（ブロック **`b`**、入力 **`x[0..255]`**）:

```text
maxv  = argmax_{j} x[j]   （符号付き。abs 最大の要素そのもの）
iscale = -127 / maxv
qs[j]  = clamp(round(iscale * x[j]), ..., 127)
d      = 1 / iscale
bsums[k] = Σ_{j=0}^{15} qs[k*16 + j]
```

近似復元 **`x̂[j] ≈ d * qs[j]`**。ggml の Q8_K 活性化は **対称 int8**（**−127..127** だが実装は **0..127 側に clamp**）で、後段の **Q4_K / Q5_K dot** が **`d · Σ(scale·q_weight·q8)`** 形式に落ちる。

**重み側スーパーブロック**（GGML packed layout、`#pragma pack(1)`）:

| 型 | サイズ | 主なフィールド |
|----|--------|----------------|
| **`BlockQ4_K`** | 144 B | **`d`/`dmin` FP16**, **`scales[12]`**（6-bit packed）, **`qs[128]`**（4-bit nibble） |
| **`BlockQ5_K`** | 176 B | 上記 + **`qh[32]`**（第 5 bit） |
| **`BlockIQ2_S`** | 82 B | **`d` FP16**, grid lookup + signs + 4-bit scales |
| **`BlockIQ3_S`** | 110 B | IQ2 系 + **`signs[32]`**, 64 要素ペア scale |

**`row_bytes_quant(type, n)`** = **`(n / QK_K) * sizeof(Block*)`**。GEMV **`y = W·x`**（**`W`** は **`[d, n]`** 行 major、mmap 上）では出力 **`i`** の **`y[i] = dot(row_i, x)`** を **`vec_dot_row_q8_K(n, row_i, type, q8_blocks)`** で求める。

##### `State.q8` とバッファ寿命

- **確保**: **`calloc(hidden_dim / QK_K, sizeof(BlockQ8_K))`**。Qwen3-VL-8B（**`dim=4096`**, **`hidden_dim=14336`**）では **56 ブロック ≈ 15 KiB**。
- **使用パターン**:
  - **Attention / gate/up**（**`n=dim`**）: 先頭 **16 ブロック**のみ書き込み。
  - **down**（**`n=hidden_dim`**）: **56 ブロック全体**を上書き（SwiGLU 後 **`hb`** を quantize）。
  - **wo / output**: **`n=dim`** で都度上書き。
- **層跨ぎ・token 跨ぎの再利用なし**: 各 **`forward(token, pos)`** 内で内容は都度更新される。**KV キャッシュ（`kc`/`vc`）は float32 のまま**（Q8 化しない）。

##### API（旧 `mm_quant_rows` からの分離）

旧 **`mm_quant_rows(o, x, w, n, d, type, q8)`** は **quantize + dot を不可分**にしていたため、同一 **`x`** に対する quantize 重複を **`forward` 側で削れなかった**。

| 関数 | 役割 |
|------|------|
| **`mm(o, x, w, n, d, type, q8, q8_ready)`** | 型分岐。量子化型かつ **`q8_ready=0`** のとき **`quantize_row_q8_K(x, q8, n)`** → **`mm_quant_dot_rows`**。**`q8_ready=1`** なら quantize 省略 |
| **`mm_quant_dot_rows(..., const q8)`** | OpenMP で **`i=0..d-1`**: **`o[i] = vec_dot_row_q8_K(n, row_i, type, q8)`** |
| **`mm_argmax_row(x, w, n, d, type, q8)`** | greedy 用。**全 **`o[]`** 非確保**。OpenMP 行 argmax（量子化型 / F32 **`cblas_sdot`** / F16 スカラー） |
| **`vec_dot_row_q8_K`** | **`switch(type)`** → 各 **`vec_dot_*_q8_K`**（AVX2 または generic） |
| **`is_q8_mm_type(type)`** | Q4_K / Q5_K / IQ2_S / IQ3_S のみ真 |

**`n % QK_K != 0`** の量子化 GEMV は **`mm` 内で exit**（Qwen3-VL-8B の **`dim`/`hidden_dim` は 256 の倍数）。

##### 層内 Q8_K 量子化共有（`forward` 制御）

```text
rmsnorm → xb
q8_att = is_q8_mm_type(wq_t[l])
if (q8_att) quantize_row_q8_K(xb → q8, n=dim)    // 層内 1 回
mm(q/k/v, xb, ..., q8, q8_att)                   // 3 GEMV で Q8 読み取り共有
... OpenBLAS attention (cblas_sgemv × 2 × n_heads) ...
mm(xb2, xb, wo, ..., q8, 0)                      // attn 出力で xb の意味が変わる → 都度 quantize

rmsnorm → xb
q8_ffn = is_q8_mm_type(gate_t[l])
if (q8_ffn) quantize_row_q8_K(xb → q8, n=dim)
mm(gate/up, xb, ..., q8, q8_ffn)
SwiGLU: hb[i] = silu(hb[i]) * hb2[i]
mm(xb, hb, down, n=hidden, ..., q8, 0)           // 入力 hb・長さ hidden → 都度 quantize（56 ブロック）
...
rmsnorm → x
// lm_mode に応じて LM head（generate が決定）:
//   FWD_NO_LM     → スキップ（prefill 中 pos < n_prompt-1）
//   FWD_LM_FULL   → rmsnorm → mm(logits)
//   FWD_LM_ARGMAX → rmsnorm → mm_argmax_row → argmax_tok
```

| 呼び出し | 入力 **`x`** | **`n`** | 共有 | 判定キー / 備考 |
|---|---|---|---|---|
| **wq / wk / wv** | attn RMSNorm 後 **`xb`** | **`dim`** | ○ **1× quantize** | **`wq_t[l]`** のみ。wk/wv 型は見ない |
| **wo** | attn 出力 **`xb`** | **`dim`** | × | attention 計算で **`xb` が上書き**される |
| **gate / up** | ffn RMSNorm 後 **`xb`** | **`dim`** | ○ **1× quantize** | **`gate_t[l]`** のみ |
| **down** | SwiGLU 後 **`hb`** | **`hidden_dim`** | × | 活性ベクトルが **`xb` → `hb` に変更** |
| **output** | 最終 **`x`** | **`dim`** | × | LM head |

**quantize 回数（量子化 GEMV 全テンソル・理想ケース）**:

| | 旧 `mm_quant_rows` | 新（層内共有） |
|---|---|---|
| wq/wk/wv | 3 | **1** |
| wo | 1 | 1 |
| gate/up | 2 | **1** |
| down | 1 | 1 |
| **層計** | **7** | **4** |
| **28 層/token 削減** | — | **~112 回** |

**エッジケース**: **`wq` が F16・`wk` が Q4_K** 等では **`q8_att=0`**。**`mm(wq)`** は F16 経路、**`mm(wk/wv)`** は **それぞれ内部で quantize**（旧挙動）。IQ2_M GGUF では wq/wk/wv/gate/up が同型 IQ2_S のため通常は **`q8_att=q8_ffn=1`**。

**OpenMP 安全性**: **`quantize_row_q8_K`** は **単スレッド**（`forward` 本体も逐次）。**`mm_quant_dot_rows`** の parallel 領域は **quantize 完了後**に **`const q8`** を読むだけ → **data race なし**。**`openblas_set_num_threads(1)`** により OpenBLAS 内部並列と OpenMP の **二重並列化を回避**（並列度は **`OMP_NUM_THREADS`**）。

##### 整数内積：`vec_dot_*_q8_K`（型別）

**共通**: 入力長 **`n`** は **`nb = n/QK_K`** スーパーブロックに分割。**`q8`** と重み row のブロック **`i`** を対応させ **`sumf += ...`**。**1 出力要素 = 1 重み行 dot**。

**IQ2_S — `vec_dot_iq2_s_q8_K`**（**`__AVX2__`** / **`_generic`**）:

- **generic**: 32 要素サブブロックごと **`iq2s_grid`** 参照 + **`signs`** ビット分岐 + **`0.125f * sumf`**。
- **AVX2**（ggml 準拠）:
  - **`iq2s_grid`** を **`_mm256_set_epi64x`** で 4 組ロード（**`qs`/`qh`** index）。
  - **`signs`**: **`k_mask1/k_mask2` + `shuffle_epi8`** → **`cmpeq` + `xor/sub`** で Q8 符号反転。
  - **`maddubs_epi16(q2, q8s)` → `madd_epi16(scale,·)`** を **`sumi1/sumi2`** に累積 → **`fmadd(d, sumi, accumf)`**。
  - **`*out = 0.125f * hsum_float_8(accumf)`**。

**IQ3_S — `vec_dot_iq3_s_q8_K`**（**`__AVX2__`** / **`_generic`**）:

- **generic**: 64 要素ペア + **`iq3s_grid`** + **`signs`**、**`2*ls+1`** scale。
- **AVX2**: grid index を **`sllv_epi32` + `iq3s_grid[ix]` gather**（16 index）。signs は IQ2_S 同型。**`*out = hsum_float_8(accumf)`**。

**Q4_K — `vec_dot_q4_K_q8_K`**（AVX2 / generic）:

- **1 スーパーブロックの数学**:

```text
d    = y.d * f16(x.d)
dmin = -y.d * f16(x.dmin)
dot += d * Σ_j (scale_j * Σ_k q4_{j,k} * q8_k)  -  dmin * Σ_m (min_m * bsum_m)
```

- **`scales[12]`** は 6-bit packed。**`kmask1=0x3f3f3f3f`**, **`kmask2=0x0f0f0f0f`**, **`kmask3=0x03030303`** で **4×uint32** に展開（ggml **`ggml_vec_dot_q4_K_q8_K`** と同一ビット操作）。
- **generic**: **`aux8[256]`** に nibble を **全面展開**してから 8 要素 **`aux16=q8·a`** ループ。**スタック ~300 B+/呼び出し**、命令数多。
- **AVX2**: nibble **オンザフライ**（**`and 0xF` / `srli 4`**）。**64 要素 ×4 サブループ**で **`_mm256_maddubs_epi16(q4,q8)`** → **`_mm256_madd_epi16(scale,·)`** → **`sumi`（int32）**。**`aux8` 不要**。

**Q5_K — `vec_dot_q5_K_q8_K`**（AVX2 / generic）:

- Q4_K に **`qh[32]`**（第 5 bit）を加え **`q5 = (q5l & 0xF) + ((qh_bit) << 4)`** 相当を復元。
- **AVX2**: **`hmask`** を lane ごとに 1 bit shift しながら **`q5h`** を取り出し **`add_epi8`**。**64 要素を `q5_0`/`q5_1` × `q8_0`/`q8_1` の 2 組**で処理。
- **dmin 項**: Q4_K は **`acc_m`**（**`__m128`** fmadd）、Q5_K は **`summs`** スカラー（**`hadd_epi32` チェーン**）。

##### AVX2 — `quantize_row_q8_K`（参照: **`quantize_row_q8_K_ref`**）

**`#if defined(__AVX2__)`** でコンパイル時分岐。**`-march=native`** 既定。

**ブロックループ**（**`i = 0 .. nb-1`**）:

1. **abs-max 走査**: **`j += 8`** で **`__m256 load` → `andnot(-0.0f, ·)`**（絶対値）。**`maxv`（符号付き）** 更新は **スカラー** — ggml は **abs 最大位置の符号付き値**を scale 基準にするため、**`_mm256_max_ps` だけでは不十分**。
2. **ゼロブロック**: **`d=0`**, **`memset qs/bsums`**。
3. **量子化**: **`j += 32`**（4×**`__m256`**）— **`× iscale` → `_mm256_round_ps(NEAREST)` → `_mm256_cvtps_epi32` → `_mm256_min_epi32(127)` → `packs_epi32`×2 → `packs_epi16` → `_mm256_permutevar8x32_epi32(0,4,1,5,2,6,3,7)` → store**。
4. **`bsums`**: **16 要素ずつスカラー sum**（AVX 化なし。dot 側が参照）。
5. **`y[i].d = 1/iscale`**。

**ref との差**: ref は **`nearest_int`（`lrintf`）**、AVX2 は **`round_ps`**。いずれも **127 キャップ**。**非 AVX2 CPU** は **`#else`** で ref に委譲。

##### AVX2 共通ユーティリティ

| 関数 | 役割 |
|------|------|
| **`hsum_float_8(__m256)`** | 8 lane float の水平和（**`extractf128` → add → movehl → movehdup → addss**） |
| **`get_scale_shuffle_k4(i)`** | **256 B `k_shuffle[]`** から **`__m256i`** load。**Q4/Q5 の 6-bit scale** を **`_mm256_shuffle_epi8`** で 32 byte lane に複製 |
| **`MM256_SET_M128I(a,b)`** | 128-bit scale を 256-bit に複製（**`_mm256_insertf128`**） |

**SIMD 対象**（**`__AVX2__`** 時）: **`quantize_row_q8_K`**、**`vec_dot_iq2_s_q8_K`**、**`vec_dot_iq3_s_q8_K`**、**`vec_dot_q4_K_q8_K`**、**`vec_dot_q5_K_q8_K`**（いずれも **`_generic`/`_ref` フォールバック**）。**bsums 計算**・**signed-max 決定**はスカラー。**F16 埋め込み**は **`__F16C__`** 追加時に **`_mm256_cvtph_ps`**。

##### OpenBLAS との分担（同一 `forward` 内）

| 処理 | 実装 |
|------|------|
| **F32 重み GEMV** | **`mm_f32`**: OpenMP 行分割 + **`cblas_sgemv(NoTrans)`** |
| **F16 重み GEMV** | **`mm_f16`**: OpenMP 行ループ（host F16→F32 変換しながら内積） |
| **量子化 GEMV** | 本節の **Q8_K + `vec_dot_*`** |
| **Attention K 内積** | ヘッド **`h`**: **`cblas_sgemv`**（**`(pos+1) × head_dim`** × **`qh`**）— 旧 CPU 版の pos ループを **1 BLAS 呼び出し**に |
| **Attention V 合成** | **`softmax(att_h)`** 後 **`cblas_sgemv(Trans)`** で value 重み付き和 |
| **OpenBLAS スレッド** | **`openblas_set_num_threads(1)`** 固定 |

| **OpenBLAS スレッド** | **`openblas_set_num_threads(1)`** 固定 |
| **greedy LM head** | **`mm_argmax_row`**: OpenMP 行並列 max（**`logits[vocab]` 非確保**） |
| **F16 埋め込み** | **`emb_lookup`**: **F16C+AVX2** で 8 要素 **`cvtph_ps`**（単スレッド） |

##### RoPE cos/sin キャッシュ

- **`init_rope_cache(m)`**（**`main`** で **`load_weights` 後**）: **`pos = 0..max_seq-1`**, **`i = 0..head_dim/2-1`** で **`rope_cr[pos*hd2+i] = cos(pos·freq)`**, **`rope_ci[...] = sin(...)`**。**`freq = 1/rope_theta^(2i/head_dim)`**。
- **`apply_rope(vec, n_heads, head_dim, pos, rope_cr, rope_ci)`**: **`pcr/pci = cache + pos*hd2`** を参照。**`powf/cosf/sinf` を forward 中に呼ばない**。
- メモリ: **2 × max_seq × (head_dim/2) × sizeof(float)**。Qwen3-VL-8B 既定（**512×64×4×2 ≈ 256 KiB**）。

##### `forward` の LM head モード（`lm_mode`）

| 値 | 意味 | LM head 処理 |
|----|------|-------------|
| **`FWD_NO_LM`** | prefill 中（最終プロンプト token 以外） | **`output_norm` + LM head スキップ** |
| **`FWD_LM_FULL`** | サンプリング（**`temp > 0`** または top-p） | **`rmsnorm` → `mm(logits, ...)`** → **`sample_token(logits)`** |
| **`FWD_LM_ARGMAX`** | greedy（**`temp <= 0`**） | **`rmsnorm` → `mm_argmax_row` → `State.argmax_tok`** |

**`generate`** の選択ロジック:

```text
lm_mode = FWD_NO_LM
if (pos >= n_prompt - 1)
    lm_mode = (temp <= 0) ? FWD_LM_ARGMAX : FWD_LM_FULL
forward(m, token, pos, lm_mode)
...
next = (lm_mode == FWD_LM_ARGMAX) ? s->argmax_tok : sample_token(s->logits, ...)
```

- **prefill スキップ効果**: プロンプト長 **`N`** なら **~(N-1) ×（output_norm + vocab 行 GEMV）** を省略（最後の prefill token のみ LM 実行）。
- **greedy argmax 効果**: **`s->logits[vocab_size]`**（~600k float ≈ 2.4 MiB）への書き込みと **softmax 不要**。量子化 LM head では **1 回 Q8 quantize + OpenMP 行 argmax**。

##### `mm_argmax_row`

- **入力**: RMSNorm 後 **`x[dim]`**、重み **`output.weight`**（**`[vocab, dim]`** 行 major）。
- **量子化型**: **`quantize_row_q8_K(x, q8, dim)`** → OpenMP **`for i in 0..vocab-1`**: **`vec_dot_row_q8_K`** → thread-local **`(lb, lv)`** → **`critical`** で global max。
- **F32 型**: 行ごと **`cblas_sdot(n, row, 1, x, 1)`**。
- **F16 型**: 行ごとスカラー F16 内積。
- **戻り値**: argmax token id。**`State.argmax_tok`** に格納。

##### IQ2_M モデルでの gain の内訳

Qwen3-VL-8B **IQ2_M** では **大部分の線形が IQ2_S**。

| 最適化 | IQ2_S 重みへの効果 |
|--------|-------------------|
| **層内 Q8 共有** | **大**（quantize 3→1 / 2→1） |
| **Q8_K 整数 dot** | **`cpu-multicore` dequant 比で有利** |
| **AVX2 IQ2_S dot** | **大**（IQ2_M の主ボトルネックだった dot を SIMD 化） |
| **AVX2 quantize** | **あり** |
| **prefill LM スキップ** | **大**（prefill 各 token で vocab GEMV 省略） |
| **greedy argmax** | **大**（decode で logits 全確保・softmax 省略） |
| **RoPE キャッシュ** | **中**（全 layer × head で **cos/sin 再計算**削減） |

Q4_K / Q5_K 混在テンソルでは **AVX2 Q4/Q5 dot** も有効。

##### スコープ外（明示）

- **token 量子化 `emb_lookup`**: 量子化 embedding 行は **block-wise `dequant_one_block_to`**（Q8 経路外）。F16 のみ **F16C SIMD** 化。
- **層跨ぎ / token 跨ぎ Q8 再利用**
- **KV キャッシュ**の量子化
- **top-p / 温度サンプリング**時の argmax ショートカット（**`FWD_LM_FULL`** 必須）
- **`-ffast-math`**（IQ / Q8_K / RMSNorm の数値崩れ。Makefile 無効）

変更履歴: **`doc/ChangeLog.md`**（**2026-05-23 04:53:10**、**04:34:38**）。

ROCm 版および CUDA **`gpu-cuda`（FP16）** はロード時に一度だけホスト上で量子化 tensor を F32 に復元し、F16 staging 経由で **全線形を FP16 VRAM** に載せる。norm は F32 のまま GPU。実行時 GEMV は FP16 カーネル（ROCm: `mm_f16_gemv_kernel`、CUDA: 同名相当）。

CUDA **`gpu-cuda-nvfp4`（NVFP4）** は線形 tensor をホストで F16 化したうえで **NVFP4 キャッシュ**（**`fp4_qwen3_weight_from_f16_host`** → **`fp4_quantize_weights_host_f16`**）にのみ H2D し、**線形の FP16 VRAM 複製は行わない**。**`token_embd.weight` だけ FP16 VRAM**、norm 系は **F32 VRAM**。起動時の **`gpu_model_create` で FP16→NVFP4 再変換は行わない**（**`wq_fp4` 等**をそのまま採用）。実行時の線形は **`fp4_qwen3_mm`**（M&lt;128 → FP4 GEMV、M≥128 → Tensor Core GEMM）。いずれも推論中の GGUF 逐次デ量子化は行わない。

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

出力時は特殊トークンを表示せず、GPT-2 byte fallback の Unicode codepoint 表現を raw byte に戻して端末へ書き出す。**Thinking ブロック内の文字列は `is_special` 対象外**（ChatML 特殊 ID とは別に通常テキストとしてトークン化）のため、thinking 対応モデルでは reasoning テキストがそのまま stdout に出る可能性がある（詳細は **「高度な機能（マルチターン・Thinking）」**）。

## 高度な機能（マルチターン・Thinking）

Qwen3 ファミリー（QwQ 等の reasoning 系を含む）では、公式スタックは **マルチターン ChatML** と **Thinking モード**（`enable_thinking`、`/think` / `/no_think`、thinking ブロック）を前提とする。本リポジトリは **1 ターン固定の `chat_encode`** と **1 プロセス 1 推論**のみを実装し、次は **スコープ外**である。

| 機能 | Qwen3 ファミリー（公式） | 本リポジトリ |
|---|---|---|
| ChatML 1 ターン | ○ | ○ |
| マルチターン履歴 | ○ | × |
| プロセス間 KV / 会話状態 | ○（FW 側） | × |
| Thinking オン／オフ | ○ | × |
| `/think`・`/no_think` | ○ | × |
| thinking ブロックの分離表示 | ○ | × |

**マルチターン（公式）** — ChatML で system / user / assistant をターン順に並べ、KV キャッシュまたは再 prefill で文脈を引き継ぐ。function calling では、過去の assistant 発話・tool 結果・（必要に応じた）reasoning 区間も次ターンへ渡す。

**マルチターン（本リポジトリ）** — 各 `main.c` の `chat_encode` は system + **1 回の user**（`-p`）+ assistant 開始のみ。過去ターンを CLI で渡す経路はなく、KV を次実行に引き継ぐ API もない（prefill は毎回ゼロから）。マルチターンに近づけるには、(1) ChatML を手で組み立てて `-p` に載せる、(2) `chat_encode` を拡張する、(3) KV 再利用（未対応）。`-l` を超える履歴は切り詰めまたは要約が必要。

**Thinking（公式）** — **hard switch**（`enable_thinking` / `apply_chat_template`）と **soft switch**（`/think` / `/no_think`）で thinking オン／オフ。thinking ブロック（テンプレートが挿入する reasoning 区間）は ChatML 特殊トークン（`<|im_start|>` 等）とは別に、通常テキストとしてトークン化される。

**Thinking（本リポジトリ）** — `enable_thinking` 相当の **assistant 直前プロンプト制御**、生成出力の **reasoning と最終回答の分離**（`print_tok` は ChatML 特殊 ID のみ抑制）、**`thinking_budget`** 等は未実装。thinking 対応 GGUF ではテンプレート不一致や reasoning 生出力が起きうる。拡張時は GGUF の **`tokenizer.chat_template`**（Jinja）に追随し、thinking 区間の encode／decode を追加する。

**技術参考（外部）** — テンプレート背景・ChatML / function calling・Thinking モードの公式解説 URL 一覧は **`README.md`** / **`README.en.md`** の **「高度な機能（マルチターン対話・Thinking モード）について」** を参照（Transformers chat templating、Qwen Function Calling / Quickstart / Transformers 推論 / vLLM、Qwen Cloud Thinking、Hugging Face モデルカード・ブログ等）。

### CPU forward

**`cpu`** / **`cpu-multicore`** の forward は、すべて `float` の activation buffer 上で逐次実行する。重みは GGUF mmap 上の raw tensor を参照し、dtype に応じて **`mm_f32`**、**`mm_f16`**、**`mm_quant_rows`**（ブロック dequant + 内積）に分岐する。

**`cpu-blas`** は上記と同一のレイヤー順序だが、F32 行列積と Attention を OpenBLAS に委譲し、量子化 GEMV は **「`cpu-blas`：Q8_K 活性化 GEMV」** の **`mm(..., q8_ready)`** / **`mm_quant_dot_rows`** 経路を使う（**`mm_quant_rows` は使用しない**）。

**`cpu-blas` レイヤー内の GEMV / BLAS 呼び出し順**（層 **`l`**、量子化 GEMV 全テンソル想定）:

1. **`rmsnorm(x → xb)`** — スカラー/OpenMP（他 CPU 版同様）。
2. **`q8_att`** 判定 → 必要なら **`quantize_row_q8_K(xb → q8)`**（**1 回**）。
3. **`mm(q)`**, **`mm(k)`**, **`mm(v)`** — 量子化型なら **`q8_ready=1`** で **共有 Q8** を参照。**`d=dim` / `kv_dim`** 行の OpenMP 行並列 dot。
4. **`rmsnorm_head_inplace(q/k)`**, **`apply_rope`**（**`Model.rope_cr/ci` キャッシュ参照**）, **KV cache 書込** — スカラー/OpenMP。
5. **Attention** — 各 head **`cblas_sgemv`**（K 内積）→ **`softmax`** → **`cblas_sgemv`**（V 合成）。**`openblas_set_num_threads(1)`** 下で OpenMP が **`n_heads`** 並列。
6. **`mm(xb2, xb, wo, q8_ready=0)`** — attn 出力 **`xb`** を入力に **都度 quantize**。
7. **残差 `x += xb2`**。
8. **`rmsnorm(x → xb)`** → **`q8_ffn`** → gate/up **共有 quantize** → **`mm(gate/up)`**。
9. **SwiGLU** — **`expf`/`sigmoid` 相当**の要素演算（OpenMP）。
10. **`mm(xb, hb, down, n=hidden, q8_ready=0)`** — **56 ブロック quantize** + **`d=dim`** 行 dot。
11. **残差 `x += xb`**。

**LM head**（**`generate`** が **`lm_mode`** を決定）:

| **`lm_mode`** | 条件 | 処理 |
|---|---|---|
| **`FWD_NO_LM`** | **`pos < n_prompt - 1`** | スキップ |
| **`FWD_LM_FULL`** | decode または prefill 最終 token、**`temp > 0`** | **`rmsnorm` → `mm(logits)` → `sample_token`** |
| **`FWD_LM_ARGMAX`** | 同上、**`temp <= 0`** | **`rmsnorm` → `mm_argmax_row` → `argmax_tok`** |

**埋め込み `emb_lookup`**: 量子化 token 行は **`dequant_one_block_to`**。F16 行は **F16C+AVX2** 時 **8 要素 SIMD `cvtph_ps`**（それ以外 OpenMP **`host_f16f32`**）。

1 token の処理は次の順序である。

1. `token_embd.weight` から token ID の行を読み、`x` に展開する。
2. 各レイヤーで `attn_norm` による RMSNorm を `xb` に出す。
3. `xb` に対して Q/K/V の GEMV を行い、`q` / `k` / `v` を作る。
4. Qwen3 固有の `attn_q_norm` / `attn_k_norm` を head ごとに in-place 適用する。
5. Q/K に RoPE を適用し、現在位置 `pos` の K/V を KV cache に書く。
6. GQA に従い、query head `h` は `kvh = h / kv_mul` の KV head を参照する。過去 `0..pos` の score を softmax し、Value の重み付き和を作る。
7. attention 出力を `attn_output.weight` で射影し、残差として `x` に加える。
8. `ffn_norm`、`ffn_gate`、`ffn_up`、`SiLU(gate) * up`、`ffn_down` の順で FFN を実行し、再び残差を加える。
9. logits が必要な位置だけ `output_norm` と `output.weight` を実行する（**`cpu-blas`** は **`lm_mode`** で **prefill スキップ / greedy argmax / フル logits** を切替）。

プロンプト消費中は次 token が既知なので、最後のプロンプト token 以外では LM head を省略できる（**`cpu-blas`** は **`FWD_NO_LM`** で **`forward` 内から省略**。**`cpu` / `cpu-multicore`** も同趣旨だが **`lm_mode` 引数はない**）。

### ROCm / CUDA forward（GPU）

ROCm 版の `forward_gpu` および CUDA 版の `gpu_forward` / `gpu_forward_prefill` は、embedding から LM head まで device buffer 上で実行する（CUDA は Prefill と Decode でエントリが分かれる）。各カーネル起動の依存は default stream の順序に任せ、forward の最後に `hipDeviceSynchronize()` する。

主なカーネルは次の通りである。

- `emb_f16_kernel`: token embedding の 1 行を FP16 から float activation に展開する。
- `rmsnorm_kernel`: block 内 reduction で二乗平均を求め、`float4` 単位も使って RMSNorm を適用する。
- `mm_f16_gemv_kernel`: 1 warp が 1 出力行を担当し、warp reduction で GEMV の和を作る。1 block は複数行を処理する（**`gpu-cuda`（FP16）** の線形層。**`gpu-cuda-nvfp4`** の線形は **`fp4_gemv_kernel`** 経路）。
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

利用する GGUF のファイル名は **`qwen3-8b/Makefile` の `MODEL`** を参照する。モデル本体は著作権とファイルサイズの都合でリポジトリに含めず、既定モデルの取得元は **`qwen3-8b/gguf.txt`** に URL として置く。

取得・検証の手順は次の 2 通りである。

1. **`cd qwen3-8b && make model`**: **`gguf.txt`** 先頭 URL の `blob/main` を `resolve/main` に置換して **`wget`** し、リポジトリ同梱の **`$(MODEL).sha256sum`** で **`sha256sum --check`** する。チェックサムファイルが無い・検証に失敗した場合はメッセージを出して終了し、破損ダウンロードは **`$(MODEL)`** を削除する。
2. **手動**: 上記と同様に URL を `resolve/main` に直して **`wget`** 等で取得し、**`sha256sum -c $(MODEL).sha256sum`** で確認する。

**`qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum`** は既定 **`MODEL`** 用の参照。別量子化・別サイズに切り替える場合は **`MODEL`** と本書の前提（メタキー `qwen3vl.*`・テンソル名）が実装と一致するかを確認すること。

## 制約・既知の制限

- **CPU 版**: IQ 混在 8B は計算量が大きく、**実用的な速度は期待しにくい**。OpenMP はアルゴリズム忠実なまま並列化するが、帯域 bound のため環境次第では伸びが限定的な場合がある。**`cpu-blas`** は F32 経路の OpenBLAS 化に加え量子化 GEMV を **Q8_K + 全型 AVX2 整数内積**（層内 Q8 共有）に置き換え、**RoPE キャッシュ**・**prefill LM スキップ**・**greedy `mm_argmax_row`** でオーバーヘッドを削るため **`cpu-multicore` より速くなることが多い**。**`-ffast-math`** を付けると IQ / Q8_K 量子化で出力が壊れる。**`-march=native`** は **AVX2/F16C** 等を有効化（移植性より当該 CPU 向け最適化）。
- **ROCm 版**: AMD GPU・ROCm・`hipcc`。通常は **`rocminfo` による `GPU_ARCH` 自動検出**だが、検出失敗時やクロスビルド時は手動 **`GPU_ARCH`** が必要。
- **CUDA 版（`qwen3-gpu-cuda` / `qwen3-gpu-cuda-nvfp4`）**: NVIDIA GPU・**`nvcc`**・**`libcudart`**。集約 Makefile 未統合。**汎用 GPU** は **`gpu-cuda`**（PTX 可）。**Blackwell NVFP4** は **`gpu-cuda-nvfp4`**（**CUDA 13**・CUTLASS・**`sm_120a`**）。**`build.polarquant`** は KV のみ PolarQuant-R（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**、任意 GPU 可／NVFP4 併用可）。**`gpu-cuda-nvfp4/third_party/cutlass`** と **`fp4_verify`** は clone／ビルド生成物で常時同梱されない。**`gpu-cuda-nvfp4`** では線形重みは NVFP4 のみ VRAM に載り、**`gpu-cuda` 比で線形 FP16 分（8B 級で約 15 GiB 相当）を節約**できる（代わりに **`token_embd`** は FP16 のまま）。
- **XDNA2 版（`qwen3-xdna2`）**: 恒久の全レイヤー **BF16 重み複製は行わない**。**mmap + 単一 GEMV 用 BF16 スクラッチ**（および `scratch_f32`）であり、代表的 8B 級 IQ 量子化モデルでも **`main-omp.c` に近い「GGUF を載せつつ増分バッファ」**になる（スクラッチの最大要素数は **`output.weight`** クラスの巨大行列にひもづき、VRAM／DRAM の余裕が依然必要になる場合がある）。変換済み GGUF でない限りロード済みモデルサイズより **桁違いの常駐 BF16 が乗らない**。NPU 本線には **MLIR-AIE / IRON** が生成した制御コード（`XDNA_GEMV_DIR`）。未配置時は OpenMP CPU フォールバック。`/dev/accel/accel0` は `render`。**推論レイテンシは GEMV のたびフル復号するため増えうる**。
- **テキストのみ**: Vision・マルチモーダル入力は未対応。
- **マルチターン対話**: 組み込み CLI・履歴管理・KV 再利用・公式 chat template の完全再現は未対応（**「高度な機能（マルチターン・Thinking）」** 参照）。
- **Thinking モード**: `enable_thinking` / `/think` / `/no_think` / thinking ブロック分離表示は未対応。
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

レイヤーあたり、`pos+1` 位置分の Key/Value で **`(pos+1) * kv_dim`** 要素（×4 バイト if float32）。全レイヤーで `n_layers` 倍。**`build.polarquant`**（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**）では **`(pos+1) * n_kv_heads * PQ_BYTES_HEAD`** バイト（K と V で 2 倍、**`PQ_BYTES_HEAD=64`**）。`-l` を大きくすると最悪ケースの割り当てが増える。

## 補足：トラブルシューティング早見表

| 現象 | 想定原因 | 確認・対処 |
|------|----------|------------|
| `hipcc` not found | `ROCM` 誤り | `make ROCM=/opt/rocm` 等 |
| ISA 不一致 / `GPU_ARCH not detected` | 自動検出失敗・手動指定の誤り | **`make -C gpu-rocm detect-gpu-arch`**。失敗時は **`rocminfo`** の **`Name: gfx*`** を確認し **`make build.gpu-rocm GPU_ARCH=…`** |
| `nvcc` not found / `nvlink` 失敗 | `PATH` に CUDA `bin` が無い | `export PATH=/usr/local/cuda/bin:$PATH` または各 CUDA ディレクトリの **`Makefile`** の `CUDA_HOME` を確認 |
| CUDA で PTX は動くが極端に遅い | `CUDA_GENCODE` が PTX のみ | 実機 **`sm_XX`** を `code=sm_XX` で指定して再ビルド（**`gpu-cuda`**） |
| Blackwell NVFP4 ビルド失敗 | CUDA 11 と 13 の混在・CUTLASS 未取得 | **`gpu-cuda-nvfp4`** で **`make blackwell`** または **`make cutlass`** → **`make build`** |
| 非 Blackwell で NVFP4 ビルドを試した | **`gpu-cuda-nvfp4`** は **`sm_120a`** 前提 | 汎用 GPU では **`gpu-cuda`** で **`make build`**（FP16）を使用 |
| **`NVFP4 quantize failed`** | **`gpu-cuda-nvfp4`** を非 Blackwell で実行、CUTLASS 未導入 | **`sm_120a`**・CUDA 13・**`make cutlass`**。汎用 GPU は **`gpu-cuda`** |
| **`make fp4-test` が FAIL / `Arch conditional MMA instruction... Aborting`** | **`fp4_gemm`** を **`compute_86` PTX** のみでビルドした古い **`fp4_gemm.o`** | **`gpu-cuda-nvfp4`** で **`make clean`** → **`make fp4-test`**（**`fp4_gemm.sm120a.o`** は **`BLACKWELL_NVCCFLAGS`** 固定）。Blackwell GPU 必須 |
| **NVFP4 で decode が極端に遅い** | 古いバイナリが M=1 を CUTLASS M=128 パディング経路に落としている | 最新 **`fp4_gemv_cached`** 入り **`gpu-cuda-nvfp4`** ビルドか確認。起動ログに **`GEMM M>=128, GEMV decode`** が出ること |
| **`gpu-cuda-nvfp4` の PolarQuant ビルドが `Killed`（Error 137）** | **`kernels.cu`** の **`sm_120a` + FP4 + PolarQuant** コンパイルが RAM 不足 | スワップ増設・並列ビルド停止後に再実行 |
| **`polarquant_init: head_dim must be 128`** | モデルの **`head_dim`** が 128 でない | 現状 **Qwen3-VL-8B** のみ想定。**`gpu-cuda`**（FP16）または PolarQuant 無効ビルドを使用 |
| **`pq-test` FAIL** | コードブック未初期化・GPU 非対応 | **`make pq-test`** の **`max_abs_err` / `rel`** を確認（閾値 rel ≤ 0.35） |
| mmap 失敗 | パス・権限 | `MODEL` を確認 |
| CPU が極端に遅い | IQ デ量子化コスト | **`cpu-blas`** を試す、ROCm 版の利用、`-n` を小さく |
| **`cpu-blas` ビルド失敗 / `cblas.h` not found** | OpenBLAS 開発パッケージ未導入・ヘッダ非標準パス | **`cd cpu-blas && make openblas`** または **`libopenblas-dev`** を手動インストール。**`CPPFLAGS=-I/.../openblas-pthread`** 等（**`cpu-blas/Makefile`** コメント・ビルド失敗時メッセージ参照） |
| **`cpu-blas` 出力が意味不明（同じ文字の連打等）** | **`-ffast-math`** による IQ 量子化の数値崩れ | リポジトリ同梱 **`cpu-blas/Makefile`** は **`-ffast-math` 無効**。手元で CFLAGS 上書きしている場合は外す |
| `/dev/accel/accel0` を開けない | `render` グループ未参加 / `amdxdna` 未ロード | `sudo usermod -aG render "$USER"`、`lsmod \| grep amdxdna` を確認 |
| XDNA2 で速度が出ない／NPU が効いていない | `XDNA_GEMV_DIR` 未設定、形状欠け、DRM 不可、`XDNA_FORCE_CPU` 等 | 起動時の **`=== XDNA GEMV / NPU ctrlcode status ===`** で各形状の `MISS` を確認。**`./xdna2/qwen3-xdna2 model.gguf --xdna-status`** で軽量診断。推論後の **NPU GEMV / CPU GEMV** カウントが **CPU のみ**なら NPU 経路は実行されていない |
| **`CREATE_HWCTX` が EINVAL（`qwen3-xdna2` / `qwen3-xdna2-bfpx`）** | ドライバが列数・タイル数・QoS・DEV_HEAP・ファームを拒否 | 実装は **`vaddr=0`** で `CREATE_BO` し **必ず `mmap()`** して `userptr` を確立、**QoS は全 0**（`qos_meet` 抵触回避）、`num_tiles` は **`ncol×core.row_count`** を主軸に `core+mem+shim` 合算と `1` を順次フォールバック。なお解消しない場合は **`XDNA_NUM_COL`**・**`XDNA_NUM_TILES`**・**`XDNA_HEAP_SIZE`** 上書きと `dmesg` の `amdxdna` 行（`MAP_HOST_BUFFER status 0x4000003` などのファームエラー、ファーム `amdnpu/<vendor>_<rev>/npu.sbin` の有無）を確認 |
| XDNA2 ビルドで `drm/drm.h` not found | カーネル UAPI ヘッダ未インストール | `apt install linux-libc-dev` 等で `<drm/drm.h>` を導入 |

## 補足：ドキュメント間の役割

- **`README.md`**: ビルド・実行・バリアント選択の手順、ライブラリ非依存の方針とその意義、**マルチターン対話・Thinking モードの対応状況と技術参考 URL**（日本語）。
- **`README.en.md`**: 上記と同等の内容（英語）。**公式仕様 vs 本リポジトリ**の対比と外部参考リンクは **「Advanced features (multi-turn chat and Thinking mode)」** に集約。
- **`qwen3-8b/xdna2/xdna-gemv/README.md`**: **`qwen3-8b/xdna2/xdna-gemv/`** 配下（**`kernels/`**・**`toolchain/`**・スタブ生成）への入口。
- **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md`**: XDNA2 の **ctrlcode**・GEMV 形状・スタブ／実機バイナリ・**`--xdna-status`** の入門。**GPU（ROCm/HIP）カーネルとの対比**（§3）もここで扱う。
- **`qwen3-8b/xdna2/xdna-gemv/toolchain/README.md`**: **mlir-aie / IRON / Peano / `aiecc`** に沿った **NPU 用 `bf16-gemv-*.bin` 自前生成**の手引き（コマンド列と注意点）。**`qwen3-xdna2` は ioctl のみで XRT 非依存**であること、IRON／mlir-aie の公式サンプルが取る **XRT 検証パス**、および Linux カーネル文書 **AMD NPU** における **`ctrlcode`** の整理を冒頭で対照するための参照になっている。**東京科学大学（2026年現在の名称。旧・東京工業大学）ACRi** ルームの日本語チュートリアル（外部リンク）も紹介される。本文は日本語（です・ます調）。
- **`doc/design.md`（本書）**: 現行の設計・仕様。
- **`doc/ChangeLog.md`**: 日付付き変更履歴。
- **外部（参考）**: **AMD XDNA** のアーキテクチャ概要・世代・ソフトウェアスタック等は、別リポジトリ **[thamada/xdna-overview](https://github.com/thamada/xdna-overview)** にまとめてある（本リポジトリの実装説明とは独立した背景資料）。

実装の詳細は **`qwen3-8b/*/main.c`**（経路ごとにディレクトリが分かれ、ファイル名はいずれも **`main.c`**）の先頭コメントとソースを参照する。

## 補足：`design.md` 更新時のチェックリスト

1. **`qwen3-8b/`** にソースまたはターゲットを増やしたら、**構成表**と **バイナリ表**を更新する。  
2. **`Makefile`** の変更と本書を同期する。  
3. 仕様変更は **`doc/ChangeLog.md`** にも記載する。  
4. 利用者向け手順や方針を **`README.md`** で変えたら、 **`README.en.md`** も同趣旨に揃える（またはその逆）。

## 補足：CUDA NVFP4（`gpu-cuda-nvfp4`）実装メモ

**`qwen3-8b/gpu-cuda-nvfp4/`** の Blackwell 向け経路。**`BONSAI_FP4=1`** 固定（**`-DBONSAI_FP4=1`**）。共有ソース（**`main.c` / `kernels.cu` / `gpu.h`**）は **`../gpu-cuda/`** を参照する。

### ロード（H2D）

1. **`upload_weights_gpu`**（**`main.c`**）: GGUF を CPU で逆量子化し、ホスト **F32/F16 ステージング**（`materialize_host_f16`）へ展開。
2. 線形 tensor（Q/K/V/O、gate/up/down、LM head）: **`upload_linear_fp4`** が **`fp4_qwen3_weight_from_f16_host`** を呼び、**NVFP4 キャッシュ**（`d_fp4` + CUTLASS インターリーブ **`d_sf`**）だけを VRAM に載せる。**線形 FP16 の device バッファは作らない**。
3. **`token_embd.weight`**: 従来どおり **FP16 VRAM**（embedding lookup 用）。
4. norm 系: **F32 VRAM**。
5. **`gpu_model_create`**（**`kernels.cu`**）: **`wq_fp4` 等**を **`dev_adopt_layers_fp4`** で採用。**`use_fp4=1`**。起動時の FP16→NVFP4 変換ループは **ない**。

### 実行（線形層）

**`fp4_qwen3_mm`**（**`fp4_qwen3.cu`**）が M で分岐する。

| M | 経路 | 実装 |
|---|------|------|
| **1**（decode） | FP4 GEMV | **`fp4_gemv_cached`** — warp リダクション、SF インデックスは **`sfb_sf_index`**（CuTe **`layout_SFB`** と一致） |
| **2〜127**（短い Prefill） | FP4 GEMV バッチ | **`fp4_gemv_batch_cached`** |
| **≥128**（長い Prefill） | CUTLASS Tensor Core GEMM | **`fp4_gemm_run_cached`**（活性 BF16 をその場量子化） |

**`kernels.cu`** の **`gpu_mm` / `gpu_mm_batch`** は **`BONSAI_FP4`** 時、線形をすべて **`fp4_qwen3_mm`** に委譲する。

### ビルド上の注意

- **`MAIN_OBJ`**（**`main.pq$(BONSAI_POLARQUANT).o`**）も **`BONSAI_FP4=1`** でコンパイル（**`gpu-cuda-nvfp4/Makefile`**）。ソースは **`$(SRC_DIR)/main.c`**。
- **`kernels.fabr$(FA_BR).pq$(BONSAI_POLARQUANT).o`** で PolarQuant / FA_BR 変更時の stale `.o` を防止。ソースは **`$(SRC_DIR)/kernels.cu`**。
- **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** は **`CUDA_GENCODE`** とは独立し、常に **`BLACKWELL_NVCCFLAGS`**（**`-std=c++17`** + **`arch=compute_120a,code=sm_120a`**）でビルドする。CUTLASS NVFP4 は **C++17 必須**。CUTLASS NVFP4 の sm_120 MMA を **`compute_86` PTX** だけに載せると実行時 abort する。
- **CUTLASS バージョン**: **`CUTLASS_TAG=v4.5.0`**（旧 **`v3.9.0`** から更新）。初回またはタグ変更後は **`make cutlass`**（**`third_party/cutlass`** が無い／タグ不一致なら clone／再 clone）。
- **CUDA 13 非推奨ベクトル型（`long4` 等）**: v4.5.0 の **`platform.h`** で CUTLASS 側が解消済み。旧版で必要だった **`-Wno-deprecated-declarations`** は不要。
- **nvcc 警告 #20012**（**`= default` コンストラクタ + `__device__`/`__host__`**）: v4.5.0 の **`sm100_static_tile_scheduler.hpp`** が CUTLASS 側コードとして発する。**`third_party/cutlass`** は当リポジトリ側では改変しない。CUTLASS 公式 nvcc ビルド（**`python/cutlass_cppgen/backend/compiler.py`**）と同じ **`-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored`** を **`BLACKWELL_NVCCFLAGS`** に付ける。背景・参考 URL は **`gpu-cuda-nvfp4/Makefile`** コメントに記載。
- 検証: **`cd qwen3-8b/gpu-cuda-nvfp4 && make fp4-test`**（**`fp4_verify.cu`** — square / wk / wq_M256 の 3 ケース、ratio &lt; 0.45 で PASS。**Blackwell GPU 必須**）。

## 補足：CUDA PolarQuant-R（`build.polarquant`）実装メモ

論文 [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)（**PolarQuant-R**）に沿った KV キャッシュ圧縮。**`BONSAI_FP4`** とは独立（**`BONSAI_POLARQUANT=1`**）。NVFP4 線形重みと併用は **`gpu-cuda-nvfp4`** で **`make build.polarquant`**。

### アルゴリズム概要

1. **前処理（PolarQuant-R）**: head 次元ベクトル \(x \in \mathbb{R}^{128}\) に固定ランダム符号 \(\sigma_i \in \{\pm1\}\) を掛け、**FWHT-128**（Walsh-Hadamard）後 \(1/\sqrt{128}\) でスケール（`polarquant_kernels.cuh`）。
2. **ブロック分割**: 128 次元を **8 ブロック × 16 座標**。各ブロックは **L=4 再帰 polar 量子化**（**`pq_polar_encode_block` / `pq_polar_decode_block`**）。
3. **量子化ビット幅**（論文 Section 4.1 準拠）: L1 は 8 角度 × **4 bit**（\([0,2\pi)\)）；L2〜L4 は各 **2 bit**（\([0,\pi/2)\)）。半径は **fp16**。
4. **`PQBlock`（8 bytes）**: fp16 半径 + L1 インデックス（32 bit）+ L2/L3/L4 インデックス（各 uint8 に 2 bit ずつパック）。
5. **コードブック**: 起動時 **`polarquant_init`**（`polarquant.cu`）で L1 は等分割、L2〜L4 は Lloyd-Max 最適化セントロイドを **`PQState`** として H2D。

### VRAM 削減（Qwen3-VL-8B 想定）

| 項目 | F32 KV | PolarQuant-R |
|------|--------|--------------|
| head あたり | 512 B | **64 B**（`PQ_BYTES_HEAD`） |
| 圧縮率 | — | **約 8×** |
| 36 layer × 512 seq（K+V 概算） | ~144 MiB | ~18 MiB |

F32 復号は Attention タイルの共有メモリ上のみ。永続 VRAM に F32 KV は載せない。

### Flash Attention 統合（`kernels.cu`）

- **`BONSAI_POLARQUANT`** 時、`kc`/`vc`（F32）の代わりに **`kc_pq`/`vc_pq`**（`PQBlock` 配列）。
- 書き込み: **`polarquant_kv_write_one`**（decode）/ **`polarquant_kv_write_batch`**（prefill）。
- カーネル: **`flash_attn_gqa_pq_kernel`** / **`flash_attn_prefill_gqa_pq_kernel`** — タイル読み出し時 **`pq_decode_head`** で F32 復号後、既存 Flash Attention と同様に QK^T / softmax / V。
- **制約**: **`head_dim=128` 固定**（`PQ_HEAD_DIM`）。Qwen3-VL-8B（`n_kv_heads=8`）向け。

### ビルド・検証

```bash
cd qwen3-8b/gpu-cuda
make build.polarquant
make pq-test    # ラウンドトリップ rel ≤ 0.35 で OK

cd ../gpu-cuda-nvfp4
make cutlass    # 初回のみ
make build.polarquant
make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
```

起動ログ（PolarQuant 有効時）: **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**

**`gpu-cuda-nvfp4` + `build.polarquant`** では上記に加え **`dequant -> NVFP4 linear layers`** と **`FP4 Tensor Core path enabled (GEMM M>=128, GEMV decode)`** も出る。

## 補足：XDNA2 NPU 実装メモ

`qwen3-8b/xdna2/main.c` と **`qwen3-8b/xdna2-bfp16/main.c`** はいずれも **`amdxdna` カーネルモジュール**（`drivers/accel/amdxdna`）の DRM ioctl を直接呼ぶ単一ソース実装である。XRT (Xilinx Runtime) / xdna-driver ユーザランドや C++ shim 等の外部依存を持たない。**主な違いはホスト側の線形重みの持ち方**（**mmap を推論中も読む BF16/GEMV スクラッチ方式** と **変換済み BFPX＋チャンクステージング＋mmap 早期解放**）である。`XdnaDev` / `ERT_START_NPU` / GEMV 入力・出力の短命 SHMEM とフォールバックの枠組みは共通である。

主要コンポーネント:

- **UAPI 取り込み**: `<drm/amdxdna_accel.h>` の構造体・ioctl 番号を inline で持つ（外部ヘッダ未配備でもビルド可）。
- **`XdnaDev`**: `open("/dev/accel/accelN")` した fd、`hwctx_handle`、`syncobj_handle`、AIE 列数・行数・FW バージョン等を保持。
- **`XdnaBo`**: 1 つの DRM バッファオブジェクト（`AMDXDNA_BO_SHMEM` / `AMDXDNA_BO_DEV_HEAP` / `AMDXDNA_BO_CMD`）を `handle`、`mmap()` 後の userspace ポインタ、NPU 側仮想アドレス `xdna_addr` の組で持つ。
- **`npu_open`**: `DRM_IOCTL_AMDXDNA_GET_INFO` で AIE topology / FW を取得し、`DRM_IOCTL_AMDXDNA_CREATE_HWCTX` でハードウェアコンテキストを作成、64 MiB の `DEV_HEAP` 命令バッファと `CMD`/`SHMEM` 補助 BO を準備する。
- **`npu_submit_start_npu`**: `ert_packet` ヘッダ + `cu_mask` + `amdxdna_cmd_start_npu` 構造体（命令バッファアドレス・サイズ・引数）を `CMD` BO に書き込み、`DRM_IOCTL_AMDXDNA_EXEC_CMD` で投入。
- **`npu_wait`**: `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` でコマンド完了を待つ。
- **`load_gemv_kernel`**: `XDNA_GEMV_DIR/bf16-gemv-<n>x<d>.bin` から事前コンパイル済の control code を命令バッファアリーナにロードする。MLIR-AIE / IRON ツールチェイン側で生成する想定。**マジック **`GQF3XDNA`** のスタブはロードしない**（誤実行防止）。
- **`launch_mm_bf16`**: NPU ディスパッチが可能な場合（`have_device && !force_cpu && XDNA_GEMV_DIR`）にのみ **`weight_prepare_bf16`** で単一 **`w_scratch_bo`** に復号転送し、NPU 経路（ctrlcode あり）か CPU OpenMP BF16 GEMV を呼ぶ。**最初から NPU 不可**な run では BF16 スクラッチを確保せず、**`main-omp.c` と同じブロック単位の量子化直 GEMV**（`mm_quant_rows_xdna` / `mm_f32_xdna` / `mm_f16_xdna`）へ自動分岐し、メモリと走行コストを削減する。
- **`print_xdna_gemv_ctrlcode_report`**: `npu_open` 後に、DRM・環境変数・6 形状分の **`bf16-gemv-<n>x<d>.bin`** 読み取り可否を一覧する。推論ループとは独立し **`--xdna-status` / `-X`** からも呼ばれる。
- **重みのレイアウト**: 線形層は **GGUF mmap** の量子化レイアウトを **`WeightsDev` が参照**。各 GEMV 前にだけ **単一 BF16 SHMEM (`w_scratch_bo`)** と **FP32 ステージング (`scratch_f32`)** で展開。小型の norm は mmap 指す。
- **CPU 上の処理**: RMSNorm、Qwen3 ヘッド RMSNorm、RoPE、KV cache 書き込み、FlashAttention 相当、SwiGLU、残差加算、softmax、サンプリングをすべて OpenMP で並列化（`main-omp.c` と同じ粒度）。

NPU 上で実際に高速 GEMV を回すには **MLIR-AIE / IRON** で生成した形状 `(n, d)` 専用 BF16 GEMV 制御コードバイナリが必要である。**詳細なセットアップコマンド・`aiecc` オプション・IRON `GEMV(M,K)` とファイル名の対応**は **`qwen3-8b/xdna2/xdna-gemv/toolchain/README.md`** を参照。**同一ファイル名のスタブ（先頭マジック `GQF3XDNA`）はリポジトリに同梱されるが、`load_gemv_kernel` はこれをロードしない**（実機は非スタブに差し替える。**ctrlcode の全体像・GPU（ROCm/HIP）カーネルとの対比・ユーザー主導で書けるかどうか**は **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md`**（§0・§2・§3 ほか）参照）。バイナリ未配置などで CPU に落ちる際、**`qwen3-xdna2`** は BF16 GEMV を OpenMP で実行する（実装では NPU 経路との **bit-identical** が謳われている。**`xdna2/main.c` の先頭コメント**参照）。一方 **`qwen3-xdna2-bfpx`** は **`mm_bfpx`** であり、出力が **`qwen3-xdna2` とビット単位では一致しない**。

### `xdna2-bfp16/main.c`（BFPX）メモ

- **`BfpxMat`**: 各行についてブロックごとに BF16 スケールと int8 係数を保持する。
- **`bfpx_convert_weight_2d`**: 期待する **`n_out × n_in`** と GGUF の **`ne[0]×ne[1]`** を突き合わせ、`main-omp.c` と同じ GEMV 向け論理形状に正規化する（転置 `[n_in,n_out]` は full dequant 後に行抽出）。
- **`model_drop_gguf_mmap`**: 変換完了後に tensor 名・mmap・モデル fd を解放し、推論中は BFPX とノルム用 F32 のみを参照する。

## 補足：本書の保守方針

設計書は **実装の下位互換の参照**である。実装と矛盾する場合は、実装か本書のいずれかを修正する。本リポジトリは **Qwen3 系のみ**を対象とするため、**他アーキテクチャ専用のファイル名・ターゲットを本書に書かない**。
