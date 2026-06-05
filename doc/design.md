# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog.md` を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

### リポジトリの目的とスコープ

本リポジトリは、**Qwen3 系（Qwen3-VL-8B-Instruct）** の **GGUF** 形式モデルを、**単一または少数の C／HIP／CUDA／Vulkan compute ソース**からビルド可能な形で **推論（テキスト生成）**するエンジンである。**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムにはリンクしない。** コアは **標準 C と `libm`**。AMD GPU 版は **ROCm/HIP**（コンパイラ・ランタイムでありニューラルネット用の高レベルフレームワークではない）、**Vulkan compute 版**（**`gpu-vulkan/`** — **`libvulkan`** + GLSL compute シェーダ。ROCm/CUDA 不要のベンダー非依存 GPU 向け付録）、NVIDIA GPU 版は **CUDA Toolkit / `nvcc`**（同様に低レベル）、CPU 並列は **OpenMP**（**`cpu-multicore`**）または **OpenMP + OpenBLAS**（**`cpu-blas`** — BLAS は F32 GEMV / Attention 集約。量子化 GEMV は **Q8_K 活性化 + ggml 準拠の整数内積**）、XDNA2 NPU 版は **`amdxdna` DRM ioctl（UAPI）** を直接利用する。Python ランタイムや `torch` に依存する層は置かない。**GGUF の読み取り・トークナイズ・Transformer フォワード・サンプリング**を一連のコードパスとして理解・改変しやすくすることを目的とする。学習・ファインチューニング・バッチ推論の最適化はスコープ外であり、主に **単発プロンプトからのテキスト生成**（`-p` + ChatML 1 ターン）を想定する。**マルチターン対話・Thinking モード**は Qwen3 ファミリーで重要だが本リポジトリでは未対応（**「高度な機能（マルチターン・Thinking）」** 参照）。

#### ライブラリ非依存とその意義

高レベルフレームワークに載せた推論は実装が簡潔になり高速化もしやすいが、**計算手順・メモリ配置・アライメント・量子化レイアウト**等がランタイム内部に隠れやすい。本リポジトリはその抽象層に依存せず、推論の実体を **C の明示的なコードパス**として観察・検証・変更できる状態に置く。フレームワークの代替を第一目的とするものではない。

- **理解可能性**: モデルファイルからの読み取り、バッファ配置、演算順序をソースと本書で追跡できる。
- **依存関係の単純化**: Python 環境や大規模 ML スタックを前提とせず、コンパイラと必要最小限の実行環境で経路を確認できる。
- **実験の自由度**: 量子化・メモリ表現（例: BFPX）・CPU/GPU/NPU の分担・`/dev/accel` への直接アクセスなど、抽象化に縛られやすい領域を試しやすい。
- **参照実装としての価値**: 最小構成で Qwen3 系デコーダ推論が成立する見取り図として、他スタックとの比較・検証の基準になる。

**最高性能や機能網羅を第一目的とはしない。** 主眼は、LLM 推論をブラックボックスにせず、実装の細部を把握したうえで改造できることである。利用者向けの入口説明は **`README.md`**（日本語）および **`README.en.md`**（英語）に詳しい。**README 本文の主眼は CPU 3 バリアント**（**`cpu` / `cpu-multicore` / `cpu-blas`**）。**ROCm / Vulkan / CUDA / XDNA2** は **付録**（README 末尾）として扱う。

文中の「decoder-only」「GQA」「FlashAttention 系デコードカーネル」等は、**Transformer デコーダの一般的なパターン**を指す。**推論ソースはすべて `qwen3-8b/` 配下であり、実行経路ごとに `cpu/`・`cpu-multicore/`・`cpu-blas/`・`gpu-rocm/`・`gpu-vulkan/`・`gpu-cuda/`・`gpu-cuda-nvfp4/`・`xdna2/`・`xdna2-bfp16/` の各ディレクトリに **`main.c`（CUDA 版は **`kernels.cu`**、Vulkan 版は **`vk_*.c`** + **`shaders/*.comp`**）を置いた単一ソース構成**とする。** 対象例は **Qwen3-VL-8B-Instruct** の **IQ2_S / IQ3_S 等が混在した GGUF**（例: `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`）。**Vision（画像エンコード・deepstack・画像トークン）は実装しない**。**テキスト用デコーダのみ**を実行する。

### 実装バリアント（本リポジトリに含まれるもの）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `qwen3-8b/cpu/main.c` | CPU、単スレッド | GGUF mmap、`qwen3vl.*` パース。線形層は **IQ2_S / IQ3_S / Q4_K / Q5_K** 等を **`QK_K=256` ブロック単位**に逆量子化しつつ GEMV（全重みの float 一括展開なし）。`libm` のみ。**Prefill** は 1 トークンずつ forward し stderr に **progress bar**（**`Prefill [====...]`**、幅 40）と prefill / decode / total の **スループット要約**を出力。 |
| `qwen3-8b/cpu-multicore/main.c` | CPU、**OpenMP** | 上記と同一アルゴリズム。**GEMV** は出力行並列、**Attention** はヘッド並列、`qwen3-8b/gpu-rocm/main.c`（ROCm 版）のカーネル粒度に相当する並列化（RoPE、RMSNorm、残差、SiLU 等）。 |
| `qwen3-8b/cpu-blas/main.c` | CPU、**OpenMP + OpenBLAS** | **`cpu-multicore`** と同一デコーダ・同一 GGUF。**F32 行列積**（**`cblas_sgemv`**）と **Attention の K 内積・V 合成**を OpenBLAS に委譲。量子化 GEMV は **Q8_K 活性化 + 全型 `vec_dot_*_q8_K` 整数内積**（**`__AVX2__`** で IQ2_S/IQ3_S/Q4_K/Q5_K）。**層内 Q8 共有**（**`mm(..., q8_ready)`**）。**RoPE cos/sin キャッシュ**、**prefill 中 LM head スキップ**（**`FWD_NO_LM`**）、**greedy 時 `mm_argmax_row`**（**`FWD_LM_ARGMAX`**）。**F16 埋め込み**は **F16C+AVX2** で 8 要素 SIMD 変換。**Prefill progress bar** とスループット要約を stderr に出力。**`-march=native`** 既定。 |
| `qwen3-8b/gpu-rocm/main.c` | **ROCm / HIP** | 量子化重みを **行単位融合逆量子化** → **F16 VRAM**（**`max_tensor_nelements` ステージング廃止**）。任意で **`<model>.gguf.fp16`** オフラインキャッシュ（**`make pack-cache`** / **`--pack-fp16-cache`**。**`make run`**（引数なし **`make`**）は **`manifest`** 欠落・更新時に **`pack-cache`** を自動実行）。**Prefill バッチ**（**`forward_prefill_gpu`** — 全プロンプトを 1 回 forward。線形層は **hipBLAS `GemmEx`**（[llama.cpp](https://github.com/ggml-org/llama.cpp/) の `cublasGemmEx` 経路と同趣旨）。フォールバック **`mm_f16_gemv_batch_kernel`**）+ **Decode 1 トークン**（**`forward_gpu`** — **`mm_f16_gemv_kernel`** + Flash decode）。**Flash 系 Prefill/Decode 注意**・**KV カーネル書き込み**・**レイヤー間のホスト非介在**・GPU サンプリング（top-p 時は logits D2H フォールバック）等。**Prefill progress bar** と prefill / decode / total の **スループット要約**を stderr に出力。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へベンチ＋**VRAM 内訳**（**`GpuVramProfile`** / **`model_vram_profile`**）。**`make log` / `make log.push`**（ファイル読取）。**`gpu-rocm/` で引数なし `make`**（**`.DEFAULT_GOAL := run`**。**`ensure-model`** + **`pack-cache`**）が AMD GPU 向け既定エントリ。**`make build`** はバイナリ +（MODEL 存在時）**`manifest`** 生成用。ビルド時 **`GPU_ARCH`** は **`gpu-rocm/Makefile`** が **`rocminfo`** から自動検出。**リンク**: `-lhipblas -lrocblas -lstdc++`（**`fp16_cache_io.o`** リンクのため **g++ / libstdc++-dev** 要）。 |
| `qwen3-8b/gpu-vulkan/` | **Vulkan 1.1 compute** | **`main.c`** + **`vk_context` / `vk_alloc` / `vk_pipeline` / `vk_kernels`** + **`shaders/*.comp`**（18 本 → **`glslangValidator`** で **`.spv`**）。**ROCm / CUDA 不要**（**`-lvulkan`**）。重みは **`gpu-rocm` / `gpu-cuda` と同じ FP16 経路**（**`<model>.gguf.fp16`** または GGUF 行単位融合逆量子化）。**Prefill / Decode** とも **FP16 GEMV compute シェーダ**（**hipBLAS GemmEx / cuBLAS 相当の batched GEMM 未実装**）。**Flash Attention**（GLSL）。**`gpu.h`** API（**`gpu_forward_prefill`** / **`gpu_forward`**）。各 **`vkCmdDispatch`** で descriptor set 確保・更新・**fence 待ち**（同期実行 — ホストオーバーヘッド大）。デバイスポインタは **`vk_malloc` ハンドル**（整数 ID）。**`QWEN3_VK_SHADER_DIR`** で SPIR-V 探索。**Prefill progress bar**。**`BENCH_LOG_FILE`**（**`vram_total`** + 簡易 **`[vram_breakdown]`**（線形重みは **`vram_total` に含む**・**`vram_weights_linear` キー未出力**）。**`vram_device_used`** は未取得）。**`make log` / `make log.push`**（第 2 列は **`vulkaninfo` の `deviceName`**。**`GPU_VK`** は **`awk -F'='`** で抽出）。**初期実装** — 長プロンプト **`make log.push`** 実測 **~3 tok/s** 級（**`gpu-rocm` 比 10〜40× 遅い**。README 付録参照）。 |
| `qwen3-8b/gpu-cuda/main.c` + `kernels.cu` | **NVIDIA CUDA（FP16）** | **Prefill バッチ** + **Decode 1 トークン**、**Flash Attention**（GQA）。全線形 **FP16 VRAM**。GGUF **行単位融合逆量子化**、または **`<model>.gguf.fp16`** オフラインキャッシュ（**`make run`** / **`make pack-cache`** / **`--pack-fp16-cache`**）。**`Makefile`** は **`nvidia-smi`** で **`CUDA_GENCODE`** / **`FA_BR`** を自動選択（Blackwell **12.x → `sm_120` + FA_BR=32**）。任意で **`build.polarquant`**: KV **PolarQuant-R**（64 B/head、F32 比 ~8×）。サンプリング **logits D2H**。**Prefill progress bar** とスループット要約。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へベンチ＋**VRAM 内訳**。**`make log` / `make log.push`**（ファイル読取）。**`gpu-cuda/` で単体ビルド**。 |
| `qwen3-8b/gpu-cuda-nvfp4/` + 共有 `gpu-cuda/` | **NVIDIA CUDA（NVFP4）** | 上記と同じ Prefill / Decode / Flash Attention。線形層はロード時 **NVFP4 のみ**（**`fp4_qwen3`**、**`BONSAI_FP4=1`** 固定）。GGUF 行単位融合逆量子化、または **`<model>.gguf.nvfp4`** オフラインキャッシュ（**`make run`** / **`make pack-cache`** / **`--pack-nvfp4-cache`**。**`FP4_CACHE_VERSION=2`**。旧 v1 は再 **`pack-cache`**）。**`token_embd`** は **FP16**（量子化 GGUF から行単位 H2D）、norm は **F32**。線形は **`fp4_qwen3_mm`** — prefill / decode とも **CUTLASS NVFP4 GEMM**（**`fp4_gemm_run_cached`**、**`M` を 128 整列**、活性はその場 FP4 量子化。**`FP4_QUANT_MAX_ABS=1024`** で極大活性の NaN を防止）。起動 smoke・回帰は **`make fp4-test`** / **`make sfa-verify`**。**`--fa-debug`**（共有 **`fa_debug.*`**）で FA/KV/層トレース。調査ログは **`gpu-cuda-nvfp4/DEBUG.md`**（ベース **`433319eb`**）。**`fp4_quantize_weights`** は GPU カーネル（**`FP4_WEIGHT_SFB_LAYOUT_M=128`** + **`d_sf_lut`**）。任意で **`build.polarquant`**。**`BENCH_LOG_FILE`** + VRAM 内訳。**`make log` / `make log.push`**（**`BENCH_TEMP`** / **`BENCH_SEED`** 反映）。要 **CUDA 13 + CUTLASS + sm_120 系 GPU**。 |
| `qwen3-8b/xdna2/main.c` | **AMD Ryzen AI NPU (XDNA2)** | **CPU OpenMP 版と同様**に線形ウェイトは **GGUF mmap 上の量子化形式を参照**。埋め込みは行単位ブロック復号。各 **GEMV ごとに**当該重み行列を **`AMDXDNA_BO_SHMEM` の単一 BF16 スクラッチ**へ展開して NPU が DMA、`scratch_f32` で逆量子化～BF16 を兼用。rmsnorm などの小型 F32 も mmap 指す。`DRM ioctl` と **`ERT_START_NPU`** 経路、`/dev/accel/accelN` 不可／制御コード未配置時の **OpenMP BF16 CPU フォールバック（NPU と bit-identical）**は従来どおり。XRT 不要・UAPI inline 持ち運びは不変。**スクラッチサイズはテキスト経路 GEMV に必要な最大要素数のみ**（パーサ済み名前走査、`TensorInfo` は推論前に開放しうる）。**起動時レポートと `--xdna-status` / `-X`** で各形状の **`bf16-gemv-<n>x<d>.bin`** 可否・推論後の NPU/CPU GEMV カウンタを確認できる。 |
| `qwen3-8b/xdna2-bfp16/main.c` | **AMD Ryzen AI NPU (XDNA2) + BFPX ホスト重み** | **`qwen3-8b/xdna2/main.c` と同一の DRM ioctl** および **チャンク BF16 GEMV（NPU 経路の枠組み）** を共有する。**密行列レイアウト**の重みはロード時に **BFPX（ブロックごとに BF16 スケールと int8 係数、ブロック長 64）** に変換しホストのみ保持し、GGUF mmap は変換完了後に解放する。**論理形状は OpenMP CPU 版（`cpu-multicore/main.c`）の `mm(..., n_in, n_out)` と一致**させ、`[n_in,n_out]` 型の GGUF 転置は **`bfpx_convert_weight_2d`** で吸収。量子化に加えブロック近似のため、**GEMV で逐次 BF16 に展開する mmap スクラッチ方式（`xdna2/main.c`）と同一ビットでの一致は期待できず**、品質が劣ることがある。NPU 不可時の CPU は **`mm_bfpx`** が単精度浮動小数点数の活性と BFPX 形式の重みの積を計算する。 |

メタデータキーは **`qwen3vl.*`**。Qwen3 固有として、線形射影の直後に **`attn_q_norm` / `attn_k_norm`**（ヘッド長に対する RMSNorm）を挟み、その後 **RoPE** を適用する。チャットは **ChatML**（`<|im_start|>` / `<|im_end|>` 等）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` | ビルド・実行・方針の説明（日本語）。 |
| `README.en.md` | 同上（英語）。 |
| `qwen3-8b/cpu/main.c` | CPU 単スレッド推論。**Prefill progress bar**（**`prefill_progress_*`**）と prefill / decode スループット要約を stderr に出力。 |
| `qwen3-8b/cpu-multicore/main.c` | CPU OpenMP 並列推論。**ソース先頭**に **`qwen3-8b/gpu-rocm/main.c`**（ROCm/HIP）との並列粒度対応、当ディレクトリ **`make build`**（**`qwen3-cpu-omp`**）を記載。 |
| `qwen3-8b/cpu-blas/main.c` | CPU OpenMP + OpenBLAS。**Q8_K GEMV**（全型 AVX2 整数内積・層内 Q8 共有）、**RoPE キャッシュ**、**lm_mode**（prefill LM スキップ / greedy argmax）、**F16 emb F16C**。詳細は **「量子化と行列積」→「`cpu-blas`：Q8_K 活性化 GEMV」**。**Prefill progress bar** を stderr に出力。 |
| `qwen3-8b/cpu-blas/Makefile` | **`qwen3-cpu-blas`** をビルド。**`.DEFAULT_GOAL := run`**。**`ensure-model`**。**`-ffast-math` 無効**（IQ 量子化の精度維持）。**`-march=native`** 既定。**`openblas_set_num_threads(1)`** は **`main.c`** 実行時。**`make openblas`** で **`libopenblas-dev`** / **`libgomp1`** を apt 導入。**`cblas.h` 未検出時**はエラーメッセージで **`make openblas`** と **`CPPFLAGS`** 例を案内。 |
| `qwen3-8b/gpu-rocm/main.c` | ROCm 推論。**Prefill バッチ**（**`forward_prefill_gpu`** — 線形層 **hipBLAS GemmEx** + カスタム Attention/Norm 等）+ **Decode**（**`forward_gpu`**）。重み H2D: オフライン **`.fp16bin`** または **GGUF 行単位融合逆量子化**（**`upload_fp16_linear`** / **`upload_fp16_tensor_streaming`**）。起動ログ: **`Loading FP16 cache from …`** または **`Uploading weights (row dequant -> FP16)...`**。8 レイヤーごとに **`layer N/L uploaded: X.XX sec, X.XX GB/sec`**。**Prefill progress bar** と prefill / decode / total スループット要約を stderr に出力。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value ベンチ＋**`[vram_breakdown]`**（**`GpuVramProfile`** / **`model_vram_profile`**。**`make log.push`** がパース。推論区間のみ）。 |
| `qwen3-8b/gpu-rocm/fp16_cache.h` / `fp16_cache_io.c` | **FP16 オフラインキャッシュ** I/O（**`gpu-cuda/`** と同 API。**`FP16HostWeight`** の save/load、**`<model>.gguf.fp16`** パス生成、**`manifest`**（GGUF サイズ・mtime 検証）。 |
| `qwen3-8b/gpu-rocm/Makefile` | **`qwen3-rocm`** を **`hipcc`** で **`main.o` + `fp16_cache_io.o`** からリンク（**`-lhipblas -lrocblas -lstdc++`**。**g++** で libstdc++ ヘッダ／リンクパスを検出）。**`GPU_ARCH_DETECTED`** は **`rocminfo`** の最初の **`gfx*`**。**`gfx1152` / `gfx1153`** では rocBLAS 未同梱のため **`HIP_OFFLOAD_ARCH=gfx1151`** と **`HSA_OVERRIDE_GFX_VER=11.5.1`**（**`HSA_OVERRIDE_GFX_VER` が設定されているときのみ** **`HSA_OVERRIDE_GFX_VERSION` を export** — 空 export は **`gfx1030`** 等で HIP を壊す）。**`.DEFAULT_GOAL := run`**。**`ensure-model`**: **`$(MODEL)`** 未存在時 **`make -C .. model`**。**`detect-gpu-arch`**。**`FP16_CACHE_MANIFEST`**: **`$(MODEL).fp16/manifest`**。**`pack-cache`**: **`manifest`** 欠落または **MODEL / バイナリ** が新しいとき **`--pack-fp16-cache`**。**`build`**: バイナリ + **`manifest`**（MODEL 存在時）。**`run`**: **`ensure-model`** + **`qwen3-rocm`** + **`pack-cache`**。**`make log`** / **`make log.push`** / **`make wmma`**。詳細は **「環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS」**。 |
| `qwen3-8b/gpu-rocm/wmma_probe.c` | RDNA **gfx11 / gfx12** 向け **WMMA 校正用**最小 HIP プローブ。**gfx11**: **`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`**。**gfx12**: **`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`**。**`make wmma-probe`** の出力 **`wmma-probe`**。**`make wmma`** の **`llvm-objdump`** 検出器校正に使用。 |
| `qwen3-8b/gpu-rocm/scripts/check_wmma.sh` | **`make wmma`** から呼ばれる検証スクリプト。**`main.c` / `qwen3-rocm` に直接 WMMA が無いこと**、**`wmma-probe` に WMMA があること**（RDNA **gfx11 / gfx12**）、rocBLAS バンドル ISA、任意で実行時 hipBLAS 経路・**`rocprofv3 --kernel-trace`** カーネル trace。**`count_wmma_in_obj`**: HIP バイナリは **`--offloading`** で AMDGPU コードを抽出して **`v_wmma`** をカウント。 |
| `qwen3-8b/gpu-vulkan/main.c` | Vulkan compute 推論ホスト（**`gpu-cuda/main.c`** ベース）。GGUF・トークナイザ・FP16 重み H2D・生成ループ。**`gpu.h`** 経由で **`vk_kernels.c`** に forward を委譲。推論終了時 **`BENCH_LOG_FILE`** へ key=value ベンチ + 簡易 **`[vram_breakdown]`**（**`write_benchmark_log`**。**`vram_weights_linear` は未出力**）。 |
| `qwen3-8b/gpu-vulkan/gpu.h` | GPU API 境界（**`GpuConfig`** / **`GpuModel`** / **`gpu_forward`** / **`gpu_forward_prefill`** / **`GpuVramProfile`**）。**`gpu-cuda/gpu.h`** と同趣旨。 |
| `qwen3-8b/gpu-vulkan/vk_context.c` / `vk_context.h` | Vulkan インスタンス・物理デバイス・論理デバイス・compute キュー・コマンドプール。 |
| `qwen3-8b/gpu-vulkan/vk_alloc.c` / `vk_alloc.h` | デバイスメモリ（**`vk_malloc`** / **`vk_free`** / H2D / D2H）。バッファハンドルは整数 ID（ポインタ算術不可）。 |
| `qwen3-8b/gpu-vulkan/vk_pipeline.c` / `vk_pipeline.h` | compute パイプライン 18 種（**`VkPipeKind`**）。SPIR-V ロード・descriptor set layout 保持・**`vk_dispatch`**。descriptor pool **4096 sets**、dispatch 後 **`vkFreeDescriptorSets`**。 |
| `qwen3-8b/gpu-vulkan/vk_kernels.c` | **`gpu_model_create`** / **`gpu_forward`** / **`gpu_forward_prefill`** 実装。レイヤーごとに多数の compute dispatch。 |
| `qwen3-8b/gpu-vulkan/fp16_cache.h` / `fp16_cache_io.c` | **FP16 オフラインキャッシュ** I/O（**`gpu-rocm` / `gpu-cuda`** と同 API・同 **`<model>.gguf.fp16`** 形式）。 |
| `qwen3-8b/gpu-vulkan/shaders/*.comp` | GLSL compute 18 本（embedding、FP16 GEMV、RMSNorm、RoPE、SiLU、KV write、Flash Attention 等）。**`common.inc`** を include。**`make`** で **`.spv`** 生成。 |
| `qwen3-8b/gpu-vulkan/Makefile` | **`cc`** + **`-lvulkan`** で **`qwen3-vulkan`** をリンク。**`glslangValidator`** で **`shaders/*.spv`**。**`QWEN3_VK_SHADER_DIR`**（**`run`** / **`pack-cache`** で **`$(CURDIR)/shaders`** を設定）。**`GPU_VK`**: **`vulkaninfo --summary`** の **`deviceName=`** 行を **`awk -F'='`** でパース（旧 **`print $2`** は **`=` のみ**になり誤記録）。**`.DEFAULT_GOAL := run`**。**`ensure-model`**。**`FP16_CACHE_MANIFEST`** + **`pack-cache`**（ROCm / CUDA と同形式）。**`run`**: **`ensure-model`** + **`qwen3-vulkan`** + **`pack-cache`**。**`make log`** / **`make log.push`**（**`BENCH_LOG_FILE`** 読取、第 2 列 GPU 名）。**`clean`**: バイナリ・**`*.o`**・**`shaders/*.spv`**。 |
| `qwen3-8b/gpu-cuda/main.c` | NVIDIA CUDA 推論ホスト（**`BONSAI_FP4=0`** 時 FP16 線形 + オフラインキャッシュ。**`BONSAI_FP4=1`** 時 NVFP4 ロード・**`--pack-nvfp4-cache`** も本ファイル）。**`kernels.cu`** がデバイス forward。**`gpu.h`** が C/CUDA 境界（**`GpuVramProfile`** / **`gpu_model_vram_profile`** / **`gpu_get_device_desc`**）。重み H2D 時、8 レイヤーごとに **`layer N/L uploaded: X.XX sec, X.XX GB/sec`**。**Prefill progress bar** と prefill / decode / total スループット要約を stderr に出力。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value ベンチ＋**`[vram_breakdown]`**（**`make log.push`** が読取）。**`gpu-cuda-nvfp4`** も共有参照。 |
| `qwen3-8b/gpu-cuda/Makefile` | **`nvcc`** で **`qwen3-gpu-cuda`** をビルド。**`.DEFAULT_GOAL := run`**（**`ensure-model`** + **`pack-cache`** + 推論）。**`build`**: バイナリのみ。**`FP16_CACHE_MANIFEST`**: **`$(MODEL).fp16/manifest`**（**`pack-cache`** が **`build`** 依存で再生成）。**`nvidia-smi`** で **`CUDA_GENCODE`** / **`FA_BR`** を自動選択（Blackwell **12.x → `sm_120` + FA_BR=32**、未検出時 **`compute_86` PTX**）。**`.build_config.stamp`** で **`kernels.*.o`** の stale 回避。**`build.polarquant`** / **`run.polarquant`**、**`pack-cache`**、**`pq-test`**、**`fa-debug`**。**`BENCH_LOG_FILE`**。**`make log`** / **`make log.push`**（**`GPU_SM=sm_<ccap>`**）。NVFP4 関連は含まない。 |
| `qwen3-8b/gpu-cuda/fp16_cache.h` / `fp16_cache_io.c` | **FP16 オフラインキャッシュ** I/O。**`FP16HostWeight`** の save/load、**`<model>.gguf.fp16`** パス生成、**`manifest`**（GGUF サイズ・mtime 検証）。 |
| `qwen3-8b/gpu-cuda/kernels.cu` | FP16 GEMV・Flash Attention（decode / prefill）・RoPE 等。PolarQuant 時は KV を **`PQBlock`** 経路に切替。**`BONSAI_FP4`** 時は **`fp4_qwen3_mm`** へ委譲。**`fa_debug.h`** フック（**`--fa-debug`**）。 |
| `qwen3-8b/gpu-cuda/fa_debug.c` / `fa_debug.h` | **Flash Attention / KV / prefill 層トレース**（stderr **`FA_DEBUG`**）。**`make fa-debug`**（**`gpu-cuda`** / **`gpu-cuda-nvfp4`**）。 |
| `qwen3-8b/gpu-cuda/polarquant.cu` / `polarquant.h` | **PolarQuant-R** KV キャッシュ圧縮。ホスト側コードブック初期化（Lloyd-Max L=2〜4）、デバイス **`PQState`**、KV 書き込み API。 |
| `qwen3-8b/gpu-cuda/polarquant_kernels.cuh` | デバイス側 encode/decode（ランダム符号 + FWHT-128、L=4 再帰 polar 量子化、**`PQBlock`** 8 bytes × 8 blocks）。 |
| `qwen3-8b/gpu-cuda/polarquant_verify.cu` | **`make pq-test`** 用のラウンドトリップ検証（推論バイナリには未リンク）。 |
| `qwen3-8b/gpu-cuda-nvfp4/Makefile` | **`nvcc`** で **`qwen3-gpu-cuda-nvfp4`** をビルド。**`.DEFAULT_GOAL := run`**（**`sm_120a`** + **`BONSAI_FP4=1`** + **`FA_BR=32`**）。**`ensure-model`**。**`FP4_CACHE_MANIFEST`**: **`$(MODEL).nvfp4/manifest`**。**`build`**: バイナリのみ。**`run`**: **`ensure-model`** + **`pack-cache`** + 推論。**`build.polarquant`** / **`run.polarquant`**、**`blackwell`**、**`cutlass`**、**`fp4-test`**、**`sfa-verify`**、**`fa-debug`**、**`pq-test`**。**`BENCH_LOG_FILE`**。**`make log`** / **`make log.push`**（**`-t $(BENCH_TEMP) -s $(BENCH_SEED)`**）。**`main.c` / `kernels.cu` / `gpu.h` / `polarquant.*` / `fa_debug.*`** は **`../gpu-cuda/`** を参照。 |
| `qwen3-8b/gpu-cuda-nvfp4/DEBUG.md` | NVFP4 異常出力の調査・修正ログ（ベースコミット **`433319eb31c3c992536afb5c9a3717084ea5d137`**、層トレース、コマンド早見表）。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_cache.h` / `fp4_cache_io.c` | **NVFP4 オフラインキャッシュ** I/O。**`FP4_CACHE_VERSION=2`**（SFB + LUT レイアウト）。**`FP4HostWeight`** の save/load、**`<model>.gguf.nvfp4`** パス生成、**`manifest`**（GGUF サイズ・mtime 検証）。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_gemm.cu` / `fp4_gemm.h` | CUTLASS **NVFP4** GEMM（推論 **`fp4_gemm_run_cached`**、**`fp4-test`**、起動 smoke）。活性量子化に **`FP4_QUANT_MAX_ABS=1024`**。**`fp4_quantize_weights`**（GPU、**`FP4_WEIGHT_SFB_LAYOUT_M=128`** + **`d_sf_lut`**）。**`fp4_host_weight_build`** / **`fp4_weight_cache_upload`**。**`fp4_gemm_vram_bytes`**。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_qwen3.cu` / `fp4_qwen3.h` | GGUF 行逆量子化から **NVFP4 キャッシュ**。**`fp4_qwen3_mm`**（F32 活性 → BF16 pad → GEMM → F32 出力）。**`fp4_qwen3_init`** で **128³** smoke。**`fp4_qwen3_vram_bytes`**（BF16 活性／出力スクラッチ）。 |
| `qwen3-8b/gpu-cuda-nvfp4/fp4_verify.cu` | **`make fp4-test`**（経路一致・**`run_batch_row_parity`**・**`run_extreme_act`**）。 |
| `qwen3-8b/gpu-cuda-nvfp4/sfa_index_verify.cpp` | **`make sfa-verify`** — **`compute_sf_index`** と CUTLASS SFA/SFB layout の一致。 |
| `qwen3-8b/gpu-cuda-nvfp4/third_party/cutlass/` | **`make cutlass` / `make blackwell`** で clone される CUTLASS **v4.5.0**（**`CUTLASS_TAG`**）。リポジトリ同梱ではない。タグ不一致時は **`make cutlass`** が再 clone する。 |
| `qwen3-8b/xdna2/main.c` | AMD Ryzen AI（XDNA2）NPU。**mmap ウェイト + GEMV 毎 BF16 スクラッチ**・`amdxdna` ioctl 直叩き。**`--xdna-status` / `-X`** で制御コード環境の軽量診断。 |
| `qwen3-8b/xdna2-bfp16/main.c` | **`xdna2/main.c` と同一の IOCTL／チャンク BF16 GEMV（枠組み）。密行列レイアウトの重みをロード時に BFPX 化しホストのみ保持、mmap は変換完了後に解放。** |
| `qwen3-8b/Makefile` | **`model` のみ**。既定 **`MODEL`** の GGUF を **`gguf.txt`** の URL から **`wget`** し **`$(MODEL).sha256sum`** で検証（**既存ファイルが検証済みならダウンロードをスキップ**）。失敗時は破損ファイルを削除。ビルド・実行・クリーンは各サブディレクトリの **`Makefile`** で行う。 |
| `qwen3-8b/cpu/Makefile` ほか（各経路直下） | **`cpu` / `cpu-multicore` / `cpu-blas` / `xdna2` / `xdna2-bfp16`** 等。**.DEFAULT_GOAL := run`**（引数なし **`make`** で推論）。**`ensure-model`**: **`$(MODEL)`** 未存在時 **`make -C .. model`**。**`run`**: **`build`**（またはバイナリターゲット）+ **`ensure-model`** + 推論。**`make build`** / **`clean`** も可。出力バイナリは **`cpu/qwen3-cpu`** のように経路直下に生成。既定 **`MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**（サブディレクトリからの相対パス）。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog.md` | 変更履歴。 |
| `qwen3-8b/xdna2/xdna-gemv/README.md` | **NPU GEMV 用 ctrlcode まわりの入口**。**`kernels/`**・**`toolchain/`**・スタブ再生成スクリプトへの導線。 |
| `qwen3-8b/xdna2/xdna-gemv/kernels/Makefile` | **`bf16-gemv-*.bin` を `curl`/`wget` で一括取得**。既定の **`XDNA_GEMV_BIN_URL_BASE`** を Makefile 内に記述（上書き可）。詳細は **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md` §8.1**。 |
| `qwen3-8b/xdna2/xdna-gemv/kernels/README.md` | **ctrlcode**（**`bf16-gemv-<n>x<d>.bin`**）と GEMV 形状、ホスト／NPU 分担、用語ミニ辞典、スタブ **`GQF3XDNA`** と **`--xdna-status`**、8B 対応表、差し替え・再生成。**§3 で ROCm/HIP の GPU カーネルとの対比と「ユーザーが HIP のように書けるか」** の整理を含む入門。 |
| `qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py` | **`qwen3-8b/xdna2/xdna-gemv/kernels/`** のスタブ `.bin` を生成（リポジトリルートから **`python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels`**）。 |
| `qwen3-8b/xdna2/xdna-gemv/toolchain/README.md` | **NPU 用 `bf16-gemv-*.bin` を自前生成する手引き**（Xilinx **mlir-aie**／**AMD IRON**／**Peano**／**`aiecc`** の公式手順に沿ったコマンド、Qwen-VL-8B と **IRON `GEMV(M,K)`** の対応、`--aie-generate-npu-insts` と `qwen3-xdna2` 統合時の注意）。**冒頭**で Linux カーネル文書 **AMD NPU** における **`ctrlcode`** と本リポジトリ実装の対応、**`qwen3-xdna2`（XRT 非依存）と公式サンプル（XRT 経由）の違い**を整理。**東京科学大学（2026年現在の名称。旧・東京工業大学）ACRi** ルーム公開の日本語チュートリアル（外部リンク・vadd 題材）への導線あり。本文は日本語（です・ます調）。 |
| `.gitignore` | ビルド生成バイナリ・**`*.gguf`**・オフラインキャッシュ **`*.gguf.nvfp4/`** / **`*.gguf.fp16/`** 等に加え、**`gpu-rocm/*.o`**・**`gpu-vulkan/*.o`**・**`gpu-vulkan/shaders/*.spv`**・**`gpu-cuda/*.o`** 等のオブジェクト、**Python の `__pycache__/` と `*.py[cod]`** を除外。 |
| `qwen3-8b/gguf.txt` | 既定 GGUF の取得元 URL 参照。Hugging Face の `blob/main` URL を `resolve/main` に置換して **`wget`** する（**`make model`** が同処理を実行）。 |
| `qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum` | 既定 GGUF の SHA256 参照（**`make model`** および手動 **`sha256sum -c`** 用）。 |

### 生成バイナリと Make ターゲット（`qwen3-8b/`）

**GGUF 取得**は **`qwen3-8b/`** の **`make model`**（各サブディレクトリの **`make run`** / 引数なし **`make`** は **`ensure-model`** 経由で同処理を自動委譲）。**GPU 4 経路**（**`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`**）の **`make run`** は **`manifest`** 欠落・更新時に **`pack-cache`** も自動実行。**ビルド・実行**は各サブディレクトリに移動して **`make`**（= **`run`**）または **`make build`**（バイナリのみ）する。ROCm 版の ISA は **`gpu-rocm/Makefile`** が **`rocminfo`** から **`GPU_ARCH_DETECTED`** を取得（**`gfx1152` 等は `gfx1151` + HSA オーバーライド** — **「環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS」**）。手動で先に取得する場合は **「モデル参照」** を参照。

#### トップ `qwen3-8b/Makefile`

| ターゲット | 出力 | 備考 |
|------------|------|------|
| **`model`** | **`$(MODEL)`**（既定 **`Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**） | **`gguf.txt`** → **`wget`**。**`$(MODEL).sha256sum`** で **`sha256sum --check`**。既存かつ検証 OK ならスキップ |

#### 各サブディレクトリの `Makefile`

**共通ターゲット**（**`cpu`** / **`cpu-multicore`** / **`cpu-blas`** / **`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`** / **`xdna2`** / **`xdna2-bfp16`**）:

| ターゲット | 意味 |
|------------|------|
| **（既定）`run`** | **`.DEFAULT_GOAL := run`** — 引数なし **`make`** で推論 |
| **`ensure-model`** | **`$(MODEL)`** が無ければ **`make -C .. model`**（親 **`qwen3-8b/`** へ委譲） |
| **`pack-cache`** | **GPU 4 経路**: **`$(MODEL).fp16/manifest`** または **`$(MODEL).nvfp4/manifest`** を生成（**`manifest`** 欠落・**MODEL / バイナリ** 更新時に再実行） |
| **`run`** | **`ensure-model`** + バイナリビルド +（**GPU 4 経路**）**`pack-cache`** + 推論 |
| **`build`** | バイナリのみ（推論なし）。**`gpu-rocm`** のみ MODEL 存在時 **`manifest`** まで（**`gpu-cuda`** / **`gpu-cuda-nvfp4`** の **`build`** はバイナリのみ） |

| ディレクトリ | `make build` 出力 | ソース |
|--------------|-------------------|--------|
| **`cpu/`** | `qwen3-cpu` | `cpu/main.c` |
| **`cpu-multicore/`** | `qwen3-cpu-omp` | `cpu-multicore/main.c`（`-fopenmp`、`OMP_NUM_THREADS`） |
| **`cpu-blas/`** | `qwen3-cpu-blas` | `cpu-blas/main.c`（`-fopenmp`、**`-march=native`**、**`-lopenblas`**。OpenBLAS は実行時 1 スレッド固定） |
| **`gpu-rocm/`** | `qwen3-rocm` | `gpu-rocm/main.c` + **`fp16_cache_io.c`**。**`make run`** が **`pack-cache`** を自動実行（**`make build`** も MODEL 存在時 **`manifest`** 生成） |
| **`gpu-vulkan/`** | `qwen3-vulkan` | **`main.c`** + **`vk_*.c`** + **`shaders/*.spv`**。**`libvulkan-dev`** + **`glslang-tools`**。**`QWEN3_VK_SHADER_DIR`**。**`make run`** が FP16 **`pack-cache`** を自動実行 |
| **`gpu-cuda/`** | `qwen3-gpu-cuda` | FP16。**`nvidia-smi`** で **`CUDA_GENCODE`** / **`FA_BR`** 自動選択。**`build.polarquant`** で PolarQuant-R KV。**`make run`** が FP16 **`pack-cache`** を自動実行 |
| **`gpu-cuda-nvfp4/`** | `qwen3-gpu-cuda-nvfp4` | **`sm_120a`** + **`BONSAI_FP4=1`**。**`build.polarquant`** / **`blackwell`** / **`cutlass`** / **`fp4-test`** / **`sfa-verify`** / **`fa-debug`**。**`make run`** が NVFP4 **`pack-cache`** を自動実行 |
| **`xdna2/`** | `qwen3-xdna2` | `xdna2/main.c`（`-fopenmp`。`amdxdna` カーネルモジュール） |
| **`xdna2-bfp16/`** | `qwen3-xdna2-bfpx` | `xdna2-bfp16/main.c`（BFPX ホスト重み） |

```bash
cd qwen3-8b
make model                   # 既定 GGUF を gguf.txt から取得し .sha256sum で検証（済みならスキップ）
cd cpu && make               # = make run（ensure-model → 未取得なら上記 model を自動実行）
cd cpu && make build
cd ../cpu-multicore && make build
cd ../cpu-blas && make build               # libopenblas-dev 等が必要
cd ../gpu-rocm && make                     # = make run（ensure-model + pack-cache + 推論。初回は FP16 cache 生成で時間がかかる）
cd ../gpu-rocm && make build               # hipcc・ROCm 必須（GPU_ARCH は rocminfo 自動検出。MODEL あれば manifest も）
cd ../gpu-vulkan && make                   # ensure-model + pack-cache + 推論
cd ../gpu-vulkan && make build             # libvulkan + glslangValidator（shaders/*.spv 生成）
cd ../gpu-cuda && make                     # ensure-model + FP16 pack-cache + 推論
cd ../gpu-cuda && make build               # 汎用 NVIDIA GPU（FP16・PTX 可。バイナリのみ）
cd ../gpu-cuda && make build.polarquant    # PolarQuant-R KV キャッシュ（FP16 線形重み・任意 GPU）
cd ../gpu-cuda-nvfp4 && make                 # ensure-model + NVFP4 pack-cache + 推論
cd ../gpu-cuda-nvfp4 && make build         # Blackwell + NVFP4（要 CUDA 13 + CUTLASS。バイナリのみ）
cd ../gpu-cuda-nvfp4 && make fp4-test      # CUTLASS GEMM 単体・バッチ行・極大活性
cd ../gpu-cuda-nvfp4 && make sfa-verify    # SFA/SFB 索引検証
cd ../gpu-cuda-nvfp4 && make fa-debug      # FA 診断（FP16 vs NVFP4、Hello）
cd ../gpu-cuda-nvfp4 && make build.polarquant   # NVFP4 線形 + PolarQuant-R KV（Blackwell）
cd ../gpu-cuda-nvfp4 && make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
cd ../xdna2 && make build                  # Linux >= 6.10 + amdxdna カーネルモジュール（XRT 不要）
cd ../xdna2-bfp16 && make build            # BFPX ホスト重み版バイナリ
# xdna2/xdna-gemv/kernels スタブ再生成（実 NPU ctrlcode ではない）:
python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels
cd cpu-multicore
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
cd ../cpu-blas
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 4
```

**CPU（IQ 混在 8B）**はブロック単位逆量子化のため **非常に遅くなり得る**。**`cpu-blas`** は F32 経路の OpenBLAS 化に加え、量子化 GEMV を **Q8_K + 整数内積**に置き換えるため **`cpu-multicore` より速くなることが多い**（README 本文の推奨 CPU 経路）。実用スループットは **ROCm 版**（AMD GPU）または **`gpu-cuda` / `gpu-cuda-nvfp4`**（NVIDIA GPU）を優先する想定である（いずれも README **付録**）。**`gpu-vulkan`** は **ROCm/CUDA 不要の検証用** — 正しく推論できることを確認済みだが **スループットは `gpu-rocm` より大幅に低い**（dispatch オーバーヘッド・GEMM 未実装）。

## ビルドと実行

### 共通（`qwen3-8b/Makefile` — モデル取得のみ）

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `MODEL` | GGUF ファイル名（`qwen3-8b/` 直下） | `Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |

各バリアントの **`Makefile`** では **`MODEL`**（既定 **`../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**）・**`PROMPT`** 等を **`make run`**（引数なし **`make`**）に渡す。**`ensure-model`** が **`$(MODEL)`** 未存在時に親の **`make model`** を呼ぶ。**GPU 4 経路**の **`run`** は **`pack-cache`** も依存し、**`manifest`** 欠落・更新時にオフラインキャッシュを自動生成する。ROCm / Vulkan / CUDA / CPU 向けのコンパイラ変数は各サブディレクトリの **`Makefile`** を参照。

| 変数（代表） | 意味 | 既定例 |
|------|------|--------|
| `CC` | C コンパイラ（CPU / OpenMP / XDNA） | `cc` |
| `CFLAGS` | C コンパイルフラグ | `-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter` |
| `LDFLAGS` | リンクフラグ・ライブラリ | `-lm` |
| `ROCM` | ROCm ルート（**`gpu-rocm/`**） | `/opt/rocm` |
| `HIPCC` | HIP コンパイラ | `$(ROCM)/bin/hipcc` |
| `GPU_ARCH` / `GPU_ARCH_DETECTED` | **`rocminfo` の `Name: gfx*`**（検出値は **`GPU_ARCH_DETECTED`**。**`GPU_ARCH`** は上書き可） | （自動。未検出時はビルド失敗） |
| `HIP_OFFLOAD_ARCH` | **`hipcc --offload-arch=`** の実際の値（**`gfx1152`/`gfx1153` 時は `gfx1151` にマップ**） | 検出 ISA または **`gfx1151`** |
| `HSA_OVERRIDE_GFX_VER` | Makefile 内部変数。**`gfx1152`/`gfx1153` 検出時のみ `11.5.1`**。非空のときだけ **`export HSA_OVERRIDE_GFX_VERSION`**（空 export は **`gfx1030`** 等で HIP 初期化を壊す） | （該当 GPU のみ設定） |
| `HSA_OVERRIDE_GFX_VERSION` | 上記 **`HSA_OVERRIDE_GFX_VER`** の export 名。**`make run`** の **`RUN_ENV`** でも同値を付与 | （該当 GPU のみ設定） |
| `XDNA_INCS` | `<drm/drm.h>` が標準外にあるときの `-I…`（**`xdna2/`** / **`xdna2-bfp16/`**） | 未定義（空で可） |
| `PROMPT` | ユーザプロンプト（**`make run`**） | `Hello, how are you?` |

### CPU

```bash
cd qwen3-8b/cpu
make build
make run PROMPT="質問" MODEL=../path/to/model.gguf
```

### CPU（OpenMP）

```bash
cd qwen3-8b/cpu-multicore
make build
OMP_NUM_THREADS=8 ./qwen3-cpu-omp ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hi" -n 8
```

### CPU（OpenMP + OpenBLAS）

**OpenBLAS**（`libopenblas-dev` 等）と OpenMP ランタイムが必要。**`pkg-config openblas`** が使える環境では **`cpu-blas/Makefile`** が include / link フラグを自動取得する。ヘッダが標準パスに無い場合（Debian/Ubuntu の pthread ビルド等）は **`CPPFLAGS`** で指定する。

```bash
cd qwen3-8b/cpu-blas && make openblas   # libopenblas-dev / libgomp1 を apt 導入
cd qwen3-8b/cpu-blas
make build
OMP_NUM_THREADS=8 ./qwen3-cpu-blas ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hi" -n 8
# ヘッダパスが必要な例:
make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
```

**`openblas_set_num_threads(1)`** で OpenBLAS 側は 1 スレッド固定（OpenMP との二重並列化を避ける）。**`-ffast-math`** は IQ2_S / IQ3_S 量子化内積で数値が崩れるため **`cpu-blas/Makefile`** では無効。

### ROCm

**`GPU_ARCH_DETECTED`** はビルド前に **`gpu-rocm/Makefile`** が **`$(ROCM)/bin/rocminfo`** から自動検出する（**`make -C gpu-rocm detect-gpu-arch`** で **検出 ISA** と **ビルド offload ISA** を表示）。検出に失敗する環境や別 ISA 向けビルドでは **`GPU_ARCH=gfx1100`** 等を明示する。ビルドは **`-lhipblas -lrocblas -lstdc++`** をリンクする（Prefill 線形層の **hipBLAS GemmEx** 用。**`fp16_cache_io.o`** のため **g++ / libstdc++-dev** が必要）。**`make build`**: **`qwen3-rocm`** + **`$(MODEL).fp16/manifest`**（**`MODEL`** が存在する場合）。**`make run`**（引数なし **`make`**）: **`ensure-model`** + **`qwen3-rocm`** + **`pack-cache`**（**`manifest`** 欠落・**MODEL / バイナリ** 更新時に **`--pack-fp16-cache`**）。**`make pack-cache`**: 上記 **`manifest`** ターゲットの明示呼び出し。

**Ryzen AI 5 340（Radeon 840M / `gfx1152`）** など **rocBLAS が `gfx1152` 用 Tensile を同梱していない GPU** では、Makefile が **`HIP_OFFLOAD_ARCH=gfx1151`** と **`HSA_OVERRIDE_GFX_VERSION=11.5.1`** を自動適用する（詳細は直後の専用節）。**ROCm を 7.2.1 に上げること自体はこの GPU 専用の必須条件ではない**（7.1.x でも同様のワークアラウンドで動作しうる）。

```bash
cd qwen3-8b/gpu-rocm
make build
make run PROMPT="Hello"
# オフライン FP16 キャッシュ（2 回目以降の起動を高速化）:
make pack-cache
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# 手動上書きの例:
make build GPU_ARCH=gfx1100
make detect-gpu-arch
# ベンチマーク履歴（gpu-rocm/ 単体 Makefile）:
cd gpu-rocm
make log.push              # 既定 BENCH_PROMPT (~128 tok) / BENCH_N=128 / -t 0
make log                   # Makefile 内 BENCH_LOG を表表示
make log.push BENCH_N=64   # 生成トークン数など上書き可
# WMMA 利用状況（hipBLAS 経路・自前コードに WMMA が無いことの確認）:
cd gpu-rocm
make wmma                           # build + wmma-probe + check_wmma.sh（MODEL 要）
make wmma WMMA_SKIP_RUN=1           # 静的チェックのみ（MODEL 不要）
make wmma WMMA_SKIP_ROCPROF=0       # rocprofv3 カーネル ISA も試行
make wmma-probe                     # 校正用 wmma-probe のみビルド
```

#### 環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS

本節は **AMD Ryzen AI 系 APU の内蔵 Radeon（RDNA 3.5）** で **`qwen3-rocm`** を動かす際の、**ROCm / rocBLAS と実機 ISA のずれ**をまとめる。検証環境の一例は **Ryzen AI 5 340 + Radeon 840M（`gcnArchName: gfx1152`）**。

##### ハードウェアと ISA の対応（参考）

| 製品例 | 内蔵 GPU | `rocminfo` の `Name`（代表） |
|--------|----------|------------------------------|
| Ryzen AI 9 HX 370 等 | Radeon 890M 系 | **`gfx1150`**（Strix Point） |
| 一部 Halo 系 | — | **`gfx1151`** |
| **Ryzen AI 5 340** 等 | **Radeon 840M** | **`gfx1152`**（Krackan Point） |
| 今後の APU | — | **`gfx1153`**（同様に rocBLAS 未同梱の可能性） |

**`qwen3-rocm` の自前 HIP カーネル**（Attention・Norm・GEMV 等）は **`hipcc --offload-arch=`** でコンパイルされる。**Prefill の線形層 GEMM** だけは **hipBLAS `GemmEx` → rocBLAS（Tensile 事前ビルドカーネル）** に依存する。この二系統の「アーキテクチャの見え方」がずれると、モデルロード後・Prefill 0% で落ちる。

##### 典型エラー（ROCm 7.1.x / 7.2.x 共通）

Prefill 開始直後（**`Prefill linear: hipBLAS GemmEx`** のあと）に次が出て **`Aborted (core dumped)`** することがある。

```text
rocBLAS error: Cannot read /opt/rocm/lib/rocblas/library/TensileLibrary.dat:
  No such file or directory for GPU arch : gfx1152
 List of available TensileLibrary Files :
  ... TensileLibrary_lazy_gfx1150.dat
  ... TensileLibrary_lazy_gfx1151.dat
  （gfx1152 はリストに無い）
```

**原因の整理:**

| 層 | 内容 |
|----|------|
| 直接原因 | **`/opt/rocm/lib/rocblas/library/`** に **`TensileLibrary_lazy_gfx1152.dat`**（および関連 **`Kernels.so-000-gfx1152.hsaco`** 等）が **公式 DEB に無い** |
| トリガ | Prefill 線形層の初回 **`hipblasGemmEx`** が rocBLAS に GPU arch **`gfx1152`** を渡す |
| 本リポジトリ | **`main.c` のバグではない**（重み H2D・トークナイズは成功しうる） |
| ROCm バージョン | **7.1.1 → 7.2.1 へ上げても**（2026-06 時点の apt 同梱では）**`gfx1152` 用 Tensile は依然未収録**。上げただけでは直らない |

AMD 側では **`gfx1152` / `gfx1153` の rocBLAS 同梱**が開発中（[ROCm/TheRock#2310](https://github.com/ROCm/TheRock/issues/2310) 等）。将来 **`TensileLibrary_lazy_gfx1152.dat`** が入った ROCm では、下記ワークアラウンドを外せる可能性がある。

##### Makefile の自動ワークアラウンド（`gpu-rocm/Makefile`）

**`GPU_ARCH_DETECTED`** が **`gfx1152`** または **`gfx1153`** のとき:

| 変数 | 値 | 役割 |
|------|-----|------|
| **`HIP_OFFLOAD_ARCH`** | **`gfx1151`** | **`hipcc --offload-arch=`** — 自前 HIP カーネルを **同梱されている最寄り ISA** でコンパイル |
| **`HSA_OVERRIDE_GFX_VER`** | **`11.5.1`** | Makefile 内部。**非空のときのみ** **`export HSA_OVERRIDE_GFX_VERSION`** |
| **`HSA_OVERRIDE_GFX_VERSION`** | **`11.5.1`**（export 時） | ランタイムで HSA が GPU を **`gfx1151`** として報告し、rocBLAS が **`TensileLibrary_lazy_gfx1151.dat`** を選択 |
| **`RUN_ENV`** | 上記を **`make run`** / **`log.push`** 等に付与 | 手動 export を忘れないため |

**`gfx1152` / `gfx1153` 以外**（**`gfx1030`** / **`gfx1100`** / **`gfx1201`** 等）では **`HSA_OVERRIDE_GFX_VER` は未設定**のため、**`HSA_OVERRIDE_GFX_VERSION` は export も **`RUN_ENV` も付与されない**。以前のように **空の `HSA_OVERRIDE_GFX_VERSION` を常 export** すると、オーバーライド不要な GPU で HIP が失敗しうる。

**`make build`** / **`make detect-gpu-arch`** のログ例:

```text
  Detected GPU arch: gfx1152
  Build offload arch: gfx1151
  HSA_OVERRIDE_GFX_VERSION: 11.5.1
```

**`make run`** 成功時の起動ログ例（Ryzen AI 5 340）:

```text
ROCm HIP device 0: AMD Radeon 840M Graphics (gcnArchName: gfx1151)
Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)
Prefill complete: 25 tokens in ~1.9s (~13 tok/s)   # 短プロンプト・環境依存
```

##### ROCm 7.2.1 へのアップグレードは必須か

**いいえ。** 本問題は **「ROCm が古い」** より **「rocBLAS の同梱アーキテクチャ一覧に `gfx1152` が無い」** ことが本質である。

- **7.2.1 に上げる利点**: バグ修正・他 GPU 向け更新・ツールチェーン新しさ（任意）
- **7.2.1 だけでは不十分**: **`gfx1150` / `gfx1151`** の lazy Tensile のみ同梱という状態は 7.1.1 でも 7.2.1 でも同様
- **本リポジトリで実際に効いた対処**: **`HIP_OFFLOAD_ARCH=gfx1151` + `HSA_OVERRIDE_GFX_VERSION=11.5.1`**（Makefile 自動化）

##### 避けるべき対処（検証で失敗した例）

| 対処 | 結果 |
|------|------|
| **`gfx1151` の `.dat` / `.hsaco` を `gfx1152` 名でコピー／シンボリックリンクのみ** | Tensile の読み込みは進むが **`hipBLAS error: 6`**（**`HIPBLAS_STATUS_INTERNAL_ERROR`**）になりうる |
| **`HSA_OVERRIDE_GFX_VERSION=11.5.1` のみ**（バイナリは **`--offload-arch=gfx1152`** のまま） | **Segmentation fault**（自前カーネル ISA とランタイム報告の不一致） |
| **別 ISA への無根拠な `HSA_OVERRIDE`**（例: **`11.0.2`** で **`gfx1102` 扱い） | 動く報告はあるが **非推奨**（本プロジェクトでは **RDNA 3.5 最寄りの `gfx1151`** を採用） |

##### 手動での再現・上書き

Makefile を使わない場合の最小セット:

```bash
cd qwen3-8b/gpu-rocm
make clean build GPU_ARCH=gfx1152   # 内部で HIP_OFFLOAD_ARCH=gfx1151 になる
export HSA_OVERRIDE_GFX_VERSION=11.5.1
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hello" -n 16
```

**`rocminfo`** で **`gfx1150`** が報告される機種（例: Ryzen AI 9 HX 370）では、通常 **オーバーライド不要**（**`TensileLibrary_lazy_gfx1150.dat`** が同梱されている前提）。

##### 確認コマンド

```bash
# 検出 ISA
/opt/rocm/bin/rocminfo | awk '/^  Name:/ { print }' | head -5

# rocBLAS が同梱する gfx115x Tensile（gfx1152 が無いことを確認）
ls /opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx115*.dat

# 本リポジトリの検出・マップ
cd qwen3-8b/gpu-rocm && make detect-gpu-arch
```

起動時に **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** が出れば Prefill 線形層は hipBLAS 経路が有効。Prefill / Decode の分離と 3 段階の改善経緯は **「ROCm Prefill 高速化の詳細（3 段階）」** を参照。

**WMMA と rocBLAS**: **WMMA が実際に使われるかは rocBLAS のカーネル選択次第**であり、**`qwen3-rocm` 側で ON/OFF する手段はない**。Prefill GEMM は hipBLAS **`GemmEx`** → rocBLAS に渡され、行列サイズ・**`GPU_ARCH`**・精度等で内部カーネルが選ばれる。WMMA 命令を含むカーネルが選ばれることも、**`v_fmac_f32`** 等 FMAC 系のみのこともある（**`gfx1201`** では **`make wmma`** の lib チェックで WMMA 0 件 WARN があり得る）。

**`make wmma`** は Prefill が **hipBLAS / rocBLAS ライブラリ経由**であること（**`main.c` に直接 WMMA を書いていない**こと）を **`llvm-objdump`** で検証する。**`qwen3-rocm` バイナリ自体に WMMA 命令が無い**のが正常。**rocBLAS** バンドル ISA（**`Kernels.so-000-$(GPU_ARCH).hsaco`**）の WMMA 有無は静的参考。**`wmma-probe`** は RDNA **gfx11 / gfx12** 向け校正用（検出器が **`v_wmma`** を拾えることの確認）。実行時 ISA は **`WMMA_SKIP_ROCPROF=0`** または下記手動 **`rocprofv3`** で確認。詳細は **`scripts/check_wmma.sh`** および **`README.md`** の **「WMMA 利用状況の確認」** を参照。

#### 手動 `rocprofv3 --kernel-trace`（Prefill 実行時 ISA）

**`make wmma WMMA_SKIP_ROCPROF=0`** と同趣旨。Prefill 中にロードされた **`.hsaco` / `.co`** を trace し **`v_wmma`** を数える。

```bash
cd qwen3-8b/gpu-rocm
MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
TRACE_DIR=/tmp/qwen3-wmma-trace
LLVM_OBJDUMP=${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}

mkdir -p "$TRACE_DIR"
rocprofv3 --kernel-trace -d "$TRACE_DIR" -f csv -- \
  ./qwen3-rocm "$MODEL" -p "Hello" -n 0 -t 0 -s 42

find "$TRACE_DIR" -type f \( -name '*.hsaco' -o -name '*.co' \) -print0 | while IFS= read -r -d '' f; do
  n=$("$LLVM_OBJDUMP" -d "$f" 2>/dev/null | grep -ciE '\tv_wmma|\bv_wmma_' || true)
  printf '%4d  %s\n' "$n" "$f"
done
```

合計 **0** なら当該 Prefill で WMMA カーネル未使用の可能性が高い。**1 以上**なら trace 中に WMMA カーネルが実行された。**`.hsaco`** が無い場合は **`rocprofv3`** / GPU アクセス / **`libdw.so`** 等を確認。

| 変数（`gpu-rocm/Makefile`） | 意味 | 既定例 |
|-----------------------------|------|--------|
| `BENCH_PROMPT` | **`log.push`** のプロンプト文字列（~128 token 想定） | 英語長文（Makefile 内） |
| `BENCH_N` | **`log.push`** の **`-n`**（生成トークン上限） | `128` |
| `BENCH_SEED` | **`log.push`** の **`-s`** | `42` |
| `BENCH_LOG_FILE` | 推論終了時に **`qwen3-rocm`** が書き込むベンチログ（**`log.push`** が読み取り） | `/tmp/benchmark.log` |
| `WMMA_PROMPT` | **`make wmma`** 実行時プロンプト | `Hello` |
| `WMMA_N` | **`make wmma`** の **`-n`**（0 で prefill のみ短時間） | `0` |
| `WMMA_SKIP_RUN` | **`1`** で実行時チェック省略（静的 ISA のみ） | `0` |
| `WMMA_SKIP_ROCPROF` | **`0`** で **`rocprofv3 --kernel-trace`** を試行 | `1` |
| `LLVM_OBJDUMP` | WMMA 命令カウント用 **`llvm-objdump`** | **`$(ROCM)/llvm/bin/llvm-objdump`** |

**`log.push`** の 1 行形式（パイ区切り）: **`YYYY-MM-DDTHH:MM:SS|GPU_ARCH|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**（**`date +%Y-%m-%dT%H:%M:%S`**、タイムゾーンオフセットなし）。**`make log`** は上記を表表示（列幅調整。旧エントリに **`+00:00`** 等が付いていても表示時に除去）。スループットは推論のみ（prefill+decode）。モデル重み H2D は含まない。**VRAM 内訳は `BENCH_LOG_FILE` の `[vram_breakdown]` を参照**（**`Makefile` の `BENCH_LOG` 行には含めない**）。

**ベンチログファイル**（**`qwen3-rocm`** が推論終了時に上書き。環境変数 **`BENCH_LOG_FILE`** でパス変更可。既定 **`/tmp/benchmark.log`**）:

| キー | 意味 |
|------|------|
| `timestamp` | ローカル日時 **`YYYY-MM-DDTHH:MM:SS`** |
| `hostname` | **`gethostname`** |
| `model` | GGUF パス |
| `gpu` | **`hipGetDeviceProperties`** の device 名 + **`gcnArchName`** |
| `max_new` / `temperature` / `top_p` / `seed` / `max_seq` | CLI 相当 |
| `prompt_tokens` / `gen_tokens` | プロンプト長・生成トークン数 |
| `prefill_sec` / `decode_sec` / `total_sec` | 区間秒数 |
| `prefill_tps` / `decode_tps` / `total_tps` | 推論スループット（H2D 除外） |
| **`vram_total`** / **`vram_total_mib`** | **`GpuVramProfile.total_bytes`**（推定合計） |
| **`vram_device_used`** / **`vram_device_total`** | **`hipMemGetInfo`**（各 **bytes** + **`_mib`**） |
| **`[vram_breakdown]`** | **`vram_weights_embd`**, **`vram_weights_f32_norm`**, **`vram_weights_linear`**, **`vram_kv_cache`**, **`vram_decode_activations`**, **`vram_prefill_batch`**（**`d_scratch_f16`** 含む） |
| `--- prompt ---` … `--- end prompt ---` | プロンプト全文 |

**参考計測**（**`Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**・**`gfx1201`**・**`-l 512`**・**`make log.push`** **2026-05-29 08:12**、132+16 トークン）:

| 項目 | bytes | MiB | 備考 |
|---|---:|---:|---|
| **`vram_total`** | 16,634,527,232 | **15863.92** | 下記カテゴリの合計 |
| **`vram_device_used`** | 16,989,028,352 | **16202.00** | **`hipMemGetInfo`** |
| **`vram_device_total`** | 34,208,743,424 | **32624.00** | GPU 全体 VRAM |
| `vram_weights_embd` | 1,244,659,712 | **1187.00** | FP16 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm 重み |
| `vram_weights_linear` | 15,136,194,560 | **14435.00** | FP16 線形重み |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`** |
| `vram_decode_activations` | 779,776 | **0.74** | decode 用バッファ |
| `vram_prefill_batch` | 100,665,344 | **96.00** | prefill バッチ + **`d_scratch_f16`** |

**`vram_prefill_batch`** は **`gpu-cuda`**（**~84 MiB**）より大きい（Prefill 用 **`d_scratch_f16`** = **`max_seq × hidden_dim`** FP16 を含む）。スループット（同計測）: prefill **124.95** / decode **28.88** / total **91.89** tok/s。

**`make log.push`** は実行時に **`BENCH_LOG_FILE="$(BENCH_LOG_FILE)"`** を子プロセスへ渡し、終了後に上記ファイルから **`prompt_tokens=`** 等を **`sed`** で抽出して **`Makefile` の `BENCH_LOG`** に 1 行追記する。**ROCm / Vulkan / CUDA いずれも stdout パースに依存しない**。

### Vulkan compute（`gpu-vulkan`）

**`qwen3-8b/gpu-vulkan/`** で単体ビルド。**ROCm も CUDA も不要**。**Vulkan 1.1+** 対応 GPU と **`libvulkan-dev`**、シェーダ用 **`glslang-tools`** が必要。

```bash
cd qwen3-8b/gpu-vulkan
make build          # shaders/*.comp → shaders/*.spv、qwen3-vulkan をリンク
make run PROMPT="Hello"
# オフライン FP16 キャッシュ（gpu-rocm / gpu-cuda と同形式）:
make pack-cache
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# バイナリ直接（SPIR-V ディレクトリ必須）:
QWEN3_VK_SHADER_DIR=$(pwd)/shaders ./qwen3-vulkan ../model.gguf -p "Hello" -n 64

# ベンチマーク履歴（gpu-vulkan/ 単体 Makefile）:
make log.push
make log
```

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `GLSLANG` | SPIR-V コンパイラ | `glslangValidator` |
| `QWEN3_VK_SHADER_DIR` | **`.spv`** 探索ディレクトリ（実行時） | **`make run`** / **`pack-cache`** が **`$(CURDIR)/shaders`** を設定 |
| `MODEL` | GGUF パス | `../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |
| `BENCH_LOG_FILE` | ベンチログ出力先 | `/tmp/benchmark.log` |

**`log.push`** の 1 行形式: **`timestamp|GPU名(vulkaninfo)|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**

**参考計測**（**`Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**・**RADV GFX1201**・**`-l 512`**・**`make log.push`** **2026-05-29 09:39**、132+16 トークン。推論区間のみ）:

| 指標 | tok/s |
|------|------:|
| prefill | **2.97** |
| decode | **2.17** |
| total | **2.86** |

**VRAM**（**`BENCH_LOG_FILE`**）: **`vram_total`** **15851.92 MiB**。**`[vram_breakdown]`**: embedding **1187** / norm **1.18** / KV **144** / decode 活性 **0.74** / prefill batch **84** MiB（線形 **~14435 MiB** は **`vram_total` に含む**）。**`vram_device_used`** は **0**（Vulkan 未実装）。

同一 GPU・同一モデルの **`gpu-rocm`**（**2026-05-29 08:12**）: prefill **124.95** / decode **28.88** / total **91.89** tok/s → Vulkan は prefill **約 42×**・decode **約 13×** 遅い。

**既知の制約**（詳細は **`README.md`** 付録）:

- レイヤーごとに多数の **`vkCmdDispatch`** + descriptor set 確保/解放 + fence 待ち（同期）→ **CPU オーバーヘッド大**。
- Prefill 線形層に **hipBLAS GemmEx 相当の batched GEMM 未実装**（Prefill も FP16 GEMV 系シェーダ）。
- WMMA / cooperative matrix 等の **ハードウェア固有 intrinsics 未使用**（GLSL 可搬性優先）。

### CUDA（NVIDIA GPU）

**`qwen3-8b/Makefile` はモデル取得のみ**。**`qwen3-8b/gpu-cuda/`**（FP16）または **`qwen3-8b/gpu-cuda-nvfp4/`**（NVFP4）で単体ビルドする。

```bash
cd qwen3-8b/gpu-cuda
# 既定: nvidia-smi で CUDA_GENCODE / FA_BR を自動選択（Blackwell 12.x → sm_120 + FA_BR=32）
make build
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# オフライン FP16 キャッシュ（2 回目以降の起動を高速化）:
make pack-cache
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# 手動でアーキテクチャを指定する例:
make build CUDA_GENCODE=arch=compute_89,code=sm_89
# Blackwell で PTX compute_86 JIT は推論が壊れるため非推奨 — 上記自動検出または sm_120 を使用

# PolarQuant-R KV キャッシュ（線形 FP16 のまま・任意 GPU）
make build.polarquant
make pq-test

cd ../gpu-cuda-nvfp4
# Blackwell（RTX 50 系等）— CUDA 13 + CUTLASS + NVFP4
make cutlass                            # 初回: third_party/cutlass を clone
make build                              # sm_120a, BONSAI_FP4=1, FA_BR=32
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# オフライン NVFP4 キャッシュ（2 回目以降の起動を高速化）:
make pack-cache
make run MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"
# 環境構築から一式（apt CUDA 11 除去 → CUDA 13、要 root 相当）:
make blackwell

# Blackwell — NVFP4 線形 + PolarQuant-R KV（最大 VRAM 節約）
make build.polarquant
make run.polarquant MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf PROMPT="Hello"

# ベンチマーク履歴（gpu-cuda/ または gpu-cuda-nvfp4/ 単体 Makefile）:
cd qwen3-8b/gpu-cuda
make log.push              # 既定 BENCH_PROMPT (~128 tok) / BENCH_N=128 / -t 0
make log                   # Makefile 内 BENCH_LOG を表表示
make log.push BENCH_N=64   # 生成トークン数など上書き可
cd ../gpu-cuda-nvfp4
make log.push
make log
```

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `CUDA_HOME` | CUDA Toolkit ルート | `/usr/local/cuda` |
| `CUDA13_NVCC` | Blackwell 向けに優先する nvcc パス（**`gpu-cuda/Makefile`**） | `/usr/local/cuda/bin/nvcc` |
| `NVCC` | CUDA コンパイラ | Blackwell 検出時は **`CUDA13_NVCC`**（存在すれば）、それ以外は **`$(CUDA_HOME)/bin/nvcc`** |
| `GPU_CCAP_NUM` | **`nvidia-smi`** の compute capability（**`12.0` → `120`**） | 自動（**`make build`** 時に表示） |
| `CUDA_GENCODE` | `-gencode` 引数（**`gpu-cuda`** の **`kernels.cu`**） | **`nvidia-smi` 自動**（**12.0 → `sm_120`**、**12.1 → `sm_120a`**、**90/89/86**、未検出 **`compute_86` PTX**）。**`gpu-cuda-nvfp4`**: `arch=compute_120a,code=sm_120a` |
| `BLACKWELL_GENCODE` | **`gpu-cuda-nvfp4`** の **`fp4_*`** オブジェクト用 `-gencode` | `arch=compute_120a,code=sm_120a` |
| `CUTLASS_TAG` | **`third_party/cutlass`** の clone タグ（**`gpu-cuda-nvfp4/Makefile`**） | **`v4.5.0`** |
| `BLACKWELL_NVCCFLAGS` | **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** / **`fp4-test`** 用 | **`-std=c++17`**（CUTLASS 必須）+ **`BLACKWELL_GENCODE`** 固定（NVFP4 MMA 必須）。**`-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored`**（CUTLASS v4.5.0 の **`sm100_static_tile_scheduler.hpp`** 由来 nvcc #20012 用。CUTLASS 公式ビルドと同オプション。詳細は **`gpu-cuda-nvfp4/Makefile`** コメント参照） |
| `FP4_GEMM_OBJ` / `FP4_QWEN3_OBJ` | NVFP4 オブジェクト名（**`gpu-cuda-nvfp4`** のみ） | **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** |
| `SRC_DIR` | 共有ソース参照（**`gpu-cuda-nvfp4/Makefile`**） | `../gpu-cuda` |
| `MODEL` | GGUF パス（各 CUDA ディレクトリからの相対） | `../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf` |
| `BONSAI_FP4` | NVFP4 線形層（**`fp4_qwen3`**） | **`gpu-cuda-nvfp4`** で常に `1`（**`-DBONSAI_FP4=1`** 固定）。**`gpu-cuda`** では未使用 |
| `BONSAI_POLARQUANT` | PolarQuant-R KV キャッシュ | `0`（**`build.polarquant`** で `1`） |
| `FA_BR` | Flash Attention の K/V タイル幅 | **`gpu-cuda`**: **`nvidia-smi` 自動**（Blackwell **12.x → 32**、それ以外 **64**）。**`gpu-cuda-nvfp4`**: 未指定時 **32** |
| `BUILD_STAMP` | **`CUDA_GENCODE` / FA_BR / PolarQuant 変更検知**（**`.build_config.stamp`**） | **`gpu-cuda/Makefile`** のみ |
| `KERNELS_OBJ` | **`kernels.cu` の出力名** | **`gpu-cuda`**: `kernels.fabr$(FA_BR).pq$(BONSAI_POLARQUANT).o`。**`gpu-cuda-nvfp4`**: 同上 |
| `MAIN_OBJ` | **`main.c` の出力名** | `main.pq$(BONSAI_POLARQUANT).o` |
| `BENCH_PROMPT` | **`log.push`** のプロンプト文字列（~128 token 想定） | 英語長文（各 CUDA Makefile 内） |
| `BENCH_N` | **`log.push`** の **`-n`**（生成トークン上限） | `128` |
| `BENCH_SEED` | **`log.push`** の **`-s`** | `42` |
| `BENCH_LOG_FILE` | 推論終了時に **`qwen3-gpu-cuda`** / **`qwen3-gpu-cuda-nvfp4`** が書き込むベンチログ（**`log.push`** が読取） | `/tmp/benchmark.log` |
| `GPU_SM` | **`log.push`** 記録用 GPU 識別子 | **`gpu-cuda`**: **`sm_$(GPU_CCAP_NUM)`**（**`nvidia-smi`**）。**`gpu-cuda-nvfp4`**: **`sm_120a`** |

**`log.push`** の 1 行形式（パイ区切り）: **`YYYY-MM-DDTHH:MM:SS|GPU_SM|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps`**（**`date +%Y-%m-%dT%H:%M:%S`**、タイムゾーンオフセットなし）。**`make log`** は上記を表表示（列幅調整。旧エントリに **`+00:00`** 等が付いていても表示時に除去）。スループットは推論のみ（prefill+decode）。モデル重み H2D は含まない。ROCm 版は第 2 列が **`GPU_ARCH`**（**`gfx*`**）。**`gpu-vulkan`** は第 2 列が **`vulkaninfo` 由来の GPU 名**。**VRAM 内訳は `BENCH_LOG_FILE` の `[vram_breakdown]` を参照**（**`gpu-vulkan`** は簡易内訳 + **`vram_total`**。線形重みは **`vram_total` に含む**。**`Makefile` の `BENCH_LOG` 行には tok/s のみ）。

**ベンチログファイル**（**`qwen3-rocm`** / **`qwen3-vulkan`** / **`qwen3-gpu-cuda`** / **`qwen3-gpu-cuda-nvfp4`** が推論終了時に上書き。環境変数 **`BENCH_LOG_FILE`** でパス変更可。既定 **`/tmp/benchmark.log`**）:

| キー | 意味 |
|------|------|
| （共通） | **`timestamp`**, **`hostname`**, **`model`**, **`gpu`**, CLI 相当, **`prompt_tokens`**, **`gen_tokens`**, **`prefill_sec`**, **`decode_sec`**, **`total_sec`**, **`prefill_tps`**, **`decode_tps`**, **`total_tps`**, プロンプト全文 |
| **`vram_total`** / **`vram_total_mib`** | **`GpuVramProfile.total_bytes`**（推定合計） |
| **`vram_device_used`** / **`vram_device_total`** | **`cudaMemGetInfo`** / **`hipMemGetInfo`**（各 **bytes** + **`_mib`**） |
| **`[vram_breakdown]`** | **`vram_weights_embd`**, **`vram_weights_f32_norm`**, **`vram_weights_linear`**（FP16）または **`vram_weights_fp4`**（NVFP4）, **`vram_kv_cache`**, **`vram_decode_activations`**, **`vram_prefill_batch`**（ROCm は **`d_scratch_f16`** 含む）, （NVFP4）**`vram_fp4_gemm_scratch`** |

**`cpu`** / **`cpu-multicore`** / **`cpu-blas`** / **`gpu-rocm`** / **`gpu-vulkan`** / **`gpu-cuda`** / **`gpu-cuda-nvfp4`** / **`xdna2`** / **`xdna2-bfp16`** の各 **`Makefile`** は **`.DEFAULT_GOAL := run`**（引数なし **`make`** で推論）。**`run`** は **`ensure-model`** に依存し、GGUF 未配置時は **`make -C .. model`** を自動実行する。**GPU 4 経路**の **`run`** は **`pack-cache`** も依存（**`manifest`** 欠落・**MODEL / バイナリ** 更新時にオフラインキャッシュを再生成）。**`gpu-cuda`** / **`gpu-cuda-nvfp4`** の **`pack-cache`** は **`build`**（または **`run.polarquant`** 時の **`build.polarquant`**）に依存。**`nvcc` / `nvlink` は `PATH` に CUDA の `bin` を通す**。apt **`nvidia-cuda-toolkit`（CUDA 11）** と **CUDA 13** は併存させない（**`gpu-cuda-nvfp4` の `make blackwell`** が 11.x を除去して 13 を入れる）。

### XDNA2（AMD Ryzen AI NPU）

```bash
cd qwen3-8b/xdna2
make build
# ユーザを render グループに追加して /dev/accel/accel0 を開けるようにしておく
sudo usermod -aG render "$USER"
# 既定では NPU 行列乗算用の制御コードバイナリは XDNA_GEMV_DIR から検索する。
# リポジトリ同梱は xdna-gemv/kernels のスタブ（実機では使われない）— MLIR-AIE 生成物に差し替える。
# 未設定/未配置の場合は OpenMP CPU フォールバックに自動切り替え。
XDNA_GEMV_DIR=xdna-gemv/kernels ./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hi" -n 8
# 強制的に NPU を使わず CPU OpenMP で実行する場合:
XDNA_FORCE_CPU=1 ./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hi" -n 8
# 上級者向け: ハードウェアコンテキストの試行値を直接上書き
#   XDNA_NUM_COL=<n>     CREATE_HWCTX で要求する列数の上限
#   XDNA_NUM_TILES=<n>   num_tiles を直接固定（core.row_count 整数倍が必要）
#   XDNA_HEAP_SIZE=<bytes>  DEV_HEAP のサイズ（既定 64 MiB、ファーム上限による）
XDNA_NUM_COL=1 XDNA_HEAP_SIZE=33554432 ./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status
# NPU / XDNA_GEMV_DIR / 各形状の bf16-gemv-<n>x<d>.bin の可否だけ確認し終了（重みロード・推論なし）
./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --xdna-status
# 同上の短い別名
./qwen3-xdna2 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -X
# BFPX ホスト重み版（ビルド後バイナリは qwen3-xdna2-bfpx）
cd ../xdna2-bfp16
make build
./qwen3-xdna2-bfpx ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf -p "Hi" -n 8
```

**詳細（入門）**: **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md`** に **ctrlcode の定義**（ERT・Instruction Buffer・`ERT_START_NPU`）、**GEMV と** **`(n,d)`**、**GPU（ROCm/HIP）カーネルとの対比**（オーバーレイ＋ctrlcode、標準ワークフローでのユーザー主導性の違い）、公開ミラーがある場合の **`qwen3-8b/xdna2/xdna-gemv/kernels/Makefile` での一括ダウンロード（README §8.1）**、**MLIR-AIE / IRON での自前ビルド手順（`qwen3-8b/xdna2/xdna-gemv/toolchain/README.md`）**、各 **`bf16-gemv-<n>x<d>.bin`** と Qwen3-VL-8B の対応、スタブ **`GQF3XDNA`**、**`--xdna-status`**、プレースホルダ再生成と MLIR-AIE / IRON 生成物への差し替えを記す。

## 実行時の挙動

**CPU（`qwen3-cpu` / `qwen3-cpu-omp` / `qwen3-cpu-blas`）**: 重みは mmap 上の GGUF を参照。KV・活性は主に float32。サンプリングはホスト上の logits に対して実施。**`qwen3-cpu`** / **`qwen3-cpu-omp`** は量子化行を都度ブロック逆量子化してから内積。**`qwen3-cpu-blas`** は F32 行列積と Attention を **`cblas_sgemv`** に集約。量子化 GEMV は **Q8_K + 全型 AVX2 整数内積**（詳細は **「量子化と行列積」** 参照）。**prefill** 中（最終プロンプト token 以外）は **LM head をスキップ**。**`-t 0`（greedy）** では **全 vocab logits を確保せず `mm_argmax_row`**。**RoPE** は起動時 **cos/sin キャッシュ**参照。プロンプト区間は **1 トークンずつ teacher forcing**。**Prefill progress bar** と tok/s 要約を stderr に出力。

**ROCm（`qwen3-rocm`）**: 量子化重みを **行単位融合逆量子化** → **F16 VRAM**、または **`<model>.gguf.fp16`** オフラインキャッシュ（**`.fp16bin`** + **`manifest`** 一致時）から H2D。起動ログ: **`Loading FP16 cache from …`** または **`Uploading weights (row dequant -> FP16)...`**。**`max_tensor_nelements` の F32/F16 全テンソルステージングは廃止**。H2D 進捗は 8 レイヤーごとに **`layer N/L uploaded: X.XX sec, X.XX GB/sec`** を stdout に出力。**Prefill**（**`n_prompt > 1`**）は **`forward_prefill_gpu`** で全プロンプトトークンを **1 回の batched forward** で処理する。線形層（Q/K/V/O、gate/up/down）は **hipBLAS `hipblasGemmEx`**（[llama.cpp](https://github.com/ggml-org/llama.cpp/) HIP バックエンドが **rocBLAS / hipBLAS** を使うのと同趣旨。内部は **`ggml_cuda_op_mul_mat_cublas`** と同一の **`OP_T, OP_N`** 行列レイアウト）。活性化は **`f32_to_f16_batch_kernel`** で **`d_scratch_f16`** に変換してから GEMM し、出力は **FP32**（**`HIPBLAS_COMPUTE_32F`**）。同一入力を共有する q/k/v や gate/up では **FP16 変換を 1 回**にまとめる。**`n_tokens < 2`** または hipBLAS 未使用時は **`mm_f16_gemv_batch_kernel`** にフォールバック。Attention 以降（RoPE、**`attn_flash_prefill_kernel`**、残差、SiLU 等）はカスタム HIP カーネルのまま。最終プロンプト token のみ **LM head** で logits を計算。**Decode** は **`forward_gpu`**（1 トークンずつ **`mm_f16_gemv_kernel`** + Flash decode）。**`n_prompt == 1`** は単一 **`forward_gpu`**。**`0 < top-p < 1`** の nucleus は **logits 全語彙 D2H** して CPU 処理する場合がある。それ以外は GPU で argmax / softmax＋多項サンプル等。stderr に **Prefill progress bar**（バッチ prefill は **0 → 完了**）と prefill / decode / total の **スループット要約**（**`cpu-blas`** 同形式）。stdout には **`--- N prompt tokens + M generated tokens ---`** のみ。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value ベンチ＋**`[vram_breakdown]`**（**`model_vram_profile`**。**`hipMemGetInfo`** で device 使用量も記録。推論区間のみ。重み H2D は計測外）。

**CUDA（`qwen3-gpu-cuda` / `qwen3-gpu-cuda-nvfp4`）**: プロンプトは **`gpu_forward_prefill`**、生成は **`gpu_forward`**（1 トークン）。Attention・RoPE・残差は **`kernels.cu`**（**`../gpu-cuda/kernels.cu`** を **`gpu-cuda-nvfp4`** が参照）。**サンプリングはホスト**（logits D2H）。重み H2D 進捗は 8 レイヤーごとに **`layer N/L uploaded: X.XX sec, X.XX GB/sec`**（ROCm 版と同形式）。stderr に **Prefill progress bar** と prefill / decode / total の **スループット要約**（**`cpu-blas`** / ROCm 同形式）。stdout には **`--- N prompt tokens + M generated tokens ---`** のみ。推論終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value ベンチ＋**`[vram_breakdown]`**（**`gpu_model_vram_profile`**。推論区間のみ。重み H2D は計測外）。

- **`gpu-cuda`（FP16）**: 線形（Q/K/V/O、gate/up/down、LM head）と **`token_embd`** は **`upload_fp16_linear`** / 行単位 **`upload_fp16_tensor_streaming`** で **FP16 VRAM** へ。ロード経路は (1) **オフラインキャッシュ**（**`<model>.gguf.fp16`**、`manifest` 一致時）から **`.fp16bin`** を H2D、(2) キャッシュ無効・ミス時は **GGUF 行単位融合逆量子化**（**`dequant_tensor_row`** → 逐次 H2D）。**`--no-fp16-cache`** で常に (2)。norm 系は **F32**。実行時線形は **`mm_f16_gemv_kernel`** / Prefill バッチ。**起動ログ例**: **`Loading FP16 cache from …`** または **`Uploading weights (fused dequant -> FP16)...`**。
- **`gpu-cuda-nvfp4`（NVFP4）**: 線形（Q/K/V/O、gate/up/down、LM head）は **`upload_linear_fp4`** で **NVFP4 キャッシュのみ** VRAM へ（線形 FP16 複製なし）。ロード経路は (1) **オフラインキャッシュ**（**`<model>.gguf.nvfp4`**、`manifest` が GGUF と一致する場合）から **`.fp4bin`** を H2D（**`FP4_CACHE_VERSION=2`**。旧 v1 は **`make pack-cache`** で再生成）、(2) キャッシュ無効・ミス時は **GGUF 行単位融合逆量子化**（**`dequant_tensor_row`** → **`fp4_qwen3_weight_from_rows`**）。**`token_embd`** は **`upload_embd_gpu_streaming`** で量子化 GGUF から行単位 FP16 H2D。**`--no-nvfp4-cache`** で常に (2)。norm 系は **F32**。**`gpu_model_create`** が **`wq_fp4` 等**を採用し **`use_fp4=1`**。起動ログ例: **`Loading NVFP4 cache from …`** または **`Uploading weights (fused dequant -> NVFP4 linear layers)...`**、**`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`**。
- **`build.polarquant`**（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**）: KV キャッシュは F32 の代わりに **`PQBlock`** 配列（**64 B/head**）。K/V 書き込みは **`polarquant_kv_write_one`** / **`polarquant_kv_write_batch`** でエンコード。Attention は **`flash_attn_gqa_pq_kernel`** / **`flash_attn_prefill_gqa_pq_kernel`** がタイル単位で **`pq_decode_head`** により F32 復号。起動ログ例: **`PolarQuant-R: KV cache enabled (head_dim=128, 64 bytes/head, ~8.00x vs F32)`**。**`head_dim=128` 固定**（Qwen3-VL-8B 向け）。
- **`gpu-cuda-nvfp4` + `build.polarquant`**: NVFP4 線形経路と PolarQuant-R KV 経路を同時有効化。起動ログに **NVFP4** と **PolarQuant-R** の両方が出る。
- **NVFP4 線形の実行**（**`gpu-cuda-nvfp4`** のみ）: **`fp4_qwen3_mm`** — prefill / decode とも **CUTLASS `fp4_gemm_run_cached`**（実トークン数 **`M`** を **128 整列した `M_pad`** で GEMM。活性は BF16 パッド後にその場 FP4 量子化）。極大活性は **`FP4_QUANT_MAX_ABS=1024`** でスケール飽和による NaN を防止（詳細 **`gpu-cuda-nvfp4/DEBUG.md`**）。**`--fa-debug`** で FA/KV/層ごとの数値トレースが可能。

**XDNA2（`qwen3-xdna2`）**: 線形ウェイトは **mmap された GGUF** を **`main-omp.c` と同様**に参照する（埋め込みは mmap 上行の量子化レイアウトからブロック単位復号）。各 **GEMV** のたび、その行列だけを **`AMDXDNA_BO_SHMEM` に確保した単一 BF16 スクラッチ**へ CPU で復号・BF16 化し、`SYNC_BO` でデバイス可視にしたうえで、入力 BF16・重み・出力への `xdna_addr` を **`ERT_START_NPU`** で `DRM_IOCTL_AMDXDNA_EXEC_CMD` に渡す構成は従来どおり。**レイヤー分の恒久 BF16 重み BO は保持しない**。RMSNorm／Qwen3 ヘッド RMSNorm／Attention 等も **CPU**。NPU が使えないときは BF16 GEMV が **OpenMP** にフォールバックする（実装どおり bit-identical）。

推論開始前に **`=== XDNA GEMV / NPU ctrlcode status ===`** ブロックを標準出力へ出し、`XDNA_FORCE_CPU`・DRM オープン可否・`XDNA_GEMV_DIR`・テキスト経路で使う **6 種類の GEMV 形状**それぞれについて `bf16-gemv-<n>x<d>.bin` を **`[ OK ]`（実 ctrlcode）／`[STUB]`（リポジトリ同梱プレースホルダ・マジック `GQF3XDNA`）／`MISS`** で表示する（実機でロードされるのは非 STUB のみ）。推論後は **NPU GEMV 回数／CPU GEMV 回数**に加え、**すべて NPU／すべて CPU／混在**を短文で表示する（実際に `EXEC_CMD` が成功したかはランタイムカウントが基準）。**`--xdna-status`** または **`-X`** は GGUF パースと `npu_open` のみ行い当該レポートを出力して **終了**する（重みロード・生成ループなし）。**`qwen3-8b/xdna2/xdna-gemv/kernels/`** の `.bin` プレースホルダの再生成はリポジトリルートで `python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py qwen3-8b/xdna2/xdna-gemv/kernels`。

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
| `--pack-fp16-cache [dir]` | **FP16 版（`gpu-cuda` / `gpu-rocm` / `gpu-vulkan`）**。全 FP16 対象 tensor をオフライン FP16 キャッシュへ書き出して終了。省略時 **`dir=<model>.gguf.fp16`** | — |
| `--no-fp16-cache` | **FP16 版（`gpu-cuda` / `gpu-rocm` / `gpu-vulkan`）**。オフラインキャッシュを無視し GGUF から再逆量子化 | 既定はキャッシュ利用（`manifest` 有効時） |
| `--pack-nvfp4-cache [dir]` | **NVFP4 版のみ**。全線形 tensor をオフライン NVFP4 キャッシュへ書き出して終了。省略時 **`dir=<model>.gguf.nvfp4`** | — |
| `--no-nvfp4-cache` | **NVFP4 版のみ**。オフラインキャッシュを無視し GGUF から再量子化 | 既定はキャッシュ利用（`manifest` 有効時） |
| `--fa-debug` | **CUDA 版（`gpu-cuda` / `gpu-cuda-nvfp4`）**。prefill/decode の FA・KV・層トレースを stderr に出力（**`FA_DEBUG`** 行） | 無効 |

## アーキテクチャ

### 実装のレイヤー構成

各実装ファイルは、外部ライブラリに分割せず、ほぼ同じ順序で機能を持つ。

1. **GGUF と量子化形式の定義**: GGUF の値型、GGML tensor dtype、`QK_K=256` の K-quant / IQ ブロック構造を定義する。`BlockQ4_K`、`BlockQ5_K`、`BlockIQ2_S`、`BlockIQ3_S` は GGML の packed layout に合わせて `#pragma pack(push, 1)` で定義する。
2. **IQ2_S / IQ3_S の復元テーブル**: `kmask_iq2xs`、`iq2s_grid`、`iq3s_grid` などを持ち、GGML 側の小さな格子表現を `float` に戻す（詳細は **「量子化と行列積」→「IQ2_S / IQ3_S グリッドテーブル」**）。
3. **モデル構造体**: `Config` がモデル形状、`TensorInfo` が GGUF 内 tensor descriptor、`Tok` が tokenizer、`Weights` / `WeightsDev` が重み、`State` が実行時バッファ、`Model` がそれらをまとめる。
4. **ロード処理**: `mmap` した GGUF からメタデータと tensor descriptor を読み、CPU 版は tensor へのポインタを保持し、ROCm / Vulkan / CUDA 版は重みを GPU にアップロードする。
5. **推論処理**: CPU 版は 1 トークン単位の forward を teacher forcing で繰り返し、生成区間は logits から次トークンを選ぶ。ROCm / Vulkan / CUDA GPU 版はプロンプトを **Prefill バッチ**（**`forward_prefill_gpu`** / **`gpu_forward_prefill`**）で先に処理し、以降 decode を 1 トークンずつ行う。

この構成により、CPU 版は「GGUF の量子化重みをその場で読む参照実装」、ROCm / Vulkan / CUDA 版は「同じモデル構造を GPU 常駐重みに変換して動かす実装」として対応づけられる。

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
| **`gpu-rocm` / `gpu-cuda`** | ロード時 **全線形 F16 VRAM 展開** → Prefill 線形は **hipBLAS/rocBLAS GEMM**（ROCm）または **カスタムバッチ GEMV**（CUDA FP16）／Decode は **GEMV** | VRAM 使用量（8B 級で数 GiB 級）+ Prefill 用 **S×dim バッチバッファ**（ROCm **`d_*_batch`**）+ **`d_scratch_f16`**（**max_seq × hidden_dim** 要素） |

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

#### IQ2_S / IQ3_S グリッドテーブル

各 **`main.c`**（**`cpu`** / **`cpu-multicore`** / **`cpu-blas`** / **`gpu-rocm`** / **`gpu-cuda`**）は **`ggml-common.h`** 由来の静的配列 3 つを **`dequant_iq2_s`** / **`dequant_iq3_s`**（および **`cpu-blas`** の **`vec_dot_*_q8_K`**）で参照する。いずれも **符号なしの大きさ** を格子ルックアップで復元し、**±1 の符号** は別フィールド + **`kmask_iq2xs`** で復元する。

##### `kmask_iq2xs[8]`

| `j` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----|---|---|---|---|---|---|---|---|
| mask (hex) | 01 | 02 | 04 | 08 | 10 | 20 | 40 | 80 |

**`kmask_iq2xs[j] = 1 << j`**。**`(signs[l] & kmask_iq2xs[j])`** が非ゼロなら **`-1`**、ゼロなら **`+1`** を重みに掛ける。

- **`dequant_iq2_s`**: **`signs[l]`** 1 バイトが **8 重み**分。
- **`dequant_iq3_s`**: 同一バイトの **下位 4 bit** が **grid1**（4 重み）、**上位 4 bit** が **grid2**（4 重み）。

##### `iq2s_grid[1024]`（IQ2_S、2.5 bpw）

| 項目 | 内容 |
|------|------|
| エントリ型 | **`uint64_t`** 1 個 = **`uint8_t` 8 個**の符号なし大きさ（リトルエンディアン） |
| 例 | **`0x0808080808080808`** → 8 重みすべて大きさ **8** |
| インデックス | **0..1023**（**2^10**）= **`qs[l]`** 下位 8 bit \| **`((qh[ib32] << (8-2*l)) & 0x300)`** 上位 2 bit |
| スケール | **`dl = d * (0.5 + 4bit_scale) * 0.25`**（32 要素サブブロックごとに **`db[0]`/`db[1]`** の 2 段） |
| 逆量子化 | **`y[j] = dl * grid[j] * (signs[l] の j 番目ビット ? -1 : +1)`** |
| 格子値 | 量子化時 **L∈{0,1,2}**（**q = 2L+1 → 1,3,5**）の **8 個組み合わせ**のうち on-grid のみ。典型バイト値 **0x08(8), 0x19(25), 0x2b(43) ≈ 8×{1,3,5}** |
| 符号 | **`BlockIQ2_S.qs[]` 後半**（**`signs = qs + QK_K/8`**） |

##### `iq3s_grid[512]`（IQ3_S、3.44 bpw）

| 項目 | 内容 |
|------|------|
| エントリ型 | **`uint32_t`** 1 個 = **`uint8_t` 4 個**の符号なし大きさ（リトルエンディアン） |
| 例 | **`0x01010101`** → 4 重みすべて大きさ **1** |
| インデックス | **0..511**（**2^9**）。**grid1**: **`qs[2*l+0] \| ((qh[k] << (8-2*l)) & 256)`**。**grid2**: **`qs[2*l+1] \| ((qh[k] << (7-2*l)) & 256)`** |
| 8 重みの構成 | **grid1 の 4 個 + grid2 の 4 個**（**`dequant_iq3_s`** で 2 ルックアップ） |
| スケール | **`db = d * (1 + 2 * 4bit_scale)`**（64 要素ペアごとに **`db1`/`db2`**） |
| 逆量子化 | **`y[j+0] = db * grid1[j] * (sign ±1)`**、**`y[j+4] = db * grid2[j] * (sign ±1)`**（**`j=0..3`**） |
| 格子値 | **L∈{0..7}**（**q = 2L+1 → 1,3,5,7,9,11,13,15**）の **4 個組み合わせ**のうち on-grid のみ。各バイトは **q そのもの**（**0x01, 0x03, … 0x0f**） |
| 符号 | **`BlockIQ3_S.signs[]`**（専用 32 バイト） |

##### IQ2_S と IQ3_S の比較

| | IQ2_S | IQ3_S |
|---|-------|-------|
| コードブック | **`iq2s_grid[1024]`** | **`iq3s_grid[512]`** |
| 1 ルックアップあたり | 8 重み | 4 重み |
| インデックス幅 | 10 bit | 9 bit |
| 8 重みの復元 | grid **1 回** | grid1 + grid2 **2 回** |
| スケール式 | **`d * (0.5+s) * 0.25`** | **`d * (1 + 2*s)`** |

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

ROCm 版は **GGUF 行単位融合逆量子化**（**`dequant_tensor_row`** → **`upload_fp16_tensor_streaming`**）で **全線形を FP16 VRAM** に載せる（**`max_tensor_nelements` ステージング廃止**）。**`<model>.gguf.fp16`** に **`manifest`** と **`.fp16bin`** がある場合は **`fp16_host_weight_load`** → H2D を省略（**`make pack-cache`** / **`--pack-fp16-cache`**）。norm は F32 のまま GPU。Decode 時 GEMV は FP16 カーネル（**`mm_f16_gemv_kernel`**）。**ROCm Prefill** の線形層は **hipBLAS `hipblasGemmEx`**（FP16 重み × FP16 活性 → FP32 出力）。CUDA Prefill は **`mm_f16_gemv_batch_kernel`** 相当のカスタムカーネル。

CUDA **`gpu-cuda`（FP16）** は **GGUF 行単位融合逆量子化**（**`dequant_tensor_row`** → **`upload_fp16_tensor_streaming`**）で **全線形を FP16 VRAM** に載せる（**`max_tensor_nelements` の F32/F16 ステージングは廃止**）。**`<model>.gguf.fp16`** に **`manifest`** と **`.fp16bin`** がある場合は **`fp16_host_weight_load`** → H2D を省略（**`make pack-cache`** / **`--pack-fp16-cache`**）。norm 系は **F32 VRAM**。実行時の線形は **`mm_f16_gemv_kernel`** / Prefill バッチ。推論中の GGUF 逐次逆量子化は行わない。

CUDA **`gpu-cuda-nvfp4`（NVFP4）** は線形 tensor を **GGUF 行単位で F32 復号**したうえで **NVFP4 キャッシュ**（**`fp4_qwen3_weight_from_rows`** → **`fp4_host_weight_build`**）にのみ H2D し、**線形の FP16 VRAM 複製は行わない**。**`<model>.gguf.nvfp4`** に **`manifest`** と **`.fp4bin`**（**`FP4_CACHE_VERSION=2`**）がある場合は **`fp4_host_weight_load`** → **`fp4_qwen3_weight_from_host`** で H2D を省略できる（**`make pack-cache`** / **`--pack-nvfp4-cache`** で事前生成。旧 v1 キャッシュは再生成必須）。**`token_embd.weight`** は **`upload_embd_gpu_streaming`** で **FP16 VRAM**（量子化 GGUF から行単位 H2D）、norm 系は **F32 VRAM**。起動時の **`gpu_model_create` で FP16→NVFP4 再変換は行わない**（**`wq_fp4` 等**をそのまま採用）。実行時の線形は **`fp4_qwen3_mm`**（全 M で **CUTLASS NVFP4 GEMM**）。いずれも推論中の GGUF 逐次逆量子化は行わない。

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

### ROCm / Vulkan / CUDA forward（GPU）

ROCm 版は **Prefill**（**`forward_prefill_gpu`**）と **Decode**（**`forward_gpu`**）でエントリが分かれる。CUDA 版は **`gpu_forward_prefill`** / **`gpu_forward`**。**Vulkan 版**（**`gpu-vulkan`**）も **`gpu.h`** 経由で同 API。**Vulkan** は各 compute dispatch 後に **fence 待ち**（同期実行）するためホストオーバーヘッドが大きい。いずれも embedding から LM head まで device buffer 上で実行する。各カーネル起動の依存は default stream / 単一キューの順序に任せ、forward の最後に `hipDeviceSynchronize()`（CUDA は `cudaDeviceSynchronize()`、Vulkan は各 dispatch の fence）する。

**ROCm Prefill バッチ**（**`forward_prefill_gpu`**）では **`Model`** に **`d_x_batch` / `d_xb_batch` / `d_q_batch` 等**（**`batch_cap = max_seq`**）、**`d_tokens`**、**`d_scratch_f16`**（**`max_seq × hidden_dim`** 要素・hipBLAS 用 FP16 活性スクラッチ）、**`hipblasHandle_t`** を確保し、プロンプト token ID を **`d_tokens`** へ H2D したうえで全レイヤーを **S トークン並列**で走らせる。

主なカーネルは次の通りである。

**Decode 用（1 トークン）**:

- `emb_f16_kernel`: token embedding の 1 行を FP16 から float activation に展開する。
- `rmsnorm_kernel`: block 内 reduction で二乗平均を求め、`float4` 単位も使って RMSNorm を適用する。
- `mm_f16_gemv_kernel`: 1 warp が 1 出力行を担当し、warp reduction で GEMV の和を作る。1 block は複数行を処理する（**`gpu-cuda`（FP16）** の線形層。**`gpu-cuda-nvfp4`** の線形は **`fp4_qwen3_mm`** → CUTLASS NVFP4 GEMM 経路）。
- `rmsnorm_head_kernel`: Q/K の各 head を 1 block で処理し、Qwen3 の head RMSNorm を in-place で適用する。
- `rope_kernel`: head 内の偶数・奇数ペアに対して RoPE 回転を適用する。
- `kv_cache_write_kernel`: 現在 token の K/V を layer offset と position offset から求めた cache 位置へ書く。
- `attn_flash_decode_kernel_hd128` / `attn_flash_decode_kernel`: decode 用 attention。query head ごとに 1 block を使い、過去 token を tile 化しながら online softmax の形で max と分母を更新する。
- `attn_mha_kernel`: head dimension が Flash decode の上限を超える場合の fallback。
- `silu_mul_kernel`: `SiLU(gate) * up` を要素ごとに計算する。
- `vec_add_kernel`: attention / FFN の残差加算を行う。

**ROCm Prefill バッチ用**（**`gpu-rocm/main.c`** のみ）:

- `emb_f16_batch_kernel`: 全プロンプト token の embedding を **`[S, dim]`** に一括展開。
- `rmsnorm_batch_kernel` / `rmsnorm_head_batch_kernel`: トークン次元で RMSNorm（head 版は Q/K）。
- **`launch_mm_f16_batch` → `hipblasGemmEx`**（**`n_tokens >= 2`**）: **`O[S,d] = X[S,n] @ W[d,n]^T`**。llama.cpp **`ggml_cuda_op_mul_mat_cublas`** と同じ **`OP_T, OP_N`** 呼び出し（FP16 重み・FP16 活性・FP32 出力・**`HIPBLAS_COMPUTE_32F`**）。活性化は事前に **`f32_to_f16_batch_kernel`** で **`d_scratch_f16`** へ変換。フォールバック: **`mm_f16_gemv_batch_kernel`**（出力行ごとに重み行を S トークン分再利用するカスタム GEMV バッチ）。
- `rope_prefill_batch_kernel`: token index を position として RoPE。
- `kv_write_batch_kernel`: 全プロンプト位置 **`0..S-1`** へ K/V を一括書込。
- `attn_flash_prefill_kernel`: 因果マスク付き Flash Attention。位置 **`t`** は K/V **`0..t`** のみ参照（**`n_tokens × n_heads`** block 並列）。K/V タイル **`FA_BR=32`**（gfx1100 では 64 だと shared memory 65 KiB 上限超過）。共有リダクション（**`fa_sh_reduce_max`** / **`fa_sh_reduce_sum`**）および分岐条件では **`threadIdx.x`**（符号なし）を **`(int)threadIdx.x`** にキャストして **`hd`** / **`tc`** / **`s`** と比較（**`-Wsign-compare`** 回避。挙動は同一）。
- `silu_mul_batch_kernel` / `vec_add_batch_kernel`: FFN・残差のバッチ版。

**Vulkan compute 用**（**`gpu-vulkan/shaders/*.comp`** → SPIR-V。HIP/CUDA カーネル名と対応）:

- **`emb_f16` / `emb_f16_batch`**: token embedding（decode / prefill）。
- **`mm_f16_gemv` / `mm_f16_gemv_batch`**: FP16 GEMV（Prefill も GEMV バッチ — **batched GEMM ライブラリ未使用**）。
- **`rmsnorm` / `rmsnorm_batch`**, **`rmsnorm_head` / `rmsnorm_head_batch`**: RMSNorm。
- **`rope` / `rope_prefill_batch`**: RoPE。
- **`kv_write` / `kv_write_batch`**: KV キャッシュ書込（F32）。
- **`flash_attn_decode` / `flash_attn_prefill`**: Flash Attention（GLSL）。
- **`silu_mul` / `silu_mul_batch`**, **`vec_add` / `vec_add_batch`**: FFN・残差。

### ROCm Prefill 高速化の詳細（3 段階）

LLM 推論の Prefill は、プロンプト全トークンが既知なため **Decode（1 トークンずつ GEMV）** とは異なり、**S トークン分をまとめて GEMM** として処理すると演算強度が上がり、重み HBM 読み出しを S で割れる（[Prefill 解説（wiki.llm）](https://github.com/thamada/wiki.llm/blob/main/wiki/LLM%E6%8E%A8%E8%AB%96%E3%81%AEPrefill%E8%A9%B3%E8%AA%AC%20%E2%80%95%20%E3%83%88%E3%83%BC%E3%82%AF%E3%83%B3%E4%B8%A6%E5%88%97%E3%83%BBChunked%20Prefill%E3%83%BBPD%E5%88%86%E9%9B%A2.md) 参照）。ROCm 版は **2026-05-23** にかけて次の 3 段階で Prefill を改善した。

#### 段階 0（改善前）: 1 トークンずつ forward

- **`generate`** がプロンプト長 **`S`** について **`forward_gpu`** を **`S` 回**呼び出し（Decode と同じ **GEMV** 経路）。
- 各 forward で層ごとに重み行列を HBM から **約 1 回読み**、**1 行分**の活性化にしか使わない → **Memory-bound** に近い。
- ベンチ（**`gpu-rocm/Makefile` `BENCH_LOG`**、132 prompt tokens、RX 7900 XTX / gfx1100）: **prefill ≈ 28.7 tok/s**。

#### 段階 1: Prefill / Decode 分離 + バッチ forward

**目的**: プロンプトを **1 回の GPU forward** にまとめ、線形層で **重み行の再利用** を可能にする（CUDA 版 **`gpu_forward_prefill`** と同趣旨）。

| 変更 | 内容 |
|------|------|
| **`forward_prefill_gpu`** | 全プロンプト token を **1 回**処理。Prefill 終了時のみ最終 token の hidden から **LM head** で logits を 1 回計算。 |
| **バッチ GPU バッファ** | **`Model`**: **`d_x_batch` / `d_xb_batch` / `d_q_batch` 等**（**`batch_cap = max_seq`**）、**`d_tokens`**。 |
| **バッチカーネル群** | **`emb_f16_batch_kernel`**、**`rmsnorm_batch_kernel`**、**`rmsnorm_head_batch_kernel`**、**`rope_prefill_batch_kernel`**、**`kv_write_batch_kernel`**、**`attn_flash_prefill_kernel`**、**`silu_mul_batch_kernel`**、**`vec_add_batch_kernel`**。 |
| **`mm_f16_gemv_batch_kernel`** | 出力 **行（重み行）ごとに 1 block**。重み行を **1 回読んで S トークン分ループ**（`for (t = 0; t < n_tokens; t++)`）。S 次元は **逐次**だが、HBM 上の重み読みは **層あたり 1 回/行** に削減。 |
| **`generate` 再構成** | **`n_prompt > 1`**: Prefill → サンプル → Decode ループ。**`n_prompt == 1`**: 単一 **`forward_gpu`**。 |

- ベンチ: **prefill ≈ 54 tok/s**（段階 0 の **約 1.9 倍**）。線形層は改善するが、カスタムカーネルは **rocBLAS 級のタイル GEMM・Tensor Core 活用** には及ばない。

#### 段階 2: hipBLAS GemmEx（llama.cpp 経路）

**参考**: [llama.cpp](https://github.com/ggml-org/llama.cpp/) の HIP バックエンド（**`ggml-hip`**）は **`rocBLAS` / `hipBLAS`** をリンクし、FP16 重みの行列積に **`cublasGemmEx` / `hipblasGemmEx`** を使う（**`ggml_cuda_op_mul_mat_cublas`**）。ROCm 版 Prefill 線形層もこの経路に合わせた。

**行列レイアウト**（row-major での意味）:

```
O[S, d] = X[S, n] @ W[d, n]^T
```

- **`W`**: VRAM 上 FP16、row-major **`[d, n]`**（**`upload_fp16_linear`** / **`upload_fp16_tensor_streaming`** の配置）。
- **`X`**: バッチ活性 **`[S, n]`** float32 → **`f32_to_f16_batch_kernel`** で **`d_scratch_f16`** に変換。
- **`O`**: float32 **`[S, d]`**（既存 **`d_q_batch`** 等）。

**hipBLAS 呼び出し**（llama.cpp **`cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_N, row_diff, src1_ncols, ne10, ...)`** と同型）:

```c
hipblasGemmEx(handle,
    HIPBLAS_OP_T, HIPBLAS_OP_N,
    d, n_tokens, n,           /* m, n, k */
    &alpha,
    w, HIPBLAS_R_16F, n,      /* A: 重み FP16 */
    x_f16, HIPBLAS_R_16F, n,  /* B: 活性 FP16 */
    &beta,
    o, HIPBLAS_R_32F, d,      /* C: 出力 FP32 */
    HIPBLAS_COMPUTE_32F,
    HIPBLAS_GEMM_DEFAULT);
```

**実装上の要点**:

| 項目 | 説明 |
|------|------|
| **`hipblasHandle_t`** | **`alloc_state_gpu`** で **`hipblasCreate`**。**`free_hipblas`** で破棄。 |
| **`d_scratch_f16`** | サイズ **`max_seq × hidden_dim`** 要素（最大 **`S × 12288`**）。層内で q/k/v や gate/up 等 **同一 FP32 入力** は **FP16 変換 1 回**で使い回し。 |
| **`launch_mm_f16_batch`** | **`n_tokens >= 2`** かつ **`x_f16 != NULL`** なら hipBLAS。それ以外は **`mm_f16_gemv_batch_kernel`**。 |
| **Makefile** | **`-lhipblas -lrocblas`** を追加。 |
| **起動ログ** | **`Prefill linear: hipBLAS GemmEx (llama.cpp cublas path)`** |
| **WMMA** | **`main.c` に直接 WMMA なし**（Prefill GEMM は hipBLAS → rocBLAS）。WMMA 使用は rocBLAS カーネル選択依存で **`qwen3-rocm` から ON/OFF 不可**。**`make wmma`** / 手動 **`rocprofv3 --kernel-trace`** で事後確認 |
| **未変更** | **Decode**（**`mm_f16_gemv_kernel`**）、**Attention**（**`attn_flash_prefill_kernel`**）、サンプリング、重みロード。 |

**なぜ段階 1 より大幅に速いか**:

- 段階 1 のカスタムカーネルは **S トークン次元を block 内 for ループ**で処理し、**rocBLAS 並列 GEMM** ほどの演算ユニット利用率にならない。
- hipBLAS は **M=S, K=n, N=d** の GEMM を **ハードウェア最適化カーネル**（行列積ライブラリ）で実行し、**演算強度と Tensor Core 相当のスループット**を得やすい。
- Attention は依然 **O(S²)** の総仕事量で **Prefill ボトルネックの一部**になりうるが、段階 2 では **線形層（36 層 × 6 matmul/層）** が支配的だったため、全体で **10 倍超**の改善になった。

**ベンチマーク推移**（同一環境: RX 7900 XTX / gfx1100、132 prompt tokens、`BENCH_PROMPT`、`-t 0`）:

| 段階 | prefill tok/s | decode tok/s | total tok/s | 備考 |
|------|---------------|--------------|-------------|------|
| 0: 1 トークンずつ GEMV | 28.74 | 29.35 | 28.77 | **`BENCH_LOG` 2026-05-23T15:54:16** |
| 1: バッチ forward + カスタム GEMV | 53.95 | 25.49 | 48.14 | **`BENCH_LOG` 2026-05-23T16:22:01** |
| 2: hipBLAS GemmEx | **549.94** | 25.67 | **171.41** | **`BENCH_LOG` 2026-05-23T16:37:12** |
| 2: 再計測（`threadIdx` キャスト後） | **556.77** / **556.95** | 25.77–25.78 | **172.50**–**172.56** | **`BENCH_LOG` 2026-05-24T04:56:58** / **05:05:36** |

Prefill は段階 0 比 **約 19 倍**、段階 1 比 **約 10 倍**。Decode はほぼ不変（設計どおり Prefill 専用の改善）。段階 2 再計測は初回と同程度（計測ばらつき範囲）。

**今後の余地**（未実装）:

- Attention Prefill を llama.cpp **`fattn-tile` / `fattn-mma`** 相当に近づける（因果 **S×S** をタイル並列。現状は **`(t, head)` block 並列**で位置 t により仕事量が不均等）。
- **`FA_BR=64`** は gfx1100 の shared memory 上限（65 KiB）を超えるため不可。**`FA_BR=32`** 固定。

Qwen3-VL-8B の代表形状では `head_dim=128` なので、専用の `attn_flash_decode_kernel_hd128` / `attn_flash_prefill_kernel`（Prefill）が使われる。Flash decode は K tile を shared memory に置き、Q と K の dot、online softmax、Value の重み付き和を 1 kernel 内で処理する。attention score 行列全体を global memory に持たないため、decode 時のメモリ転送を抑えられる。

### 生成ループとサンプリング

**CPU 版**（**`cpu` / `cpu-multicore` / `cpu-blas`**）は `prompt[0]` から開始し、`pos` を 0 から進める。`pos < n_prompt - 1` の間は teacher forcing として `prompt[pos + 1]` を次 token に使う。`pos >= n_prompt - 1` になったら logits から次 token をサンプリングし、`eos` または `eot` なら停止する。

**ROCm 版**（**`gpu-rocm`**）および **CUDA 版**は Prefill と Decode を分離する。**`n_prompt > 1`** のとき Prefill は **`forward_prefill_gpu`** / **`gpu_forward_prefill`** で全プロンプトを 1 回処理し、最終 token の logits から最初の生成 token をサンプルする。以降は **`forward_gpu`** / **`gpu_forward`** で 1 トークンずつ decode し、各 step でサンプルして `eos` / `eot` または `max_seq` で停止。**`n_prompt == 1`** の ROCm 版は単一 **`forward_gpu`** にフォールバックする。

サンプリングは次の分岐を持つ。

- `temp <= 0`: greedy。ROCm 版は GPU argmax 経路を使える。
- `temp > 0` かつ `top-p` 無効相当: logits を temperature で割り、softmax 後に多項サンプルする。ROCm 版は GPU softmax / multinomial 経路を使える。
- `0 < top-p < 1`: nucleus sampling。ROCm 版でも語彙全体の logits を host に戻して CPU で sort / 累積確率処理を行う fallback がある。

乱数は xorshift 系の 64-bit state を使う。seed が 0 の場合は 1 に置き換え、CPU fallback と GPU sampling の間で host 側の state を同期する。

## モデル参照

利用する GGUF のファイル名は **`qwen3-8b/Makefile` の `MODEL`** を参照する。モデル本体は著作権とファイルサイズの都合でリポジトリに含めず、既定モデルの取得元は **`qwen3-8b/gguf.txt`** に URL として置く。

取得・検証の手順は次の 3 通りである。

1. **各サブディレクトリから推論実行**（推奨）: **`cd qwen3-8b/cpu && make`**（= **`make run`**）。**`ensure-model`** が **`../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf`**（既定 **`MODEL`**）の存在を確認し、無ければ **`make -C .. model`** を実行してから推論する。**GPU 4 経路**では **`pack-cache`** も自動実行され、初回はオフラインキャッシュ生成のため起動に時間がかかる。
2. **`cd qwen3-8b && make model`**: **`gguf.txt`** 先頭 URL の `blob/main` を `resolve/main` に置換して **`wget`** し、リポジトリ同梱の **`$(MODEL).sha256sum`** で **`sha256sum --check`** する。**`$(MODEL)` が既に存在し検証に成功した場合はダウンロードをスキップ**する。チェックサムファイルが無い・検証に失敗した場合はメッセージを出して終了し、破損ダウンロードは **`$(MODEL)`** を削除する。
3. **手動**: 上記と同様に URL を `resolve/main` に直して **`wget`** 等で取得し、**`sha256sum -c $(MODEL).sha256sum`** で確認する。

**`qwen3-8b/Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf.sha256sum`** は既定 **`MODEL`** 用の参照。別量子化・別サイズに切り替える場合は **`MODEL`** と本書の前提（メタキー `qwen3vl.*`・テンソル名）が実装と一致するかを確認すること。

## 制約・既知の制限

- **CPU 版**: IQ 混在 8B は計算量が大きく、**実用的な速度は期待しにくい**。OpenMP はアルゴリズム忠実なまま並列化するが、帯域 bound のため環境次第では伸びが限定的な場合がある。**`cpu-blas`** は F32 経路の OpenBLAS 化に加え量子化 GEMV を **Q8_K + 全型 AVX2 整数内積**（層内 Q8 共有）に置き換え、**RoPE キャッシュ**・**prefill LM スキップ**・**greedy `mm_argmax_row`** でオーバーヘッドを削るため **`cpu-multicore` より速くなることが多い**。**`-ffast-math`** を付けると IQ / Q8_K 量子化で出力が壊れる。**`-march=native`** は **AVX2/F16C** 等を有効化（移植性より当該 CPU 向け最適化）。
- **ROCm 版**: AMD GPU・ROCm・**`hipcc`**・**g++ / libstdc++-dev`**（**`fp16_cache_io.o`** リンク用）。**Prefill** 線形層は **hipBLAS / rocBLAS**（**`-lhipblas -lrocblas -lstdc++`**）。重み H2D は **オフライン FP16 キャッシュ**または **GGUF 行単位融合逆量子化**（**`make run`** / **`make pack-cache`** / **`make build`** で事前 pack 可）。通常は **`rocminfo` による `GPU_ARCH_DETECTED` 自動検出**だが、検出失敗時やクロスビルド時は手動 **`GPU_ARCH`** が必要。**`gfx1152` / `gfx1153`**（例: **Ryzen AI 5 340 / Radeon 840M**）では **rocBLAS に公式 `gfx1152` Tensile が無い**ため Makefile が **`gfx1151` ビルド + `HSA_OVERRIDE_GFX_VERSION=11.5.1`** を適用（**ROCm 7.2.1 へ上げるだけでは解決しない** — **「環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS」**）。**Prefill Attention** の **`attn_flash_prefill_kernel`** は **`FA_BR=32`** 固定（gfx1100 等で 64 だと shared memory 65 KiB 上限超過）。Prefill 高速化の詳細は **「ROCm Prefill 高速化の詳細（3 段階）」**。ベンチは **`BENCH_LOG_FILE`**（**`make log.push`** が読取。VRAM 内訳は **`[vram_breakdown]`** / **`model_vram_profile`**）。
- **CUDA 版（`qwen3-gpu-cuda` / `qwen3-gpu-cuda-nvfp4`）**: NVIDIA GPU・**`nvcc`**・**`libcudart`**。**`gpu-cuda/`** または **`gpu-cuda-nvfp4/`** で単体ビルド。**`gpu-cuda`** は **`nvidia-smi`** で **`CUDA_GENCODE`** / **`FA_BR`** を自動選択（Blackwell **12.x → `sm_120` + FA_BR=32**）。**PTX `compute_86` JIT** は Blackwell 等で推論が壊れるため非推奨。**Blackwell NVFP4** は **`gpu-cuda-nvfp4`**（**CUDA 13**・CUTLASS・**`sm_120a`**）。**`build.polarquant`** は KV のみ PolarQuant-R（**`gpu-cuda`** または **`gpu-cuda-nvfp4`**、任意 GPU 可／NVFP4 併用可）。ベンチは **`BENCH_LOG_FILE`**（**`make log.push`** が読取。VRAM 内訳は **`[vram_breakdown]`**）。**`gpu-cuda-nvfp4/third_party/cutlass`** と **`fp4_verify`** は clone／ビルド生成物で常時同梱されない。**`gpu-cuda-nvfp4`** では線形重みは NVFP4 のみ VRAM に載り、**`gpu-cuda` 比で線形 FP16 分（8B 級で約 15 GiB 相当）を節約**できる（代わりに **`token_embd`** は FP16 のまま）。
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
| **`undefined reference to hipblas*`** / **`-lhipblas` リンク失敗** | ROCm の hipBLAS / rocBLAS 未インストール・不完全 | **`$(ROCM)/lib`** に **`libhipblas.so`** / **`librocblas.so`** があるか確認。ROCm 再インストールまたは **`ROCM=`** パス修正 |
| **ROCm 初回起動が遅い** | **`manifest`** 未生成のため **`make run`** が **`pack-cache`**（GGUF 行逆量子化）を実行 | 初回 **`make run`** で自動生成される。明示のみなら **`make pack-cache`**。2 回目以降は **`Loading FP16 cache from …`** |
| **ROCm FP16 キャッシュ miss 警告** | **`.fp16bin`** 欠落・形状不一致・**`manifest`** 無効 | **`make pack-cache`** を再実行。**`--no-fp16-cache`** で強制再逆量子化 |
| **`Cannot open shader: …/xxx.spv`**（**`gpu-vulkan`**） | **`.spv` 未生成** または **`QWEN3_VK_SHADER_DIR` 未設定** | **`cd gpu-vulkan && make build`**。**`QWEN3_VK_SHADER_DIR=$(pwd)/shaders`** を指定 |
| **`vkAllocateDescriptorSets` / OUT_OF_POOL_MEMORY**（**`gpu-vulkan`**） | 1 forward で descriptor pool 枯渇 | **`make clean && make build`**（プール拡大・**`vkFreeDescriptorSets`** 版） |
| **Vulkan が ROCm/CUDA より極端に遅い** | dispatch オーバーヘッド・GEMM 未実装 | 性能目的なら **`gpu-rocm`** / **`gpu-cuda`**。README 付録 **「既知の制約」** 参照 |
| **RADV `not a conformant Vulkan implementation` 警告** | Mesa 開発ドライバの警告 | 検証用付録。動作確認は可能 |
| **`gpu-rocm` ビルド失敗（C++ headers not found）** | **g++ / libstdc++-dev** 未導入 | **`apt install g++ libstdc++-dev`** |
| **ROCm Prefill が遅い（~30 tok/s 程度）** | 古いバイナリ・hipBLAS 未リンク | 起動ログに **`Prefill linear: hipBLAS GemmEx`** があるか確認。**`make -C gpu-rocm clean build`**。詳細は **「ROCm Prefill 高速化の詳細（3 段階）」** |
| **`make wmma` が FAIL** | **`qwen3-rocm` に WMMA 命令**・hipBLAS 経路未報告・**`wmma-probe` 校正失敗**（**gfx11/gfx12** で probe に **`v_wmma` 無し**） | **`make -C gpu-rocm wmma WMMA_SKIP_RUN=1`** で静的のみ確認。**`llvm-objdump`** パス（**`LLVM_OBJDUMP=`**）。MODEL 未配置時は **`WMMA_SKIP_RUN=1`**。実行時 ISA は **`WMMA_SKIP_ROCPROF=0`** または **`README.md`** の手動 **`rocprofv3`** 手順 |
| **`make wmma-probe` で `cmath` / `cstdlib` エラー** | **`hipcc`** が libstdc++ ヘッダを見つけられない | **`apt install g++ libstdc++-dev`**。**`wmma-probe`** は **`$(HIP_CFLAGS)`** 経由でヘッダパスを付与（2026-05-29 以降の Makefile） |
| **`rocBLAS error: … for GPU arch : gfx1152`**（**TensileLibrary** 欠落） | 公式 rocBLAS に **`gfx1152`** 用 lazy Tensile が無い（**ROCm 7.1.x / 7.2.x 共通**） | **`cd gpu-rocm && make clean build && make run`**（Makefile の **`gfx1151` offload + `HSA_OVERRIDE_GFX_VERSION=11.5.1`**）。詳細は **「環境依存：gfx1152（Ryzen AI / Radeon 840M）と rocBLAS」**。**ROCm バージョンアップのみ**では直らない場合がある |
| **`hipBLAS error: 6`**（Prefill 0%・**`HIPBLAS_STATUS_INTERNAL_ERROR`**） | **`gfx1152` 名で `gfx1151` ライブラリを無理に symlink した**、または **HSA オーバーライドと `--offload-arch` の不一致** | **`make clean build`**（**`HIP_OFFLOAD_ARCH=gfx1151`** を確認）。**`HSA_OVERRIDE_GFX_VERSION=11.5.1`** を **`make run`** とセットで使用。手動実行時は両方必須 |
| **`Segmentation fault`**（Prefill 直後・**`gcnArchName: gfx1151`** 表示あり） | **`HSA_OVERRIDE_GFX_VERSION` のみ**でバイナリが **`gfx1152` offload** のまま | **`make clean build`** 後 **`make run`**。ビルドログの **`Build offload arch: gfx1151`** を確認 |
| HIP 初期化失敗・デバイス未検出（**`gfx1030`** / **`gfx1100`** 等・オーバーライド不要 GPU） | シェルや親 Makefile から **空の `HSA_OVERRIDE_GFX_VERSION`** が export されている | **`unset HSA_OVERRIDE_GFX_VERSION`** 後 **`make run`**。本リポジトリの Makefile は **`HSA_OVERRIDE_GFX_VER` 非空時のみ export**（2026-06-05 以降） |
| ISA 不一致 / `GPU_ARCH not detected` | 自動検出失敗・手動指定の誤り | **`cd qwen3-8b/gpu-rocm && make detect-gpu-arch`**。失敗時は **`rocminfo`** の **`Name: gfx*`** を確認し **`make build GPU_ARCH=…`**（**`gfx1152` 実機では内部で `gfx1151` offload**） |
| `nvcc` not found / `nvlink` 失敗 | `PATH` に CUDA `bin` が無い | `export PATH=/usr/local/cuda/bin:$PATH` または各 CUDA ディレクトリの **`Makefile`** の `CUDA_HOME` を確認 |
| CUDA FP16 で出力が文字化け（Blackwell / RTX 50 系） | **`CUDA_GENCODE=compute_86` PTX JIT** で **`kernels.cu`** が不整合 | **`make build`** で **`nvidia-smi` 自動検出**（**`sm_120` + FA_BR=32**）を確認。**`=== build: GPU_CCAP=… ===`** 行を参照。手動 **`CUDA_GENCODE=arch=compute_120,code=sm_120`** |
| CUDA で PTX は動くが極端に遅い／非 Blackwell で PTX のみ | `CUDA_GENCODE` が PTX のみ | 実機 **`sm_XX`** を `code=sm_XX` で指定して再ビルド（**`gpu-cuda`**） |
| **FP16 初回起動が遅い**（**`gpu-cuda`** / **`gpu-vulkan`**） | **`manifest`** 未生成のため **`make run`** が **`pack-cache`** を実行 | 初回 **`make run`** で自動生成。明示のみなら **`make pack-cache`**。2 回目以降は **`Loading FP16 cache from …`** |
| **FP16 キャッシュ miss 警告**（**`gpu-cuda`** / **`gpu-vulkan`**） | **`.fp16bin`** 欠落・形状不一致・**`manifest`** 無効 | **`make pack-cache`** を再実行。**`--no-fp16-cache`** で強制再逆量子化 |
| Blackwell NVFP4 ビルド失敗 | CUDA 11 と 13 の混在・CUTLASS 未取得 | **`gpu-cuda-nvfp4`** で **`make blackwell`** または **`make cutlass`** → **`make build`** |
| 非 Blackwell で NVFP4 ビルドを試した | **`gpu-cuda-nvfp4`** は **`sm_120a`** 前提 | 汎用 GPU では **`gpu-cuda`** で **`make build`**（FP16）を使用 |
| **NVFP4 初回起動が遅い** | **`manifest`** 未生成のため **`make run`** が **`pack-cache`** を実行 | 初回 **`make run`** で自動生成。明示のみなら **`make pack-cache`**。2 回目以降は **`Loading NVFP4 cache from …`** |
| **NVFP4 キャッシュ miss 警告** | **`.fp4bin`** 欠落・形状不一致・**`manifest`** 無効 | **`make pack-cache`** を再実行。GGUF 更新後はキャッシュ再生成。**`--no-nvfp4-cache`** で強制再量子化 |
| **`NVFP4 quantize failed`** | **`gpu-cuda-nvfp4`** を非 Blackwell で実行、CUTLASS 未導入 | **`sm_120a`**・CUDA 13・**`make cutlass`**。汎用 GPU は **`gpu-cuda`** |
| **`make fp4-test` が FAIL / `Arch conditional MMA instruction... Aborting`** | **`fp4_gemm`** を **`compute_86` PTX** のみでビルドした古い **`fp4_gemm.o`** | **`gpu-cuda-nvfp4`** で **`make clean`** → **`make fp4-test`**（**`fp4_gemm.sm120a.o`** は **`BLACKWELL_NVCCFLAGS`** 固定）。Blackwell GPU 必須 |
| **NVFP4 で短 prefill 後に `?,` 連打等** | L6 **`down`** で極大 **`hb`** → 活性量子化スケールが UE4M3 危険域 → GEMM **NaN** → KV 破壊 | 最新ビルド（**`FP4_QUANT_MAX_ABS=1024`**）。**`make fp4-test`** PASS。**`gpu-cuda-nvfp4/DEBUG.md`** 参照 |
| **NVFP4 で同文繰り返し・「LLM」連発等** | 旧実装の **GEMM prefill** と **GEMV decode** の数値不一致、または未修正の活性 NaN | 最新 **`gpu-cuda-nvfp4`**。**`FP4_CACHE_VERSION=2`** で **`make pack-cache`** 再実行。**`make fp4-test`**。**`--fa-debug`** で L17 **`kc[t=0]`** を確認 |
| **NVFP4 prefill 終了時 `k_batch[t=0]==0` と見える** | **`k_batch` は最終層 wk バッファ**（KV ではない） | **`--fa-debug`** の **`kc` スライス**（L0/L17）を参照。**`DEBUG.md`** の誤解メモ |
| **`gpu-cuda-nvfp4` の PolarQuant ビルドが `Killed`（Error 137）** | **`kernels.cu`** の **`sm_120a` + FP4 + PolarQuant** コンパイルが RAM 不足 | スワップ増設・並列ビルド停止後に再実行 |
| **`polarquant_init: head_dim must be 128`** | モデルの **`head_dim`** が 128 でない | 現状 **Qwen3-VL-8B** のみ想定。**`gpu-cuda`**（FP16）または PolarQuant 無効ビルドを使用 |
| **`pq-test` FAIL** | コードブック未初期化・GPU 非対応 | **`make pq-test`** の **`max_abs_err` / `rel`** を確認（閾値 rel ≤ 0.35） |
| mmap 失敗 | パス・権限 | `MODEL` を確認 |
| CPU が極端に遅い | IQ 逆量子化コスト | **`cpu-blas`** を試す、ROCm 版の利用、`-n` を小さく |
| **`cpu-blas` ビルド失敗 / `cblas.h` not found** | OpenBLAS 開発パッケージ未導入・ヘッダ非標準パス | **`cd cpu-blas && make openblas`** または **`libopenblas-dev`** を手動インストール。**`CPPFLAGS=-I/.../openblas-pthread`** 等（**`cpu-blas/Makefile`** コメント・ビルド失敗時メッセージ参照） |
| **`cpu-blas` 出力が意味不明（同じ文字の連打等）** | **`-ffast-math`** による IQ 量子化の数値崩れ | リポジトリ同梱 **`cpu-blas/Makefile`** は **`-ffast-math` 無効**。手元で CFLAGS 上書きしている場合は外す |
| `/dev/accel/accel0` を開けない | `render` グループ未参加 / `amdxdna` 未ロード | `sudo usermod -aG render "$USER"`、`lsmod \| grep amdxdna` を確認 |
| XDNA2 で速度が出ない／NPU が効いていない | `XDNA_GEMV_DIR` 未設定、形状欠け、DRM 不可、`XDNA_FORCE_CPU` 等 | 起動時の **`=== XDNA GEMV / NPU ctrlcode status ===`** で各形状の `MISS` を確認。**`./xdna2/qwen3-xdna2 model.gguf --xdna-status`** で軽量診断。推論後の **NPU GEMV / CPU GEMV** カウントが **CPU のみ**なら NPU 経路は実行されていない |
| **`CREATE_HWCTX` が EINVAL（`qwen3-xdna2` / `qwen3-xdna2-bfpx`）** | ドライバが列数・タイル数・QoS・DEV_HEAP・ファームを拒否 | 実装は **`vaddr=0`** で `CREATE_BO` し **必ず `mmap()`** して `userptr` を確立、**QoS は全 0**（`qos_meet` 抵触回避）、`num_tiles` は **`ncol×core.row_count`** を主軸に `core+mem+shim` 合算と `1` を順次フォールバック。なお解消しない場合は **`XDNA_NUM_COL`**・**`XDNA_NUM_TILES`**・**`XDNA_HEAP_SIZE`** 上書きと `dmesg` の `amdxdna` 行（`MAP_HOST_BUFFER status 0x4000003` などのファームエラー、ファーム `amdnpu/<vendor>_<rev>/npu.sbin` の有無）を確認 |
| XDNA2 ビルドで `drm/drm.h` not found | カーネル UAPI ヘッダ未インストール | `apt install linux-libc-dev` 等で `<drm/drm.h>` を導入 |

## 補足：ドキュメント間の役割

- **`README.md`**: ビルド・実行・バリアント選択の手順、ライブラリ非依存の方針とその意義、**マルチターン対話・Thinking モードの対応状況と技術参考 URL**（日本語）。**本文は CPU 3 バリアント**（**`cpu` / `cpu-multicore` / `cpu-blas`**）中心。**ROCm / Vulkan / CUDA / XDNA2** は **付録**（末尾）。
- **`README.en.md`**: 上記と同等の内容（英語）。**公式仕様 vs 本リポジトリ**の対比と外部参考リンクは **「Advanced features (multi-turn chat and Thinking mode)」** に集約。構成は日本語版と同様（CPU 主眼 + GPU/NPU 付録）。
- **`qwen3-8b/xdna2/xdna-gemv/README.md`**: **`qwen3-8b/xdna2/xdna-gemv/`** 配下（**`kernels/`**・**`toolchain/`**・スタブ生成）への入口。
- **`qwen3-8b/xdna2/xdna-gemv/kernels/README.md`**: XDNA2 の **ctrlcode**・GEMV 形状・スタブ／実機バイナリ・**`--xdna-status`** の入門。**GPU（ROCm/HIP）カーネルとの対比**（§3）もここで扱う。
- **`qwen3-8b/xdna2/xdna-gemv/toolchain/README.md`**: **mlir-aie / IRON / Peano / `aiecc`** に沿った **NPU 用 `bf16-gemv-*.bin` 自前生成**の手引き（コマンド列と注意点）。**`qwen3-xdna2` は ioctl のみで XRT 非依存**であること、IRON／mlir-aie の公式サンプルが取る **XRT 検証パス**、および Linux カーネル文書 **AMD NPU** における **`ctrlcode`** の整理を冒頭で対照するための参照になっている。**東京科学大学（2026年現在の名称。旧・東京工業大学）ACRi** ルームの日本語チュートリアル（外部リンク）も紹介される。本文は日本語（です・ます調）。
- **`doc/design.md`（本書）**: 現行の設計・仕様。
- **`doc/ChangeLog.md`**: 日付付き変更履歴。
- **`qwen3-8b/gpu-cuda-nvfp4/DEBUG.md`**: NVFP4 異常出力の調査・修正ログ（**`433319eb`** 起点。FA 診断・**`FP4_QUANT_MAX_ABS`**・ベンチ注意）。
- **外部（参考）**: **AMD XDNA** のアーキテクチャ概要・世代・ソフトウェアスタック等は、別リポジトリ **[thamada/xdna-overview](https://github.com/thamada/xdna-overview)** にまとめてある（本リポジトリの実装説明とは独立した背景資料）。

実装の詳細は **`qwen3-8b/*/main.c`**（経路ごとにディレクトリが分かれ、ファイル名はいずれも **`main.c`**）の先頭コメントとソースを参照する。

## 補足：`design.md` 更新時のチェックリスト

1. **`qwen3-8b/`** にソースまたはターゲットを増やしたら、**構成表**と **バイナリ表**を更新する。  
2. **`Makefile`** の変更と本書を同期する。  
3. 仕様変更は **`doc/ChangeLog.md`** にも記載する。  
4. 利用者向け手順や方針を **`README.md`** で変えたら、 **`README.en.md`** も同趣旨に揃える（またはその逆）。

## 補足：ROCm FP16（`gpu-rocm`）オフラインキャッシュ

**`qwen3-8b/gpu-rocm/`** の FP16 経路。**`gpu-cuda/fp16_cache.h`** / **`fp16_cache_io.c`** / **`gpu-vulkan/fp16_cache_io.c`** と同 API（**`<model>.gguf.fp16`** 形式は共通）。

### ロード（H2D）

1. **`upload_weights_gpu`**（**`main.c`**）: **`max_tensor_nelements` の F32/F16 ステージングは使わない**。
2. **オフラインキャッシュ**（任意）: 既定 **`dir=<model>.gguf.fp16`**。**`manifest`** が一致すれば **`Loading FP16 cache from …`**。各 tensor は **`<dir>/<name>.fp16bin`**（magic **`FPH1`**、version **1**）を **`fp16_host_weight_load`** → H2D。ミス時は GGUF フォールバック（警告 **`FP16 cache miss for …`**）。
3. **GGUF 融合逆量子化**（キャッシュ無効・ミス時）: **`dequant_tensor_row`** → **`upload_fp16_tensor_streaming`**（行単位 FP16 化 → 逐次 H2D）。起動ログ: **`Uploading weights (row dequant -> FP16)...`**。
4. **`token_embd`** と線形 tensor（Q/K/V/O、gate/up/down、LM head）: **`upload_fp16_linear`** が (2) または (3) で **FP16 VRAM** へ。norm 系は **F32 VRAM**。

### オフラインキャッシュ（`pack-cache`）

```bash
cd qwen3-8b/gpu-rocm
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
# または（MODEL 存在時 make build が自動 pack）:
make build
# または:
./qwen3-rocm ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-rocm ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-rocm ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

- 出力: **`token_embd` + L×7 + `output.weight`**（計 **L×7 + 2**）個の **`.fp16bin`** と **`manifest`**。
- **`make run`**（引数なし **`make`**）は **`pack-cache`** を依存（**`manifest`** 欠落・**MODEL / `qwen3-rocm`** 更新時に再生成）。**`make build`** も **`MODEL`** 存在時は **`manifest`** まで生成。
- **`.gitignore`**: **`*.gguf.fp16/`**（**`gpu-cuda`** / **`gpu-vulkan`** と共有形式）。

## 補足：Vulkan compute（`gpu-vulkan`）実装メモ

**`qwen3-8b/gpu-vulkan/`** — ROCm / CUDA を使わず **Vulkan 1.1 compute** のみで GPU 推論。**`gpu-cuda/main.c`** をベースにホスト側を移植し、デバイス側は **`vk_kernels.c`** + GLSL **`shaders/*.comp`**（18 本）。

### アーキテクチャ

| モジュール | 役割 |
|------------|------|
| **`vk_context`** | インスタンス・デバイス・compute キュー |
| **`vk_alloc`** | デバイスバッファ（**`vk_malloc`** ハンドル = 整数 ID。オフセットは push constant **`loff`** 等） |
| **`vk_pipeline`** | SPIR-V ロード、パイプライン作成、**`vk_dispatch`**（descriptor set 確保 → 更新 → dispatch → **fence 待ち** → **解放**） |
| **`vk_kernels`** | **`gpu_model_create`** / **`gpu_forward`** / **`gpu_forward_prefill`** |
| **`gpu.h`** | **`gpu-cuda`** と同趣旨の C API |

### ロード（H2D）・キャッシュ

**`gpu-rocm` / `gpu-cuda`** と同じ FP16 経路。**`<model>.gguf.fp16`** があれば **`.fp16bin`** から H2D。CLI: **`--pack-fp16-cache`** / **`--no-fp16-cache`**。

```bash
cd qwen3-8b/gpu-vulkan
make build
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
QWEN3_VK_SHADER_DIR=$(pwd)/shaders ./qwen3-vulkan ../model.gguf -p "Hello" -n 64
```

### 既知の制約（2026-05-29 時点）

1. **Prefill 線形層** — hipBLAS GemmEx / cuBLAS 相当の **batched GEMM 未実装**（Prefill も **`mm_f16_gemv_batch`** compute シェーダ）。
2. **ホストオーバーヘッド** — レイヤーあたり多数の **`vkCmdDispatch`**。各 dispatch で descriptor set の都度確保（プール **4096 sets**、`vkFreeDescriptorSets` で解放）。
3. **性能** — RADV **GFX1201** 長プロンプト **`make log.push`**（**2026-05-29 09:39**）: prefill **2.97** / decode **2.17** / total **2.86** tok/s（132+16 トークン）。同一条件の **`gpu-rocm`**: **124.95** / **28.88** / **91.89** tok/s。
4. **最適化候補** — descriptor / コマンドバッファ再利用、**`VK_KHR_cooperative_matrix`**、mega-kernel 融合。

## 補足：CUDA FP16（`gpu-cuda`）オフラインキャッシュ

**`qwen3-8b/gpu-cuda/`** の FP16 経路。**`BONSAI_FP4`** は未使用（NVFP4 は **`gpu-cuda-nvfp4/`** が **`../gpu-cuda/main.c`** を **`BONSAI_FP4=1`** で再コンパイル）。

### ロード（H2D）

1. **`upload_weights_gpu`**（**`main.c`**）: **`max_tensor_nelements` の F32/F16 ステージングは使わない**。
2. **オフラインキャッシュ**（任意）: 既定 **`dir=<model>.gguf.fp16`**。**`manifest`** が一致すれば **`Loading FP16 cache from …`**。各 tensor は **`<dir>/<name>.fp16bin`**（magic **`FPH1`**、version **1**）を **`fp16_host_weight_load`** → H2D。ミス時は GGUF フォールバック（警告 **`FP16 cache miss for …`**）。
3. **GGUF 融合逆量子化**（キャッシュ無効・ミス時）: **`dequant_tensor_row`** → **`upload_fp16_tensor_streaming`**（行単位 FP16 化 → 逐次 H2D）。起動ログ: **`Uploading weights (fused dequant -> FP16)...`**。
4. 線形 tensor と **`token_embd`** / **`output.weight`**: **`upload_fp16_linear`** が (2) または (3) で **FP16 VRAM** へ。norm 系は **F32 VRAM**。

### オフラインキャッシュ（`pack-cache`）

```bash
cd qwen3-8b/gpu-cuda
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
# または:
./qwen3-gpu-cuda ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-fp16-cache
./qwen3-gpu-cuda ../model.gguf --pack-fp16-cache /path/to/cache
./qwen3-gpu-cuda ../model.gguf --no-fp16-cache -p "Hello" -n 64
```

- 出力: **`L×7 + 1`** 個の **`.fp16bin`** と **`manifest`**。
- **`.gitignore`**: **`*.gguf.fp16/`**（NVFP4 版は **`*.gguf.nvfp4/`**）。

## 補足：CUDA NVFP4（`gpu-cuda-nvfp4`）実装メモ

**`qwen3-8b/gpu-cuda-nvfp4/`** の Blackwell 向け経路。**`BONSAI_FP4=1`** 固定（**`-DBONSAI_FP4=1`**）。共有ソース（**`main.c` / `kernels.cu` / `gpu.h`**）は **`../gpu-cuda/`** を参照する。

### ロード（H2D）

1. **`upload_weights_gpu`**（**`main.c`**）: 線形 tensor は **F16 全行列ステージングを使わない**。
2. **オフラインキャッシュ**（任意）: 既定 **`dir=<model>.gguf.nvfp4`**。**`manifest`**（GGUF サイズ・mtime）が一致すれば **`Loading NVFP4 cache from …`**。各 tensor は **`<dir>/<name>.fp4bin`**（magic **`NFAQ`**、version **1**）を **`fp4_host_weight_load`** → **`fp4_qwen3_weight_from_host`** で H2D。ミス時は GGUF フォールバック（警告 **`NVFP4 cache miss for …`**）。
3. **GGUF 融合逆量子化**（キャッシュ無効・ミス時）: **`dequant_tensor_row`** が mmap 上の **IQ2_S / IQ3_S / Q4_K / Q5_K / F16 / F32** を行単位 F32 復号 → **`fp4_qwen3_weight_from_rows`** → **GPU `fp4_quantize_weights`**（**`compute_sf_index`** / **`FP4_WEIGHT_SFB_LAYOUT_M=128`**）→ H2D。起動ログ: **`Uploading weights (fused dequant -> NVFP4 linear layers)...`**。
4. 線形 tensor（Q/K/V/O、gate/up/down、LM head）: **`upload_linear_fp4`** が上記 (2) または (3) で **NVFP4 キャッシュ**（`d_fp4` + CUTLASS インターリーブ **`d_sf`**）だけを VRAM に載せる。**線形 FP16 の device バッファは作らない**。
5. **`token_embd.weight`**: **`upload_embd_gpu_streaming`** — F32/F16 は一括 H2D、量子化型は行単位逆量子化 → FP16 H2D。
6. norm 系: **F32 VRAM**。
7. **`gpu_model_create`**（**`kernels.cu`**）: **`wq_fp4` 等**を **`dev_adopt_layers_fp4`** で採用。**`use_fp4=1`**。起動時の FP16→NVFP4 変換ループは **ない**。

### オフラインキャッシュ（`pack-cache`）

```bash
cd qwen3-8b/gpu-cuda-nvfp4
make pack-cache MODEL=../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
# または:
./qwen3-gpu-cuda-nvfp4 ../Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf --pack-nvfp4-cache
./qwen3-gpu-cuda-nvfp4 ../model.gguf --pack-nvfp4-cache /path/to/cache
```

- 出力: **`L×7 + 1`** 個の **`.fp4bin`**（各レイヤーの Q/K/V/O/gate/up/down + **`output.weight`**）と **`manifest`**。
- **`make run`** は **`pack-cache`** を依存（**`manifest`** 欠落・**MODEL / バイナリ** 更新時に再生成）。**`manifest`** 有効時の 2 回目以降はキャッシュから H2D（GGUF 逆量子化・NVFP4 量子化をスキップ）。
- **`--no-nvfp4-cache`**: キャッシュを無視して毎回 GGUF から再量子化。

### 実行（線形層）

**`fp4_qwen3_mm`**（**`fp4_qwen3.cu`**）は推論で **常に CUTLASS NVFP4 GEMM**（**`fp4_gemm_run_cached`**）を使う（prefill / decode で経路を分けない）。

| 段階 | 処理 |
|------|------|
| 入力 | F32 活性 **`x`**（**`M × K`**）を **`f32_to_bf16_pad_kernel`** で **`M_pad × K_pad`** にパッド（**`M_pad = align128(M)`**） |
| 量子化 | GEMM 内で活性を FP4 + ブロックスケール（**`max_abs` は `FP4_QUANT_MAX_ABS=1024` でクランプ**） |
| 演算 | **`fp4_gemm_run_cached`**（重みはキャッシュ済み NVFP4 + **`d_sf_lut`**） |
| 出力 | **`bf16_to_f32_trunc_kernel`** で実 **`M × N`** 行のみ F32 へ |

**`fp4_qwen3_set_gemm_row`** は no-op（互換 API）。

**`kernels.cu`** の **`gpu_mm` / `gpu_mm_batch`** は **`BONSAI_FP4`** 時、線形をすべて **`fp4_qwen3_mm`** に委譲する。起動時に **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** を stderr へ出力。

**診断**: **`--fa-debug`**（**`fa_debug.c`**）で FA CPU replay・KV **`kc`/`vc`**・L6 **`hb`/`down-out`** 等をトレース。**調査の詳細は `DEBUG.md`**（ベースコミット **`433319eb31c3c992536afb5c9a3717084ea5d137`**）。

### ビルド上の注意

- **`MAIN_OBJ`**（**`main.pq$(BONSAI_POLARQUANT).o`**）も **`BONSAI_FP4=1`** でコンパイル（**`gpu-cuda-nvfp4/Makefile`**）。ソースは **`$(SRC_DIR)/main.c`**。
- **`kernels.fabr$(FA_BR).pq$(BONSAI_POLARQUANT).o`** で PolarQuant / FA_BR 変更時の stale `.o` を防止。ソースは **`$(SRC_DIR)/kernels.cu`**。
- **`fp4_qwen3_init`**: モデルロード前に **128³** smoke GEMM で CUTLASS／workspace を検証。失敗時は **`NVFP4 sanity GEMM failed`** で終了。
- **`FP4_CACHE_VERSION=2`**: 重みキャッシュに SFB + **`d_sf_lut`** を含む。v1 **`.fp4bin`** は読み込まない — **`make pack-cache`** で再生成。
- **`FP4_WEIGHT_SFB_LAYOUT_M=128`**: 重みキャッシュの SFB レイアウトと **`fp4_gemm_run_cached`** の **`layout_SFB`** 参照（LUT 経由）を一致させる。**`make sfa-verify`** で索引式を検証。
- **`fp4_gemm_vram_bytes`** / **`fp4_qwen3_vram_bytes`**: **`gpu_model_vram_profile`** の **`fp4_gemm_scratch`** 見積もりに使用。
- **`fp4_gemm.sm120a.o`** / **`fp4_qwen3.sm120a.o`** は **`CUDA_GENCODE`** とは独立し、常に **`BLACKWELL_NVCCFLAGS`**（**`-std=c++17`** + **`arch=compute_120a,code=sm_120a`**）でビルドする。CUTLASS NVFP4 は **C++17 必須**。CUTLASS NVFP4 の sm_120 MMA を **`compute_86` PTX** だけに載せると実行時 abort する。
- **CUTLASS バージョン**: **`CUTLASS_TAG=v4.5.0`**（旧 **`v3.9.0`** から更新）。初回またはタグ変更後は **`make cutlass`**（**`third_party/cutlass`** が無い／タグ不一致なら clone／再 clone）。
- **CUDA 13 非推奨ベクトル型（`long4` 等）**: v4.5.0 の **`platform.h`** で CUTLASS 側が解消済み。旧版で必要だった **`-Wno-deprecated-declarations`** は不要。
- **nvcc 警告 #20012**（**`= default` コンストラクタ + `__device__`/`__host__`**）: v4.5.0 の **`sm100_static_tile_scheduler.hpp`** が CUTLASS 側コードとして発する。**`third_party/cutlass`** は当リポジトリ側では改変しない。CUTLASS 公式 nvcc ビルド（**`python/cutlass_cppgen/backend/compiler.py`**）と同じ **`-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored`** を **`BLACKWELL_NVCCFLAGS`** に付ける。背景・参考 URL は **`gpu-cuda-nvfp4/Makefile`** コメントに記載。
- 検証: **`make fp4-test`**（経路一致 + **`run_batch_row_parity`** M=20 + **`run_extreme_act`**）、**`make sfa-verify`**（SFA 索引）、**`make fa-debug`**（Hello・FP16/NVFP4 比較）。**Blackwell GPU 必須**。

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

**`gpu-cuda-nvfp4` + `build.polarquant`** では上記に加え **`Loading NVFP4 cache from …`** または **`Uploading weights (fused dequant -> NVFP4 linear layers)...`** と **`GPU: FP4 Tensor Core GEMM path enabled (prefill + decode)`** も出る。

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
