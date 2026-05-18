# `xdna-gemv/` — NPU GEMV 用 ctrlcode（`bf16-gemv-*.bin`）の準備

**`xdna2/qwen3-xdna2`** が環境変数 **`XDNA_GEMV_DIR`** で探す **`bf16-gemv-<n>x<d>.bin`** まわりの、**ドキュメント・取得用 Makefile・スタブ生成スクリプト**は、このディレクトリにまとめてあります（推論ソースは **`qwen3-8b/xdna2/main.c`**）。

## 構成

| パス | 内容 |
|------|------|
| **`kernels/`** | ランタイムが参照する既定の親ディレクトリ想定。**`README.md`**（初学者向け）、**`Makefile`**（ミラーから一括ダウンロード）、および（再生成すると）スタブ／本物の **`.bin` ファイル**が置かれる場所。 |
| **`toolchain/`** | **MLIR-AIE / AMD IRON / Peano / `aiecc`** で本物を自前ビルドする際の詳細手引き（**`README.md`**）。 |
| **`gen-xdna-gemv-stubs.py`** | リポジトリ同梱用の **64 バイトスタブ `bf16-gemv-*.bin`** を出力（マジック `GQF3XDNA`。NPU には載りません）。 |

## 最短の状態確認・スタブ再生成（リポジトリルートから）

```bash
python3 qwen3-8b/xdna2/xdna-gemv/gen-xdna-gemv-stubs.py
export XDNA_GEMV_DIR="$(pwd)/qwen3-8b/xdna2/xdna-gemv/kernels"
cd qwen3-8b && ./xdna2/qwen3-xdna2 /path/to/model.gguf --xdna-status
```

または **`qwen3-8b`** で `make gen-xdna-kernels`。

詳細は **`kernels/README.md`** と **`toolchain/README.md`** を参照してください。
