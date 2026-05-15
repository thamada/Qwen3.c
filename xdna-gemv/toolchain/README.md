# XDNA GEMV 用 ctrlcode を自前ビルドする手引き（公式ツールチェイン）

このディレクトリの文書は、**本リポジトリの `qwen3-xdna2`** が環境変数 `XDNA_GEMV_DIR` から読み込む **`bf16-gemv-<n>x<d>.bin`** を、公開されている **MLIR-AIE／IRON／Peano** 一式を揃えて生成するところまでの入口です。このファイルは XAie のトランザクション列に相当するバイナリを想定しています。

実装側の説明は `qwen3-8b/xdna2/main.c` の `load_gemv_kernel` 付近のコメント、および形状リストがある [`xdna-gemv/gen-xdna-gemv-stubs.py`](../gen-xdna-gemv-stubs.py) を参照してください。

---

## 0. 最初にお読みください（重要）

### 本リポジトリ（Qwen3.c）のランタイム

- **`qwen3-xdna2`** は、**XRT なし**で動きます。裏では **`amdxdna` の DRM ioctl だけ**を使っています。
- 一方で **IRON／mlir-aie の公式サンプル**は、**XRT と Peano でデザインをビルドし、ホストから NPU にロードして検証する**のが一般的です（[Xilinx/mlir-aie README — Getting Started](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux)、[AMD IRON README](https://github.com/amd/IRON/blob/devel/README.md)）。

Linux カーネル公式の [**AMD NPU**](https://docs.kernel.org/accel/amdxdna/amdnpu.html) では、**ドライバとファームウェアまわりの全体像**が整理されています。クライアント APU 内蔵の NPU は **`amdxdna`** が管理し、ワークロードは **XDNA Array のオーバーレイ**（空間パーティションの設定など）と **`ctrlcode`**（配下のオーケストレーション）の **2 種類のバイナリ**から成ります。**`ctrlcode`** はマイコン上の ERT で **`XAie_TxnOpcode` 列**として実行され、そのあいだにホスト DDR と L2 の間で DMA が走ります。各コンテキストは **命令用ホストバッファ**（ドキュメントでは例として 64 MiB 程度）へ `ctrlcode` を写してから、メールボックス経由で渡します。`qwen3-xdna2` が XRT なしで ioctl に渡しているバイナリが、ここで説明される **`ctrlcode`** に相当する、という整理を押さえるのに便利です。

東京科学大学（2026年現在の名称。旧・東京工業大学）ACRi ルームの [**Ryzen NPU の利用方法**](https://gw.acri.c.titech.ac.jp/wp/manual/ryzen-npu)（リンクは従来どおり `titech.ac.jp`）は、**浮動小数ベクトル加算（`vadd`）** を題材にした日本語チュートリアルで、手を動かしやすくまとまっています。同ルームの **Ryzen NPU サーバー**（記事ではホスト名 `ds001`）向けに、AIE カーネル（C++／Peano）、mlir-aie の **IRON（`aie.iron`）** による AIE プログラム（Python から MLIR を生成）、`aiecc` での **`xclbin` と NPU 向け命令バイナリ（例: `vadd_inst.bin`）** の生成まで、さらに **XRT を使うホスト（C++）からのロードと検証**まで、コマンド例つきで追えるようになっています。文中のパス（例: `/tools/repo/Xilinx/mlir-aie/`）はその環境での一例にすぎないので、自前のマシンでは本 README や mlir-aie／XRT の公式手順に沿って読み替え、同じ「カーネル → MLIR → `aiecc` → バイナリ → ホスト実行」の流れに乗れば十分です。ツールチェインに初めて触れる人にとっても、頼りになる参照になるはずです。**執筆の安藤潤さん**には、公開サーバーでの実践を丁寧に文章化していただきました。ここに敬意と感謝をお伝えします。

以上を前提に、本書の流れはおおむね次の二段になります。

1. **開発用マシンに、XRT と mlir-aie／IRON が求める構成をそろえ、`aiecc` や IRON が出力する「NPU 向けバイナリ」の意味を把握する。**  
2. **そのうち、`ERT_START_NPU` に載せられる「txn／NPU insts」に相当する部分だけを `bf16-gemv-*.bin` として取り出し、`XDNA_GEMV_DIR` に置く。**  
   ここで、公式が「この C ランタイム用に、このファイル名でこの中身を置け」と一本化した文書が必ずあるわけではないため、**`aiecc` の出力と `qwen3-8b/xdna2/main.c` が期待するレイアウトが一致するかは、ビルドごとに確認・調査が必要**です。

**正直な整理:** 以下のコマンド列は、[Xilinx mlir-aie の README](https://github.com/Xilinx/mlir-aie)、[mlir-aie のサイト](https://xilinx.github.io/mlir-aie/)、[AMD IRON](https://github.com/amd/IRON) が示す標準的なセットアップに沿っています。**最終的に `bf16-gemv-<n>x<d>.bin` と機械的に対応づける自動スクリプトは、本リポジトリにはまだ含めていません**（環境やアーキテクチャの差をこちらで吸収しきれないためです）。

---

## 1. Qwen3-VL-8B（テキスト経路）で必要なファイル名と GEMV の形

対応する実ファイルは **5 本**です（実装および [`gen-xdna-gemv-stubs.py`](../gen-xdna-gemv-stubs.py) と同じです）。

| ファイル名（`n` = 入力ベクトル長、`d` = 出力ベクトル長） | 主に対応する重み |
|------------------------------------------------------------|------------------|
| `bf16-gemv-4096x4096.bin` | 注意機構の Q／WO など dim×dim |
| `bf16-gemv-4096x1024.bin` | 注意機構の K／V など dim×kv_dim |
| `bf16-gemv-4096x12288.bin` | FFN の gate／up など dim×hidden |
| `bf16-gemv-12288x4096.bin` | FFN の down など hidden×dim |
| `bf16-gemv-4096x151936.bin` | lm_head など dim×vocab |

### AMD IRON の `GEMV` オペレータと行列の向き

[IRON の `iron/operators/gemv/op.py`](https://github.com/amd/IRON/blob/devel/iron/operators/gemv/op.py) では、ランタイム引数として次の形が定義されています。

- 行列 **`(M, K)`**
- ベクトル **`(K,)`**
- 出力 **`(M,)`**

`qwen3-xdna2` の線形層では、重み行列を **行 `d` × 列 `n`**、入力の長さ `n`、出力の長さ `d` として **y = W x** を計算します。IRON 側ではこれに対応させるため、**`M = d`、`K = n`** と設定すればよいです。

IRON には `K >= kernel_vector_size` かつ `K % kernel_vector_size == 0`（既定で `kernel_vector_size = 64`）という制約があります。上記 5 形状はいずれも **`K` が 64 の倍数**なので、既定のまま GEMV を構成できる想定です。

---

## 2. 開発ホストの前提（Ryzen AI／XDNA で IRON が想定する条件）

以下は [Xilinx mlir-aie README — Getting Started for AMD Ryzen™ AI on Linux](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux) に基づきます。

- **OS:** 手順書の標準は **Ubuntu 24.04 または 24.10** です。
- **カーネル:** README では、**Ubuntu 24.04 では 6.11 以降が必要になる場合がある**旨と、`linux-generic-hwe-24.04` を入れてから再起動する例が載っています。また **PPA 経由パッケージの節では Linux 6.17 以降が前提**と読める記述があります。まずは現在のカーネルを確認してください。

```bash
uname -r
```

不足していれば、[README の Initial Setup](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux) に従い HWE で上げ、`sudo reboot` してください。

- **BIOS:** NPU が有効であること。**Secure Boot を無効にする**案内があります（モジュール読み込みの都合）。
- **`render` グループ:** NPU にアクセスするユーザーに付与します。

```bash
sudo usermod -aG render "$USER"
# グループを反映するため、ログアウト／ログインし直してください
```

---

## 3. XDNA ドライバと XRT（公式パッケージの例）

[mlir-aie README — Install the XDNA™ Driver and XRT](https://github.com/Xilinx/mlir-aie#install-the-xdna-driver-and-xrt)（Ubuntu 24.04 でカーネル 6.17 以上を満たす場合）に従う例です。

```bash
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
sudo reboot
```

再起動後、環境を読み込み、デバイスを確認します。

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

出力の末尾付近に NPU が列挙されることを確認してください（README では `[0000:..] : NPU Strix` のような例があります）。

**独自ビルドのカーネルやほかのディストリビューションで上記 PPA が使えない場合:** 同じ README の [Alternative: Build XDNA™ Driver and XRT from source](https://github.com/Xilinx/mlir-aie#alternative-build-xdna-driver-and-xrt-from-source) に、`mlir-aie/utils/build_drivers.sh` を使う流れがあります（`sudo` が必要で、再起動を求められることがあります）。

---

## 4. mlir-aie（IRON の共通基盤）として Python／wheel／Peano をそろえる

[Xilinx mlir-aie README — Install IRON for AMD Ryzen™ AI AIE Application Development](https://github.com/Xilinx/mlir-aie#install-iron-for-amd-ryzen-ai-aie-application-development) に沿います。**pip の wheel と Git のコミット、手順書の版を揃える**必要があります（README でも強調されています）。

作業ディレクトリは任意です（このリポジトリの外に clone してかまいません）。

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
```

Python の仮想環境を作成します。

```bash
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
```

ディストリビューション付属の CMake が古い場合は、README の注記どおり、仮想環境側に CMake を入れます。

```bash
python3 -m pip install --upgrade 'cmake>=3.30'
```

**mlir-aie の wheel と、clone したリポジトリのコミットは対応させる**（README の注意どおりです）。

**最新の wheel に追従する例**（README にあるコマンド例）:

```bash
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3
```

**安定版リリースに固定する場合**は、README のとおり、`releases/latest` などからタグを取得し、`mlir_aie==<該当バージョン>` と **`git checkout` で同一タグ**を組み合わせます。

**Peano（llvm-aie）の wheel:**

```bash
python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

**環境セットアップスクリプト**（mlir-aie リポジトリ内）:

```bash
cd ~/src/mlir-aie
source utils/env_setup.sh
```

（`PATH` や `PYTHONPATH` などが調整されます。**新しいシェルを開くたびに** `source ironenv/bin/activate`、`source utils/env_setup.sh`、`source /opt/xilinx/xrt/setup.sh` が必要になります。）

mlir-aie 側の開発用テストに必要な依存だけ入れる場合（任意）は、README に従ってください。

```bash
python3 -m pip install -r python/requirements_dev.txt
```

---

## 5. AMD IRON リポジトリの入手とセットアップ

[AMD IRON README（Installation Linux）](https://github.com/amd/IRON/blob/devel/README.md#installation-linux) は、**Ubuntu 24.04／24.10 をほぼ素の状態から** Ryzen AI NPU で動かす前提で書かれており、上流の mlir-aie README と同系統です。

手順の概略は次のとおりです。

```bash
cd ~/src
git clone https://github.com/amd/IRON.git
cd IRON
# 開発ブランチに追従する例（README に従ってください）
git checkout devel

source ~/src/mlir-aie/ironenv/bin/activate
source ~/src/mlir-aie/utils/env_setup.sh
source /opt/xilinx/xrt/setup.sh

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**動作確認の例（IRON README の単体サンプル `axpy`）:**

```bash
cd ~/src/IRON
pytest ./iron/operators/axpy/
```

**GEMV がビルドされ実行できるか**（ご自身の環境のスモークテスト）:

```bash
pytest ./iron/operators/gemv/
```

ここまで通れば、IRON が **GEMV を含む mlir-aie のパイプラインでビルドし、XRT 経由で NPU に載せる**状態になっているはずです。ただし **`pytest` が生成する成果物の名前や構造が、本リポジトリの `bf16-gemv-*.bin` とそのまま一致するとは限りません**。§7 で述べる `aiecc` の出力と突き合わせる作業が別途必要になります。

---

## 6. IRON の GEMV と Qwen の 5 形状の対応（メモ）

| `bf16-gemv-<n>x<d>.bin` の `n` と `d` | IRON `GEMV(M,K)` の **`M` と `K`** |
|---------------------------------------|--------------------------------------|
| 4096×4096                             | `M=4096`, `K=4096`                  |
| 4096×1024                             | `M=4096`, `K=1024`                  |
| 4096×12288                            | `M=4096`, `K=12288`                 |
| 12288×4096                            | `M=12288`, `K=4096`                 |
| 4096×151936                           | `M=4096`, `K=151936`                |

IRON の `GEMV` には **`num_aie_columns`**（使う列数）やタイル関連のパラメータがあります。モデルや性能目標に応じて、**列数を増やす**といった調整もありえます。**デバイス名**（Phoenix／Strix に対応する `npu1`／`npu2` など）は [mlir-aie の Devices ドキュメント](https://github.com/Xilinx/mlir-aie/blob/main/docs/Devices.md) と [公式サイト](https://xilinx.github.io/mlir-aie/) を参照してください（具体的なフラグ値はプラットフォームごとに異なります）。

---

## 7. `aiecc` で NPU 命令トランザクション（.bin）を明示的に出すオプション

[mlir-aie の `tools/aiecc/README.md`](https://github.com/Xilinx/mlir-aie/blob/main/tools/aiecc/README.md) には、入力 MLIR に対して少なくとも次のフラグが記載されています。

- **`--aie-generate-npu-insts`** … NPU 命令の TXN 列を生成する  
- **`--npu-insts-name`** … 出力ファイル名のパターン（既定は `{0}_{1}.bin`）

例として、コンパイル対象の `design.mlir` がある場合のイメージです（パスは環境に合わせて読み替えてください）。

```bash
# 仮想環境の有効化と env_setup.sh、XRT の setup.sh は事前に source しておいてください
which aiecc   # mlir-aie が PATH に載っていることを確認

aiecc --verbose --aie-generate-npu-insts design.mlir
# ランタイムシーケンスが複数ある設計では、README に記載のとおり --sequence-name で絞り込める場合があります
```

**本リポジトリが期待する `bf16-gemv-*.bin` が、このトランザクションバイナリと単一の表で完全一致すると証明できるわけではありません**。実運用では例えば次のように進めることになります。

1. mlir-aie／IRON のサンプルを `--aie-generate-npu-insts` でビルドし、出力ファイルのサイズや先頭バイトなどを記録する。  
2. `qwen3-xdna2` に **スタブではない**そのバイナリを渡し、`EXEC_CMD` が成功するか、`--xdna-status` と推論ログの **NPU GEMV 回数**で確認する。  
3. 必要に応じて [amdnpu.rst（xdna-driver の説明）](https://github.com/amd/xdna-driver/blob/main/src/driver/doc/amdnpu.rst)、Linux カーネル文書の [**AMD NPU**](https://docs.kernel.org/accel/amdxdna/amdnpu.html)（Application Binaries／High-level Use Flow など）、または mlir-aie の Issue などで、**txn と制御コードのレイアウト**を照らし合わせる。

**オーバーレイ（`CONFIG_HWCTX`）との関係:** `qwen3-8b/xdna2/main.c` の冒頭コメントでは、`overlay` と `ctrlcode` の**ペア**が必要となる旨が述べられている箇所もあります。`bf16-gemv-*.bin` だけでは足りず、mlir-aie が生成する PDI や xclbin との整合まで必要になる場合があります。**すべてを一致させる作業は、統合側の開発者の責務**となります。オーバーレイと `ctrlcode` の分担は、上記カーネル文書の *Application Binaries* でも公式に定義されています。

別ルートとして `aiecc` には **`--aie-generate-txn`** や **`--aie-generate-ctrlpkt`** もあり、構成や世代によって **txn と制御パケットが分かれる**ことも README にあります。**どの出力が、ご自分のプラットフォーム・ファームウェアでの `ERT_START_NPU` に適合するかは環境依存**です。

---

## 8.（任意）クローズドツールチェイン：Vitis AIE Essentials

[mlir-aie README — Optional: Install AIETools](https://github.com/Xilinx/mlir-aie#optional-install-aietools) にあるとおり、**AIE2／AIE2P をオープンソースの Peano だけで十分な場合は省略してかまわません**。  
**`xchesscc` を含むクローズドなツールチェイン**で AIE コアのコンパイルを行いたい場合は、README が案内する **Ryzen AI Software（EA）から Vitis™ AIE Essentials** をインストールし、`AIETOOLS_ROOT` とライセンス **`LM_LICENSE_FILE`** を設定し、**xchesscc でコアをビルドする**構成になります。本リポジトリ側の説明では **既定で Peano あればソース中心で足りる**前提と読めるため、両者のどちらを選ぶかは用途に応じて判断してください。

---

## 9. 成果物を `qwen3-xdna2` に渡す手順の例（ビルド済みである場合）

検証済みの `.bin` を **`xdna-gemv/kernels/`** に置いたと仮定した例です。

```bash
# 例（パスはご自身の clone 場所に合わせてください）
export XDNA_GEMV_DIR="/absolute/path/to/gguf.Qwen3.c/xdna-gemv/kernels"

ls -la "$XDNA_GEMV_DIR"/bf16-gemv-*.bin

cd /absolute/path/to/gguf.Qwen3.c/qwen3-8b
./xdna2/qwen3-xdna2 /path/to/model.gguf --xdna-status
```

- **`[STUB]`** のときは、スタブか先頭マジック `GQF3XDNA` のファイルであり、NPU には載りません。  
- **非スタブで `[ OK ]`** でも必ずしも互換があるとは限りません。続けて短い生成（例: 小さな `-n`）で推論し、**NPU GEMV のカウンタ**で確認してください。

---

## 10. 参照リンク（調べるときの優先順）

| 資料 | URL |
|------|-----|
| Xilinx mlir-aie（Ryzen AI 向けセットアップの本体） | [https://github.com/Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) |
| mlir-aie ドキュメントサイト | [https://xilinx.github.io/mlir-aie/](https://xilinx.github.io/mlir-aie/) |
| mlir-aie `aiecc`（NPU txn 関連フラグ） | [tools/aiecc/README.md](https://github.com/Xilinx/mlir-aie/blob/main/tools/aiecc/README.md) |
| デバイス一覧（npu1／npu2 など） | [docs/Devices.md](https://github.com/Xilinx/mlir-aie/blob/main/docs/Devices.md) |
| AMD IRON | [https://github.com/amd/IRON](https://github.com/amd/IRON) |
| IRON の GEMV（`GEMV(M,K)` の定義） | [iron/operators/gemv/op.py](https://github.com/amd/IRON/blob/devel/iron/operators/gemv/op.py) |
| Peano llvm-aie | [https://github.com/Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) |
| xdna-driver ドキュメント（ctrlcode と DMA の背景） | [amdnpu.rst](https://github.com/amd/xdna-driver/blob/main/src/driver/doc/amdnpu.rst) |
| Linux カーネル文書：AMD NPU（`amdxdna`／overlay／`ctrlcode`／ERT） | [AMD NPU](https://docs.kernel.org/accel/amdxdna/amdnpu.html) |
| 本リポジトリ側の GEMV 形状（スタブ生成） | [`xdna-gemv/gen-xdna-gemv-stubs.py`](../gen-xdna-gemv-stubs.py) |
| 東京科学大学 ACRi ルーム（2026年現在の名称。旧・東京工業大学）：Ryzen NPU サーバー上の vadd チュートリアル（日本語。執筆：安藤潤氏） | [Ryzen NPU の利用方法](https://gw.acri.c.titech.ac.jp/wp/manual/ryzen-npu) |

---

本書の多くは、オープンソースの README／ツールのヘルプを**転記・要約**したものです。ツールチェイン側のアップストリームの変更にあわせて内容を更新してください。本リポジトリへの自動ビルドスクリプトの追加を検討する場合は、Issue や Pull Request で相談するとよいでしょう。
