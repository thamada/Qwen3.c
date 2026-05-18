# SPDX-License-Identifier: Apache-2.0
# Download bf16-gemv-<n>x<d>.bin (real NPU ctrlcode), not GQF3XDNA stubs.
#
# 配布ベース URL の末尾は / がなくても可（Makefile が補う）。対象ファイル名は次のとおり置くこと。
#   bf16-gemv-4096x4096.bin
#   bf16-gemv-4096x1024.bin
#   bf16-gemv-4096x12288.bin
#   bf16-gemv-12288x4096.bin
#   bf16-gemv-4096x151936.bin
#
# 既定の配布ベース URL（環境変数または make の変数代入で上書き可）。
# 例: make XDNA_GEMV_BIN_URL_BASE=https://mirror.example.jp/gemv/
#
# Releases にアセットを載せない場合、この行の URL が 404 のままになるため、そのときは手元サーバ等へ置き換えてください。
XDNA_GEMV_BIN_URL_BASE ?= https://github.com/thamada/Qwen3.c/releases/download/xdna-gemv-qwen3vl-8b/

.DEFAULT_GOAL := all

# このディレクトリの親に gen-xdna-gemv-stubs.py がある。SHAPES は qwen3-8b/xdna2/main.c のテキスト経路と一致させること
GEMV_BINARIES := \
	bf16-gemv-4096x4096.bin \
	bf16-gemv-4096x1024.bin \
	bf16-gemv-4096x12288.bin \
	bf16-gemv-12288x4096.bin \
	bf16-gemv-4096x151936.bin

CURL  := $(shell command -v curl 2>/dev/null)
WGET  := $(shell command -v wget 2>/dev/null)

.PHONY: all help check-url

all: check-url $(GEMV_BINARIES)

help:
	@echo '本物の ctrlcode (.bin) をまとめてダウンロードします。'
	@echo ''
	@echo '  make                           # Makefile 内の既定 URL から取得'
	@echo '  make XDNA_GEMV_BIN_URL_BASE=URL/ # 別ミラーへ上書き'
	@echo ''
	@echo '既定ベース URL:'
	@echo '  $(strip $(XDNA_GEMV_BIN_URL_BASE))'
	@echo ''
	@echo '末尾は / がなくても可。上記 5 ファイルがその場所にあること。依存: curl または wget'

check-url:
	@if [ -z "$(strip $(XDNA_GEMV_BIN_URL_BASE))" ]; then \
		echo >&2 'エラー: XDNA_GEMV_BIN_URL_BASE が空です。'; \
		echo >&2 '  Makefile 先頭の既定 URL を復元するか、make の変数で指定してください。'; \
		exit 1; \
	fi
	@if [ -z "$(CURL)" ] && [ -z "$(WGET)" ]; then \
		echo >&2 'エラー: curl または wget が見つかりません。'; \
		exit 1; \
	fi

$(GEMV_BINARIES): check-url
	@echo "Downloading $@ ..."
	@BASE="$(XDNA_GEMV_BIN_URL_BASE)"; \
	case "$$BASE" in */) ;; *) BASE="$$BASE/" ;; esac; \
	if [ -n "$(CURL)" ]; then \
		$(CURL) -fSL --retry 3 --retry-delay 2 \
			-o "$@" "$$BASE$@"; \
	elif [ -n "$(WGET)" ]; then \
		$(WGET) -O "$@" "$$BASE$@" || exit $$?; \
	fi
	@SZ=$$(wc -c < "$@" | tr -d ' '); \
	if [ "$$SZ" -le 128 ]; then \
		echo >&2 "警告: $@ のサイズが $$SZ バイトのみです。スタブかダウンロード失敗の可能性があります。"; \
	fi
