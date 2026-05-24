MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf

.PHONY: model

# GGUF モデル（gguf.txt の URL からダウンロード）
model:
	@checksum_ok_banner() { \
		subtitle="$$1"; \
		echo ""; \
		printf '\033[1;32m'; \
		echo "+==============================================================+"; \
		echo "| ***  SHA256 INTEGRITY VERIFIED - ALL CLEAR  ***              |"; \
		echo "+==============================================================+"; \
		echo "|                                                              |"; \
		echo "|      ***   CHECKSUM VERIFICATION PASSED   ***                |"; \
		echo "|                                                              |"; \
		echo "+--------------------------------------------------------------+"; \
		echo "| [OK] File integrity confirmed - safe to use                  |"; \
		echo "+==============================================================+"; \
		printf '\033[0m'; \
		echo ""; \
		printf '\033[1m  File:\033[0m     \033[1;32m$(MODEL)\033[0m\n'; \
		printf '\033[1m  Expected:\033[0m $(MODEL).sha256sum\n'; \
		echo ""; \
		if [ -n "$$subtitle" ]; then \
			printf '\033[1;32m  > %s\033[0m\n' "$$subtitle"; \
			echo ""; \
		fi; \
	}; \
	if [ ! -f "$(MODEL).sha256sum" ]; then \
		echo "Error: checksum file '$(MODEL).sha256sum' not found."; \
		echo "Update the repository to the latest version, or verify the file exists."; \
		exit 1; \
	fi; \
	if [ -f "$(MODEL)" ] && sha256sum --check "$(MODEL).sha256sum" >/dev/null 2>&1; then \
		checksum_ok_banner "Skipping download - model already present."; \
	else \
		url=$$(sed 's|/blob/main/|/resolve/main/|' gguf.txt); \
		wget -O "$(MODEL)" "$$url" && \
		if sha256sum --check "$(MODEL).sha256sum"; then \
			checksum_ok_banner "Model download and verification complete."; \
		else \
			echo ""; \
			printf '\033[1;31m'; \
			echo "+==============================================================+"; \
			echo "|        [FAIL]  CHECKSUM VERIFICATION FAILED  [FAIL]          |"; \
			echo "+==============================================================+"; \
			printf '\033[0m'; \
			echo ""; \
			echo "  File:     $(MODEL)"; \
			echo "  Expected: $(MODEL).sha256sum"; \
			echo ""; \
			echo "The file may have been corrupted during download."; \
			echo "Removing the corrupted file. Check your network connection, then run"; \
			echo "  make model"; \
			echo "again."; \
			rm -f "$(MODEL)"; \
			exit 1; \
		fi; \
	fi
