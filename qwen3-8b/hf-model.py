#!/usr/bin/env python3
"""
Hugging Face から gguf.txt に記載されたモデル（blob/resolve URL）をダウンロードし、
Hub API の LFS メタデータと SHA256 を照合する。

前提:
  - `hf` CLI が PATH 上にあること（例: pip install -U huggingface_hub）

認証:
  スクリプトは `hf auth login` を実行する。
  非対話利用は環境変数 HF_TOKEN（または --token）を設定する。

使用例:
  python3 hf-model.py
  python3 hf-model.py --gguf-txt /path/to/gguf.txt --local-dir ./models
  HF_TOKEN=hf_xxx python3 hf-model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HF_BLOB_OR_RESOLVE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """(repo_id, revision, file_path_within_repo) を返す。file_path は URL デコード済みのスラッシュ区切り。"""
    url = url.strip()
    m = HF_BLOB_OR_RESOLVE.match(url)
    if not m:
        raise ValueError(
            "想定する URL 形式: https://huggingface.co/<org>/<repo>/blob/<rev>/<path> "
            "または .../resolve/<rev>/<path>"
        )
    repo = m.group("repo")
    rev = m.group("rev")
    path_in_repo = urllib.parse.unquote(m.group("path"))
    return repo, rev, path_in_repo


def read_url_from_gguf_txt(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    raise ValueError(f"{path}: 有効な URL 行が見つかりません")


def hub_tree_api_url(repo_id: str, revision: str) -> str:
    # repo_id は org/name のスラッシュをそのまま使う（%2F にしない）
    rev_q = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo_id}/tree/{rev_q}?recursive=true"


def fetch_lfs_sha256(
    repo_id: str, revision: str, file_path: str, token: str | None
) -> str:
    url = hub_tree_api_url(repo_id, revision)
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data: Any = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hub tree API 失敗 HTTP {e.code}: {body[:500]}") from e

    if not isinstance(data, list):
        raise RuntimeError(f"Hub tree API の応答形式が不正です: {type(data)}")

    for entry in data:
        if entry.get("type") != "file":
            continue
        if entry.get("path") != file_path:
            continue
        lfs = entry.get("lfs") or {}
        oid = lfs.get("oid")
        if isinstance(oid, str) and len(oid) == 64:
            return oid.lower()
        raise RuntimeError(f"{file_path}: LFS oid が必要ですがメタデータにありません: {entry}")
    raise RuntimeError(f"{file_path}: リポジトリ {repo_id}@{revision} のツリーに見つかりません")


def file_sha256_hex(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_auth_login(token: str | None) -> None:
    cmd = ["hf", "auth", "login"]
    if token:
        cmd.extend(["--token", token])
    subprocess.run(cmd, check=True)


def hf_download(
    repo_id: str, revision: str, filenames: list[str], local_dir: Path, token: str | None
) -> None:
    cmd = [
        "hf",
        "download",
        repo_id,
        *filenames,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
    ]
    if token:
        cmd.extend(["--token", token])
    subprocess.run(cmd, check=True)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="HF から GGUF を dl して SHA256 検証")
    ap.add_argument(
        "--gguf-txt",
        type=Path,
        default=here / "gguf.txt",
        help="1 行目に Hugging Face blob/resolve URL があるテキスト（既定: スクリプトと同じ場所の gguf.txt）",
    )
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=here,
        help="保存先ディレクトリ（既定: スクリプトと同じディレクトリ）",
    )
    ap.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="省略時は環境変数 HF_TOKEN を使用（いずれも無ければ対話ログイン）",
    )
    ap.add_argument(
        "--skip-login",
        action="store_true",
        help="既にログイン済みなら hf auth login をスキップ",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="SHA256 照合をスキップ（非推奨）",
    )
    args = ap.parse_args()

    token: str | None = args.token
    if token is not None:
        token = token.strip() or None
    url_line = read_url_from_gguf_txt(args.gguf_txt)
    repo_id, revision, file_path = parse_hf_url(url_line)
    base_name = Path(file_path).name

    if not args.skip_login:
        hf_auth_login(token)

    args.local_dir.mkdir(parents=True, exist_ok=True)

    expected = None
    if not args.skip_verify:
        expected = fetch_lfs_sha256(repo_id, revision, file_path, token)
        print(f"期待 SHA256 (Hub LFS): {expected}")

    hf_download(repo_id, revision, [file_path], args.local_dir, token)

    out_file = args.local_dir / base_name
    if not out_file.is_file():
        # hf がサブディレクトリに置く場合へのフォールバック
        alt = args.local_dir / file_path
        if alt.is_file():
            out_file = alt
        else:
            print(
                f"エラー: ダウンロード後に期待ファイルが見つかりません: {out_file} / {alt}",
                file=sys.stderr,
            )
            return 1

    got = file_sha256_hex(out_file)
    print(f"実測 SHA256:           {got}")
    print(f"保存先: {out_file.resolve()}")

    if expected is not None and got != expected:
        print("SHA256 が一致しません。", file=sys.stderr)
        return 2

    if expected is not None:
        print("SHA256 チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
