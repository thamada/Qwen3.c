#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate bf16-gemv-<n>x<d>.bin placeholder files for qwen3-xdna2 status checks.

These binaries are NOT executable control microcode. They carry an 8-byte magic
``GQF3XDNA`` so ``qwen3-8b/xdna2/main.c`` rejects them for ERT_START_NPU and falls back
to CPU. Replace them with MLIR-AIE / IRON produced txn blobs for real NPU GEMV.

Usage:
  python3 xdna-gemv/gen-xdna-gemv-stubs.py [output_dir]

Default output_dir: ./xdna-gemv/kernels (repository root relative).
"""
from __future__ import annotations

import os
import struct
import sys

# Must match qwen3-8b/xdna2/main.c and qwen3-8b/xdna2-bfp16/main.c
MAGIC = b"GQF3XDNA"
VERSION = 1
FLAG_PLACEHOLDER = 1

# Unique (n, d) for Qwen3-VL-8B text path (dim=4096, hidden=12288, ...)
SHAPES = [
    (4096, 4096),    # wq, wo
    (4096, 1024),    # wk, wv
    (4096, 12288),   # gate / up
    (12288, 4096),   # down
    (4096, 151936),  # lm_head
]

STUB_HEADER_BYTES = 64


def write_stub(path: str, n: int, d: int) -> None:
    head = struct.pack("<8sIIII", MAGIC, VERSION, FLAG_PLACEHOLDER, n, d)
    assert len(head) == 24
    pad = b"\x00" * (STUB_HEADER_BYTES - len(head))
    data = head + pad
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes -> {path}")


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, "xdna-gemv", "kernels")
    if len(sys.argv) >= 2:
        out = os.path.abspath(sys.argv[1])
    for n, d in SHAPES:
        name = f"bf16-gemv-{n}x{d}.bin"
        write_stub(os.path.join(out, name), n, d)
    print("\nSet: export XDNA_GEMV_DIR=%s" % out)
    print("Stub files are ignored by the runtime for NPU dispatch; use MLIR-AIE for real kernels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
