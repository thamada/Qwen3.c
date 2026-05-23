#!/usr/bin/env bash
# WMMA usage check for qwen3-8b/gpu-rocm (invoked by `make wmma`).
#
# Checks:
#   [static]  qwen3-rocm source/binary — direct WMMA in our code (expect: none)
#   [probe]   wmma_probe binary — WMMA detector calibration (expect: WMMA ops)
#   [lib]     rocBLAS bundled kernels for GPU_ARCH — WMMA in library ISA
#   [path]    runtime — Prefill uses hipBLAS GemmEx (requires MODEL, optional)
#   [runtime] rocprof kernel trace + objdump (requires MODEL, optional)
#
# Environment:
#   ROCM, GPU_ARCH, MODEL, WMMA_PROMPT, WMMA_N, WMMA_SKIP_RUN, WMMA_SKIP_ROCPROF

set -uo pipefail

ROCM="${ROCM:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-}"
MODEL="${MODEL:-}"
WMMA_PROMPT="${WMMA_PROMPT:-Hello}"
WMMA_N="${WMMA_N:-0}"
WMMA_SKIP_RUN="${WMMA_SKIP_RUN:-0}"
WMMA_SKIP_ROCPROF="${WMMA_SKIP_ROCPROF:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

LLVM_OBJDUMP="${LLVM_OBJDUMP:-$ROCM/llvm/bin/llvm-objdump}"
ROCBLAS_KERNELS="$ROCM/lib/rocblas/library/Kernels.so-000-${GPU_ARCH}.hsaco"

PASS=0
FAIL=0
WARN=0

ok()   { echo "  OK   $*"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL $*"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN $*"; WARN=$((WARN + 1)); }
skip() { echo "  SKIP $*"; }

section() {
    echo ""
    echo "=== $1 ==="
}

count_wmma_in_obj() {
    local obj="$1"
    if [ ! -f "$obj" ]; then
        echo "-1"
        return
    fi
    if [ ! -x "$LLVM_OBJDUMP" ]; then
        echo "-1"
        return
    fi
    "$LLVM_OBJDUMP" -d "$obj" 2>/dev/null | grep -ciE 'wmma|v_wmma' || true
}

# ---------------------------------------------------------------------------
# [static] Source — no direct WMMA / rocWMMA / MFMA in main.c
# ---------------------------------------------------------------------------
section "static: source (main.c)"
if grep -qE 'rocwmma|__builtin_amdgcn_wmma|__builtin_amdgcn_mfma|\bwmma_|\bmfma\b' main.c 2>/dev/null; then
    fail "direct WMMA/MFMA/rocWMMA references found in main.c"
else
    ok "no direct WMMA/MFMA/rocWMMA in main.c"
fi

# ---------------------------------------------------------------------------
# [static] Binary — qwen3-rocm should not embed WMMA (library path only)
# ---------------------------------------------------------------------------
section "static: binary (qwen3-rocm)"
if [ ! -f qwen3-rocm ]; then
    fail "qwen3-rocm not built (run: make build)"
else
    wmma_cnt="$(count_wmma_in_obj qwen3-rocm)"
    if [ "$wmma_cnt" = "-1" ]; then
        warn "llvm-objdump unavailable — skipped binary ISA check"
    elif [ "$wmma_cnt" -gt 0 ]; then
        fail "qwen3-rocm contains $wmma_cnt WMMA instruction(s) (unexpected)"
    else
        ok "qwen3-rocm binary has no WMMA instructions"
    fi
fi

# ---------------------------------------------------------------------------
# [probe] wmma_probe — calibration: detector must find WMMA on gfx11
# ---------------------------------------------------------------------------
section "probe: wmma-probe (detector calibration)"
if [ ! -f wmma-probe ]; then
    fail "wmma-probe not built (run: make wmma-probe)"
else
    probe_cnt="$(count_wmma_in_obj wmma-probe)"
    if [ "$probe_cnt" = "-1" ]; then
        warn "llvm-objdump unavailable — skipped probe ISA check"
    elif [ "$probe_cnt" -gt 0 ]; then
        ok "wmma-probe contains $probe_cnt WMMA instruction(s) — detector works"
    else
        if [[ "$GPU_ARCH" == gfx11* ]]; then
            fail "wmma-probe has no WMMA on $GPU_ARCH — fix detector or toolchain"
        else
            warn "wmma-probe has no WMMA (GPU_ARCH=$GPU_ARCH may not use RDNA WMMA)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# [lib] rocBLAS bundled kernels for this GPU
# ---------------------------------------------------------------------------
section "lib: rocBLAS kernels ($GPU_ARCH)"
if [ -z "$GPU_ARCH" ]; then
    skip "GPU_ARCH not set"
elif [ ! -f "$ROCBLAS_KERNELS" ]; then
    warn "rocBLAS kernel library not found: $ROCBLAS_KERNELS"
else
    lib_cnt="$(count_wmma_in_obj "$ROCBLAS_KERNELS")"
    fmac_cnt=$("$LLVM_OBJDUMP" -d "$ROCBLAS_KERNELS" 2>/dev/null | grep -c 'v_fmac_f32' || true)
    if [ "$lib_cnt" = "-1" ]; then
        warn "llvm-objdump unavailable — skipped rocBLAS ISA check"
    elif [ "$lib_cnt" -gt 0 ]; then
        ok "rocBLAS $GPU_ARCH library: $lib_cnt WMMA instruction(s) present"
    else
        warn "rocBLAS $GPU_ARCH library: 0 WMMA ops (v_fmac_f32 count: $fmac_cnt) — hipBLAS may use FMAC not WMMA"
    fi
fi

# ---------------------------------------------------------------------------
# [path] Runtime — Prefill linear uses hipBLAS GemmEx
# ---------------------------------------------------------------------------
section "path: runtime (hipBLAS prefill)"
if [ "$WMMA_SKIP_RUN" = "1" ]; then
    skip "WMMA_SKIP_RUN=1"
elif [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    skip "model not found ($MODEL) — set MODEL= or WMMA_SKIP_RUN=1"
else
    echo "  Running: ./qwen3-rocm \"\$MODEL\" -p \"\$WMMA_PROMPT\" -n $WMMA_N -t 0 -s 42 ..."
    set +e
    run_out="$(./qwen3-rocm "$MODEL" -p "$WMMA_PROMPT" -n "$WMMA_N" -t 0 -s 42 2>&1)"
    run_rc=$?
    set -e
    if [ "$run_rc" -ne 0 ]; then
        fail "qwen3-rocm exited with status $run_rc"
        echo "$run_out" | tail -20
    elif echo "$run_out" | grep -q 'Prefill linear: hipBLAS GemmEx'; then
        ok "Prefill linear: hipBLAS GemmEx active"
    else
        fail "Prefill linear: hipBLAS GemmEx not reported in output"
    fi
    if [ "$run_rc" -eq 0 ] && echo "$run_out" | grep -q 'prefill_tps:'; then
        pf="$(echo "$run_out" | sed -n 's/^prefill_tps: \([0-9.][0-9.]*\)/\1/p' | tail -1)"
        ok "prefill completed (prefill_tps=$pf)"
    fi
fi

# ---------------------------------------------------------------------------
# [runtime] rocprofv3 kernel trace — WMMA in kernels launched during prefill
# ---------------------------------------------------------------------------
section "runtime: kernel ISA (rocprofv3)"
if [ "$WMMA_SKIP_ROCPROF" = "1" ]; then
    skip "WMMA_SKIP_ROCPROF=1 (set WMMA_SKIP_ROCPROF=0 to enable)"
elif [ "$WMMA_SKIP_RUN" = "1" ] || [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    skip "model unavailable"
elif ! command -v rocprofv3 >/dev/null 2>&1; then
    skip "rocprofv3 not found"
else
    trace_dir="$(mktemp -d /tmp/qwen3-wmma-XXXXXX)"
    echo "  Profiling into $trace_dir ..."
    if rocprofv3 --kernel-trace -d "$trace_dir" -f csv -- \
        ./qwen3-rocm "$MODEL" -p "$WMMA_PROMPT" -n "$WMMA_N" -t 0 -s 42 \
        >/dev/null 2>&1; then
        rt_wmma=0
        rt_kernels=0
        while IFS= read -r -d '' f; do
            rt_kernels=$((rt_kernels + 1))
            c="$(count_wmma_in_obj "$f")"
            if [ "$c" != "-1" ] && [ "$c" -gt 0 ]; then
                rt_wmma=$((rt_wmma + c))
            fi
        done < <(find "$trace_dir" -type f \( -name '*.hsaco' -o -name '*.co' \) -print0 2>/dev/null)
        if [ "$rt_kernels" -eq 0 ]; then
            warn "rocprofv3 produced no code objects — check ROCm profiler setup"
        elif [ "$rt_wmma" -gt 0 ]; then
            ok "prefill kernels: $rt_wmma WMMA instruction(s) in traced code objects"
        else
            warn "prefill kernels: 0 WMMA in $rt_kernels traced code object(s)"
        fi
    else
        warn "rocprofv3 failed (missing libdw.so or GPU access?) — skipped runtime ISA"
    fi
    rm -rf "$trace_dir"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==============================================="
echo "  WMMA check summary (GPU_ARCH=${GPU_ARCH:-unknown})"
echo "==============================================="
echo "  PASS: $PASS   FAIL: $FAIL   WARN: $WARN"
echo ""
echo "  Notes:"
echo "    - qwen3-rocm uses hipBLAS/rocBLAS for prefill GEMM (not direct WMMA)."
echo "    - WMMA in rocBLAS depends on kernel selection; 0 WMMA is common on gfx11."
echo "    - Use WMMA_SKIP_RUN=1 for static-only checks without a model."
echo ""

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
