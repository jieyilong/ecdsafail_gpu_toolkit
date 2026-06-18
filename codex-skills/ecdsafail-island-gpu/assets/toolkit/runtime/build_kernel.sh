#!/usr/bin/env bash
# Runs ON the GPU machine. Compiles the CUDA kernel for the local GPU(s), auto-detecting
# the compute capability, with a PTX-JIT fallback for GPUs newer than the installed CUDA.
# Works for A100(8.0) H100/H200(9.0) RTX30(8.6) RTX40(8.9) RTX50(12.0) and others.
#   usage: build_kernel.sh SRC OUT [ARCH]    ARCH = auto | sm_XX | XX
set -uo pipefail
SRC="${1:?SRC}"; OUT="${2:?OUT}"; ARCH="${3:-auto}"
export PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin"
command -v nvcc >/dev/null || { echo "ERROR: nvcc not on PATH. Install the CUDA toolkit (nvidia-cuda-toolkit)." >&2; exit 2; }
if [ "$ARCH" = auto ] || [ -z "$ARCH" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. \r')
fi
ARCH="${ARCH#sm_}"
err=/tmp/nvcc_err.$$; : > "$err"
# 1) native SASS for the detected arch + forward-compatible PTX of the same family
if [ -n "$ARCH" ] && nvcc -O3 -gencode arch=compute_"$ARCH",code=sm_"$ARCH" \
        -gencode arch=compute_"$ARCH",code=compute_"$ARCH" "$SRC" -o "$OUT" 2>"$err"; then
  echo ">> built native sm_$ARCH (+PTX) | $(nvcc --version | grep -oE 'release [0-9.]+')"
# 2) nvcc too old for this arch -> PTX at compute_80 (any GPU >= sm_80 JITs it)
elif nvcc -O3 -gencode arch=compute_80,code=compute_80 "$SRC" -o "$OUT" 2>>"$err"; then
  echo ">> built PTX compute_80 (JIT to sm_$ARCH at runtime). Update CUDA for native perf."
# 3) last resort: let nvcc pick its default arch
elif nvcc -O3 "$SRC" -o "$OUT" 2>>"$err"; then
  echo ">> built with nvcc default arch."
else
  echo "ERROR: nvcc build failed:" >&2; tail -8 "$err" >&2; rm -f "$err"; exit 1
fi
rm -f "$err"
