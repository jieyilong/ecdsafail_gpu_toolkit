#!/usr/bin/env bash
# Runs ON the GPU machine. Multi-GPU parallel chunked island search over a nonce range.
# Splits [START, START+COUNT) across all (or GPUS) visible GPUs, one process per GPU,
# each pinned via CUDA_VISIBLE_DEVICES. Emits "CLEAN nonce=N" lines (merged, deduped).
#   env:  GPU_ISLAND_BIN, GPU_STATE_FILE, BLOCKS (opt)
#   args: START COUNT [CHUNK] [NGPU=auto]
set -uo pipefail
export PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin"
START="${1:?START}"; COUNT="${2:?COUNT}"; CHUNK="${3:-200000}"; NGPU="${4:-auto}"
BIN="${GPU_ISLAND_BIN:?set GPU_ISLAND_BIN}"; STATE="${GPU_STATE_FILE:?set GPU_STATE_FILE}"; BLOCKS="${BLOCKS:-512}"
[ -x "$BIN" ] || { echo "ERROR: kernel binary not found/executable: $BIN (run build)" >&2; exit 1; }
[ -f "$STATE" ] || { echo "ERROR: state file not found: $STATE" >&2; exit 1; }
# Kernel env: KERNEL3=1 (batch-inv) or KERNEL2=1 (original shot-parallel) or neither (serial)
KFLAG=""
[ "${KERNEL3:-0}" = 1 ] && KFLAG="KERNEL3=1"
[ "${KERNEL2:-0}" = 1 ] && KFLAG="KERNEL2=1"
if [ "$NGPU" = auto ] || [ -z "$NGPU" ]; then
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
fi
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
per=$(( (COUNT + NGPU - 1) / NGPU ))
for (( g=0; g<NGPU; g++ )); do
  gstart=$(( START + g*per )); gcount=$per; end=$(( START + COUNT ))
  (( gstart + gcount > end )) && gcount=$(( end - gstart ))
  (( gcount <= 0 )) && continue
  (
    d=0
    while [ "$d" -lt "$gcount" ]; do
      c=$(( gcount-d < CHUNK ? gcount-d : CHUNK )); s=$(( gstart+d )); d=$(( d+CHUNK ))
      CUDA_VISIBLE_DEVICES="$g" GPU_STATE="$STATE" $KFLAG BLOCKS="$BLOCKS" \
        "$BIN" "$s" "$c" 2>/dev/null | grep -oE "CLEAN nonce=[0-9]+" >> "$TMP/g$g"
    done
  ) &
done
wait
cat "$TMP"/g* 2>/dev/null | sort -u
