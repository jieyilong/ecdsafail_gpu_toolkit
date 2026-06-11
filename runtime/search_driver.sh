#!/usr/bin/env bash
# Runs ON the GPU machine. Multi-GPU parallel chunked island search over a nonce range.
# Splits [START, START+COUNT) across all (or GPUS) visible GPUs, one process per GPU,
# each pinned via CUDA_VISIBLE_DEVICES. Emits "CLEAN nonce=N" lines (merged, deduped).
#   env:  GPU_ISLAND_BIN, GPU_STATE_FILE, BLOCKS (opt)
#   args: START COUNT [CHUNK] [NGPU=auto]
set -uo pipefail
export PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin"
truthy(){ case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0;; *) return 1;; esac; }
START="${1:?START}"; COUNT="${2:?COUNT}"; CHUNK="${3:-500000}"; NGPU="${4:-auto}"
BIN="${GPU_ISLAND_BIN:?set GPU_ISLAND_BIN}"; STATE="${GPU_STATE_FILE:?set GPU_STATE_FILE}"; BLOCKS="${BLOCKS:-512}"
GPU_BATCH_INV="${GPU_BATCH_INV:-${BATCH_INV:-0}}"
GPU_COMB_BITS="${GPU_COMB_BITS:-8}"
case "${GPU_LARGE_COMB:-0}" in 1|true|TRUE|yes|YES|on|ON) GPU_COMB_BITS=16;; esac
GPU_GCD_MODE="${GPU_GCD_MODE:-${GCD_MODE:-full_first}}"
GPU_WAVE="${GPU_WAVE:-${WAVE:-128}}"
GPU_FAN_BITS="${GPU_FAN_BITS:-0}"
GPU_STREAM_CANDIDATES="${GPU_STREAM_CANDIDATES:-1}"
[ -x "$BIN" ] || { echo "ERROR: kernel binary not found/executable: $BIN (run build)" >&2; exit 1; }
[ -f "$STATE" ] || { echo "ERROR: state file not found: $STATE" >&2; exit 1; }
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
      hits="$(CUDA_VISIBLE_DEVICES="$g" GPU_STATE="$STATE" KERNEL2=1 BLOCKS="$BLOCKS" \
        GPU_BATCH_INV="$GPU_BATCH_INV" GPU_COMB_BITS="$GPU_COMB_BITS" \
        GPU_GCD_MODE="$GPU_GCD_MODE" GPU_WAVE="$GPU_WAVE" GPU_FAN_BITS="$GPU_FAN_BITS" \
        "$BIN" "$s" "$c" 2>/dev/null | grep -oE "CLEAN nonce=[0-9]+" || true)"
      if [ -n "$hits" ]; then
        printf '%s\n' "$hits" >> "$TMP/g$g"
        if truthy "$GPU_STREAM_CANDIDATES"; then
          printf '%s\n' "$hits"
        fi
      fi
    done
  ) &
done
wait
if ! truthy "$GPU_STREAM_CANDIDATES"; then
  cat "$TMP"/g* 2>/dev/null | sort -u
fi
