#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 8 ]; then
  echo "usage: $0 GPU_NAME STATE START COUNT CHUNK OUT_LOG STATUS_FILE CFG" >&2
  exit 2
fi

GPU_NAME="$1"
STATE="$2"
START="$3"
COUNT="$4"
CHUNK="$5"
OUT_LOG="$6"
STATUS_FILE="$7"
CFG="$8"

DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/gpu_island2"
DRIVER="$DIR/search_driver.sh"
SCAN_START_EPOCH="$(date +%s)"
done_count=0
candidate_count=0

mkdir -p "$(dirname "$OUT_LOG")" "$(dirname "$STATUS_FILE")"
: > "$OUT_LOG"

write_status() {
  local now elapsed rate remaining eta pct
  now="$(date +%s)"
  elapsed=$((now - SCAN_START_EPOCH))
  [ "$elapsed" -le 0 ] && elapsed=1
  rate="$(awk -v d="$done_count" -v e="$elapsed" 'BEGIN { printf "%.2f", d/e }')"
  remaining=$((COUNT - done_count))
  [ "$remaining" -lt 0 ] && remaining=0
  eta="$(awk -v r="$remaining" -v rate="$rate" 'BEGIN { if (rate > 0) printf "%.0f", r/rate; else print -1 }')"
  pct="$(awk -v d="$done_count" -v c="$COUNT" 'BEGIN { if (c > 0) printf "%.2f", 100*d/c; else print "0.00" }')"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$GPU_NAME" "$START" "$COUNT" "$done_count" "$candidate_count" "$elapsed" "$rate" "$remaining" "$eta" "$pct" "$CFG" \
    > "$STATUS_FILE"
  printf "PROGRESS gpu=%s processed=%s/%s pct=%s rate=%s n/s candidates=%s eta_s=%s cfg=%s\n" \
    "$GPU_NAME" "$done_count" "$COUNT" "$pct" "$rate" "$candidate_count" "$eta" "$CFG"
}

write_status

offset=0
while [ "$offset" -lt "$COUNT" ]; do
  this_chunk="$CHUNK"
  if [ $((offset + this_chunk)) -gt "$COUNT" ]; then
    this_chunk=$((COUNT - offset))
  fi
  chunk_start=$((START + offset))
  tmp="$(mktemp)"
  t0="$(date +%s)"
  GPU_ISLAND_BIN="$BIN" GPU_STATE_FILE="$STATE" BLOCKS=512 \
    GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=trunc_first GPU_WAVE=128 GPU_FAN_BITS=22 GPU_STREAM_CANDIDATES=1 \
    bash "$DRIVER" "$chunk_start" "$this_chunk" "$this_chunk" 1 > "$tmp"
  rc=$?
  t1="$(date +%s)"
  if [ "$rc" -ne 0 ]; then
    echo "ERROR gpu=$GPU_NAME chunk_start=$chunk_start chunk=$this_chunk rc=$rc" >&2
    cat "$tmp" >&2
    rm -f "$tmp"
    exit "$rc"
  fi
  if [ -s "$tmp" ]; then
    cat "$tmp" >> "$OUT_LOG"
  fi
  new_candidates="$(grep -c '^CLEAN nonce=' "$tmp" 2>/dev/null || true)"
  candidate_count=$((candidate_count + new_candidates))
  done_count=$((done_count + this_chunk))
  chunk_elapsed=$((t1 - t0))
  [ "$chunk_elapsed" -le 0 ] && chunk_elapsed=1
  chunk_rate="$(awk -v c="$this_chunk" -v e="$chunk_elapsed" 'BEGIN { printf "%.2f", c/e }')"
  printf "CHUNK gpu=%s start=%s count=%s rate=%s n/s new_candidates=%s\n" \
    "$GPU_NAME" "$chunk_start" "$this_chunk" "$chunk_rate" "$new_candidates"
  rm -f "$tmp"
  write_status
  offset=$((offset + this_chunk))
done

echo "DONE gpu=$GPU_NAME candidates=$candidate_count"
