#!/usr/bin/env bash
# ecdsafail-island-gpu — GPU-accelerated Fiat-Shamir island search for ecdsa.fail.
# Portable across NVIDIA GPUs (A100/H100/H200, RTX 30/40/50), multi-GPU, local or remote SSH.
#
#   ./island.sh init-local  [CHALLENGE]                 # configure for a local GPU + build
#   ./island.sh init-remote "ssh -p PORT user@IP" [CHALLENGE]  # configure for a remote GPU box + build
#   ./island.sh doctor                                  # report GPU/CUDA on the target
#   ./island.sh install                                 # build the Rust helpers in the challenge repo
#   ./island.sh measure [CFG]                           # Toffoli (CCX) cost of a config
#   ./island.sh build                                   # compile the CUDA kernel (auto-arch)
#   ./island.sh dump   CFG OUT.bin                      # gpu_state dump for a config
#   ./island.sh probe  STATE                            # GPU Keccak probe cross-check
#   ./island.sh search STATE START N [CHUNK]            # multi-GPU search -> CLEAN nonce=...
#   ./island.sh stage2 CFG CANDIDATES [RESULTS] [JOBS]  # exact validator-backed prefilter
#   ./island.sh obligations emit-default|emit-dialog-gcd [OUT]
#   ./island.sh obligations check CFG MANIFEST CANDIDATES [RESULTS] [JOBS]
#   ./island.sh test-gpu-knobs [CFG] [START] [N]        # correctness smoke for GPU knobs
#   ./island.sh bench-gpu-knobs [CFG] [START] [N]       # throughput benchmark for GPU knobs
#   ./island.sh validate CFG NONCE...                  # quantum-confirm 0/0/0 + score
#   ./island.sh bake   KEY VALUE [...]                  # CRLF-safe mod.rs edit + ecdsafail run
#   ./island.sh hunt   CFG START N                      # measure -> dump -> search -> validate
#   CFG = a space-free env assignment, e.g. DIALOG_GCD_ACTIVE_ITERATIONS=258
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/island.sh"
CFGF="${ISLAND_CONFIG:-$HERE/config.env}"
die(){ echo "ERROR: $*" >&2; exit 1; }
truthy(){ case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0;; *) return 1;; esac; }
line_score_suffix(){ # tof qubits
  local tof="${1:-}" q="${2:-}" score
  [ -n "$tof" ] && [ -n "$q" ] || return 0
  score=$(awk -v t="$tof" -v q="$q" 'BEGIN{printf "%d", int(t + 0.5) * q}')
  printf ' score=%s' "$score"
}
emit_validation_record(){ # line
  local line="$1" dest lock dir
  echo "$line"
  [ -n "${VALIDATE_RESULTS_LOG:-}" ] || return 0
  case "$line" in
    ERROR*)
      dest="${VALIDATE_ERRORS_LOG:-$(dirname "$VALIDATE_RESULTS_LOG")/errors.log}"
      ;;
    *)
      dest="$VALIDATE_RESULTS_LOG"
      ;;
  esac
  dir="$(dirname "$dest")"
  mkdir -p "$dir"
  lock="${VALIDATE_LOCK_FILE:-$VALIDATE_RESULTS_LOG.lock}"
  mkdir -p "$(dirname "$lock")"
  if command -v flock >/dev/null 2>&1; then
    { flock 9; printf '%s\n' "$line" >> "$dest"; } 9>"$lock"
  else
    printf '%s\n' "$line" >> "$dest"
  fi
}

# ---- config helpers (init-* don't require an existing config.env) ----
cmd="${1:-help}"; shift || true
write_cfg(){ # GPU REMOTE_SSH CHALLENGE
  cat > "$CFGF" <<EOF
CHALLENGE="$3"
GPU="$1"
REMOTE_SSH="$2"
REMOTE_DIR=".ecdsafail_island"
NVCC_ARCH="auto"
GPUS="auto"
BLOCKS="512"
EOF
  echo ">> wrote $CFGF"
}

case "$cmd" in
init-local)
  CH="${1:-}"; [ -n "$CH" ] || { [ -f "$CFGF" ] && CH="$(. "$CFGF"; echo "${CHALLENGE:-}")"; }
  [ -d "${CH:-}" ] || die "usage: init-local <path-to-ecdsafail-challenge>"
  write_cfg local "" "$CH"; bash "$SELF" doctor; bash "$SELF" build; exit 0;;
init-remote)
  SSHCMD="${1:?usage: init-remote \"ssh -p PORT user@IP\" [CHALLENGE]}"
  CH="${2:-}"; [ -n "$CH" ] || { [ -f "$CFGF" ] && CH="$(. "$CFGF"; echo "${CHALLENGE:-}")"; }
  [ -d "${CH:-}" ] || die "give the challenge repo path: init-remote \"$SSHCMD\" <path-to-ecdsafail-challenge>"
  case "$SSHCMD" in ssh\ *) ;; *) SSHCMD="ssh $SSHCMD";; esac   # tolerate "-p PORT user@IP" without leading "ssh"
  write_cfg remote "$SSHCMD" "$CH"; bash "$SELF" doctor; bash "$SELF" build; exit 0;;
esac

[ -f "$CFGF" ] || die "not configured. Run: ./island.sh init-local <CHALLENGE>   OR   ./island.sh init-remote \"ssh -p PORT user@IP\" <CHALLENGE>"
# shellcheck disable=SC1090
. "$CFGF"
: "${CHALLENGE:?set CHALLENGE in config.env}"; : "${GPU:=local}"; : "${REMOTE_SSH:=}"
: "${REMOTE_DIR:=.ecdsafail_island}"; : "${NVCC_ARCH:=auto}"; : "${GPUS:=auto}"; : "${BLOCKS:=512}"
: "${GPU_BATCH_INV:=0}"; : "${GPU_COMB_BITS:=8}"; : "${GPU_GCD_MODE:=full_first}"; : "${GPU_WAVE:=128}"; : "${GPU_FAN_BITS:=0}"; : "${GPU_FILTER:=dialog}"; : "${GPU_TRAILMIX_THIN:=0}"; : "${GPU_TRAILMIX_SLACK:=0}"; : "${GPU_TRAILMIX_WINDOW:=0}"; : "${GPU_STREAM_CANDIDATES:=1}"
: "${EVAL_STAGE2_SHOTS:=9024}"
BIN="$CHALLENGE/target/release"; KSRC="$HERE/cuda/gpu_island2.cu"; RDIR="$REMOTE_DIR"
need_remote(){ [ -n "$REMOTE_SSH" ] || die "GPU=remote needs REMOTE_SSH (init-remote)"; }
rhost(){ echo "$REMOTE_SSH" | grep -oE '[^ ]+@[^ ]+' | head -1; }
rport(){ echo "$REMOTE_SSH" | grep -oE '\-p +[0-9]+' | grep -oE '[0-9]+'; }
rkey(){ local k; k=$(echo "$REMOTE_SSH" | grep -oE '\-i +[^ ]+' | awk '{print $2}'); echo "${k/#\~/$HOME}"; }
rsh(){ $REMOTE_SSH "$@"; }
rcp(){ local p k; p="$(rport)"; k="$(rkey)"; scp ${p:+-P "$p"} ${k:+-i "$k"} -o StrictHostKeyChecking=no "$1" "$(rhost):$RDIR/$2" >/dev/null; }
push_runtime(){ rsh "mkdir -p $RDIR"; rcp "$KSRC" gpu_island2.cu; for s in build_kernel search_driver doctor; do rcp "$HERE/runtime/$s.sh" "$s.sh"; done; }
tail_nonce(){
  grep -E 'set_default_env\("DIALOG_TAIL_NONCE", "[0-9]+"\)' "$CHALLENGE/src/point_add/mod.rs" \
    | tail -1 | grep -oE '"[0-9]+"' | tr -d '"'
}
eval_supports_tail_nonce(){
  grep -a -q "EVAL_TAIL_NONCE" "$BIN/eval_circuit" 2>/dev/null
}
emit_validation_line(){ # nonce out eval_rc
  local nonce="$1" out="$2" eval_rc="$3" cls pha anc tof q shots detail prefix
  cls=$(echo "$out"|grep "classical mismatches"|grep -oE '[0-9]+$')
  pha=$(echo "$out"|grep "phase-garbage"|grep -oE '[0-9]+$')
  anc=$(echo "$out"|grep "ancilla-garbage"|grep -oE '[0-9]+$')
  tof=$(echo "$out"|grep "avg executed Toffoli"|grep -oE '[0-9.]+'|head -1)
  q=$(echo "$out"|grep -E '^  qubits '|grep -oE '[0-9]+$'|head -1)
  shots=$(echo "$out"|grep "tested shots"|grep -oE '[0-9]+$'|head -1)
  if [ -z "${cls:-}" ] || [ -z "${pha:-}" ] || [ -z "${anc:-}" ]; then
    detail=$(echo "$out" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g')
    emit_validation_record "ERROR nonce=$nonce stage=eval rc=$eval_rc detail=${detail:-eval_circuit failed}"
    return
  fi
  prefix=0
  truthy "${EVAL_STAGE2_MODE:-0}" && prefix=1
  [ -n "${EVAL_SHOT_LIMIT:-}" ] && prefix=1
  if [ "${cls:-x}" = 0 ] && [ "${pha:-x}" = 0 ] && [ "${anc:-x}" = 0 ]; then
    if [ "$prefix" = 1 ]; then
      emit_validation_record "stage2-pass nonce=$nonce shots=${shots:-?} cls=0 pha=0 anc=0 tof=${tof:-?} qubits=${q:-?}"
    else
      emit_validation_record "CLEAN nonce=$nonce cls=0 pha=0 anc=0 tof=$tof qubits=$q$(line_score_suffix "$tof" "$q")"
    fi
  else
    if [ "$prefix" = 1 ]; then
      emit_validation_record "stage2-reject nonce=$nonce shots=${shots:-?} cls=${cls:-?} pha=${pha:-?} anc=${anc:-?} tof=${tof:-?} qubits=${q:-?}"
    else
      emit_validation_record "dirty nonce=$nonce cls=${cls:-?} pha=${pha:-?} anc=${anc:-?} tof=${tof:-?} qubits=${q:-?}"
    fi
  fi
}
run_variant_search(){ # envs state start n chunk
  local envs="$1" state="$2" start="$3" n="$4" chunk="${5:-200000}"
  env $envs bash "$SELF" search "$state" "$start" "$n" "$chunk" | sort -u
}
require_variant_clean(){ # name envs state nonce
  local name="$1" envs="$2" state="$3" nonce="$4" out
  echo ">> known-clean check: $name"
  out=$(run_variant_search "$envs" "$state" "$nonce" 1 1)
  echo "$out"
  echo "$out" | grep -qx "CLEAN nonce=$nonce" || die "$name did not report known clean nonce $nonce"
}
compare_variant_range(){ # name envs state start n chunk baseline_file
  local name="$1" envs="$2" state="$3" start="$4" n="$5" chunk="$6" baseline="$7" got
  got="$(mktemp)"
  echo ">> exact range check: $name over [$start, $((start+n)))"
  run_variant_search "$envs" "$state" "$start" "$n" "$chunk" > "$got"
  if ! diff -u "$baseline" "$got"; then
    rm -f "$got"
    die "$name candidate set differs from baseline"
  fi
  rm -f "$got"
}
check_subset(){ # name envs state start n chunk baseline_file
  local name="$1" envs="$2" state="$3" start="$4" n="$5" chunk="$6" baseline="$7" got missing
  got="$(mktemp)"; missing="$(mktemp)"
  echo ">> noisy superset check: $name over [$start, $((start+n)))"
  run_variant_search "$envs" "$state" "$start" "$n" "$chunk" > "$got"
  comm -23 "$baseline" "$got" > "$missing"
  if [ -s "$missing" ]; then
    echo "Missing baseline candidates:" >&2
    cat "$missing" >&2
    rm -f "$got" "$missing"
    die "$name missed baseline candidates"
  fi
  rm -f "$got" "$missing"
}
run_raw_search(){ # envs state start n
  local envs="$1" state="$2" start="$3" n="$4"
  if [ "$GPU" = local ]; then
    [ -x "$HERE/gpu_island2" ] || die "run './island.sh build' first"
    env $envs GPU_STATE="$state" KERNEL2=1 BLOCKS="$BLOCKS" "$HERE/gpu_island2" "$start" "$n"
  else
    need_remote
    rsh "GPU_STATE=$state KERNEL2=1 BLOCKS=$BLOCKS $envs \$HOME/$RDIR/gpu_island2 $start $n"
  fi
}
default_jobs(){
  if [ -n "${STAGE2_JOBS:-}" ]; then echo "$STAGE2_JOBS"; return; fi
  if command -v nproc >/dev/null 2>&1; then nproc; return; fi
  if command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu 2>/dev/null && return; fi
  echo 4
}
extract_nonces(){
  sed -nE '
    s/.*nonce=([0-9]+).*/\1/p
    /^[[:space:]]*[0-9]+[[:space:]]*$/ {
      s/[[:space:]]//g
      p
    }
  ' | sort -n -u
}
extract_file_nonces(){ # file
  [ -s "$1" ] || return 0
  extract_nonces < "$1"
}
bench_variant(){ # name envs state start n summary_file
  local name="$1" envs="$2" state="$3" start="$4" n="$5" summary="$6"
  local warmups="${GPU_BENCH_WARMUPS:-1}" runs="${GPU_BENCH_RUNS:-3}"
  local total=$((warmups + runs)) rates out rate secs clean build i rate_file
  rate_file="$(mktemp)"
  echo ">> bench: $name  env='$envs'"
  for ((i=1; i<=total; i++)); do
    out=$(run_raw_search "$envs" "$state" "$start" "$n")
    rate=$(echo "$out" | sed -n 's/.*(\([0-9][0-9]*\) nonce\/s).*/\1/p' | tail -1)
    secs=$(echo "$out" | sed -n 's/.* in \([0-9.][0-9.]*\)s .*/\1/p' | tail -1)
    clean=$(echo "$out" | sed -n 's/.* clean=\([0-9][0-9]*\).*/\1/p' | tail -1)
    build=$(echo "$out" | sed -n 's/.*table built: .* in \([0-9.][0-9.]*\)s.*/\1/p' | tail -1)
    [ -n "$build" ] || build="-"
    [ -n "$rate" ] || { echo "$out"; rm -f "$rate_file"; die "could not parse benchmark rate for $name"; }
    if [ "$i" -le "$warmups" ]; then
      printf "   warmup %d/%d: %s nonce/s in %ss clean=%s build=%ss\n" "$i" "$warmups" "$rate" "${secs:-?}" "${clean:-?}" "$build"
    else
      printf "   run %d/%d: %s nonce/s in %ss clean=%s build=%ss\n" "$((i-warmups))" "$runs" "$rate" "${secs:-?}" "${clean:-?}" "$build"
      echo "$rate" >> "$rate_file"
    fi
  done
  rates=$(awk '{sum+=$1; if(NR==1||$1<min)min=$1; if($1>max)max=$1} END{if(NR){printf "%.0f %d %d", sum/NR, min, max}}' "$rate_file")
  rm -f "$rate_file"
  set -- $rates
  printf "%s %s %s %s\n" "$name" "$1" "$2" "$3" >> "$summary"
}

case "$cmd" in

doctor)
  if [ "$GPU" = local ]; then bash "$HERE/runtime/doctor.sh"
  else need_remote; push_runtime >/dev/null 2>&1 || true; rsh "bash $RDIR/doctor.sh"; fi
  ;;

install)
  cp "$HERE"/rust/*.rs "$CHALLENGE/src/bin/"
  ( cd "$CHALLENGE" && cargo build --release \
      --bin build_circuit --bin eval_circuit --bin dump_gpu_state --bin count_tof --bin island_search --bin obligation_filter )
  echo ">> installed. Sanity: (cd $CHALLENGE && ecdsafail run) must print the leaderboard score, 0/0/0."
  ;;

measure)
  echo "baseline: $("$BIN/count_tof")"; [ $# -gt 0 ] && echo "$1: $(env "$1" "$BIN/count_tof")" ;;

dump)
  CFG="${1:-}"; OUT="${2:?usage: dump CFG OUT.bin}"
  env ${CFG:+$CFG} "$BIN/dump_gpu_state" "$OUT" | grep -E "cfg:|n_ops|KECCAK MATCH"; echo ">> wrote $OUT" ;;

build)
  if [ "$GPU" = local ]; then
    bash "$HERE/runtime/build_kernel.sh" "$KSRC" "$HERE/gpu_island2" "$NVCC_ARCH"
  else
    need_remote; push_runtime
    rsh "bash $RDIR/build_kernel.sh \$HOME/$RDIR/gpu_island2.cu \$HOME/$RDIR/gpu_island2 $NVCC_ARCH"
  fi
  ;;

probe)
  STATE="${1:?usage: probe STATE}"
  if [ "$GPU" = local ]; then
    [ -x "$HERE/gpu_island2" ] || die "run './island.sh build' first"
    GPU_STATE="$STATE" "$HERE/gpu_island2" 0 1 probe
  else
    need_remote; rcp "$STATE" state.bin
    rsh "GPU_STATE=\$HOME/$RDIR/state.bin \$HOME/$RDIR/gpu_island2 0 1 probe"
  fi
  ;;

search)
  STATE="${1:?usage: search STATE START N [CHUNK]}"; START="${2:?}"; N="${3:?}"; CHUNK="${4:-500000}"
  if [ "$GPU" = local ]; then
    [ -x "$HERE/gpu_island2" ] || die "run './island.sh build' first"
    GPU_ISLAND_BIN="$HERE/gpu_island2" GPU_STATE_FILE="$STATE" BLOCKS="$BLOCKS" \
      GPU_BATCH_INV="$GPU_BATCH_INV" GPU_COMB_BITS="$GPU_COMB_BITS" \
      GPU_GCD_MODE="$GPU_GCD_MODE" GPU_WAVE="$GPU_WAVE" GPU_FAN_BITS="$GPU_FAN_BITS" \
      GPU_FILTER="$GPU_FILTER" GPU_TRAILMIX_THIN="$GPU_TRAILMIX_THIN" \
      GPU_TRAILMIX_SLACK="$GPU_TRAILMIX_SLACK" GPU_TRAILMIX_WINDOW="$GPU_TRAILMIX_WINDOW" \
      GPU_STREAM_CANDIDATES="$GPU_STREAM_CANDIDATES" \
      bash "$HERE/runtime/search_driver.sh" "$START" "$N" "$CHUNK" "$GPUS"
  else
    need_remote; rcp "$STATE" state.bin
    rcp "$HERE/runtime/search_driver.sh" search_driver.sh
    rsh "GPU_ISLAND_BIN=\$HOME/$RDIR/gpu_island2 GPU_STATE_FILE=\$HOME/$RDIR/state.bin BLOCKS=$BLOCKS \
         GPU_BATCH_INV=$GPU_BATCH_INV GPU_COMB_BITS=$GPU_COMB_BITS GPU_GCD_MODE=$GPU_GCD_MODE GPU_WAVE=$GPU_WAVE GPU_FAN_BITS=$GPU_FAN_BITS GPU_STREAM_CANDIDATES=$GPU_STREAM_CANDIDATES \
         GPU_FILTER=$GPU_FILTER GPU_TRAILMIX_THIN=$GPU_TRAILMIX_THIN GPU_TRAILMIX_SLACK=$GPU_TRAILMIX_SLACK GPU_TRAILMIX_WINDOW=$GPU_TRAILMIX_WINDOW \
         bash \$HOME/$RDIR/search_driver.sh $START $N $CHUNK $GPUS"
  fi
  ;;

test-gpu-knobs)
  CFG="${1:-}"; START="${2:-0}"; N="${3:-4096}"; CHUNK="${4:-$N}"
  [ -f "$CHALLENGE/src/point_add/mod.rs" ] || die "bad CHALLENGE path: $CHALLENGE"
  KNOWN="$(tail_nonce)"; [ -n "$KNOWN" ] || die "could not extract DIALOG_TAIL_NONCE"
  echo ">> testing against CHALLENGE=$CHALLENGE"
  echo ">> config override file: $CFGF"
  echo ">> known clean DIALOG_TAIL_NONCE=$KNOWN"
  if [ "${GPU_TEST_SKIP_INSTALL:-0}" != 1 ]; then
    echo ">> [1/5] installing/rebuilding Rust helpers from current SOTA"
    bash "$SELF" install >/dev/null
  else
    echo ">> [1/5] skipping Rust helper rebuild (GPU_TEST_SKIP_INSTALL=1)"
  fi
  if [ "${GPU_TEST_SKIP_BUILD:-0}" != 1 ]; then
    echo ">> [2/5] building CUDA kernel"
    bash "$SELF" build
  else
    echo ">> [2/5] skipping CUDA rebuild (GPU_TEST_SKIP_BUILD=1)"
  fi
  STATE="$(mktemp).bin"; BASE="$(mktemp)"
  trap 'rm -f "$STATE" "$BASE"' EXIT
  echo ">> [3/5] dumping GPU state"
  bash "$SELF" dump "$CFG" "$STATE"
  echo ">> [4/5] probing GPU Keccak derivation"
  probe_out=$(bash "$SELF" probe "$STATE")
  echo "$probe_out" | grep -E "options:|probe nonce|k1:|k2:"
  echo "$probe_out" | grep -q "k1:OK k2:OK" || die "GPU probe mismatch"

  base_env="GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128"
  require_variant_clean "baseline" "$base_env" "$STATE" "$KNOWN"
  require_variant_clean "trunc_first" "GPU_GCD_MODE=trunc_first GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "single_pass" "GPU_GCD_MODE=single_pass GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "fan${GPU_TEST_FAN_BITS:-20}" "GPU_FAN_BITS=${GPU_TEST_FAN_BITS:-20} GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "batch_comb16_single" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=single_pass GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "wave64" "GPU_WAVE=64" "$STATE" "$KNOWN"
  require_variant_clean "wave256" "GPU_WAVE=256" "$STATE" "$KNOWN"
  require_variant_clean "batch_inv" "GPU_BATCH_INV=1 GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "comb16" "GPU_COMB_BITS=16 GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "batch_wave256" "GPU_BATCH_INV=1 GPU_WAVE=256" "$STATE" "$KNOWN"
  require_variant_clean "batch_comb16" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_WAVE=128" "$STATE" "$KNOWN"
  require_variant_clean "all_exact" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=trunc_first GPU_WAVE=256" "$STATE" "$KNOWN"
  TEST_COMB_BITS="${GPU_TEST_COMB_BITS:-}"
  if [ -z "$TEST_COMB_BITS" ] && truthy "${GPU_TEST_LARGE_COMB:-0}"; then TEST_COMB_BITS="20 22"; fi
  for bits in $TEST_COMB_BITS; do
    require_variant_clean "comb$bits" "GPU_COMB_BITS=$bits GPU_WAVE=128" "$STATE" "$KNOWN"
    require_variant_clean "batch_comb$bits" "GPU_BATCH_INV=1 GPU_COMB_BITS=$bits GPU_WAVE=128" "$STATE" "$KNOWN"
  done
  require_variant_clean "trunc_only" "GPU_GCD_MODE=trunc_only GPU_WAVE=128" "$STATE" "$KNOWN"

  echo ">> [5/5] baseline range over [$START, $((START+N)))"
  run_variant_search "$base_env" "$STATE" "$START" "$N" "$CHUNK" > "$BASE"
  echo "baseline candidates: $(wc -l < "$BASE" | tr -d ' ')"
  compare_variant_range "trunc_first" "GPU_GCD_MODE=trunc_first GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "single_pass" "GPU_GCD_MODE=single_pass GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "fan${GPU_TEST_FAN_BITS:-20}" "GPU_FAN_BITS=${GPU_TEST_FAN_BITS:-20} GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "batch_comb16_single" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=single_pass GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "wave64" "GPU_WAVE=64" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "wave256" "GPU_WAVE=256" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "batch_inv" "GPU_BATCH_INV=1 GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "comb16" "GPU_COMB_BITS=16 GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "batch_wave256" "GPU_BATCH_INV=1 GPU_WAVE=256" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "batch_comb16" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  compare_variant_range "all_exact" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=trunc_first GPU_WAVE=256" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  for bits in $TEST_COMB_BITS; do
    compare_variant_range "comb$bits" "GPU_COMB_BITS=$bits GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
    compare_variant_range "batch_comb$bits" "GPU_BATCH_INV=1 GPU_COMB_BITS=$bits GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  done
  check_subset "trunc_only" "GPU_GCD_MODE=trunc_only GPU_WAVE=128" "$STATE" "$START" "$N" "$CHUNK" "$BASE"
  echo "PASS: GPU knob correctness smoke passed for known nonce $KNOWN and range [$START, $((START+N)))"
  ;;

bench-gpu-knobs)
  CFG="${1:-}"; START="${2:-0}"; N="${3:-16384}"
  [ -f "$CHALLENGE/src/point_add/mod.rs" ] || die "bad CHALLENGE path: $CHALLENGE"
  echo ">> benchmarking against CHALLENGE=$CHALLENGE"
  echo ">> config override file: $CFGF"
  echo ">> range [$START, $((START+N)))"
  echo ">> measured runs=${GPU_BENCH_RUNS:-3}, warmups=${GPU_BENCH_WARMUPS:-1}"
  if [ "${GPU_BENCH_SKIP_INSTALL:-0}" != 1 ]; then
    echo ">> [1/4] installing/rebuilding Rust helpers"
    bash "$SELF" install >/dev/null
  else
    echo ">> [1/4] skipping Rust helper rebuild (GPU_BENCH_SKIP_INSTALL=1)"
  fi
  if [ "${GPU_BENCH_SKIP_BUILD:-0}" != 1 ]; then
    echo ">> [2/4] building CUDA kernel"
    bash "$SELF" build
  else
    echo ">> [2/4] skipping CUDA rebuild (GPU_BENCH_SKIP_BUILD=1)"
  fi
  STATE="$(mktemp).bin"; SUMMARY="$(mktemp)"
  RAW_STATE="$STATE"
  trap 'rm -f "$STATE" "$SUMMARY"' EXIT
  echo ">> [3/4] dumping GPU state"
  bash "$SELF" dump "$CFG" "$STATE"
  if [ "$GPU" != local ]; then
    echo ">> uploading benchmark state once"
    rcp "$STATE" bench_state.bin
    RAW_STATE="\$HOME/$RDIR/bench_state.bin"
  fi
  echo ">> [4/4] measuring kernel throughput"
  bench_variant "baseline" "GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "trunc_first" "GPU_GCD_MODE=trunc_first GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "single_pass" "GPU_GCD_MODE=single_pass GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "wave64" "GPU_WAVE=64" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "wave256" "GPU_WAVE=256" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "batch_inv" "GPU_BATCH_INV=1 GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "comb16" "GPU_COMB_BITS=16 GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "batch_wave256" "GPU_BATCH_INV=1 GPU_WAVE=256" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "batch_comb16" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "batch_comb16_single" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=single_pass GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "fan${GPU_BENCH_FAN_BITS:-22}" "GPU_FAN_BITS=${GPU_BENCH_FAN_BITS:-22} GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "best_combo" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=single_pass GPU_FAN_BITS=${GPU_BENCH_FAN_BITS:-22} GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  bench_variant "all_exact" "GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=trunc_first GPU_WAVE=256" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  BENCH_COMB_BITS="${GPU_BENCH_COMB_BITS:-}"
  if [ -z "$BENCH_COMB_BITS" ] && truthy "${GPU_BENCH_LARGE_COMB:-0}"; then BENCH_COMB_BITS="20 22"; fi
  for bits in $BENCH_COMB_BITS; do
    bench_variant "comb$bits" "GPU_COMB_BITS=$bits GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
    bench_variant "batch_comb$bits" "GPU_BATCH_INV=1 GPU_COMB_BITS=$bits GPU_WAVE=128" "$RAW_STATE" "$START" "$N" "$SUMMARY"
  done
  echo
  echo "variant        avg_nonce_s   min       max       speedup_vs_baseline"
  awk '
    NR==1 { base=$2 }
    {
      speed = (base > 0) ? $2 / base : 0
      printf "%-14s %11.0f %9.0f %9.0f %8.3fx\n", $1, $2, $3, $4, speed
    }
  ' "$SUMMARY"
  ;;

validate)
  CFG="${1:-}"; shift || true; [ $# -gt 0 ] || die "usage: validate CFG NONCE..."
  if truthy "${VALIDATE_REUSE_OPS:-0}"; then
    eval_supports_tail_nonce || die "VALIDATE_REUSE_OPS=1 needs patches/eval_stage2_prefilter.diff applied and eval_circuit rebuilt"
    d="$(mktemp -d)"
    build_out=$( cd "$d" && env ${CFG:+$CFG} DIALOG_TAIL_NONCE=0 "$BIN/build_circuit" 2>&1 )
    build_rc=$?
    if [ "$build_rc" -ne 0 ]; then
      detail=$(echo "$build_out" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g')
      rm -rf "$d"
      for nonce in "$@"; do emit_validation_record "ERROR nonce=$nonce stage=build rc=$build_rc detail=${detail:-build_circuit failed}"; done
      exit 0
    fi
    for nonce in "$@"; do
      out=$( cd "$d" && env ${CFG:+$CFG} EVAL_FAST_REJECT="${EVAL_FAST_REJECT:-1}" EVAL_TAIL_NONCE="$nonce" "$BIN/eval_circuit" --note "isl-$nonce" 2>&1 )
      eval_rc=$?
      emit_validation_line "$nonce" "$out" "$eval_rc"
    done
    rm -rf "$d"
  else
    for nonce in "$@"; do
      d="$(mktemp -d)"
      build_out=$( cd "$d" && env ${CFG:+$CFG} DIALOG_TAIL_NONCE="$nonce" "$BIN/build_circuit" 2>&1 )
      build_rc=$?
      if [ "$build_rc" -ne 0 ]; then
        detail=$(echo "$build_out" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g')
        rm -rf "$d"
        emit_validation_record "ERROR nonce=$nonce stage=build rc=$build_rc detail=${detail:-build_circuit failed}"
        continue
      fi
      out=$( cd "$d" && env ${CFG:+$CFG} EVAL_FAST_REJECT="${EVAL_FAST_REJECT:-1}" DIALOG_TAIL_NONCE="$nonce" "$BIN/eval_circuit" --note "isl-$nonce" 2>&1 )
      eval_rc=$?
      rm -rf "$d"
      emit_validation_line "$nonce" "$out" "$eval_rc"
    done
  fi
  ;;

stage2)
  CFG="${1:-}"; CAND="${2:-}"; OUT="${3:-stage2-results.log}"; JOBS="${4:-$(default_jobs)}"
  [ -n "$CAND" ] || die "usage: stage2 CFG CANDIDATES|- [RESULTS.log] [JOBS]"
  [ "$JOBS" -gt 0 ] 2>/dev/null || die "JOBS must be a positive integer"
  eval_supports_tail_nonce || die "stage2 needs patches/eval_stage2_prefilter.diff applied and eval_circuit rebuilt with EVAL_TAIL_NONCE support"
  ALL="$(mktemp)"; SEEN="$(mktemp)"; TODO="$(mktemp)"
  WORK="$(mktemp -d)"
  trap 'rm -f "$ALL" "$SEEN" "$TODO"; rm -rf "$WORK"' EXIT
  if [ "$CAND" = "-" ]; then
    extract_nonces > "$ALL"
  else
    [ -f "$CAND" ] || die "candidate file not found: $CAND"
    extract_nonces < "$CAND" > "$ALL"
  fi
  extract_file_nonces "$OUT" > "$SEEN"
  if [ -s "$SEEN" ]; then
    awk 'NR==FNR { done[$1]=1; next } !($1 in done)' "$SEEN" "$ALL" > "$TODO"
  else
    cp "$ALL" "$TODO"
  fi
  total=$(wc -l < "$ALL" | tr -d ' ')
  seen=$(wc -l < "$SEEN" | tr -d ' ')
  todo=$(wc -l < "$TODO" | tr -d ' ')
  EVAL_STAGE2_PREFIX="${EVAL_SHOT_LIMIT:-$EVAL_STAGE2_SHOTS}"
  [ "$EVAL_STAGE2_PREFIX" -gt 0 ] 2>/dev/null || die "EVAL_STAGE2_SHOTS/EVAL_SHOT_LIMIT must be a positive integer"
  EVAL_FAST_REJECT="${EVAL_FAST_REJECT:-1}"
  echo ">> stage2 exact prefilter: candidates=$total already_logged=$seen todo=$todo shots=$EVAL_STAGE2_PREFIX jobs=$JOBS fast_reject=$EVAL_FAST_REJECT out=$OUT"
  if [ "$todo" = 0 ]; then
    echo ">> stage2: no new candidates"
    exit 0
  fi
  mkdir -p "$(dirname "$OUT")"
  if [ -n "${STAGE2_BATCH:-}" ]; then
    echo ">> stage2: STAGE2_BATCH is ignored; this command now builds once and evaluates one nonce per worker"
  fi
  echo ">> stage2: building reusable nonce-0 ops.bin"
  build_out=$( cd "$WORK" && env ${CFG:+$CFG} DIALOG_TAIL_NONCE=0 "$BIN/build_circuit" 2>&1 )
  build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    detail=$(echo "$build_out" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g' | cut -c1-2000)
    die "stage2 build failed rc=$build_rc detail=${detail:-build_circuit failed}"
  fi
  [ -s "$WORK/ops.bin" ] || die "stage2 build did not produce ops.bin in $WORK"
  export EVAL_STAGE2_PREFIX EVAL_FAST_REJECT
  xargs -n 1 -P "$JOBS" bash -c '
    bin="$1"; work="$2"; cfg="$3"; out_file="$4"; lock_file="$5"; prefix="$6"; fast="$7"; nonce="$8"
    out=$( cd "$work" && env ${cfg:+$cfg} EVAL_STAGE2_MODE=1 EVAL_FAST_REJECT="$fast" EVAL_SHOT_LIMIT="$prefix" EVAL_TAIL_NONCE="$nonce" "$bin/eval_circuit" --note "stage2-$nonce" 2>&1 )
    rc=$?
    cls=$(printf "%s\n" "$out" | grep "classical mismatches" | grep -oE "[0-9]+$" | tail -1)
    pha=$(printf "%s\n" "$out" | grep "phase-garbage" | grep -oE "[0-9]+$" | tail -1)
    anc=$(printf "%s\n" "$out" | grep "ancilla-garbage" | grep -oE "[0-9]+$" | tail -1)
    tof=$(printf "%s\n" "$out" | grep "avg executed Toffoli" | grep -oE "[0-9.]+" | head -1)
    q=$(printf "%s\n" "$out" | grep -E "^  qubits " | grep -oE "[0-9]+$" | head -1)
    shots=$(printf "%s\n" "$out" | grep "tested shots" | grep -oE "[0-9]+$" | head -1)
    if [ -z "${cls:-}" ] || [ -z "${pha:-}" ] || [ -z "${anc:-}" ]; then
      detail=$(printf "%s\n" "$out" | tr "\n" " " | sed -E "s/[[:space:]]+/ /g" | cut -c1-2000)
      line="ERROR nonce=$nonce stage=stage2 rc=$rc detail=${detail:-eval_circuit failed}"
    elif [ "$cls" = 0 ] && [ "$pha" = 0 ] && [ "$anc" = 0 ]; then
      line="stage2-pass nonce=$nonce shots=${shots:-?} cls=0 pha=0 anc=0 tof=${tof:-?} qubits=${q:-?}"
    else
      line="stage2-reject nonce=$nonce shots=${shots:-?} cls=${cls:-?} pha=${pha:-?} anc=${anc:-?} tof=${tof:-?} qubits=${q:-?}"
    fi
    if command -v flock >/dev/null 2>&1; then
      { flock 9; printf "%s\n" "$line" >> "$out_file"; } 9>"$lock_file"
    else
      printf "%s\n" "$line" >> "$out_file"
    fi
    printf "%s\n" "$line"
  ' _ "$BIN" "$WORK" "$CFG" "$OUT" "$OUT.lock" "$EVAL_STAGE2_PREFIX" "$EVAL_FAST_REJECT" < "$TODO"
  echo ">> stage2 survivors: $(grep -c '^stage2-pass nonce=' "$OUT" 2>/dev/null || true)"
  ;;

obligations)
  SUB="${1:-help}"; shift || true
  case "$SUB" in
  emit-default|emit-dialog-gcd)
    OUT="${1:-}"
    MODE="${SUB#emit-}"
    [ -x "$BIN/obligation_filter" ] || die "obligation_filter not built. Run: ./island.sh install"
    if [ -n "$OUT" ]; then
      mkdir -p "$(dirname "$OUT")"
      "$BIN/obligation_filter" "emit-$MODE" > "$OUT"
      echo ">> wrote obligation manifest $OUT"
    else
      "$BIN/obligation_filter" "emit-$MODE"
    fi
    ;;
  check)
    CFG="${1:-}"; MANIFEST="${2:-}"; CAND="${3:-}"; OUT="${4:-obligation-results.log}"; JOBS="${5:-${OBLIGATION_JOBS:-$(default_jobs)}}"
    [ -n "$CAND" ] || die "usage: obligations check CFG MANIFEST CANDIDATES|- [RESULTS.log] [JOBS]"
    [ -f "$MANIFEST" ] || die "manifest file not found: $MANIFEST"
    [ "$JOBS" -gt 0 ] 2>/dev/null || die "JOBS must be a positive integer"
    [ -x "$BIN/obligation_filter" ] || die "obligation_filter not built. Run: ./island.sh install"
    ALL="$(mktemp)"; SEEN="$(mktemp)"; TODO="$(mktemp)"
    trap 'rm -f "$ALL" "$SEEN" "$TODO"' EXIT
    if [ "$CAND" = "-" ]; then
      extract_nonces > "$ALL"
    else
      [ -f "$CAND" ] || die "candidate file not found: $CAND"
      extract_nonces < "$CAND" > "$ALL"
    fi
    extract_file_nonces "$OUT" > "$SEEN"
    if [ -s "$SEEN" ]; then
      awk 'NR==FNR { done[$1]=1; next } !($1 in done)' "$SEEN" "$ALL" > "$TODO"
    else
      cp "$ALL" "$TODO"
    fi
    total=$(wc -l < "$ALL" | tr -d ' ')
    seen=$(wc -l < "$SEEN" | tr -d ' ')
    todo=$(wc -l < "$TODO" | tr -d ' ')
    OBLIGATION_SHOTS="${OBLIGATION_SHOTS:-9024}"
    OBLIGATION_BATCH="${OBLIGATION_BATCH:-64}"
    [ "$OBLIGATION_SHOTS" -gt 0 ] 2>/dev/null || die "OBLIGATION_SHOTS must be a positive integer"
    [ "$OBLIGATION_SHOTS" -le 9024 ] 2>/dev/null || die "OBLIGATION_SHOTS must be <= 9024"
    [ "$OBLIGATION_BATCH" -gt 0 ] 2>/dev/null || die "OBLIGATION_BATCH must be a positive integer"
    ERR="${OBLIGATION_ERRORS_LOG:-$(dirname "$OUT")/obligation-errors.log}"
    echo ">> exact obligation prefilter: candidates=$total already_logged=$seen todo=$todo shots=$OBLIGATION_SHOTS jobs=$JOBS batch=$OBLIGATION_BATCH out=$OUT errors=$ERR"
    if [ "$todo" = 0 ]; then
      echo ">> obligations: no new candidates"
      exit 0
    fi
    mkdir -p "$(dirname "$OUT")" "$(dirname "$ERR")"
    export OBLIGATION_SHOTS
    xargs -n "$OBLIGATION_BATCH" -P "$JOBS" bash -c '
      bin="$1"; cfg="$2"; manifest="$3"; out_file="$4"; out_lock="$5"; err_file="$6"; err_lock="$7"; shift 7
      out=$(env ${cfg:+$cfg} "$bin/obligation_filter" check "$manifest" "$@" 2>&1)
      rc=$?
      if [ "$rc" -ne 0 ]; then
        detail=$(printf "%s\n" "$out" | tr "\n" " " | sed -E "s/[[:space:]]+/ /g" | cut -c1-2000)
        for nonce in "$@"; do
          line="ERROR nonce=$nonce stage=obligations rc=$rc detail=${detail:-obligation_filter failed}"
          if command -v flock >/dev/null 2>&1; then
            { flock 9; printf "%s\n" "$line" >> "$err_file"; } 9>"$err_lock"
          else
            printf "%s\n" "$line" >> "$err_file"
          fi
          printf "%s\n" "$line"
        done
        exit 0
      fi
      printf "%s\n" "$out" | while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
          obligation-pass\ nonce=*|obligation-reject\ nonce=*) ;;
          *) printf "%s\n" "$line" >&2; continue ;;
        esac
        if command -v flock >/dev/null 2>&1; then
          { flock 9; printf "%s\n" "$line" >> "$out_file"; } 9>"$out_lock"
        else
          printf "%s\n" "$line" >> "$out_file"
        fi
        printf "%s\n" "$line"
      done
    ' _ "$BIN" "$CFG" "$MANIFEST" "$OUT" "$OUT.lock" "$ERR" "$ERR.lock" < "$TODO"
    echo ">> obligation survivors: $(grep -c '^obligation-pass nonce=' "$OUT" 2>/dev/null || true)"
    ;;
  help|*)
    echo "usage:"
    echo "  ./island.sh obligations emit-default [OUT]"
    echo "  ./island.sh obligations emit-dialog-gcd [OUT]"
    echo "  ./island.sh obligations check CFG MANIFEST CANDIDATES|- [RESULTS.log] [JOBS]"
    ;;
  esac
  ;;

bake)
  [ $# -ge 2 ] || die "usage: bake KEY VALUE [KEY VALUE ...]"
  M="$CHALLENGE/src/point_add/mod.rs"; cr0=$(grep -c $'\r' "$M" || true)
  while [ $# -ge 2 ]; do perl -i -pe 's/(set_default_env\("'"$1"'", ")[^"]*("\))/${1}'"$2"'${2}/' "$M"; shift 2; done
  cr1=$(grep -c $'\r' "$M" || true)
  [ "$cr0" = "$cr1" ] || echo "WARNING: CR count changed ($cr0->$cr1) — CRLF corrupted! revert: (cd $CHALLENGE && git checkout -- src/point_add/mod.rs)"
  echo ">> diff (must be exactly your lines):"; ( cd "$CHALLENGE" && git --no-pager diff src/point_add/mod.rs | grep -E "^[-+]" | grep -vE "^[-+][-+]" || true )
  echo ">> ecdsafail run:"; ( cd "$CHALLENGE" && ecdsafail run 2>&1 | grep -iE "classical mismatch|phase-garbage|ancilla|score" | tail -4 )
  ;;

hunt)
  CFG="${1:?usage: hunt CFG START N [CHUNK]}"; START="${2:?}"; N="${3:?}"; CHUNK="${4:-500000}"
  echo ">> [1/3] Toffoli cost:"; bash "$SELF" measure "$CFG"
  echo ">> [2/3] dump + multi-GPU search ($N nonces from $START, chunk $CHUNK):"
  STATE="$(mktemp).bin"; bash "$SELF" dump "$CFG" "$STATE" >/dev/null
  # CHUNK only affects throughput, not correctness: the batch+large-comb combo is exact and
  # scale-invariant (verified — identical candidates at 200k/1M/6M; see measured-speedups.md
  # "Per-process startup cost & chunk sizing"). Chunk size just amortizes the ~1s per-process
  # table build (~6% at 200k, ~1.3% at 1M). Raise CHUNK to ~1M for long/billion-scale runs.
  cands=$(bash "$SELF" search "$STATE" "$START" "$N" "$CHUNK" | grep -oE '[0-9]+' | sort -un)
  echo "GCD-clean candidates: $(echo "$cands" | grep -c . || true)"
  echo ">> [3/3] validating (looking for 0/0/0):"; found=0
  for n in $cands; do line=$(bash "$SELF" validate "$CFG" "$n"); echo "$line"; case "$line" in CLEAN*) found=1;; esac; done
  [ "$found" = 1 ] || echo "(no fully-clean island in this range — search a larger N)"
  ;;

help|*) sed -n '2,20p' "$SELF" ;;
esac
