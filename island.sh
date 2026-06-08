#!/usr/bin/env bash
# ecdsafail-island-gpu — GPU-accelerated Fiat-Shamir island search for ecdsa.fail.
# Portable across NVIDIA GPUs (A100/H100/H200, RTX 30/40/50), multi-GPU, local or remote SSH.
#
#   ./island.sh init-local  [CHALLENGE]                 # configure for a local GPU + build
#   ./island.sh init-remote "ssh -p PORT user@IP" [CHALLENGE]  # configure for a remote GPU box + build
#   ./island.sh doctor                                  # report GPU/CUDA on the target
#   ./island.sh install                                 # build the Rust helpers in the challenge repo
#   ./island.sh measure [CFG]                           # Toffoli (CCX) cost of a config
#   ./island.sh build  [2|3]                            # compile CUDA kernel (2=original, 3=batch-inv)
#   ./island.sh dump   CFG OUT.bin                      # gpu_state dump for a config
#   ./island.sh search STATE START N [CHUNK]            # multi-GPU search -> CLEAN nonce=...
#   ./island.sh validate CFG NONCE...                  # quantum-confirm 0/0/0 + score
#   ./island.sh bake   KEY VALUE [...]                  # CRLF-safe mod.rs edit + ecdsafail run
#   ./island.sh hunt   CFG START N                      # measure -> dump -> search -> validate
#   ./island.sh validate-port                           # verify GPU filter matches current base
#   CFG = a space-free env assignment, e.g. DIALOG_GCD_ACTIVE_ITERATIONS=258
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CFGF="$HERE/config.env"
die(){ echo "ERROR: $*" >&2; exit 1; }

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
KERNEL="3"
EOF
  echo ">> wrote $CFGF"
}

case "$cmd" in
init-local)
  CH="${1:-}"; [ -n "$CH" ] || { [ -f "$CFGF" ] && CH="$(. "$CFGF"; echo "${CHALLENGE:-}")"; }
  [ -d "${CH:-}" ] || die "usage: init-local <path-to-ecdsafail-challenge>"
  write_cfg local "" "$CH"; "$0" doctor; "$0" build; exit 0;;
init-remote)
  SSHCMD="${1:?usage: init-remote \"ssh -p PORT user@IP\" [CHALLENGE]}"
  CH="${2:-}"; [ -n "$CH" ] || { [ -f "$CFGF" ] && CH="$(. "$CFGF"; echo "${CHALLENGE:-}")"; }
  [ -d "${CH:-}" ] || die "give the challenge repo path: init-remote \"$SSHCMD\" <path-to-ecdsafail-challenge>"
  case "$SSHCMD" in ssh\ *) ;; *) SSHCMD="ssh $SSHCMD";; esac   # tolerate "-p PORT user@IP" without leading "ssh"
  write_cfg remote "$SSHCMD" "$CH"; "$0" doctor; "$0" build; exit 0;;
esac

[ -f "$CFGF" ] || die "not configured. Run: ./island.sh init-local <CHALLENGE>   OR   ./island.sh init-remote \"ssh -p PORT user@IP\" <CHALLENGE>"
# shellcheck disable=SC1090
. "$CFGF"
: "${CHALLENGE:?set CHALLENGE in config.env}"; : "${GPU:=local}"; : "${REMOTE_SSH:=}"
: "${REMOTE_DIR:=.ecdsafail_island}"; : "${NVCC_ARCH:=auto}"; : "${GPUS:=auto}"; : "${BLOCKS:=512}"; : "${KERNEL:=3}"
BIN="$CHALLENGE/target/release"; RDIR="$REMOTE_DIR"
ksrc(){ [ "$KERNEL" = 3 ] && echo "$HERE/cuda/gpu_island3.cu" || echo "$HERE/cuda/gpu_island2.cu"; }
kbin(){ [ "$KERNEL" = 3 ] && echo "$HERE/gpu_island3" || echo "$HERE/gpu_island2"; }
KSRC="$(ksrc)"; KBIN="$(kbin)"
need_remote(){ [ -n "$REMOTE_SSH" ] || die "GPU=remote needs REMOTE_SSH (init-remote)"; }
rhost(){ echo "$REMOTE_SSH" | grep -oE '[^ ]+@[^ ]+' | head -1; }
rport(){ echo "$REMOTE_SSH" | grep -oE '\-p +[0-9]+' | grep -oE '[0-9]+'; }
rkey(){ local k; k=$(echo "$REMOTE_SSH" | grep -oE '\-i +[^ ]+' | awk '{print $2}'); echo "${k/#\~/$HOME}"; }
rsh(){ $REMOTE_SSH "$@"; }
rcp(){ local p k; p="$(rport)"; k="$(rkey)"; scp ${p:+-P "$p"} ${k:+-i "$k"} -o StrictHostKeyChecking=no "$1" "$(rhost):$RDIR/$2" >/dev/null; }
push_runtime(){ rsh "mkdir -p $RDIR"; rcp "$KSRC" "gpu_island.cu"; for s in build_kernel search_driver doctor; do rcp "$HERE/runtime/$s.sh" "$s.sh"; done; }

case "$cmd" in

doctor)
  if [ "$GPU" = local ]; then bash "$HERE/runtime/doctor.sh"
  else need_remote; push_runtime >/dev/null 2>&1 || true; rsh "bash $RDIR/doctor.sh"; fi
  ;;

install)
  cp "$HERE"/rust/*.rs "$CHALLENGE/src/bin/"
  ( cd "$CHALLENGE" && cargo build --release \
      --bin build_circuit --bin eval_circuit --bin dump_gpu_state --bin count_tof --bin island_search 2>&1 | grep -E 'Compiling|Finished|error' )
  rc=$?; [ $rc -ne 0 ] && die "cargo build failed — fix errors before proceeding"
  bins="build_circuit eval_circuit dump_gpu_state count_tof island_search"
  for b in $bins; do [ -x "$BIN/$b" ] || die "$BIN/$b missing after build — stale binary risk"; done
  echo ">> installed $(echo $bins | wc -w) binaries. Sanity: (cd $CHALLENGE && ecdsafail run) must print 0/0/0."
  ;;

measure)
  echo "baseline: $("$BIN/count_tof")"; [ $# -gt 0 ] && echo "$1: $(env "$1" "$BIN/count_tof")" ;;

dump)
  CFG="${1:-}"; OUT="${2:?usage: dump CFG OUT.bin}"
  env ${CFG:+$CFG} "$BIN/dump_gpu_state" "$OUT" | grep -E "cfg:|n_ops|KECCAK MATCH"; echo ">> wrote $OUT" ;;

build)
  KN="${1:-$KERNEL}"
  KS="$( [ "$KN" = 3 ] && echo "$HERE/cuda/gpu_island3.cu" || echo "$HERE/cuda/gpu_island2.cu" )"
  KB="$( [ "$KN" = 3 ] && echo "$HERE/gpu_island3" || echo "$HERE/gpu_island2" )"
  if [ "$GPU" = local ]; then
    bash "$HERE/runtime/build_kernel.sh" "$KS" "$KB" "$NVCC_ARCH"
  else
    need_remote; rcp "$KS" "gpu_island.cu"; rsh "mkdir -p $RDIR"
    for s in build_kernel search_driver doctor; do rcp "$HERE/runtime/$s.sh" "$s.sh"; done
    rsh "bash $RDIR/build_kernel.sh \$HOME/$RDIR/gpu_island.cu \$HOME/$RDIR/gpu_island $NVCC_ARCH"
  fi
  ;;

search)
  STATE="${1:?usage: search STATE START N [CHUNK]}"; START="${2:?}"; N="${3:?}"; CHUNK="${4:-200000}"
  KENV=$( [ "$KERNEL" = 3 ] && echo "KERNEL3=1" || echo "KERNEL2=1" )
  if [ "$GPU" = local ]; then
    [ -x "$KBIN" ] || die "run './island.sh build' first (kernel $KERNEL)"
    GPU_ISLAND_BIN="$KBIN" GPU_STATE_FILE="$STATE" BLOCKS="$BLOCKS" $KENV \
      bash "$HERE/runtime/search_driver.sh" "$START" "$N" "$CHUNK" "$GPUS"
  else
    need_remote; rcp "$STATE" state.bin
    rsh "GPU_ISLAND_BIN=\$HOME/$RDIR/gpu_island GPU_STATE_FILE=\$HOME/$RDIR/state.bin BLOCKS=$BLOCKS $KENV \
         bash \$HOME/$RDIR/search_driver.sh $START $N $CHUNK $GPUS"
  fi
  ;;

validate)
  CFG="${1:-}"; shift || true; [ $# -gt 0 ] || die "usage: validate CFG NONCE..."
  for nonce in "$@"; do
    d="$(mktemp -d)"
    ( cd "$d" && env ${CFG:+$CFG} DIALOG_TAIL_NONCE="$nonce" "$BIN/build_circuit" >/dev/null 2>&1 )
    out=$( cd "$d" && env ${CFG:+$CFG} DIALOG_TAIL_NONCE="$nonce" "$BIN/eval_circuit" --note "isl-$nonce" 2>&1 ); rm -rf "$d"
    cls=$(echo "$out"|grep "classical mismatches"|grep -oE '[0-9]+$'); pha=$(echo "$out"|grep "phase-garbage"|grep -oE '[0-9]+$')
    anc=$(echo "$out"|grep "ancilla-garbage"|grep -oE '[0-9]+$'); tof=$(echo "$out"|grep "avg executed Toffoli"|grep -oE '[0-9.]+'|head -1)
    q=$(echo "$out"|grep -E '^  qubits '|grep -oE '[0-9]+$')
    if [ "${cls:-x}" = 0 ] && [ "${pha:-x}" = 0 ] && [ "${anc:-x}" = 0 ]; then
      echo "CLEAN nonce=$nonce tof=$tof qubits=$q score=$(python3 -c "print(int(round(float('$tof')))*int('$q'))")"
    else echo "dirty nonce=$nonce cls=${cls:-?} pha=${pha:-?} anc=${anc:-?}"; fi
  done
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
  CFG="${1:?usage: hunt CFG START N}"; START="${2:?}"; N="${3:?}"
  echo ">> [1/3] Toffoli cost:"; "$0" measure "$CFG"
  echo ">> [2/3] dump + multi-GPU search ($N nonces from $START):"
  STATE="$(mktemp).bin"; "$0" dump "$CFG" "$STATE" >/dev/null
  cands=$("$0" search "$STATE" "$START" "$N" | grep -oE '[0-9]+' | sort -un)
  echo "GCD-clean candidates: $(echo "$cands" | grep -c . || true)"
  echo ">> [3/3] validating (looking for 0/0/0):"; found=0
  for n in $cands; do line=$("$0" validate "$CFG" "$n"); echo "$line"; case "$line" in CLEAN*) found=1;; esac; done
  [ "$found" = 1 ] || echo "(no fully-clean island in this range — search a larger N)"
  ;;

validate-port)
  NONCE=$(grep -oP 'DIALOG_TAIL_NONCE.*?=\K[0-9]+' "$CHALLENGE/src/point_add/mod.rs" | head -1)
  [ -n "$NONCE" ] || die "can't read DIALOG_TAIL_NONCE from $CHALLENGE/src/point_add/mod.rs"
  echo ">> validating port against base nonce=$NONCE"
  STATE="$(mktemp).bin"; "$0" dump "" "$STATE" >/dev/null
  result=$("$0" search "$STATE" "$NONCE" 1 2>/dev/null | grep -c "CLEAN nonce=$NONCE" || true)
  rm -f "$STATE"
  if [ "$result" -ge 1 ]; then echo ">> PORT OK — GPU flags base nonce as CLEAN"; else
    echo ">> PORT FAILED — GPU did NOT flag base nonce as CLEAN"; echo "   Re-run './island.sh install' after sync, then './island.sh build'."; exit 1; fi
  ;;

help|*) sed -n '2,20p' "$0" ;;
esac
