# 1216-balanced scan recipe

This branch records the 1216-qubit pivot config used for the distributed island scan.
It is intentionally less aggressive than the 1211-qubit attempt, which produced many
GCD-clean candidates but dirty full-eval triples around `cls/pha = 60..80 / 27..34`.

## Circuit config

```bash
CFG="DIALOG_GCD_APPLY_FUSED_FOLD=0 ROUND84_FOLD_FAST_ADD=0 SQUARE_ROW_MAX_SEG=184 DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS=12"
```

Measured locally with `count_tof`:

```text
qubits=1216
CCX/Toffoli=1506570
score=1216 * 1506570 = 1831989120
```

The first dirty inherited nonce run had much smaller violations than the 1211-qubit
variant: `cls/pha/anc = 16 / 6 / 0`.

## Phase 1: GPU scan

Use the safer fast scan settings. They preserve the production GCD filter while reducing
per-nonce work and stream candidates after each chunk.

```bash
./island.sh dump "$CFG" s-1216-balanced.bin

GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=trunc_first GPU_FAN_BITS=22 GPU_WAVE=128 \
  ./island.sh search s-1216-balanced.bin <START> <N>
```

For long multi-node scans, deploy `runtime/remote_gpu_scan_loop.sh` with `gpu_island2`,
`search_driver.sh`, and the dumped state. It writes:

```text
<tag>.status      tab-separated progress row
<tag>.candidates  streamed CLEAN nonce=... lines
```

The status row is:

```text
gpu_name start count done_count candidate_count elapsed rate remaining eta pct cfg
```

## Phase 2: CPU validation

Every GPU candidate must pass both fast and full validation. The fast path may stop at the
first dirty shot, so exact triples are only required to match on a clean island.

```bash
EVAL_FAST_REJECT=1 ./island.sh validate "$CFG" <nonce...>
EVAL_FAST_REJECT=0 ./island.sh validate "$CFG" <same nonce...>
```

For dirty candidates, check the componentwise prefix-count invariant:

```text
cls_fast <= cls_full
pha_fast <= pha_full
anc_fast <= anc_full
```

If a nonce validates as `0 / 0 / 0` under full eval, refresh the public SOTA before
submitting:

```bash
cd "$CHALLENGE"
ecdsafail submissions --all
ecdsafail sync
```

Submit only if `1216 * 1506570` is lower than the current promoted SOTA score.
