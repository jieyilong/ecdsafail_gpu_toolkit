---
name: ecdsafail-island
description: >-
  GPU-accelerated Fiat-Shamir "island" (DIALOG_TAIL_NONCE) search for the ecdsa.fail
  quantum point-add challenge. Use whenever optimizing the ecdsa.fail / ecdsafail
  reversible secp256k1 point-addition circuit and you need to (re-)hunt a clean tail
  nonce after tightening a config lever (DIALOG_GCD_ACTIVE_ITERATIONS, COMPARE_BITS,
  APPLY_CLEAN_COMPARE_BITS, WIDTH_SLOPE/MARGIN, KAL carry truncs, etc.), or to measure a
  lever's Toffoli cost, or to bake + submit a found island. Triggers: "find an island",
  "re-hunt the nonce", "lower active_iterations", "beat the ecdsa.fail SOTA".
---

# ecdsafail-island skill

You are driving a GPU island searcher for the ecdsa.fail challenge. Score =
`avg_executed_Toffoli × peak_qubits` (lower better); the circuit is validated on 9,024
SHAKE256-derived inputs reseeded by a free 96-gate identity tail (`DIALOG_TAIL_NONCE`).
Every config tightening reseeds those inputs, so you must re-hunt a clean nonce.

## Setup (once)
Pick local or remote based on what the user says — you do NOT need to know the GPU's `sm_XX`
(it's auto-detected on the target; works for A100/H100/H200 and RTX 30/40/50, single/multi-GPU).

- **Local GPU:** `./island.sh init-local <path-to-ecdsafail-challenge>`
- **Remote GPU:** when the user says something like *"run on a remote GPU machine, here's the
  ssh command: `ssh -p 40162 ubuntu@1.2.3.4`"*, take that exact ssh command and run:
  `./island.sh init-remote "ssh -p 40162 ubuntu@1.2.3.4" <path-to-ecdsafail-challenge>`
  (key-based auth is assumed; if the user only gave `-p PORT user@IP`, prepend `ssh`.)

`init-*` writes `config.env`, runs `doctor` (prints the GPU(s)/CUDA — show this to the user),
and builds the kernel. Then:
1. `./island.sh install` (builds the Rust helpers in the challenge repo). Confirm
   `(cd $CHALLENGE && ecdsafail run)` prints the leaderboard score (stale-binary check).
2. **Validate the port on this base**: read `DIALOG_TAIL_NONCE` from
   `$CHALLENGE/src/point_add/mod.rs`, then
   `./island.sh dump "" /tmp/base.bin && ./island.sh search /tmp/base.bin <that_nonce> 1`
   MUST print `CLEAN`. If not, STOP — the filter/dump doesn't match this base; do not trust results.

To switch GPUs later (e.g. the user gives a new box), just re-run `init-remote`/`init-local`.
The script can be invoked as `./island.sh ...` or `bash island.sh ...`; recursive self-calls
resolve through the script directory.

## GPU search knobs
Every improvement is an independent on/off knob, all defaulting to the exact/conservative
baseline (`GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128 GPU_FAN_BITS=0`
for the scan, `EVAL_FAST_REJECT=0` for the eval). They compose; benchmark combinations with
`bench-gpu-knobs`. Two are new:

To compare against the previous release's search behavior, explicitly set the scan baseline
and clear compatibility aliases:

```bash
unset BATCH_INV GPU_LARGE_COMB GCD_MODE WAVE
GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128 GPU_FAN_BITS=0 ./island.sh search s.bin <START> <N>
```

For strict previous-release validation behavior, add `EVAL_FAST_REJECT=0`; `island.sh
validate` sets it to `1` by default for faster dirty-candidate rejection. On the RTX 5090,
the previous-release binary and this branch with the scan baseline both measured about
10k nonce/s on the same dumped state.

- `GPU_FAN_BITS=K` — **nonce-fan**: precompute the SHAKE sponge for the low `K` tail bits so
  each nonce only absorbs its high bits. Exact. Table is `2^K * 208 B`. Measured ~+1.5% on the
  current SOTA base (`squeeze_init` is not the bottleneck there).
- `EVAL_FAST_REJECT=1` — **eval early-exit / exact apply pre-scan**: defers the per-shot
  EC-muls into the batch loop and stops at the first failing batch. **~8.5× avg** on dirty
  candidates (16.1s → ~1.9s), exact (clean islands still read `0/0/0`; the full eval already
  checks apply-cleanliness, so a fast-rejecting eval *is* the apply pre-scan with no false
  negatives). Lives in the challenge `eval_circuit`; re-apply `patches/eval_fast_reject.diff`
  after `ecdsafail sync`. `island.sh validate` sets it to `1` by default.

For production island searches on a large NVIDIA GPU, prefer the fastest exact mode that has
passed `test-gpu-knobs` on the current base. As of the RTX 5090 measurements, the best
measured exact scan mode is:

```bash
GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_FAN_BITS=22 GPU_WAVE=128 ./island.sh search s.bin <START> <N>
```

(`GPU_FAN_BITS=22` is the fastest measured exact scan, ~13,676 n/s ≈ 1.42× baseline; its
~872 MiB table builds in ~0.3s and amortizes at the default 500k chunk. Drop it only for
tiny chunks ≪200k.)

`GPU_GCD_MODE=single_pass` is a valid necessary filter (won't miss true islands) and adds a
free ~0-4% scan, but it is **not** candidate-set-identical to `full_first` -- it checks
*truncated* GCD convergence (what the circuit runs), `full_first` checks untruncated. They
disagree on borderline eval-dirty nonces: measured, `single_pass` is **looser** (it found
`{46719, 644403}` where `full_first` found only `{644403}` -- a superset -- with `46719`
`single_pass`-specific and eval-dirty). So `single_pass` passes a few more eval-dirty
false-positives for the faster scan; use `full_first` for the strictest pre-filter. Validate
against the eval, not a sparse candidate range. See `docs/measured-speedups.md`.
The `comb22` table is ~3.0 GiB and was only ~2.8% over `comb16`, but its build is only ~0.33s
(measured), so process startup does **not** dominate even at 200k chunks -- prefer `comb22`
unless VRAM is constrained, then `GPU_COMB_BITS=16`.
`GPU_FAN_BITS=22` (with the comb22 combo) is the fastest measured exact scan; its ~872 MiB
table also builds in ~0.3s. For long/billion-scale runs use `CHUNK≈1000000` to amortize the
~1s startup to ~1.3% (vs ~6% at 200k). The combo is exact and scale-invariant at any size, so
chunk size is a throughput/memory knob only -- see "Per-process startup cost & chunk sizing".

Calibrate expectations: the absolute speedup of these exact knobs is **strongly
base-dependent** (e.g. `GPU_BATCH_INV` is ~1.42x on slow-reject bases but only +1.2% on the
current fast-reject frontier base). Native `sm_120` compilation was measured to give no
benefit over PTX-JIT on a 581-series driver. See `docs/measured-speedups.md` for the full
measured table, and always re-measure with `bench-gpu-knobs` on the actual base before
assuming a number.

Overall end-to-end speedup: scan and eval are **sequential** stages, so the scan (≤1.65x) and
eval (~8.5x) speedups **do not multiply** — combined is **up to ~8.5x** where candidate
validation is the bottleneck (apply-bound configs) and **~1.6x** where the GPU scan dominates
(the current frontier base). The lazy eval (`EVAL_FAST_REJECT`) is the dominant lever. See the
"Overall pipeline speedup" section of `docs/measured-speedups.md`.

Before trusting new GPU knob combinations on a fresh base or GPU, run:

```bash
GPU_TEST_COMB_BITS="20 22" ./island.sh test-gpu-knobs "" <START> <N>
```

To compare throughput fairly, run:

```bash
GPU_BENCH_COMB_BITS="20 22" GPU_BENCH_RUNS=2 ./island.sh bench-gpu-knobs "" <START> <N>
```

Always correctness-test first, then benchmark. `GPU_GCD_MODE=trunc_only` is intentionally
noisy and should only be used as a candidate generator followed by normal validation.

## The optimization loop
1. **Measure levers, don't guess.** For each candidate tightening, run
   `./island.sh measure <CFG>` to get its exact Toffoli (CCX). Score win = (baseline_CCX −
   CFG_CCX) × peak. Pick the **biggest** win that is plausibly findable. Historically
   `DIALOG_GCD_ACTIVE_ITERATIONS` is the largest lever (~2,860 Toffoli/step) and is often
   uncontested; comparator bits are small (~144–516). See `docs/levers.md`.
2. **Hunt.** `./island.sh hunt <CFG> <START> <N>` (e.g. `... DIALOG_GCD_ACTIVE_ITERATIONS=258
   1 2000000`). It measures, dumps, GPU-searches, and quantum-confirms candidates, printing
   `CLEAN nonce=... score=...` for any fully-0/0/0 island. If none, search a larger range
   (rarer islands need more nonces; lower active_iterations = rarer).
3. **Bake + submit ONLY a confirmed island.** Use `./island.sh bake <KEY> <VAL> DIALOG_TAIL_NONCE
   <nonce>` — it edits `mod.rs` CRLF-safely (NEVER use a normal file-edit tool on mod.rs; it
   corrupts CRLF and breaks promotion), shows the diff (must be exactly your lines), and runs
   `ecdsafail run`. Then in `$CHALLENGE`: `ecdsafail submit --note-file <note> --model <m>
   --claimed-score <score>`. Watch `ecdsafail submissions` until `promoted`.
4. **Stay on the frontier.** Periodically `cd $CHALLENGE && ecdsafail benchmark` / `submissions`.
   If someone else takes the lead, `ecdsafail sync --force`, re-`install`, re-validate the port,
   and re-run measure→hunt on the new base (apply your lever idea on top of their base).

## Hard rules
- A GPU `CLEAN` is a candidate; only a `./island.sh validate` 0/0/0 is submittable.
- Confirm `ecdsafail run` == leaderboard after every `install`/`sync` (stale-binary trap).
- Only submit if the confirmed score strictly beats the current best.
- Bake with `bake`/`perl` only. Verify the `git diff` is exactly the changed config lines.
