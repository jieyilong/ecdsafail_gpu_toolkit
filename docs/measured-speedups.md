# Measured Speedups

Honest, measured numbers for every GPU/eval knob in this toolkit, so nobody re-derives
them from scratch or trusts an estimate that didn't pan out. All scan-kernel numbers are
RTX 5090 (CUDA 11.5 PTX-JIT to `sm_120`); eval numbers are CPU.

## The headline: speedups are strongly base-dependent

The same knob gives wildly different gains depending on how fast nonces reject in the GPU
filter, which is set by the challenge config (the truncation schedule baked into
`gpu_state.bin`):

- **Slow-reject base** — many shots run before the first hard one, so per-shot field
  arithmetic dominates. `GPU_BATCH_INV` (which attacks the per-shot inversions) shines.
- **Fast-reject base** — nonces die after a few shots, so the per-nonce `squeeze_init`
  (~39 Keccak-f) plus a wave of shot work dominate, and per-shot levers barely run.

Always re-measure with `bench-gpu-knobs` on the *actual* base before assuming a number.

## Scan kernel (exact knobs — candidate set unchanged)

| knob | current SOTA base *(fast-reject)* | earlier base *(slow-reject)* | notes |
|---|---:|---:|---|
| **baseline** | 1.00× (10,209 n/s) | 1.00× (8,400 n/s) | default exact path |
| `GPU_GCD_MODE=single_pass` | ~1.00× | +2–4% (in stack) | folds the two GCD passes into one |
| `GPU_BATCH_INV=1` | **1.01×** (+1.2%) | **1.42×** | the only large lever; only on slow-reject |
| `GPU_COMB_BITS=16` | 1.11× | 1.10× | ~64 MiB table |
| `GPU_COMB_BITS=22` | +3% over comb16 | +3% over comb16 | ~3.0 GiB table; diminishing returns |
| `GPU_FAN_BITS` (nonce-fan) | **1.015×** (+1.5%) | not measured | `squeeze_init` is not the bottleneck here |
| native `sm_120` build (CUDA 12.8) | **1.00×** (no gain) | — | PTX-JIT matches/beats native codegen |
| `GPU_WAVE=64` | 0.78× *(slower)* | 0.84× *(slower)* | smaller waves = more overhead |
| **best exact combo** | **1.18×** (`batch_comb16` + `single_pass`) | **1.65×** (`batch_comb22`) | |

Combination caveat: `GPU_WAVE=256` *hurts* in batch mode (lower occupancy), so the best
combo uses `GPU_WAVE=128` — e.g. `all_exact` (which forces wave256) measured slower than
`batch_comb16`.

## Eval phase (challenge `eval_circuit`, exact) — the one big exact win

| knob | effect | status |
|---|---:|---|
| `EVAL_FAST_REJECT=1` (lazy derivation + early-exit) | **~8.5× avg** on dirty candidates (16.1s → ~1.9s) | built |
| clean island under fast-reject | unchanged: `0/0/0`, all 9024 shots OK | exact |
| `FAST=0` scoring path | byte-identical to original (same full counts) | exact |

`EVAL_FAST_REJECT` **defers the per-shot EC scalar-mults into the batch loop** and stops at
the first failing batch, so a dirty candidate exits after ~the first bad batch (~1–3.6 s)
instead of paying the full ~9 s upfront derivation + 141-batch sim. The earlier
simple-early-exit version was capped at ~1.5× by that upfront derivation; deferring it is
what unlocks the ~10×.

This is the **exact realization of the "apply pre-scan"**: the full eval already checks
apply-cleanliness, so a fast-rejecting eval *is* the apply pre-scan — with **zero false
negatives** and no GPU re-implementation of the apply phase. A clean island still derives and
simulates all 9024 shots and reads `0/0/0`.

Measured dirty-candidate eval times: 1.13, 1.24, 1.75, 1.75, 1.85, 3.57 s (vs 16.1 s full) →
**~8.5× average**. Spread is because the exit time depends on where the first bad shot falls.

## Overall pipeline speedup (scan and eval do NOT multiply)

The scan (GPU) and the eval (CPU) are **sequential** stages: total time = `scan + eval`. So
their speedups **do not multiply** — you sum the *reduced* times, and the stage that's still
slower after speedup caps the result. The two levers are:

- scan, all knobs: **1.65×** (slow-reject base) ... **1.18×** (current fast-reject base)
- eval, lazy fast-reject: **~8.5×** per dirty candidate

If the baseline spent fraction `f` of its time scanning and `1-f` evaling, the combined
speedup is `1 / (f/1.65 + (1-f)/8.5)`:

| where the baseline's time went | combined end-to-end |
|---|---:|
| ~all eval (apply-bound configs) | **~8.5×** |
| 50/50 scan/eval | **~2.8×** |
| ~all scan (scan-bound frontier base) | **~1.6×** |

So **combining all improvements gives up to ~8.5× end-to-end** where validating candidates is
the bottleneck (apply-bound — the regime these tools target), and **~1.6×** where the GPU scan
dominates (the current frontier base, where the island density is the structural wall). It is
**never** `1.65 × 8.5 ≈ 14×` — that product would require the two stages to run in parallel,
but they are sequential. The single biggest contributor by far is the lazy eval (the exact
apply pre-scan); the scan knobs add a modest `1.2–1.65×` only when scanning is a meaningful
share of the time.

## What did NOT pan out (and why)

Three predicted multipliers came in at low single digits — recorded so they aren't retried
on the same reasoning:

- **`single_pass` GCD** (predicted ~1.2×): the convergence pass it removes already
  early-exits cheaply, so removing it barely helps.
- **native `sm_120`** (predicted ~1.2×): the 581-series driver's PTX-JIT from `compute_80`
  matches/beats nvcc-12.8 native Blackwell codegen — measured *slightly slower* native.
- **nonce-fan** (predicted ~1.4×): `squeeze_init` is only ~3% of per-nonce time on the
  current base, not the bottleneck; halving it bought +1.5%.

The lesson: the scan kernel is near its wall on the current SOTA base, and predicting
multipliers from crude profiles overshot repeatedly. The genuine remaining headroom is a
**speculative apply pre-scan** between the GCD scan and the full eval (the apply-bound
regime), not further scan-kernel micro-optimization.

## Methodology

- Scan: `bench-gpu-knobs` (warmups + measured repeats, raw `gpu_island2` `nonce/s` timer)
  over a fixed dirty range; speedup is an on-vs-off ratio within one run on one base.
- Exactness: `test-gpu-knobs` confirms each knob finds the known island and produces a
  bit-identical candidate set vs the `full_first` baseline before any speed claim.
- Eval: wall-clock of `eval_circuit` on a dirty candidate (`EVAL_FAST_REJECT` 0 vs 1), with
  the clean island re-checked to confirm `0/0/0` is preserved.
