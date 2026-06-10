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

## Eval phase (challenge `eval_circuit`, exact)

| knob | effect | status |
|---|---:|---|
| `EVAL_FAST_REJECT=1` (simple early-exit) | **~1.5×** on dirty candidates (16.3s → ~10.5s) | built |
| lazy-derivation upgrade | ~10× projected | not built |
| clean island under fast-reject | unchanged: `0/0/0`, all 9024 shots OK | exact |

`EVAL_FAST_REJECT` stops at the first failing batch. The ~1.5× ceiling is because the eval
derives **all 9024 test inputs upfront** (~9 s of CPU EC scalar-mults) before the batch loop
starts, and early-exit only shortcuts the batch loop. Deferring those muls into the batch
loop (lazy derivation) would reach ~10× but requires a careful rewrite of the *trusted
scoring* binary — worth it mainly for apply-bound configs where eval dominates the search.

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
