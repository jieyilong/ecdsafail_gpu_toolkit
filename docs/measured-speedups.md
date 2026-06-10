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

## Matching the previous release baseline

The previous release hard-coded the shot-parallel `gpu_island2` path with `WAVE=128` and no
runtime performance knobs. To make this branch behave like that search path, use:

```bash
unset BATCH_INV GPU_LARGE_COMB GCD_MODE WAVE
GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128 GPU_FAN_BITS=0
```

For validation parity with the previous release, also use `EVAL_FAST_REJECT=0`. The search
knobs above match the candidate set; `EVAL_FAST_REJECT=1` only changes how quickly dirty
candidates are rejected during CPU eval.

Same RTX 5090, same dumped state, 32,768-nonce dirty range:

| binary / knob set | measured throughput |
|---|---:|
| previous-release binary | ~10,057 nonce/s |
| this branch with previous-release-compatible knobs | ~10,062 nonce/s |
| `speculative-and-fan` recommended exact stack | ~12,505 nonce/s |

So the previous release baseline on this machine is about 10k nonce/s. The earlier 7k-ish
number corresponds to slower wave settings such as `GPU_WAVE=64`, not to the previous
release.

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

## Recommended settings

For normal exact GPU search on a large NVIDIA GPU:

```bash
GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_WAVE=128 GPU_FAN_BITS=0
```

Add `EVAL_FAST_REJECT=1` for candidate validation unless you intentionally need byte-for-byte
previous-release diagnostics/counts. For large single-process chunks (roughly 500k nonces or
more), `GPU_FAN_BITS=20` can be tested as a small extra exact scan win; on 200k chunks it was
a wash after the 208 MiB table build.
## Known issues (found by a large-range A/B; small-range tests missed them)

A 6M-nonce A/B (baseline vs `batch_inv+comb22+single_pass+fan22`) surfaced two things that
the sparse `test-gpu-knobs` ranges (0–8192 + the island window) never hit:

1. **The batch+large-comb combo corrupts its output over a very large *single* kernel
   launch (≳2–6M nonces).** It emits false-positive candidates whose verdict depends on the
   total scan size (e.g. nonce `5000046719` appears clean at 6M but dirty at 200k/1M/2M and
   in a 1-nonce scan). Baseline-6M and the individual knobs over ≤200k are exact, so the
   production path (which chunks into 200k) is safe — but a single multi-million launch is
   not. Ground truth: every observed false-positive is eval-dirty, so no *island was missed*
   in this run, but corrupted output could drop a real island too. **Mitigation:** keep
   `CHUNK` bounded (≤~200k–1M); do not run the combo as one huge launch until this is
   root-caused (`compute-sanitizer`). Suspected cause: the batch-inversion kernel + the 3 GiB
   comb22 / 832 MiB fan tables over a long grid-stride scan.

2. **`single_pass` is NOT candidate-set-identical to `full_first`** (correcting the earlier
   "exact identical candidate set" claim). `single_pass` checks *truncated*-GCD convergence —
   what the circuit actually runs — whereas `full_first` checks *untruncated* convergence. So
   `single_pass` correctly rejects GCD false-positives that `full_first` accepts (verified:
   `5000644403` is `full_first`-clean but eval-dirty `cls=1`, and `single_pass` rejects it).
   It is a *different, arguably more circuit-faithful* necessary filter (a true island is
   truncated-GCD-clean on all shots, so `single_pass` still accepts true islands), not a
   drop-in exact replacement for `full_first`. The "identical candidate set" check passed only
   because the test ranges contained no divergent nonce.

Lesson: validate exactness on a **candidate-dense** range (or against the eval), not just a
range that happens to contain 0–1 candidates.

## Methodology

- Scan: `bench-gpu-knobs` (warmups + measured repeats, raw `gpu_island2` `nonce/s` timer)
  over a fixed dirty range; speedup is an on-vs-off ratio within one run on one base.
- Exactness: `test-gpu-knobs` confirms each knob finds the known island and produces a
  bit-identical candidate set vs the `full_first` baseline before any speed claim.
- Eval: wall-clock of `eval_circuit` on a dirty candidate (`EVAL_FAST_REJECT` 0 vs 1), with
  the clean island re-checked to confirm `0/0/0` is preserved.
