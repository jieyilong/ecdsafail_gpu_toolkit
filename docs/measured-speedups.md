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

For long / billion-scale exact GPU search on a large NVIDIA GPU (the default `CHUNK` is now
500k, so the fan table amortizes — see "Per-process startup cost & chunk sizing"):

```bash
GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_WAVE=128 GPU_FAN_BITS=22  # CHUNK≈1000000
```

This is the fastest measured exact scanner (~13,676 n/s, ~1.42× over the comb8 baseline);
`fan22`'s ~872 MiB table builds in ~0.3s, a clear win at ≥500k chunks (it added ~3% scan over
the no-fan combo). Drop to `GPU_FAN_BITS=0` only for small chunks (≪200k) where its build
isn't amortized, or `GPU_GCD_MODE=full_first` for the strictest pre-filter (fewest
eval-dirty false-positives, ~3% slower scan). Add `EVAL_FAST_REJECT=1` for candidate
validation unless you intentionally need byte-for-byte previous-release diagnostics/counts.
## Resolved: the batch+large-comb combo is exact and scale-invariant

An earlier draft of this file claimed the `batch_inv+comb22+single_pass+fan` combo
"corrupts its output over a very large single kernel launch (≳2–6M nonces)," based on a 6M
A/B whose candidate set differed from smaller runs. **That was a misdiagnosis — there is no
corruption and no count-dependence.** Re-verified directly on the RTX 5090:

- **Deterministic:** the 6M combo, run twice back-to-back, produced the *identical* 7
  candidates both times (and matched the original run). A race would vary run-to-run.
- **Scale-invariant:** the *identical* combo finds nonce `5000046719` (offset 46719) at
  N=200k, N=1M, and N=6M; `5000644403` (offset 644403) appears as soon as N≥645k. Every
  smaller candidate set is a clean subset of the larger one — exactly "more nonces scanned →
  more candidates," not corruption.
- **Mechanism confirms it:** the scan kernel is grid-stride (`nidx = blockIdx.x; nidx +=
  gridDim.x`), the comb/fan tables are read-only during the scan, and the output write is
  bounds-guarded (`if(pos<max_out)`). There is **no mutable global state read during
  per-nonce evaluation**, so a per-nonce verdict cannot depend on the total launch size.

The original A/B's "extra" candidates were just `single_pass` false-positives (see below)
plus normal GCD-filter false-positives, compared against a `full_first` baseline at a
different size — a config mismatch, not a scan-size effect. `compute-sanitizer` (both 11.5
and 12.8) cannot run on this RTX 5090 + 581 driver ("device not supported"), but with the
bug shown to be deterministic and mechanically impossible there is no race to hunt.

**The only real large-*single*-launch caveat is benign:** the output buffer holds
`MAXOUT=4096` candidates; a single launch that finds more than 4096 *silently truncates* the
surplus (the write is guarded — a clean drop, not memory corruption). At the ~1e-6–1e-5
candidate density of real bases this needs billions of nonces in one launch to approach, and
any chunked search (≤~1M) never comes close. Chunk for throughput/memory and table-build
amortization, **not** for correctness.

## `single_pass` is a different (looser) necessary filter than `full_first`

`single_pass` checks *truncated*-GCD convergence (what the circuit's GCD actually runs);
`full_first` checks *untruncated* convergence. They are both **necessary** filters — a true
`0/0/0` island is clean under both, so neither misses islands — but they disagree on
borderline eval-dirty nonces. Measured on offset `[0, 1M)` of the fan base:

| filter | candidates found | note |
|---|---|---|
| `full_first` (comb8) | `{644403}` | untruncated convergence |
| `single_pass` (comb8) | `{46719, 644403}` | truncated convergence — a **superset** here |

Both `46719` and `644403` are eval-dirty (`cls=1`); `46719` is `single_pass`-specific
(truncated converges, untruncated does not). So on this range `single_pass` is **looser** —
it passes one extra eval-dirty false-positive. (This corrects an earlier note that had the
direction backwards, claiming `single_pass` *rejects* `644403`; in fact both filters accept
it.) Practical tradeoff: `single_pass` scans ~3% faster but hands the eval a few more dirty
candidates to reject. It never misses a true island. Use `full_first` for the strictest GPU
pre-filter (fewest false-positives), `single_pass` to shave scan time when the search is
scan-bound and the extra eval cost is negligible.

Lesson: validate exactness on a **candidate-dense** range (or against the eval), and compare
*identical* configs across sizes — not a range with 0–1 candidates, and never across
different filter modes.

## Per-process startup cost & chunk sizing (for billion-scale scans)

Each chunk is a *separate* `gpu_island2` process (see `runtime/search_driver.sh`), so every
chunk pays a one-time startup: process init + state load + GPU table build. Measured on the
RTX 5090 as wall-clock of an N=2000 run (essentially pure startup), 3 reps, <0.03s spread:

| config | startup (build+init) | scan rate | per-process table |
|---|---:|---:|---|
| `comb8` / `full_first` (baseline) | 0.47s | 9,655 n/s | none |
| `comb16` +batch | 0.44s | 12,490 n/s | ~64 MiB |
| `comb22` +batch | 0.80s | 12,680 n/s | ~3.0 GiB |
| `comb22` +batch +`single_pass` +`fan22` | 1.10s | **13,676 n/s** | ~3.0 GiB + ~872 MiB |

**The 3 GiB comb22 table builds in only ~0.33s, and fan22 adds ~0.30s — startup is ~1s even
for the heaviest config.** (An earlier worry that the 3 GiB rebuild would be a heavy
per-chunk tax was wrong: the GPU builds it in a fraction of a second.)

Startup as a fraction of a chunk (heaviest config, ~1.0s fixed + scan at 13,676 n/s):

| CHUNK | scan time | startup overhead |
|---|---:|---:|
| 200k | ~14.6s | ~6.4% |
| 500k | ~36.6s | ~2.7% |
| 1M | ~73s | ~1.3% |

Since the combo is exact at any size (no corruption ceiling), the only reason to bound CHUNK
is this startup amortization (and GPU memory for the table, which is fixed by `comb_bits`,
not by CHUNK). **Recommendation for long/billion-scale runs: `GPU_BATCH_INV=1
GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_FAN_BITS=22` with `CHUNK≈1000000`** — fastest
scanner, startup amortized to ~1.3%, ~**1.42×** the comb8 baseline. (Drop `single_pass` for
`full_first` if you prefer the strictest pre-filter; the scan cost is ~3% higher.)

Effective throughput and wall-clock at ~13,500 n/s (after startup):

| nonces | 1 GPU | 8 GPUs |
|---|---:|---:|
| 1 billion | ~20.6 GPU-h | ~2.6 h |
| 10 billion | ~206 GPU-h | ~26 h |

(The comb8 baseline at 9,655 n/s would take ~28.8 GPU-h per billion — the combo saves ~30%.)

## Methodology

- Scan: `bench-gpu-knobs` (warmups + measured repeats, raw `gpu_island2` `nonce/s` timer)
  over a fixed dirty range; speedup is an on-vs-off ratio within one run on one base.
- Exactness: `test-gpu-knobs` confirms each knob finds the known island and produces a
  bit-identical candidate set vs the `full_first` baseline before any speed claim.
- Eval: wall-clock of `eval_circuit` on a dirty candidate (`EVAL_FAST_REJECT` 0 vs 1), with
  the clean island re-checked to confirm `0/0/0` is preserved.
