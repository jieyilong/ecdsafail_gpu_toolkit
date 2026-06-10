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

Same RTX 5090, same 1221-qubit SOTA dumped state (`155ebc5`, local commit `572bba4`),
131,072-nonce dirty range, two measured runs:

| binary / knob set | measured throughput |
|---|---:|
| previous-release binary | ~10,174 nonce/s |
| this branch with previous-release-compatible knobs | ~10,053 nonce/s |
| safer stack: `batch_inv+comb22+trunc_first+fan22` | ~12,267 nonce/s |
| requested `batch_inv+comb22+single_pass+fan22` | ~12,576 nonce/s, **invalid** (missed known clean nonce) |

So the previous release baseline on this machine is about 10k nonce/s. The earlier 7k-ish
number corresponds to slower wave settings such as `GPU_WAVE=64`, not to the previous
release.

## Scan kernel (safe knobs — candidate set preserved)

| knob | current SOTA base *(fast-reject)* | earlier base *(slow-reject)* | notes |
|---|---:|---:|---|
| **baseline** | 1.00× (10,209 n/s) | 1.00× (8,400 n/s) | default exact path |
| `GPU_GCD_MODE=trunc_first` | known-clean pass | +1–2% when width rejects dominate | safe reorder: truncated width check, then full convergence check |
| `GPU_GCD_MODE=single_pass` | **unsafe on 1221-SOTA** | +2–4% (historical stack) | fused truncated convergence; missed known clean nonce `165002130437` |
| `GPU_BATCH_INV=1` | **1.01×** (+1.2%) | **1.42×** | the only large lever; only on slow-reject |
| `GPU_COMB_BITS=16` | 1.11× | 1.10× | ~64 MiB table |
| `GPU_COMB_BITS=22` | +3% over comb16 | +3% over comb16 | ~3.0 GiB table; diminishing returns |
| `GPU_FAN_BITS` (nonce-fan) | **1.015×** (+1.5%) | not measured | `squeeze_init` is not the bottleneck here |
| native `sm_120` build (CUDA 12.8) | **1.00×** (no gain) | — | PTX-JIT matches/beats native codegen |
| `GPU_WAVE=64` | 0.78× *(slower)* | 0.84× *(slower)* | smaller waves = more overhead |
| **best safer combo** | **~1.2×** (`batch_comb22` + `trunc_first` + `fan22`) | **1.65×** (`batch_comb22`) | revalidate per base |

Combination caveat: `GPU_WAVE=256` *hurts* in batch mode (lower occupancy), so the best
combo uses `GPU_WAVE=128` — e.g. `all_exact` (which forces wave256) measured slower than
`batch_comb16`.

## Per-candidate validation cost: it's all eval, not build

A common misconception is that `build_circuit` is the slow part. Profiled on the RTX-5090
SOTA base:

| step | time | breakdown |
|---|---:|---|
| `build_circuit` | **~1.2 s** | `point_add::build()` 0.43 s + serialize/write 550 MB 0.76 s |
| `eval_circuit` (stock, clean) | **~16.9 s** | full 9024-shot simulation |
| `eval_circuit` (stock, **dirty**) | **~16.9 s** | ⚠️ stock eval does **not** fail-fast — it simulates all shots and *counts* mismatches |

So per-candidate time is dominated by `eval_circuit` (~17 s), and the stock eval is ~17 s even
for *dirty* candidates. `build_circuit` (~1.2 s) is not worth optimizing; trimming the 550 MB
disk round-trip (in-memory pipe) would save <1 s.

## Eval phase (`EVAL_FAST_REJECT`, exact) — the one big exact win

`EVAL_FAST_REJECT=1` defers the per-shot EC scalar-mults into the batch loop and stops at the
**first failing shot**. It is exact: a clean island still simulates all 9024 shots and reads
`0/0/0` (re-verified on the current base), and with the var unset the path is byte-identical
(`ecdsafail run` still scores 1766121990).

| candidate type | stock eval | `EVAL_FAST_REJECT=1` | speedup |
|---|---:|---:|---:|
| early-failing dirty | ~17 s | ~1.9 s | ~8.5× |
| **GCD-clean but eval-dirty** (what a GPU hunt feeds the validator) | ~17 s | **~6 s** | **~2.6×** |
| clean island | ~17 s | ~17 s (must check all shots) | 1× |

The realized speedup is **candidate-dependent** — it exits at the *first* bad shot, so it
helps most when failures are early. GCD-clean candidates already passed the GCD filter, so
they fail *later* (in the apply/phase tail), landing around ~6 s rather than the ~1.9 s of
arbitrary dirty nonces. Still a real win on the dominant cost: per-candidate validation drops
from ~18 s to ~7 s.

This is the **exact realization of the "apply pre-scan"**: the full eval already checks
apply-cleanliness, so a fast-rejecting eval *is* the apply pre-scan — with **zero false
negatives** and no GPU re-implementation of the apply phase.

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
  early-exits cheaply, so removing it barely helps; it is also unsafe on the 1221-qubit SOTA
  because it missed the baked clean nonce.
- **native `sm_120`** (predicted ~1.2×): the 581-series driver's PTX-JIT from `compute_80`
  matches/beats nvcc-12.8 native Blackwell codegen — measured *slightly slower* native.
- **nonce-fan** (predicted ~1.4×): `squeeze_init` is only ~3% of per-nonce time on the
  current base, not the bottleneck; halving it bought +1.5%.

The lesson: the scan kernel is near its wall on the current SOTA base, and predicting
multipliers from crude profiles overshot repeatedly. The genuine remaining headroom is a
**speculative apply pre-scan** between the GCD scan and the full eval (the apply-bound
regime), not further scan-kernel micro-optimization.

## Recommended settings

For long / billion-scale safer GPU search on a large NVIDIA GPU (the default `CHUNK` is now
500k, so the fan table amortizes — see "Per-process startup cost & chunk sizing"):

```bash
GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=trunc_first GPU_WAVE=128 GPU_FAN_BITS=22  # CHUNK≈1000000
```

On the 1221-qubit SOTA, this found the baked clean nonce and measured ~12,267 n/s on the
RTX 5090, about 1.2x the previous-release baseline. `fan22`'s ~872 MiB table builds in ~0.3s,
a clear win at >=500k chunks. Drop to `GPU_FAN_BITS=0` only for small chunks (≪200k) where
its build is not amortized, or `GPU_GCD_MODE=full_first` for the strictest pre-filter. Add
`EVAL_FAST_REJECT=1` for candidate validation unless you intentionally need byte-for-byte
previous-release diagnostics/counts.

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

## `single_pass` false-negative on the 1221-qubit SOTA

`single_pass` checks *truncated*-GCD convergence (what the circuit's GCD actually runs);
`full_first` checks *untruncated* convergence. Earlier ranges only showed `single_pass`
passing extra eval-dirty false positives. On the 2026-06-10 1221-qubit SOTA, it also missed a
real clean island:

| filter / stack | known-clean range `[165002130430, 165002130446)` |
|---|---|
| previous-release default | found `165002130437` |
| latest main baseline (`full_first`) | found `165002130437` |
| `batch_inv+comb22+trunc_first+fan22` | found `165002130437` |
| `batch_inv+comb22+single_pass+fan22` | **missed** `165002130437` |

CPU validation confirmed the missed nonce is clean:

```text
CLEAN nonce=165002130437 tof=1428172.241 qubits=1221 score=1743798012
```

So `single_pass` is **not** a safe necessary filter on all current bases. The failure is the
fused truncated-convergence check: `trunc_first` still runs the full convergence check after
the truncated width-envelope check, and that safer order found the clean nonce.

Historical note: on an earlier offset `[0, 1M)` of the fan base, `single_pass` looked merely
looser:

| filter | candidates found | note |
|---|---|---|
| `full_first` (comb8) | `{644403}` | untruncated convergence |
| `single_pass` (comb8) | `{46719, 644403}` | truncated convergence — a **superset** here |

Both `46719` and `644403` are eval-dirty (`cls=1`); `46719` is `single_pass`-specific
(truncated converges, untruncated does not). That range was not enough to prove safety. Use
`trunc_first` for production scans; treat `single_pass` as an experiment that must pass a
known-clean nonce check and candidate-dense A/B on the exact base.

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
| `comb22` +batch +`single_pass` +`fan22` | 1.10s | **13,676 n/s** | ~3.0 GiB + ~872 MiB; historical, now unsafe on 1221-SOTA |
| `comb22` +batch +`trunc_first` +`fan22` | ~1.1s | **12,267 n/s** | ~3.0 GiB + ~872 MiB; safer 1221-SOTA recipe |

**The 3 GiB comb22 table builds in only ~0.33s, and fan22 adds ~0.30s — startup is ~1s even
for the heaviest config.** (An earlier worry that the 3 GiB rebuild would be a heavy
per-chunk tax was wrong: the GPU builds it in a fraction of a second.)

Startup as a fraction of a chunk (heaviest config, ~1.0s fixed; use the measured scan rate
for the recipe you choose):

| CHUNK | scan time | startup overhead |
|---|---:|---:|
| 200k | ~16s at the safer 12.3k n/s recipe | ~6% |
| 500k | ~41s at the safer 12.3k n/s recipe | ~2.4% |
| 1M | ~81s at the safer 12.3k n/s recipe | ~1.2% |

The batch+comb+fan tables are deterministic and scale-invariant; the reason to bound CHUNK is
startup amortization plus output-buffer headroom, not a corruption ceiling. **Recommendation
for long/billion-scale runs: `GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=trunc_first
GPU_FAN_BITS=22` with `CHUNK≈1000000`**. Use `full_first` for the strictest pre-filter, and
reserve `single_pass` for explicitly unsafe experiments.

Effective throughput and wall-clock at ~12,300 n/s (after startup, safer recipe):

| nonces | 1 GPU | 8 GPUs |
|---|---:|---:|
| 1 billion | ~22.6 GPU-h | ~2.8 h |
| 10 billion | ~226 GPU-h | ~28 h |

(The comb8 baseline around 10k n/s takes ~27.8 GPU-h per billion — the safer combo saves
roughly 19%.)

## Methodology

- Scan: `bench-gpu-knobs` (warmups + measured repeats, raw `gpu_island2` `nonce/s` timer)
  over a fixed dirty range; speedup is an on-vs-off ratio within one run on one base.
- Exactness: `test-gpu-knobs` confirms each knob finds the known island and produces a
  bit-identical candidate set vs the `full_first` baseline before any speed claim.
- Eval: wall-clock of `eval_circuit` on a dirty candidate (`EVAL_FAST_REJECT` 0 vs 1), with
  the clean island re-checked to confirm `0/0/0` is preserved.
