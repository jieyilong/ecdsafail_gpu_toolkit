# Changelog

This file records not only what changed, but also why we made the change, what speedup
we expect, and what still needs to be validated. For future work, add an entry whenever a
change affects search behavior, performance assumptions, correctness risk, or workflow.

## 2026-06-10 - Retract the "combo corruption" bug; measure startup & chunk sizing

Branch: `speculative-and-fan`

### Summary

Root-caused the alleged "batch+large-comb combo corrupts its output over a very large single
launch" issue. **It was a misdiagnosis — there is no corruption and no count-dependence.**
Also measured the real per-process startup cost (it is ~1s even for the 3 GiB comb22 + fan22,
not the heavy per-chunk tax previously assumed), and set a billion-scale chunk recommendation.

### What we verified (RTX 5090)

- **Deterministic:** the 6M combo run twice produced the identical 7 candidates both times
  (and matched the original). Refutes the "non-deterministic race" hypothesis.
- **Scale-invariant:** the *identical* combo finds offset 46719 at N=200k, 1M, and 6M; offset
  644403 appears once N≥645k. Smaller candidate sets are clean subsets of larger ones.
- **Mechanism:** grid-stride scan (`nidx += gridDim.x`), read-only comb/fan tables during the
  scan, and a bounds-guarded output write (`if(pos<max_out)`) mean no mutable global state is
  read per-nonce — a verdict *cannot* depend on launch size. The old A/B's "extra" candidates
  were `single_pass` false-positives vs a `full_first` baseline at a different size (a config
  mismatch). `compute-sanitizer` 11.5 and 12.8 both refuse to run on this 5090+581 driver
  ("device not supported"), but no race exists to find.
- **`single_pass` direction corrected:** it is *looser* than `full_first`, not stricter.
  Measured on `[0,1M)`: `single_pass`={46719,644403}, `full_first`={644403}. Both are
  necessary filters (neither misses a true island); `single_pass` passes a few more eval-dirty
  false-positives for ~3% faster scan. (Earlier note claimed `single_pass` *rejects* 644403 —
  backwards; both accept it.)

### Measured startup (N=2000 wall, 3 reps, <0.03s spread)

`comb8` 0.47s | `comb16`+batch 0.44s | `comb22`+batch 0.80s | `comb22`+batch+sp+fan22 1.10s.
The 3 GiB comb22 build is ~0.33s; fan22 (~872 MiB) adds ~0.30s. Scan rate climbs
9,655 → 13,676 n/s (1.42×) across that range.

### Main Changes

- Replaced the wrong "Known issues #1 (corruption)" in `docs/measured-speedups.md` with a
  "Resolved: exact and scale-invariant" section + a new "Per-process startup cost & chunk
  sizing" section (startup table, amortization table, billion-scale throughput table).
- Corrected the `single_pass` vs `full_first` direction in `docs/measured-speedups.md`,
  `README.md`, and `SKILL.md` (looser, superset, not stricter).
- Updated the recommended exact scanner to include `GPU_FAN_BITS=22` (fastest measured) for
  the now-default larger chunks.
- **Bumped default `CHUNK` 200000 → 500000** in `island.sh` (search + hunt) and
  `runtime/search_driver.sh` — the 200k cap only existed for the now-retracted corruption
  fear; 500k cuts startup overhead from ~6% to ~2.7% with finer output granularity than 1M.
  Recommend `CHUNK≈1000000` for long/billion-scale runs (~1.3% overhead).

## 2026-06-10 - Document previous-release-compatible baseline knobs

Branch: `speculative-and-fan`

### Summary

Clarified how to run the latest branch as a strict previous-release search baseline and how
to switch from that baseline to the recommended exact-performance stack.

### Main Changes

- Added explicit previous-release-compatible search recipe:
  `GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128 GPU_FAN_BITS=0`,
  plus `unset BATCH_INV GPU_LARGE_COMB GCD_MODE WAVE` for clean comparisons.
- Documented that validation parity with the previous release requires `EVAL_FAST_REJECT=0`;
  the branch's `island.sh validate` defaults it to `1` for faster dirty-candidate rejection.
- Recorded same-machine RTX 5090 comparison: the previous-release binary measured
  ~10,057 nonce/s, and this branch with baseline knobs measured ~10,062 nonce/s on the same
  dumped state. This confirms the previous release baseline is about 10k nonce/s on this
  machine, while the 7k-ish number corresponds to slower wave settings such as `GPU_WAVE=64`.
- Added recommended exact scan settings:
  `GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_WAVE=128 GPU_FAN_BITS=0`,
  with `GPU_FAN_BITS=20` only as a large-chunk option.

### Rationale

The branch now contains several exact on/off knobs, so "baseline" was ambiguous. These notes
separate three cases: the previous release, this branch with previous-release-compatible
knobs, and the recommended fast exact stack. That makes future speedup claims auditable and
avoids comparing against the wrong baseline.
## 2026-06-10 - Large-range A/B: correctness findings + corrections

Branch: `speculative-and-fan`

> **⚠ PARTIALLY RETRACTED (see the "Retract the combo corruption bug" entry above).** The
> "combo corrupts output over a very large single launch" claim below was a **misdiagnosis** —
> the combo is exact and scale-invariant (verified: identical candidates at 200k/1M/6M, a 6M
> run reproduces bit-for-bit, no mutable global state is read per-nonce). The `single_pass`
> direction below is also **backwards**: it is *looser* than `full_first` (a superset), not
> stricter — both accept `5000644403`; `single_pass` additionally accepts `46719`. The 1.40x
> throughput number and the chunking change are still valid. Kept for history.

A 6M-nonce A/B (baseline vs `batch_inv+comb22+single_pass+fan22`) measured **1.40x** scan
throughput (9,794 -> 13,719 n/s) but also exposed two correctness issues the sparse
`test-gpu-knobs` ranges missed. See `docs/measured-speedups.md` -> "Known issues".

- **Combo corrupts output over a very large single kernel launch (>~2-6M).** It emits
  false-positive candidates whose verdict depends on scan size (non-deterministic). Baseline
  and individual knobs over <=200k are exact; the production path chunks into 200k so it is
  safe. Ground truth: observed false-positives are all eval-dirty (no island missed in this
  run). Suspected: batch-inversion kernel + 3 GiB comb22 / 832 MiB fan tables over a long
  grid-stride scan. Needs `compute-sanitizer`.
- **Reverted the earlier `hunt` "whole-range chunk" change** (it removed the protective
  chunking and would trigger the above). `hunt` CHUNK defaults back to 200k, override allowed.
- **Corrected the `single_pass` docs:** it is NOT candidate-set-identical to `full_first`.
  It checks *truncated* GCD convergence (the circuit's actual behavior), so it rejects GCD
  false-positives `full_first` accepts (e.g. `5000644403`: `full_first`-clean but eval-dirty).
  It remains a valid necessary filter (won't miss true islands) and is arguably more
  circuit-faithful. The old "validated identical candidate set" claim relied on a test range
  with no divergent nonce.
- The 1.40x figure is a valid *throughput* number but came from a non-exact combo run; do not
  use the combo as one huge launch until the corruption is root-caused.

## 2026-06-10 - Nonce-fan, eval early-exit, and a measured-speedups record

Branch: `speculative-and-fan` (not yet merged)

### Summary

Two new exact, independently-toggleable knobs plus an honest measured-speedups doc.

### Main Changes

- `GPU_FAN_BITS=K` (**nonce-fan**, kernel): precompute the SHAKE sponge for the low `K` tail
  bits (`2^K * 208 B` table) so each nonce only absorbs its high bits. Exact (validated:
  finds the island, candidate set unchanged). Wired through `island.sh` + `search_driver.sh`.
- `EVAL_FAST_REJECT=1` (**eval early-exit**, challenge `eval_circuit`): stop the validation
  eval at the first failing batch. Default off so scoring runs stay complete; `island.sh
  validate` sets it to `1`. Saved as `patches/eval_fast_reject.diff` (the challenge repo is
  reset by `ecdsafail sync`).
- Added `docs/measured-speedups.md` with the full measured table, and updated `README.md`,
  `SKILL.md`, `docs/theory-knobs.md`, `docs/kernel-notes.md`, `docs/how-it-works.md`.

### Measured Impact (RTX 5090, current SOTA base)

- nonce-fan: **~+1.5%** (squeeze_init is not the bottleneck here; predicted ~1.4x, did not
  pan out).
- eval early-exit: now **lazy** (defers the per-shot EC-muls into the batch loop +
  early-exit) -> **~8.5x average** on dirty candidates (16.1s -> ~1.9s), exact. This is the
  exact realization of the "apply pre-scan" (#3/4): the full eval already checks
  apply-cleanliness, so a fast-rejecting eval *is* the apply pre-scan, with zero false
  negatives and no GPU re-implementation of the apply phase. `FAST=0` scoring path is
  byte-identical to the original.
- Every improvement is now an independent on/off knob: `GPU_BATCH_INV`, `GPU_COMB_BITS`,
  `GPU_GCD_MODE`, `GPU_WAVE`, `GPU_FAN_BITS`, `EVAL_FAST_REJECT`.
- Overall end-to-end: scan and eval are sequential stages, so the scan (<=1.65x) and eval
  (~8.5x) speedups do NOT multiply. Combined is **up to ~8.5x** where candidate validation
  dominates (apply-bound) and **~1.6x** where the GPU scan dominates (current frontier base);
  never ~14x. Documented in the new "Overall pipeline speedup" section of
  `docs/measured-speedups.md`.

### Validation Status

- Kernel compiles (PTX compute_80 JIT to sm_120); `bash -n` passes for `island.sh` and
  `search_driver.sh`. nonce-fan verified exact on the live SOTA base; eval early-exit
  verified to preserve the island's `0/0/0`. Not committed/pushed.

## 2026-06-10 - Single-pass exact GCD + measured limits of further exact speedups

Branch: `exact-speedups` (not yet merged)

### Summary

Added `GPU_GCD_MODE=single_pass`, an exact one-walk GCD filter, and — more importantly —
ran a careful A/B pass on the RTX 5090 that establishes the *realistic* ceiling for further
exact-mode speedups. The headline result is sobering: beyond the existing `batch_inv`, the
remaining exact levers are single-digit-percent, and several proposed ones give ~0.

### Main Changes

- New exact GCD mode `GPU_GCD_MODE=single_pass` (kernel + `parse_gcd_mode` + harness). It
  folds the two baseline GCD passes (untruncated convergence walk + truncated overflow walk)
  into one truncated walk that also tests `v==0` convergence. Exact by the same
  within-envelope argument the two-pass filter already relies on; see `docs/theory-knobs.md`.
- Wired `single_pass` and `batch_comb16_single` into `test-gpu-knobs` and `bench-gpu-knobs`.

### Correctness (validated on RTX 5090, SOTA base f6f9536, island 1000005782829)

`single_pass` and `batch_comb16_single` both find the known island AND produce the
**bit-identical candidate set** as the `full_first` baseline over the candidate-bearing
range. `PASS`.

### Speedup (measured, the honest numbers)

The gains are **strongly base-dependent**, which is the main finding. On the current SOTA
base, `[0, 32768)`:

```text
variant              avg_nonce_s   speedup
baseline                  10209    1.000x
single_pass               10228    1.002x   (~0 on its own)
batch_inv                 10333    1.012x   (only +1.2% here!)
batch_comb16              11821    1.158x
batch_comb16_single       12065    1.182x   (best exact)
```

On the base benchmarked the day before, the SAME `batch_inv` gave ~`1.5x` and
`batch_comb16` ~`1.6x`. The difference is reject speed: when nonces reject after only a few
shots (as on this base), the per-NONCE `squeeze_init` (~35 Keccak-f permutations, paid once
regardless of early-exit) dominates, and the per-SHOT levers (`batch_inv`, `comb`,
`single_pass`) barely run. When nonces reject slowly, the per-shot inversions dominate and
`batch_inv` shines. So the real bottleneck shifts between SHAKE-init and field-arithmetic
depending on the config.

### Tricks evaluated and NOT implemented (with reasons)

- **Native `sm_120` build (CUDA 12.8):** built and measured — **no benefit**. On this card's
  driver (581.x) the PTX-JIT from `compute_80` matches or beats nvcc-12.8 native codegen
  (baseline native 9185 vs PTX-JIT 10515 n/s). Not worth carrying a second toolchain.
- **Montgomery multiplication, fused single inversion, lazy reduction:** these speed up
  per-shot field arithmetic, which is NOT the bottleneck in the fast-reject (SHAKE-bound)
  regime, and where it IS the bottleneck `batch_inv` already captured most of it. Predicted
  ~1.05-1.1x for a large, correctness-risky refactor — declined on ROI grounds after
  `single_pass` and native both underdelivered their estimates.
- **Faster/incremental SHAKE squeeze_init:** this is the *right* lever for fast-reject bases
  (squeeze_init dominates there), but an exact incremental update across sequential nonces is
  blocked by the tail feed order (the nonce's low bits are fed first, so an increment changes
  the first absorbed block and forces a full re-permute). Left as an open problem; any future
  attempt must preserve the exact op-stream byte order or it changes the derived inputs.

### Expected Impact

`single_pass` is exact and never hurts, so it is safe to enable, but it is a ~0-4% lever, not
a multiplier. The practical recommendation is unchanged: `batch_inv` + `comb16` remains the
default-good exact mode; add `single_pass` for a free few-percent; expect the absolute win to
vary with the base's reject dynamics.

### Validation Status

- `bash -n island.sh` passed; kernel compiles clean (PTX compute_80 JIT + native sm_120).
- `test-gpu-knobs` PASS (candidate-set equality incl single_pass) on f6f9536.
- `bench-gpu-knobs` table above, RTX 5090, 2 measured repeats.
- NOT committed/pushed (per request).

## 2026-06-10 - Large comb VRAM tradeoff and shell self-call fix

Branch: `quick_filtering`

### Summary

Added opt-in `GPU_COMB_BITS=20` and `GPU_COMB_BITS=22` modes to test whether spare VRAM on
large GPUs can buy down scalar-multiplication compute, and fixed `island.sh` recursive
self-calls so the script works when invoked as either `./island.sh ...` or
`bash island.sh ...`.

### Rationale

`nvidia-smi` showed low VRAM capacity usage, so the natural experiment was to spend memory
on larger fixed-base tables. The existing `comb16` table already showed that this path is
exact and composable with `GPU_BATCH_INV=1`; `comb20` and `comb22` test the next two
reasonable points before memory traffic and table size become likely bottlenecks.

The self-call fix came from review: several commands invoke `island.sh` recursively. Using
`"$0"` breaks when the user runs `bash island.sh ...` because `$0` may not be an executable
path on `PATH`. Resolving `SELF="$HERE/island.sh"` and invoking `bash "$SELF"` makes both
entry styles reliable.

### Main Changes

- Replaced the hard-coded `comb16` device table path with a generic runtime-built large
  comb table for explicit `GPU_COMB_BITS=16`, `20`, or `22`.
- Added arbitrary-window scalar digit extraction and a generic table-builder kernel that
  composes entries from the dumped 8-bit comb table.
- Added opt-in large-comb coverage to the correctness and benchmark harnesses:
  `GPU_TEST_COMB_BITS="20 22"` and `GPU_BENCH_COMB_BITS="20 22"`.
- Printed large-comb table build time in benchmark runs so kernel throughput is not
  confused with wall-clock startup cost.
- Fixed `island.sh` self-invocation by routing recursive calls through `bash "$SELF"`.
- Updated `README.md`, `docs/kernel-notes.md`, `docs/how-it-works.md`, and
  `docs/theory-knobs.md` with the new knob values, memory tradeoffs, and measured result.

### Expected Impact

No circuit-score change. The search-speed effect is real but small: larger comb tables
reduce scalar-mul mixed additions, but after batch inversion the kernel is no longer
dominated solely by comb work. On the RTX 5090, `batch_comb22` is the fastest exact mode
measured so far, but only by a few percent over `batch_comb16`.

### Validation Status

- `bash -n island.sh` passed.
- `git diff --check` passed.
- Remote CUDA build passed on the RTX 5090 machine, still using CUDA 11.5 PTX fallback to
  `sm_120`.
- Correctness passed on current base `f6f9536`, baked nonce `1000005782829`:
  - `GPU_TEST_COMB_BITS="20"` over dirty range `[0, 256)`;
  - `GPU_TEST_COMB_BITS="22"` over dirty range `[0, 64)`;
  - `GPU_TEST_COMB_BITS="20 22"` over candidate-bearing range
    `[1000005782824, 1000005782840)`, where baseline candidates = 1.
- Speed benchmark on the RTX 5090 over `[0, 32768)` with two measured repeats:

```text
variant        avg_nonce_s   speedup_vs_baseline
baseline              8508    1.000x
batch_comb16         13245    1.557x
batch_comb20         13498    1.587x
batch_comb22         13622    1.601x
```

`batch_comb22` was about `2.8%` faster than `batch_comb16` on this longer run. Table build
time was about `0.10s` for `comb20` and `0.37s` for `comb22` on the RTX 5090.

## 2026-06-09 - GPU knob throughput benchmark

Branch: `quick_filtering`

### Summary

Added `./island.sh bench-gpu-knobs` so speedup claims for the experimental CUDA knobs can
be measured from the same SOTA state, same nonce range, same warmup policy, and same raw
kernel timer.

### Rationale

Correctness checks tell us whether a knob is safe, but they do not tell us whether it is
worth using in the island search loop. The previous manual approach mixed kernel time,
remote chunk orchestration, first-run PTX JIT, and operator memory. A dedicated benchmark
command makes the A/B test repeatable and produces a single table that can be pasted into
future research notes.

### Main Changes

- Added `run_raw_search`, which invokes `gpu_island2` directly with `KERNEL2=1` while
  preserving the binary's `scanned ... (nonce/s)` timing output.
- Added `bench_variant`, which runs configurable warmups and measured repeats, then records
  average/min/max nonce throughput.
- Added `./island.sh bench-gpu-knobs [CFG] [START] [N]` with default variants for the
  baseline, isolated knobs, and combined exact paths.
- Extended `test-gpu-knobs` to cover the same combined exact paths, so benchmarked
  combinations have an explicit candidate-set guard.
- In remote mode, uploaded the benchmark state once and reused it across variants instead
  of re-copying it before every measured run.
- Documented the speed-validation workflow and caveats in `README.md`.

### Expected Impact

No circuit-score change. The expected workflow speedup is decision speed: we can reject
unhelpful kernel knobs after a small controlled benchmark instead of discovering later
during a long island run. The command should also make architecture-specific tuning more
honest, especially on RTX 50 cards where old CUDA toolkits may JIT PTX instead of building
native `sm_120` code.

### Validation Plan

- Shell syntax checks and `git diff --check` passed locally.
- `test-gpu-knobs` passed on the RTX 5090 remote for the baked SOTA nonce `4591773` and
  range `[0, 1024)`, including combined exact variants.
- `bench-gpu-knobs` ran on the RTX 5090 remote over range `[0, 8192)` with one warmup and
  two measured repeats per variant. The remote used CUDA 11.5 PTX fallback on `sm_120`, so
  absolute rates may improve with a newer CUDA toolkit, but relative numbers are still an
  apples-to-apples comparison:

```text
variant        avg_nonce_s   speedup_vs_baseline
baseline              7550    1.000x
trunc_first           7697    1.019x
wave64                5515    0.730x
wave256               7836    1.038x
batch_inv            11310    1.498x
comb16                8440    1.118x
batch_wave256         9225    1.222x
batch_comb16         12813    1.697x
all_exact            10288    1.363x
```

The best measured exact combination was `GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_WAVE=128`
at about `1.70x` kernel throughput over this slice.

## 2026-06-09 - GPU knob correctness smoke tests

Branch: `quick_filtering`

### Summary

Added an integrated `./island.sh test-gpu-knobs` command and `./island.sh probe` command
so the experimental GPU search paths can be checked against the latest promoted SOTA state
on a real CUDA machine.

### Rationale

The new knobs are meant to preserve the exact candidate set except for the intentionally
noisy `trunc_only` mode. That makes correctness testing straightforward: every exact knob
must find the baked clean `DIALOG_TAIL_NONCE`, and every exact knob must match the baseline
candidate set over a fixed dirty range. The probe check also catches SHAKE-prefix or state
upload mistakes before any candidate-set comparison.

### Main Changes

- `ISLAND_CONFIG=/path/to/config.env` lets tests target a temporary remote GPU config
  without committing machine-specific SSH settings into the tracked `config.env`.
- `./island.sh probe STATE` runs the CUDA binary's built-in first-shot Keccak probe.
- `./island.sh test-gpu-knobs [CFG] [START] [N] [CHUNK]` rebuilds Rust helpers, builds the
  CUDA kernel, dumps state, checks the probe, verifies the known clean nonce under each
  knob, compares exact candidate sets for baseline, `trunc_first`, `wave64`, `wave256`,
  `batch_inv`, and `comb16`, and checks that `trunc_only` does not miss baseline
  candidates.

### Expected Impact

No score or search-speed change. The expected benefit is faster failure localization when
testing on machines like RTX 5090/H100/A100: a broken SHAKE state, comb table, batch
inversion, GCD ordering, or wave-size path should fail a small deterministic test before a
long island search wastes time.

## 2026-06-09 - Theory note for GPU search knobs

Branch: `quick_filtering`

### Summary

Added `docs/theory-knobs.md` to explain the mathematical and performance background behind
the major runtime knobs introduced in the quick-filtering work.

### Rationale

The GPU search knobs are not just code switches; they encode assumptions about field
arithmetic, elliptic-curve scalar multiplication, GCD rejection modes, and CUDA block
scheduling. Capturing those assumptions next to the code makes future review easier and
reduces the chance that we forget why an experimental path exists.

### Main Topics Covered

- why `dx = tx - ox` can be checked before constructing `rx` and `c`;
- Montgomery batch inversion with prefix/suffix products;
- how the batch kernel applies that idea to Jacobian `Z` values and affine-add
  denominators;
- why a 16-bit comb table reduces scalar-multiplication windows but costs GPU memory and
  startup time;
- why `trunc_first` is exact while `trunc_only` is a noisy candidate generator;
- why `GPU_WAVE` trades off fewer waves, occupancy, shared memory, and early-exit latency.

### Expected Impact

No runtime behavior changes. The value is research memory: future code reviews and
benchmark sessions should be able to compare measured results against the original theory
and decide which knobs deserve more engineering effort.

## 2026-06-09 - Quick filtering and experimental GPU search knobs

Commit: `f473416`
Branch: `quick_filtering`

### Summary

Added a runtime-configurable experimental layer on top of the production
`gpu_island2.cu` search kernel while keeping the default path compatible with the existing
toolkit behavior.

Default behavior remains:

```text
GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128
```

### Main Changes

- Added an exact `dx`-first structural quick filter in the CUDA and Rust search paths.
  The point-addition circuit feeds two dialog-GCD factors, `dx = tx - ox` and
  `c = ox - rx`. The first factor is available immediately after affine coordinates are
  known, so the kernel now checks it before computing the affine-add denominator inverse
  and second factor.
- Added `GPU_BATCH_INV=1`, which launches a separate cooperative block kernel. This path
  batch-inverts per-wave Jacobian `Z` products and affine-add denominators using one
  block-wide Fermat inversion plus parallel prefix/suffix products.
- Added `GPU_COMB_BITS=16`, which builds a 16x65536 comb table on the GPU at process
  startup from the existing dumped 32x256 table. This halves the scalar-multiplication
  window loop from 32 table windows to 16.
- Added `GPU_GCD_MODE=full_first|trunc_first|trunc_only`.
  `full_first` preserves the old exact order. `trunc_first` is still exact but checks the
  truncated width envelope before the full convergence counter. `trunc_only` intentionally
  skips the convergence counter and can produce extra false positives.
- Added `GPU_WAVE=32..256` to tune the CUDA block size per nonce wave. The value is rounded
  to a warp multiple and capped at 256.
- Wired the knobs through `island.sh` and `runtime/search_driver.sh` for both local and
  remote GPU searches.
- Documented the knobs in `README.md`, `docs/how-it-works.md`, and
  `docs/kernel-notes.md`.

### Rationale

The existing production kernel is already strong because it uses one CUDA block per nonce
and lets the block stop as soon as any shot is hard. The next search-speed gains are less
likely to come from checking fewer true shots, because dirty nonces already fail after a
few hundred shots on average. Bigger wins should come from reducing repeated per-shot
field work and making the search binary easy to A/B.

The changes therefore keep the known-good baseline as the default and expose independent
switches for the main plausible speed levers:

- batch inversion attacks the expensive field inversions in affine conversion and
  denominator inversion;
- comb16 attacks scalar multiplication table-add cost;
- GCD mode tuning attacks wasted work inside factor checks when width overflow is the
  common failure mode;
- wave tuning lets us balance fewer shot waves against occupancy and shared-memory use.

The `dx`-first filter is exact and low risk, so it is enabled in the normal path. The other
levers are behind explicit runtime options because their performance depends on GPU
architecture, occupancy, island density, and chunk size.

### Expected Impact

These are hypotheses until benchmarked on a CUDA machine:

- `dx`-first quick filter: expected to be small on the production GPU path, probably below
  1%, because first-factor failures are rare relative to all evaluated shots. It is still
  beneficial because it is exact and skips a costly denominator inversion on those shots.
- `GPU_BATCH_INV=1`: expected to be the largest potential win, plausibly 1.5x to 3x if
  Fermat inversions are still a dominant cost. Actual speed depends on shared-memory use,
  occupancy, and the cost of the block-wide scan.
- `GPU_COMB_BITS=16`: expected to help scalar multiplication by reducing table windows
  from 32 to 16, with a rough target of 1.3x to 2x for the scalar-mul portion. It costs
  about 64 MiB of GPU memory and startup time per process, so it is better for larger
  chunks than tiny test chunks.
- `GPU_GCD_MODE=trunc_first`: expected to help only if truncated width failures dominate
  convergence failures. It is exact, so it is safe to test broadly.
- `GPU_GCD_MODE=trunc_only`: expected to be useful only as an aggressive candidate
  generator. It may increase false positives and must always be followed by normal
  validation.
- `GPU_WAVE`: expected to be a tuning knob rather than a guaranteed win. Larger waves
  reduce the number of SHAKE/shot waves per nonce but can reduce occupancy, especially in
  batch mode.

### Validation Status

- Local shell syntax checks passed for `island.sh`, `runtime/search_driver.sh`,
  `runtime/build_kernel.sh`, and `runtime/doctor.sh`.
- `git diff --check` passed.
- CUDA compile and performance benchmarking were not run locally because this laptop does
  not have `nvcc`.

### Follow-Up Benchmark Plan

Run each setting against a known-clean nonce and a representative dirty range before
combining knobs:

```bash
./island.sh build
./island.sh dump "" /tmp/base.bin
./island.sh search /tmp/base.bin <known_nonce> 1

GPU_GCD_MODE=trunc_first ./island.sh search /tmp/base.bin <known_nonce> 1
GPU_BATCH_INV=1 ./island.sh search /tmp/base.bin <known_nonce> 1
GPU_COMB_BITS=16 ./island.sh search /tmp/base.bin <known_nonce> 1
GPU_WAVE=64 ./island.sh search /tmp/base.bin <known_nonce> 1
GPU_WAVE=256 ./island.sh search /tmp/base.bin <known_nonce> 1
```

After correctness checks, benchmark fixed-size dirty ranges, for example 100k to 1M
nonces, and compare nonce/s plus candidate counts. Only combine knobs after each one has
passed the known-nonce sanity check.
