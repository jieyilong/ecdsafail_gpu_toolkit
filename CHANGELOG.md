# Changelog

This file records not only what changed, but also why we made the change, what speedup
we expect, and what still needs to be validated. For future work, add an entry whenever a
change affects search behavior, performance assumptions, correctness risk, or workflow.

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
