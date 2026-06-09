
## UPDATE: shot-parallel kernel (gpu_island2.cu) — 7.3x throughput, 77x latency
The original kernel (1 thread/nonce) uses 168 registers → ~20% occupancy, and at high
active_iterations each dirty nonce runs ~670 shots before bailing → only 236 nonce/s,
with clean nonces gating 74s each (single-thread, all 9024 shots).
Fix: **block-per-nonce shot-parallelism** — 1 CUDA block per nonce, WAVE=128 threads
split the 9024 shots in waves; thread 0 advances the SHAKE squeeze into shared memory
per wave; each thread runs shot_is_hard() on its shot; a shared hard_flag gives
block-wide early-exit. Launch KERNEL2=1 BLOCKS=512 ./gpu_island2.
Result (validated bit-exact: 431581 → CLEAN, agrees with serial):
- clean-nonce latency 74.7s → **0.97s (77x)**
- search throughput (a257 cfg) 236 → **1713 nonce/s (7.3x)** — the win is block-
  synchronous early-exit eliminating warp divergence (occupancy ~unchanged, registers
  still 162 — the EC+filter is the register hog, not the Squeezer).
2 GPUs ≈ 3400 nonce/s → a ~1/1M island ≈ 5 min (was 12+ hr).
Runtime knobs now keep this path as the baseline while letting experiments A/B against it:
`GPU_WAVE` changes the block size, `GPU_BATCH_INV=1` launches a cooperative batch-inversion
kernel, `GPU_COMB_BITS=16` builds a larger runtime comb table, and `GPU_GCD_MODE` controls
the GCD check order.

## UPDATE: dx-first quick filter
Point addition gives the dialog-GCD two factors:

1. `dx = tx - ox`
2. `c = ox - rx`, where `rx` requires the affine-add slope and a denominator inversion.

The kernel now checks `dx` immediately after deriving the two affine x-coordinates. If
`dx` fails the same `check_gcd_factor` predicate, the shot is hard regardless of `c`, so
the kernel rejects it before computing `rx`. This is an exact structural prefilter: no
clean nonce can be lost, but dirty shots whose first factor is already hard skip the most
expensive part of the second-factor construction.

## UPDATE: experimental runtime knobs

Default settings preserve the existing production behavior:

```text
GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128
```

- `GPU_BATCH_INV=1` runs `search_kernel2_batch`, where every block batch-inverts the two
  Jacobian `Z` values and the affine-add denominator across the current wave. This is exact
  and shares one Fermat inversion per batch instead of one per lane.
- `GPU_COMB_BITS=16` builds a 16x65536 comb table from the dumped 32x256 table at process
  startup. The table is 64 MiB and halves the scalar-mul table-add loop from 32 windows to
  16 windows. It is exact, but repeated small chunks pay repeated build cost.
- `GPU_GCD_MODE=trunc_first` is exact and checks the truncated width envelope before the
  full convergence counter. It should help when width overflows dominate failures.
  `GPU_GCD_MODE=trunc_only` skips the convergence counter and is intentionally noisy:
  it can emit extra false positives, so it is for candidate-generation experiments only.
- `GPU_WAVE` accepts 32..256 and is rounded up to a warp multiple. Batch mode uses more
  shared memory per block, so large waves can trade fewer waves for lower occupancy.

## Lever value (exact CCX counts on b55ede3 base, peak 1309, tof 1,503,871):
active257 −2989 (3.9M score win), active256 −5978 (8.1M), apply20 −516 (675k),
compare48 −144 (188k). ACTIVE_ITERATIONS is the dominant lever; competition sits at 258.
Risk: reducing active_iters couples to the apply phase (phase-garbage) — TBD if a
GCD-clean nonce evals fully clean.
