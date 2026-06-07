
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
Next lever to add if needed: shared-memory block-level batch inversion (removes the
768-modmul/shot Fermat inverse → ~2x more), now feasible since the wave already has
128 shots' points co-resident.

## Lever value (exact CCX counts on b55ede3 base, peak 1309, tof 1,503,871):
active257 −2989 (3.9M score win), active256 −5978 (8.1M), apply20 −516 (675k),
compare48 −144 (188k). ACTIVE_ITERATIONS is the dominant lever; competition sits at 258.
Risk: reducing active_iters couples to the apply phase (phase-garbage) — TBD if a
GCD-clean nonce evals fully clean.
