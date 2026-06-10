# Theory Background for GPU Search Knobs

This note explains the main ideas behind the experimental `gpu_island2.cu` knobs. The goal
is to make the code reviewable months later: what mathematical fact is being used, why it
should help, and where the correctness or performance risk lives.

## Search Context

For each candidate `DIALOG_TAIL_NONCE`, the GPU reproduces the benchmark's 9,024
hash-derived point-add inputs. Each shot roughly does:

1. derive two scalars, `k1` and `k2`, from SHAKE256;
2. compute `t = k1 * G` and `o = k2 * G`;
3. convert both points from Jacobian coordinates to affine coordinates;
4. construct the point-addition inversion factors;
5. run the dialog-GCD filter on those factors.

The knobs below try to reduce the cost of steps 2 to 5 without changing which exact inputs
the benchmark tests.

## `dx`-First Quick Filter

Point addition needs the slope

```text
lambda = (oy - ty) / (ox - tx)
rx = lambda^2 - tx - ox
```

The circuit's dialog-GCD path checks two relevant factors:

```text
dx = tx - ox
c  = ox - rx
```

The important ordering fact is that `dx` is known as soon as `tx` and `ox` are affine.
The second factor `c` requires `rx`, and `rx` requires the affine-add denominator inverse.

So the exact short-circuit is:

```text
if dx is hard:
    reject the shot immediately
else:
    compute lambda, rx, c
    check c
```

This cannot reject a clean nonce incorrectly, because a hard `dx` is already sufficient for
the shot to fail the same GCD predicate. The expected speedup is small on the production
GPU path because first-factor failures are rare among all evaluated shots, but each saved
case skips a relatively expensive denominator inversion and second-factor construction.

## `GPU_BATCH_INV=1`: Batch Inversion

Field inversion is expensive. In this toolkit the CUDA field inverse is Fermat-style:

```text
a^-1 = a^(p-2) mod p
```

That means one inverse is hundreds of modular squarings and multiplications. A normal
modular multiplication is much cheaper, so it is often better to trade many inversions for
one inversion plus extra multiplications.

The identity is often called Montgomery's trick. Given nonzero field elements:

```text
a1, a2, ..., an

P = a1 * a2 * ... * an
P_inv = P^-1
```

Then each inverse can be recovered from the inverse of the product:

```text
a1^-1 = P_inv * a2 * a3 * ... * an
a2^-1 = P_inv * a1 * a3 * ... * an
...
an^-1 = P_inv * a1 * a2 * ... * a(n-1)
```

In practice, prefix and suffix products avoid recomputing those long products:

```text
prefix[i] = a1 * a2 * ... * ai
suffix[i] = ai * a(i+1) * ... * an

ai^-1 = prefix[i-1] * suffix[i+1] * P_inv
```

So `n` inversions become:

```text
1 inversion + O(n) multiplications
```

### How the GPU Kernel Uses It

In a wave of `GPU_WAVE` shots, every lane has two Jacobian points, `t` and `o`, with
projective coordinates:

```text
t = (Xt, Yt, Zt)
o = (Xo, Yo, Zo)
```

Affine conversion needs:

```text
zt_inv = Zt^-1
zo_inv = Zo^-1

tx = Xt * zt_inv^2
ty = Yt * zt_inv^3
ox = Xo * zo_inv^2
oy = Yo * zo_inv^3
```

The batch kernel first combines the two `Z` inversions per lane:

```text
zprod = Zt * Zo
zprod_inv = zprod^-1

Zt^-1 = Zo * zprod_inv
Zo^-1 = Zt * zprod_inv
```

Then it batch-inverts all `zprod` values across the block. Later, after `dx` passes, it
batch-inverts the affine-add denominators:

```text
den = ox - tx
den_inv = den^-1
```

### Why It Might Be Faster

The old path makes each lane run several independent Fermat inversions. The batch path
uses block-wide cooperation so the wave pays one inversion for many lanes, plus prefix and
suffix multiplication scans in shared memory.

The expected gain depends on whether the saved inversions dominate the added costs:

- shared-memory traffic;
- `__syncthreads()` barriers;
- lower occupancy from larger shared-memory use;
- extra modular multiplications for prefix/suffix scans.

That is why this is behind `GPU_BATCH_INV=1` instead of replacing the baseline immediately.

## `GPU_COMB_BITS=16/20/22`: Larger Scalar-Multiplication Tables

Scalar multiplication by the base point is done with a fixed-window comb table. The default
table uses 8-bit digits:

```text
k = d0 + d1 * 2^8 + d2 * 2^16 + ... + d31 * 2^248
k * G = table[0][d0] + table[1][d1] + ... + table[31][d31]
```

That means up to 32 mixed additions per scalar multiply. Since each shot computes both
`k1 * G` and `k2 * G`, scalar multiplication is a large part of per-shot work.

With a 16-bit table:

```text
k = e0 + e1 * 2^16 + e2 * 2^32 + ... + e15 * 2^240
k * G = table16[0][e0] + table16[1][e1] + ... + table16[15][e15]
```

The loop length drops from 32 windows to 16 windows. The tradeoff is table size:

```text
8-bit table:  32 * 256      affine points
16-bit table: 16 * 65536    affine points  (~64 MiB)
20-bit table: 13 * 1048576  affine points  (~832 MiB)
22-bit table: 12 * 4194304  affine points  (~3.0 GiB)
```

Each affine point stores two 256-bit field elements. The current implementation builds the
larger table at GPU process startup from the existing dumped 8-bit table. That keeps the
state file backward-compatible, but small benchmark chunks pay the startup cost repeatedly.

This knob is exact: it changes how `k * G` is computed, not what point is computed.

The measured return diminishes quickly. On the RTX 5090 current SOTA base, `batch_comb22`
was about `2.8%` faster than `batch_comb16` over a 32,768-nonce slice. The larger tables
are therefore useful as opt-in tuning knobs, but the main speedup still comes from batch
inversion rather than VRAM alone.

## `GPU_GCD_MODE`: GCD Check Ordering

The dialog-GCD filter has two kinds of rejection:

1. the full classical GCD does not converge within `active_iterations`;
2. the truncated circuit-style replay overflows a configured width envelope.

The baseline order is:

```text
full convergence check
then truncated width check
```

That is `GPU_GCD_MODE=full_first`, and it preserves the original behavior.

`GPU_GCD_MODE=trunc_first` swaps the exact order:

```text
truncated width check
then full convergence check
```

The accepted set is the same because both checks still run. This can help when many hard
factors fail due to width overflow, because the kernel can avoid the full convergence
counter on those factors.

`GPU_GCD_MODE=trunc_only` runs only the truncated width check. This is intentionally not an
exact replacement for the full filter. It can emit extra candidates that later fail
validation, so it should be treated as a noisy candidate generator, not as proof of
cleanliness.

`GPU_GCD_MODE=single_pass` is the *exact* version of `trunc_only`. The full filter rejects a
factor for either of two reasons: (1) the untruncated GCD does not converge within
`active_iters`, or (2) the truncated circuit-style walk overflows the width envelope. The
baseline runs these as two separate passes (a full untruncated convergence walk plus the
truncated overflow walk). `single_pass` folds them into one truncated walk that also tests
"did `v` reach 0 within `active_iters`?":

```text
for step in 0..active_iters:
    if truncated_step overflows -> reject (width)
    if v == 0                   -> accept (converged)
reject (ran out of steps without converging)
```

Why it is exact: inside the width envelope the truncated step is identical to the full step
(truncation only drops bits that are zero on the support / the truncated comparator resolves
the same branch) — which is precisely the support condition under which the two-pass filter
is itself exact. So when the truncated walk reaches `v==0` with no overflow at step `k`, the
full GCD also converges at `k`; when it runs all `active_iters` with neither, the full GCD
also failed to converge in time. The candidate-set equality test vs the `full_first` baseline
confirms this empirically.

Performance note (measured, RTX 5090): the saving is **small** — about `+4%` stacked on
`batch_inv`+`comb` and ~`0%` on its own. The reason is that the full convergence walk it
removes is already nearly free: it early-exits as soon as `v==0`, and for the common hard
factors the dominant cost is elsewhere (per-shot field arithmetic + SHAKE), not the second
GCD pass. So `single_pass` is worth enabling (it is exact and never hurts), but it is not a
large lever. See the changelog for the full A/B table.

## `GPU_WAVE`: Threads Per Nonce Wave

`gpu_island2.cu` assigns one CUDA block to one nonce. The block processes the 9,024 shots
in waves:

```text
number_of_waves = ceil(9024 / GPU_WAVE)
```

Larger waves have an obvious benefit:

```text
fewer waves per nonce
```

But larger waves can also hurt:

```text
more threads per block
more shared memory in batch mode
possibly lower occupancy
more work issued before a hard_flag can stop the block
```

That last point is subtle. If a hard shot appears early inside a large wave, every lane in
that wave still does its assigned shot before the block observes the shared `hard_flag`.
Smaller waves can stop sooner, while larger waves reduce loop overhead and may improve
parallelism on clean or late-failing nonces.

So `GPU_WAVE` is a real tuning knob, not a monotonic "bigger is better" setting. It should
be benchmarked separately for baseline mode and batch-inversion mode.

## `GPU_FAN_BITS`: Nonce-Fan (shared SHAKE prefix)

Each nonce reseeds SHAKE by absorbing a 96-op identity tail whose ops are selected by the
nonce's 48 bits (bit `i` -> two ops, fed low bit first). That `squeeze_init` is ~39 Keccak-f
permutations per nonce. The nonce-fan exploits that the *low* tail bits are absorbed
*first*: precompute the sponge state after absorbing the low `K` bits for all `2^K` prefixes,
then each nonce loads its prefix (a 208-byte table read) and only absorbs the high `48-K`
bits.

```text
2^K prefix states, each = 25 sponge words + pt   (208 B/entry)
K=20 -> 208 MiB, K=22 -> 856 MiB, K=24 -> 3.5 GiB
```

This is **exact** — identical total absorption, just cached prefix — and the table builds
once (amortized over the whole scan). It cuts the per-nonce tail absorption by `K/48`.

Performance note (measured, RTX 5090): only **~+1.5%** on the current SOTA base. The
hypothesis that `squeeze_init` dominates was wrong here — cutting it in half moved the needle
1.5%, i.e. it is only ~3% of per-nonce time in this fast-reject regime (the per-wave squeeze
and shot work dominate). May help more on init-bound bases; re-measure with `bench-gpu-knobs`.

## `EVAL_FAST_REJECT`: Eval Early-Exit (eval phase, not scan)

This knob is in the challenge's `eval_circuit`, the slow trusted stage that validates a
GCD-clean candidate by simulating the whole circuit over 9024 shots in batches of 64. With
`EVAL_FAST_REJECT=1`, the eval stops at the first batch that records any failure (the
search/validate path only needs a clean/dirty verdict, not the full mismatch count). Default
`0` keeps scoring/submission runs complete and byte-identical; a clean island still runs all
batches and reads `0/0/0`.

It also **defers the per-shot EC scalar-mults into the batch loop**, so an early exit skips
the muls for batches it never reaches. That is the key: a simple batch-only early-exit was
capped at ~1.5× by the ~9 s upfront derivation of all 9024 inputs; deferring it gives
**~8.5× average** on dirty candidates (16.1 s → ~1.9 s). The `FAST=0` scoring path is
byte-identical to the original (same full mismatch counts), and a clean island still derives
and simulates all 9024 shots and reads `0/0/0`.

This is the **exact "apply pre-scan"**: the full eval already checks apply-cleanliness, so a
fast-rejecting eval *is* the apply pre-scan — with zero false negatives and no GPU
re-implementation of the apply phase. The change lives in the challenge repo (reset by
`ecdsafail sync`); re-apply `patches/eval_fast_reject.diff`.

## Recommended Evaluation Order

Test exact knobs before noisy knobs:

1. default baseline;
2. `GPU_GCD_MODE=trunc_first` and `GPU_GCD_MODE=single_pass` (both exact);
3. `GPU_WAVE=64`, `128`, `256`;
4. `GPU_BATCH_INV=1` with wave sweeps;
5. `GPU_COMB_BITS=16` on sufficiently large chunks;
6. combinations of the winners (e.g. `GPU_BATCH_INV=1 GPU_COMB_BITS=16 GPU_GCD_MODE=single_pass`);
7. `GPU_GCD_MODE=trunc_only` only as an aggressive candidate-generation experiment.

For every setting, first check a known-clean nonce, then benchmark a dirty range and compare
both nonce/s and candidate count.

## Measured Reality Check (what actually moves the needle)

After A/B'ing all of the above on an RTX 5090, the practical takeaways are:

- `GPU_BATCH_INV=1` is the only large exact lever, but its size is **base-dependent**: ~1.5x
  on bases where nonces reject slowly (many per-shot inversions run) and only a few percent on
  bases where nonces reject fast (the per-nonce SHAKE `squeeze_init`, ~35 Keccak-f, dominates
  instead). `GPU_COMB_BITS` adds ~10%; `single_pass` adds ~0-4%.
- Native `sm_120` compilation (CUDA 12.8) was measured to give **no** speedup over the
  driver's PTX-JIT from `compute_80` on a 581-series driver — sometimes slightly slower. Keep
  the PTX path; don't carry a second toolchain for it.
- Per-shot field-arithmetic tricks (Montgomery multiply, fused single inversion, lazy
  reduction) only help the slow-reject regime, where `GPU_BATCH_INV` already captures most of
  the inversion cost. The genuinely under-optimized cost on fast-reject bases is the per-nonce
  `squeeze_init`, but an exact incremental update across sequential nonces is blocked by the
  tail feed order (low nonce bits are absorbed first), so it is left as an open problem.
