---
name: ecdsafail-island-gpu
description: "GPU-accelerated Fiat-Shamir island search for the ecdsa.fail reversible secp256k1 point-addition challenge, adapted from jieyilong/ecdsafail_gpu_toolkit. Use when optimizing ecdsa.fail or ecdsafail and needing to install or run island.sh, set up local or remote NVIDIA GPU search, measure Toffoli impact of config levers, hunt or re-hunt a clean DIALOG_TAIL_NONCE after tightening DIALOG_GCD_ACTIVE_ITERATIONS, comparator bits, width margins, KAL carry truncations, or similar levers, test GPU knob correctness, validate GPU-clean candidates, bake CRLF-safe config changes, or prepare a submission. Triggers include: find an island, re-hunt the nonce, GPU island search, lower active_iterations, DIALOG_TAIL_NONCE, island.sh, and beat the ecdsa.fail SOTA."
---

# ECDSA Fail Island GPU

Drive the GPU island-search toolkit for the ecdsa.fail point-add benchmark. The goal is to reduce `avg_executed_Toffoli * peak_qubits` by tightening a legal circuit config lever, then finding a `DIALOG_TAIL_NONCE` whose 9,024 hash-derived inputs still validate.

## Non-Negotiables

- Treat a GPU `CLEAN` result as a prefilter candidate only. Call a nonce submittable only after `./island.sh validate <CFG> <nonce>` or `ecdsafail run` confirms full `0/0/0`.
- Re-run the base sanity check after every `ecdsafail sync`, promoted-base change, toolkit rebuild, GPU switch, or helper reinstall.
- Use `./island.sh bake` or CRLF-safe `perl` edits for `src/point_add/mod.rs`; do not use ordinary text editing for baked config lines.
- Inspect `git diff` before submission. The diff should show only the intended benchmark-legal lines.
- Keep the local challenge repo and `ecdsafail` CLI on the laptop. Remote mode sends only the GPU search work to the GPU box.
- Pair with `ecdsafail-cli` for login, benchmark, sync, submit, notes, and CLI-specific errors. Pair with `redsky` for broader frontier audit or strategy.

## Toolkit Location

This skill bundles the upstream toolkit under `assets/toolkit`.

Use the bundled installer when no toolkit checkout already exists:

```bash
scripts/install_toolkit.sh /path/to/ecdsafail-island-gpu-workdir
cd /path/to/ecdsafail-island-gpu-workdir
```

If the user has an existing clone of `jieyilong/ecdsafail_gpu_toolkit`, use that clone instead after confirming it has the expected `island.sh`, `cuda/`, `rust/`, and `runtime/` layout.

The bundled source snapshot is recorded in `references/toolkit-source.md`.

## Setup Flow

1. Locate the challenge repo.
   - Prefer the user-provided path.
   - If none is provided, check the current workspace and `/Users/olifreuler/Documents/New project/ecdsafail-challenge`.
   - Confirm `ecdsafail run` works before spending GPU time.
2. Initialize the toolkit.

```bash
# Local NVIDIA GPU
./island.sh init-local /path/to/ecdsafail-challenge

# Remote NVIDIA GPU; preserve the user's exact SSH command
./island.sh init-remote "ssh -p 40162 ubuntu@1.2.3.4" /path/to/ecdsafail-challenge
```

3. Build helper binaries and confirm the base.

```bash
./island.sh install
(cd "$CHALLENGE" && ecdsafail run)
```

4. Validate the GPU port against the current base nonce. Read `DIALOG_TAIL_NONCE` from `$CHALLENGE/src/point_add/mod.rs`.

```bash
./island.sh dump "" /tmp/base.bin
./island.sh search /tmp/base.bin <known_nonce> 1
```

Continue only if the search prints `CLEAN nonce=<known_nonce>`.

## Optimization Loop

1. Measure candidate levers before searching.

```bash
./island.sh measure DIALOG_GCD_ACTIVE_ITERATIONS=258
```

Score movement is `(baseline_CCX - candidate_CCX) * peak_qubits`. Read `references/levers.md` when choosing between GCD iterations, comparator widths, width margins, carry truncations, and peak-qubit knobs.

2. Hunt a nonce.

```bash
./island.sh hunt DIALOG_GCD_ACTIVE_ITERATIONS=258 1 2000000
```

`hunt` chains measure, dump, GPU search, and CPU validation. If no fully clean island appears, increase the range or choose a less rare lever.

3. For manual runs, keep the phases explicit.

```bash
./island.sh dump DIALOG_GCD_ACTIVE_ITERATIONS=258 s.bin
./island.sh search s.bin 1 2000000
./island.sh validate DIALOG_GCD_ACTIVE_ITERATIONS=258 <nonce> [<nonce>...]
```

4. Bake only a fully validated winner.

```bash
./island.sh bake DIALOG_GCD_ACTIVE_ITERATIONS 258 DIALOG_TAIL_NONCE <nonce>
(cd "$CHALLENGE" && ecdsafail run)
```

5. Submit only if the confirmed score strictly improves the current best.

```bash
cd "$CHALLENGE"
ecdsafail submit --note-file note.md --model "<model>" --claimed-score <score>
ecdsafail submissions
```

Use `assets/toolkit/examples/note-template.md` as a submission-note starting point.

## GPU And Eval Knobs

Keep scan knobs and validation knobs separate:

```bash
# Phase 1: GPU scan
GPU_BATCH_INV=1 GPU_COMB_BITS=22 GPU_GCD_MODE=single_pass GPU_FAN_BITS=22 GPU_WAVE=128 \
  ./island.sh search s.bin <START> <N>

# Phase 2: CPU validation
EVAL_FAST_REJECT=1 ./island.sh validate "<CFG>" <nonce> [<nonce>...]
```

- `EVAL_FAST_REJECT` is a validation-phase knob. It is a no-op on `search`.
- `GPU_GCD_MODE=single_pass` is a valid necessary filter but looser than `full_first`; it may hand more eval-dirty false positives to validation. Use `full_first` for the strictest prefilter.
- Use `GPU_COMB_BITS=16` instead of `22` when VRAM is constrained.
- For long runs, prefer chunk sizes around 500k to 1M to amortize startup and table build time.

Before trusting new GPU settings on a base or machine, run correctness tests before benchmarks:

```bash
GPU_TEST_COMB_BITS="20 22" ./island.sh test-gpu-knobs "" 0 4096
GPU_BENCH_COMB_BITS="20 22" GPU_BENCH_RUNS=2 ./island.sh bench-gpu-knobs "" 0 32768
```

Read `references/measured-speedups.md` and `references/theory-knobs.md` before changing the recommended stack.

## Remote GPU Notes

- `init-remote` writes `config.env`, copies runtime files to a remote workdir, builds the CUDA kernel on the target, and keeps Rust dump/validate steps local.
- Do not ask the user for `sm_XX`; the runtime auto-detects compute capability and can fall back to PTX JIT.
- For multi-hour searches, prefer `tmux` or `screen` on the remote machine, or split the range into smaller `search` calls.
- Re-run `./island.sh doctor` when changing boxes or debugging CUDA/NVIDIA setup.

## References

- `references/levers.md`: lever catalog and score arithmetic.
- `references/walkthrough.md`: end-to-end lower-`ACTIVE_ITERATIONS` example.
- `references/measured-speedups.md`: tested scan/eval knobs, chunk sizing, and throughput caveats.
- `references/theory-knobs.md`: background for each GPU knob.
- `references/how-it-works.md`: GCD filter and dump/kernel architecture.
- `references/kernel-notes.md`: kernel implementation notes and measured historical values.
- `references/toolkit-source.md`: source URL, bundled commit, and included asset list.
