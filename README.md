# ecdsafail-island-gpu

**GPU-accelerated Fiat-Shamir "island" search for the [ecdsa.fail](https://www.ecdsa.fail/) quantum point-addition challenge.**

The challenge asks you to minimize `avg_executed_Toffoli × peak_qubits` for a reversible
secp256k1 point-add circuit, validated against **9,024 hash-derived test inputs**. Almost
every score improvement comes from *tightening a config lever* (narrowing a comparator,
dropping a GCD iteration, truncating a register width) — but each tightening only stays
correct on a lucky set of those 9,024 inputs. That lucky set is selected by a
`DIALOG_TAIL_NONCE` (a free 96-gate identity tail that reseeds the inputs), and **after
every lever change you must re-hunt a clean nonce.**

Re-hunting by running the quantum simulator on candidate nonces is brutally slow
(~minutes per nonce). This repo gives you a **bit-exact GPU port of the circuit's own
classical pre-filter** that screens nonces ~1000× faster than the simulator, plus a
shot-parallel CUDA kernel that's **7× faster than a naive GPU port** — turning a
multi-hour island hunt into **minutes**.

> It was built while taking the public leaderboard to a new SOTA by pushing
> `DIALOG_GCD_ACTIVE_ITERATIONS` lower than anyone had — a lever that's only reachable
> *because* the search is fast enough to find its rare islands. See `docs/levers.md`.

---

## What it does (and why it's fast)

A "clean island" = a nonce whose 9,024 derived point-add inputs are all safe under your
tightened config. The circuit ships a classical pre-filter (`dialog_gcd_classical_filter`,
analysis-only, not called by `build()`) that classically replays the truncated K2
binary-GCD transcript on both inversion factors of each input and rejects any nonce with a
width-envelope overflow or non-convergence. This repo:

1. **`dump_gpu_state.rs`** — exports a `gpu_state.bin` capturing everything the GPU needs:
   the SHAKE256 prefix state (so the 96-op tail is the only per-nonce hashing), the
   per-step width/compare/carry arrays (precomputed in Rust so the GPU stays integer-exact),
   the windowed comb table, and the filter config. Its byte-exact Keccak is validated `==`
   the `sha3` crate.
2. **`gpu_island2.cu`** — a CUDA kernel that, per nonce, derives all 9,024 inputs (SHAKE256
   → k1,k2 → comb `k·G` → point-add factors) and runs the GCD filter, with **one block per
   nonce and 128 threads splitting the 9,024 shots** (cooperative squeeze in shared memory,
   block-wide early-exit). ~**1,700 nonce/s/GPU** on an A100 (vs ~236/s naive).
   It now checks the first GCD factor `dx = tx - ox` before constructing the second factor
   `c = ox - rx`, so many dirty shots avoid the affine-add denominator inversion entirely.

Throughput stacks across GPUs, so a ~1/1M-density island is ~5 minutes on 2×A100.

**It is bit-exact.** A nonce reported `CLEAN` by the GPU is then quantum-confirmed with the
real `eval_circuit` (the GCD filter doesn't model the apply phase, so ~9% of GCD-clean
candidates fail the full 0/0/0 check — you confirm, then submit).

---

## Requirements

- The challenge repo checked out locally (`ecdsafail clone` or `git clone`) + the
  `ecdsafail` CLI logged in, and a working `ecdsafail run`. The Rust helpers build against
  the repo's `quantum_ecc` crate. **Keep the repo + `ecdsafail` CLI on your laptop** — even
  when the GPU is remote, only the search runs on the GPU box.
- An **NVIDIA GPU + CUDA toolkit (`nvcc`)** — either local, or a rented box you `ssh` into
  (vast.ai / Lambda / RunPod / CoreWeave / any). No GPU? `rust/island_search.rs` is a CPU
  fallback (slower, but fine for low-density levers).
- `python3` (score arithmetic), `perl` (CRLF-safe config edits — see Gotchas).

### GPU support — portable across the whole NVIDIA lineup, single- or multi-GPU
The kernel source is **architecture-agnostic** (plain integer ops, shared memory, atomics)
and is **compiled on the target machine with its compute capability auto-detected**, so the
same tool runs on:

| class | examples | compute cap | arch |
|---|---|---|---|
| datacenter | **H200, H100** | 9.0 | sm_90 |
| datacenter | **A100**, A30 | 8.0 | sm_80 |
| workstation/consumer | **RTX 50** (5090/5080) | 12.0 | sm_120 |
| | **RTX 40** (4090/4080) | 8.9 | sm_89 |
| | **RTX 30** (3090/3080) | 8.6 | sm_86 |
| older | V100 / T4 | 7.0 / 7.5 | sm_70 / sm_75 |

`NVCC_ARCH=auto` (the default) detects the card and builds native SASS; if your CUDA
toolkit is older than your GPU (e.g. a brand-new RTX 50), it **falls back to PTX that the
driver JIT-compiles** so it still runs. `GPUS=auto` (default) **uses every GPU on the box**
— the search range is split across all of them, one process pinned per device. Set
`GPUS=N` or `NVCC_ARCH=sm_90` to pin explicitly.

> The kernel was validated bit-exact on A100 (sm_80). Other arches use the same source and
> the auto-detect/JIT build, so they compile and run unchanged — after `build`, always run
> the quickstart sanity check (GPU must flag the base's known nonce as `CLEAN`).

---

## 60-second quickstart

```bash
git clone <this-repo> ecdsafail-island-gpu && cd ecdsafail-island-gpu

# Local GPU:
./island.sh init-local /path/to/ecdsafail-challenge      # detects GPU, builds the kernel

# OR a remote GPU box (any provider) — just paste the ssh command your provider gave you:
./island.sh init-remote "ssh -p 40162 ubuntu@1.2.3.4" /path/to/ecdsafail-challenge

./island.sh install        # build the Rust helpers in the challenge repo
./island.sh doctor         # shows the GPU(s), compute cap, and nvcc on the target
```

`init-*` writes `config.env`, runs `doctor`, and compiles the kernel (auto-arch, all GPUs).
Now confirm the port is correct **for your current base** by checking it flags the base's
*known* clean nonce (read it from `mod.rs`: `DIALOG_TAIL_NONCE`):

```bash
./island.sh dump "" /tmp/base.bin                 # dump for the CURRENT (unchanged) config
./island.sh search /tmp/base.bin <known_nonce> 1  # MUST print: CLEAN nonce=<known_nonce>
```

If that prints `CLEAN`, you're wired up correctly. Now hunt an island for a tighter config:

```bash
# one command: measure the lever, dump, GPU-search, and quantum-confirm candidates
./island.sh hunt DIALOG_GCD_ACTIVE_ITERATIONS=258 1 2000000
# -> prints "CLEAN nonce=12345 tof=... qubits=1309 score=..." for any fully-clean island
```

Bake the winner (CRLF-safe) and submit:

```bash
./island.sh bake DIALOG_GCD_ACTIVE_ITERATIONS 258 DIALOG_TAIL_NONCE 12345
# -> shows a clean 2-line diff and the ecdsafail run score
cd $CHALLENGE && ecdsafail submit --note-file note.md --model "..." --claimed-score <score>
```

---

## The workflow, step by step

| step | command | what it does |
|---|---|---|
| 1. install | `./island.sh install` | drop `dump_gpu_state`, `count_tof`, `island_search` into `CHALLENGE/src/bin`, build |
| 2. pick a lever | `./island.sh measure DIALOG_GCD_ACTIVE_ITERATIONS=258` | print exact Toffoli (CCX) for baseline vs the tighter config → `Δ × peak` = your score win |
| 3. build kernel | `./island.sh build` | `nvcc` the kernel (local, or scp+build on your remote box) |
| 4. dump | `./island.sh dump DIALOG_GCD_ACTIVE_ITERATIONS=258 s.bin` | encode the GCD filter+comb+prefix for that config |
| 5. search | `./island.sh search s.bin 1 2000000` | GPU-screen 2M nonces → `CLEAN nonce=...` candidates |
| 6. validate | `./island.sh validate DIALOG_GCD_ACTIVE_ITERATIONS=258 <n>...` | quantum-confirm 0/0/0 + print score |
| 7. bake | `./island.sh bake DIALOG_GCD_ACTIVE_ITERATIONS 258 DIALOG_TAIL_NONCE <n>` | CRLF-safe edit + `ecdsafail run` |
| 8. submit | `ecdsafail submit ...` | (in the challenge repo) |

`./island.sh hunt CFG START N` chains steps 2/4/5/6. See `examples/walkthrough.md`.

### Experimental search-kernel knobs
The default search path remains the original production `gpu_island2` behavior:
`GPU_BATCH_INV=0 GPU_COMB_BITS=8 GPU_GCD_MODE=full_first GPU_WAVE=128`.

Set any of these on `./island.sh search` or `./island.sh hunt`; local and remote modes both
forward them to the GPU binary:

```bash
GPU_BATCH_INV=1 GPU_WAVE=128 ./island.sh search s.bin 1 2000000
GPU_COMB_BITS=16 ./island.sh search s.bin 1 2000000
GPU_GCD_MODE=trunc_first ./island.sh search s.bin 1 2000000
```

| option | values | effect |
|---|---|---|
| `GPU_BATCH_INV` | `0`/`1` | `1` launches the cooperative block kernel that batch-inverts the two Jacobian `Z` values and the affine-add denominator across a wave. Exact candidate set. |
| `GPU_COMB_BITS` | `8`/`16` | `16` builds a 64 MiB comb16 table on the GPU at process startup and halves scalar-mul table lookups/adds. Exact, but startup-heavy. |
| `GPU_GCD_MODE` | `full_first`, `trunc_first`, `trunc_only` | `full_first` is the current exact order. `trunc_first` is exact but checks width overflow before convergence. `trunc_only` is a noisy experimental prefilter that can emit extra false positives, so always validate. |
| `GPU_WAVE` | `32`..`256` | CUDA block threads per nonce wave. Default `128`; values are rounded up to a warp multiple and capped at `256`. |

Before trusting a new GPU build, run the integrated correctness smoke:

```bash
./island.sh test-gpu-knobs "" 0 4096
```

It rebuilds the helper binaries, builds the CUDA kernel, dumps the current state, checks the
GPU Keccak probe, verifies that every knob still finds the baked `DIALOG_TAIL_NONCE`, and
compares exact candidate sets over the requested range, including the combined exact paths
used by the benchmark. Use `ISLAND_CONFIG=/tmp/box.env` to point the same repo at a one-off
remote GPU without editing the tracked `config.env`.

To validate speedups after correctness passes, run the fixed-range throughput benchmark:

```bash
GPU_BENCH_RUNS=3 GPU_BENCH_WARMUPS=1 ./island.sh bench-gpu-knobs "" 0 16384
```

This dumps the current SOTA state once, uploads it once in remote mode, warms up each
variant, then runs the raw `gpu_island2` binary over the same nonce interval and prints
average/min/max `nonce/s` plus speedup relative to the default baseline. The default
benchmark variants include isolated knobs (`trunc_first`, `wave64`, `wave256`,
`batch_inv`, `comb16`) and combined exact paths (`batch_wave256`, `batch_comb16`,
`all_exact`). Use `GPU_BENCH_SKIP_INSTALL=1` or `GPU_BENCH_SKIP_BUILD=1` when the Rust
helpers or CUDA binary are already fresh.

Interpretation caveats:

- Run `test-gpu-knobs` first; speed is only meaningful for variants that preserve the
  expected candidate set, except intentionally noisy modes.
- Use a fixed challenge commit, config, start nonce, and range size for every comparison.
- Warmups matter on new cards or old toolkits because PTX may be JIT-compiled on the first
  run.
- The benchmark measures single-process kernel throughput. Multi-GPU scheduling speed is
  still best checked with `./island.sh search`.
- For `GPU_COMB_BITS=16`, the one-time table construction cost is outside the timed CUDA
  event, so also check wall-clock behavior for very small chunks.

### Local vs remote GPU
`init-local` / `init-remote` set this up for you. In remote mode, `build`/`search`/`doctor`
automatically `scp` the kernel + runtime scripts + the (tiny, ~515 KB) state dump into a
working dir under the remote home (`~/.ecdsafail_island`, so it works for `ubuntu@`, `root@`,
any user) and run over SSH; the Rust steps (`dump`/`validate`, which need the challenge repo)
stay on your laptop. You keep the repo + `ecdsafail` CLI local and rent GPUs only for the
search. Arch is auto-detected on the remote, so you don't need to know the card's `sm_XX`.

**Long unattended searches:** an `init`+`search` over millions of nonces holds the SSH
connection open for the duration. For multi-hour runs, wrap the node-side search in `tmux`/
`screen` (or split the range and run several `./island.sh search` calls).

---

## Using it inside an autoresearch / agent harness

The CLI is the integration surface — your harness (or an AI agent) drives the loop:

```
measure many candidate levers  ->  pick the biggest score win that's plausibly findable
  ->  hunt an island for it     ->  if a 0/0/0 island is found, bake + submit
  ->  on a new SOTA from others, `ecdsafail sync` and repeat on the new base.
```

There's a Claude-Code / agent **skill** in [`SKILL.md`](SKILL.md): copy this folder to
`~/.claude/skills/ecdsafail-island/` (or point your harness at it) and the agent can invoke
"find me an island for `DIALOG_GCD_...=X`" as a tool. The skill encodes the
measure→search→validate→bake loop and the gotchas below.

**Natural-language remote setup.** With the skill installed, you can just tell the agent in
plain language: *"Run the search on a remote GPU machine — here's the ssh command:
`ssh -p 40162 ubuntu@1.2.3.4`."* The agent turns that into
`./island.sh init-remote "ssh -p 40162 ubuntu@1.2.3.4" <challenge>`, which provisions the
box (detects the GPU(s), builds the kernel) and runs every subsequent search there — no
manual `sm_XX` / multi-GPU wiring needed.

**Key principle for the harness:** *measure every lever's true Toffoli cost first* (step 2)
and target the biggest, least-contested one — don't just push whatever lever you searched
last time. `docs/levers.md` catalogs the levers and their measured per-unit Toffoli cost.

---

## Gotchas (these cost real time — read them)

- **CRLF line endings.** `mod.rs` uses CRLF. Editing it with a tool that rewrites line
  endings (most editors / IDE "format on save" / naive sed) corrupts the touched region
  into a huge diff that **fails leaderboard promotion** even when the score is correct.
  Always edit config with `perl`/`./island.sh bake`, and verify `git diff` is *exactly* your
  changed lines (the `bake` command checks the CR count for you).
- **Validate before you trust.** The GPU filter models the GCD only (width + convergence),
  not the apply phase. A `CLEAN` from the GPU is a *candidate* — always `validate` (real
  `eval_circuit`) before submitting. ~9% of candidates pass; you need ~10 per winner.
- **Re-validate the port on each new base.** After `ecdsafail sync` to a new SOTA base,
  re-run the quickstart sanity check (GPU must flag the base's known nonce as `CLEAN`). The
  dump reads the filter config dynamically, so it tracks bases that keep the dialog-GCD
  architecture — but confirm, don't assume.
- **Stale binaries.** `cargo build --bin A --bin B` aborts *all* targets if one is missing.
  Then `build_circuit`/`eval_circuit` silently stay stale from the previous base. Always
  confirm `ecdsafail run` equals the leaderboard score after `install`.
- **Remote process management.** For long unattended searches, run the node loop inside a
  `tmux`/`screen` or a backgrounded ssh that stays connected — detached `setsid &` jobs do
  not reliably survive on some rented boxes. `kill -9` on a running kernel is *pending*
  until that kernel returns.

---

## Layout

```
island.sh              unified CLI (init-local/init-remote/doctor/install/measure/build/
                                     dump/search/validate/bake/hunt)
config.env.example     reference config (init-* writes config.env for you)
SKILL.md               agent/Claude-Code skill manifest
CHANGELOG.md           decision log for major search changes, rationale, expected impact
runtime/               scripts that run ON the GPU machine (local or scp'd to the remote):
  build_kernel.sh        auto-detect compute cap -> nvcc (native + PTX-JIT fallback)
  search_driver.sh       multi-GPU parallel chunked search (splits range across all GPUs)
  doctor.sh              report GPUs / compute cap / nvcc
cuda/
  gpu_island2.cu       PRODUCTION shot-parallel kernel (KERNEL2=1)
  gpu_island.cu        reference serial kernel (for cross-checking)
rust/
  dump_gpu_state.rs    exports gpu_state.bin (Keccak validated == sha3)
  count_tof.rs         static Toffoli (CCX) lever meter
  island_search.rs     CPU reference searcher (no-GPU fallback)
docs/
  how-it-works.md      the dialog-GCD circuit + island search, explained
  levers.md            the lever catalog + measured Toffoli costs
  kernel-notes.md      kernel design / throughput / validation notes
  theory-knobs.md      theoretical background for the experimental GPU knobs
examples/
  walkthrough.md       end-to-end example (lower ACTIVE_ITERATIONS, find island, submit)
```

## License
MIT — see `LICENSE`. Not affiliated with ecdsa.fail; built by challenge participants.
Contributions welcome (more kernels, batch inversion, other GPU vendors).
