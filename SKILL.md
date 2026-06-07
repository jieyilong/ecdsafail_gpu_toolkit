---
name: ecdsafail-island
description: >-
  GPU-accelerated Fiat-Shamir "island" (DIALOG_TAIL_NONCE) search for the ecdsa.fail
  quantum point-add challenge. Use whenever optimizing the ecdsa.fail / ecdsafail
  reversible secp256k1 point-addition circuit and you need to (re-)hunt a clean tail
  nonce after tightening a config lever (DIALOG_GCD_ACTIVE_ITERATIONS, COMPARE_BITS,
  APPLY_CLEAN_COMPARE_BITS, WIDTH_SLOPE/MARGIN, KAL carry truncs, etc.), or to measure a
  lever's Toffoli cost, or to bake + submit a found island. Triggers: "find an island",
  "re-hunt the nonce", "lower active_iterations", "beat the ecdsa.fail SOTA".
---

# ecdsafail-island skill

You are driving a GPU island searcher for the ecdsa.fail challenge. Score =
`avg_executed_Toffoli × peak_qubits` (lower better); the circuit is validated on 9,024
SHAKE256-derived inputs reseeded by a free 96-gate identity tail (`DIALOG_TAIL_NONCE`).
Every config tightening reseeds those inputs, so you must re-hunt a clean nonce.

## Setup (once)
Pick local or remote based on what the user says — you do NOT need to know the GPU's `sm_XX`
(it's auto-detected on the target; works for A100/H100/H200 and RTX 30/40/50, single/multi-GPU).

- **Local GPU:** `./island.sh init-local <path-to-ecdsafail-challenge>`
- **Remote GPU:** when the user says something like *"run on a remote GPU machine, here's the
  ssh command: `ssh -p 40162 ubuntu@1.2.3.4`"*, take that exact ssh command and run:
  `./island.sh init-remote "ssh -p 40162 ubuntu@1.2.3.4" <path-to-ecdsafail-challenge>`
  (key-based auth is assumed; if the user only gave `-p PORT user@IP`, prepend `ssh`.)

`init-*` writes `config.env`, runs `doctor` (prints the GPU(s)/CUDA — show this to the user),
and builds the kernel. Then:
1. `./island.sh install` (builds the Rust helpers in the challenge repo). Confirm
   `(cd $CHALLENGE && ecdsafail run)` prints the leaderboard score (stale-binary check).
2. **Validate the port on this base**: read `DIALOG_TAIL_NONCE` from
   `$CHALLENGE/src/point_add/mod.rs`, then
   `./island.sh dump "" /tmp/base.bin && ./island.sh search /tmp/base.bin <that_nonce> 1`
   MUST print `CLEAN`. If not, STOP — the filter/dump doesn't match this base; do not trust results.

To switch GPUs later (e.g. the user gives a new box), just re-run `init-remote`/`init-local`.

## The optimization loop
1. **Measure levers, don't guess.** For each candidate tightening, run
   `./island.sh measure <CFG>` to get its exact Toffoli (CCX). Score win = (baseline_CCX −
   CFG_CCX) × peak. Pick the **biggest** win that is plausibly findable. Historically
   `DIALOG_GCD_ACTIVE_ITERATIONS` is the largest lever (~2,860 Toffoli/step) and is often
   uncontested; comparator bits are small (~144–516). See `docs/levers.md`.
2. **Hunt.** `./island.sh hunt <CFG> <START> <N>` (e.g. `... DIALOG_GCD_ACTIVE_ITERATIONS=258
   1 2000000`). It measures, dumps, GPU-searches, and quantum-confirms candidates, printing
   `CLEAN nonce=... score=...` for any fully-0/0/0 island. If none, search a larger range
   (rarer islands need more nonces; lower active_iterations = rarer).
3. **Bake + submit ONLY a confirmed island.** Use `./island.sh bake <KEY> <VAL> DIALOG_TAIL_NONCE
   <nonce>` — it edits `mod.rs` CRLF-safely (NEVER use a normal file-edit tool on mod.rs; it
   corrupts CRLF and breaks promotion), shows the diff (must be exactly your lines), and runs
   `ecdsafail run`. Then in `$CHALLENGE`: `ecdsafail submit --note-file <note> --model <m>
   --claimed-score <score>`. Watch `ecdsafail submissions` until `promoted`.
4. **Stay on the frontier.** Periodically `cd $CHALLENGE && ecdsafail benchmark` / `submissions`.
   If someone else takes the lead, `ecdsafail sync --force`, re-`install`, re-validate the port,
   and re-run measure→hunt on the new base (apply your lever idea on top of their base).

## Hard rules
- A GPU `CLEAN` is a candidate; only a `./island.sh validate` 0/0/0 is submittable.
- Confirm `ecdsafail run` == leaderboard after every `install`/`sync` (stale-binary trap).
- Only submit if the confirmed score strictly beats the current best.
- Bake with `bake`/`perl` only. Verify the `git diff` is exactly the changed config lines.
