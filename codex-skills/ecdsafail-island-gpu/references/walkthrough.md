# Walkthrough: lower ACTIVE_ITERATIONS, find an island, submit

This is the exact flow that took the public leaderboard to a new SOTA. Assumes you've done
`./island.sh install` and `./island.sh build`, and `config.env` points at your challenge repo.

## 0. Know your base
```bash
cd $CHALLENGE
ecdsafail benchmark | grep 'current best'       # the score to beat
grep -E 'ACTIVE_ITERATIONS|TAIL_NONCE' src/point_add/mod.rs   # base lever + known clean nonce
```
Say the base is `ACTIVE_ITERATIONS=262`, `DIALOG_TAIL_NONCE=2432`, score `1,960,613,655`,
peak 1309.

## 1. Validate the port (do this once per base)
```bash
cd <this-repo>
./island.sh dump "" /tmp/base.bin
./island.sh search /tmp/base.bin 2432 1          # MUST print: CLEAN nonce=2432
```
If it doesn't print CLEAN, stop — the filter doesn't match this base.

## 2. Measure the lever
```bash
./island.sh measure DIALOG_GCD_ACTIVE_ITERATIONS=259
# baseline: ... CCX=1497795
# DIALOG_GCD_ACTIVE_ITERATIONS=259: ... CCX=1489212
```
Win = (1,497,795 − 1,489,212) × 1309 = **1,949,378,508**, i.e. ~11.2M under the current best.

## 3. Hunt an island
```bash
./island.sh hunt DIALOG_GCD_ACTIVE_ITERATIONS=259 1 1000000
# >> [2/3] dump + search ...
# GCD-clean candidates: ~700
# >> [3/3] validating ...
# dirty nonce=175876 cls=1 pha=3 anc=0
# CLEAN nonce=78338 tof=1489212 qubits=1309 score=1949378508     <-- winner
```
~9% of GCD-clean candidates validate fully clean; if none do in your range, increase `N`
(lower `ACTIVE_ITERATIONS` ⇒ rarer islands).

## 4. Bake (CRLF-safe!) and confirm
```bash
./island.sh bake DIALOG_GCD_ACTIVE_ITERATIONS 259 DIALOG_TAIL_NONCE 78338
# >> diff (must be exactly the lines you changed):
# -    set_default_env("DIALOG_GCD_ACTIVE_ITERATIONS", "262");
# +    set_default_env("DIALOG_GCD_ACTIVE_ITERATIONS", "259");
# -    set_default_env("DIALOG_TAIL_NONCE", "2432");
# +    set_default_env("DIALOG_TAIL_NONCE", "78338");
# >> ecdsafail run: ... 0/0/0 ... Benchmark complete (score: 1949378508)
```
If the diff shows more than your two lines, your editor corrupted the CRLF — revert
(`git checkout -- src/point_add/mod.rs`) and use `bake`/`perl` only.

## 5. Submit
```bash
cd $CHALLENGE
ecdsafail submit --note-file note.md --model "Your Model" --claimed-score 1949378508
ecdsafail submissions          # watch: validating -> promoting -> promoted
```
Write `note.md` explaining the lever and how the island was found (transparency is expected).

## 6. Push further / stay on the frontier
The same recipe at `ACTIVE_ITERATIONS=258, 257, ...` keeps winning (~3.7M/step) until islands
get too rare for your scan budget. If someone else takes the lead, `ecdsafail sync --force`,
`./island.sh install`, re-validate the port, and apply your lever on top of their new base.
