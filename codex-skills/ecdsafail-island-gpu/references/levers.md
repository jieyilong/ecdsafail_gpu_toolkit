# Lever catalog

These are the tunable `set_default_env(...)` knobs in `configure_ecdsafail_submission_route()`
(`src/point_add/mod.rs`) that change the Toffoli count. **Always measure the exact cost on
your current base** with `./island.sh measure <CFG>` — the numbers below are representative
(measured on a 1309q base, ~1.49–1.50M Toffoli) and shift between bases.

## The big lever — GCD iteration count
| knob | direction | ~Toffoli / step | notes |
|---|---|---|---|
| `DIALOG_GCD_ACTIVE_ITERATIONS` | lower = fewer | **~2,860 / step** | unrolled GCD loop length. Worst-case-sized; lower it and re-hunt an island where every input converges in the smaller budget. The **largest single lever** and frequently uncontested. Rarer islands as you go lower (convergence-bound). |

## Comparator-width levers (value-exact bit truncation)
| knob | direction | ~Toffoli / bit | notes |
|---|---|---|---|
| `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS` | lower = fewer | **~516 / bit** | apply-phase overflow-correction comparator. GCD-transcript-independent (cleanest to land). |
| `DIALOG_GCD_COMPARE_BITS` | lower = fewer | **~144 / bit** | GCD branch comparator. Changing it alters the GCD transcript (couples to the apply phase). |

## Width-envelope / carry truncation
| knob | direction | notes |
|---|---|---|
| `DIALOG_GCD_WIDTH_SLOPE_X1000` | higher = narrower | u,v shrink as the GCD runs; the active register width tapers `N − step·slope + margin`. Steeper slope = fewer Toffoli on later steps but more overflow risk. |
| `DIALOG_GCD_WIDTH_MARGIN` | lower = narrower | safety margin on the width taper. |
| `DIALOG_GCD_BODY_CARRY_BAND_TRIMS` | trims | per-band carry-width trims in the GCD body. |
| `KAL_DOUBLE_CARRY_TRUNC_W`, `KAL_FOLD_CARRY_TRUNC_W` | tune | Solinas fold carry widths in the apply phase. Often have a single non-obvious sweet spot (neighbors blow up the peak). |

## Peak-qubit knobs (the other factor)
| knob | notes |
|---|---|
| `SELECTED_BODY_GATE_SUFFIX_CARRIES`, `KAL_*` packing, log-hosting flags | affect `peak_qubits`. Worth ~1.5M score/qubit but usually pinned at an architectural floor (1309 "round84"). Read the peak from `eval_circuit`'s `qubits:` line — it prints *before* the correctness tests, so you can read it for any config without a clean nonce. |

## How to choose
1. `./island.sh measure <CFG>` each candidate; compute `(baseline − CFG) × peak`.
2. Prefer the biggest win that's still findable. Comparator/iteration tightenings that change
   the GCD transcript couple to the apply phase (lower clean-island yield); apply-side and
   pure-iteration tightenings are cleaner.
3. Stack carefully: two tightenings together need one nonce clean for **both** (rarer).
4. After picking, `./island.sh hunt <CFG> <START> <N>`.

## Measuring peak without a clean nonce
```bash
cd $CHALLENGE && (cd $(mktemp -d) && env <CFG> DIALOG_TAIL_NONCE=1 build_circuit >/dev/null && \
  env <CFG> DIALOG_TAIL_NONCE=1 eval_circuit 2>&1 | grep -m1 'qubits')
```
`eval_circuit` prints `qubits:` before correctness, so a dirty nonce still reveals the peak.
