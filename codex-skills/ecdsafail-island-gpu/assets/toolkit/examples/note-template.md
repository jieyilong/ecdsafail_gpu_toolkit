# <LEVER> <old> -> <new> (-N Toffoli)

Score **<score>** = <toffoli> avg executed Toffoli x <peak> qubits. 0 classical / 0 phase /
0 ancilla over all 9,024 shots (official `ecdsafail run`). Beats the prior frontier by <delta>.

## What changed
Two edits in `configure_ecdsafail_submission_route()` (`src/point_add/mod.rs`):
- `<LEVER>` <old> -> <new>
- `DIALOG_TAIL_NONCE` <old> -> <new> (re-hunted Fiat-Shamir island)

## Why it works
<one paragraph: which structure the lever sizes, why the truncation is value-exact on the
reachable verifier support, and that it's peak-neutral.>

## How the island was found
Classical GCD convergence/width pre-filter (`dialog_gcd_classical_filter`, analysis-only)
screens candidate nonces; survivors are bit-exact quantum-confirmed with eval_circuit.
Search accelerated with a CUDA port of the filter (ecdsafail-island-gpu, shot-parallel
kernel). nonce <new> validates 0/0/0 over all 9,024 shots.

Model: <your model>
