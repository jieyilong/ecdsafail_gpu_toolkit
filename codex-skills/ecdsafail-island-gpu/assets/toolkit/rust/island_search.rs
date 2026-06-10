//! LOCAL TOOLING (not part of the submission).
//!
//! Fiat-Shamir island searcher for the dialog-GCD point-add circuit.
//!
//! The deployed circuit truncates the binary-GCD width envelope / iteration
//! count, so a small fraction (~1e-3) of random point-add inputs are "hard"
//! (a register overflows the truncated width or the GCD fails to converge in
//! ACTIVE_ITERATIONS). The circuit is validated against 9024 Fiat-Shamir-derived
//! random inputs whose seed is the SHAKE256 hash of the *entire* op stream. A
//! `DIALOG_TAIL_NONCE` appends a fixed-length 96-op identity tail (X;X pairs on
//! tx[0]/tx[1]) that reseeds those inputs WITHOUT changing the circuit action,
//! Toffoli count, or peak qubits. A "clean island" is a nonce whose 9024 inputs
//! contain no hard input.
//!
//! This binary finds clean islands fast by:
//!   1. building the op stream ONCE (so the SHAKE prefix is hashed once),
//!   2. per candidate nonce, cloning the prefix hash state and feeding only the
//!      96 tail-op bytes (O(1) instead of re-hashing 10M ops),
//!   3. deriving the 9024 inputs with a fast Jacobian k*G (the reference adder
//!      does an affine inversion per bit, ~1000x slower),
//!   4. running the bit-exact `dialog_gcd_classical_filter` with early-exit on
//!      the first hard input.
//!
//! The filter models the dominant hard-input source (width/convergence). It does
//! NOT model the apply-phase double/fold-carry truncations, so a filter-clean
//! nonce is a CANDIDATE that must be confirmed with the real `eval_circuit`.
//!
//! Usage:
//!   [CONFIG_ENV=...] ISLAND_THREADS=11 ./island_search <start_nonce> <count>
//!
//! Prints `CLEAN nonce=N` for each filter-clean nonce found.

use alloy_primitives::U256;
use quantum_ecc::circuit::{analyze_ops, QubitOrBit};
use quantum_ecc::point_add::dialog_gcd_classical_filter::{
    check_gcd_factor, point_add_gcd_factors, DialogGcdFilterConfig,
};
use quantum_ecc::point_add::{self, SECP256K1_P};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const P: U256 = SECP256K1_P;
const NONCE_BITS: u32 = 48;
const NUM_TESTS: usize = 9024;
const DOMAIN: &[u8] = b"quantum_ecc-fiat-shamir-v2";

// ───────────────────────── field arithmetic ─────────────────────────

#[inline]
fn fadd(a: U256, b: U256) -> U256 {
    a.add_mod(b, P)
}
#[inline]
fn fsub(a: U256, b: U256) -> U256 {
    if a >= b {
        a - b
    } else {
        P - (b - a)
    }
}
#[inline]
fn fmul(a: U256, b: U256) -> U256 {
    a.mul_mod(b, P)
}
#[inline]
fn fsqr(a: U256) -> U256 {
    a.mul_mod(a, P)
}

// ───────────────────────── Jacobian k*G ─────────────────────────
//
// Jacobian point (X,Y,Z) ~ affine (X/Z^2, Y/Z^3). Z==0 is the point at infinity.
// secp256k1 has a == 0.

#[derive(Clone, Copy)]
struct Jac {
    x: U256,
    y: U256,
    z: U256,
}

impl Jac {
    const INF: Jac = Jac {
        x: U256::ZERO,
        y: U256::ZERO,
        z: U256::ZERO,
    };
    #[inline]
    fn is_inf(&self) -> bool {
        self.z.is_zero()
    }
}

#[inline]
fn jac_double(p: Jac) -> Jac {
    if p.is_inf() || p.y.is_zero() {
        return Jac::INF;
    }
    // a == 0 doubling.
    let yy = fsqr(p.y);
    let s = fmul(U256::from(4u64), fmul(p.x, yy)); // 4*X*Y^2
    let m = fmul(U256::from(3u64), fsqr(p.x)); // 3*X^2
    let x3 = fsub(fsqr(m), fadd(s, s)); // M^2 - 2S
    let yyyy = fsqr(yy);
    let y3 = fsub(fmul(m, fsub(s, x3)), fmul(U256::from(8u64), yyyy)); // M(S-X3) - 8Y^4
    let z3 = fmul(fadd(p.y, p.y), p.z); // 2*Y*Z
    Jac { x: x3, y: y3, z: z3 }
}

/// Mixed addition: Jacobian `p` + affine `(qx,qy)`.
#[inline]
fn jac_add_affine(p: Jac, qx: U256, qy: U256) -> Jac {
    if p.is_inf() {
        return Jac {
            x: qx,
            y: qy,
            z: U256::from(1u64),
        };
    }
    let z1z1 = fsqr(p.z);
    let u2 = fmul(qx, z1z1);
    let s2 = fmul(qy, fmul(z1z1, p.z));
    let u1 = p.x;
    let s1 = p.y;
    if u1 == u2 {
        if s1 == s2 {
            return jac_double(p);
        }
        return Jac::INF; // P == -Q
    }
    let h = fsub(u2, u1);
    let r = fsub(s2, s1);
    let h2 = fsqr(h);
    let h3 = fmul(h2, h);
    let u1h2 = fmul(u1, h2);
    let x3 = fsub(fsub(fsqr(r), h3), fadd(u1h2, u1h2));
    let y3 = fsub(fmul(r, fsub(u1h2, x3)), fmul(s1, h3));
    let z3 = fmul(p.z, h);
    Jac { x: x3, y: y3, z: z3 }
}

#[inline]
fn jac_to_affine(p: Jac) -> (U256, U256) {
    if p.is_inf() {
        return (U256::ZERO, U256::ZERO);
    }
    let zinv = p.z.inv_mod(P).expect("z invertible");
    let zinv2 = fsqr(zinv);
    let zinv3 = fmul(zinv2, zinv);
    (fmul(p.x, zinv2), fmul(p.y, zinv3))
}

/// Width-8 windowed comb: `tbl[j][d] = d * 2^(8j) * G` (affine), j in 0..32,
/// d in 0..256. A scalar mult is then <=32 mixed Jacobian adds (one per nonzero
/// byte) instead of ~128.
struct Comb {
    tbl: Vec<[(U256, U256); 256]>, // [32][256]
}

impl Comb {
    fn new(gx: U256, gy: U256) -> Self {
        let inf = (U256::ZERO, U256::ZERO);
        let mut tbl: Vec<[(U256, U256); 256]> = vec![[inf; 256]; 32];
        // base_j = 2^(8j) * G, computed by repeated doubling.
        let mut base = Jac {
            x: gx,
            y: gy,
            z: U256::from(1u64),
        };
        for j in 0..32 {
            let base_aff = jac_to_affine(base);
            // table[j][d] = d * base_j, accumulated by affine add of base_j.
            tbl[j][0] = inf;
            tbl[j][1] = base_aff;
            for d in 2..256 {
                tbl[j][d] = affine_add(tbl[j][d - 1].0, tbl[j][d - 1].1, base_aff.0, base_aff.1);
            }
            // base_{j+1} = 2^8 * base_j
            for _ in 0..8 {
                base = jac_double(base);
            }
        }
        Comb { tbl }
    }

    /// k*G in affine, (0,0) for the identity.
    #[inline]
    fn mul(&self, k: U256) -> (U256, U256) {
        jac_to_affine(self.mul_jac(k))
    }

    /// k*G in Jacobian (no inversion); Z==0 for the identity.
    #[inline]
    fn mul_jac(&self, k: U256) -> Jac {
        let bytes = k.to_le_bytes::<32>();
        let mut acc = Jac::INF;
        for (j, &byte) in bytes.iter().enumerate() {
            if byte != 0 {
                let (x, y) = self.tbl[j][byte as usize];
                acc = jac_add_affine(acc, x, y);
            }
        }
        acc
    }
}

/// Montgomery batch inversion: replace each (nonzero) element with its inverse
/// mod P using a single field inversion + O(n) multiplications.
fn batch_invert(vals: &mut [U256], scratch: &mut Vec<U256>) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    scratch.clear();
    scratch.reserve(n);
    let mut acc = U256::from(1u64);
    for &v in vals.iter() {
        scratch.push(acc);
        acc = fmul(acc, v);
    }
    let mut inv = acc.inv_mod(P).expect("batch product invertible");
    for i in (0..n).rev() {
        let vi = vals[i];
        vals[i] = fmul(inv, scratch[i]);
        inv = fmul(inv, vi);
    }
}

// ───────────────────────── affine add (for R = P+Q) ─────────────────────────
// Reference-identical add; only used once per shot (rx needed for factor c).

fn affine_add(x1: U256, y1: U256, x2: U256, y2: U256) -> (U256, U256) {
    if x1.is_zero() && y1.is_zero() {
        return (x2, y2);
    }
    if x2.is_zero() && y2.is_zero() {
        return (x1, y1);
    }
    if x1 == x2 {
        if fadd(y1, y2).is_zero() {
            return (U256::ZERO, U256::ZERO);
        }
        // doubling
        let num = fadd(fmul(U256::from(3u64), fsqr(x1)), U256::ZERO);
        let den = fmul(U256::from(2u64), y1);
        let lambda = fmul(num, den.inv_mod(P).unwrap());
        let x3 = fsub(fsqr(lambda), fmul(U256::from(2u64), x1));
        let y3 = fsub(fmul(lambda, fsub(x1, x3)), y1);
        return (x3, y3);
    }
    let num = fsub(y2, y1);
    let den = fsub(x2, x1);
    let lambda = fmul(num, den.inv_mod(P).unwrap());
    let x3 = fsub(fsub(fsqr(lambda), x1), x2);
    let y3 = fsub(fmul(lambda, fsub(x1, x3)), y1);
    (x3, y3)
}

// ───────────────────────── Fiat-Shamir seed ─────────────────────────

/// Feed one op's 41 hash-bytes (kind u8 + 6 u64 LE) into the hasher, in the same
/// order as `eval_circuit::fiat_shamir_seed`. Only X ops in the tail are needed,
/// all-NO_QUBIT/NO_BIT/NO_REG except q_target.
#[inline]
fn feed_x_op(h: &mut Shake256, q_target: u64) {
    const NO: u64 = u64::MAX;
    h.update(&[6u8]); // OperationType::X
    h.update(&NO.to_le_bytes()); // q_control2
    h.update(&NO.to_le_bytes()); // q_control1
    h.update(&q_target.to_le_bytes()); // q_target
    h.update(&NO.to_le_bytes()); // c_target
    h.update(&NO.to_le_bytes()); // c_condition
    h.update(&NO.to_le_bytes()); // r_target
}

/// True iff nonce produces a filter-clean 9024-input set.
fn nonce_is_clean(
    base: &Shake256,
    tx0: u64,
    tx1: u64,
    nonce: u64,
    comb: &Comb,
    cfg: &DialogGcdFilterConfig,
) -> bool {
    let mut h = base.clone();
    for i in 0..NONCE_BITS {
        let q = if (nonce >> i) & 1 == 1 { tx1 } else { tx0 };
        feed_x_op(&mut h, q);
        feed_x_op(&mut h, q);
    }
    let mut xof = h.finalize_xof();

    // Chunked batch-inversion derivation: compute t=k1*G, o=k2*G in Jacobian for a
    // chunk, batch-invert all Z's at once (1 field inversion instead of 2/shot) to
    // get affine (t,o), then batch-invert the affine-add denominators (o.x - t.x)
    // to get e.x (1 more inversion instead of 1/shot). Early-exit between chunks.
    const CHUNK: usize = 256;
    let mut rb = [[0u8; 32]; 2];
    let mut tj = Vec::with_capacity(CHUNK);
    let mut oj = Vec::with_capacity(CHUNK);
    let mut zs: Vec<U256> = Vec::with_capacity(2 * CHUNK);
    let mut scratch: Vec<U256> = Vec::with_capacity(2 * CHUNK);
    // affine x,y of t and o for the chunk
    let mut ta: Vec<(U256, U256)> = Vec::with_capacity(CHUNK);
    let mut oa: Vec<(U256, U256)> = Vec::with_capacity(CHUNK);
    // per-shot: skip flag, and index into the denominator batch
    let mut dens: Vec<U256> = Vec::with_capacity(CHUNK);
    let mut keep: Vec<usize> = Vec::with_capacity(CHUNK);

    let mut remaining = NUM_TESTS;
    while remaining > 0 {
        let m = remaining.min(CHUNK);
        remaining -= m;
        tj.clear();
        oj.clear();
        zs.clear();
        for _ in 0..m {
            xof.read(&mut rb[0]);
            xof.read(&mut rb[1]);
            let k1 = U256::from_le_bytes(rb[0]);
            let k2 = U256::from_le_bytes(rb[1]);
            let t = comb.mul_jac(k1);
            let o = comb.mul_jac(k2);
            // collect Z's of non-infinity points for the batch
            if !t.is_inf() {
                zs.push(t.z);
            }
            if !o.is_inf() {
                zs.push(o.z);
            }
            tj.push(t);
            oj.push(o);
        }
        // batch-invert all Z's
        batch_invert(&mut zs, &mut scratch);
        // map back to affine
        ta.clear();
        oa.clear();
        let mut zi = 0usize;
        for s in 0..m {
            let t = tj[s];
            if t.is_inf() {
                ta.push((U256::ZERO, U256::ZERO));
            } else {
                let zinv = zs[zi];
                zi += 1;
                let zinv2 = fsqr(zinv);
                let zinv3 = fmul(zinv2, zinv);
                ta.push((fmul(t.x, zinv2), fmul(t.y, zinv3)));
            }
            let o = oj[s];
            if o.is_inf() {
                oa.push((U256::ZERO, U256::ZERO));
            } else {
                let zinv = zs[zi];
                zi += 1;
                let zinv2 = fsqr(zinv);
                let zinv3 = fmul(zinv2, zinv);
                oa.push((fmul(o.x, zinv2), fmul(o.y, zinv3)));
            }
        }
        // Circuit-structure quick filter: the circuit checks two dialog-GCD
        // factors per point-add input. The first one, dx=tx-ox, is known before
        // the affine-add slope/result is built. Reject dx-hard shots here, before
        // spending the denominator batch inversion needed only for the second
        // factor c=ox-rx.
        dens.clear();
        keep.clear();
        for s in 0..m {
            let (tx, ty) = ta[s];
            let (ox, oy) = oa[s];
            let _ = (ty, oy);
            if tx == ox {
                continue;
            }
            if tx.is_zero() && ta[s].1.is_zero() {
                continue;
            }
            if ox.is_zero() && oa[s].1.is_zero() {
                continue;
            }
            let dx = fsub(tx, ox);
            if check_gcd_factor(dx, cfg).is_err() {
                return false;
            }
            dens.push(fsub(ox, tx));
            keep.push(s);
        }
        batch_invert(&mut dens, &mut scratch);
        // check the second factor for each dx-clean kept shot
        for (di, &s) in keep.iter().enumerate() {
            let (tx, ty) = ta[s];
            let (ox, oy) = oa[s];
            let den_inv = dens[di];
            let lambda = fmul(fsub(oy, ty), den_inv);
            let ex = fsub(fsub(fsqr(lambda), tx), ox); // e.x = lambda^2 - tx - ox
            let (_dx, c) = point_add_gcd_factors(tx, ox, ex);
            if check_gcd_factor(c, cfg).is_err() {
                return false;
            }
        }
    }
    true
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <start_nonce> <count> [step]", args[0]);
        std::process::exit(2);
    }
    let start: u64 = args[1].parse().expect("start nonce");
    let count: u64 = args[2].parse().expect("count");
    let step: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let threads: usize = std::env::var("ISLAND_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // Build the op stream once with a tail present (nonce 0 -> all-tx0 tail), so
    // we can split off the 96-op tail and hash the prefix once. This also fills
    // the configure_ecdsafail_submission_route() env defaults the filter reads.
    std::env::set_var("DIALOG_TAIL_NONCE", "0");
    let ops = point_add::build();
    let n_ops = ops.len();
    assert!(n_ops > 96);
    let (_q, _b, _r, regs) = analyze_ops(ops.iter());
    let tx0 = match regs[0][0] {
        QubitOrBit::Qubit(q) => q.0,
        _ => panic!("reg0[0] not qubit"),
    };
    let tx1 = match regs[0][1] {
        QubitOrBit::Qubit(q) => q.0,
        _ => panic!("reg0[1] not qubit"),
    };

    // Sanity: the last 96 ops are the nonce-0 tail (all X on tx0).
    for op in &ops[n_ops - 96..] {
        assert_eq!(op.kind as u8, 6, "tail op not X");
        assert_eq!(op.q_target.0, tx0, "tail op not on tx0 at nonce 0");
    }

    // Hash domain + count + prefix (all ops EXCEPT the 96-op tail) once.
    let mut base = Shake256::default();
    base.update(DOMAIN);
    base.update(&(n_ops as u64).to_le_bytes());
    for op in &ops[..n_ops - 96] {
        base.update(&[op.kind as u8]);
        base.update(&op.q_control2.0.to_le_bytes());
        base.update(&op.q_control1.0.to_le_bytes());
        base.update(&op.q_target.0.to_le_bytes());
        base.update(&op.c_target.0.to_le_bytes());
        base.update(&op.c_condition.0.to_le_bytes());
        base.update(&op.r_target.0.to_le_bytes());
    }

    let cfg = DialogGcdFilterConfig::from_env();
    eprintln!(
        "filter cfg: active_iters={} compare_bits={} width_margin={} width_slope={:.4} k2={} odd_u={} var_w={} sched={}",
        cfg.active_iterations, cfg.compare_bits, cfg.width_margin, cfg.width_slope,
        cfg.k2, cfg.odd_u_lowbit_fastpath, cfg.variable_width, cfg.pa9024_compare_schedule
    );
    eprintln!(
        "n_ops={} tx0={} tx1={} threads={} scan [{}, {}) step {}",
        n_ops, tx0, tx1, threads, start, start + count * step, step
    );

    // G for the comb.
    let gx = U256::from_str_radix(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        16,
    )
    .unwrap();
    let gy = U256::from_str_radix(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
        16,
    )
    .unwrap();
    let comb = Arc::new(Comb::new(gx, gy));

    // Self-check the fast comb against the reference adder on a few scalars.
    {
        use quantum_ecc::weierstrass_elliptic_curve::WeierstrassEllipticCurve;
        let curve = WeierstrassEllipticCurve {
            modulus: P,
            a: U256::ZERO,
            b: U256::from(7u64),
            gx,
            gy,
            order: U256::from_str_radix(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
                16,
            )
            .unwrap(),
        };
        for s in [1u64, 2, 3, 7, 12345, 0xdead_beef, 0x1234_5678_9abc_def0] {
            let k = U256::from(s);
            assert_eq!(comb.mul(k), curve.mul(gx, gy, k), "comb mismatch at {}", s);
        }
        eprintln!("comb self-check OK");
    }

    let base = Arc::new(base);
    let cfg = Arc::new(cfg);
    let found = Arc::new(AtomicBool::new(false));
    let scanned = Arc::new(AtomicU64::new(0));
    let next = Arc::new(AtomicU64::new(0));
    const CHUNK: u64 = 256;

    let t0 = std::time::Instant::now();
    let mut handles = Vec::new();
    for _ in 0..threads {
        let base = Arc::clone(&base);
        let cfg = Arc::clone(&cfg);
        let comb = Arc::clone(&comb);
        let found = Arc::clone(&found);
        let scanned = Arc::clone(&scanned);
        let next = Arc::clone(&next);
        handles.push(std::thread::spawn(move || {
            loop {
                let chunk_start = next.fetch_add(CHUNK, Ordering::Relaxed);
                if chunk_start >= count {
                    break;
                }
                let chunk_end = (chunk_start + CHUNK).min(count);
                for idx in chunk_start..chunk_end {
                    let nonce = start + idx * step;
                    if nonce_is_clean(&base, tx0, tx1, nonce, &comb, &cfg) {
                        println!("CLEAN nonce={}", nonce);
                        found.store(true, Ordering::Relaxed);
                    }
                    let s = scanned.fetch_add(1, Ordering::Relaxed) + 1;
                    if s % 20000 == 0 {
                        let el = t0.elapsed().as_secs_f64();
                        eprintln!(
                            "  scanned {} in {:.0}s ({:.0} nonce/s)",
                            s, el, s as f64 / el
                        );
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let el = t0.elapsed().as_secs_f64();
    eprintln!(
        "done: scanned {} nonces in {:.1}s ({:.0} nonce/s); found_any={}",
        scanned.load(Ordering::Relaxed),
        el,
        scanned.load(Ordering::Relaxed) as f64 / el,
        found.load(Ordering::Relaxed)
    );
}
