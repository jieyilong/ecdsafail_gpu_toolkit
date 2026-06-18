//! LOCAL TOOLING (not part of the submission).
//!
//! Dumps everything the CUDA island searcher needs to derive the 9024 Fiat-Shamir
//! inputs per nonce and run the dialog-GCD filter, bit-identically to the Rust
//! `island_search`:
//!   - the SHAKE256 prefix state (after absorbing domain+count+prefix ops), so the
//!     GPU only has to absorb the 96-op tail per nonce;
//!   - tx0/tx1 (the two tail-target qubit ids);
//!   - per-step (active_width, compare_bits, body_w) arrays (precomputed in f64
//!     here so the GPU never touches floats);
//!   - the width-8 comb table (32x256 affine points);
//!   - cfg bools (odd_u, k2, k2_force0) and scalars (active_iters, compare_bits).
//!
//! Also self-validates: my byte-oriented Keccak (resumed from the dumped state +
//! tail) reproduces the exact `sha3` XOF stream, and prints the ground-truth first
//! (k1,k2) for a probe nonce so the GPU can cross-check its derivation.

use alloy_primitives::U256;
use quantum_ecc::circuit::{analyze_ops, QubitOrBit};
use quantum_ecc::point_add::dialog_gcd_classical_filter::DialogGcdFilterConfig;
use quantum_ecc::point_add::{self, SECP256K1_P};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::io::Write;

const P: U256 = SECP256K1_P;
const DOMAIN: &[u8] = b"quantum_ecc-fiat-shamir-v2";

// ───── field ─────
fn fadd(a: U256, b: U256) -> U256 { a.add_mod(b, P) }
fn fsub(a: U256, b: U256) -> U256 { if a >= b { a - b } else { P - (b - a) } }
fn fmul(a: U256, b: U256) -> U256 { a.mul_mod(b, P) }
fn fsqr(a: U256) -> U256 { a.mul_mod(a, P) }

#[derive(Clone, Copy)]
struct Jac { x: U256, y: U256, z: U256 }
impl Jac {
    const INF: Jac = Jac { x: U256::ZERO, y: U256::ZERO, z: U256::ZERO };
    fn is_inf(&self) -> bool { self.z.is_zero() }
}
fn jac_double(p: Jac) -> Jac {
    if p.is_inf() || p.y.is_zero() { return Jac::INF; }
    let yy = fsqr(p.y);
    let s = fmul(U256::from(4u64), fmul(p.x, yy));
    let m = fmul(U256::from(3u64), fsqr(p.x));
    let x3 = fsub(fsqr(m), fadd(s, s));
    let yyyy = fsqr(yy);
    let y3 = fsub(fmul(m, fsub(s, x3)), fmul(U256::from(8u64), yyyy));
    let z3 = fmul(fadd(p.y, p.y), p.z);
    Jac { x: x3, y: y3, z: z3 }
}
fn jac_to_affine(p: Jac) -> (U256, U256) {
    if p.is_inf() { return (U256::ZERO, U256::ZERO); }
    let zinv = p.z.inv_mod(P).expect("z invertible");
    let zinv2 = fsqr(zinv);
    let zinv3 = fmul(zinv2, zinv);
    (fmul(p.x, zinv2), fmul(p.y, zinv3))
}
fn affine_add(x1: U256, y1: U256, x2: U256, y2: U256) -> (U256, U256) {
    if x1.is_zero() && y1.is_zero() { return (x2, y2); }
    if x2.is_zero() && y2.is_zero() { return (x1, y1); }
    if x1 == x2 {
        if fadd(y1, y2).is_zero() { return (U256::ZERO, U256::ZERO); }
        let num = fmul(U256::from(3u64), fsqr(x1));
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
fn build_comb(gx: U256, gy: U256) -> Vec<[(U256, U256); 256]> {
    let inf = (U256::ZERO, U256::ZERO);
    let mut tbl: Vec<[(U256, U256); 256]> = vec![[inf; 256]; 32];
    let mut base = Jac { x: gx, y: gy, z: U256::from(1u64) };
    for j in 0..32 {
        let ba = jac_to_affine(base);
        tbl[j][0] = inf;
        tbl[j][1] = ba;
        for d in 2..256 {
            tbl[j][d] = affine_add(tbl[j][d - 1].0, tbl[j][d - 1].1, ba.0, ba.1);
        }
        for _ in 0..8 { base = jac_double(base); }
    }
    tbl
}

// ───── byte-oriented Keccak / SHAKE256, matching the CUDA impl ─────
const RC: [u64; 24] = [
    0x0000000000000001,0x0000000000008082,0x800000000000808a,0x8000000080008000,
    0x000000000000808b,0x0000000080000001,0x8000000080008081,0x8000000000008009,
    0x000000000000008a,0x0000000000000088,0x0000000080008009,0x000000008000000a,
    0x000000008000808b,0x800000000000008b,0x8000000000008089,0x8000000000008003,
    0x8000000000008002,0x8000000000000080,0x000000000000800a,0x800000008000000a,
    0x8000000080008081,0x8000000000008080,0x0000000080000001,0x8000000080008008];
const ROTC: [u32; 24] = [1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44];
const PILN: [usize; 24] = [10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1];
fn keccakf(st: &mut [u64; 25]) {
    for round in 0..24 {
        let mut bc = [0u64; 5];
        for i in 0..5 { bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20]; }
        for i in 0..5 {
            let t = bc[(i+4)%5] ^ bc[(i+1)%5].rotate_left(1);
            let mut j = 0;
            while j < 25 { st[j+i] ^= t; j += 5; }
        }
        let mut t = st[1];
        for i in 0..24 {
            let j = PILN[i];
            let tmp = st[j];
            st[j] = t.rotate_left(ROTC[i]);
            t = tmp;
        }
        let mut j = 0;
        while j < 25 {
            let mut tb = [0u64; 5];
            for i in 0..5 { tb[i] = st[j+i]; }
            for i in 0..5 { st[j+i] ^= (!tb[(i+1)%5]) & tb[(i+2)%5]; }
            j += 5;
        }
        st[0] ^= RC[round];
    }
}
const RATE: usize = 136;
struct Keccak { st: [u64; 25], pt: usize }
impl Keccak {
    fn new() -> Self { Keccak { st: [0u64; 25], pt: 0 } }
    fn absorb(&mut self, data: &[u8]) {
        for &b in data {
            self.st[self.pt / 8] ^= (b as u64) << ((self.pt % 8) * 8);
            self.pt += 1;
            if self.pt == RATE { keccakf(&mut self.st); self.pt = 0; }
        }
    }
    fn finalize_squeeze(&self, out: &mut [u8]) {
        let mut st = self.st;
        st[self.pt / 8] ^= 0x1Fu64 << ((self.pt % 8) * 8);
        st[(RATE - 1) / 8] ^= 0x80u64 << (((RATE - 1) % 8) * 8);
        keccakf(&mut st);
        let mut p = 0usize;
        for o in out.iter_mut() {
            if p == RATE { keccakf(&mut st); p = 0; }
            *o = ((st[p / 8] >> ((p % 8) * 8)) & 0xff) as u8;
            p += 1;
        }
    }
}
fn feed_x_op_bytes(k: &mut Keccak, q_target: u64) {
    const NO: u64 = u64::MAX;
    k.absorb(&[6u8]);
    k.absorb(&NO.to_le_bytes());
    k.absorb(&NO.to_le_bytes());
    k.absorb(&q_target.to_le_bytes());
    k.absorb(&NO.to_le_bytes());
    k.absorb(&NO.to_le_bytes());
    k.absorb(&NO.to_le_bytes());
}

fn main() {
    std::env::set_var("DIALOG_TAIL_NONCE", "0");
    let ops = point_add::build();
    let n_ops = ops.len();
    assert!(n_ops > 96);
    let (_q, _b, _r, regs) = analyze_ops(ops.iter());
    let tx0 = match regs[0][0] { QubitOrBit::Qubit(q) => q.0, _ => panic!("reg0[0]") };
    let tx1 = match regs[0][1] { QubitOrBit::Qubit(q) => q.0, _ => panic!("reg0[1]") };

    // sha3 prefix (ground truth) and my-keccak prefix (state to dump).
    let mut sha_base = Shake256::default();
    let mut my = Keccak::new();
    // domain + count
    sha_base.update(DOMAIN);
    sha_base.update(&(n_ops as u64).to_le_bytes());
    my.absorb(DOMAIN);
    my.absorb(&(n_ops as u64).to_le_bytes());
    for op in &ops[..n_ops - 96] {
        let mut bytes = Vec::with_capacity(49);
        bytes.push(op.kind as u8);
        bytes.extend_from_slice(&op.q_control2.0.to_le_bytes());
        bytes.extend_from_slice(&op.q_control1.0.to_le_bytes());
        bytes.extend_from_slice(&op.q_target.0.to_le_bytes());
        bytes.extend_from_slice(&op.c_target.0.to_le_bytes());
        bytes.extend_from_slice(&op.c_condition.0.to_le_bytes());
        bytes.extend_from_slice(&op.r_target.0.to_le_bytes());
        sha_base.update(&bytes);
        my.absorb(&bytes);
    }
    eprintln!("prefix absorbed: n_ops={} my.pt={}", n_ops, my.pt);

    let cfg = DialogGcdFilterConfig::from_env();
    eprintln!(
        "cfg: active_iters={} compare_bits={} margin={} slope={:.6} k2={} odd_u={} k2f0={} var_w={} strict={} trunc_w={} trims={:?}",
        cfg.active_iterations, cfg.compare_bits, cfg.width_margin, cfg.width_slope,
        cfg.k2, cfg.odd_u_lowbit_fastpath, cfg.k2_force0, cfg.variable_width,
        cfg.strict_compare, cfg.body_carry_trunc_w, cfg.body_carry_trims
    );

    // Per-step arrays.
    let ai = cfg.active_iterations;
    let mut aw = vec![0u32; ai];
    let mut cb = vec![0u32; ai];
    let mut bw = vec![0u32; ai];
    for step in 0..ai {
        let a = cfg.active_width(step);
        aw[step] = a as u32;
        cb[step] = cfg.compare_bits_for_step(step, a) as u32;
        bw[step] = cfg.body_carry_trunc_width(a, step) as u32;
    }

    // Comb.
    let gx = U256::from_str_radix("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16).unwrap();
    let gy = U256::from_str_radix("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16).unwrap();
    let tbl = build_comb(gx, gy);

    // ── Validate my-keccak == sha3, resumed from prefix + tail(probe) ──
    let probe: u64 = 264497;
    // sha3 ground truth
    let mut sha = sha_base.clone();
    for i in 0..48u32 {
        let q = if (probe >> i) & 1 == 1 { tx1 } else { tx0 };
        for _ in 0..2 {
            const NO: u64 = u64::MAX;
            sha.update(&[6u8]);
            sha.update(&NO.to_le_bytes());
            sha.update(&NO.to_le_bytes());
            sha.update(&q.to_le_bytes());
            sha.update(&NO.to_le_bytes());
            sha.update(&NO.to_le_bytes());
            sha.update(&NO.to_le_bytes());
        }
    }
    let mut xof = sha.finalize_xof();
    let mut g64 = [0u8; 64];
    xof.read(&mut g64);
    // my-keccak resumed
    let mut myk = Keccak { st: my.st, pt: my.pt };
    for i in 0..48u32 {
        let q = if (probe >> i) & 1 == 1 { tx1 } else { tx0 };
        feed_x_op_bytes(&mut myk, q);
        feed_x_op_bytes(&mut myk, q);
    }
    let mut m64 = [0u8; 64];
    myk.finalize_squeeze(&mut m64);
    assert_eq!(g64, m64, "my-keccak XOF != sha3 XOF for probe nonce {}", probe);
    eprintln!("KECCAK MATCH: my-keccak resumed-from-state == sha3 (64 bytes), probe={}", probe);
    let k1 = U256::from_le_bytes(<[u8; 32]>::try_from(&g64[0..32]).unwrap());
    let k2 = U256::from_le_bytes(<[u8; 32]>::try_from(&g64[32..64]).unwrap());
    eprintln!("probe first k1={:x}\nprobe first k2={:x}", k1, k2);

    // ── write dump ──
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/gpu_state.bin".into());
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let w32 = |f: &mut dyn Write, v: u32| f.write_all(&v.to_le_bytes()).unwrap();
    let w64 = |f: &mut dyn Write, v: u64| f.write_all(&v.to_le_bytes()).unwrap();
    let wu256 = |f: &mut dyn Write, v: U256| f.write_all(&v.to_le_bytes::<32>()).unwrap();
    w32(&mut f, 0x47505531); // magic "GPU1"
    w64(&mut f, n_ops as u64);
    w64(&mut f, tx0);
    w64(&mut f, tx1);
    w32(&mut f, my.pt as u32);
    for &s in &my.st { w64(&mut f, s); }
    w32(&mut f, cfg.odd_u_lowbit_fastpath as u32);
    w32(&mut f, cfg.k2 as u32);
    w32(&mut f, cfg.k2_force0 as u32);
    w32(&mut f, cfg.active_iterations as u32);
    w32(&mut f, cfg.compare_bits as u32);
    for step in 0..ai { w32(&mut f, aw[step]); }
    for step in 0..ai { w32(&mut f, cb[step]); }
    for step in 0..ai { w32(&mut f, bw[step]); }
    for j in 0..32 {
        for d in 0..256 {
            wu256(&mut f, tbl[j][d].0);
            wu256(&mut f, tbl[j][d].1);
        }
    }
    // probe cross-check block
    w64(&mut f, probe);
    wu256(&mut f, k1);
    wu256(&mut f, k2);
    f.flush().unwrap();
    eprintln!("wrote {} ({} bytes header+arrays+comb)", path, "?");
    eprintln!("DONE");
}
