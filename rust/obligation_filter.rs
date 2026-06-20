//! LOCAL TOOLING (not part of the submission).
//!
//! Exact manifest-driven prefilter for candidate nonces.
//!
//! The CUDA filters are intentionally narrow: they check the fast, stable GCD
//! obligations that can be ported once. This helper is the next layer up. It
//! derives the same Fiat-Shamir point-add inputs as `eval_circuit`, computes a
//! small set of stable classical values per shot, and evaluates a line-oriented
//! manifest of exact obligations emitted by, or manually audited against, a
//! circuit builder.
//!
//! This is deliberately conservative. Unknown obligation kinds are hard errors,
//! and the default manifest is empty. A manifest must only contain conditions
//! that a real clean circuit execution is guaranteed to satisfy.
//!
//! Usage:
//!   obligation_filter emit-default
//!   obligation_filter emit-dialog-gcd
//!   OBLIGATION_SHOTS=9024 obligation_filter check manifest.txt NONCE...

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
use std::fs;

const P: U256 = SECP256K1_P;
const NONCE_BITS: u32 = 48;
const NUM_TESTS: usize = 9024;
const DOMAIN: &[u8] = b"quantum_ecc-fiat-shamir-v2";

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
    let yy = fsqr(p.y);
    let s = fmul(U256::from(4u64), fmul(p.x, yy));
    let m = fmul(U256::from(3u64), fsqr(p.x));
    let x3 = fsub(fsqr(m), fadd(s, s));
    let yyyy = fsqr(yy);
    let y3 = fsub(fmul(m, fsub(s, x3)), fmul(U256::from(8u64), yyyy));
    let z3 = fmul(fadd(p.y, p.y), p.z);
    Jac {
        x: x3,
        y: y3,
        z: z3,
    }
}

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
        return Jac::INF;
    }
    let h = fsub(u2, u1);
    let r = fsub(s2, s1);
    let h2 = fsqr(h);
    let h3 = fmul(h2, h);
    let u1h2 = fmul(u1, h2);
    let x3 = fsub(fsub(fsqr(r), h3), fadd(u1h2, u1h2));
    let y3 = fsub(fmul(r, fsub(u1h2, x3)), fmul(s1, h3));
    let z3 = fmul(p.z, h);
    Jac {
        x: x3,
        y: y3,
        z: z3,
    }
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

struct Comb {
    tbl: Vec<[(U256, U256); 256]>,
}

impl Comb {
    fn new(gx: U256, gy: U256) -> Self {
        let inf = (U256::ZERO, U256::ZERO);
        let mut tbl: Vec<[(U256, U256); 256]> = vec![[inf; 256]; 32];
        let mut base = Jac {
            x: gx,
            y: gy,
            z: U256::from(1u64),
        };
        for j in 0..32 {
            let base_aff = jac_to_affine(base);
            tbl[j][0] = inf;
            tbl[j][1] = base_aff;
            for d in 2..256 {
                tbl[j][d] = affine_add(tbl[j][d - 1].0, tbl[j][d - 1].1, base_aff.0, base_aff.1);
            }
            for _ in 0..8 {
                base = jac_double(base);
            }
        }
        Comb { tbl }
    }

    #[inline]
    fn mul(&self, k: U256) -> (U256, U256) {
        let bytes = k.to_le_bytes::<32>();
        let mut acc = Jac::INF;
        for (j, &byte) in bytes.iter().enumerate() {
            if byte != 0 {
                let (x, y) = self.tbl[j][byte as usize];
                acc = jac_add_affine(acc, x, y);
            }
        }
        jac_to_affine(acc)
    }
}

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

#[inline]
fn feed_x_op(h: &mut Shake256, q_target: u64) {
    const NO: u64 = u64::MAX;
    h.update(&[6u8]);
    h.update(&NO.to_le_bytes());
    h.update(&NO.to_le_bytes());
    h.update(&q_target.to_le_bytes());
    h.update(&NO.to_le_bytes());
    h.update(&NO.to_le_bytes());
    h.update(&NO.to_le_bytes());
}

#[derive(Clone, Copy, Debug)]
enum ValueName {
    Tx,
    Ty,
    Ox,
    Oy,
    Rx,
    Ry,
    Dx,
    C,
}

impl ValueName {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "tx" => Ok(Self::Tx),
            "ty" => Ok(Self::Ty),
            "ox" => Ok(Self::Ox),
            "oy" => Ok(Self::Oy),
            "rx" => Ok(Self::Rx),
            "ry" => Ok(Self::Ry),
            "dx" => Ok(Self::Dx),
            "c" => Ok(Self::C),
            _ => Err(format!("unknown value '{s}'")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Tx => "tx",
            Self::Ty => "ty",
            Self::Ox => "ox",
            Self::Oy => "oy",
            Self::Rx => "rx",
            Self::Ry => "ry",
            Self::Dx => "dx",
            Self::C => "c",
        }
    }
}

struct ShotValues {
    tx: U256,
    ty: U256,
    ox: U256,
    oy: U256,
    rx: U256,
    ry: U256,
    dx: U256,
    c: U256,
}

impl ShotValues {
    fn get(&self, name: ValueName) -> U256 {
        match name {
            ValueName::Tx => self.tx,
            ValueName::Ty => self.ty,
            ValueName::Ox => self.ox,
            ValueName::Oy => self.oy,
            ValueName::Rx => self.rx,
            ValueName::Ry => self.ry,
            ValueName::Dx => self.dx,
            ValueName::C => self.c,
        }
    }
}

enum Obligation {
    GcdFactorFits {
        name: String,
        value: ValueName,
    },
    HighZero {
        name: String,
        value: ValueName,
        keep_bits: usize,
    },
    LowEq {
        name: String,
        left: ValueName,
        right: ValueName,
        bits: usize,
    },
    CompareWindowAgrees {
        name: String,
        left: ValueName,
        right: ValueName,
        lo: usize,
        width: usize,
    },
    AddNoCarry {
        name: String,
        left: ValueName,
        right: ValueName,
        bits: usize,
    },
    SubNoBorrow {
        name: String,
        left: ValueName,
        right: ValueName,
        bits: usize,
    },
    NonZero {
        name: String,
        value: ValueName,
    },
}

impl Obligation {
    fn name(&self) -> &str {
        match self {
            Self::GcdFactorFits { name, .. }
            | Self::HighZero { name, .. }
            | Self::LowEq { name, .. }
            | Self::CompareWindowAgrees { name, .. }
            | Self::AddNoCarry { name, .. }
            | Self::SubNoBorrow { name, .. }
            | Self::NonZero { name, .. } => name,
        }
    }

    fn check(&self, v: &ShotValues, cfg: &DialogGcdFilterConfig) -> Result<(), String> {
        match self {
            Self::GcdFactorFits { value, .. } => check_gcd_factor(v.get(*value), cfg)
                .map_err(|e| format!("{} does not fit dialog-GCD schedule: {e:?}", value.as_str())),
            Self::HighZero {
                value, keep_bits, ..
            } => {
                let x = v.get(*value);
                if fits_bits(x, *keep_bits) {
                    Ok(())
                } else {
                    Err(format!(
                        "{} has nonzero bits above {}",
                        value.as_str(),
                        keep_bits
                    ))
                }
            }
            Self::LowEq {
                left, right, bits, ..
            } => {
                let l = low_bits(v.get(*left), *bits);
                let r = low_bits(v.get(*right), *bits);
                if l == r {
                    Ok(())
                } else {
                    Err(format!(
                        "low {} bits differ for {} and {}",
                        bits,
                        left.as_str(),
                        right.as_str()
                    ))
                }
            }
            Self::CompareWindowAgrees {
                left,
                right,
                lo,
                width,
                ..
            } => {
                let l = v.get(*left);
                let r = v.get(*right);
                let full = l.cmp(&r);
                let wl = window_bits(l, *lo, *width);
                let wr = window_bits(r, *lo, *width);
                let narrowed = wl.cmp(&wr);
                if full == narrowed {
                    Ok(())
                } else {
                    Err(format!(
                        "window compare {}:{}+{} disagrees with full compare of {} and {}",
                        left.as_str(),
                        lo,
                        width,
                        left.as_str(),
                        right.as_str()
                    ))
                }
            }
            Self::AddNoCarry {
                left, right, bits, ..
            } => {
                let sum = low_bits(v.get(*left), *bits) + low_bits(v.get(*right), *bits);
                if fits_bits(sum, *bits) {
                    Ok(())
                } else {
                    Err(format!(
                        "{} + {} carries beyond {} bits",
                        left.as_str(),
                        right.as_str(),
                        bits
                    ))
                }
            }
            Self::SubNoBorrow {
                left, right, bits, ..
            } => {
                let l = low_bits(v.get(*left), *bits);
                let r = low_bits(v.get(*right), *bits);
                if l >= r {
                    Ok(())
                } else {
                    Err(format!(
                        "{} - {} borrows beyond {} low bits",
                        left.as_str(),
                        right.as_str(),
                        bits
                    ))
                }
            }
            Self::NonZero { value, .. } => {
                if v.get(*value).is_zero() {
                    Err(format!("{} is zero", value.as_str()))
                } else {
                    Ok(())
                }
            }
        }
    }
}

fn fits_bits(x: U256, bits: usize) -> bool {
    bits >= 256 || (x >> bits).is_zero()
}

fn low_bits(x: U256, bits: usize) -> U256 {
    if bits >= 256 {
        x
    } else if bits == 0 {
        U256::ZERO
    } else {
        x & ((U256::from(1u64) << bits) - U256::from(1u64))
    }
}

fn window_bits(x: U256, lo: usize, width: usize) -> U256 {
    if width == 0 || lo >= 256 {
        return U256::ZERO;
    }
    let w = width.min(256 - lo);
    low_bits(x >> lo, w)
}

fn parse_usize(s: &str, what: &str, line_no: usize) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("line {line_no}: invalid {what} '{s}'"))
}

fn require_len(parts: &[&str], want: usize, line_no: usize) -> Result<(), String> {
    if parts.len() == want {
        Ok(())
    } else {
        Err(format!(
            "line {line_no}: expected {want} fields for '{}', got {}",
            parts.first().copied().unwrap_or("?"),
            parts.len()
        ))
    }
}

fn parse_manifest(path: &str) -> Result<Vec<Obligation>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let obligation = match parts[0] {
            "gcd_factor_fits" => {
                require_len(&parts, 3, line_no)?;
                Obligation::GcdFactorFits {
                    name: parts[1].to_string(),
                    value: ValueName::parse(parts[2])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                }
            }
            "high_zero" => {
                require_len(&parts, 4, line_no)?;
                Obligation::HighZero {
                    name: parts[1].to_string(),
                    value: ValueName::parse(parts[2])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                    keep_bits: parse_usize(parts[3], "keep_bits", line_no)?,
                }
            }
            "low_eq" => {
                require_len(&parts, 5, line_no)?;
                Obligation::LowEq {
                    name: parts[1].to_string(),
                    left: ValueName::parse(parts[2]).map_err(|e| format!("line {line_no}: {e}"))?,
                    right: ValueName::parse(parts[3])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                    bits: parse_usize(parts[4], "bits", line_no)?,
                }
            }
            "compare_window_agrees" => {
                require_len(&parts, 6, line_no)?;
                Obligation::CompareWindowAgrees {
                    name: parts[1].to_string(),
                    left: ValueName::parse(parts[2]).map_err(|e| format!("line {line_no}: {e}"))?,
                    right: ValueName::parse(parts[3])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                    lo: parse_usize(parts[4], "lo", line_no)?,
                    width: parse_usize(parts[5], "width", line_no)?,
                }
            }
            "add_no_carry" => {
                require_len(&parts, 5, line_no)?;
                Obligation::AddNoCarry {
                    name: parts[1].to_string(),
                    left: ValueName::parse(parts[2]).map_err(|e| format!("line {line_no}: {e}"))?,
                    right: ValueName::parse(parts[3])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                    bits: parse_usize(parts[4], "bits", line_no)?,
                }
            }
            "sub_no_borrow" => {
                require_len(&parts, 5, line_no)?;
                Obligation::SubNoBorrow {
                    name: parts[1].to_string(),
                    left: ValueName::parse(parts[2]).map_err(|e| format!("line {line_no}: {e}"))?,
                    right: ValueName::parse(parts[3])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                    bits: parse_usize(parts[4], "bits", line_no)?,
                }
            }
            "nonzero" => {
                require_len(&parts, 3, line_no)?;
                Obligation::NonZero {
                    name: parts[1].to_string(),
                    value: ValueName::parse(parts[2])
                        .map_err(|e| format!("line {line_no}: {e}"))?,
                }
            }
            other => {
                return Err(format!(
                    "line {line_no}: unsupported obligation kind '{other}'"
                ))
            }
        };
        out.push(obligation);
    }
    Ok(out)
}

struct Context {
    base: Shake256,
    tx0: u64,
    tx1: u64,
    comb: Comb,
    cfg: DialogGcdFilterConfig,
}

fn build_context() -> Context {
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
    for op in &ops[n_ops - 96..] {
        assert_eq!(op.kind as u8, 6, "tail op not X");
        assert_eq!(op.q_target.0, tx0, "tail op not on tx0 at nonce 0");
    }

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
    let comb = Comb::new(gx, gy);
    let cfg = DialogGcdFilterConfig::from_env();
    Context {
        base,
        tx0,
        tx1,
        comb,
        cfg,
    }
}

fn nonce_xof(ctx: &Context, nonce: u64) -> impl XofReader {
    let mut h = ctx.base.clone();
    for i in 0..NONCE_BITS {
        let q = if (nonce >> i) & 1 == 1 {
            ctx.tx1
        } else {
            ctx.tx0
        };
        feed_x_op(&mut h, q);
        feed_x_op(&mut h, q);
    }
    h.finalize_xof()
}

fn shot_values(ctx: &Context, xof: &mut impl XofReader) -> ShotValues {
    let mut rb = [[0u8; 32]; 2];
    xof.read(&mut rb[0]);
    xof.read(&mut rb[1]);
    let k1 = U256::from_le_bytes(rb[0]);
    let k2 = U256::from_le_bytes(rb[1]);
    let (tx, ty) = ctx.comb.mul(k1);
    let (ox, oy) = ctx.comb.mul(k2);
    let (rx, ry) = affine_add(tx, ty, ox, oy);
    let (dx, c) = point_add_gcd_factors(tx, ox, rx);
    ShotValues {
        tx,
        ty,
        ox,
        oy,
        rx,
        ry,
        dx,
        c,
    }
}

struct Failure {
    shot: usize,
    obligation: String,
    reason: String,
}

fn check_nonce(
    ctx: &Context,
    obligations: &[Obligation],
    nonce: u64,
    shot_limit: usize,
) -> Result<usize, Failure> {
    let mut xof = nonce_xof(ctx, nonce);
    for shot in 0..shot_limit {
        let values = shot_values(ctx, &mut xof);
        for obligation in obligations {
            if let Err(reason) = obligation.check(&values, &ctx.cfg) {
                return Err(Failure {
                    shot,
                    obligation: obligation.name().to_string(),
                    reason,
                });
            }
        }
    }
    Ok(shot_limit)
}

fn print_default_manifest() {
    println!("# ecdsafail obligation manifest v1");
    println!("# Default is intentionally empty: it cannot reject any clean nonce.");
    println!("# Add only exact obligations emitted by or audited against the circuit builder.");
    println!("# Supported forms:");
    println!("#   gcd_factor_fits <name> <tx|ty|ox|oy|rx|ry|dx|c>");
    println!("#   high_zero <name> <value> <keep_bits>");
    println!("#   low_eq <name> <left> <right> <bits>");
    println!("#   compare_window_agrees <name> <left> <right> <lo> <width>");
    println!("#   add_no_carry <name> <left> <right> <bits>");
    println!("#   sub_no_borrow <name> <left> <right> <bits>");
    println!("#   nonzero <name> <value>");
}

fn print_dialog_gcd_manifest() {
    println!("# ecdsafail obligation manifest v1");
    println!("# Use only for dialog-GCD routes whose GCD schedule is represented by");
    println!("# DialogGcdFilterConfig. Do not use this manifest for unrelated jump-GCD");
    println!("# schedules unless you have verified known clean nonces.");
    println!("gcd_factor_fits gcd_dx dx");
    println!("gcd_factor_fits gcd_c c");
}

fn usage(program: &str) -> ! {
    eprintln!("usage:");
    eprintln!("  {program} emit-default");
    eprintln!("  {program} emit-dialog-gcd");
    eprintln!("  OBLIGATION_SHOTS=9024 {program} check MANIFEST NONCE...");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
    }
    match args[1].as_str() {
        "emit-default" => {
            print_default_manifest();
        }
        "emit-dialog-gcd" => {
            print_dialog_gcd_manifest();
        }
        "check" => {
            if args.len() < 4 {
                usage(&args[0]);
            }
            let manifest = match parse_manifest(&args[2]) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("manifest error: {e}");
                    std::process::exit(2);
                }
            };
            let shot_limit = std::env::var("OBLIGATION_SHOTS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(NUM_TESTS);
            if shot_limit > NUM_TESTS {
                eprintln!("OBLIGATION_SHOTS must be <= {NUM_TESTS}");
                std::process::exit(2);
            }
            if manifest.is_empty() {
                for nonce_s in &args[3..] {
                    let nonce: u64 = nonce_s.parse().expect("nonce");
                    println!("obligation-pass nonce={nonce} shots=0 obligations=0");
                }
                return;
            }
            let ctx = build_context();
            for nonce_s in &args[3..] {
                let nonce: u64 = nonce_s.parse().expect("nonce");
                match check_nonce(&ctx, &manifest, nonce, shot_limit) {
                    Ok(shots) => {
                        println!(
                            "obligation-pass nonce={nonce} shots={shots} obligations={}",
                            manifest.len()
                        );
                    }
                    Err(f) => {
                        println!(
                            "obligation-reject nonce={nonce} shot={} obligation={} reason={}",
                            f.shot, f.obligation, f.reason
                        );
                    }
                }
            }
        }
        _ => usage(&args[0]),
    }
}
