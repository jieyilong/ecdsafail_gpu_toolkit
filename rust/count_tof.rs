//! Count emitted CCX (Toffoli) in the built circuit for the current config.
use quantum_ecc::point_add;
use quantum_ecc::circuit::OperationType;
fn main() {
    let ops = point_add::build();
    let mut ccx = 0usize; let mut ccz = 0usize;
    for op in &ops {
        match op.kind { OperationType::CCX => ccx += 1, OperationType::CCZ => ccz += 1, _ => {} }
    }
    println!("n_ops={} CCX={} CCZ={}", ops.len(), ccx, ccz);
}
