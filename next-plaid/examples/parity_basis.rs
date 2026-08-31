//! Parity gate: does the NumPy reference used by the basis study agree with the
//! deployed binary kernel?
//!
//! Rule 7 exists because the int8 query quantizer silently dropped its per-token
//! scale for the entire programme — a defect that changed no shape and raised no
//! error, and was only found by reading the production path beside the reference
//! one. The Householder basis fix is applied *upstream* of this kernel (the kernel
//! only ever sees already-rotated embeddings), so the transform itself needs no
//! kernel change. What DOES need checking is whether the retention numbers the
//! study reports would reproduce in production at all.
//!
//! Two concrete divergence risks were identified by reading both paths:
//!   1. Rust rounds with `f32::round` (half AWAY FROM ZERO); NumPy's `rint`
//!      rounds half to EVEN. Codes can differ by 1 on exact .5 ties.
//!   2. Rust accumulates the query·doc dot in exact `i32` and applies the row
//!      scale once; the reference dequantises per element into f32 first, so it
//!      carries f32 rounding through the whole sum.
//!
//! Emits deterministic inputs (a shared LCG, so Python reproduces the SAME bits)
//! and the kernel's scores as JSON. `parity_basis.py` recomputes the reference and
//! compares.

use ndarray::Array2;
use next_plaid::binary::{binarize, maxsim_binary_i8, packed_dim, quantize_query_i8};

/// Same 32-bit LCG on both sides, so the two implementations see identical bits.
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // numerical recipes constants; take the top 24 bits as a mantissa
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

fn l2_normalize(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        row.iter_mut().for_each(|x| *x /= n);
    }
}

fn main() {
    // Dimensions chosen to hit several kernel paths: 48 is the crux model's dim
    // and is byte-aligned; 128 is the fleet's common dim and hits the fused path.
    for &dim in &[48u32, 96, 128] {
        let n_doc = 512usize;
        let n_q = 8usize;
        let mut rng = Lcg(0x1234_5678 ^ dim);

        let mut docs = Array2::<f32>::zeros((n_doc, dim as usize));
        for v in docs.iter_mut() {
            *v = rng.next_f32();
        }
        // add a shared offset so the cloud looks like a real ColBERT token cloud
        let mut off = vec![0.0f32; dim as usize];
        for o in off.iter_mut() {
            *o = rng.next_f32();
        }
        let onorm = off.iter().map(|x| x * x).sum::<f32>().sqrt();
        for o in off.iter_mut() {
            *o = *o / onorm * 2.0;
        }
        for mut row in docs.rows_mut() {
            for (d, v) in row.iter_mut().enumerate() {
                *v += off[d];
            }
        }
        l2_normalize(&mut docs);

        let mut queries = Array2::<f32>::zeros((n_q, dim as usize));
        for v in queries.iter_mut() {
            *v = rng.next_f32();
        }
        l2_normalize(&mut queries);

        let packed = binarize(&docs.view());
        let qi8 = quantize_query_i8(&queries.view());
        let score = maxsim_binary_i8(&qi8, &packed.view(), dim as usize);

        println!(
            "{{\"dim\":{},\"n_doc\":{},\"n_q\":{},\"packed_dim\":{},\"score\":{:.9},\
             \"scales\":[{}],\"codes_row0\":[{}]}}",
            dim,
            n_doc,
            n_q,
            packed_dim(dim as usize),
            score,
            qi8.scales
                .iter()
                .map(|s| format!("{s:.9}"))
                .collect::<Vec<_>>()
                .join(","),
            qi8.values
                .row(0)
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
