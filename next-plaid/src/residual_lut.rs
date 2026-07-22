//! Asymmetric int8-query × LUT scoring for residual indexes.
//!
//! The residual float path decompresses every candidate token to `f32`
//! (`centroid[cid] + bucket_weights[code_d]`, then a renormalize) and runs a
//! BLAS MaxSim. This module scores the *stored* codes directly, the same
//! compute-only Stage-2 swap [`crate::binary`] made for sign indexes:
//!
//! ```text
//! q · token  =  q · centroid[cid]              (cdot — already computed by
//!                                               the search for IVF probing)
//!            +  Σ_d q_d · bucket_weights[code_d]  (int8 query × int8 LUT,
//!                                                  integer multiply-adds)
//! ```
//!
//! The int8 query touches only the small residual-correction term; the
//! dominant centroid term stays float, which is why the quality cost of the
//! asymmetric path measures at < 0.002 NDCG@10 across checkpoints and
//! corpora (exp/quant-grid LUT cells, 22 measured deltas, incl. long-query
//! ArguAna).
//!
//! Storage is untouched: a residual index already persists codes, packed
//! residuals, centroids and bucket weights. The path is selected per-search
//! via [`crate::search::SearchParameters::residual_asym`] — the same index
//! can be A/B'd with and without it.
//!
//! The float path L2-normalizes each decompressed token; this path applies
//! the identical normalization via a cached per-token `1/||recon||`
//! ([`compute_inv_norms`]) — measured as load-bearing (skipping it costs up
//! to 0.17 NDCG@10 at nbits=1). The one remaining delta vs the float path is
//! int8 quantization of the residual term (measured ≈ 0.001 NDCG@10).

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;

use crate::binary::QueryI8;
use crate::codec::ResidualCodec;

/// Highest embedding dim the fused expansion buffer supports (matches the
/// binary kernels' `fused_dim` ceiling).
pub const MAX_DIM: usize = 256;

/// The document-side lookup state for asymmetric residual scoring: one fused
/// table turning each packed residual *byte* directly into its `8/nbits`
/// int8 bucket weights.
///
/// The table composes the codec's own decode maps —
/// `byte_reversed_bits_map` (undoes the LSB-first-in-group bit packing of
/// `quantize_residuals`) then `bucket_weight_indices_lookup` (splits the
/// reversed byte into natural nbits groups) then the int8-quantized
/// `bucket_weights` — so it inherits the exact packing semantics of
/// [`ResidualCodec::decompress`] by construction.
pub struct ResidualLut {
    /// `[256 * keys_per_byte]` int8 weights, row `b` = expansion of byte `b`.
    pub fused: Vec<i8>,
    /// `8 / nbits`: how many dims one packed byte carries.
    pub keys_per_byte: usize,
    /// Dequantization scale: `fused as f32 * scale ≈ bucket_weights`.
    pub scale: f32,
}

/// Build the fused byte→weights table from a residual codec.
///
/// Returns `None` for codecs without bucket artifacts (binary indexes).
pub fn quantize_lut(codec: &ResidualCodec) -> Option<ResidualLut> {
    let weights = codec.bucket_weights.as_ref()?;
    let lookup = codec.bucket_weight_indices_lookup.as_ref()?;
    let max_abs = weights.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = (max_abs / 127.0).max(1e-12);
    let vals: Vec<i8> = weights
        .iter()
        .map(|&w| (w / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let keys_per_byte = 8 / codec.nbits;
    let mut fused = vec![0i8; 256 * keys_per_byte];
    for byte in 0..256usize {
        let reversed = codec.byte_reversed_bits_map[byte] as usize;
        for k in 0..keys_per_byte {
            fused[byte * keys_per_byte + k] = vals[lookup[[reversed, k]]];
        }
    }
    Some(ResidualLut {
        fused,
        keys_per_byte,
        scale,
    })
}

/// Per-token `1 / ||centroid + dequantized residual||` for a whole index —
/// the exact normalization [`ResidualCodec::decompress`] applies to every
/// reconstructed token (computed with the f32 bucket weights, so it
/// normalizes by the same quantity the float path does).
///
/// This is *derived* data: recomputable from the stored codes at any time,
/// cached once per index by `MmapIndex::residual_inv_norms`. Without it the
/// asymmetric path scores un-normalized reconstructions, whose per-token
/// norm spread MaxSim's argmax amplifies (measured: up to -0.17 NDCG@10 at
/// nbits=1 on long-query corpora).
pub fn compute_inv_norms(
    codec: &ResidualCodec,
    codes: &[i64],
    packed: &ArrayView2<u8>,
) -> Option<Vec<f32>> {
    let weights = codec.bucket_weights.as_ref()?;
    let lookup = codec.bucket_weight_indices_lookup.as_ref()?;
    let dim = codec.embedding_dim();
    Some(
        (0..codes.len())
            .into_par_iter()
            .map(|t| {
                let centroid = codec.centroids.row(codes[t] as usize);
                let mut sq = 0.0f32;
                let mut d = 0usize;
                'row: for &byte in packed.row(t).iter() {
                    let reversed = codec.byte_reversed_bits_map[byte as usize] as usize;
                    for &bi in lookup.row(reversed).iter() {
                        if d == dim {
                            break 'row;
                        }
                        let v = centroid[d] + weights[bi];
                        sq += v * v;
                        d += 1;
                    }
                }
                1.0 / sq.sqrt().max(1e-12)
            })
            .collect(),
    )
}

/// MaxSim of an int8 query against one document's stored residual codes.
///
/// * `doc_packed` — `[n_tokens, packed_dim]` packed residual rows (sliced
///   straight from the mmap, no decompression).
/// * `doc_codes` — the tokens' centroid ids.
/// * `cdot` — `[n_query_tokens, num_centroids]` query×centroid scores (the
///   dense matrix the search already computes for IVF probing).
///
/// Scalar reference implementation; the SIMD paths must match it exactly on
/// the integer accumulator (same contract as the binary kernels).
pub fn maxsim_residual_lut_scalar(
    q8: &QueryI8,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    debug_assert!(dim <= MAX_DIM);
    debug_assert_eq!(doc_packed.nrows(), doc_codes.len());
    let nq = q8.values.nrows();
    if nq == 0 || doc_packed.nrows() == 0 {
        return 0.0;
    }
    let qv = q8.values.as_slice().expect("QueryI8.values is contiguous");
    let mut best = vec![f32::NEG_INFINITY; nq];
    let mut w = [0i8; MAX_DIM];

    // Doc-token-outer: expand each stored token's bytes to int8 weights once,
    // amortized over all query tokens (the binary kernels' loop order).
    for (t, row) in doc_packed.axis_iter(Axis(0)).enumerate() {
        let mut d = 0usize;
        'expand: for &byte in row.iter() {
            let base = byte as usize * lut.keys_per_byte;
            for k in 0..lut.keys_per_byte {
                if d == dim {
                    break 'expand;
                }
                w[d] = lut.fused[base + k];
                d += 1;
            }
        }
        let cid = doc_codes[t] as usize;
        let inv = inv_norms[t];
        for (qi, best_q) in best.iter_mut().enumerate() {
            let qrow = &qv[qi * dim..(qi + 1) * dim];
            let mut acc = 0i32;
            for (qd, wd) in qrow.iter().zip(&w[..dim]) {
                acc += *qd as i32 * *wd as i32;
            }
            let score = (q8.scales[qi] * lut.scale * acc as f32 + cdot[[qi, cid]]) * inv;
            if score > *best_q {
                *best_q = score;
            }
        }
    }
    best.iter().sum()
}

/// Public entry: runtime-dispatched MaxSim over stored residual codes.
///
/// Uses the fused NEON SDOT path on aarch64 with `dotprod` for byte-aligned
/// dims ≤ [`MAX_DIM`]; otherwise the scalar reference. All paths compute the
/// identical integer accumulator, so results are bit-equal across dispatch.
pub fn maxsim_residual_lut_i8(
    q8: &QueryI8,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if dim.is_multiple_of(8)
            && dim <= MAX_DIM
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            return unsafe {
                neon::maxsim_residual_lut_neon(q8, doc_packed, doc_codes, cdot, lut, inv_norms, dim)
            };
        }
    }
    maxsim_residual_lut_scalar(q8, doc_packed, doc_codes, cdot, lut, inv_norms, dim)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Fused NEON path: expand each doc token's packed bytes once through the
    /// fused table into a zero-padded weights buffer, then score all query
    /// rows with SDOT against [`QueryI8::padded`] (whose zero lanes make the
    /// buffer's padding contribute nothing).
    ///
    /// # Safety
    /// Requires the `dotprod` CPU feature; `dim % 8 == 0 && dim <= MAX_DIM`.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn maxsim_residual_lut_neon(
        q8: &QueryI8,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot: &ArrayView2<f32>,
        lut: &ResidualLut,
        inv_norms: &[f32],
        dim: usize,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let ps = crate::binary::padded_stride(dim);
        let qp_base = q8.padded.as_ptr();
        let sqw: Vec<f32> = q8.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut w = [0i8; MAX_DIM];

        for (t, row) in doc_packed.axis_iter(Axis(0)).enumerate() {
            let mut d = 0usize;
            'expand: for &byte in row.iter() {
                let base = byte as usize * lut.keys_per_byte;
                for k in 0..lut.keys_per_byte {
                    if d == dim {
                        break 'expand;
                    }
                    w[d] = lut.fused[base + k];
                    d += 1;
                }
            }
            let cid = doc_codes[t] as usize;
            let inv = inv_norms[t];
            let wp = w.as_ptr();
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qp = qp_base.add(qi * ps);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                let mut k = 0usize;
                // Partial tail chunks are exact: both sides zero-pad past dim.
                while k < dim {
                    a = crate::binary::sdot_asm(a, vld1q_s8(qp.add(k)), vld1q_s8(wp.add(k)));
                    if k + 16 < dim {
                        b = crate::binary::sdot_asm(
                            b,
                            vld1q_s8(qp.add(k + 16)),
                            vld1q_s8(wp.add(k + 16)),
                        );
                    }
                    k += 32;
                }
                let acc = vaddvq_s32(vaddq_s32(a, b));
                let score = (sqw[qi] * acc as f32 + cdot[[qi, cid]]) * inv;
                if score > *best_qi {
                    *best_qi = score;
                }
            }
        }
        best.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Build a small residual codec with synthetic centroids and quantile
    /// buckets, mirroring the training in `index.rs`.
    fn toy_codec(dim: usize, nbits: usize, k: usize, rng: &mut StdRng) -> ResidualCodec {
        let centroids = Array2::from_shape_fn((k, dim), |_| rng.gen_range(-1.0f32..1.0));
        let residuals: Vec<f32> = (0..40_000).map(|_| rng.gen_range(-0.3f32..0.3)).collect();
        let mut sorted = residuals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n_options = 1usize << nbits;
        let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
        let cutoffs: Array1<f32> = (1..n_options)
            .map(|i| q(i as f64 / n_options as f64))
            .collect();
        let weights: Array1<f32> = (0..n_options)
            .map(|i| q((i as f64 + 0.5) / n_options as f64))
            .collect();
        ResidualCodec::new(
            nbits,
            centroids,
            Array1::zeros(dim),
            Some(cutoffs),
            Some(weights),
        )
        .unwrap()
    }

    /// The fused table must expand a packed byte to exactly the bucket
    /// weights `quantize_residuals` encoded — i.e. the composition through
    /// `byte_reversed_bits_map` and `bucket_weight_indices_lookup` matches
    /// an independent per-dim bucketing against the cutoffs.
    #[test]
    fn fused_table_matches_packing() {
        let mut rng = StdRng::seed_from_u64(7);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 48, 128] {
                let codec = toy_codec(dim, nbits, 16, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let cutoffs = codec.bucket_cutoffs.as_ref().unwrap();
                let weights = codec.bucket_weights.as_ref().unwrap();

                let res = Array2::from_shape_fn((5, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();

                for (row, pr) in res.axis_iter(Axis(0)).zip(packed.axis_iter(Axis(0))) {
                    // independent reference bucketing (strict >, as encode does)
                    let expect: Vec<i8> = row
                        .iter()
                        .map(|&v| {
                            let b = cutoffs.iter().filter(|&&c| v > c).count();
                            let w = weights[b];
                            (w / lut.scale).round().clamp(-127.0, 127.0) as i8
                        })
                        .collect();
                    let mut got = Vec::with_capacity(dim);
                    'row: for &byte in pr.iter() {
                        let base = byte as usize * lut.keys_per_byte;
                        for k in 0..lut.keys_per_byte {
                            if got.len() == dim {
                                break 'row;
                            }
                            got.push(lut.fused[base + k]);
                        }
                    }
                    assert_eq!(got, expect, "nbits={nbits} dim={dim}");
                }
            }
        }
    }

    /// The NEON path must equal the scalar reference bit-for-bit: both
    /// compute the identical integer accumulator, and the float epilogue is
    /// the same expression.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernel_matches_scalar_bitwise() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(23);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 16, 40, 48, 128, 200, 256] {
                let k = 12;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let query = Array2::from_shape_fn((7, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((13, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..13).map(|_| rng.gen_range(0..k as i64)).collect();
                let cdot = Array2::from_shape_fn((7, k), |_| rng.gen_range(-1.0f32..1.0));
                let inv: Vec<f32> = (0..13).map(|_| rng.gen_range(0.5f32..1.5)).collect();

                let scalar = maxsim_residual_lut_scalar(
                    &q8,
                    &packed.view(),
                    &codes,
                    &cdot.view(),
                    &lut,
                    &inv,
                    dim,
                );
                let neon = unsafe {
                    super::neon::maxsim_residual_lut_neon(
                        &q8,
                        &packed.view(),
                        &codes,
                        &cdot.view(),
                        &lut,
                        &inv,
                        dim,
                    )
                };
                assert_eq!(
                    scalar.to_bits(),
                    neon.to_bits(),
                    "nbits={nbits} dim={dim}: scalar {scalar} != neon {neon}"
                );
            }
        }
    }

    /// End-to-end decoder parity: the kernel with [`compute_inv_norms`] must
    /// approximate `Σ_q max_t q · (recon_t / ||recon_t||)` — i.e. exactly
    /// what the float path scores after `decompress` — with only int8
    /// residual rounding as the difference.
    #[test]
    fn normalized_scoring_matches_decompress_reference() {
        let mut rng = StdRng::seed_from_u64(31);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[48usize, 128] {
                let k = 8;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let weights = codec.bucket_weights.as_ref().unwrap();
                let lookup = codec.bucket_weight_indices_lookup.as_ref().unwrap();

                let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.3f32..0.3));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
                let cents = Array2::from_shape_fn((k, dim), |(i, d)| codec.centroids.row(i)[d]);
                let cdot = query.dot(&cents.t());
                let inv = compute_inv_norms(&codec, &codes, &packed.view()).unwrap();

                let got = maxsim_residual_lut_i8(
                    &q8,
                    &packed.view(),
                    &codes,
                    &cdot.view(),
                    &lut,
                    &inv,
                    dim,
                );

                // Reference: float query x exact normalized reconstruction.
                let mut expect = 0.0f64;
                for qi in 0..6 {
                    let mut best = f64::NEG_INFINITY;
                    for (t, &code) in codes.iter().enumerate() {
                        // exact reconstruction (decompress semantics)
                        let centroid = codec.centroids.row(code as usize);
                        let mut recon = vec![0.0f64; dim];
                        let mut d = 0usize;
                        'r: for &byte in packed.row(t).iter() {
                            let rev = codec.byte_reversed_bits_map[byte as usize] as usize;
                            for &bi in lookup.row(rev).iter() {
                                if d == dim {
                                    break 'r;
                                }
                                recon[d] = centroid[d] as f64 + weights[bi] as f64;
                                d += 1;
                            }
                        }
                        let norm = recon.iter().map(|v| v * v).sum::<f64>().sqrt();
                        let dot: f64 = (0..dim)
                            .map(|d| query[[qi, d]] as f64 * recon[d] / norm)
                            .sum();
                        best = best.max(dot);
                    }
                    expect += best;
                }
                assert!(
                    (got as f64 - expect).abs() < 0.05,
                    "nbits={nbits} dim={dim}: got {got} expect {expect}"
                );
            }
        }
    }

    /// The scalar kernel must equal a float reference computing
    /// `max_t [ scale·(q8 · w) + cdot[q, cid_t] ]` summed over query tokens.
    #[test]
    fn scalar_kernel_matches_float_reference() {
        let mut rng = StdRng::seed_from_u64(11);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 48, 128, 256] {
                let k = 8;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();

                let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
                let cdot = Array2::from_shape_fn((6, k), |_| rng.gen_range(-1.0f32..1.0));
                let inv: Vec<f32> = (0..9).map(|_| rng.gen_range(0.5f32..1.5)).collect();

                let got = maxsim_residual_lut_i8(
                    &q8,
                    &packed.view(),
                    &codes,
                    &cdot.view(),
                    &lut,
                    &inv,
                    dim,
                );

                // f64 reference over the same integers
                let mut expect = 0.0f64;
                for qi in 0..6 {
                    let mut best = f64::NEG_INFINITY;
                    for t in 0..9 {
                        let mut acc = 0i64;
                        let mut d = 0usize;
                        'e: for &byte in packed.row(t).iter() {
                            let base = byte as usize * lut.keys_per_byte;
                            for kk in 0..lut.keys_per_byte {
                                if d == dim {
                                    break 'e;
                                }
                                acc += q8.values[[qi, d]] as i64 * lut.fused[base + kk] as i64;
                                d += 1;
                            }
                        }
                        let s = (q8.scales[qi] as f64 * lut.scale as f64 * acc as f64
                            + cdot[[qi, codes[t] as usize]] as f64)
                            * inv[t] as f64;
                        best = best.max(s);
                    }
                    expect += best;
                }
                assert!(
                    (got as f64 - expect).abs() < 1e-3,
                    "nbits={nbits} dim={dim}: got {got} expect {expect}"
                );
            }
        }
    }
}
