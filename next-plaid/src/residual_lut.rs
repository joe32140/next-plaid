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
    /// Nibble-factored form of `fused` for the SIMD expand paths.
    pub nibble: Option<NibbleLut>,
}

/// The fused table factored per key position into 16-entry nibble tables —
/// the shape NEON `tbl` / SSE `pshufb` consume (one in-register lookup per
/// key position per 16 packed bytes, instead of a scalar walk over dims).
///
/// Codes are `nbits ∈ {1,2,4}` wide and bit-packing never crosses a nibble
/// boundary, so key `k` of a packed byte is a function of exactly one of its
/// nibbles. [`derive_nibble_lut`] builds each table from `fused` and then
/// *verifies* the factorization over all 256 bytes, so the SIMD paths can
/// never silently diverge from the scalar reference's table.
pub struct NibbleLut {
    /// Per key position: weights indexed by the source nibble's value.
    pub tables: [[i8; 16]; 8],
    /// Whether key `k` reads the byte's high nibble (else the low one).
    pub from_hi: [bool; 8],
}

/// Factor `fused` into per-key nibble tables; `None` if any key position is
/// not a function of a single nibble (never for the current codec — this
/// guards future packing changes by failing back to the scalar path).
fn derive_nibble_lut(fused: &[i8], keys_per_byte: usize) -> Option<NibbleLut> {
    let mut tables = [[0i8; 16]; 8];
    let mut from_hi = [false; 8];
    for k in 0..keys_per_byte {
        let hi: [i8; 16] = std::array::from_fn(|x| fused[(x << 4) * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == hi[b >> 4]) {
            tables[k] = hi;
            from_hi[k] = true;
            continue;
        }
        let lo: [i8; 16] = std::array::from_fn(|x| fused[x * keys_per_byte + k]);
        if (0..256).all(|b| fused[b * keys_per_byte + k] == lo[b & 15]) {
            tables[k] = lo;
            from_hi[k] = false;
            continue;
        }
        return None;
    }
    Some(NibbleLut { tables, from_hi })
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
    let nibble = derive_nibble_lut(&fused, keys_per_byte);
    Some(ResidualLut {
        fused,
        keys_per_byte,
        scale,
        nibble,
    })
}

/// The int8 query permuted to *plane order*: plane `k` holds the dims byte
/// position `i` carries at key `k` (`d = i·keys_per_byte + k`), so the SIMD
/// expand can store each `tbl`/`pshufb` result contiguously instead of
/// interleaving back to dim order. A dot product is permutation-invariant
/// and the integer accumulator is order-invariant, so scores stay bit-equal
/// to the scalar reference. Rows are zero-padded to
/// [`crate::binary::padded_stride`] lanes (a multiple of both the NEON
/// 16-lane and AVX2 32-lane chunk widths); padding contributes `q·0 = 0`.
pub struct QueryPlanes {
    pub data: Vec<i8>,
    pub stride: usize,
}

/// Build [`QueryPlanes`] from already-quantized query codes. `dim` must be a
/// multiple of 8 (the SIMD dispatch precondition), so every plane holds
/// exactly `dim / keys_per_byte` lanes.
pub fn build_query_planes(q8: &QueryI8, keys_per_byte: usize, dim: usize) -> QueryPlanes {
    let nq = q8.values.nrows();
    let stride = crate::binary::padded_stride(dim);
    let pdim = dim / keys_per_byte;
    let qv = q8.values.as_slice().expect("QueryI8.values is contiguous");
    let mut data = vec![0i8; nq * stride];
    for qi in 0..nq {
        let row = &qv[qi * dim..(qi + 1) * dim];
        let out = &mut data[qi * stride..qi * stride + dim];
        for i in 0..pdim {
            for k in 0..keys_per_byte {
                out[k * pdim + i] = row[i * keys_per_byte + k];
            }
        }
    }
    QueryPlanes { data, stride }
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
/// With `planes` (and a nibble-factorable table) byte-aligned dims ≤
/// [`MAX_DIM`] take a fused SIMD path — `tbl`+SDOT on aarch64 with
/// `dotprod`, `pshufb`+`maddubs` on x86_64 with AVX2; otherwise the scalar
/// reference. All paths compute the identical integer accumulator, so
/// results are bit-equal across dispatch.
pub fn maxsim_residual_lut_i8(
    q8: &QueryI8,
    planes: Option<&QueryPlanes>,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = planes;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let (Some(planes), Some(nib)) = (planes, lut.nibble.as_ref()) {
        if dim.is_multiple_of(8) && dim <= MAX_DIM {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return unsafe {
                    neon::maxsim_residual_lut_neon(
                        q8, planes, doc_packed, doc_codes, cdot, lut, nib, inv_norms, dim,
                    )
                };
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                return unsafe {
                    avx2::maxsim_residual_lut_avx2(
                        q8, planes, doc_packed, doc_codes, cdot, lut, nib, inv_norms, dim,
                    )
                };
            }
        }
    }
    maxsim_residual_lut_scalar(q8, doc_packed, doc_codes, cdot, lut, inv_norms, dim)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Fused NEON path: expand each doc token's packed bytes through the
    /// nibble tables with `tbl` — one lookup per key position per 16 packed
    /// bytes, stored straight to that key's plane (no interleave) — then
    /// score all query rows with SDOT against the matching
    /// [`QueryPlanes`] rows (whose zero padding makes the buffer's padding
    /// contribute nothing).
    ///
    /// # Safety
    /// Requires the `dotprod` CPU feature; `dim % 8 == 0 && dim <= MAX_DIM`.
    #[target_feature(enable = "dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_neon(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot: &ArrayView2<f32>,
        lut: &ResidualLut,
        nib: &NibbleLut,
        inv_norms: &[f32],
        dim: usize,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb; // packed bytes per token (dim % 8 == 0)
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot.as_slice().expect("cdot must be standard layout");
        let ncent = cdot.ncols();
        let sqw: Vec<f32> = q8.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut w = [0i8; MAX_DIM];
        let mut tabs = [vdupq_n_s8(0); 8];
        for k in 0..kpb {
            tabs[k] = vld1q_s8(nib.tables[k].as_ptr());
        }
        let low_mask = vdupq_n_u8(0x0F);

        for (t, &code) in doc_codes.iter().enumerate() {
            let row = &d_all[t * pb..t * pb + pb];
            let wp = w.as_mut_ptr();
            let mut i = 0usize;
            while i + 16 <= pdim {
                let v = vld1q_u8(row.as_ptr().add(i));
                let hi = vshrq_n_u8(v, 4);
                let lo = vandq_u8(v, low_mask);
                for k in 0..kpb {
                    let idx = if nib.from_hi[k] { hi } else { lo };
                    vst1q_s8(wp.add(k * pdim + i), vqtbl1q_s8(tabs[k], idx));
                }
                i += 16;
            }
            // Sub-16 tail: pad the remaining packed bytes into a zeroed
            // 16-byte scratch, expand with the same tbl, and copy out only
            // the valid lanes — a direct 16-lane store would clobber the
            // next plane's already-written low bytes. This keeps narrow
            // dims on the SIMD path (dim 48 at nbits 2/1 packs to 12/6
            // bytes — under one chunk — and previously fell to a scalar
            // walk). Bit-identical: the nibble tables are verified against
            // the fused table over all 256 byte values, zero-pad included.
            if i < pdim {
                let rem = pdim - i;
                let mut src = [0u8; 16];
                src[..rem].copy_from_slice(&row[i..pdim]);
                let v = vld1q_u8(src.as_ptr());
                let hi = vshrq_n_u8(v, 4);
                let lo = vandq_u8(v, low_mask);
                let mut dst = [0i8; 16];
                for k in 0..kpb {
                    let idx = if nib.from_hi[k] { hi } else { lo };
                    vst1q_s8(dst.as_mut_ptr(), vqtbl1q_s8(tabs[k], idx));
                    w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
                }
            }
            let cid = code as usize;
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
                let score = (sqw[qi] * acc as f32 + cd[qi * ncent + cid]) * inv;
                if score > *best_qi {
                    *best_qi = score;
                }
            }
        }
        best.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Fused AVX2 path, mirroring the NEON kernel: `pshufb` nibble-table
    /// expansion into plane order, then a 32-lane `maddubs`/`madd` int8 dot
    /// against the [`QueryPlanes`] rows.
    ///
    /// Exactness: both operands are clamped to ±127 at quantization, so
    /// `_mm256_sign_epi8` never sees −128 and each `maddubs` pair-sum is
    /// bounded by 2·127·127 < i16::MAX — the i32 accumulator is exact.
    ///
    /// # Safety
    /// Requires AVX2; `dim % 8 == 0 && dim <= MAX_DIM`.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_avx2(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot: &ArrayView2<f32>,
        lut: &ResidualLut,
        nib: &NibbleLut,
        inv_norms: &[f32],
        dim: usize,
    ) -> f32 {
        let nq = q8.values.nrows();
        if nq == 0 || doc_packed.nrows() == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb;
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot.as_slice().expect("cdot must be standard layout");
        let ncent = cdot.ncols();
        let sqw: Vec<f32> = q8.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut w = [0i8; MAX_DIM];
        let mut tabs = [_mm_setzero_si128(); 8];
        for k in 0..kpb {
            tabs[k] = _mm_loadu_si128(nib.tables[k].as_ptr() as *const __m128i);
        }
        let low_mask = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);

        for (t, &code) in doc_codes.iter().enumerate() {
            let row = &d_all[t * pb..t * pb + pb];
            let wp = w.as_mut_ptr();
            let mut i = 0usize;
            while i + 16 <= pdim {
                let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
                let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                let lo = _mm_and_si128(v, low_mask);
                for k in 0..kpb {
                    let idx = if nib.from_hi[k] { hi } else { lo };
                    _mm_storeu_si128(
                        wp.add(k * pdim + i) as *mut __m128i,
                        _mm_shuffle_epi8(tabs[k], idx),
                    );
                }
                i += 16;
            }
            // Sub-16 tail: same padded-scratch expand as the NEON kernel —
            // see the comment there. Keeps narrow dims on pshufb instead of
            // a scalar walk; copy-out of only the valid lanes protects the
            // next plane's low bytes.
            if i < pdim {
                let rem = pdim - i;
                let mut src = [0u8; 16];
                src[..rem].copy_from_slice(&row[i..pdim]);
                let v = _mm_loadu_si128(src.as_ptr() as *const __m128i);
                let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
                let lo = _mm_and_si128(v, low_mask);
                let mut dst = [0i8; 16];
                for k in 0..kpb {
                    let idx = if nib.from_hi[k] { hi } else { lo };
                    _mm_storeu_si128(
                        dst.as_mut_ptr() as *mut __m128i,
                        _mm_shuffle_epi8(tabs[k], idx),
                    );
                    w[k * pdim + i..k * pdim + pdim].copy_from_slice(&dst[..rem]);
                }
            }
            let cid = code as usize;
            let inv = inv_norms[t];
            let wp = w.as_ptr();
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qp = qp_base.add(qi * ps);
                let mut acc = _mm256_setzero_si256();
                let mut k = 0usize;
                // Partial tail chunks are exact: both sides zero-pad past dim
                // (w to MAX_DIM, query rows to their 64-lane stride).
                while k < dim {
                    let qv = _mm256_loadu_si256(qp.add(k) as *const __m256i);
                    let wv = _mm256_loadu_si256(wp.add(k) as *const __m256i);
                    let prod =
                        _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(qv, wv));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
                    k += 32;
                }
                let hi128 = _mm256_extracti128_si256(acc, 1);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
                let acc = _mm_cvtsi128_si32(s32);
                let score = (sqw[qi] * acc as f32 + cd[qi * ncent + cid]) * inv;
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

    /// The fused table must factor into per-key nibble tables for every
    /// nbits — the precondition of both SIMD expand paths.
    #[test]
    fn nibble_factorization_holds() {
        let mut rng = StdRng::seed_from_u64(3);
        for &nbits in &[1usize, 2, 4] {
            let codec = toy_codec(64, nbits, 8, &mut rng);
            let lut = quantize_lut(&codec).unwrap();
            let nib = lut
                .nibble
                .as_ref()
                .unwrap_or_else(|| panic!("nbits={nbits}: fused table not nibble-separable"));
            for b in 0..256usize {
                for k in 0..lut.keys_per_byte {
                    let nibble = if nib.from_hi[k] { b >> 4 } else { b & 15 };
                    assert_eq!(
                        lut.fused[b * lut.keys_per_byte + k],
                        nib.tables[k][nibble],
                        "nbits={nbits} byte={b} key={k}"
                    );
                }
            }
        }
    }

    /// Every SIMD path must equal the scalar reference bit-for-bit: all
    /// compute the identical integer accumulator, and the float epilogue is
    /// the same expression.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn simd_kernel_matches_scalar_bitwise() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(23);
        for &nbits in &[1usize, 2, 4] {
            for &dim in &[8usize, 16, 40, 48, 128, 200, 256] {
                let k = 12;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();
                let nib = lut.nibble.as_ref().expect("nibble tables");
                let query = Array2::from_shape_fn((7, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let planes = build_query_planes(&q8, lut.keys_per_byte, dim);
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
                #[cfg(target_arch = "aarch64")]
                let simd = unsafe {
                    super::neon::maxsim_residual_lut_neon(
                        &q8,
                        &planes,
                        &packed.view(),
                        &codes,
                        &cdot.view(),
                        &lut,
                        nib,
                        &inv,
                        dim,
                    )
                };
                #[cfg(target_arch = "x86_64")]
                let simd = unsafe {
                    super::avx2::maxsim_residual_lut_avx2(
                        &q8,
                        &planes,
                        &packed.view(),
                        &codes,
                        &cdot.view(),
                        &lut,
                        nib,
                        &inv,
                        dim,
                    )
                };
                assert_eq!(
                    scalar.to_bits(),
                    simd.to_bits(),
                    "nbits={nbits} dim={dim}: scalar {scalar} != simd {simd}"
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
                let planes = build_query_planes(&q8, lut.keys_per_byte, dim);

                let got = maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
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
                let planes = build_query_planes(&q8, lut.keys_per_byte, dim);

                let got = maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
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
