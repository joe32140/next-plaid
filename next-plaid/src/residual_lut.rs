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
//! asymmetric path measures at < 0.002 NDCG@10 (3 ColBERT checkpoints × 3
//! BEIR corpora × nbits 4/2/1, incl. long-query ArguAna).
//!
//! New residual indexes also persist one `f32` inverse reconstruction norm per
//! token. Search memory-maps that sidecar, so it adds file-backed storage but
//! no whole-index heap allocation. Legacy indexes without the sidecar remain
//! supported and compute norms for shortlisted documents on demand. This is
//! the residual scoring path — the float decompress→MaxSim code remains only
//! as the automatic fallback for shapes the kernels cannot serve.
//!
//! The float path L2-normalizes each decompressed token; this path applies
//! the identical normalization via `1/||recon||` values — measured as
//! load-bearing (skipping it costs up to 0.17 NDCG@10 at nbits=1). The one
//! remaining delta vs the float path is int8 quantization of the residual
//! term (measured ≈ 0.001 NDCG@10).

use ndarray::{ArrayView2, Axis};

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
/// nibbles. `derive_nibble_lut` builds each table from `fused` and then
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
/// `crate::binary::padded_stride` lanes, a multiple of every kernel's chunk
/// width; padding contributes `q·0 = 0`. The NEON kernel reads this layout
/// directly; on x86 it is the source the `tiles` rearrangement is built
/// from.
pub struct QueryPlanes {
    pub data: Vec<i8>,
    pub stride: usize,
    /// Per query row: `q8.scales[qi] * lut.scale` — the query-constant
    /// factor the fold applies to each integer accumulator, built once per
    /// query rather than once per scored document.
    pub sqw: Vec<f32>,
    /// The same plane-order codes as *unsigned* GEMM tiles, the layout the
    /// x86 kernels' `u8 × s8` dot instructions want:
    /// `[⌈nq/16⌉ tiles][stride/4 groups][16 rows][4 bytes]`, each byte
    /// `code + 128`. One 64-byte load is then 16 query rows × 4 consecutive
    /// plane dims, which the kernel multiplies against a 4-byte broadcast of
    /// the token's weights — so an accumulator lane *is* one row's running
    /// sum and no horizontal reduction is ever needed. Rows past `nq` hold
    /// 128 (code 0) and contribute nothing.
    ///
    /// The `+128` offset is exact in integer arithmetic: AVX-512 subtracts
    /// `128·Σw` once per token, and AVX2 recovers the signed code with
    /// `xor 0x80` (so it needs no second query buffer).
    ///
    /// Built only on x86_64. Preparing a query is on the interactive latency
    /// path, and an aarch64 host has no kernel that reads this.
    #[cfg(target_arch = "x86_64")]
    pub tiles: Vec<u8>,
}

/// Build [`QueryPlanes`] from already-quantized query codes. `dim` must be a
/// multiple of 8 (the SIMD dispatch precondition), so every plane holds
/// exactly `dim / lut.keys_per_byte` lanes.
pub fn build_query_planes(q8: &QueryI8, lut: &ResidualLut, dim: usize) -> QueryPlanes {
    let nq = q8.values.nrows();
    let keys_per_byte = lut.keys_per_byte;
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
    #[cfg(target_arch = "x86_64")]
    let tiles = {
        let d4n = stride / 4;
        let n16 = nq.div_ceil(16);
        let mut tiles = vec![128u8; n16 * d4n * 64];
        for qi in 0..nq {
            let (t, r) = (qi / 16, qi % 16);
            for d in 0..dim {
                let v = data[qi * stride + d];
                tiles[(t * d4n + d / 4) * 64 + r * 4 + (d % 4)] = (v as i16 + 128) as u8;
            }
        }
        tiles
    };
    let sqw = q8.scales.iter().map(|&s| s * lut.scale).collect();
    QueryPlanes {
        data,
        stride,
        sqw,
        #[cfg(target_arch = "x86_64")]
        tiles,
    }
}

/// Per-token `1 / ||centroid + dequantized residual||` for a document —
/// the exact normalization [`ResidualCodec::decompress`] applies to every
/// reconstructed token (computed with the f32 bucket weights, so it
/// normalizes by the same quantity the float path does).
///
/// Index creation uses this for the mmap sidecar. Legacy indexes call the
/// reusable-buffer variant below only for shortlisted documents. Without the
/// normalization the asymmetric path scores un-normalized reconstructions,
/// whose per-token norm spread MaxSim's argmax amplifies (measured: up to
/// -0.17 NDCG@10 at nbits=1 on long-query corpora).
pub fn compute_inv_norms(
    codec: &ResidualCodec,
    codes: &[i64],
    packed: &ArrayView2<u8>,
) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(codes.len());
    compute_inv_norms_into(codec, codes, packed, &mut out)?;
    Some(out)
}

/// Fill a reusable inverse-norm buffer for one shortlisted document.
/// Sequential evaluation avoids nested Rayon overhead at document lengths;
/// document-level scoring already runs across the Rayon pool.
pub(crate) fn compute_inv_norms_into(
    codec: &ResidualCodec,
    codes: &[i64],
    packed: &ArrayView2<u8>,
    out: &mut Vec<f32>,
) -> Option<()> {
    let weights = codec.bucket_weights.as_ref()?;
    let lookup = codec.bucket_weight_indices_lookup.as_ref()?;
    let dim = codec.embedding_dim();
    assert_eq!(codes.len(), packed.nrows());
    out.clear();
    out.reserve(codes.len());
    for (t, &code) in codes.iter().enumerate() {
        let centroid = codec.centroids.row(code as usize);
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
        out.push(1.0 / sq.sqrt().max(1e-12));
    }
    Some(())
}

/// MaxSim of an int8 query against one document's stored residual codes.
///
/// * `doc_packed` — `[n_tokens, packed_dim]` packed residual rows (sliced
///   straight from the mmap, no decompression).
/// * `doc_codes` — the tokens' centroid ids.
/// * `cdot_t` — `[num_centroids, n_query_tokens]` query×centroid scores,
///   **centroid-major**: one centroid's scores across all query rows are
///   contiguous, so the vectorized fold loads them as one vector (and one
///   doc token touches one small contiguous strip instead of `nq` loads
///   scattered `num_centroids` apart — the search transposes its stage-1
///   matrix once per query to pay for this).
///
/// Scalar reference implementation; the SIMD paths must match it exactly on
/// the integer accumulator (same contract as the binary kernels).
pub fn maxsim_residual_lut_scalar(
    q8: &QueryI8,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot_t: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    assert!(dim <= MAX_DIM, "dim {dim} exceeds MAX_DIM {MAX_DIM}");
    assert_eq!(doc_packed.nrows(), doc_codes.len());
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
            let score = (q8.scales[qi] * lut.scale * acc as f32 + cdot_t[[cid, qi]]) * inv;
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
/// `dotprod`, `pshufb`+`maddubs` on x86_64 with AVX2, `vpdpbusd` with
/// AVX-512 VNNI; otherwise the scalar reference. `cdot_t` is centroid-major
/// (see [`maxsim_residual_lut_scalar`]). Each kernel blocks over several doc
/// tokens at once and folds 4 (NEON), 8 (AVX2) or 16 (AVX-512) query rows
/// per step, but every path computes the identical integer accumulator and
/// applies the identical float epilogue expression, so results are bit-equal
/// across dispatch.
#[allow(clippy::too_many_arguments)]
pub fn maxsim_residual_lut_i8(
    q8: &QueryI8,
    planes: Option<&QueryPlanes>,
    doc_packed: &ArrayView2<u8>,
    doc_codes: &[i64],
    cdot_t: &ArrayView2<f32>,
    lut: &ResidualLut,
    inv_norms: &[f32],
    dim: usize,
) -> f32 {
    // This is a safe public entry over kernels that do raw pointer loads, so
    // every precondition the SIMD paths rely on is a hard assert here — a
    // shape mismatch or out-of-range centroid id must panic like the
    // ndarray-indexed scalar path, never read out of bounds. One pass over
    // the doc's codes is noise next to the scoring work.
    let nq = q8.values.nrows();
    assert!(dim <= MAX_DIM, "dim {dim} exceeds MAX_DIM {MAX_DIM}");
    assert_eq!(
        cdot_t.ncols(),
        nq,
        "cdot_t must be centroid-major [num_centroids, n_query_tokens]"
    );
    assert_eq!(q8.scales.len(), nq, "QueryI8 scales/values row mismatch");
    assert_eq!(doc_packed.nrows(), doc_codes.len(), "packed rows != codes");
    assert_eq!(inv_norms.len(), doc_codes.len(), "inv_norms != codes");
    assert!(
        doc_packed.ncols() >= dim.div_ceil(lut.keys_per_byte),
        "packed row too short for dim {dim} at {} keys/byte",
        lut.keys_per_byte
    );
    let ncent = cdot_t.nrows() as u64;
    for &c in doc_codes {
        // A negative i64 wraps to a huge u64 and fails the same check.
        assert!((c as u64) < ncent, "centroid id {c} out of range {ncent}");
    }
    if let Some(p) = planes {
        // The kernels read whole SIMD chunks past `dim` (up to 64 lanes on
        // AVX-512), relying on the row padding `build_query_planes` provides
        // — so the stride floor is the padded stride, not `dim`. A
        // hand-built QueryPlanes with `stride == dim` would pass a weaker
        // check and read out of bounds.
        assert!(
            p.stride >= crate::binary::padded_stride(dim) && p.data.len() >= nq * p.stride,
            "QueryPlanes stride/len too small for nq {nq} x dim {dim}"
        );
        assert_eq!(p.sqw.len(), nq, "QueryPlanes sqw/rows mismatch");
        // The x86 kernels read the tile layout instead of `data`, so its
        // extent is a precondition of its own.
        #[cfg(target_arch = "x86_64")]
        assert!(
            p.tiles.len() >= nq.div_ceil(16) * (p.stride / 4) * 64,
            "QueryPlanes tiles too small for nq {nq} x dim {dim}"
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = planes;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let (Some(planes), Some(nib)) = (planes, lut.nibble.as_ref()) {
        if dim.is_multiple_of(8) {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return SCRATCH.with(|s| {
                    let best = &mut *s.borrow_mut();
                    unsafe {
                        neon::maxsim_residual_lut_neon(
                            q8, planes, doc_packed, doc_codes, cdot_t, lut, nib, inv_norms, dim,
                            best,
                        )
                    }
                });
            }
            #[cfg(target_arch = "x86_64")]
            if has_avx512_vnni() {
                return SCRATCH.with(|s| {
                    let best = &mut *s.borrow_mut();
                    unsafe {
                        avx512::maxsim_residual_lut_avx512(
                            q8, planes, doc_packed, doc_codes, cdot_t, lut, nib, inv_norms, dim,
                            best,
                        )
                    }
                });
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                return SCRATCH.with(|s| {
                    let best = &mut *s.borrow_mut();
                    unsafe {
                        avx2::maxsim_residual_lut_avx2(
                            q8, planes, doc_packed, doc_codes, cdot_t, lut, nib, inv_norms, dim,
                            best,
                        )
                    }
                });
            }
        }
    }
    maxsim_residual_lut_scalar(q8, doc_packed, doc_codes, cdot_t, lut, inv_norms, dim)
}

/// Does this CPU have the full AVX-512 set the fused kernel needs?
#[cfg(target_arch = "x86_64")]
fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
}

/// Name of the kernel this process will actually run, for benchmark output.
/// A speedup attributed to a path that never executed is the easiest
/// measurement error to make and the hardest to notice, so harnesses print
/// this next to their numbers.
pub fn active_kernel_name(dim: usize, nibble_ok: bool) -> &'static str {
    if !nibble_ok || !dim.is_multiple_of(8) || dim > MAX_DIM {
        return "scalar (no SIMD dispatch)";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512_vnni() {
            return "avx512-vnni";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return "neon-sdot";
        }
    }
    "scalar"
}

/// Will the fused SIMD kernel actually run for this shape on this CPU?
///
/// Mirrors the dispatch in [`maxsim_residual_lut_i8`], so callers can report
/// the outcome without re-deriving — and re-deriving it wrongly — the
/// condition. Every way of answering `false` here is silent and still returns
/// correct scores; the only symptom is that the rescore speedup collapses to
/// roughly the scalar kernel's margin over float.
pub fn simd_dispatch_available(dim: usize, nibble_ok: bool) -> bool {
    if !nibble_ok || !dim.is_multiple_of(8) || dim > MAX_DIM {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_avx512_vnni() || is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// Per-thread `best` buffer, reused across the ~1024 per-candidate kernel
// calls of a search. Each rayon worker gets its own copy, and the kernels
// size-and-initialize it on entry, so no state leaks between calls. The
// integer accumulators live in registers (see the kernels' `block`), so this
// is the only scratch they need.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Document tokens held in registers per block. Four tokens × four query
    /// rows is 16 accumulators, which sits well inside the 32 NEON registers
    /// with four query chunks and one weight chunk live. Neighbouring widths
    /// (2, 3, 6, 8) measure within ±8% of this one.
    const NT: usize = 4;

    /// Fold four query rows' final integer accumulators (lane `r` = row
    /// `base + r`) through the shared float tail.
    ///
    /// Bit-identical to the scalar kernel's tail on purpose:
    /// `vcvtq_f32_s32` is the same round-to-nearest as `acc as f32`; the
    /// multiply and add stay SEPARATE (`vmulq` then `vaddq`, never fused)
    /// and the `inv` multiply comes last, matching the scalar
    /// `(sqw·acc + crow) · inv` rounding-for-rounding; and for the finite
    /// scores this loop produces, `vmaxq_f32(best, s)` equals the scalar
    /// `if s > best` select.
    #[inline(always)]
    unsafe fn fold4(
        accv: int32x4_t,
        base: usize,
        sqw: &[f32],
        crow: *const f32,
        inv: f32,
        best: &mut [f32],
    ) {
        let a = vcvtq_f32_s32(accv);
        let s = vmulq_f32(
            vaddq_f32(
                vmulq_f32(vld1q_f32(sqw.as_ptr().add(base)), a),
                vld1q_f32(crow.add(base)),
            ),
            vdupq_n_f32(inv),
        );
        let b = vld1q_f32(best.as_ptr().add(base));
        vst1q_f32(best.as_mut_ptr().add(base), vmaxq_f32(b, s));
    }

    /// Scalar fold for the `nq % 4` leftover rows — same expression, same
    /// order as [`fold4`] and the scalar kernel.
    #[inline(always)]
    unsafe fn fold_tail(accs: &[i32], sqw: &[f32], crow: *const f32, inv: f32, best: &mut [f32]) {
        for (i, &acc) in accs.iter().enumerate() {
            let s = (sqw[i] * acc as f32 + *crow.add(i)) * inv;
            if s > best[i] {
                best[i] = s;
            }
        }
    }

    /// Expand one stored token's `pdim` packed bytes into `kpb` planes of
    /// int8 weights: one `tbl` per key position per 16 packed bytes, stored
    /// straight to that key's plane (no interleave back to dim order).
    ///
    /// # Safety
    /// `row.len() >= pdim`; `kpb * pdim <= MAX_DIM`; `tabs` holds the
    /// nibble tables for `kpb` keys. Writes land strictly below
    /// `kpb * pdim == dim`, which is what keeps `w`'s tail zero for the dot
    /// loop's chunk over-read.
    #[inline(always)]
    unsafe fn expand(
        row: &[u8],
        tabs: &[int8x16_t; 8],
        nib: &NibbleLut,
        kpb: usize,
        pdim: usize,
        w: &mut [i8; MAX_DIM],
    ) {
        let low_mask = vdupq_n_u8(0x0F);
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= pdim {
            let v = vld1q_u8(row.as_ptr().add(i));
            let hi = vshrq_n_u8(v, 4);
            let lo = vandq_u8(v, low_mask);
            for (k, tab) in tabs.iter().enumerate().take(kpb) {
                let idx = if nib.from_hi[k] { hi } else { lo };
                vst1q_s8(wp.add(k * pdim + i), vqtbl1q_s8(*tab, idx));
            }
            i += 16;
        }
        // Sub-16 tail: pad the remaining packed bytes into a zeroed 16-byte
        // scratch, expand with the same tbl, and copy out only the valid
        // lanes — a direct 16-lane store would clobber the next plane's
        // already-written low bytes. This keeps narrow dims on the SIMD path
        // (dim 48 at nbits 2/1 packs to 12/6 bytes, under one chunk).
        // Bit-identical: the nibble tables are verified against the fused
        // table over all 256 byte values, zero-pad included.
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
    }

    /// Score `N` already-expanded doc tokens against every query row,
    /// updating `best`.
    ///
    /// Register blocking is what this buys over a token-at-a-time loop: with
    /// one token live, every `sdot` needs two loads (a query chunk and a
    /// weight chunk), which caps a three-load-per-cycle core near 1.5
    /// `sdot`/cycle. Holding `N` tokens' expanded weights and four query
    /// rows in registers shares each query load across `N` products and each
    /// weight load across four rows — `4 + N` loads per `4·N` `sdot`s — and
    /// hands the out-of-order core `4·N` independent accumulator chains
    /// instead of four.
    ///
    /// The integer accumulator is unchanged: the same products are summed,
    /// and integer addition is exact and associative, so a different
    /// grouping cannot move a bit. The float epilogue is untouched. Hence
    /// the blocked form is bit-identical to the unblocked one, and to
    /// scalar.
    ///
    /// # Safety
    /// Requires `dotprod`; `dim % 8 == 0 && dim <= MAX_DIM`; the query planes
    /// zero-padded to a 16-lane multiple past `dim` (`padded_stride` is a
    /// multiple of 64); `crows[j]` readable for `nq` f32s; `best.len() == nq`
    /// and `sqw.len() == nq`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn block<const N: usize>(
        ws: &[[i8; MAX_DIM]; N],
        dim: usize,
        nq: usize,
        qp_base: *const i8,
        ps: usize,
        sqw: &[f32],
        crows: &[*const f32; N],
        invs: &[f32; N],
        best: &mut [f32],
    ) {
        let mut qi = 0usize;
        while qi + 4 <= nq {
            let mut acc = [[vdupq_n_s32(0); 4]; N];
            let qp = [
                qp_base.add(qi * ps),
                qp_base.add((qi + 1) * ps),
                qp_base.add((qi + 2) * ps),
                qp_base.add((qi + 3) * ps),
            ];
            let mut k = 0usize;
            // Partial tail chunks are exact: both sides zero-pad past dim.
            while k < dim {
                let q = [
                    vld1q_s8(qp[0].add(k)),
                    vld1q_s8(qp[1].add(k)),
                    vld1q_s8(qp[2].add(k)),
                    vld1q_s8(qp[3].add(k)),
                ];
                for j in 0..N {
                    let wv = vld1q_s8(ws[j].as_ptr().add(k));
                    for r in 0..4 {
                        acc[j][r] = crate::binary::sdot_asm(acc[j][r], q[r], wv);
                    }
                }
                k += 16;
            }
            for j in 0..N {
                // Pairwise tree -> [Σrow0, Σrow1, Σrow2, Σrow3] in one
                // register, instead of four per-row `vaddvq` reduces.
                let accv = vpaddq_s32(
                    vpaddq_s32(acc[j][0], acc[j][1]),
                    vpaddq_s32(acc[j][2], acc[j][3]),
                );
                fold4(accv, qi, sqw, crows[j], invs[j], best);
            }
            qi += 4;
        }
        if qi < nq {
            let rem = nq - qi;
            let mut tail = [0i32; 4];
            for j in 0..N {
                for (r, slot) in tail.iter_mut().enumerate().take(rem) {
                    let qp = qp_base.add((qi + r) * ps);
                    let mut acc0 = vdupq_n_s32(0);
                    let mut acc1 = vdupq_n_s32(0);
                    let mut k = 0usize;
                    while k < dim {
                        acc0 = crate::binary::sdot_asm(
                            acc0,
                            vld1q_s8(qp.add(k)),
                            vld1q_s8(ws[j].as_ptr().add(k)),
                        );
                        if k + 16 < dim {
                            acc1 = crate::binary::sdot_asm(
                                acc1,
                                vld1q_s8(qp.add(k + 16)),
                                vld1q_s8(ws[j].as_ptr().add(k + 16)),
                            );
                        }
                        k += 32;
                    }
                    *slot = vaddvq_s32(vaddq_s32(acc0, acc1));
                }
                fold_tail(
                    &tail[..rem],
                    &sqw[qi..],
                    crows[j].add(qi),
                    invs[j],
                    &mut best[qi..],
                );
            }
        }
    }

    /// Fused NEON path: expand `NT` doc tokens' packed bytes through the
    /// nibble tables, then score that block of tokens against every query
    /// row with SDOT against the matching [`QueryPlanes`] rows (whose zero
    /// padding makes the buffer's padding contribute nothing).
    ///
    /// # Safety
    /// Requires the `dotprod` CPU feature; `dim % 8 == 0 && dim <= MAX_DIM`;
    /// `planes.stride >= padded_stride(dim)` (the dot loop reads 16-byte
    /// chunks past `dim`, into the rows' zero padding).
    #[target_feature(enable = "dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_neon(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        nib: &NibbleLut,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        let n_tokens = doc_packed.nrows();
        if nq == 0 || n_tokens == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb; // packed bytes per token (dim % 8 == 0)
        let ps = planes.stride;
        let qp_base = planes.data.as_ptr();
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        let mut tabs = [vdupq_n_s8(0); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = vld1q_s8(src.as_ptr());
        }

        // Zeroed once: `expand` writes only below `dim`, so the tail of each
        // buffer stays zero for every token and the dot loop's over-read past
        // `dim` contributes exactly nothing.
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut crows = [std::ptr::null::<f32>(); NT];
        let mut invs = [0f32; NT];
        let mut t = 0usize;
        while t + NT <= n_tokens {
            for j in 0..NT {
                let tt = t + j;
                expand(
                    &d_all[tt * pb..tt * pb + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                crows[j] = cd.as_ptr().add(doc_codes[tt] as usize * nq);
                invs[j] = inv_norms[tt];
            }
            block::<NT>(&ws, dim, nq, qp_base, ps, sqw, &crows, &invs, best);
            t += NT;
        }
        while t < n_tokens {
            expand(
                &d_all[t * pb..t * pb + pdim],
                &tabs,
                nib,
                kpb,
                pdim,
                &mut ws[0],
            );
            let one_w = [ws[0]];
            let one_c = [cd.as_ptr().add(doc_codes[t] as usize * nq)];
            let one_i = [inv_norms[t]];
            block::<1>(&one_w, dim, nq, qp_base, ps, sqw, &one_c, &one_i, best);
            t += 1;
        }
        best.iter().sum()
    }
}

/// AVX2 path for the fused asym kernel, in GEMM-tile form.
///
/// The row-per-accumulator form pays an 8-lane horizontal reduction per
/// *(query row, doc token)* — the most frequent event in the kernel. Putting
/// the query *rows* into the lanes removes it entirely: a 32-byte load of the
/// tile layout ([`QueryPlanes::tiles`]) is 8 rows × 4 consecutive plane dims,
/// the token's expanded weights are broadcast 4 bytes at a time, and each
/// accumulator lane then *is* one row's running sum — so the fold reads the
/// lanes directly.
///
/// The only AVX2 `u8 × s8` dot is `vpmaddubsw`, whose pair sums saturate at
/// i16, so the `+128` offset the AVX-512 path relies on is unavailable here
/// (`(q+128)·w` pairs overflow). Instead the unsigned operand is `|q|` and
/// q's sign is moved onto the broadcast weight with `vpsignb`:
/// `|q| · sign(q)·w = q·w` exactly, and `|q|,|w| <= 127` keeps every pair sum
/// under 32 767. The signed code is recovered from the stored `u8` tile by
/// `xor 0x80`, so the query needs no second buffer; `-128` never appears
/// (codes are clamped to ±127 at quantization, and padding rows store 128 =
/// code 0), so `vpabsb` is exact.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Document tokens held in registers per block.
    const NT: usize = 4;

    /// Expand one stored token's `pdim` packed bytes into `kpb` planes of
    /// int8 weights with `pshufb` — the x86 twin of the NEON `expand`.
    ///
    /// # Safety
    /// `row.len() >= pdim`; `kpb * pdim <= MAX_DIM`; `tabs` holds the nibble
    /// tables for `kpb` keys. Writes land strictly below `kpb * pdim == dim`.
    #[inline(always)]
    unsafe fn expand(
        row: &[u8],
        tabs: &[__m128i; 8],
        nib: &NibbleLut,
        kpb: usize,
        pdim: usize,
        w: &mut [i8; MAX_DIM],
    ) {
        let low_mask = _mm_set1_epi8(0x0F);
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= pdim {
            let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
            let lo = _mm_and_si128(v, low_mask);
            for (k, tab) in tabs.iter().enumerate().take(kpb) {
                let idx = if nib.from_hi[k] { hi } else { lo };
                _mm_storeu_si128(
                    wp.add(k * pdim + i) as *mut __m128i,
                    _mm_shuffle_epi8(*tab, idx),
                );
            }
            i += 16;
        }
        // Sub-16 tail through a zero-padded scratch so the store cannot
        // clobber the next plane's already-written low bytes; see the NEON
        // `expand` for the full argument.
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
    }

    /// Mask selecting the first `rem` of 8 f32 lanes, for the last partial
    /// row tile.
    ///
    /// # Safety
    /// Requires AVX2.
    #[inline(always)]
    unsafe fn lane_mask(rem: usize) -> __m256i {
        let idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _mm256_cmpgt_epi32(_mm256_set1_epi32(rem as i32), idx)
    }

    /// One block: `RT` eight-row query tiles × `N` doc tokens over all `d4n`
    /// dim groups, then the fold for exactly those rows and tokens.
    ///
    /// The float epilogue is the same expression in the same order as the
    /// scalar kernel (`_mm256_cvtepi32_ps` rounds to nearest like `as f32`;
    /// separate mul then add, never fused; `inv` last; `_mm256_max_ps`
    /// matches the scalar select for finite scores), so results are
    /// bit-identical.
    ///
    /// # Safety
    /// Requires AVX2. `tile_ptrs[rt]` addresses 8-row tile `row0/8 + rt`
    /// inside the query's tile array, which must carry `d4n` groups of 64
    /// bytes past it; `row0 + 8·(RT−1) < nq`; `crows[j]` readable for `nq`
    /// f32s; `sqw.len() == best.len() == nq`; `ws[j]` readable for `4·d4n`
    /// bytes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn block<const RT: usize, const N: usize>(
        ws: &[[i8; MAX_DIM]; N],
        crows: &[*const f32; N],
        invs: &[f32; N],
        tile_ptrs: &[*const u8; RT],
        d4n: usize,
        row0: usize,
        nq: usize,
        sqw: &[f32],
        best: &mut [f32],
    ) {
        let zero = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let flip = _mm256_set1_epi8(-128);
        let mut acc = [[zero; RT]; N];
        for g in 0..d4n {
            let mut qs = [zero; RT];
            let mut qa = [zero; RT];
            for rt in 0..RT {
                let qu = _mm256_loadu_si256(tile_ptrs[rt].add(g * 64) as *const __m256i);
                qs[rt] = _mm256_xor_si256(qu, flip);
                qa[rt] = _mm256_abs_epi8(qs[rt]);
            }
            for j in 0..N {
                let wb =
                    _mm256_set1_epi32((ws[j].as_ptr().add(g * 4) as *const i32).read_unaligned());
                for rt in 0..RT {
                    let prod = _mm256_maddubs_epi16(qa[rt], _mm256_sign_epi8(wb, qs[rt]));
                    acc[j][rt] = _mm256_add_epi32(acc[j][rt], _mm256_madd_epi16(prod, ones));
                }
            }
        }
        for j in 0..N {
            let invv = _mm256_set1_ps(invs[j]);
            for (rt, acc_rt) in acc[j].iter().enumerate() {
                let r0 = row0 + rt * 8;
                let rem = (nq - r0).min(8);
                let m = lane_mask(rem);
                let a = _mm256_cvtepi32_ps(*acc_rt);
                let s = _mm256_mul_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(_mm256_maskload_ps(sqw.as_ptr().add(r0), m), a),
                        _mm256_maskload_ps(crows[j].add(r0), m),
                    ),
                    invv,
                );
                let b = _mm256_maskload_ps(best.as_ptr().add(r0), m);
                _mm256_maskstore_ps(best.as_mut_ptr().add(r0), m, _mm256_max_ps(b, s));
            }
        }
    }

    /// Address of 8-row tile `r8`: the tile array is built in 16-row tiles,
    /// whose second half is the next 8 rows at a 32-byte offset.
    ///
    /// # Safety
    /// `r8 < 2 * ⌈nq/16⌉`.
    #[inline(always)]
    unsafe fn tile8(tiles: *const u8, tile_stride: usize, r8: usize) -> *const u8 {
        tiles.add((r8 / 2) * tile_stride + (r8 % 2) * 32)
    }

    /// Every 8-row tile for `N` expanded tokens: pairs of tiles, then the
    /// odd one.
    ///
    /// # Safety
    /// As [`block`], for the whole tile array (`n8 = ⌈nq/8⌉` tiles).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tokens<const N: usize>(
        ws: &[[i8; MAX_DIM]; N],
        crows: &[*const f32; N],
        invs: &[f32; N],
        tiles: *const u8,
        tile_stride: usize,
        d4n: usize,
        n8: usize,
        nq: usize,
        sqw: &[f32],
        best: &mut [f32],
    ) {
        let mut t0 = 0usize;
        while n8 - t0 >= 2 {
            let tp = [
                tile8(tiles, tile_stride, t0),
                tile8(tiles, tile_stride, t0 + 1),
            ];
            block::<2, N>(ws, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
            t0 += 2;
        }
        if n8 - t0 == 1 {
            let tp = [tile8(tiles, tile_stride, t0)];
            block::<1, N>(ws, crows, invs, &tp, d4n, t0 * 8, nq, sqw, best);
        }
    }

    /// # Safety
    /// Requires AVX2; `dim % 8 == 0 && dim <= MAX_DIM`;
    /// `planes.tiles` built for `nq` rows at `planes.stride`.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_avx2(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        nib: &NibbleLut,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        let n_tokens = doc_packed.nrows();
        if nq == 0 || n_tokens == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb;
        let d4n = dim / 4;
        let n8 = nq.div_ceil(8);
        let tile_stride = (planes.stride / 4) * 64;
        let tiles = planes.tiles.as_ptr();
        debug_assert!(planes.tiles.len() >= nq.div_ceil(16) * tile_stride);
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }

        // Zeroed once: `expand` writes only below `dim`, so each buffer's
        // tail stays zero and a 4-byte weight group never carries stale data.
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut crows = [std::ptr::null::<f32>(); NT];
        let mut invs = [0f32; NT];
        let mut t = 0usize;
        while t + NT <= n_tokens {
            for j in 0..NT {
                let tt = t + j;
                expand(
                    &d_all[tt * pb..tt * pb + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                crows[j] = cd.as_ptr().add(doc_codes[tt] as usize * nq);
                invs[j] = inv_norms[tt];
            }
            tokens::<NT>(
                &ws,
                &crows,
                &invs,
                tiles,
                tile_stride,
                d4n,
                n8,
                nq,
                sqw,
                best,
            );
            t += NT;
        }
        while t < n_tokens {
            expand(
                &d_all[t * pb..t * pb + pdim],
                &tabs,
                nib,
                kpb,
                pdim,
                &mut ws[0],
            );
            let one_w = [ws[0]];
            let one_c = [cd.as_ptr().add(doc_codes[t] as usize * nq)];
            let one_i = [inv_norms[t]];
            tokens::<1>(
                &one_w,
                &one_c,
                &one_i,
                tiles,
                tile_stride,
                d4n,
                n8,
                nq,
                sqw,
                best,
            );
            t += 1;
        }
        best.iter().sum()
    }
}

/// AVX-512 + VNNI path for the fused asym kernel, in GEMM-tile form.
///
/// The expand stays 128-bit `pshufb` (charged once per doc token and
/// amortized over every query row); what goes wide is the part charged per
/// *(query row, token)*, plus the fold.
///
/// A row-per-accumulator form would end every row with a 16-lane horizontal
/// reduction and need a sign fix-up per chunk, because `vpdpbusd` is
/// unsigned × signed. Both costs vanish when the query *rows* occupy the
/// lanes instead ([`QueryPlanes::tiles`]):
///
/// * the query is stored as tiles of 16 rows × 4 consecutive plane dims (64
///   bytes), every code offset by `+128` so it is a valid `u8`;
/// * the token's expanded weights are broadcast 4 bytes at a time;
/// * `acc = vpdpbusd(acc, q_tile, w_bcast)` then adds 4 dims for 16 rows at
///   once, and after `dim/4` steps each lane holds one row's full sum, offset
///   by `128·Σw`. One vector subtract per (token, tile) removes that offset
///   exactly, and the fold runs straight on the lanes.
///
/// Per `vpdpbusd`: one broadcast load and a fraction of a tile load, no
/// shuffles, no sign ops, no reductions.
///
/// Exactness: `Σ (q+128)·w = Σ q·w + 128·Σw` in exact integers, and
/// `|q|,|w| <= 127` over `dim <= 256` keeps every partial sum far inside i32
/// (the accumulator is bounded by 256·127·255 < 2^24). The fold is the same
/// op sequence as the other kernels, so results are bit-identical to scalar.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// Document tokens held in registers per block.
    const NT: usize = 4;

    /// Expand one stored token's packed bytes into `kpb` planes of int8
    /// weights, returning `Σw` over all `dim` weights for the `+128` offset
    /// correction.
    ///
    /// # Safety
    /// Requires `avx512f,avx512bw,avx512vnni`; `row.len() >= pdim`;
    /// `kpb * pdim <= MAX_DIM`; `tabs` holds the nibble tables for `kpb`
    /// keys; `w` is zero beyond `kpb * pdim` (the `Σw` reduction reads whole
    /// 64-byte chunks, so stale bytes there would corrupt the correction).
    #[inline(always)]
    unsafe fn expand(
        row: &[u8],
        tabs: &[__m128i; 8],
        nib: &NibbleLut,
        kpb: usize,
        pdim: usize,
        w: &mut [i8; MAX_DIM],
    ) -> i32 {
        // The expansion itself is the AVX2 module's loop, duplicated rather
        // than shared: a cross-module call would have to carry `ssse3` in its
        // own target-feature set to be inlinable here, and an out-of-line
        // `pshufb` per key position per 16 bytes would cost more than the
        // duplication.
        let low_mask = _mm_set1_epi8(0x0F);
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= pdim {
            let v = _mm_loadu_si128(row.as_ptr().add(i) as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
            let lo = _mm_and_si128(v, low_mask);
            for (k, tab) in tabs.iter().enumerate().take(kpb) {
                let idx = if nib.from_hi[k] { hi } else { lo };
                _mm_storeu_si128(
                    wp.add(k * pdim + i) as *mut __m128i,
                    _mm_shuffle_epi8(*tab, idx),
                );
            }
            i += 16;
        }
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
        // Σw: u8 ones × s8 weights, over the 64-byte chunks covering `dim`.
        // `dim <= MAX_DIM` and the chunks start at 0, so the last one ends at
        // or before byte 255.
        let dim = kpb * pdim;
        let ones = _mm512_set1_epi8(1);
        let mut s = _mm512_setzero_si512();
        let mut k = 0usize;
        while k < dim {
            s = _mm512_dpbusd_epi32(s, ones, _mm512_loadu_si512(w.as_ptr().add(k) as *const _));
            k += 64;
        }
        _mm512_reduce_add_epi32(s)
    }

    /// One block: `RT` sixteen-row query tiles × `N` doc tokens over all
    /// `d4n` dim groups, then the fold for exactly those rows and tokens.
    ///
    /// # Safety
    /// Requires `avx512f,avx512bw,avx512vnni`. `tiles` addresses row tile
    /// `row0/16` of the query's tile array, with at least `RT` tiles of
    /// `tile_stride` bytes available; `row0 + 16·(RT−1) < nq`; `crows[j]`
    /// readable for `nq` f32s; `sqw.len() == best.len() == nq`; `ws[j]`
    /// readable for `4·d4n` bytes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn block<const RT: usize, const N: usize>(
        ws: &[[i8; MAX_DIM]; N],
        corr: &[__m512i; N],
        crows: &[*const f32; N],
        invs: &[f32; N],
        tiles: *const u8,
        tile_stride: usize,
        d4n: usize,
        row0: usize,
        nq: usize,
        sqw: &[f32],
        best: &mut [f32],
    ) {
        let zero = _mm512_setzero_si512();
        let mut acc = [[zero; RT]; N];
        for g in 0..d4n {
            let mut q = [zero; RT];
            for (rt, qt) in q.iter_mut().enumerate() {
                *qt = _mm512_loadu_si512(tiles.add(rt * tile_stride + g * 64) as *const _);
            }
            for j in 0..N {
                let wb =
                    _mm512_set1_epi32((ws[j].as_ptr().add(g * 4) as *const i32).read_unaligned());
                for rt in 0..RT {
                    acc[j][rt] = _mm512_dpbusd_epi32(acc[j][rt], q[rt], wb);
                }
            }
        }
        for j in 0..N {
            let invv = _mm512_set1_ps(invs[j]);
            for (rt, acc_rt) in acc[j].iter().enumerate() {
                let r0 = row0 + rt * 16;
                let rem = (nq - r0).min(16);
                // Masked loads/stores keep the last, partial row tile inside
                // `sqw` / `crow` / `best`.
                let mask: __mmask16 = if rem == 16 { 0xFFFF } else { (1u16 << rem) - 1 };
                let a = _mm512_cvtepi32_ps(_mm512_sub_epi32(*acc_rt, corr[j]));
                let s = _mm512_mul_ps(
                    _mm512_add_ps(
                        _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, sqw.as_ptr().add(r0)), a),
                        _mm512_maskz_loadu_ps(mask, crows[j].add(r0)),
                    ),
                    invv,
                );
                let b = _mm512_maskz_loadu_ps(mask, best.as_ptr().add(r0));
                _mm512_mask_storeu_ps(best.as_mut_ptr().add(r0), mask, _mm512_max_ps(b, s));
            }
        }
    }

    /// Every row tile for `N` expanded tokens: blocks of four tiles, then the
    /// remainder.
    ///
    /// # Safety
    /// As [`block`], for the whole tile array (`n16` tiles).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tokens<const N: usize>(
        ws: &[[i8; MAX_DIM]; N],
        corr: &[__m512i; N],
        crows: &[*const f32; N],
        invs: &[f32; N],
        tiles: *const u8,
        tile_stride: usize,
        d4n: usize,
        n16: usize,
        nq: usize,
        sqw: &[f32],
        best: &mut [f32],
    ) {
        let mut t0 = 0usize;
        while n16 - t0 >= 4 {
            let tp = tiles.add(t0 * tile_stride);
            let r0 = t0 * 16;
            block::<4, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                r0,
                nq,
                sqw,
                best,
            );
            t0 += 4;
        }
        let tp = tiles.add(t0 * tile_stride);
        let r0 = t0 * 16;
        match n16 - t0 {
            3 => block::<3, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                r0,
                nq,
                sqw,
                best,
            ),
            2 => block::<2, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                r0,
                nq,
                sqw,
                best,
            ),
            1 => block::<1, N>(
                ws,
                corr,
                crows,
                invs,
                tp,
                tile_stride,
                d4n,
                r0,
                nq,
                sqw,
                best,
            ),
            _ => {}
        }
    }

    /// # Safety
    /// Requires `avx512f,avx512bw,avx512vnni`; `dim % 8 == 0 && dim <=
    /// MAX_DIM`; `planes.tiles` built for `nq` rows at `planes.stride`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn maxsim_residual_lut_avx512(
        q8: &QueryI8,
        planes: &QueryPlanes,
        doc_packed: &ArrayView2<u8>,
        doc_codes: &[i64],
        cdot_t: &ArrayView2<f32>,
        lut: &ResidualLut,
        nib: &NibbleLut,
        inv_norms: &[f32],
        dim: usize,
        best: &mut Vec<f32>,
    ) -> f32 {
        let nq = q8.values.nrows();
        let n_tokens = doc_packed.nrows();
        if nq == 0 || n_tokens == 0 {
            return 0.0;
        }
        let kpb = lut.keys_per_byte;
        let pdim = dim / kpb;
        let d4n = dim / 4;
        let n16 = nq.div_ceil(16);
        let tile_stride = (planes.stride / 4) * 64;
        let tiles = planes.tiles.as_ptr();
        debug_assert!(planes.tiles.len() >= n16 * tile_stride);
        let d_all = doc_packed.as_slice().expect("doc bytes must be contiguous");
        let pb = doc_packed.ncols();
        let cd = cdot_t.as_slice().expect("cdot_t must be standard layout");
        debug_assert_eq!(cdot_t.ncols(), nq);
        let sqw: &[f32] = &planes.sqw;
        best.clear();
        best.resize(nq, f32::NEG_INFINITY);
        let mut tabs = [_mm_setzero_si128(); 8];
        for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
            *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        }

        // Zeroed once, and `expand` writes only below `dim`, which is what
        // makes the `Σw` reduction's 64-byte over-read contribute nothing.
        let mut ws = [[0i8; MAX_DIM]; NT];
        let mut corr = [_mm512_setzero_si512(); NT];
        let mut crows = [std::ptr::null::<f32>(); NT];
        let mut invs = [0f32; NT];
        let mut t = 0usize;
        while t + NT <= n_tokens {
            for j in 0..NT {
                let tt = t + j;
                let sumw = expand(
                    &d_all[tt * pb..tt * pb + pdim],
                    &tabs,
                    nib,
                    kpb,
                    pdim,
                    &mut ws[j],
                );
                corr[j] = _mm512_set1_epi32(128 * sumw);
                crows[j] = cd.as_ptr().add(doc_codes[tt] as usize * nq);
                invs[j] = inv_norms[tt];
            }
            tokens::<NT>(
                &ws,
                &corr,
                &crows,
                &invs,
                tiles,
                tile_stride,
                d4n,
                n16,
                nq,
                sqw,
                best,
            );
            t += NT;
        }
        while t < n_tokens {
            let sumw = expand(
                &d_all[t * pb..t * pb + pdim],
                &tabs,
                nib,
                kpb,
                pdim,
                &mut ws[0],
            );
            let one_w = [ws[0]];
            let one_corr = [_mm512_set1_epi32(128 * sumw)];
            let one_c = [cd.as_ptr().add(doc_codes[t] as usize * nq)];
            let one_i = [inv_norms[t]];
            tokens::<1>(
                &one_w,
                &one_corr,
                &one_c,
                &one_i,
                tiles,
                tile_stride,
                d4n,
                n16,
                nq,
                sqw,
                best,
            );
            t += 1;
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
        // nq values chosen to reach every row-blocking branch. NEON folds
        // four rows at a time: 3 = pure scalar tail, 9 = vector blocks + a
        // 1-row tail, 32 = the production query shape, no tail. The x86
        // kernels block over whole row tiles instead, and the remainder arms
        // are selected by how many tiles a query fills: AVX2 works in 8-row
        // tiles (7 and 9 give a partial tile, 17 an odd tile after a pair),
        // AVX-512 in 16-row tiles (17 -> 2, 48 -> the 3-tile arm, 65 -> a
        // 4-tile block plus a single). 8 is exactly one AVX2 tile.
        for &nq in &[3usize, 7, 8, 9, 17, 32, 48, 65] {
            for &nbits in &[1usize, 2, 4] {
                for &dim in &[8usize, 16, 40, 48, 128, 200, 256] {
                    let k = 12;
                    let codec = toy_codec(dim, nbits, k, &mut rng);
                    let lut = quantize_lut(&codec).unwrap();
                    let nib = lut.nibble.as_ref().expect("nibble tables");
                    let query = Array2::from_shape_fn((nq, dim), |_| rng.gen_range(-1.0f32..1.0));
                    let q8 = crate::binary::quantize_query_i8(&query.view());
                    let planes = build_query_planes(&q8, &lut, dim);
                    let res = Array2::from_shape_fn((13, dim), |_| rng.gen_range(-0.4f32..0.4));
                    let packed = codec.quantize_residuals(&res).unwrap();
                    let codes: Vec<i64> = (0..13).map(|_| rng.gen_range(0..k as i64)).collect();
                    let cdot_t = Array2::from_shape_fn((k, nq), |_| rng.gen_range(-1.0f32..1.0));
                    let inv: Vec<f32> = (0..13).map(|_| rng.gen_range(0.5f32..1.5)).collect();

                    let scalar = maxsim_residual_lut_scalar(
                        &q8,
                        &packed.view(),
                        &codes,
                        &cdot_t.view(),
                        &lut,
                        &inv,
                        dim,
                    );
                    // Fresh scratch per call — also proves the kernels fully
                    // initialize it (no state carried between calls).
                    let mut best = Vec::new();
                    #[cfg(target_arch = "aarch64")]
                    let simd = unsafe {
                        super::neon::maxsim_residual_lut_neon(
                            &q8,
                            &planes,
                            &packed.view(),
                            &codes,
                            &cdot_t.view(),
                            &lut,
                            nib,
                            &inv,
                            dim,
                            &mut best,
                        )
                    };
                    #[cfg(target_arch = "x86_64")]
                    let simd = unsafe {
                        super::avx2::maxsim_residual_lut_avx2(
                            &q8,
                            &planes,
                            &packed.view(),
                            &codes,
                            &cdot_t.view(),
                            &lut,
                            nib,
                            &inv,
                            dim,
                            &mut best,
                        )
                    };
                    assert_eq!(
                        scalar.to_bits(),
                        simd.to_bits(),
                        "nq={nq} nbits={nbits} dim={dim}: scalar {scalar} != simd {simd}"
                    );
                    // On AVX-512 VNNI hardware, also pin the 512-bit kernel
                    // to the reference directly — the dispatcher check below
                    // would otherwise be this kernel's only coverage, and on
                    // AVX2-only hardware it gets none (dispatch never
                    // reaches it there).
                    #[cfg(target_arch = "x86_64")]
                    if has_avx512_vnni() {
                        let mut best = Vec::new();
                        let v512 = unsafe {
                            super::avx512::maxsim_residual_lut_avx512(
                                &q8,
                                &planes,
                                &packed.view(),
                                &codes,
                                &cdot_t.view(),
                                &lut,
                                nib,
                                &inv,
                                dim,
                                &mut best,
                            )
                        };
                        assert_eq!(
                            scalar.to_bits(),
                            v512.to_bits(),
                            "nq={nq} nbits={nbits} dim={dim}: scalar {scalar} != avx512 {v512}"
                        );
                    }
                    // And the safe dispatcher must agree with the reference
                    // whatever path it picks on this machine.
                    let dispatched = maxsim_residual_lut_i8(
                        &q8,
                        Some(&planes),
                        &packed.view(),
                        &codes,
                        &cdot_t.view(),
                        &lut,
                        &inv,
                        dim,
                    );
                    assert_eq!(
                        scalar.to_bits(),
                        dispatched.to_bits(),
                        "nq={nq} nbits={nbits} dim={dim}: dispatcher diverged from scalar"
                    );
                }
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
                let cdot_t = cents.dot(&query.t());
                let inv = compute_inv_norms(&codec, &codes, &packed.view()).unwrap();
                let planes = build_query_planes(&q8, &lut, dim);

                let got = maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
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
            // 44 is deliberately not a multiple of 8: no query planes are
            // built, so the public dispatcher takes the scalar path. Only
            // byte-aligned payloads exist (`quantize_residuals` packs
            // `dim·nbits/8` whole bytes), so skip combinations the codec
            // cannot produce.
            for &dim in &[8usize, 44, 48, 128, 256] {
                if (dim * nbits) % 8 != 0 {
                    continue;
                }
                let k = 8;
                let codec = toy_codec(dim, nbits, k, &mut rng);
                let lut = quantize_lut(&codec).unwrap();

                let query = Array2::from_shape_fn((6, dim), |_| rng.gen_range(-1.0f32..1.0));
                let q8 = crate::binary::quantize_query_i8(&query.view());
                let res = Array2::from_shape_fn((9, dim), |_| rng.gen_range(-0.4f32..0.4));
                let packed = codec.quantize_residuals(&res).unwrap();
                let codes: Vec<i64> = (0..9).map(|_| rng.gen_range(0..k as i64)).collect();
                let cdot_t = Array2::from_shape_fn((k, 6), |_| rng.gen_range(-1.0f32..1.0));
                let inv: Vec<f32> = (0..9).map(|_| rng.gen_range(0.5f32..1.5)).collect();
                let planes = dim
                    .is_multiple_of(8)
                    .then(|| build_query_planes(&q8, &lut, dim));

                let got = maxsim_residual_lut_i8(
                    &q8,
                    planes.as_ref(),
                    &packed.view(),
                    &codes,
                    &cdot_t.view(),
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
                            + cdot_t[[codes[t] as usize, qi]] as f64)
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
