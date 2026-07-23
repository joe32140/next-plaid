//! Asymmetric binary quantization for late-interaction (MaxSim) scoring.
//!
//! Documents are stored as 1-bit signs — 32x smaller than `f32` — while queries
//! stay at higher precision (`f32`, optionally int8-rounded). Scoring is an
//! *asymmetric* MaxSim: each query token is matched against `±1` document tokens
//! and never decompressed to a learned float, unlike the residual codec which
//! reconstructs `centroid + bucket_weight` before scoring.
//!
//! This mirrors the "Int8 x Binary" scheme from mixedbread's asymmetric
//! quantization work, adapted to multi-vector ColBERT MaxSim. The key identity
//! is that a dot product against a sign vector `s ∈ {-1,+1}^d` is exact:
//! `q · s = Σ_d q_d · s_d`, so document precision can collapse to one bit per
//! dimension while the ranking stays close to full precision.
//!
//! Scoring kernels (all exact w.r.t. the stored signs):
//!   * [`maxsim_binary_i8`] — the search-path default. Dispatches to a fused
//!     doc-token-outer kernel for byte-aligned dims (`dim % 8 == 0`, up to
//!     256): AVX-512 VNNI `vpdpbusd` or AVX2 masked-SAD on x86_64, SDOT on
//!     aarch64. Ragged or larger dims take a per-pair bit-native `2P − T`
//!     SIMD dot (NEON / AVX2 / scalar).
//!   * [`maxsim_binary`] — decode to +/-1 f32 then BLAS GEMM; reference only.

use ndarray::{Array2, ArrayView2, Axis};

use crate::maxsim::maxsim_score;
use crate::utils::packbits;

/// An int8-quantized query: per-row `i8` values plus the per-row dequant scale.
///
/// This is the "int8" side of int8 × binary scoring, but unlike
/// [`quantize_query_int8`] (which rounds and immediately re-expands to `f32`),
/// the integer values are kept so the fast kernel can score with integer adds
/// only. `scales[i]` maps row `i` back to the original magnitude: the true
/// query value is `values[[i, d]] as f32 * scales[i]`.
pub struct QueryI8 {
    pub values: Array2<i8>,
    pub scales: Vec<f32>,
    /// Per-row sum of the int8 codes — the `T` in the `2P − T` identity.
    /// Hoisted here so no kernel recomputes it per candidate document.
    pub sums: Vec<i32>,
    /// Row-major biased codes (`x ^ 0x80`, i.e. `x + 128` viewed as `u8`) at a
    /// row stride of `dim` rounded up to 32 lanes, zero-filled beyond `dim` —
    /// the query layout consumed by the AVX2 masked-SAD kernel, where the sum
    /// of selected biased bytes is `P + 128 · popcount(doc bits)`. Padding
    /// lanes are never selected (their doc bits are zero-extended), so the
    /// fill value cannot reach the score.
    pub biased: Vec<u8>,
    /// Row-major signed codes at a row stride of `dim` rounded up to 64 lanes,
    /// zero-filled beyond `dim` — the query layout consumed by the fused
    /// AVX-512 VNNI and NEON SDOT kernels, whose vector loads read whole
    /// 64-/16-lane groups. Zero padding keeps partial tail groups exact: a
    /// padding lane contributes `anything · 0 = 0`.
    pub padded: Vec<i8>,
}

/// Quantize a query to symmetric per-row int8, keeping the integer codes.
///
/// Each row is scaled by `max_abs / 127` so the largest-magnitude component maps
/// to ±127. Feeds [`maxsim_binary_i8`], the multiply-free scoring path.
pub fn quantize_query_i8(query: &ArrayView2<f32>) -> QueryI8 {
    let n = query.nrows();
    let dim = query.ncols();
    let mut values = Array2::<i8>::zeros((n, dim));
    let mut scales = vec![0.0f32; n];
    for (i, row) in query.axis_iter(Axis(0)).enumerate() {
        let max_abs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if max_abs <= 0.0 {
            continue; // all-zero row: scale 0, codes 0
        }
        let scale = max_abs / 127.0;
        scales[i] = scale;
        for (d, &x) in row.iter().enumerate() {
            values[[i, d]] = (x / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let v = values.as_slice().expect("contiguous");
    let sums = (0..n)
        .map(|i| v[i * dim..(i + 1) * dim].iter().map(|&x| x as i32).sum())
        .collect();
    let bs = biased_stride(dim);
    let ps = padded_stride(dim);
    let mut biased = vec![0u8; n * bs];
    let mut padded = vec![0i8; n * ps];
    for i in 0..n {
        for (d, &x) in v[i * dim..(i + 1) * dim].iter().enumerate() {
            biased[i * bs + d] = (x as u8) ^ 0x80;
            padded[i * ps + d] = x;
        }
    }
    QueryI8 {
        values,
        scales,
        sums,
        biased,
        padded,
    }
}

/// Row stride, in lanes, of [`QueryI8::biased`]: `dim` rounded up to the
/// 32-lane ymm groups the AVX2 masked-SAD kernel loads.
#[inline]
fn biased_stride(dim: usize) -> usize {
    dim.div_ceil(32) * 32
}

/// Row stride, in lanes, of [`QueryI8::padded`]: `dim` rounded up to the
/// 64-lane zmm groups the AVX-512 VNNI kernel loads (also a multiple of the
/// NEON kernel's 16-lane chunks).
#[inline]
fn padded_stride(dim: usize) -> usize {
    dim.div_ceil(64) * 64
}

/// Dot product of an int8 query token against one packed ±1 document token,
/// via the exact identity `q · s = 2 · Σ_{bit=1} q − Σ q` (integer adds only).
///
/// `q.len() == dim`; `bits` holds `packed_dim(dim)` bytes, big-endian per byte
/// (dim `d` is bit `7 - (d % 8)` of byte `d / 8`), matching [`binarize`].
#[inline]
fn dot_pm1_scalar(q: &[i8], t: i32, bits: &[u8], dim: usize) -> i32 {
    let mut p = 0i32;
    // Whole bytes: 8 dims each. Branchless — each bit becomes a 0/-1 mask so the
    // loop autovectorizes (critical on x86 where there is no NEON path).
    let full = dim / 8;
    for (k, &byte) in bits[..full].iter().enumerate() {
        let base = k * 8;
        for j in 0..8 {
            // mask = 0xFFFF_FFFF if bit set (MSB-first), else 0.
            let bit = ((byte >> (7 - j)) & 1) as i32;
            let mask = -bit; // 0 or -1
            p += (q[base + j] as i32) & mask;
        }
    }
    // Tail dims (dim not a multiple of 8).
    for d in (full * 8)..dim {
        let bit = ((bits[d / 8] >> (7 - (d % 8))) & 1) as i32;
        p += (q[d] as i32) & (-bit);
    }
    2 * p - t
}

/// NEON dot product of int8 query token against packed ±1 doc token.
///
/// Expands 16 sign bits at a time into a ±1 `i8` lane vector and does a widening
/// multiply-accumulate against the query lanes. Handles the `dim % 16` tail with
/// the scalar `2P − T` path. Result is identical to [`dot_pm1_scalar`].
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_pm1_neon(q: &[i8], t: i32, bits: &[u8], dim: usize) -> i32 {
    use std::arch::aarch64::*;
    let blocks = dim / 16; // 16 dims -> 2 bits-bytes per block
    if blocks == 0 {
        return dot_pm1_scalar(q, t, bits, dim);
    }
    // Per-lane bit-select mask: MSB-first within each byte (dim 8k -> 0x80).
    const SEL: [u8; 16] = [
        0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01,
    ];
    unsafe {
        let sel = vld1q_u8(SEL.as_ptr());
        let pos = vdupq_n_s8(1);
        let neg = vdupq_n_s8(-1);
        let mut acc = vdupq_n_s32(0);
        for b in 0..blocks {
            let byte0 = bits[b * 2];
            let byte1 = bits[b * 2 + 1];
            // Broadcast byte0 to lanes 0..8 and byte1 to lanes 8..16.
            let bitbytes = vcombine_u8(vdup_n_u8(byte0), vdup_n_u8(byte1));
            // lane != 0 after AND  ->  bit is set.
            let anded = vandq_u8(bitbytes, sel);
            let is_set = vtstq_u8(anded, anded); // 0xFF where bit set, else 0x00
                                                 // pm1 = is_set ? +1 : -1
            let pm1 = vbslq_s8(is_set, pos, neg);
            let qv = vld1q_s8(q.as_ptr().add(b * 16));
            // Widening multiply-accumulate: (i8 × i8) -> i16 -> i32.
            let prod_lo = vmull_s8(vget_low_s8(qv), vget_low_s8(pm1)); // i16x8
            let prod_hi = vmull_s8(vget_high_s8(qv), vget_high_s8(pm1));
            acc = vpadalq_s16(acc, prod_lo);
            acc = vpadalq_s16(acc, prod_hi);
        }
        // q·s over the vectorized dims.
        let dot_vec = vaddvq_s32(acc);
        // Scalar tail for dims not covered by whole 16-blocks.
        let done = blocks * 16;
        if done == dim {
            dot_vec
        } else {
            // Recompute tail as a standalone q·s and add. dot_pm1_scalar returns
            // 2P−T over the *whole* vector, so instead sum the tail directly.
            let mut tail = 0i32;
            for d in done..dim {
                let bit = (bits[d / 8] >> (7 - (d % 8))) & 1;
                tail += if bit == 1 {
                    q[d] as i32
                } else {
                    -(q[d] as i32)
                };
            }
            dot_vec + tail
        }
    }
}

/// AVX2 dot product of int8 query token against packed ±1 doc token.
///
/// Expands 32 sign bits at a time into a ±1 `i8` lane vector, applies the sign to
/// the query lanes with `_mm256_sign_epi8` (`q · s` directly), then widens and
/// accumulates with `_mm256_madd_epi16`. Result is identical to
/// [`dot_pm1_scalar`]. Requires AVX2; the caller gates on runtime detection.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_pm1_avx2(q: &[i8], t: i32, bits: &[u8], dim: usize) -> i32 {
    use std::arch::x86_64::*;
    let blocks = dim / 32; // 32 dims -> 4 bits-bytes per block
    if blocks == 0 {
        return dot_pm1_scalar(q, t, bits, dim);
    }
    // Per-lane selector mask, MSB-first within each byte (dim 8k -> 0x80).
    const SEL: [i8; 32] = [
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01,
    ];
    let sel = _mm256_loadu_si256(SEL.as_ptr() as *const __m256i);
    // Replicate packed byte k across its 8 sign lanes: broadcast the 4 packed
    // bytes to every 32-bit element, then shuffle byte j of each element to
    // lanes 8j..8j+8. Two instructions instead of a 32-operand `set_epi8`.
    const IDX: [i8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    let idx = _mm256_loadu_si256(IDX.as_ptr() as *const __m256i);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for b in 0..blocks {
        let base = b * 4;
        let w = (bits.as_ptr().add(base) as *const u32).read_unaligned();
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
        let anded = _mm256_and_si256(bytes, sel);
        // 0xFF where the selected bit is set, else 0x00.
        let is_set = _mm256_cmpeq_epi8(anded, sel);
        // pm1 = is_set ? +1 : -1. blendv picks the second arg where the mask's
        // high bit is set (0xFF -> +1) and the first where clear (0x00 -> -1).
        let pm1 = _mm256_blendv_epi8(_mm256_set1_epi8(-1), _mm256_set1_epi8(1), is_set);
        let qv = _mm256_loadu_si256(q.as_ptr().add(b * 32) as *const __m256i);
        // Apply sign: lanes of q, negated where pm1 < 0. This is exactly q_d * s_d.
        let signed = _mm256_sign_epi8(qv, pm1);
        // Widen i8 -> i16 (two halves) and horizontally add pairs into i32.
        let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(signed));
        let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(signed, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, ones));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi, ones));
    }
    // Horizontal sum of the 8 i32 lanes.
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_00_01));
    let dot_vec = _mm_cvtsi128_si32(sum32);
    // Scalar tail for dims beyond whole 32-blocks.
    let done = blocks * 32;
    if done == dim {
        dot_vec
    } else {
        let mut tail = 0i32;
        for d in done..dim {
            let bit = (bits[d / 8] >> (7 - (d % 8))) & 1;
            tail += if bit == 1 {
                q[d] as i32
            } else {
                -(q[d] as i32)
            };
        }
        dot_vec + tail
    }
}

/// Architecture-dispatching int8×±1 token dot product (`q · s`).
#[inline]
fn dot_pm1(q: &[i8], t: i32, bits: &[u8], dim: usize) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        dot_pm1_neon(q, t, bits, dim)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_pm1_avx2(q, t, bits, dim) };
        }
        dot_pm1_scalar(q, t, bits, dim)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_pm1_scalar(q, t, bits, dim)
    }
}

/// Little-endian zero-extended load of packed bytes `8g..8g+8` of one doc
/// token's row — the 64 sign bits of dim group `g`. Bytes past the row are
/// zero-extended; a cleared bit selects nothing, so tail groups stay exact.
///
/// This runs once per group per doc token, so whole groups must compile to a
/// single unaligned load and the tail group to a shifted overlapping load —
/// the safe-slice route (`get` + `try_into`) round-trips through a stack
/// buffer (or a `memcpy` call), costing a store-forward per group per token.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn group_bits_u64(row: &[u8], g: usize) -> u64 {
    let off = g * 8;
    let len = row.len();
    if off + 8 <= len {
        // SAFETY: in bounds for 8 bytes by the check above.
        unsafe { (row.as_ptr().add(off) as *const u64).read_unaligned() }
    } else if len >= 8 {
        // Tail group: read the row's LAST 8 bytes (in bounds, overlapping the
        // previous group) and shift byte `off` down to bit 0; the top fills
        // with zeros. `off < len`, so the shift stays below 64.
        // SAFETY: `len - 8 >= 0` by the check above.
        let w = unsafe { (row.as_ptr().add(len - 8) as *const u64).read_unaligned() };
        w >> (8 * (off + 8 - len))
    } else {
        // Row shorter than one group (dim < 64, so this is group 0): assemble.
        let mut w = 0u64;
        for (k, &b) in row[off..].iter().enumerate() {
            w |= (b as u64) << (8 * k);
        }
        w
    }
}

/// 32-bit sibling of [`group_bits_u64`] for the AVX2 kernel's ymm groups.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn group_bits_u32(row: &[u8], g: usize) -> u32 {
    let off = g * 4;
    let len = row.len();
    if off + 4 <= len {
        // SAFETY: in bounds for 4 bytes by the check above.
        unsafe { (row.as_ptr().add(off) as *const u32).read_unaligned() }
    } else if len >= 4 {
        // SAFETY: `len - 4 >= 0` by the check above.
        let w = unsafe { (row.as_ptr().add(len - 4) as *const u32).read_unaligned() };
        w >> (8 * (off + 4 - len))
    } else {
        let mut w = 0u32;
        for (k, &b) in row[off..].iter().enumerate() {
            w |= (b as u32) << (8 * k);
        }
        w
    }
}

/// AVX-512 VNNI fused MaxSim for byte-aligned dims, monomorphized over the
/// per-token zmm group count `G = ceil(dim/64)` (dims ≤ 256 keep the block's
/// `4·G` expanded masks register-resident).
///
/// Doc-token-outer loop: each token's sign bits are expanded ONCE, in
/// registers, to `G` zmm vectors of `0/1` bytes (u64 broadcast + `pshufb` +
/// `vptestmb` + `maskz_mov` — ~5 instructions per group), amortized over
/// every query token. Scoring is one `vpdpbusd` per group (u8 × i8
/// dot-accumulate — the x86 twin of the NEON SDOT in mixedbread's kernel):
/// `P = Σ mask_d · q_d`, `score = 2P − T`. Doc tokens go in blocks of 4 so
/// the horizontal reductions collapse into `phaddd` pairs and the per-query
/// running max stays vectorized (`pmaxsd`).
///
/// Queries come from [`QueryI8::padded`] (row stride `G·64`, zero-filled) and
/// tail-group doc bytes are zero-extended, so partial groups score exactly:
/// every padding lane contributes `0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn maxsim_vnni<const G: usize>(
    q_pad: &[i8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let n_q = sums.len();
    debug_assert!(q_pad.len() >= n_q * G * 64);
    // After `set1_epi64` every 128-bit chunk holds the group's 8 packed bytes
    // (twice); chunk c covers dims 16c..16c+16 = packed bytes 2c, 2c+1.
    // Indices replicate byte k across its 8 lanes.
    const IDX: [i8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7,
    ];
    // MSB-first bit selector per lane group (dim 8k -> 0x80), as in `binarize`.
    const SEL: [i8; 64] = [
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08,
        0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10,
        0x08, 0x04, 0x02, 0x01,
    ];
    let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
    let sel = _mm512_loadu_si512(SEL.as_ptr() as *const __m512i);
    let one = _mm512_set1_epi8(1);

    // Expand one group's 64 packed sign bits into a zmm of 0/1 bytes.
    macro_rules! expand {
        ($w:expr) => {{
            let bc = _mm512_set1_epi64($w as i64);
            let k = _mm512_test_epi8_mask(_mm512_shuffle_epi8(bc, idx), sel);
            _mm512_maskz_mov_epi8(k, one)
        }};
    }
    // Sum the 16 i32 lanes of a zmm down to an xmm of 4 partial lanes.
    macro_rules! fold4 {
        ($a:expr) => {{
            let y = _mm256_add_epi32(_mm512_castsi512_si256($a), _mm512_extracti64x4_epi64($a, 1));
            _mm_add_epi32(_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1))
        }};
    }

    // Per-query running max over doc tokens, 4 lanes per query token (one lane
    // per doc slot in the block). Tail blocks replicate the last token, which
    // cannot change a max.
    let mut best = vec![i32::MIN; n_q * 4];
    let qp = q_pad.as_ptr();
    let mut db = 0usize;
    while db < n_d {
        let mut masks = [[_mm512_setzero_si512(); G]; 4];
        for (t, m) in masks.iter_mut().enumerate() {
            let d = (db + t).min(n_d - 1);
            let row = &d_all[d * pdim..(d + 1) * pdim];
            for (g, slot) in m.iter_mut().enumerate() {
                *slot = expand!(group_bits_u64(row, g));
            }
        }
        for (qi, &sum) in sums.iter().enumerate() {
            let q = qp.add(qi * G * 64);
            let mut acc = [_mm512_setzero_si512(); 4];
            for g in 0..G {
                let qv = _mm512_loadu_si512(q.add(g * 64) as *const __m512i);
                for (a, m) in acc.iter_mut().zip(&masks) {
                    *a = _mm512_dpbusd_epi32(*a, m[g], qv);
                }
            }
            // [P0, P1, P2, P3] for the 4 doc tokens of this block.
            let h01 = _mm_hadd_epi32(fold4!(acc[0]), fold4!(acc[1]));
            let h23 = _mm_hadd_epi32(fold4!(acc[2]), fold4!(acc[3]));
            let p4 = _mm_hadd_epi32(h01, h23);
            let sc = _mm_sub_epi32(_mm_slli_epi32(p4, 1), _mm_set1_epi32(sum));
            let bp = best.as_mut_ptr().add(qi * 4) as *mut __m128i;
            _mm_storeu_si128(bp, _mm_max_epi32(_mm_loadu_si128(bp), sc));
        }
        db += 4;
    }

    let mut total = 0.0f32;
    for qi in 0..n_q {
        let b = &best[qi * 4..qi * 4 + 4];
        let m = b.iter().copied().max().unwrap();
        total += m as f32 * scales[qi];
    }
    total
}

/// Monomorphization dispatch for [`maxsim_vnni`]: `g = ceil(dim/64) ∈ 1..=4`.
///
/// # Safety
/// Same contract as [`maxsim_vnni`]: AVX-512 F/BW/VNNI present, `q_pad` rows
/// at stride `g·64`, doc rows of `pdim` bytes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn maxsim_vnni_dyn(
    g: usize,
    q_pad: &[i8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    match g {
        1 => maxsim_vnni::<1>(q_pad, sums, scales, d_all, pdim, n_d),
        2 => maxsim_vnni::<2>(q_pad, sums, scales, d_all, pdim, n_d),
        3 => maxsim_vnni::<3>(q_pad, sums, scales, d_all, pdim, n_d),
        4 => maxsim_vnni::<4>(q_pad, sums, scales, d_all, pdim, n_d),
        _ => unreachable!("fused kernels cover dims <= 256"),
    }
}

/// AVX2 masked-SAD fused MaxSim for byte-aligned dims, monomorphized over the
/// per-token ymm group count `NG = ceil(dim/32)` (dims ≤ 256; no AVX-512
/// required).
///
/// Doc-token-outer: expand each token's bits ONCE into `NG` ymm `0xFF/0x00`
/// masks (broadcast + `pshufb` + `pcmpeqb`), amortized over all query tokens.
/// Scoring uses the biased-SAD identity: with `qb = q + 128` stored as `u8`,
/// `SAD(qb & mask, 0) = P + 128 · popcount(bits)`, so
/// `P = SAD − 128·popcount` and `score = 2P − T`. Every scoring op
/// (`pand`/`psadbw`/`paddq`) is a cheap 1-µop instruction — no widening chains.
///
/// Queries come from [`QueryI8::biased`] (row stride `NG·32`, zero-filled)
/// and tail-group doc bytes are zero-extended, so padding lanes are never
/// selected and reach neither the SAD nor the popcount.
///
/// `popcnt` is enabled alongside AVX2 so `count_ones` compiles to the
/// instruction rather than a ~17-op SWAR sequence per group per token (every
/// AVX2 CPU has it — it predates AVX2 by two µarch generations — but the
/// dispatcher still verifies at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn maxsim_avx2_sad<const NG: usize>(
    qb_pad: &[u8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let n_q = sums.len();
    debug_assert!(qb_pad.len() >= n_q * NG * 32);
    const IDX: [i8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    const SEL: [i8; 32] = [
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01,
    ];
    let idx = _mm256_loadu_si256(IDX.as_ptr() as *const __m256i);
    let sel = _mm256_loadu_si256(SEL.as_ptr() as *const __m256i);
    let zero = _mm256_setzero_si256();

    let mut best = vec![i32::MIN; n_q];
    let qp = qb_pad.as_ptr();
    // `chunks_exact` walks doc rows with an additive pointer — no per-token
    // index multiply or slice bounds check in front of the query loop.
    for row in d_all.chunks_exact(pdim).take(n_d) {
        // 0xFF mask ymm per 32-dim group, plus the token's total popcount.
        let mut masks = [_mm256_setzero_si256(); NG];
        let mut cnt = 0i32;
        for (g, m) in masks.iter_mut().enumerate() {
            let w = group_bits_u32(row, g);
            cnt += w.count_ones() as i32;
            let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
            *m = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, sel), sel);
        }
        for qi in 0..n_q {
            let q = qp.add(qi * NG * 32);
            // Two alternating accumulators keep the SAD sums a balanced tree
            // instead of one serial add dependency chain.
            let mut acc = [_mm256_setzero_si256(); 2];
            for (g, m) in masks.iter().enumerate() {
                let qv = _mm256_loadu_si256(q.add(g * 32) as *const __m256i);
                acc[g & 1] =
                    _mm256_add_epi64(acc[g & 1], _mm256_sad_epu8(_mm256_and_si256(*m, qv), zero));
            }
            let s = _mm256_add_epi64(acc[0], acc[1]);
            let x = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            let sad = _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_unpackhi_epi64(x, x))) as i32;
            let score = 2 * (sad - 128 * cnt) - sums[qi];
            if score > best[qi] {
                best[qi] = score;
            }
        }
    }

    let mut total = 0.0f32;
    for qi in 0..n_q {
        total += best[qi] as f32 * scales[qi];
    }
    total
}

/// Monomorphization dispatch for [`maxsim_avx2_sad`]:
/// `ng = ceil(dim/32) ∈ 1..=8`.
///
/// # Safety
/// Same contract as [`maxsim_avx2_sad`]: AVX2 present, `qb_pad` rows at
/// stride `ng·32`, doc rows of `pdim` bytes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn maxsim_avx2_sad_dyn(
    ng: usize,
    qb_pad: &[u8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    match ng {
        1 => maxsim_avx2_sad::<1>(qb_pad, sums, scales, d_all, pdim, n_d),
        2 => maxsim_avx2_sad::<2>(qb_pad, sums, scales, d_all, pdim, n_d),
        3 => maxsim_avx2_sad::<3>(qb_pad, sums, scales, d_all, pdim, n_d),
        4 => maxsim_avx2_sad::<4>(qb_pad, sums, scales, d_all, pdim, n_d),
        5 => maxsim_avx2_sad::<5>(qb_pad, sums, scales, d_all, pdim, n_d),
        6 => maxsim_avx2_sad::<6>(qb_pad, sums, scales, d_all, pdim, n_d),
        7 => maxsim_avx2_sad::<7>(qb_pad, sums, scales, d_all, pdim, n_d),
        8 => maxsim_avx2_sad::<8>(qb_pad, sums, scales, d_all, pdim, n_d),
        _ => unreachable!("fused kernels cover dims <= 256"),
    }
}

/// `sdot vD.4s, vN.16b, vM.16b` — signed int8 dot product into i32x4, via inline
/// asm because the `vdotq_s32` intrinsic is still nightly-only. Sound with
/// `options(pure, nomem, nostack)`: reads three registers, writes one, no memory.
///
/// # Safety
/// Requires the `dotprod` target feature at runtime (caller must check).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_asm(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::int32x4_t;
    let out: int32x4_t;
    std::arch::asm!(
        "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
        out = inout(vreg) acc => out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    out
}

/// NEON SDOT fused MaxSim for byte-aligned dims up to 256 (aarch64 with
/// `dotprod`).
///
/// The ARM analog of `maxsim_vnni`: doc-token-outer, so each token's sign
/// bits are expanded ONCE — broadcast + `vtst` + `vbsl` per 16-dim chunk into
/// a ±1 `i8` stack tile — and amortized over every query token. Scoring
/// SDOTs the query's natural-order int8 lanes ([`QueryI8::padded`], zero
/// beyond `dim`) straight against the ±1 lanes: SDOT is a true signed×signed
/// dot, so each accumulator lane sums `q_d · s_d` directly and no `2P − T`
/// detour is needed (unlike `vpdpbusd`/`psadbw`, which force the 0/1-mask
/// identity on x86). Padding bits expand to −1 but meet a zero query lane,
/// so partial tail chunks score exactly. Doc tokens go in blocks of 4 with
/// two SDOT accumulator chains per token (8 independent chains hide SDOT
/// latency), the horizontal reductions collapse into `vpaddq` pairs, and the
/// per-query running max stays vectorized (`vmaxq_s32`).
///
/// The expansion lives in a ≤1 KiB stack tile, never the index: doc bytes
/// read per token stay `pdim` (packed bits), not `8·pdim` (expanded lanes),
/// which is what wins once the candidate set outgrows L2 — index-time
/// expansion measured no faster than this on Apple M4 while costing 8× the
/// memory.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn maxsim_neon<const C: usize>(
    q_pad: &[i8],
    q_stride: usize,
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let n_q = scales.len();
    // C = 16-dim chunks per token: 2 packed bytes each, the last chunk
    // possibly holding a single byte (dim % 16 == 8). Monomorphized so the
    // chunk loops fully unroll with static addressing (mirrors
    // `maxsim_vnni<G>`; a runtime chunk count costs ~55% at dim = 128).
    let c = C;
    debug_assert_eq!(c, pdim.div_ceil(2));
    debug_assert!(c * 16 <= q_stride, "query stride must cover all chunks");
    debug_assert!(q_pad.len() >= n_q * q_stride);
    // Per-lane bit-select mask: MSB-first within each byte (dim 8k -> 0x80).
    const SEL: [u8; 16] = [
        0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01,
    ];
    let sel = vld1q_u8(SEL.as_ptr());
    let pos = vdupq_n_s8(1);
    let neg = vdupq_n_s8(-1);

    // Per-query running max over doc tokens, 4 lanes per query token (one lane
    // per doc slot in the block). Tail blocks replicate the last token, which
    // cannot change a max.
    let mut best = vec![i32::MIN; n_q * 4];
    // ±1 expansion of the current block's 4 doc tokens, chunk-major per token
    // (dims ≤ 256 → c ≤ 16 → 4·c·16 ≤ 1024 bytes).
    let mut tile = [0i8; 4 * 16 * 16];
    let mut db = 0usize;
    while db < n_d {
        for t in 0..4 {
            let d = (db + t).min(n_d - 1);
            let row = &d_all[d * pdim..(d + 1) * pdim];
            for ch in 0..c {
                let b0 = row[ch * 2];
                let b1 = if ch * 2 + 1 < pdim {
                    row[ch * 2 + 1]
                } else {
                    0
                };
                // Broadcast byte0 to lanes 0..8 and byte1 to lanes 8..16, then
                // turn each lane's bit into ±1.
                let bitbytes = vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1));
                let pm1 = vbslq_s8(vtstq_u8(bitbytes, sel), pos, neg);
                vst1q_s8(tile.as_mut_ptr().add((t * c + ch) * 16), pm1);
            }
        }
        let tp = tile.as_ptr();
        for qi in 0..n_q {
            let q = q_pad.as_ptr().add(qi * q_stride);
            // Two SDOT accumulator chains per doc token, fed in chunk pairs.
            let mut a = [vdupq_n_s32(0); 4];
            let mut b = [vdupq_n_s32(0); 4];
            let mut ch = 0usize;
            while ch + 1 < c {
                let q0 = vld1q_s8(q.add(ch * 16));
                let q1 = vld1q_s8(q.add(ch * 16 + 16));
                for (t, (at, bt)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
                    *at = sdot_asm(*at, q0, vld1q_s8(tp.add((t * c + ch) * 16)));
                    *bt = sdot_asm(*bt, q1, vld1q_s8(tp.add((t * c + ch + 1) * 16)));
                }
                ch += 2;
            }
            if ch < c {
                let q0 = vld1q_s8(q.add(ch * 16));
                for (t, at) in a.iter_mut().enumerate() {
                    *at = sdot_asm(*at, q0, vld1q_s8(tp.add((t * c + ch) * 16)));
                }
            }
            // Pairwise-add tree -> [S0, S1, S2, S3]: lane t is the full
            // `q · s` against doc token t of the block.
            let s4 = vpaddq_s32(
                vpaddq_s32(vaddq_s32(a[0], b[0]), vaddq_s32(a[1], b[1])),
                vpaddq_s32(vaddq_s32(a[2], b[2]), vaddq_s32(a[3], b[3])),
            );
            let bp = best.as_mut_ptr().add(qi * 4);
            vst1q_s32(bp, vmaxq_s32(vld1q_s32(bp), s4));
        }
        db += 4;
    }

    let mut total = 0.0f32;
    for (qi, &scale) in scales.iter().enumerate() {
        let m = vmaxvq_s32(vld1q_s32(best.as_ptr().add(qi * 4)));
        total += m as f32 * scale;
    }
    total
}

impl QueryI8 {
    /// True when the hoisted per-row layouts agree with `values` — the
    /// invariant [`quantize_query_i8`] establishes. The unsafe fused kernels
    /// index these buffers with raw pointers, so dispatch checks this and
    /// falls back to the safe slice paths for hand-built values.
    fn layouts_consistent(&self) -> bool {
        let n_q = self.values.nrows();
        self.sums.len() == n_q && self.scales.len() == n_q
    }
}

/// Monomorphization dispatch for [`maxsim_neon`]:
/// `c = ceil(dim/16) = ceil(pdim/2) ∈ 1..=16`.
///
/// # Safety
/// Same contract as [`maxsim_neon`]: `dotprod` present, `q_pad` rows at
/// stride `q_stride ≥ c·16`, doc rows of `pdim` bytes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn maxsim_neon_dyn(
    c: usize,
    q_pad: &[i8],
    q_stride: usize,
    scales: &[f32],
    d_all: &[u8],
    pdim: usize,
    n_d: usize,
) -> f32 {
    macro_rules! arm {
        ($($n:literal),+) => {
            match c {
                $($n => maxsim_neon::<$n>(q_pad, q_stride, scales, d_all, pdim, n_d),)+
                _ => unreachable!("fused kernels cover dims <= 256"),
            }
        };
    }
    arm!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
}

/// Runtime gate for the fused AVX-512 VNNI kernel — single source of truth
/// shared by the dispatcher and the benchmarking hook.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512f")
}

/// Runtime gate for the fused AVX2 masked-SAD kernel. The `popcnt` check is
/// formal — no AVX2 CPU lacks it — but the kernel enables the feature, so the
/// gate must prove it.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_popcnt() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt")
}

/// Fast asymmetric MaxSim: int8 query against a packed 1-bit document.
///
/// Scores directly on the stored sign bits — no `f32` decode, integer ops
/// only (the `2P − T` identity on x86, native signed×signed SDOT on
/// aarch64). Per query token the best `q · s` over document tokens is found
/// in the integer domain, then scaled by that token's dequant scale and
/// summed (matching float MaxSim exactly up to the query's int8 rounding).
///
/// Dispatches once per call: byte-aligned dims (`dim % 8 == 0`, up to 256)
/// take a fused doc-token-outer kernel — AVX-512 VNNI (`maxsim_vnni`) or
/// AVX2 masked-SAD (`maxsim_avx2_sad`) on x86_64, SDOT (`maxsim_neon`) on
/// aarch64; ragged and larger dims take a per-pair SIMD dot with the feature
/// check hoisted out of the loops.
pub fn maxsim_binary_i8(query: &QueryI8, doc_packed: &ArrayView2<u8>, dim: usize) -> f32 {
    let n_doc = doc_packed.nrows();
    let n_q = query.values.nrows();
    if n_doc == 0 || n_q == 0 {
        return 0.0;
    }
    // Shape contract: a mismatch would otherwise become an out-of-bounds read
    // inside the raw-pointer fused kernels instead of a clean panic.
    assert_eq!(query.values.ncols(), dim, "query dim mismatch");
    assert_eq!(
        doc_packed.ncols(),
        packed_dim(dim),
        "doc rows must hold packed_dim(dim) bytes"
    );
    // Work on raw contiguous slices to avoid per-token ndarray view overhead in
    // the hot loop. Both arrays are row-major and standard-layout here.
    let q_all = query
        .values
        .as_slice()
        .expect("query values must be contiguous");
    let d_all = doc_packed.as_slice().expect("doc bits must be contiguous");

    // The fused kernels cover byte-aligned dims up to 256 (`fused_dim`): the
    // per-token masks stay register-resident (`ceil(dim/64) ≤ 4` zmm groups /
    // `ceil(dim/32) ≤ 8` ymm groups) and the NEON ±1 tile stays ≤ 1 KiB.
    // Ragged dims keep the per-pair path, which masks the last byte's
    // padding bits per dimension instead of relying on them being zero.
    #[cfg(target_arch = "x86_64")]
    {
        if fused_dim(dim) && query.layouts_consistent() {
            if has_avx512_vnni() && query.padded.len() == n_q * padded_stride(dim) {
                return unsafe {
                    maxsim_vnni_dyn(
                        dim.div_ceil(64),
                        &query.padded,
                        &query.sums,
                        &query.scales,
                        d_all,
                        packed_dim(dim),
                        n_doc,
                    )
                };
            }
            if has_avx2_popcnt() && query.biased.len() == n_q * biased_stride(dim) {
                return unsafe {
                    maxsim_avx2_sad_dyn(
                        dim.div_ceil(32),
                        &query.biased,
                        &query.sums,
                        &query.scales,
                        d_all,
                        packed_dim(dim),
                        n_doc,
                    )
                };
            }
        }
        if is_x86_feature_detected!("avx2") {
            // Other dims: per-pair AVX2 dot, detection hoisted out of the loop.
            let pdim = packed_dim(dim);
            let mut total = 0.0f32;
            for qi in 0..n_q {
                let q = &q_all[qi * dim..qi * dim + dim];
                let t = query.sums[qi];
                let mut best = i32::MIN;
                for d in 0..n_doc {
                    let bits = &d_all[d * pdim..d * pdim + pdim];
                    let score = unsafe { dot_pm1_avx2(q, t, bits, dim) };
                    if score > best {
                        best = score;
                    }
                }
                total += best as f32 * query.scales[qi];
            }
            return total;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if fused_dim(dim)
            && query.layouts_consistent()
            && query.padded.len() == n_q * padded_stride(dim)
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            return unsafe {
                maxsim_neon_dyn(
                    packed_dim(dim).div_ceil(2),
                    &query.padded,
                    padded_stride(dim),
                    &query.scales,
                    d_all,
                    packed_dim(dim),
                    n_doc,
                )
            };
        }
    }

    maxsim_binary_i8_pairwise_inner(query, q_all, d_all, n_doc, dim)
}

/// Portable per-pair path (NEON on aarch64, scalar elsewhere).
fn maxsim_binary_i8_pairwise_inner(
    query: &QueryI8,
    q_all: &[i8],
    d_all: &[u8],
    n_doc: usize,
    dim: usize,
) -> f32 {
    let pdim = packed_dim(dim);
    let n_q = query.values.nrows();
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let q = &q_all[qi * dim..qi * dim + dim];
        let t = query.sums[qi];
        let mut best = i32::MIN;
        for d in 0..n_doc {
            let bits = &d_all[d * pdim..d * pdim + pdim];
            let score = dot_pm1(q, t, bits, dim);
            if score > best {
                best = score;
            }
        }
        total += best as f32 * query.scales[qi];
    }
    total
}

/// The pre-fusion kernel shape: per-(query, doc-token) SIMD dot with no doc-bit
/// expansion reuse. Kept for benchmarking; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_pairwise(query: &QueryI8, doc_packed: &ArrayView2<u8>, dim: usize) -> f32 {
    let n_doc = doc_packed.nrows();
    if n_doc == 0 || query.values.nrows() == 0 {
        return 0.0;
    }
    let q_all = query.values.as_slice().expect("contiguous");
    let d_all = doc_packed.as_slice().expect("contiguous");
    maxsim_binary_i8_pairwise_inner(query, q_all, d_all, n_doc, dim)
}

/// True when the fused kernels can score `dim`: byte-aligned and at most 256
/// (see the dispatch note in [`maxsim_binary_i8`]).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn fused_dim(dim: usize) -> bool {
    dim.is_multiple_of(8) && (8..=256).contains(&dim)
}

/// Force the fused AVX-512 VNNI kernel. `None` when unsupported (needs
/// AVX-512 VNNI and a byte-aligned dim ≤ 256).
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_vnni(query: &QueryI8, doc_packed: &ArrayView2<u8>) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        let dim = query.values.ncols();
        let n_q = query.values.nrows();
        if fused_dim(dim)
            && doc_packed.ncols() == packed_dim(dim)
            && query.layouts_consistent()
            && query.padded.len() == n_q * padded_stride(dim)
            && has_avx512_vnni()
        {
            let n_doc = doc_packed.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_vnni_dyn(
                    dim.div_ceil(64),
                    &query.padded,
                    &query.sums,
                    &query.scales,
                    d_all,
                    packed_dim(dim),
                    n_doc,
                )
            });
        }
    }
    let _ = (query, doc_packed);
    None
}

/// Force the fused AVX2 masked-SAD kernel. `None` when unsupported (needs
/// AVX2 and a byte-aligned dim ≤ 256).
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_avx2_sad(
    query: &QueryI8,
    doc_packed: &ArrayView2<u8>,
) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        let dim = query.values.ncols();
        let n_q = query.values.nrows();
        if fused_dim(dim)
            && doc_packed.ncols() == packed_dim(dim)
            && query.layouts_consistent()
            && query.biased.len() == n_q * biased_stride(dim)
            && has_avx2_popcnt()
        {
            let n_doc = doc_packed.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_avx2_sad_dyn(
                    dim.div_ceil(32),
                    &query.biased,
                    &query.sums,
                    &query.scales,
                    d_all,
                    packed_dim(dim),
                    n_doc,
                )
            });
        }
    }
    let _ = (query, doc_packed);
    None
}

/// Force the fused NEON SDOT kernel. `None` when unsupported (needs the
/// `dotprod` feature and a byte-aligned dim ≤ 256).
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_neon(query: &QueryI8, doc_packed: &ArrayView2<u8>) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        let dim = query.values.ncols();
        let n_q = query.values.nrows();
        if fused_dim(dim)
            && doc_packed.ncols() == packed_dim(dim)
            && query.layouts_consistent()
            && query.padded.len() == n_q * padded_stride(dim)
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            let n_doc = doc_packed.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_neon_dyn(
                    packed_dim(dim).div_ceil(2),
                    &query.padded,
                    padded_stride(dim),
                    &query.scales,
                    d_all,
                    packed_dim(dim),
                    n_doc,
                )
            });
        }
    }
    let _ = (query, doc_packed);
    None
}

/// Number of packed bytes needed to store `dim` sign bits.
#[inline]
pub fn packed_dim(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Binarize embeddings to packed sign bits, big-endian within each byte.
///
/// Bit `1` encodes a non-negative component (`x >= 0`), bit `0` a negative one.
/// The result has shape `[n_tokens, packed_dim(dim)]` — a 32x reduction versus
/// the `4 * dim` bytes of the `f32` representation.
pub fn binarize(embeddings: &ArrayView2<f32>) -> Array2<u8> {
    let n = embeddings.nrows();
    let dim = embeddings.ncols();
    let pdim = packed_dim(dim);

    let mut packed = Array2::<u8>::zeros((n, pdim));
    let mut bits = Vec::with_capacity(dim);
    for (i, row) in embeddings.axis_iter(Axis(0)).enumerate() {
        bits.clear();
        bits.extend(row.iter().map(|&x| (x >= 0.0) as u8));
        for (j, byte) in packbits(&bits).into_iter().enumerate() {
            packed[[i, j]] = byte;
        }
    }
    packed
}

/// Decode packed sign bits back into a dense `±1` matrix of shape `[n, dim]`.
///
/// Padding bits beyond `dim` (present when `dim` is not a multiple of 8) are
/// dropped. Used for BLAS-accelerated scoring and as the correctness reference.
pub fn signs_pm1(packed: &ArrayView2<u8>, dim: usize) -> Array2<f32> {
    let n = packed.nrows();
    let mut out = Array2::<f32>::zeros((n, dim));
    for i in 0..n {
        let row = packed.row(i);
        for d in 0..dim {
            let byte = row[d / 8];
            let bit = (byte >> (7 - (d % 8))) & 1;
            out[[i, d]] = if bit == 1 { 1.0 } else { -1.0 };
        }
    }
    out
}

/// Asymmetric MaxSim between a full-precision query and a binary document.
///
/// The query keeps its `f32` (or int8-rounded, see [`quantize_query_int8`])
/// values; the document contributes only `±1` per dimension. Equivalent to
/// `maxsim_score(query, signs_pm1(doc))` but named to make the asymmetry
/// explicit at the call site.
pub fn maxsim_binary(query: &ArrayView2<f32>, doc_packed: &ArrayView2<u8>, dim: usize) -> f32 {
    let doc = signs_pm1(doc_packed, dim);
    maxsim_score(query, &doc.view())
}

/// Round a query to int8 precision (symmetric, per-row scale) and back to `f32`.
///
/// Models the "int8" query side of int8 x binary: the returned values carry the
/// quantization error but stay `f32` so they can feed [`maxsim_binary`] directly.
/// A per-row uniform scale does not by itself change MaxSim ranking; the effect
/// measured is purely the rounding error. Benchmarking support; not part of
/// the stability surface.
#[doc(hidden)]
pub fn quantize_query_int8(query: &ArrayView2<f32>) -> Array2<f32> {
    let mut out = query.to_owned();
    for mut row in out.axis_iter_mut(Axis(0)) {
        let max_abs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if max_abs <= 0.0 {
            continue;
        }
        let scale = max_abs / 127.0;
        for x in row.iter_mut() {
            *x = (*x / scale).round().clamp(-127.0, 127.0) * scale;
        }
    }
    out
}

/// MaxSim between a 1-bit query and a 1-bit document — the fastest tier.
///
/// With both sides in `{-1,+1}`, a token dot product is
/// `dim − 2 · popcount(q_bits XOR d_bits)`: agreements score +1, disagreements
/// −1. This is pure XOR + `popcount` over packed words (no per-dim work), the
/// article's "binary × binary" scheme. It throws away query magnitude, so it is
/// faster but lower-quality than [`maxsim_binary_i8`].
pub fn maxsim_binary_binary(
    query_packed: &ArrayView2<u8>,
    doc_packed: &ArrayView2<u8>,
    dim: usize,
) -> f32 {
    let n_q = query_packed.nrows();
    let n_d = doc_packed.nrows();
    if n_q == 0 || n_d == 0 {
        return 0.0;
    }
    let pdim = packed_dim(dim);
    let q_all = query_packed.as_slice().expect("query bits contiguous");
    let d_all = doc_packed.as_slice().expect("doc bits contiguous");
    let dim_i = dim as i32;
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let qb = &q_all[qi * pdim..qi * pdim + pdim];
        let mut best = i32::MIN;
        for d in 0..n_d {
            let db = &d_all[d * pdim..d * pdim + pdim];
            let mut ham = 0u32;
            for k in 0..pdim {
                ham += (qb[k] ^ db[k]).count_ones();
            }
            // Padding bits (dim not multiple of 8) are 0 in both when derived
            // from binarize on the same dim, so they XOR to 0 — no correction
            // needed here for the padded high bits of the last byte.
            let score = dim_i - 2 * ham as i32;
            if score > best {
                best = score;
            }
        }
        total += best as f32;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;

    fn random(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        Array::random_using(
            (rows, cols),
            StandardNormal,
            &mut StdRng::seed_from_u64(seed),
        )
    }

    #[test]
    fn packed_dim_rounds_up() {
        assert_eq!(packed_dim(128), 16);
        assert_eq!(packed_dim(8), 1);
        assert_eq!(packed_dim(13), 2);
    }

    #[test]
    fn binarize_recovers_signs() {
        let emb = random(5, 13, 1);
        let packed = binarize(&emb.view());
        assert_eq!(packed.ncols(), packed_dim(13));

        let decoded = signs_pm1(&packed.view(), 13);
        for i in 0..emb.nrows() {
            for d in 0..emb.ncols() {
                let expected = if emb[[i, d]] >= 0.0 { 1.0 } else { -1.0 };
                assert_eq!(decoded[[i, d]], expected, "mismatch at [{i},{d}]");
            }
        }
    }

    #[test]
    fn storage_is_32x_smaller() {
        let (n, dim) = (300, 128);
        let emb = random(n, dim, 2);
        let packed_bytes = binarize(&emb.view()).len();
        let float_bytes = n * dim * std::mem::size_of::<f32>();
        assert_eq!(float_bytes / packed_bytes, 32);
    }

    #[test]
    fn maxsim_binary_matches_bruteforce_sign_math() {
        // Independent reference: MaxSim of the query against raw sign(doc).
        let query = random(8, 64, 3);
        let doc = random(20, 64, 4);
        let packed = binarize(&doc.view());

        let mut reference = 0.0f32;
        for q in query.axis_iter(Axis(0)) {
            let mut best = f32::NEG_INFINITY;
            for d in doc.axis_iter(Axis(0)) {
                let s: f32 = q
                    .iter()
                    .zip(d.iter())
                    .map(|(&qv, &dv)| qv * if dv >= 0.0 { 1.0 } else { -1.0 })
                    .sum();
                best = best.max(s);
            }
            reference += best;
        }

        let got = maxsim_binary(&query.view(), &packed.view(), 64);
        assert!(
            (got - reference).abs() < 1e-3,
            "got {got}, want {reference}"
        );
    }

    #[test]
    fn int8_query_round_trip_is_bounded() {
        let query = random(4, 128, 5);
        let q8 = quantize_query_int8(&query.view());
        for (orig, quant) in query.axis_iter(Axis(0)).zip(q8.axis_iter(Axis(0))) {
            // Rounding error is at most half a quantization step: row_max / (2 * 127).
            let row_max = orig.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let tol = row_max / (2.0 * 127.0) + 1e-6;
            for (a, b) in orig.iter().zip(quant.iter()) {
                assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
            }
        }
    }

    /// Brute-force `q · sign(doc_bits)` in the integer domain, the reference the
    /// `2P − T` kernel must reproduce exactly (no floating point involved).
    fn ref_dot(q: &[i8], bits: &[u8], dim: usize) -> i32 {
        (0..dim)
            .map(|d| {
                let bit = (bits[d / 8] >> (7 - (d % 8))) & 1;
                if bit == 1 {
                    q[d] as i32
                } else {
                    -(q[d] as i32)
                }
            })
            .sum()
    }

    #[test]
    fn dot_pm1_matches_reference_all_paths() {
        // Cover multiple-of-16, multiple-of-8, and ragged tails.
        for &dim in &[16usize, 32, 64, 128, 13, 100, 127] {
            let doc = random(6, dim, 40 + dim as u64);
            let packed = binarize(&doc.view());
            let q8 = quantize_query_i8(&random(3, dim, 90 + dim as u64).view());
            for qi in 0..q8.values.nrows() {
                let q = q8.values.row(qi);
                let q = q.as_slice().unwrap();
                let t: i32 = q.iter().map(|&x| x as i32).sum();
                for d in 0..packed.nrows() {
                    let bits = packed.row(d);
                    let bits = bits.as_slice().unwrap();
                    let want = ref_dot(q, bits, dim);
                    let scal = dot_pm1_scalar(q, t, bits, dim);
                    let disp = dot_pm1(q, t, bits, dim); // NEON on aarch64
                    assert_eq!(scal, want, "scalar dim={dim}");
                    assert_eq!(disp, want, "dispatch/neon dim={dim}");
                }
            }
        }
    }

    #[test]
    fn fused_kernels_match_scalar_reference() {
        // The fused kernels (AVX-512 VNNI, AVX2 masked-SAD, NEON SDOT) and the
        // dispatched entry point must reproduce the integer-exact scalar
        // MaxSim for every byte-aligned dim shape: whole groups (64, 128,
        // 256), partial tail groups (16, 48, 96), odd packed-byte counts
        // (8, 72, 200), and the monomorphization bounds (8, 256). Doc counts
        // cover every 4-block tail shape (the fused kernels replicate the
        // last token to fill a block).
        for &dim in &[8usize, 16, 48, 64, 72, 96, 128, 200, 256] {
            let pdim = packed_dim(dim);
            for &n_doc in &[1usize, 2, 3, 4, 5, 13, 80] {
                let query = random(9, dim, 500 + dim as u64 + n_doc as u64);
                let doc = random(n_doc, dim, 600 + dim as u64 + n_doc as u64);
                let packed = binarize(&doc.view());
                let q8 = quantize_query_i8(&query.view());

                let q_all = q8.values.as_slice().unwrap();
                let d_all = packed.as_slice().unwrap();
                let mut want = 0.0f32;
                for qi in 0..9 {
                    let q = &q_all[qi * dim..qi * dim + dim];
                    let mut best = i32::MIN;
                    for d in 0..n_doc {
                        let s = ref_dot(q, &d_all[d * pdim..d * pdim + pdim], dim);
                        if s > best {
                            best = s;
                        }
                    }
                    want += best as f32 * q8.scales[qi];
                }

                let tol = 1e-4 * want.abs().max(1.0);
                let got = maxsim_binary_i8(&q8, &packed.view(), dim);
                assert!(
                    (got - want).abs() <= tol,
                    "dispatch dim={dim} n_doc={n_doc}: {got} vs {want}"
                );
                let pw = maxsim_binary_i8_pairwise(&q8, &packed.view(), dim);
                assert!(
                    (pw - want).abs() <= tol,
                    "pairwise dim={dim} n_doc={n_doc}: {pw} vs {want}"
                );
                // On hardware with the features, the hooks must engage for
                // every one of these dims — `is_some()` keeps this test from
                // silently losing its fused coverage — and must agree.
                if let Some(v) = maxsim_binary_i8_force_vnni(&q8, &packed.view()) {
                    assert!(
                        (v - want).abs() <= tol,
                        "vnni dim={dim} n_doc={n_doc}: {v} vs {want}"
                    );
                }
                if let Some(v) = maxsim_binary_i8_force_avx2_sad(&q8, &packed.view()) {
                    assert!(
                        (v - want).abs() <= tol,
                        "avx2-sad dim={dim} n_doc={n_doc}: {v} vs {want}"
                    );
                }
                if let Some(v) = maxsim_binary_i8_force_neon(&q8, &packed.view()) {
                    assert!(
                        (v - want).abs() <= tol,
                        "neon dim={dim} n_doc={n_doc}: {v} vs {want}"
                    );
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512_vnni() {
                        assert!(maxsim_binary_i8_force_vnni(&q8, &packed.view()).is_some());
                    }
                    if is_x86_feature_detected!("avx2") {
                        assert!(maxsim_binary_i8_force_avx2_sad(&q8, &packed.view()).is_some());
                    }
                }
                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    assert!(maxsim_binary_i8_force_neon(&q8, &packed.view()).is_some());
                }
            }
        }
    }

    #[test]
    fn query_i8_layouts_are_consistent() {
        // Strides exercise every rounding case: 48 (both round to 64), 96
        // (biased exact, padded rounds to 128), 128 (both exact), 130
        // (ragged dim still gets well-formed layouts).
        for &dim in &[48usize, 96, 128, 130] {
            let n = 7;
            let q8 = quantize_query_i8(&random(n, dim, 900 + dim as u64).view());
            let v = q8.values.as_slice().unwrap();
            for qi in 0..n {
                let t: i32 = v[qi * dim..(qi + 1) * dim].iter().map(|&x| x as i32).sum();
                assert_eq!(q8.sums[qi], t, "sums dim={dim}");
            }
            let bs = biased_stride(dim);
            let ps = padded_stride(dim);
            assert_eq!(q8.biased.len(), n * bs, "biased len dim={dim}");
            assert_eq!(q8.padded.len(), n * ps, "padded len dim={dim}");
            for qi in 0..n {
                for d in 0..dim {
                    assert_eq!(
                        q8.biased[qi * bs + d] as i32,
                        v[qi * dim + d] as i32 + 128,
                        "biased dim={dim} [{qi},{d}]"
                    );
                    assert_eq!(
                        q8.padded[qi * ps + d],
                        v[qi * dim + d],
                        "padded dim={dim} [{qi},{d}]"
                    );
                }
                assert!(
                    q8.biased[qi * bs + dim..(qi + 1) * bs]
                        .iter()
                        .all(|&b| b == 0),
                    "biased padding dim={dim} row {qi}"
                );
                assert!(
                    q8.padded[qi * ps + dim..(qi + 1) * ps]
                        .iter()
                        .all(|&p| p == 0),
                    "padded padding dim={dim} row {qi}"
                );
            }
        }
    }

    #[test]
    fn binary_binary_matches_hamming_reference() {
        for &dim in &[64usize, 128, 100, 127] {
            let query = random(5, dim, 200 + dim as u64);
            let doc = random(30, dim, 201 + dim as u64);
            let qb = binarize(&query.view());
            let db = binarize(&doc.view());
            let got = maxsim_binary_binary(&qb.view(), &db.view(), dim);

            // Reference: MaxSim of sign(query) against sign(doc), agreements +1.
            let mut want = 0.0f32;
            for q in query.axis_iter(Axis(0)) {
                let mut best = f32::NEG_INFINITY;
                for d in doc.axis_iter(Axis(0)) {
                    let s: f32 = q
                        .iter()
                        .zip(d.iter())
                        .map(|(&qv, &dv)| {
                            let qs = if qv >= 0.0 { 1.0 } else { -1.0 };
                            let ds = if dv >= 0.0 { 1.0 } else { -1.0 };
                            qs * ds
                        })
                        .sum();
                    best = best.max(s);
                }
                want += best;
            }
            assert!((got - want).abs() < 1e-3, "dim={dim}: {got} vs {want}");
        }
    }

    #[test]
    fn fast_kernel_matches_float_reference() {
        // The fast int8 kernel must equal float MaxSim over the SAME int8 codes,
        // i.e. maxsim_binary on the dequantized query. This isolates the kernel
        // from query-quantization error (that error is measured separately).
        // Dims cover fused whole/partial groups (64, 96, 128, 200), a ragged
        // per-pair dim (127), and a beyond-fused dim (320).
        for &dim in &[64usize, 96, 127, 128, 200, 320] {
            let query = random(12, dim, 7 + dim as u64);
            let doc = random(50, dim, 8 + dim as u64);
            let packed = binarize(&doc.view());

            let q8 = quantize_query_i8(&query.view());
            let got = maxsim_binary_i8(&q8, &packed.view(), dim);

            // Dequantize the int8 codes back to f32 and score with the reference.
            let mut dq = Array2::<f32>::zeros((query.nrows(), dim));
            for i in 0..query.nrows() {
                for d in 0..dim {
                    dq[[i, d]] = q8.values[[i, d]] as f32 * q8.scales[i];
                }
            }
            let want = maxsim_binary(&dq.view(), &packed.view(), dim);
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "dim={dim}: fast {got} vs reference {want}"
            );
        }
    }
}
