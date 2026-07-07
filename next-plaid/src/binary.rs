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
//! Scoring kernels (all exact w.r.t. the stored signs; benchmarked in
//! `examples/kernel_bench.rs`):
//!   * [`maxsim_binary_i8`] — the search-path default. Dispatches to a fused
//!     doc-token-outer kernel for `dim == 128` (AVX-512 VNNI `vpdpbusd` or
//!     AVX2 masked-SAD on x86_64, SDOT on aarch64), else a per-pair bit-native
//!     `2P − T` SIMD dot (NEON / AVX2 / scalar).
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
    /// Row-major biased codes (`x ^ 0x80`, i.e. `x + 128` viewed as `u8`) —
    /// the query layout consumed by the AVX2 masked-SAD kernel, where the sum
    /// of selected biased bytes is `P + 128 · popcount(doc bits)`.
    pub biased: Vec<u8>,
    /// Plane-major query codes (`planes[qi*128 + p*16 + k] = values[qi][k*8+p]`)
    /// — the layout matching `extract_planes_128`'s doc-side expansion,
    /// consumed by the fused NEON SDOT kernel.
    /// Populated only for `dim == 128`; empty otherwise.
    pub planes: Vec<i8>,
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
    let biased = v.iter().map(|&x| (x as u8) ^ 0x80).collect();
    let planes = if dim == 128 {
        let mut p = vec![0i8; n * 128];
        for qi in 0..n {
            for pl in 0..8 {
                for k in 0..16 {
                    p[qi * 128 + pl * 16 + k] = v[qi * 128 + k * 8 + pl];
                }
            }
        }
        p
    } else {
        Vec::new()
    };
    QueryI8 {
        values,
        scales,
        sums,
        biased,
        planes,
    }
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

/// AVX-512 VNNI fused MaxSim for `dim == 128`.
///
/// Doc-token-outer loop: each token's 128 sign bits are expanded ONCE, in
/// registers, to two zmm vectors of `0/1` bytes (broadcast + `pshufb` +
/// `vptestmb` + `maskz_mov` — ~10 instructions), amortized over every query
/// token. Scoring is two `vpdpbusd` (u8 × i8 dot-accumulate — the x86 twin of
/// the NEON SDOT in mixedbread's kernel): `P = Σ mask_d · q_d`, `score = 2P − T`.
/// Doc tokens go in blocks of 4 so the horizontal reductions collapse into
/// `phaddd` pairs and the per-query running max stays vectorized (`pmaxsd`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn maxsim_vnni128(
    q_all: &[i8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    n_q: usize,
    n_d: usize,
) -> f32 {
    use std::arch::x86_64::*;
    // After `broadcast_i32x4` every 128-bit chunk holds all 16 packed bytes;
    // chunk c of the LOW zmm covers dims 16c..16c+16 = packed bytes 2c, 2c+1
    // (HIGH zmm: bytes 8..16). Indices replicate byte k across its 8 lanes.
    const IDX_LO: [i8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7,
    ];
    const IDX_HI: [i8; 64] = [
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    // MSB-first bit selector per lane group (dim 8k -> 0x80), as in `binarize`.
    const SEL: [i8; 64] = [
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08,
        0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10,
        0x08, 0x04, 0x02, 0x01,
    ];
    let idx_lo = _mm512_loadu_si512(IDX_LO.as_ptr() as *const __m512i);
    let idx_hi = _mm512_loadu_si512(IDX_HI.as_ptr() as *const __m512i);
    let sel = _mm512_loadu_si512(SEL.as_ptr() as *const __m512i);
    let one = _mm512_set1_epi8(1);

    // Expand one 16-byte packed token into two zmm of 0/1 bytes.
    macro_rules! expand {
        ($ptr:expr) => {{
            let bc = _mm512_broadcast_i32x4(_mm_loadu_si128($ptr as *const __m128i));
            let klo = _mm512_test_epi8_mask(_mm512_shuffle_epi8(bc, idx_lo), sel);
            let khi = _mm512_test_epi8_mask(_mm512_shuffle_epi8(bc, idx_hi), sel);
            (
                _mm512_maskz_mov_epi8(klo, one),
                _mm512_maskz_mov_epi8(khi, one),
            )
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
    let dp = d_all.as_ptr();
    let qp = q_all.as_ptr();
    let mut db = 0usize;
    while db < n_d {
        let i1 = (db + 1).min(n_d - 1);
        let i2 = (db + 2).min(n_d - 1);
        let i3 = (db + 3).min(n_d - 1);
        let (m00, m01) = expand!(dp.add(db * 16));
        let (m10, m11) = expand!(dp.add(i1 * 16));
        let (m20, m21) = expand!(dp.add(i2 * 16));
        let (m30, m31) = expand!(dp.add(i3 * 16));
        for (qi, &sum) in sums.iter().enumerate() {
            let q0 = _mm512_loadu_si512(qp.add(qi * 128) as *const __m512i);
            let q1 = _mm512_loadu_si512(qp.add(qi * 128 + 64) as *const __m512i);
            let z = _mm512_setzero_si512();
            let a0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(z, m00, q0), m01, q1);
            let a1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(z, m10, q0), m11, q1);
            let a2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(z, m20, q0), m21, q1);
            let a3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(z, m30, q0), m31, q1);
            // [P0, P1, P2, P3] for the 4 doc tokens of this block.
            let h01 = _mm_hadd_epi32(fold4!(a0), fold4!(a1));
            let h23 = _mm_hadd_epi32(fold4!(a2), fold4!(a3));
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

/// AVX2 masked-SAD fused MaxSim for `dim == 128` (no AVX-512 required).
///
/// Doc-token-outer: expand each token's bits ONCE into four ymm `0xFF/0x00`
/// masks (broadcast + `pshufb` + `pcmpeqb`), amortized over all query tokens.
/// Scoring uses the biased-SAD identity: with `qb = q + 128` stored as `u8`,
/// `SAD(qb & mask, 0) = P + 128 · popcount(bits)`, so
/// `P = SAD − 128·popcount` and `score = 2P − T`. Every scoring op
/// (`pand`/`psadbw`/`paddq`) is a cheap 1-µop instruction — no widening chains.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxsim_avx2_sad128(
    qb_all: &[u8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    n_q: usize,
    n_d: usize,
) -> f32 {
    use std::arch::x86_64::*;
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

    // 0xFF mask ymm for dims 32g..32g+32 of the token at `bp`.
    macro_rules! mask32 {
        ($bp:expr, $g:expr) => {{
            let w = ($bp.add($g * 4) as *const u32).read_unaligned();
            let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
            _mm256_cmpeq_epi8(_mm256_and_si256(bytes, sel), sel)
        }};
    }

    let mut best = vec![i32::MIN; n_q];
    let qp = qb_all.as_ptr();
    for d in 0..n_d {
        let bp = d_all.as_ptr().add(d * 16);
        let m0 = mask32!(bp, 0);
        let m1 = mask32!(bp, 1);
        let m2 = mask32!(bp, 2);
        let m3 = mask32!(bp, 3);
        let cnt = ((bp as *const u64).read_unaligned().count_ones()
            + (bp.add(8) as *const u64).read_unaligned().count_ones()) as i32;
        for qi in 0..n_q {
            let q = qp.add(qi * 128);
            let s0 = _mm256_sad_epu8(
                _mm256_and_si256(m0, _mm256_loadu_si256(q as *const __m256i)),
                zero,
            );
            let s1 = _mm256_sad_epu8(
                _mm256_and_si256(m1, _mm256_loadu_si256(q.add(32) as *const __m256i)),
                zero,
            );
            let s2 = _mm256_sad_epu8(
                _mm256_and_si256(m2, _mm256_loadu_si256(q.add(64) as *const __m256i)),
                zero,
            );
            let s3 = _mm256_sad_epu8(
                _mm256_and_si256(m3, _mm256_loadu_si256(q.add(96) as *const __m256i)),
                zero,
            );
            let s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
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

/// NEON SDOT fused MaxSim for `dim == 128` (aarch64 with `dotprod`).
///
/// The ARM analog of `maxsim_vnni128`: doc-token-outer, so each token's 128
/// sign bits are expanded ONCE — 8 NEON shift+mask ops into the plane-major
/// `0/1` layout of `extract_planes_128` — and amortized over every query
/// token. Scoring SDOTs the hoisted plane-major query row (built once per
/// query in [`quantize_query_i8`]) against the expanded planes:
/// `P = Σ mask_d · q_d`, `score = 2P − T`. Doc tokens go in blocks of 4 so the
/// four SDOT accumulator chains stay independent, the horizontal reductions
/// collapse into `vpaddq` pairs, and the per-query running max stays
/// vectorized (`vmaxq_s32`).
///
/// The expansion lives in a 512-byte stack tile, never the index: doc bytes
/// read per token stay 16 (packed bits), not 128 (stored planes), which is
/// what wins once the candidate set outgrows L2 — index-time plane extraction
/// measured no faster than this on Apple M4 while costing 8× the memory.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn maxsim_neon128(
    q_planes: &[i8],
    sums: &[i32],
    scales: &[f32],
    d_all: &[u8],
    n_q: usize,
    n_d: usize,
) -> f32 {
    use std::arch::aarch64::*;

    use crate::int8::sdot_asm;

    debug_assert_eq!(q_planes.len(), n_q * 128);
    // Per-query running max over doc tokens, 4 lanes per query token (one lane
    // per doc slot in the block). Tail blocks replicate the last token, which
    // cannot change a max.
    let mut best = vec![i32::MIN; n_q * 4];
    let mut planes = [0i8; 4 * 128];
    let mut db = 0usize;
    while db < n_d {
        for t in 0..4 {
            let d = (db + t).min(n_d - 1);
            let m: &mut [i8; 128] = (&mut planes[t * 128..t * 128 + 128])
                .try_into()
                .expect("128-byte tile slot");
            extract_planes_128(&d_all[d * 16..d * 16 + 16], m);
        }
        let pp = planes.as_ptr();
        for (qi, &sum) in sums.iter().enumerate() {
            let qp = q_planes.as_ptr().add(qi * 128);
            let q0 = vld1q_s8(qp);
            let q1 = vld1q_s8(qp.add(16));
            let q2 = vld1q_s8(qp.add(32));
            let q3 = vld1q_s8(qp.add(48));
            let q4 = vld1q_s8(qp.add(64));
            let q5 = vld1q_s8(qp.add(80));
            let q6 = vld1q_s8(qp.add(96));
            let q7 = vld1q_s8(qp.add(112));
            // One doc token: 8 SDOTs over two accumulators (depth 4 per chain,
            // enough to hide SDOT latency across the 4 tokens of the block).
            macro_rules! tok {
                ($off:expr) => {{
                    let mut a = vdupq_n_s32(0);
                    let mut b = vdupq_n_s32(0);
                    a = sdot_asm(a, q0, vld1q_s8(pp.add($off)));
                    b = sdot_asm(b, q1, vld1q_s8(pp.add($off + 16)));
                    a = sdot_asm(a, q2, vld1q_s8(pp.add($off + 32)));
                    b = sdot_asm(b, q3, vld1q_s8(pp.add($off + 48)));
                    a = sdot_asm(a, q4, vld1q_s8(pp.add($off + 64)));
                    b = sdot_asm(b, q5, vld1q_s8(pp.add($off + 80)));
                    a = sdot_asm(a, q6, vld1q_s8(pp.add($off + 96)));
                    b = sdot_asm(b, q7, vld1q_s8(pp.add($off + 112)));
                    vaddq_s32(a, b)
                }};
            }
            let t0 = tok!(0);
            let t1 = tok!(128);
            let t2 = tok!(256);
            let t3 = tok!(384);
            // Pairwise-add tree -> [P0, P1, P2, P3] for the block's 4 tokens.
            let p4 = vpaddq_s32(vpaddq_s32(t0, t1), vpaddq_s32(t2, t3));
            let sc = vsubq_s32(vshlq_n_s32::<1>(p4), vdupq_n_s32(sum));
            let bp = best.as_mut_ptr().add(qi * 4);
            vst1q_s32(bp, vmaxq_s32(vld1q_s32(bp), sc));
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

/// Runtime gate for the fused AVX-512 VNNI kernel — single source of truth
/// shared by the dispatcher and the benchmarking hook.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512f")
}

/// Fast asymmetric MaxSim: int8 query against a packed 1-bit document.
///
/// Scores directly on the stored sign bits — no `f32` decode, integer adds
/// only via the `2P − T` identity. Per query token the best `q · s` over
/// document tokens is found in the integer domain, then scaled by that token's
/// dequant scale and summed (matching float MaxSim exactly up to the query's
/// int8 rounding).
///
/// Dispatches once per call: for `dim == 128` this takes a fused
/// doc-token-outer kernel — AVX-512 VNNI (`maxsim_vnni128`) or AVX2
/// masked-SAD (`maxsim_avx2_sad128`) on x86_64, SDOT (`maxsim_neon128`)
/// on aarch64; other dims take a per-pair SIMD dot with the feature check
/// hoisted out of the loops.
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

    #[cfg(target_arch = "x86_64")]
    {
        if dim == 128 && query.layouts_consistent() {
            if has_avx512_vnni() {
                return unsafe {
                    maxsim_vnni128(q_all, &query.sums, &query.scales, d_all, n_q, n_doc)
                };
            }
            if is_x86_feature_detected!("avx2") && query.biased.len() == n_q * 128 {
                return unsafe {
                    maxsim_avx2_sad128(&query.biased, &query.sums, &query.scales, d_all, n_q, n_doc)
                };
            }
        }
        if is_x86_feature_detected!("avx2") {
            // Generic dims: per-pair AVX2 dot, detection hoisted out of the loop.
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
        if dim == 128
            && query.layouts_consistent()
            && query.planes.len() == n_q * 128
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            return unsafe {
                maxsim_neon128(&query.planes, &query.sums, &query.scales, d_all, n_q, n_doc)
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

/// Force the fused AVX-512 VNNI kernel (dim=128). `None` when unsupported.
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_vnni(query: &QueryI8, doc_packed: &ArrayView2<u8>) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if query.values.ncols() == 128
            && doc_packed.ncols() == 16
            && query.layouts_consistent()
            && has_avx512_vnni()
        {
            let n_doc = doc_packed.nrows();
            let n_q = query.values.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let q_all = query.values.as_slice().expect("contiguous");
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_vnni128(q_all, &query.sums, &query.scales, d_all, n_q, n_doc)
            });
        }
    }
    let _ = (query, doc_packed);
    None
}

/// Force the fused AVX2 masked-SAD kernel (dim=128). `None` when unsupported.
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_avx2_sad(
    query: &QueryI8,
    doc_packed: &ArrayView2<u8>,
) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if query.values.ncols() == 128
            && doc_packed.ncols() == 16
            && query.layouts_consistent()
            && query.biased.len() == query.values.nrows() * 128
            && is_x86_feature_detected!("avx2")
        {
            let n_doc = doc_packed.nrows();
            let n_q = query.values.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_avx2_sad128(&query.biased, &query.sums, &query.scales, d_all, n_q, n_doc)
            });
        }
    }
    let _ = (query, doc_packed);
    None
}

/// Force the fused NEON SDOT kernel (dim=128). `None` when unsupported.
/// Benchmarking hook; not part of the stability surface.
#[doc(hidden)]
pub fn maxsim_binary_i8_force_neon(query: &QueryI8, doc_packed: &ArrayView2<u8>) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        if query.values.ncols() == 128
            && doc_packed.ncols() == 16
            && query.layouts_consistent()
            && query.planes.len() == query.values.nrows() * 128
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            let n_doc = doc_packed.nrows();
            let n_q = query.values.nrows();
            if n_doc == 0 || n_q == 0 {
                return Some(0.0);
            }
            let d_all = doc_packed.as_slice().expect("contiguous");
            return Some(unsafe {
                maxsim_neon128(&query.planes, &query.sums, &query.scales, d_all, n_q, n_doc)
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

/// Extract the 8 bit-planes of one 128-dim (16-byte) packed doc token into a
/// plane-major `0/1` `i8` buffer `m[128]`: `m[p*16 + k] = bit p of byte k`
/// (MSB-first, so plane `p` holds dim `k*8 + p`).
///
/// On aarch64 this is 8 NEON shift+mask ops over the whole 16-byte vector —
/// one shift yields one bit from all 16 bytes at once, hence the plane-major
/// layout (an extraction scheme due to mixedbread's aarch64 kernel). Doc-side
/// expansion step of `maxsim_neon128`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn extract_planes_128(bits: &[u8], m: &mut [i8; 128]) {
    use std::arch::aarch64::*;
    unsafe {
        let v = vld1q_u8(bits.as_ptr());
        let one = vdupq_n_u8(1);
        let p = m.as_mut_ptr();
        vst1q_s8(
            p.add(0),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<7>(v), one)),
        );
        vst1q_s8(
            p.add(16),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<6>(v), one)),
        );
        vst1q_s8(
            p.add(32),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<5>(v), one)),
        );
        vst1q_s8(
            p.add(48),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(v), one)),
        );
        vst1q_s8(
            p.add(64),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<3>(v), one)),
        );
        vst1q_s8(
            p.add(80),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(v), one)),
        );
        vst1q_s8(
            p.add(96),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<1>(v), one)),
        );
        vst1q_s8(p.add(112), vreinterpretq_s8_u8(vandq_u8(v, one)));
    }
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
        // The fused dim=128 kernels (AVX-512 VNNI, AVX2 masked-SAD) and the
        // dispatched entry point must reproduce the integer-exact scalar MaxSim.
        // Doc counts cover every 4-block tail shape (VNNI replicates the last
        // token to fill a block).
        for &n_doc in &[1usize, 2, 3, 4, 5, 13, 80] {
            let query = random(9, 128, 500 + n_doc as u64);
            let doc = random(n_doc, 128, 600 + n_doc as u64);
            let packed = binarize(&doc.view());
            let q8 = quantize_query_i8(&query.view());

            let q_all = q8.values.as_slice().unwrap();
            let d_all = packed.as_slice().unwrap();
            let mut want = 0.0f32;
            for qi in 0..9 {
                let q = &q_all[qi * 128..qi * 128 + 128];
                let mut best = i32::MIN;
                for d in 0..n_doc {
                    let s = ref_dot(q, &d_all[d * 16..d * 16 + 16], 128);
                    if s > best {
                        best = s;
                    }
                }
                want += best as f32 * q8.scales[qi];
            }

            let tol = 1e-4 * want.abs().max(1.0);
            let got = maxsim_binary_i8(&q8, &packed.view(), 128);
            assert!(
                (got - want).abs() <= tol,
                "dispatch n_doc={n_doc}: {got} vs {want}"
            );
            let pw = maxsim_binary_i8_pairwise(&q8, &packed.view(), 128);
            assert!(
                (pw - want).abs() <= tol,
                "pairwise n_doc={n_doc}: {pw} vs {want}"
            );
            if let Some(v) = maxsim_binary_i8_force_vnni(&q8, &packed.view()) {
                assert!((v - want).abs() <= tol, "vnni n_doc={n_doc}: {v} vs {want}");
            }
            if let Some(v) = maxsim_binary_i8_force_avx2_sad(&q8, &packed.view()) {
                assert!(
                    (v - want).abs() <= tol,
                    "avx2-sad n_doc={n_doc}: {v} vs {want}"
                );
            }
            if let Some(v) = maxsim_binary_i8_force_neon(&q8, &packed.view()) {
                assert!((v - want).abs() <= tol, "neon n_doc={n_doc}: {v} vs {want}");
            }
        }
    }

    #[test]
    fn query_i8_sums_and_biased_are_consistent() {
        let q8 = quantize_query_i8(&random(7, 128, 900).view());
        let v = q8.values.as_slice().unwrap();
        for qi in 0..7 {
            let t: i32 = v[qi * 128..(qi + 1) * 128].iter().map(|&x| x as i32).sum();
            assert_eq!(q8.sums[qi], t);
        }
        for (i, &b) in q8.biased.iter().enumerate() {
            assert_eq!(b as i32, v[i] as i32 + 128);
        }
        // Plane-major layout: planes[qi*128 + p*16 + k] holds dim k*8 + p.
        assert_eq!(q8.planes.len(), 7 * 128);
        for qi in 0..7 {
            for p in 0..8 {
                for k in 0..16 {
                    assert_eq!(q8.planes[qi * 128 + p * 16 + k], v[qi * 128 + k * 8 + p]);
                }
            }
        }
        // Non-128 dims skip the plane layout.
        assert!(quantize_query_i8(&random(3, 64, 901).view())
            .planes
            .is_empty());
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
        for &dim in &[64usize, 128, 127] {
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
