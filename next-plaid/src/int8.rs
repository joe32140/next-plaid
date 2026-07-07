//! Symmetric int8 × int8 MaxSim — the "near-lossless, 4× smaller" tier.
//!
//! Both query and document tokens are stored as `i8` (1 byte/dim, 4× smaller
//! than `f32`). Scoring uses a hardware integer dot product per token:
//!   * aarch64: `SDOT` (via inline asm, since `vdotq_s32` is nightly-only)
//!   * x86_64:  sign-extend `i8→i16` then `_mm256_madd_epi16` under AVX2
//!   * else:    scalar
//!
//! This is the natural upper-quality comparator for the binary tiers: it keeps
//! the article's "int8 ≈ float" claim honest while still compressing 4×. The
//! per-token dequant scale is applied to the integer dot to recover the true
//! MaxSim value.
//!
//! IMPORTANT correctness note: the common `_mm256_maddubs_epi16`-based AVX2
//! int8 dot product SATURATES its i16 intermediate for full-range `i8` inputs
//! and is numerically wrong. This module deliberately uses the slower but exact
//! sign-extend + `madd_epi16` path.

use ndarray::{Array2, ArrayView2, Axis};

/// A per-row int8-quantized embedding matrix plus per-row dequant scale.
pub struct Quantized {
    pub values: Array2<i8>,
    pub scales: Vec<f32>,
}

/// Symmetric per-row int8 quantization (largest magnitude maps to ±127).
pub fn quantize_i8(m: &ArrayView2<f32>) -> Quantized {
    let n = m.nrows();
    let dim = m.ncols();
    let mut values = Array2::<i8>::zeros((n, dim));
    let mut scales = vec![0.0f32; n];
    for (i, row) in m.axis_iter(Axis(0)).enumerate() {
        let max_abs = row.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_abs <= 0.0 {
            continue;
        }
        let scale = max_abs / 127.0;
        scales[i] = scale;
        for (d, &x) in row.iter().enumerate() {
            values[[i, d]] = (x / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    Quantized { values, scales }
}

/// Scalar reference int8 dot product accumulating into i32.
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

/// AVX2 int8 dot product: sign-extend to i16, then `madd_epi16` (exact, no
/// saturation). Processes 16 dims/iteration; scalar tail for the remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= n {
        // Load 16 i8 lanes each, sign-extend to i16 (256-bit = 16×i16).
        let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        // madd: pairwise i16×i16 -> i32, summed in pairs. Exact for i8 range.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
        i += 16;
    }
    // Horizontal sum of 8 i32 lanes.
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_00_01));
    let mut dot = _mm_cvtsi128_si32(sum32);
    for k in i..n {
        dot += a[k] as i32 * b[k] as i32;
    }
    dot
}

/// NEON `SDOT` int8 dot product via inline asm (stable-Rust safe; the
/// `vdotq_s32` intrinsic requires nightly). Processes 16 dims per `sdot`,
/// 4 accumulators to hide latency; scalar tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    let mut i = 0;
    while i + 64 <= n {
        let (q0, d0) = (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
        let (q1, d1) = (
            vld1q_s8(a.as_ptr().add(i + 16)),
            vld1q_s8(b.as_ptr().add(i + 16)),
        );
        let (q2, d2) = (
            vld1q_s8(a.as_ptr().add(i + 32)),
            vld1q_s8(b.as_ptr().add(i + 32)),
        );
        let (q3, d3) = (
            vld1q_s8(a.as_ptr().add(i + 48)),
            vld1q_s8(b.as_ptr().add(i + 48)),
        );
        // sdot accumulates 4 groups of 4 signed-byte products into each i32 lane.
        acc0 = sdot_asm(acc0, q0, d0);
        acc1 = sdot_asm(acc1, q1, d1);
        acc2 = sdot_asm(acc2, q2, d2);
        acc3 = sdot_asm(acc3, q3, d3);
        i += 64;
    }
    while i + 16 <= n {
        let q = vld1q_s8(a.as_ptr().add(i));
        let d = vld1q_s8(b.as_ptr().add(i));
        acc0 = sdot_asm(acc0, q, d);
        i += 16;
    }
    let acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
    let mut dot = vaddvq_s32(acc);
    for k in i..n {
        dot += a[k] as i32 * b[k] as i32;
    }
    dot
}

/// `sdot vD.4s, vN.16b, vM.16b` — signed int8 dot product into i32x4, via inline
/// asm because the `vdotq_s32` intrinsic is still nightly-only.
///
/// The single definition of the SDOT wrapper for the whole crate (the binary
/// module's bit-plane kernel reuses this). `options(pure, nomem, nostack)` is
/// sound: the instruction reads only its three register operands and writes one
/// register — no memory, no stack — and is a pure function of its inputs.
///
/// # Safety
/// Requires the `dotprod` target feature at runtime (caller must check).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn sdot_asm(
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

/// Public wrapper for the dispatched int8 dot product (used by the binary
/// module's SDOT-based int8×1-bit kernel to score unpacked ±1 doc tokens).
#[inline]
pub fn dot_i8_pub(a: &[i8], b: &[i8]) -> i32 {
    dot_i8(a, b)
}

/// Architecture-dispatching int8×int8 token dot product.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot_i8_neon(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

/// int8 × int8 MaxSim: both sides quantized. Returns the dequantized MaxSim
/// score (per query-token best match, scaled by query and doc scales, summed).
pub fn maxsim_i8(query: &Quantized, doc: &Quantized, dim: usize) -> f32 {
    let n_q = query.values.nrows();
    let n_d = doc.values.nrows();
    if n_q == 0 || n_d == 0 {
        return 0.0;
    }
    let q_all = query.values.as_slice().expect("contiguous");
    let d_all = doc.values.as_slice().expect("contiguous");
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let q = &q_all[qi * dim..qi * dim + dim];
        let qs = query.scales[qi];
        let mut best = f32::NEG_INFINITY;
        for di in 0..n_d {
            let d = &d_all[di * dim..di * dim + dim];
            // Dequantized dot = int_dot * q_scale * d_scale.
            let sim = dot_i8(q, d) as f32 * qs * doc.scales[di];
            if sim > best {
                best = sim;
            }
        }
        if best.is_finite() {
            total += best;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;

    fn rand_i8(rows: usize, cols: usize, seed: u64) -> Array2<i8> {
        // Full-range i8 including boundaries, to expose saturation bugs.
        let f: Array2<f32> = Array::random_using(
            (rows, cols),
            Uniform::new(-128.0, 128.0),
            &mut StdRng::seed_from_u64(seed),
        );
        f.mapv(|x| x.clamp(-128.0, 127.0) as i8)
    }

    #[test]
    fn dot_i8_simd_matches_scalar_full_range() {
        for &dim in &[16usize, 32, 64, 128, 100, 127] {
            let a = rand_i8(4, dim, 10 + dim as u64);
            let b = rand_i8(4, dim, 20 + dim as u64);
            for i in 0..4 {
                let ar = a.row(i);
                let br = b.row(i);
                let (ar, br) = (ar.as_slice().unwrap(), br.as_slice().unwrap());
                let want = dot_i8_scalar(ar, br);
                let got = dot_i8(ar, br);
                assert_eq!(got, want, "dim={dim} row={i}");
            }
        }
    }

    #[test]
    fn dot_i8_extremes() {
        // -128 * -128 * 128 dims = 2_097_152, well within i32.
        let a = vec![-128i8; 128];
        let b = vec![-128i8; 128];
        assert_eq!(dot_i8(&a, &b), 128 * 128 * 128);
        assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn maxsim_i8_matches_float_reference() {
        use crate::maxsim::maxsim_score;
        for &dim in &[64usize, 128] {
            let qf: Array2<f32> = Array::random_using(
                (10, dim),
                ndarray_rand::rand_distr::StandardNormal,
                &mut StdRng::seed_from_u64(1 + dim as u64),
            );
            let df: Array2<f32> = Array::random_using(
                (40, dim),
                ndarray_rand::rand_distr::StandardNormal,
                &mut StdRng::seed_from_u64(2 + dim as u64),
            );
            let q = quantize_i8(&qf.view());
            let d = quantize_i8(&df.view());
            let got = maxsim_i8(&q, &d, dim);

            // Reference: dequantize both, run float MaxSim.
            let dq = |quant: &Quantized| -> Array2<f32> {
                let mut out = Array2::<f32>::zeros(quant.values.raw_dim());
                for i in 0..quant.values.nrows() {
                    for j in 0..quant.values.ncols() {
                        out[[i, j]] = quant.values[[i, j]] as f32 * quant.scales[i];
                    }
                }
                out
            };
            let want = maxsim_score(&dq(&q).view(), &dq(&d).view());
            assert!(
                (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                "dim={dim}: {got} vs {want}"
            );
        }
    }
}
