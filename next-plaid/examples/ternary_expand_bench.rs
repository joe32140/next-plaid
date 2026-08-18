//! Can ternary's byte→weights expansion beat the shipped one-hop table copy?
//!
//! **Measured answer: no, and not by a little.** This exists so that stays
//! reproducible, because it is the obvious optimization to propose and the
//! reasoning behind it is sound right up to the point where it is timed.
//!
//! The shipped route (A) turns each base-3 byte into five int8 weights with one
//! unaligned 8-byte copy out of a 256×8 table, in **natural dim order**. It is
//! the one stage where ternary still pays more than 2-bit (which does a `tbl`
//! per key position per 16 bytes), so it is where any remaining e2e gap lives.
//!
//! The proposal is to stop looking trits up and *compute* them. The "243 wall"
//! that blocks the nibble route is a claim about table shuffles: no
//! `tbl`/`pshufb` reaches past 16 entries. Arithmetic has no such limit —
//! `b / 3^k` is an exact multiply-shift on u8, and the five quotients are
//! computed *independently* from `b`, so there is no serial chain:
//!
//! ```text
//!   qk = b / 3^k  for k = 1..5, each one independent multiply-shift
//!   tk = q(k) - 3*q(k+1)          for k = 0..4, with q0 = b
//! ```
//!
//! Note `t4 = q4 - 3·q5`, not `q4`. The codec's table is iterated (`t = v%3;
//! v/=3`), so it defines `t4 = (b/81) % 3`, which is **0** for bytes 243..=255
//! — the 13 the encoder never emits but that the decode table still covers.
//! Dropping the final `%3` gives 3 there and silently disagrees with the scalar
//! reference on exactly those bytes. Two `b/81` magics that look right are also
//! wrong: `(b*51)>>12` first fails at b=161 and `(b*13)>>10` at b=79, both
//! *valid* bytes. The magics are asserted exhaustively over all 256 in `main`.
//!
//! Arm C additionally exploits the shipped codebook: in τ-mode the weights are
//! `{−m, 0, +m}`, so their int8 quantization is exactly `{−127, 0, +127}` —
//! `w = 127·(t−1)`. Fold the 127 into the per-query `sqw` and the weight table
//! disappears.
//!
//! ## What it measures (aarch64, Apple M4, ns/token, A = 1.00×)
//!
//! ```text
//!   pdim  tail      A one-hop   B arith+tbl   C arith+affine
//!     48  none         16.24    17.40 0.93x   17.51 0.93x
//!     16  none          4.77     5.17 0.92x    5.26 0.91x
//!     26  10 bytes     11.84    28.90 0.41x   28.95 0.41x   <- dim 128
//!     10  all tail      2.97    14.96 0.20x   14.89 0.20x   <- dim 48
//! ```
//!
//! Two findings, and the second is the general one:
//!
//! 1. **With no tail at all, arithmetic still loses by ~8%.** pdim 16 and 48
//!    tile the vector register exactly, and B trails anyway — so ~0.93× is its
//!    ceiling however well the tail is written. A at pdim 16 does 16 load/store
//!    pairs in 4.77 ns ≈ 1 cycle/byte: it is store-throughput-bound, and there
//!    is no headroom to take.
//! 2. **Base 3 is hostile to the layout, not just to the lookup.** Planar
//!    output needs plane strides that are multiples of the vector width, and
//!    `ceil(dim/5)` essentially never is — so both shipping dims (128 → pdim
//!    26, 48 → pdim 10) land in ragged-tail territory. Natural dim order
//!    sidesteps the 243-entry table *and* the ragged planes at once. Padding
//!    plane strides up to a multiple of 16 would fix the tail at the cost of
//!    25% zero lanes in the dot — far more than the 1–2% e2e at stake.
//!
//! Arm C tracks arm B to within 0.1 ns everywhere: `vqtbl1q` and `vsubq` are
//! both one op, so removing the weight table saves nothing. The identity is
//! real; it just has no cost to remove.
//!
//! Arithmetic extraction produces **planar** output, the layout the nibble
//! kernels consume, so it would cost nothing at score time —
//! `build_query_planes` permutes once per query. It cannot cheaply produce
//! natural order: that needs a 5-way interleaved store, and NEON tops out at
//! `st4`.
//!
//! Every arm is checked against `lut.fused` on every token before anything is
//! timed — a faster arm that computes something else is not a result — and the
//! arms are timed interleaved in one process, because the same benchmark moves
//! ±30 % between machines.
//!
//! ```text
//! cargo run --release -p next-plaid --example ternary_expand_bench
//! ```
//! Args: `[n_tokens] [reps]` (defaults 4096, 15). Env: `DIM` (128).

// Every arm is a NEON intrinsic, so the harness only exists on aarch64. The
// gate is per-item rather than crate-level (`#![cfg]` at the root would remove
// `main` along with everything else) and there is a stub `main` below for
// other targets.
#[cfg(target_arch = "aarch64")]
mod common;

#[cfg(target_arch = "aarch64")]
use common::{median, Lcg};
#[cfg(target_arch = "aarch64")]
use ndarray::{Array1, Array2};
#[cfg(target_arch = "aarch64")]
use next_plaid::codec::ResidualCodec;
#[cfg(target_arch = "aarch64")]
use next_plaid::residual_lut::quantize_lut;
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
const TRITS: usize = 5;

/// (A) The shipped one-hop: 256×8 table, one unaligned 8-byte copy per stored
/// byte, natural dim order. `w` needs 8 bytes of slack past `5·nbytes`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn expand_one_hop(row: &[u8], nbytes: usize, table: &[[i8; 8]; 256], w: &mut [i8]) {
    let wp = w.as_mut_ptr();
    unsafe {
        for (i, &b) in row[..nbytes].iter().enumerate() {
            std::ptr::copy_nonoverlapping(table[b as usize].as_ptr(), wp.add(TRITS * i), 8);
        }
    }
}

/// Scalar reference for the planar arms, so the SIMD versions have something
/// to be wrong against.
#[cfg(target_arch = "aarch64")]
fn expand_planar_scalar(row: &[u8], nbytes: usize, fused: &[i8], w: &mut [i8], pdim: usize) {
    for (i, &b) in row[..nbytes].iter().enumerate() {
        for k in 0..TRITS {
            w[k * pdim + i] = fused[b as usize * TRITS + k];
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_arms {
    use super::*;
    use std::arch::aarch64::*;

    /// Five trit planes from 16 packed bytes, by exact multiply-shift. The four
    /// quotients are independent of each other, so this is latency ~1 divide,
    /// not 4 chained ones.
    #[inline(always)]
    unsafe fn trits_16(v: uint8x16_t) -> [uint8x16_t; TRITS] {
        let mut out = [vdupq_n_u8(0); TRITS];
        // Widen once; every division runs on both halves in parallel.
        let halves = [vmovl_u8(vget_low_u8(v)), vmovl_u8(vget_high_u8(v))];
        let mut t: [[uint16x8_t; 2]; TRITS] = [[vdupq_n_u16(0); 2]; TRITS];
        for (h, &b) in halves.iter().enumerate() {
            let q1 = vshrq_n_u16(vmulq_n_u16(b, 171), 9);
            let q2 = vshrq_n_u16(vmulq_n_u16(b, 57), 9);
            let q3 = vshrq_n_u16(vmulq_n_u16(b, 19), 9);
            let q4 = vshrq_n_u16(vmulq_n_u16(b, 203), 14);
            // q5 = b/243 in {0,1}: only bytes 243..=255 reach it, and it is what
            // keeps t4 in 0..=2 there. Derived from `b`, not from q4, so it does
            // not lengthen the dependence chain.
            let q5 = vshrq_n_u16(vmulq_n_u16(b, 135), 15);
            t[0][h] = vmlsq_n_u16(b, q1, 3);
            t[1][h] = vmlsq_n_u16(q1, q2, 3);
            t[2][h] = vmlsq_n_u16(q2, q3, 3);
            t[3][h] = vmlsq_n_u16(q3, q4, 3);
            t[4][h] = vmlsq_n_u16(q4, q5, 3);
        }
        for k in 0..TRITS {
            out[k] = vcombine_u8(vmovn_u16(t[k][0]), vmovn_u16(t[k][1]));
        }
        out
    }

    /// (B) Arithmetic extraction, then a 16-entry `tbl` maps trit → weight.
    /// Works for any ternary codebook, equal-mass included.
    pub unsafe fn expand_planar_tbl(
        row: &[u8],
        nbytes: usize,
        wtab: int8x16_t,
        w: &mut [i8],
        pdim: usize,
    ) {
        let wp = w.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= nbytes {
            let ts = trits_16(vld1q_u8(row.as_ptr().add(i)));
            for (k, &t) in ts.iter().enumerate() {
                vst1q_s8(wp.add(k * pdim + i), vqtbl1q_s8(wtab, t));
            }
            i += 16;
        }
        if i < nbytes {
            let rem = nbytes - i;
            let mut src = [0u8; 16];
            src[..rem].copy_from_slice(&row[i..nbytes]);
            let ts = trits_16(vld1q_u8(src.as_ptr()));
            let mut dst = [0i8; 16];
            for (k, &t) in ts.iter().enumerate() {
                vst1q_s8(dst.as_mut_ptr(), vqtbl1q_s8(wtab, t));
                w[k * pdim + i..k * pdim + nbytes].copy_from_slice(&dst[..rem]);
            }
        }
    }

    /// (C) Same extraction, but the codebook is equally spaced so the weight is
    /// `127·(t−1)`: fold 127 into `sqw` and store `t−1`. No table at all.
    pub unsafe fn expand_planar_affine(row: &[u8], nbytes: usize, w: &mut [i8], pdim: usize) {
        let wp = w.as_mut_ptr();
        let one = vdupq_n_s8(1);
        let mut i = 0usize;
        while i + 16 <= nbytes {
            let ts = trits_16(vld1q_u8(row.as_ptr().add(i)));
            for (k, &t) in ts.iter().enumerate() {
                vst1q_s8(wp.add(k * pdim + i), vsubq_s8(vreinterpretq_s8_u8(t), one));
            }
            i += 16;
        }
        if i < nbytes {
            let rem = nbytes - i;
            let mut src = [0u8; 16];
            src[..rem].copy_from_slice(&row[i..nbytes]);
            let ts = trits_16(vld1q_u8(src.as_ptr()));
            let mut dst = [0i8; 16];
            for (k, &t) in ts.iter().enumerate() {
                vst1q_s8(dst.as_mut_ptr(), vsubq_s8(vreinterpretq_s8_u8(t), one));
                w[k * pdim + i..k * pdim + nbytes].copy_from_slice(&dst[..rem]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    // The magics are the whole correctness story; assert them exhaustively
    // before anything else runs.
    for b in 0..256usize {
        let (q1, q2, q3, q4, q5) = (
            (b * 171) >> 9,
            (b * 57) >> 9,
            (b * 19) >> 9,
            (b * 203) >> 14,
            (b * 135) >> 15,
        );
        assert_eq!(
            (q1, q2, q3, q4, q5),
            (b / 3, b / 9, b / 27, b / 81, b / 243),
            "byte {b}"
        );
        let t = [
            b - 3 * q1,
            q1 - 3 * q2,
            q2 - 3 * q3,
            q3 - 3 * q4,
            q4 - 3 * q5,
        ];
        for (k, &tk) in t.iter().enumerate() {
            assert_eq!(tk, (b / 3usize.pow(k as u32)) % 3, "byte {b} trit {k}");
        }
    }
    println!("magic multipliers: exact for all 256 bytes");

    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let ntok: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    let pdim = dim.div_ceil(TRITS);

    let mut rng = Lcg(0x5EED);
    let centroids = Array2::from_shape_fn((4, dim), |_| rng.next_f32() - 0.5);
    let codec = ResidualCodec::new_ternary(
        2,
        centroids,
        Array1::zeros(dim),
        Some(Array1::from_vec(vec![-0.12, 0.12])),
        Some(Array1::from_vec(vec![-0.21, 0.0, 0.21])),
    )
    .unwrap();
    let lut = quantize_lut(&codec).expect("ternary lut");
    let residuals = Array2::from_shape_fn((ntok, dim), |_| rng.next_gauss() * 0.3);
    let packed = codec.quantize_residuals(&residuals).unwrap();
    let nbytes = packed.ncols();
    assert_eq!(nbytes, pdim);

    let mut table = Box::new([[0i8; 8]; 256]);
    for (b, r) in table.iter_mut().enumerate() {
        r[..TRITS].copy_from_slice(&lut.fused[b * TRITS..(b + 1) * TRITS]);
    }
    // trit -> weight, as a 16-entry tbl source (only 0..2 are ever indexed).
    // Byte `t` has trit0 == t, so its row's first entry is w(t). (`fused[..3]`
    // would be byte 0's five trits, which are all trit value 0.)
    let mut wtab_src = [0i8; 16];
    for (t, slot) in wtab_src.iter_mut().enumerate().take(3) {
        *slot = lut.fused[t * TRITS];
    }
    let equally_spaced = wtab_src[0] == -127 && wtab_src[1] == 0 && wtab_src[2] == 127;
    println!(
        "dim={dim} bytes/tok={nbytes} pdim={pdim} tokens={ntok} reps={reps}\n\
         trit weights (int8) = {:?}  equally spaced (arm C legal) = {equally_spaced}",
        &wtab_src[..3]
    );

    let mut wa = vec![0i8; 5 * pdim + 8];
    let mut wb = vec![0i8; 5 * pdim + 8];
    let mut wc = vec![0i8; 5 * pdim + 8];
    let mut wref = vec![0i8; 5 * pdim + 8];

    // ---- correctness, every token, before any timing ----
    {
        use neon_arms::*;
        let wtab = unsafe { std::arch::aarch64::vld1q_s8(wtab_src.as_ptr()) };
        for t in 0..ntok {
            let row = packed.row(t);
            let row = row.as_slice().unwrap();
            wa.fill(0);
            wb.fill(0);
            wc.fill(0);
            wref.fill(0);
            expand_one_hop(row, nbytes, &table, &mut wa);
            expand_planar_scalar(row, nbytes, &lut.fused, &mut wref, pdim);
            unsafe {
                expand_planar_tbl(row, nbytes, wtab, &mut wb, pdim);
                expand_planar_affine(row, nbytes, &mut wc, pdim);
            }
            for (d, &got_a) in wa.iter().enumerate().take(dim) {
                let (i, k) = (d / TRITS, d % TRITS);
                let want = lut.fused[row[i] as usize * TRITS + k];
                assert_eq!(got_a, want, "A tok {t} dim {d}");
                assert_eq!(wb[k * pdim + i], want, "B tok {t} dim {d}");
                assert_eq!(wref[k * pdim + i], want, "ref tok {t} dim {d}");
                if equally_spaced {
                    assert_eq!(
                        wc[k * pdim + i] as i32 * 127,
                        want as i32,
                        "C tok {t} dim {d}"
                    );
                }
            }
        }
        println!("all arms verified against lut.fused on {ntok} tokens\n");
    }

    // ---- timing, interleaved ----
    {
        use neon_arms::*;
        let wtab = unsafe { std::arch::aarch64::vld1q_s8(wtab_src.as_ptr()) };
        let (mut ta, mut tb, mut tc) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..reps {
            let t0 = Instant::now();
            for t in 0..ntok {
                expand_one_hop(packed.row(t).as_slice().unwrap(), nbytes, &table, &mut wa);
                black_box(&wa);
            }
            ta.push(t0.elapsed().as_secs_f64() / ntok as f64 * 1e9);

            let t0 = Instant::now();
            for t in 0..ntok {
                unsafe {
                    expand_planar_tbl(
                        packed.row(t).as_slice().unwrap(),
                        nbytes,
                        wtab,
                        &mut wb,
                        pdim,
                    )
                };
                black_box(&wb);
            }
            tb.push(t0.elapsed().as_secs_f64() / ntok as f64 * 1e9);

            let t0 = Instant::now();
            for t in 0..ntok {
                unsafe {
                    expand_planar_affine(packed.row(t).as_slice().unwrap(), nbytes, &mut wc, pdim)
                };
                black_box(&wc);
            }
            tc.push(t0.elapsed().as_secs_f64() / ntok as f64 * 1e9);
        }
        let (a, b, c) = (median(ta), median(tb), median(tc));
        println!("A  one-hop table copy   (natural) : {a:6.2} ns/token   1.00x");
        println!(
            "B  arithmetic + tbl     (planar)  : {b:6.2} ns/token  {:5.2}x",
            a / b
        );
        println!(
            "C  arithmetic + affine  (planar)  : {c:6.2} ns/token  {:5.2}x",
            a / c
        );
        println!(
            "\n2-bit pays a nibble tbl only; measured at 5.00 ns/token on this shape.\n\
             What ternary owes 2-bit: A {:+.2}, B {:+.2}, C {:+.2} ns/token.",
            a - 5.0,
            b - 5.0,
            c - 5.0
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!(
        "ternary_expand_bench: the arms are NEON intrinsics; nothing to measure on this target."
    );
}
