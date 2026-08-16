//! Is the two-stage ternary expansion worth replacing with a one-stage one?
//!
//! Ternary currently reaches the SIMD kernels in two hops:
//!
//! ```text
//!   base-3 byte --repack--> 2-bit bytes --nibble tbl--> int8 weights (w)
//!                  (A)                      (B)
//! ```
//!
//! (B) is the expansion every scalar rung already does, so ternary's whole
//! asym penalty against 2-bit is (A). But `lut.fused` is *already* a
//! `256 x 5` byte→weights table — the scalar kernel's reference — so a single
//! hop is possible:
//!
//! ```text
//!   base-3 byte --fused u64 store--> int8 weights (w)          (C)
//! ```
//!
//! (C) would replace **both** stages: no 2-bit intermediate, no scratch
//! round-trip, no nibble tbl, and query planes in natural dim order. The
//! question this bench answers is whether `C < A + B` (worth doing at all) and
//! how `C` compares to `B` alone (what ternary would still owe 2-bit, since
//! 2-bit pays only B).
//!
//! Timed interleaved in one process on one core: the effect is smaller than
//! the +/-30% swing the same benchmark shows on *unchanged* code across two CI
//! runners, so only within-process comparison can resolve it. Every arm is
//! checked against the scalar reference before any timing — a faster arm that
//! computes something else is not a result.
//!
//! ```text
//! cargo run --release -p next-plaid --example ternary_expand_bench
//! ```
//! Args: `[n_tokens] [reps]` (defaults 4096, 15). Env: `DIM` (128).

mod common;

use common::{median, Lcg};
use ndarray::{Array1, Array2};
use next_plaid::codec::ResidualCodec;
use next_plaid::residual_lut::{quantize_lut, NibbleLut, MAX_DIM};
use std::hint::black_box;
use std::time::Instant;

/// (A) The shipped repack: four base-3 bytes at a time into five output bytes.
fn repack(src: &[u8], dst: &mut [u8], tab: &[u16; 256]) {
    let mut groups = src.chunks_exact(4);
    for (g, quad) in groups.by_ref().enumerate() {
        let v = tab[quad[0] as usize] as u64
            | (tab[quad[1] as usize] as u64) << 10
            | (tab[quad[2] as usize] as u64) << 20
            | (tab[quad[3] as usize] as u64) << 30;
        dst[5 * g..5 * g + 5].copy_from_slice(&v.to_le_bytes()[..5]);
    }
    let rem = groups.remainder();
    if !rem.is_empty() {
        let g = src.len() / 4;
        let mut v = 0u64;
        for (k, &b) in rem.iter().enumerate() {
            v |= (tab[b as usize] as u64) << (10 * k);
        }
        let nbytes = (10 * rem.len()).div_ceil(8);
        dst[5 * g..5 * g + nbytes].copy_from_slice(&v.to_le_bytes()[..nbytes]);
    }
}

/// (B) The kernels' nibble expansion, transcribed from `maxsim_residual_lut_*`
/// so the cost measured here is the cost they actually pay.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn nibble_expand(row: &[u8], pdim: usize, kpb: usize, nib: &NibbleLut, w: &mut [i8]) {
    use std::arch::aarch64::*;
    let mut tabs = [vdupq_n_s8(0); 8];
    for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
        *tab = vld1q_s8(src.as_ptr());
    }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_expand(row: &[u8], pdim: usize, kpb: usize, nib: &NibbleLut, w: &mut [i8]) {
    use std::arch::x86_64::*;
    let mut tabs = [_mm_setzero_si128(); 8];
    for (tab, src) in tabs.iter_mut().zip(nib.tables.iter()).take(kpb) {
        *tab = _mm_loadu_si128(src.as_ptr() as *const __m128i);
    }
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
}

/// (C) One hop: each base-3 byte's five weights are one `u64` in a 2 KB table,
/// stored straight into `w` at `5i`. Stores overlap by three bytes, but every
/// one is *pure* — the next iteration overwrites the slack, so there is no
/// read-modify-write and no store-forwarding stall. Output is natural dim
/// order, so the query needs no permutation at all.
fn direct_expand(row: &[u8], src_cols: usize, table: &[[i8; 8]; 256], w: &mut [i8]) {
    let wp = w.as_mut_ptr();
    for (i, &b) in row[..src_cols].iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(table[b as usize].as_ptr(), wp.add(5 * i), 8);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ntok: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(15);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    let mut rng = Lcg::new(0x5EED);
    let mut sample: Vec<f32> = (0..40_000).map(|_| rng.next_f32() * 0.3).collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| sample[((sample.len() - 1) as f64 * p) as usize];
    let codec = ResidualCodec::new_ternary(
        2,
        Array2::from_shape_fn((16, dim), |_| rng.next_f32()),
        Array1::zeros(dim),
        Some([1.0 / 3.0, 2.0 / 3.0].iter().map(|&p| q(p)).collect()),
        Some([1.0 / 6.0, 0.5, 5.0 / 6.0].iter().map(|&p| q(p)).collect()),
    )
    .unwrap();
    let lut = quantize_lut(&codec).expect("ternary LUT");
    let td = lut.ternary_direct.as_ref().expect("one-hop table");

    // The two-hop route this replaced, rebuilt here from the codec's own trit
    // table. The library no longer carries it, but the comparison is the whole
    // reason the one-hop route exists, so the bench keeps it reproducible.
    let trits = codec.trit_lookup.as_ref().expect("ternary trit table");
    let mut repack_tab = [0u16; 256];
    for (byte, quintet) in trits.iter().enumerate() {
        let mut v = 0u16;
        for (k, &t) in quintet.iter().enumerate() {
            v |= (t as u16) << (2 * k);
        }
        repack_tab[byte] = v;
    }
    // The 2-bit alphabet the repacked stream was scored with: [w-, w0, w+, 0].
    // Byte value j (j < 3) has trits [j,0,0,0,0], so its weight is fused[5j].
    // Key k of a 2-bit byte is `(b >> 2k) & 3`, so keys 0,1 read the low
    // nibble and keys 2,3 the high one — which is why it factored and base-3
    // does not.
    let vals2: [i8; 4] = [lut.fused[0], lut.fused[5], lut.fused[10], 0];
    let mut nib2 = NibbleLut {
        tables: [[0i8; 16]; 8],
        from_hi: [false; 8],
    };
    for k in 0..4 {
        nib2.from_hi[k] = k >= 2;
        let shift = 2 * (k % 2);
        for x in 0..16usize {
            nib2.tables[k][x] = vals2[(x >> shift) & 3];
        }
    }

    let residuals = Array2::from_shape_fn((ntok, dim), |_| rng.next_f32() * 0.3);
    let packed = codec.quantize_residuals(&residuals).unwrap();
    let src_cols = dim.div_ceil(5);
    let edim = 4 * dim.div_ceil(4);
    let pdim2 = edim / 4;
    let tcols = pdim2 + 2;

    // Correctness gate. `w_direct` is natural order; `w_nibble` is plane order
    // (`key*pdim + byte`). Checking both against `lut.fused` also re-verifies
    // the whole repack chain, not just the new arm.
    let mut tmp = vec![0u8; tcols];
    let mut w_direct = [0i8; MAX_DIM + 8];
    let mut w_nibble = [0i8; MAX_DIM + 8];
    for row in packed.rows() {
        let src = &row.as_slice().unwrap()[..src_cols];
        tmp.fill(0);
        repack(src, &mut tmp, &repack_tab);
        unsafe { nibble_expand(&tmp, pdim2, 4, &nib2, &mut w_nibble) };
        direct_expand(src, src_cols, &td.table, &mut w_direct);
        for d in 0..dim {
            let want = lut.fused[src[d / 5] as usize * 5 + d % 5];
            assert_eq!(w_direct[d], want, "direct expand wrong at dim {d}");
            assert_eq!(
                w_nibble[(d % 4) * pdim2 + d / 4],
                want,
                "repack+nibble expand wrong at dim {d}"
            );
        }
    }

    println!("\nternary expansion: two hops (repack + nibble tbl) vs one (fused u64)");
    println!(
        "  build   {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("  {ntok} tokens x dim {dim}, median of {reps} interleaved reps\n");

    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for rep in 0..=reps {
        let t = Instant::now();
        for row in packed.rows() {
            repack(&row.as_slice().unwrap()[..src_cols], &mut tmp, &repack_tab);
            black_box(&tmp);
        }
        let ta = t.elapsed().as_secs_f64() * 1e9 / ntok as f64;

        let t = Instant::now();
        for _ in 0..ntok {
            unsafe { nibble_expand(&tmp, pdim2, 4, &nib2, &mut w_nibble) };
            black_box(&w_nibble);
        }
        let tb = t.elapsed().as_secs_f64() * 1e9 / ntok as f64;

        let t = Instant::now();
        for row in packed.rows() {
            direct_expand(
                &row.as_slice().unwrap()[..src_cols],
                src_cols,
                &td.table,
                &mut w_direct,
            );
            black_box(&w_direct);
        }
        let tc = t.elapsed().as_secs_f64() * 1e9 / ntok as f64;

        if rep > 0 {
            a.push(ta);
            b.push(tb);
            c.push(tc);
        }
    }
    let (a, b, c) = (median(a), median(b), median(c));
    println!("                                    ns/token");
    println!("    A  repack (base-3 -> 2-bit)     {a:9.2}");
    println!("    B  nibble tbl (2-bit -> w)      {b:9.2}   <- 2-bit pays only this");
    println!("    A+B  what ternary pays today    {:9.2}", a + b);
    println!("    C  fused copy (base-3 -> w)     {c:9.2}   <- shipped");
    println!();
    println!(
        "    C vs A+B   {:+.1}%  (is one hop worth doing)",
        (c / (a + b) - 1.0) * 100.0
    );
    println!(
        "    C vs B     {:+.1}%  (what ternary would still owe 2-bit)",
        (c / b - 1.0) * 100.0
    );
    println!(
        "\n  Expansion only — the SDOT, fold and memory traffic around it are\n  \
         identical for both and are not in these numbers.\n"
    );
}
