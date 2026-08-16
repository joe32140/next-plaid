//! Head-to-head for the two ternary→2-bit repack strategies, in one process.
//!
//! Both produce byte-identical output (asserted here); they differ only in how
//! they write it:
//!
//! * `per-byte` — one `[4][256] → u16` lookup per stored byte, OR-ed in at
//!   output byte `5i/4`. Consecutive writes overlap, so each iteration
//!   read-modify-writes a byte its predecessor just stored.
//! * `batched`  — four stored bytes at a time (`4 × 5 = 20` trits = 40 bits =
//!   five whole output bytes), accumulated in a `u64` and stored once. No
//!   overlap, so every store is pure.
//!
//! Why a dedicated harness: the effect is ~15 %, while the same benchmark run
//! twice on different CI runners moves *unchanged* code by ±30 % (measured).
//! Cross-run comparison therefore cannot resolve this, but interleaving both
//! variants in one process on one core can — they see the same CPU, the same
//! thermal state, the same cache.
//!
//! ```text
//! cargo run --release -p next-plaid --example ternary_repack_bench
//! ```
//! Args: `[n_tokens] [reps]` (defaults 4096, 15). Env: `DIM` (128).

use ndarray::{Array1, Array2};
use next_plaid::codec::ResidualCodec;
use next_plaid::residual_lut::quantize_lut;
use std::hint::black_box;
use std::time::Instant;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// The shipped strategy: one pure 5-byte store per four stored bytes.
fn repack_batched(src: &[u8], dst: &mut [u8], tab: &[u16; 256]) {
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

/// The earlier strategy, kept here only as the comparison arm.
fn repack_per_byte(src: &[u8], dst: &mut [u8], phased: &[[u16; 256]; 4]) {
    for (i, &b) in src.iter().enumerate() {
        let v = phased[i & 3][b as usize];
        let j = 5 * i / 4;
        dst[j] |= v as u8;
        dst[j + 1] |= (v >> 8) as u8;
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

    let mut rng = Lcg(0x5EED);
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
    let ts = lut.ternary_simd.as_ref().expect("repack table");

    // Rebuild the phased table the per-byte arm used, from the shipped 1-D one.
    let mut phased = Box::new([[0u16; 256]; 4]);
    for (phase, table) in phased.iter_mut().enumerate() {
        for (byte, slot) in table.iter_mut().enumerate() {
            *slot = ts.repack[byte] << (2 * phase);
        }
    }

    let residuals = Array2::from_shape_fn((ntok, dim), |_| rng.next_f32() * 0.3);
    let packed = codec.quantize_residuals(&residuals).unwrap();
    let src_cols = dim.div_ceil(5);
    let tcols = 4 * dim.div_ceil(4) / 4 + 2;

    // Correctness gate before any timing: a faster arm that computes something
    // else is not a result.
    let mut a = vec![0u8; tcols];
    let mut b = vec![0u8; tcols];
    for row in packed.rows() {
        let src = &row.as_slice().unwrap()[..src_cols];
        a.fill(0);
        b.fill(0);
        repack_per_byte(src, &mut a, &phased);
        repack_batched(src, &mut b, &ts.repack);
        assert_eq!(a, b, "the two repack strategies must agree byte for byte");
    }

    println!("\nternary repack: per-byte vs batched");
    println!(
        "  build   {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "  {ntok} tokens x dim {dim} ({src_cols} -> {} bytes), median of {reps} interleaved reps\n",
        tcols - 2
    );

    let mut out = vec![0u8; ntok * tcols];
    let mut per_byte = Vec::new();
    let mut batched = Vec::new();
    for rep in 0..=reps {
        // Interleaved, rep 0 discarded as warmup, so neither arm pays the
        // first-touch page faults or gets a colder cache than the other.
        let t = Instant::now();
        out.fill(0); // the per-byte arm ORs, so it needs a zeroed buffer
        for (t_i, row) in packed.rows().into_iter().enumerate() {
            let src = &row.as_slice().unwrap()[..src_cols];
            repack_per_byte(src, &mut out[t_i * tcols..(t_i + 1) * tcols], &phased);
        }
        black_box(&out);
        let pb = t.elapsed().as_secs_f64() * 1e9 / ntok as f64;

        let t = Instant::now();
        for (t_i, row) in packed.rows().into_iter().enumerate() {
            let src = &row.as_slice().unwrap()[..src_cols];
            repack_batched(src, &mut out[t_i * tcols..(t_i + 1) * tcols], &ts.repack);
        }
        black_box(&out);
        let ba = t.elapsed().as_secs_f64() * 1e9 / ntok as f64;

        if rep > 0 {
            per_byte.push(pb);
            batched.push(ba);
        }
    }
    let (pb, ba) = (median(per_byte), median(batched));
    println!("               ns/token");
    println!("    per-byte  {pb:9.1}");
    println!("    batched   {ba:9.1}");
    println!("    speedup   {:8.2}x\n", pb / ba);
    println!(
        "  Repack only — it is one term of the rescore cost, alongside the\n  \
         SIMD expand and dot that both arms share unchanged.\n"
    );
}
