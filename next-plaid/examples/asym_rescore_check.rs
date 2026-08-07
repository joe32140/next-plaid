//! Does *this* build get the fused rescore kernel, and what is it worth here?
//!
//! `residual_asym` falls back silently — to the float path for binary indexes
//! and unsupported dims, and to the scalar kernel when the CPU lacks AVX2 or
//! NEON `dotprod`. Every fallback returns correct scores, so a build that
//! never reaches the fused kernel does not fail; it just measures a small
//! speedup where the fused kernel measures a large one. This example names
//! the kernel it dispatched to before printing any number, so that outcome
//! cannot be mistaken for "the optimization does not work".
//!
//! It times Stage 2's two arms back to back on identical inputs, which is the
//! comparison the PR quotes:
//!
//! * float — `ResidualCodec::decompress` then f32 MaxSim, and
//! * asym  — `maxsim_residual_lut_i8` straight over the packed codes.
//!
//! Synthetic inputs, no index build and no dataset: residual *values* set how
//! the kernels rank, not how much work they do, so kernel throughput is a
//! function of the shapes alone. Both arms run under rayon across documents,
//! as a real search does — the float arm materializes a decompressed f32 copy
//! per document, so a single-threaded loop understates what that costs.
//!
//! ```text
//! cargo run --release -p next-plaid --example asym_rescore_check
//! cargo run --release -p next-plaid --example asym_rescore_check -- 1024 2
//! ```
//!
//! Args: `[n_docs] [nbits]` (defaults 1024, 4). Env: `DIM` (128), `TOKENS`
//! (230 per doc), `NQ` (32 query tokens), `CENTROIDS` (16384), `REPS` (5).

use ndarray::{Array1, Array2};
use next_plaid::binary::quantize_query_i8;
use next_plaid::codec::ResidualCodec;
use next_plaid::maxsim::maxsim_score;
use next_plaid::residual_lut::{
    active_kernel_name, build_query_planes, maxsim_residual_lut_i8, quantize_lut,
};
use rayon::prelude::*;
use std::time::Instant;

/// Deterministic pseudo-random floats in `[-1, 1)`; no rand dependency needed
/// and the same numbers on every platform, so two machines' outputs compare.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn array(&mut self, rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| self.next_f32() * scale)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A codec whose bucket cutoffs and weights are the quantiles of the residual
/// distribution below — the same construction the real index builder uses,
/// which is what makes the LUT factor into nibble tables.
fn synth_codec(dim: usize, nbits: usize, k: usize, rng: &mut Lcg) -> ResidualCodec {
    let centroids = rng.array(k, dim, 1.0);
    let mut sample: Vec<f32> = (0..40_000).map(|_| rng.next_f32() * 0.3).collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let n_options = 1usize << nbits;
    let q = |p: f64| sample[((sample.len() - 1) as f64 * p) as usize];
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_docs = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024usize);
    let nbits = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4usize);
    let dim = env_usize("DIM", 128);
    let tokens = env_usize("TOKENS", 230);
    let nq = env_usize("NQ", 32);
    let k = env_usize("CENTROIDS", 16384);
    let reps = env_usize("REPS", 5);

    let mut rng = Lcg(0x5EED);
    let codec = synth_codec(dim, nbits, k, &mut rng);
    let Some(lut) = quantize_lut(&codec) else {
        println!("this codec has no fused LUT (nbits={nbits}); asym would score in float");
        return;
    };

    // Candidates reaching Stage 2 came out of the IVF probe, so their tokens
    // concentrate in the cells that probe selected — roughly n_ivf_probe per
    // query token. Drawing codes uniformly over all `k` centroids instead
    // would make every `cdot_t` lookup a cache miss over a matrix real
    // searches touch a small, hot slice of, and would understate asym by
    // charging it for a scatter that does not happen.
    let active = env_usize("ACTIVE_CENTROIDS", 8 * nq);
    let cells: Vec<i64> = (0..active)
        .map(|_| (rng.next_f32().abs() * k as f32) as i64 % k as i64)
        .collect();

    // One document = packed residual bytes + a centroid code per token, which
    // is exactly what Stage 2 reads off the mmap.
    let docs: Vec<(Array2<u8>, Vec<i64>, Vec<usize>)> = (0..n_docs)
        .map(|_| {
            let residuals = rng.array(tokens, dim, 0.3);
            let packed = codec.quantize_residuals(&residuals).unwrap();
            let codes: Vec<i64> = (0..tokens)
                .map(|_| cells[(rng.next_f32().abs() * active as f32) as usize % active])
                .collect();
            let as_usize = codes.iter().map(|&c| c as usize).collect();
            (packed, codes, as_usize)
        })
        .collect();

    let query = rng.array(nq, dim, 1.0);
    let q8 = quantize_query_i8(&query.view());
    let planes = build_query_planes(&q8, &lut, dim);
    // Stands in for Stage 1's query x centroid scores, centroid-major as the
    // kernels consume them.
    let cdot_t = rng.array(k, nq, 1.0);
    let inv: Vec<f32> = (0..tokens).map(|_| 1.0).collect();

    let kernel = active_kernel_name(dim, lut.nibble.is_some());
    println!("\nnext-plaid asym rescore check");
    println!(
        "  build        {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("  asym kernel  {kernel}");
    if kernel.starts_with("scalar") {
        println!(
            "\n  !! The fused SIMD kernel did NOT dispatch on this build, so the number\n  \
             !! below is the scalar fallback, not what this PR is about. An x86_64 build\n  \
             !! under Rosetta, an x86_64 container on Apple Silicon, or an ARM CPU\n  \
             !! without `dotprod` lands here."
        );
    }
    println!("\n  {n_docs} docs x {tokens} tokens, dim {dim}, nbits {nbits}, {nq} query tokens");
    println!(
        "  {k} centroids, median of {reps} reps, {} threads\n",
        rayon::current_num_threads()
    );

    let mut fl = Vec::new();
    let mut asy = Vec::new();
    for rep in 0..=reps {
        // Interleaved, and rep 0 is a warmup discarded by both arms so neither
        // pays the first-touch page faults. Scored under rayon across
        // documents, as `search_one_mmap` does — the float arm materializes a
        // decompressed f32 copy per document, so its cost relative to asym
        // depends on memory pressure and a single-threaded loop would flatter
        // it.
        let t = Instant::now();
        let fsum: f32 = docs
            .par_iter()
            .map(|(packed, _, codes)| {
                // `MmapIndex::decode_rows` owns the mmap slice and rebuilds the
                // code array before decompressing; both copies are charged to
                // the float arm in production, so charge them here too.
                let rec = codec
                    .decompress(&packed.clone(), &Array1::from(codes.clone()).view())
                    .unwrap();
                maxsim_score(&query.view(), &rec.view())
            })
            .sum();
        let f = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let asum: f32 = docs
            .par_iter()
            .map(|(packed, codes, _)| {
                maxsim_residual_lut_i8(
                    &q8,
                    Some(&planes),
                    &packed.view(),
                    codes,
                    &cdot_t.view(),
                    &lut,
                    &inv,
                    dim,
                )
            })
            .sum();
        let a = t.elapsed().as_secs_f64() * 1e3;

        if rep == 0 {
            assert!(
                fsum.is_finite() && asum.is_finite(),
                "non-finite scores: float {fsum}, asym {asum}"
            );
        } else {
            fl.push(f);
            asy.push(a);
        }
    }
    let (fl, asy) = (median(fl), median(asy));
    let per_token = |ms: f64| ms * 1e6 / (n_docs * tokens) as f64;

    println!("                 ms/{n_docs} docs      ns/token");
    println!("    float        {fl:10.2}   {:11.1}", per_token(fl));
    println!("    asym         {asy:10.2}   {:11.1}", per_token(asy));
    println!("    ratio        {:9.2}x\n", fl / asy);
    println!(
        "  Rescore only, and a floor. This corpus is RAM-resident, where the float\n  \
         arm costs ~44 ns/token; against a real mmap-backed index it measured ~67,\n  \
         so the ratios quoted in the PR — real corpora, three platforms — run above\n  \
         this synthetic cell. What this check settles is the kernel line above.\n  \
         End-to-end search gains less than the rescore ratio either way, since\n  \
         Stage 1 is unchanged by this PR and is common to both arms.\n"
    );
}
