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
//! cargo run --release -p next-plaid --example asym_rescore_check -- 1024 t
//! ```
//!
//! Args: `[n_docs] [nbits|t|all]` (defaults 1024, `all`). The second arg selects
//! a scalar rung (`1`/`2`/`4`), the ternary base-3 codec (`t`, or set
//! `TERNARY=1`), or `all` — every codec in **one process**, which is the only
//! way their numbers may be compared with each other.
//!
//! Running one codec per process, as this harness used to, produces
//! cross-codec deltas that are noise: the same benchmark on unchanged code
//! moves +/-30% between CI runners, and it moved enough between two runs to
//! report ternary as *faster* than 2-bit — impossible, since ternary does
//! everything 2-bit does and then expands from a wider alphabet. `all`
//! interleaves every codec within each rep, so they share one machine, one
//! thermal state, one page cache.
//!
//! Env: `DIM` (128), `TOKENS` (230 per doc), `NQ` (32 query tokens),
//! `CENTROIDS` (16384), `REPS` (5).

mod common;

use common::{median, Lcg};
use ndarray::{Array1, Array2};
use next_plaid::binary::quantize_query_i8;
use next_plaid::codec::ResidualCodec;
use next_plaid::maxsim::maxsim_score;
use next_plaid::residual_lut::{build_query_planes, maxsim_residual_lut_i8, quantize_lut};
use next_plaid::residual_lut::{QueryPlanes, ResidualLut};
use rayon::prelude::*;
use std::time::Instant;

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

/// The ternary (base-3 dead-zone) analogue of [`synth_codec`]: two cutoffs at the
/// residual tertiles and three weights at the bucket centers, so the `{-m,0,+m}`
/// buckets sit on the same synthetic distribution. `quantize_lut` maps this to the
/// base-3 fused table plus a `TernarySimd` transcode, so asym rides the same
/// nibble SIMD kernels as the scalar rungs (via the 2-bit transcoded stream) —
/// exactly what a real ternary index dispatches to.
fn synth_ternary_codec(dim: usize, k: usize, rng: &mut Lcg) -> ResidualCodec {
    let centroids = rng.array(k, dim, 1.0);
    let mut sample: Vec<f32> = (0..40_000).map(|_| rng.next_f32() * 0.3).collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| sample[((sample.len() - 1) as f64 * p) as usize];
    let cutoffs: Array1<f32> = [1.0 / 3.0, 2.0 / 3.0].iter().map(|&p| q(p)).collect();
    let weights: Array1<f32> = [1.0 / 6.0, 0.5, 5.0 / 6.0].iter().map(|&p| q(p)).collect();
    ResidualCodec::new_ternary(
        2,
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
    // `[nbits]` selects one scalar rung, `t`/`ternary` (or TERNARY=1) the
    // base-3 codec, `all` (the default) every codec in this one process.
    let arg2 = args.get(2).cloned().unwrap_or_else(|| "all".to_string());
    let dim = env_usize("DIM", 128);
    let tokens = env_usize("TOKENS", 230);
    let nq = env_usize("NQ", 32);
    let k = env_usize("CENTROIDS", 16384);
    let reps = env_usize("REPS", 5);

    let mut rng = Lcg(0x5EED);
    // (label, codec). Built before any documents so every codec quantizes the
    // same residuals below.
    let specs: Vec<(String, ResidualCodec)> = if arg2 == "all" {
        vec![
            ("4-bit".to_string(), synth_codec(dim, 4, k, &mut rng)),
            ("2-bit".to_string(), synth_codec(dim, 2, k, &mut rng)),
            ("ternary".to_string(), synth_ternary_codec(dim, k, &mut rng)),
            ("1-bit".to_string(), synth_codec(dim, 1, k, &mut rng)),
        ]
    } else if env_usize("TERNARY", 0) == 1 || arg2.starts_with('t') {
        vec![("ternary".to_string(), synth_ternary_codec(dim, k, &mut rng))]
    } else {
        let nbits = arg2.parse().unwrap_or(4usize);
        vec![(format!("{nbits}-bit"), synth_codec(dim, nbits, k, &mut rng))]
    };

    let luts: Vec<ResidualLut> = specs
        .iter()
        .map(|(label, codec)| {
            quantize_lut(codec).unwrap_or_else(|| panic!("{label}: no fused LUT"))
        })
        .collect();

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
    // is exactly what Stage 2 reads off the mmap. Every codec packs the *same*
    // residuals and codes: the documents differ only in how they are encoded,
    // so nothing but the codec varies across rows of the table.
    let mut docs: Vec<Vec<(Array2<u8>, Vec<i64>, Vec<usize>)>> =
        specs.iter().map(|_| Vec::with_capacity(n_docs)).collect();
    for _ in 0..n_docs {
        let residuals = rng.array(tokens, dim, 0.3);
        let codes: Vec<i64> = (0..tokens)
            .map(|_| cells[(rng.next_f32().abs() * active as f32) as usize % active])
            .collect();
        let as_usize: Vec<usize> = codes.iter().map(|&c| c as usize).collect();
        for (ci, (_, codec)) in specs.iter().enumerate() {
            let packed = codec.quantize_residuals(&residuals).unwrap();
            docs[ci].push((packed, codes.clone(), as_usize.clone()));
        }
    }

    let query = rng.array(nq, dim, 1.0);
    let q8 = quantize_query_i8(&query.view());
    let planes: Vec<QueryPlanes> = luts
        .iter()
        .map(|lut| build_query_planes(&q8, lut, dim))
        .collect();
    // Stands in for Stage 1's query x centroid scores, centroid-major as the
    // kernels consume them.
    let cdot_t = rng.array(k, nq, 1.0);
    let inv: Vec<f32> = (0..tokens).map(|_| 1.0).collect();

    println!("\nnext-plaid asym rescore check");
    println!(
        "  build        {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    for (lut, (label, _)) in luts.iter().zip(specs.iter()) {
        println!("  {label:<8} kernel  {}", lut.kernel_name(dim));
    }
    if luts
        .iter()
        .any(|l| l.kernel_name(dim).starts_with("scalar"))
    {
        println!(
            "\n  !! A fused SIMD kernel did NOT dispatch on this build, so a number\n  \
             !! below is the scalar fallback, not what this PR is about. An x86_64 build\n  \
             !! under Rosetta, an x86_64 container on Apple Silicon, or an ARM CPU\n  \
             !! without `dotprod` lands here."
        );
    }
    println!("\n  {n_docs} docs x {tokens} tokens, dim {dim}, {nq} query tokens");
    println!(
        "  {k} centroids, median of {reps} interleaved reps, {} threads\n",
        rayon::current_num_threads()
    );

    let ncell = specs.len();
    let mut fl = vec![Vec::new(); ncell];
    let mut asy = vec![Vec::new(); ncell];
    for rep in 0..=reps {
        // Every codec runs inside every rep, so a machine that speeds up or
        // slows down mid-run moves all of them together. rep 0 is a warmup
        // discarded everywhere, so no arm pays the first-touch page faults.
        for ci in 0..ncell {
            let (codec, lut) = (&specs[ci].1, &luts[ci]);
            // Scored under rayon across documents, as `search_one_mmap` does —
            // the float arm materializes a decompressed f32 copy per document,
            // so its cost relative to asym depends on memory pressure and a
            // single-threaded loop would flatter it.
            let t = Instant::now();
            let fsum: f32 = docs[ci]
                .par_iter()
                .map(|(packed, _, codes)| {
                    // `MmapIndex::decode_rows` owns the mmap slice and rebuilds
                    // the code array before decompressing; both copies are
                    // charged to the float arm in production, so charge them
                    // here too.
                    let rec = codec
                        .decompress(&packed.clone(), &Array1::from(codes.clone()).view())
                        .unwrap();
                    maxsim_score(&query.view(), &rec.view())
                })
                .sum();
            let f = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let asum: f32 = docs[ci]
                .par_iter()
                .map(|(packed, codes, _)| {
                    maxsim_residual_lut_i8(
                        &q8,
                        Some(&planes[ci]),
                        &packed.view(),
                        codes,
                        &cdot_t.view(),
                        lut,
                        &inv,
                        dim,
                    )
                })
                .sum();
            let a = t.elapsed().as_secs_f64() * 1e3;

            if rep == 0 {
                assert!(
                    fsum.is_finite() && asum.is_finite(),
                    "{}: non-finite scores: float {fsum}, asym {asum}",
                    specs[ci].0
                );
            } else {
                fl[ci].push(f);
                asy[ci].push(a);
            }
        }
    }

    let per_token = |ms: f64| ms * 1e6 / (n_docs * tokens) as f64;
    println!(
        "{:<9} {:>7} {:>12} {:>12}  ratio",
        "codec", "B/tok", "float ns/tok", "asym ns/tok"
    );
    println!("{}", "-".repeat(56));
    let mut baseline: Option<f64> = None;
    for ci in 0..ncell {
        let (f, a) = (median(fl[ci].clone()), median(asy[ci].clone()));
        let (fp, ap) = (per_token(f), per_token(a));
        let bytes = if specs[ci].1.ternary {
            dim.div_ceil(5)
        } else {
            dim / (8 / specs[ci].1.nbits)
        };
        println!(
            "{:<9} {:>6}B {:>12.1} {:>12.1}  {:.2}x",
            specs[ci].0,
            bytes,
            fp,
            ap,
            fp / ap
        );
        if specs[ci].0 == "2-bit" {
            baseline = Some(ap);
        }
    }
    if let (Some(b), Some(ti)) = (baseline, (0..ncell).find(|&i| specs[i].0 == "ternary")) {
        let ta = per_token(median(asy[ti].clone()));
        println!(
            "\n  ternary asym vs 2-bit asym: {:+.1}%  ({:+.2} ns/token)",
            (ta / b - 1.0) * 100.0,
            ta - b
        );
        println!(
            "  Both expand to the same lane count and run the same dot; the gap is\n  \
             ternary's wider-alphabet expansion (see `TernaryDirect`)."
        );
    }
    println!(
        "\n  Rescore only, and a floor. This corpus is RAM-resident, where the float\n  \
         arm costs ~44 ns/token; against a real mmap-backed index it measured ~67,\n  \
         so the ratios quoted in the PR — real corpora, three platforms — run above\n  \
         this synthetic cell. End-to-end search gains less than the rescore ratio\n  \
         either way, since Stage 1 is unchanged by this PR and common to both arms.\n"
    );
}
