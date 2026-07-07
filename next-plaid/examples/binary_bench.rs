//! Pragmatic apples-to-apples comparison of every available Stage-2 scoring
//! scheme in next-plaid, on synthetic ColBERT-style embeddings.
//!
//! Tiers compared (each against the STRONGEST available implementation):
//!   * float           — full f32 MaxSim (ground-truth ranking; uses linked BLAS)
//!   * residual nbits=4 — stock PLAID codec (centroid + 4-bit residual), the
//!     scheme the binary codec is an alternative to
//!   * residual nbits=2 — same, 2-bit
//!   * int8 x int8      — both sides int8 (SDOT / AVX2-madd), "near-lossless" 4x
//!   * int8 x 1-bit     — the asymmetric binary scheme, 2P-T integer kernel
//!   * 1-bit x 1-bit    — both binarized, XOR+popcount
//!
//! Corpus geometry is tuned to resemble a REAL checkpoint (low effective rank +
//! anisotropy), so NDCG retention lands in a believable band rather than the
//! worst-case ~57% that isotropic Gaussian noise produces.
//!
//! CAVEATS on realism (read before trusting the numbers):
//!   * This brute-force rescores ALL docs. Real PLAID rescores only the Stage-1
//!     candidate shortlist (~a few %), and Stage-1 recall — not touched here —
//!     dominates end-to-end latency. These are Stage-2 KERNEL microbenchmarks,
//!     not end-to-end query latencies.
//!   * NDCG is not the discriminating variable across kernels of the same scheme
//!     (they compute the same scores); the latency columns are the point.
//!   * Synthetic geometry ≠ a real ColBERT checkpoint. Retention here is a
//!     sanity band, not a measured model number.
//!
//! Run:
//!   cargo run -p next-plaid --release --example binary_bench                 # pure-Rust GEMM
//!   cargo run -p next-plaid --release --features accelerate --example binary_bench   # Apple BLAS
//!   cargo run -p next-plaid --release --features mkl        --example binary_bench   # Intel MKL

use std::time::{Duration, Instant};

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;

use next_plaid::binary::{binarize, maxsim_binary_binary, maxsim_binary_i8, quantize_query_i8};
use next_plaid::index::{prepare_codec_artifacts, IndexConfig};
use next_plaid::int8::{maxsim_i8, quantize_i8};
use next_plaid::kmeans::{compute_kmeans, ComputeKmeansConfig};
use next_plaid::maxsim::maxsim_score;

const DIM: usize = 128;
const N_DOCS: usize = 1000;
const DOC_TOKENS: usize = 80;
const QUERY_TOKENS: usize = 32;
const N_QUERIES: usize = 30;
const K: usize = 10;

// --- realistic-geometry knobs (tuned for ~80-95% int8x1bit retention) --------
// Higher RANK + weaker anisotropy + more topics + more noise => harder, more
// realistic binarization (avoids the "too rosy" 99% that strong anisotropy and
// a few dominant dims produce). Each query is relevant to only ~N_DOCS/N_TOPICS
// docs, so top-10 ranking is discriminative rather than trivial.
const RANK: usize = 48; // effective subspace rank
const N_TOPICS: usize = 200; // many tight clusters -> ~5 relevant docs/query
const DEAD_DIMS: usize = 6; // ~5% dead dims, demonstrates the mechanism
const ALPHA: f32 = 0.96; // gentle anisotropic decay (flatter spectrum)
const SIGMA_DOC: f32 = 0.7; // more within-cluster spread
const SIGMA_Q: f32 = 0.5;

fn randn(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    Array2::random_using(
        (rows, cols),
        StandardNormal,
        &mut StdRng::seed_from_u64(seed),
    )
}

fn normalize_rows(mut m: Array2<f32>) -> Array2<f32> {
    for mut row in m.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
    m
}

/// Embed latent codes `z` (n × RANK) through the anisotropic spectrum and shared
/// projection `proj` (RANK × DIM), add tiny full-rank noise, L2-normalize.
fn embed(z: &Array2<f32>, proj: &Array2<f32>, spectrum: &[f32], noise_seed: u64) -> Array2<f32> {
    let mut zs = z.clone();
    for mut row in zs.axis_iter_mut(Axis(0)) {
        for (j, x) in row.iter_mut().enumerate() {
            *x *= spectrum[j];
        }
    }
    let mut x = zs.dot(proj); // [n, DIM]
    let noise = randn(x.nrows(), DIM, noise_seed);
    x = x + 0.02 * noise;
    normalize_rows(x)
}

fn ndcg_at_k(ranking: &[usize], rel: &[f32], k: usize) -> f32 {
    let dcg = |order: &[usize]| -> f32 {
        order
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &doc)| (2f32.powf(rel[doc]) - 1.0) / ((i + 2) as f32).log2())
            .sum()
    };
    let mut ideal: Vec<usize> = (0..rel.len()).collect();
    ideal.sort_by(|&a, &b| rel[b].total_cmp(&rel[a]));
    let idcg = dcg(&ideal);
    if idcg <= 0.0 {
        0.0
    } else {
        dcg(ranking) / idcg
    }
}

fn rank_by<F: Fn(usize) -> f32>(n: usize, score: F) -> Vec<usize> {
    let scores: Vec<f32> = (0..n).map(&score).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

fn main() {
    println!(
        "Pragmatic binary-MaxSim comparison\n\
         dim={DIM} docs={N_DOCS} doc_tokens={DOC_TOKENS} query_tokens={QUERY_TOKENS} \
         queries={N_QUERIES}\n\
         geometry: rank={RANK} topics={N_TOPICS} dead_dims={DEAD_DIMS} alpha={ALPHA}\n"
    );

    // --- shared latent structure (docs + queries live in the SAME subspace) ---
    let mut proj = randn(RANK, DIM, 777);
    for c in 0..DEAD_DIMS {
        for r in 0..RANK {
            proj[[r, c]] = 0.0; // dead output dims -> pure-noise sign
        }
    }
    let spectrum: Vec<f32> = (0..RANK).map(|j| ALPHA.powi(j as i32)).collect();
    let clusters = randn(N_TOPICS, RANK, 778);

    // --- build corpus ---------------------------------------------------------
    let docs_f32: Vec<Array2<f32>> = (0..N_DOCS)
        .map(|i| {
            let c = clusters.row(i % N_TOPICS).to_owned();
            let jitter = randn(1, RANK, 3000 + i as u64);
            let center = &c + &(0.5 * &jitter.row(0));
            // DOC_TOKENS latent codes around this doc's center.
            let mut z = randn(DOC_TOKENS, RANK, 4000 + i as u64);
            z.mapv_inplace(|x| x * SIGMA_DOC);
            for mut row in z.axis_iter_mut(Axis(0)) {
                row += &center;
            }
            embed(&z, &proj, &spectrum, 5000 + i as u64)
        })
        .collect();

    let docs_bits: Vec<Array2<u8>> = docs_f32.iter().map(|d| binarize(&d.view())).collect();
    let docs_i8: Vec<_> = docs_f32.iter().map(|d| quantize_i8(&d.view())).collect();

    // --- build queries (each targets a latent cluster -> graded relevance) ----
    let queries: Vec<Array2<f32>> = (0..N_QUERIES)
        .map(|q| {
            let k = q % N_TOPICS;
            let center = clusters.row(k).to_owned();
            let mut z = randn(QUERY_TOKENS, RANK, 6000 + q as u64);
            z.mapv_inplace(|x| x * SIGMA_Q);
            for mut row in z.axis_iter_mut(Axis(0)) {
                row += &center;
            }
            embed(&z, &proj, &spectrum, 7000 + q as u64)
        })
        .collect();

    // --- build the residual codec (stock PLAID) at nbits=4 and nbits=2 --------
    let km_cfg = ComputeKmeansConfig {
        num_partitions: Some(256),
        seed: 42,
        force_cpu: true,
        ..Default::default()
    };
    let centroids = compute_kmeans(&docs_f32, &km_cfg).expect("kmeans");

    // Encode all docs against a codec; returns (packed_residuals, codes) per doc.
    let build_residual = |nbits: usize| {
        let cfg = IndexConfig {
            nbits,
            seed: Some(42),
            force_cpu: true,
            ..Default::default()
        };
        let art = prepare_codec_artifacts(&docs_f32, centroids.clone(), &cfg).expect("codec");
        let codec = art.codec;
        let store: Vec<(Array2<u8>, ndarray::Array1<usize>)> = docs_f32
            .iter()
            .map(|d| {
                let codes = codec.compress_into_codes_cpu(d);
                // residual = d - centroid[code]
                let mut resid = d.clone();
                for (t, mut row) in resid.axis_iter_mut(Axis(0)).enumerate() {
                    let cen = codec.centroids.row(codes[t]);
                    row -= &cen;
                }
                let packed = codec.quantize_residuals(&resid).expect("quantize");
                (packed, codes)
            })
            .collect();
        (codec, store)
    };
    let (codec4, store4) = build_residual(4);
    let (codec2, store2) = build_residual(2);

    // --- storage report (true bytes/token, including centroid codes) ----------
    let float_bpt = DIM * 4;
    let bin_bpt = DIM.div_ceil(8);
    let i8_bpt = DIM;
    let res_bpt = |nbits: usize| DIM * nbits / 8 + 4; // + 4-byte i32 centroid code
    println!("Storage (bytes/token, dim={DIM}):");
    println!("  float32:        {float_bpt:>4} B   1.0x");
    println!(
        "  residual nbits4: {:>4} B   {:.1}x",
        res_bpt(4),
        float_bpt as f64 / res_bpt(4) as f64
    );
    println!(
        "  residual nbits2: {:>4} B   {:.1}x",
        res_bpt(2),
        float_bpt as f64 / res_bpt(2) as f64
    );
    println!(
        "  int8 x int8:    {i8_bpt:>4} B   {:.1}x",
        float_bpt as f64 / i8_bpt as f64
    );
    println!(
        "  int8 x 1-bit:   {bin_bpt:>4} B   {:.1}x  (doc side)",
        float_bpt as f64 / bin_bpt as f64
    );
    println!(
        "  1-bit x 1-bit:  {bin_bpt:>4} B   {:.1}x",
        float_bpt as f64 / bin_bpt as f64
    );
    println!();

    // --- latency (time full Stage-2 rescore of all N_DOCS candidates) ---------
    // Warm up once (page-ins, feature detection, cache), then take the MIN over
    // several reps — min is the most stable estimator of true kernel cost,
    // rejecting scheduler/thermal noise.
    const REPS: usize = 3;
    let mut sink = 0.0f32;
    let time_it = |sink: &mut f32, f: &dyn Fn(&mut f32)| -> Duration {
        f(sink); // warmup
        let mut best = Duration::from_secs(u64::MAX);
        for _ in 0..REPS {
            let t = Instant::now();
            f(sink);
            best = best.min(t.elapsed());
        }
        best
    };

    let t_float = time_it(&mut sink, &|s| {
        for q in &queries {
            for d in &docs_f32 {
                *s += maxsim_score(&q.view(), &d.view());
            }
        }
    });
    let t_res4 = time_it(&mut sink, &|s| {
        for q in &queries {
            for (packed, codes) in &store4 {
                let recon = codec4.decompress(packed, &codes.view()).unwrap();
                *s += maxsim_score(&q.view(), &recon.view());
            }
        }
    });
    let t_res2 = time_it(&mut sink, &|s| {
        for q in &queries {
            for (packed, codes) in &store2 {
                let recon = codec2.decompress(packed, &codes.view()).unwrap();
                *s += maxsim_score(&q.view(), &recon.view());
            }
        }
    });
    let t_i8 = time_it(&mut sink, &|s| {
        for q in &queries {
            let q8 = quantize_i8(&q.view());
            for d in &docs_i8 {
                *s += maxsim_i8(&q8, d, DIM);
            }
        }
    });
    let t_bin = time_it(&mut sink, &|s| {
        for q in &queries {
            let q8 = quantize_query_i8(&q.view());
            for bits in &docs_bits {
                *s += maxsim_binary_i8(&q8, &bits.view(), DIM);
            }
        }
    });
    let t_bb = time_it(&mut sink, &|s| {
        for q in &queries {
            let qbits = binarize(&q.view());
            for bits in &docs_bits {
                *s += maxsim_binary_binary(&qbits.view(), &bits.view(), DIM);
            }
        }
    });

    let ms = |d: Duration| d.as_secs_f64() * 1e3 / N_QUERIES as f64;
    let sp = |d: Duration| t_float.as_secs_f64() / d.as_secs_f64();
    println!("Latency (Stage-2 rescore of {N_DOCS} candidates, per query):");
    println!("  float (BLAS):    {:>7.2} ms   1.00x", ms(t_float));
    println!(
        "  residual nbits4: {:>7.2} ms   {:.2}x",
        ms(t_res4),
        sp(t_res4)
    );
    println!(
        "  residual nbits2: {:>7.2} ms   {:.2}x",
        ms(t_res2),
        sp(t_res2)
    );
    println!("  int8 x int8:     {:>7.2} ms   {:.2}x", ms(t_i8), sp(t_i8));
    println!(
        "  int8 x 1-bit:    {:>7.2} ms   {:.2}x",
        ms(t_bin),
        sp(t_bin)
    );
    println!("  1-bit x 1-bit:   {:>7.2} ms   {:.2}x", ms(t_bb), sp(t_bb));
    println!();

    // --- quality: NDCG@10 vs float ground truth -------------------------------
    // Precompute residual reconstructions once (decode cost already measured in
    // the latency section; here we only care about ranking quality).
    let recon4: Vec<Array2<f32>> = store4
        .iter()
        .map(|(p, c)| codec4.decompress(p, &c.view()).unwrap())
        .collect();
    let recon2: Vec<Array2<f32>> = store2
        .iter()
        .map(|(p, c)| codec2.decompress(p, &c.view()).unwrap())
        .collect();

    let mut nd = [0.0f64; 6]; // float, res4, res2, i8, bin, bb
    let mut rel_count = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let float_scores: Vec<f32> = (0..N_DOCS)
            .map(|d| maxsim_score(&q.view(), &docs_f32[d].view()))
            .collect();
        let (lo, hi) = float_scores
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &s| (a.min(s), b.max(s)));
        let span = (hi - lo).max(1e-6);
        let rel: Vec<f32> = float_scores
            .iter()
            .map(|&s| 3.0 * (s - lo) / span)
            .collect();
        if qi == 0 {
            rel_count = rel.iter().filter(|&&r| r > 1.5).count();
        }

        let float_order = {
            let mut o: Vec<usize> = (0..N_DOCS).collect();
            o.sort_by(|&a, &b| float_scores[b].total_cmp(&float_scores[a]));
            o
        };
        let o_res4 = rank_by(N_DOCS, |d| maxsim_score(&q.view(), &recon4[d].view()));
        let o_res2 = rank_by(N_DOCS, |d| maxsim_score(&q.view(), &recon2[d].view()));
        let q8i = quantize_i8(&q.view());
        let o_i8 = rank_by(N_DOCS, |d| maxsim_i8(&q8i, &docs_i8[d], DIM));
        let q8b = quantize_query_i8(&q.view());
        let o_bin = rank_by(N_DOCS, |d| {
            maxsim_binary_i8(&q8b, &docs_bits[d].view(), DIM)
        });
        let qbits = binarize(&q.view());
        let o_bb = rank_by(N_DOCS, |d| {
            maxsim_binary_binary(&qbits.view(), &docs_bits[d].view(), DIM)
        });

        for (i, order) in [&float_order, &o_res4, &o_res2, &o_i8, &o_bin, &o_bb]
            .iter()
            .enumerate()
        {
            nd[i] += ndcg_at_k(order, &rel, K) as f64;
        }
    }
    for x in nd.iter_mut() {
        *x /= N_QUERIES as f64;
    }
    let ret = |x: f64| 100.0 * x / nd[0];

    println!("Quality — NDCG@{K} (ground truth = float MaxSim ranking):");
    println!("  (~{rel_count} relevant docs/query, rel>1.5)");
    println!("  float:           {:.4}   (upper bound)", nd[0]);
    println!(
        "  residual nbits4: {:.4}   ({:.1}% retention)",
        nd[1],
        ret(nd[1])
    );
    println!(
        "  residual nbits2: {:.4}   ({:.1}% retention)",
        nd[2],
        ret(nd[2])
    );
    println!(
        "  int8 x int8:     {:.4}   ({:.1}% retention)",
        nd[3],
        ret(nd[3])
    );
    println!(
        "  int8 x 1-bit:    {:.4}   ({:.1}% retention)",
        nd[4],
        ret(nd[4])
    );
    println!(
        "  1-bit x 1-bit:   {:.4}   ({:.1}% retention)",
        nd[5],
        ret(nd[5])
    );

    if sink.is_nan() {
        println!("(sink={sink})");
    }
}
