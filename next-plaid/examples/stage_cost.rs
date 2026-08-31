//! What does a query actually cost, and where? — the missing cost measurement.
//!
//! The quantization study this supports argues about *candidate depth*: a damaged
//! checkpoint needs a much larger `n_full_scores` to reach the same recall as a
//! repaired one (10–40× on the measured cells). That arithmetic only converts into
//! an end-to-end speedup if we know how query time splits between the depth-
//! independent part (IVF probe + centroid-LUT approximate scoring) and the
//! depth-proportional part (exact rescoring of the top candidates).
//!
//! So: sweep `n_full_scores` on a real mmap index and fit
//!
//!     T(C) = s + r·C
//!
//! `s` is what you pay regardless of candidate depth; `r·C` is what shrinking the
//! candidate set actually buys. Everything is measured through the public search
//! API on the real memory-mapped path — no synthetic token buffers, because a
//! synthetic rescore bench was previously measured to *undercost* the real
//! random-access path by ~1.45×.
//!
//! Run:
//!   cargo run -p next-plaid --release --example stage_cost -- <index_dir> [n_queries]
//!
//! On Apple Silicon pass `--target aarch64-apple-darwin` if the default toolchain
//! is x86_64 (a Rosetta build measures the wrong kernels entirely), and run it on
//! an otherwise idle machine.

use std::time::Instant;

use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;

use next_plaid::index::MmapIndex;
use next_plaid::search::SearchParameters;

const QUERY_TOKENS: usize = 32;
const DEPTHS: [usize; 6] = [64, 128, 256, 512, 1024, 4096];

/// L2-normalised random query tokens — the geometry the kernels see. Query
/// *content* changes which documents are visited, not the per-candidate cost, and
/// this bench measures cost, so a fixed seeded draw keeps every depth comparable.
fn random_query(dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut q: Array2<f32> = Array2::random_using((QUERY_TOKENS, dim), StandardNormal, &mut rng);
    for mut row in q.rows_mut() {
        let n = row.dot(&row).sqrt().max(1e-9);
        row.mapv_inplace(|v| v / n);
    }
    q
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let index_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/Users/joe/beir-data/demo_indexes/scifact_lateon_reg_r4".to_string());
    let n_queries: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let index = MmapIndex::load(&index_dir)?;
    let dim = index.metadata.embedding_dim;
    println!(
        "index {index_dir}\n  {} docs, {} embeddings, dim {dim}, avg doclen {:.1}, binary={}, nbits={}",
        index.metadata.num_documents,
        index.metadata.num_embeddings,
        index.metadata.avg_doclen,
        index.metadata.binary,
        index.metadata.nbits,
    );

    let queries: Vec<Array2<f32>> = (0..n_queries)
        .map(|i| random_query(dim, i as u64))
        .collect();

    // A flat T(C) can mean two very different things, and only one of them is a
    // statement about cost: either rescoring really is free, or CENTROID PRUNING
    // capped the candidate list below `n_full_scores`, in which case every depth
    // did identical work and the sweep measured nothing. So run the sweep under
    // two retrieval regimes — pruned (the shipped default) and unpruned with deep
    // IVF probing, which forces a genuinely large candidate set.
    let regimes: [(&str, Option<f32>, usize); 2] = [
        ("pruned t_cs=0.4, probe=8 (shipped default)", Some(0.4), 8),
        ("UNPRUNED t_cs=None, probe=64 (forces candidates)", None, 64),
    ];

    for (label, t_cs, probe) in regimes {
        let base_params = SearchParameters {
            batch_size: 1,
            n_full_scores: 1024,
            top_k: 10,
            n_ivf_probe: probe,
            centroid_batch_size: 100_000,
            centroid_score_threshold: t_cs,
        };
        // Warm the page cache and allocator: the first searches of a run pay for
        // faulting the mmap in, which is a one-off, not a per-query cost.
        for q in queries.iter().take(8) {
            let _ = index.search(q, &base_params, None)?;
        }

        println!("\n── {label}");
        println!(
            "{:>8}  {:>12}  {:>12}  {:>14}",
            "C", "ms/query", "rel. to C=64", "µs/candidate"
        );
        let mut points: Vec<(f64, f64)> = Vec::new();
        for c in DEPTHS {
            let params = SearchParameters {
                n_full_scores: c,
                ..base_params.clone()
            };
            // Median of per-query times: a mean over an mmap path is hostage to a
            // single page-fault outlier.
            let mut times: Vec<f64> = Vec::with_capacity(queries.len());
            for q in &queries {
                let t = Instant::now();
                let _ = index.search(q, &params, None)?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = times[times.len() / 2];
            points.push((c as f64, med));
            let base = points[0].1;
            println!(
                "{c:>8}  {med:>12.3}  {:>12.2}x  {:>14.3}",
                med / base,
                med * 1e3 / c as f64
            );
        }

        // Least squares on T(C) = s + r·C.
        let n = points.len() as f64;
        let sx: f64 = points.iter().map(|p| p.0).sum();
        let sy: f64 = points.iter().map(|p| p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let r = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let s = (sy - r * sx) / n;
        println!(
            "  fit T(C) = {s:.3} ms + {:.4} µs·C   → rescore share at C=4096: {:.1}%",
            r * 1e3,
            100.0 * r * 4096.0 / (s + r * 4096.0)
        );
        for (from, to) in [(2500.0, 250.0), (1024.0, 100.0)] {
            println!(
                "  equal-recall C {from:.0} -> {to:.0}:  {:.2}x end-to-end",
                (s + r * from) / (s + r * to)
            );
        }
    }
    println!(
        "\nNOTE: this index holds {} documents, so C=4096 is {:.0}% of the corpus —\n\
         candidate-depth economics can only be read from a corpus where C << N.",
        index.metadata.num_documents,
        100.0 * 4096.0 / index.metadata.num_documents as f64
    );
    Ok(())
}
