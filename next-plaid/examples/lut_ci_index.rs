//! Build a small synthetic residual index plus in-distribution queries, so a
//! CI runner can exercise the rescore path with no private corpus.
//!
//! Usage:
//!   cargo run -p next-plaid --release --example lut_ci_index -- <out_dir>
//! Env: DOCS (default 3000), TOKENS (per doc, default 100), QUERIES (default
//!      50), QTOKENS (default 32), DIM (default 128), NBITS (default 4).
//!
//! Writes `<out_dir>/index/` and `<out_dir>/queries.npy` ([QUERIES, QTOKENS,
//! DIM]), and does nothing if the index is already there — the two arms of an
//! A/B must read the *same* index.
//!
//! Queries are perturbed document tokens on purpose. Random unit vectors are
//! near-orthogonal to every centroid, so the default
//! `centroid_score_threshold` of 0.4 prunes everything and search returns
//! empty: a benchmark built that way measures an empty shortlist and reads as
//! a flat, fast, meaningless line.
use std::fs::File;
use std::path::PathBuf;

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::WriteNpyExt;
use next_plaid::IndexConfig;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn normalize_rows(a: &mut Array2<f32>) {
    for mut row in a.axis_iter_mut(Axis(0)) {
        let n = row.dot(&row).sqrt().max(1e-12);
        row.mapv_inplace(|x| x / n);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: lut_ci_index <out_dir>"),
    );
    let n_docs: usize = env("DOCS", 3000);
    let tokens: usize = env("TOKENS", 100);
    let n_queries: usize = env("QUERIES", 50);
    let q_tokens: usize = env("QTOKENS", 32);
    let dim: usize = env("DIM", 128);
    let nbits: usize = env("NBITS", 4);
    let index_dir = out.join("index");
    std::fs::create_dir_all(&out)?;

    // One RNG, one seed: both arms and every rerun must see the same corpus.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let n_clusters = 64usize;
    let mut centers = Array2::<f32>::from_shape_fn((n_clusters, dim), |_| rng.gen_range(-1.0..1.0));
    normalize_rows(&mut centers);

    let docs: Vec<Array2<f32>> = (0..n_docs)
        .map(|_| {
            let c = rng.gen_range(0..n_clusters);
            let mut d = Array2::<f32>::from_shape_fn((tokens, dim), |(_, j)| {
                centers[[c, j]] + rng.gen_range(-0.35..0.35)
            });
            normalize_rows(&mut d);
            d
        })
        .collect();

    if !index_dir.join("metadata.json").exists() {
        let config = IndexConfig {
            nbits,
            seed: Some(42),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let meta = next_plaid::index::create_index_with_kmeans_files(
            &docs,
            index_dir.to_str().unwrap(),
            &config,
        )?;
        println!(
            "built {} docs / {} embeddings / {} partitions, nbits {nbits}, in {:.1}s",
            n_docs,
            meta.num_embeddings,
            meta.num_partitions,
            t0.elapsed().as_secs_f64()
        );
    } else {
        println!("index already at {}, reusing", index_dir.display());
    }

    let qpath = out.join("queries.npy");
    if !qpath.exists() {
        // Each query draws its tokens from several documents, not one. A
        // query copied from a single document matches only that document's
        // cluster, the IVF shortlist collapses to a handful of candidates,
        // and the rescore depth then stops changing anything — the bench goes
        // flat and reports a kernel ratio near 1 no matter what the kernel
        // does. Real ColBERT query tokens spread over many centroids.
        let docs_per_query = 8usize;
        let mut q = Array3::<f32>::zeros((n_queries, q_tokens, dim));
        for qi in 0..n_queries {
            let picks: Vec<usize> = (0..docs_per_query)
                .map(|_| rng.gen_range(0..n_docs))
                .collect();
            for t in 0..q_tokens {
                let d = &docs[picks[t % docs_per_query]];
                let src = d.row(rng.gen_range(0..d.nrows()));
                let mut row = q.slice_mut(ndarray::s![qi, t, ..]);
                for j in 0..dim {
                    row[j] = src[j] + rng.gen_range(-0.1..0.1);
                }
                let n = row.dot(&row).sqrt().max(1e-12);
                row.mapv_inplace(|x| x / n);
            }
        }
        q.write_npy(File::create(&qpath)?)?;
        println!(
            "wrote {} queries x {q_tokens} tokens to {}",
            n_queries,
            qpath.display()
        );
    } else {
        println!("queries already at {}, reusing", qpath.display());
    }
    Ok(())
}
