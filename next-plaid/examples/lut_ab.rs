//! Real-index A/B for the residual LUT kernel: whole-search wall time plus a
//! top-k dump for exact parity.
//!
//! Build the same file from two checkouts (e.g. v1.7.0 and this branch) and
//! alternate the binaries; scores are supposed to be bit-identical, so the
//! dumps must diff clean while the times move.
//!
//! Usage:
//!   cargo run -p next-plaid --release --example lut_ab -- \
//!       <index_dir> <queries.npy> <out_dir> <tag>
//! Env: NQ (queries, default 50), ROUNDS (timed passes, default 3),
//!      WARMUP (untimed passes, default 1), NFULL (n_full_scores, default 4096).
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{s, Array2, Array3};
use ndarray_npy::ReadNpyExt;
use next_plaid::index::MmapIndex;
use next_plaid::SearchParameters;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Median and the interquartile spread as a fraction of the median — the
/// noise verdict this bench refuses to be read without.
fn stats(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = s[s.len() / 2];
    let (q1, q3) = (s[s.len() / 4], s[(3 * s.len()) / 4]);
    (med, (q3 - q1) / med)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let index_dir = args[1].clone();
    let queries_path = PathBuf::from(&args[2]);
    let out_dir = PathBuf::from(&args[3]);
    let tag = args[4].clone();
    std::fs::create_dir_all(&out_dir)?;

    let nq: usize = env("NQ", 50);
    let rounds: usize = env("ROUNDS", 3);
    let warmup: usize = env("WARMUP", 1);
    let n_full: usize = env("NFULL", 4096);

    let all: Array3<f32> = Array3::read_npy(File::open(&queries_path)?)?;
    let nq = nq.min(all.shape()[0]);
    let queries: Vec<Array2<f32>> = (0..nq)
        .map(|i| all.slice(s![i, .., ..]).to_owned())
        .collect();

    let index = MmapIndex::load(&index_dir)?;
    let params = SearchParameters {
        n_full_scores: n_full,
        ..Default::default()
    };

    println!(
        "[{tag}] {} docs, {} embeddings, nbits {}, dim {}; {nq} queries x {} tokens; \
         n_full_scores {n_full}; rayon {}",
        index.metadata.num_documents,
        index.metadata.num_embeddings,
        index.metadata.nbits,
        all.shape()[2],
        all.shape()[1],
        rayon::current_num_threads()
    );

    for _ in 0..warmup {
        for q in &queries {
            index.search(q, &params, None)?;
        }
    }

    let mut per_round = Vec::with_capacity(rounds);
    let mut last = Vec::new();
    for _ in 0..rounds {
        let t0 = Instant::now();
        let mut res = Vec::with_capacity(nq);
        for q in &queries {
            res.push(index.search(q, &params, None)?);
        }
        per_round.push(t0.elapsed().as_secs_f64() * 1e3 / nq as f64);
        last = res;
    }
    let (med, iqr) = stats(&per_round);
    for (i, ms) in per_round.iter().enumerate() {
        println!("[{tag}] round {i}: {ms:.4} ms/query");
    }
    println!(
        "[{tag}] median {med:.4} ms/query, IQR {:.1}% of median",
        iqr * 100.0
    );

    // Exact-parity dump: the score BITS, not a rounded print — the claim is
    // bit-identical scoring, and a formatted float would hide a last-bit move.
    let path = out_dir.join(format!("top_{tag}_c{n_full}.tsv"));
    let mut f = File::create(&path)?;
    for (qi, r) in last.iter().enumerate() {
        for (rank, (&doc, &score)) in r.passage_ids.iter().zip(r.scores.iter()).enumerate() {
            writeln!(f, "{qi}\t{rank}\t{doc}\t{:08x}", score.to_bits())?;
        }
    }
    println!("[{tag}] wrote {}", path.display());
    Ok(())
}
