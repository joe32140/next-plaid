//! Reproduce the asymmetric-quantization NDCG comparison on a BEIR dataset.
//!
//! Reads a ColBERT embedding bundle produced by `scripts/embed_beir_colbert.py`,
//! builds a residual (float-precision) index and a binary index over the same
//! vectors, and reports NDCG@10 and document bytes/token for each — the
//! multi-vector analogue of mixedbread's "Int8 x Binary" retention table.
//!
//! Usage: `cargo run --release --example binary_ndcg -- <bundle_dir>`

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use ndarray::{s, Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::index::MmapIndex;
use next_plaid::{IndexConfig, SearchParameters};

type Qrels = HashMap<String, HashMap<String, i64>>;

/// Split a concatenated `[total_tokens, dim]` array into per-item matrices using
/// the per-item token counts.
fn unpack(concat: &Array2<f32>, lens: &Array1<i64>) -> Vec<Array2<f32>> {
    let mut items = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &len in lens {
        let len = len as usize;
        items.push(concat.slice(s![off..off + len, ..]).to_owned());
        off += len;
    }
    items
}

fn read_f32(path: &Path) -> Array2<f32> {
    Array2::read_npy(File::open(path).unwrap()).unwrap()
}
fn read_i64(path: &Path) -> Array1<i64> {
    Array1::read_npy(File::open(path).unwrap()).unwrap()
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    serde_json::from_reader(File::open(path).unwrap()).unwrap()
}

/// NDCG@k with the exponential gain `2^rel - 1` used by BEIR / pytrec_eval.
fn ndcg_at_k(ranked_ids: &[String], rels: &HashMap<String, i64>, k: usize) -> f64 {
    let gain = |r: i64| 2f64.powi(r as i32) - 1.0;
    let discount = |i: usize| 1.0 / ((i + 2) as f64).log2();

    let dcg: f64 = ranked_ids
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| gain(*rels.get(d).unwrap_or(&0)) * discount(i))
        .sum();

    let mut ideal: Vec<i64> = rels.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &r)| gain(r) * discount(i))
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Search params for two evaluation regimes:
///   isolated  = probe everything, no centroid pruning -> the quantizer's ceiling
///               (the index analogue of exhaustive MaxSim, ANN recall removed).
///   deployed  = PLAID defaults (n_ivf_probe=8, n_full_scores=4096, pruning ON)
///               -> the real two-stage system a user actually ships.
fn regime_params(deployed: bool, n_docs: usize) -> SearchParameters {
    if deployed {
        SearchParameters {
            top_k: 10,
            ..Default::default()
        }
    } else {
        SearchParameters {
            top_k: 100,
            n_full_scores: n_docs,
            n_ivf_probe: 256,
            centroid_score_threshold: None,
            ..Default::default()
        }
    }
}

/// Build an index with the given config and return (mean NDCG@10, bytes/token).
#[allow(clippy::too_many_arguments)]
fn evaluate(
    name: &str,
    config: &IndexConfig,
    params: &SearchParameters,
    docs: &[Array2<f32>],
    corpus_ids: &[String],
    queries: &[Array2<f32>],
    query_ids: &[String],
    qrels: &Qrels,
) -> (f64, usize) {
    let dir = std::env::temp_dir().join(format!("np_ndcg_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    let index = MmapIndex::create_with_kmeans(docs, dir.to_str().unwrap(), config).unwrap();
    let bytes_per_token = index.mmap_residuals.ncols();

    let mut sum_ndcg = 0.0;
    let mut counted = 0;
    for (q, qid) in queries.iter().zip(query_ids) {
        let Some(rels) = qrels.get(qid) else { continue };
        let result = index.search(q, params, None).unwrap();
        let ranked: Vec<String> = result
            .passage_ids
            .iter()
            .map(|&pid| corpus_ids[pid as usize].clone())
            .collect();
        sum_ndcg += ndcg_at_k(&ranked, rels, 10);
        counted += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    (sum_ndcg / counted as f64, bytes_per_token)
}

fn main() {
    let bundle = PathBuf::from(
        std::env::args()
            .nth(1)
            .or_else(|| std::env::var("NDCG_BUNDLE").ok())
            .expect("usage: binary_ndcg <bundle_dir>"),
    );

    let docs = unpack(
        &read_f32(&bundle.join("corpus.npy")),
        &read_i64(&bundle.join("corpus_lens.npy")),
    );
    let queries = unpack(
        &read_f32(&bundle.join("queries.npy")),
        &read_i64(&bundle.join("query_lens.npy")),
    );
    let corpus_ids: Vec<String> = read_json(&bundle.join("corpus_ids.json"));
    let query_ids: Vec<String> = read_json(&bundle.join("query_ids.json"));
    let qrels: Qrels = read_json(&bundle.join("qrels.json"));
    let dim = docs[0].ncols();
    println!("docs={} queries={} dim={dim}\n", docs.len(), queries.len());

    let base = IndexConfig {
        nbits: 4,
        seed: Some(42),
        ..Default::default()
    };
    let residual = IndexConfig {
        binary: false,
        ..base.clone()
    };
    let binary = IndexConfig {
        binary: true,
        ..base
    };

    // The exhaustive (probe-all) isolated pass is redundant when the Python
    // decomposition already gives that ceiling and is slow on big corpora, so
    // NDCG_DEPLOYED_ONLY=1 skips it and reports just the deployed PLAID number.
    let deployed_only = std::env::var("NDCG_DEPLOYED_ONLY").is_ok();
    let f32_bytes = dim * 4;
    println!(
        "{:<22} {:>10} {:>10} {:>14} {:>10}",
        "scheme", "isolated", "deployed", "doc bytes/tok", "vs f32"
    );

    // Each row: quantizer-isolated ceiling (probe all, no pruning) next to the
    // deployed PLAID number (n_ivf_probe=8, n_full_scores=4096, pruning ON).
    let retained = |name: &str, config: &IndexConfig| -> (f64, f64) {
        let iso = if deployed_only {
            f64::NAN
        } else {
            evaluate(
                name,
                config,
                &regime_params(false, docs.len()),
                &docs,
                &corpus_ids,
                &queries,
                &query_ids,
                &qrels,
            )
            .0
        };
        let (dep, bytes) = evaluate(
            name,
            config,
            &regime_params(true, docs.len()),
            &docs,
            &corpus_ids,
            &queries,
            &query_ids,
            &qrels,
        );
        println!(
            "{name:<22} {iso:>10.4} {dep:>10.4} {bytes:>14} {:>9}x",
            f32_bytes / bytes
        );
        (iso, dep)
    };

    let (res_iso, res_dep) = retained("residual (nbits=4)", &residual);
    let (bin_iso, bin_dep) = retained("binary (int8 x 1-bit)", &binary);

    println!(
        "\nbinary retains {:.1}% (deployed) of residual NDCG@10",
        100.0 * bin_dep / res_dep
    );
    if !deployed_only {
        println!(
            "binary retains {:.1}% (isolated) of residual; deployed two-stage keeps \
             {:.1}% (residual) / {:.1}% (binary) of the isolated ceiling",
            100.0 * bin_iso / res_iso,
            100.0 * res_dep / res_iso,
            100.0 * bin_dep / bin_iso
        );
    }

    // Machine-readable line for the Modal harness to parse (NDCG_JSON=1).
    if std::env::var("NDCG_JSON").is_ok() {
        println!(
            "NDCG_JSON {{\"docs\":{},\"queries\":{},\"dim\":{},\
             \"residual_isolated\":{:.6},\"residual_deployed\":{:.6},\
             \"binary_isolated\":{:.6},\"binary_deployed\":{:.6}}}",
            docs.len(),
            queries.len(),
            dim,
            res_iso,
            res_dep,
            bin_iso,
            bin_dep
        );
    }
}
