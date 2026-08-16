//! NDCG@10 quality ladder for the residual codecs on a real ColBERT bundle.
//!
//! This isolates *codec reconstruction fidelity*. For each profile we build an
//! index, reconstruct every document (`centroid + decoded residual`), then rank
//! with exhaustive float MaxSim — so there is **no** stage-1 (centroid pruning /
//! `n_ivf_probe`) confound and the only variable across rows is the residual
//! codec. The `float` row ranks the *original* embeddings for the lossless
//! ceiling. Because k-means uses a fixed seed, all profiles share identical
//! centroids; the NDCG gaps are pure quantization loss.
//!
//! Bundle layout (BEIR ColBERTv2 export, one directory):
//!   * `corpus.npy`      `[T, dim]`  f32   — all doc tokens, concatenated
//!   * `corpus_lens.npy` `[N]`       i64   — tokens per doc
//!   * `corpus_ids.json` `[N]`       str   — external doc id per doc
//!   * `queries.npy`     `[Tq, dim]` f32
//!   * `query_lens.npy`  `[Q]`       i64
//!   * `query_ids.json`  `[Q]`       str
//!   * `qrels.json`      `{ qid: { docid: rel } }`
//!
//! Run: `cargo run -p next-plaid --release --example ndcg_eval -- <bundle_dir>`

use ndarray::{s, Array1, Array2, Axis};
use ndarray_npy::read_npy;
use next_plaid::index::MmapIndex;
use next_plaid::IndexConfig;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Split a `[sum(lens), dim]` token matrix into one `[len_i, dim]` matrix per row
/// group (per document / per query).
fn split_rows(mat: &Array2<f32>, lens: &[i64]) -> Vec<Array2<f32>> {
    let mut out = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        out.push(mat.slice(s![off..off + l, ..]).to_owned());
        off += l;
    }
    out
}

/// Flatten a per-doc token list into one `[sum(len), dim]` matrix plus each
/// doc's `(offset, len)` window into it. One big `Q · Dᵀ` GEMM per query is far
/// faster than a small matmul per (query, doc) pair.
fn flatten(docs: &[Array2<f32>], dim: usize) -> (Array2<f32>, Vec<(usize, usize)>) {
    let total: usize = docs.iter().map(|d| d.nrows()).sum();
    let mut flat = Array2::<f32>::zeros((total, dim));
    let mut index = Vec::with_capacity(docs.len());
    let mut off = 0usize;
    for d in docs {
        let len = d.nrows();
        flat.slice_mut(s![off..off + len, ..]).assign(d);
        index.push((off, len));
        off += len;
    }
    (flat, index)
}

/// Standard NDCG@k with the `2^rel - 1` gain (graded relevance).
fn ndcg_at_k(ranked: &[usize], rels: &HashMap<usize, f32>, k: usize) -> f32 {
    let dcg: f32 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = *rels.get(id).unwrap_or(&0.0);
            (2f32.powf(rel) - 1.0) / (i as f32 + 2.0).log2()
        })
        .sum();
    let mut ideal: Vec<f32> = rels.values().copied().collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idcg: f32 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, rel)| (2f32.powf(*rel) - 1.0) / (i as f32 + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Mean per-token cosine between reconstructed and original embeddings — a
/// direct read on how much the codec distorts each vector.
fn mean_recon_cosine(recon: &[Array2<f32>], orig: &[Array2<f32>]) -> f32 {
    let mut cos_sum = 0.0f64;
    let mut n = 0u64;
    for (r, o) in recon.iter().zip(orig.iter()) {
        for (gr, go) in r.axis_iter(Axis(0)).zip(o.axis_iter(Axis(0))) {
            let dot = gr.dot(&go);
            let ng = gr.dot(&gr).sqrt().max(1e-12);
            let no = go.dot(&go).sqrt().max(1e-12);
            cos_sum += (dot / (ng * no)) as f64;
            n += 1;
        }
    }
    (cos_sum / n.max(1) as f64) as f32
}

/// Mean NDCG@10 over all judged queries, ranking every doc by exhaustive MaxSim.
/// `flat`/`dindex` come from [`flatten`]; scoring is one `Q · Dᵀ` GEMM per query
/// followed by a per-doc segment-max over the query-token axis.
fn eval_ndcg(
    flat: &Array2<f32>,
    dindex: &[(usize, usize)],
    queries: &[Array2<f32>],
    query_ids: &[String],
    corpus_ids: &[String],
    qrels: &HashMap<String, HashMap<String, f32>>,
) -> f32 {
    let row_of_docid: HashMap<&str, usize> = corpus_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let dt = flat.t(); // [dim, total_tokens]

    let scores: Vec<f32> = (0..queries.len())
        .into_par_iter()
        .filter_map(|qi| {
            let rel_map = qrels.get(&query_ids[qi])?;
            let rels_row: HashMap<usize, f32> = rel_map
                .iter()
                .filter_map(|(d, &r)| row_of_docid.get(d.as_str()).map(|&row| (row, r)))
                .collect();
            if rels_row.is_empty() {
                return None;
            }
            // sims[t, j] = <query_token_t, doc_token_j> for all corpus tokens.
            let sims = queries[qi].dot(&dt); // [n_q_tok, total_tokens]
            let mut score = vec![0f32; dindex.len()];
            for row in sims.axis_iter(Axis(0)) {
                let rs = row.as_slice().unwrap();
                for (di, &(off, len)) in dindex.iter().enumerate() {
                    let mut m = f32::NEG_INFINITY;
                    for &v in &rs[off..off + len] {
                        if v > m {
                            m = v;
                        }
                    }
                    score[di] += m;
                }
            }
            let mut scored: Vec<(usize, f32)> = score.into_iter().enumerate().collect();
            // Descending by score; ties broken by doc id for determinism.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            let ranked: Vec<usize> = scored.iter().map(|(i, _)| *i).collect();
            Some(ndcg_at_k(&ranked, &rels_row, 10))
        })
        .collect();

    let sum: f32 = scores.iter().sum();
    sum / scores.len().max(1) as f32
}

fn scalar_cfg(nbits: usize) -> IndexConfig {
    IndexConfig {
        nbits,
        seed: Some(42),
        ..Default::default()
    }
}

/// Ternary with the *equal-mass* bucket split (cutoffs at the 1/3 and 2/3
/// residual quantiles). This was the default until the tau sweep; it stays in
/// the ladder as the control the shipped default has to beat.
fn ternary_cfg() -> IndexConfig {
    IndexConfig {
        nbits: 2, // nominal; the ternary codec supersedes it
        seed: Some(42),
        ternary: true,
        ternary_tau: None,
        ..Default::default()
    }
}

/// Ternary with an explicit dead-zone width (`|r| < tau*sigma` stores 0)
/// instead of the equal-mass 1/3–2/3 quantile split.
fn ternary_tau_cfg(tau: f32) -> IndexConfig {
    IndexConfig {
        ternary_tau: Some(tau),
        ..ternary_cfg()
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: ndcg_eval <bundle_dir>");
    let dir = Path::new(&dir);
    let read_json = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();

    let corpus: Array2<f32> = read_npy(dir.join("corpus.npy")).unwrap();
    let corpus_lens: Array1<i64> = read_npy(dir.join("corpus_lens.npy")).unwrap();
    let queries: Array2<f32> = read_npy(dir.join("queries.npy")).unwrap();
    let query_lens: Array1<i64> = read_npy(dir.join("query_lens.npy")).unwrap();
    let corpus_ids: Vec<String> = serde_json::from_str(&read_json("corpus_ids.json")).unwrap();
    let query_ids: Vec<String> = serde_json::from_str(&read_json("query_ids.json")).unwrap();
    let qrels: HashMap<String, HashMap<String, f32>> =
        serde_json::from_str(&read_json("qrels.json")).unwrap();

    let docs = split_rows(&corpus, corpus_lens.as_slice().unwrap());
    let qs = split_rows(&queries, query_lens.as_slice().unwrap());
    let dim = docs[0].ncols();
    let n_judged = query_ids.iter().filter(|q| qrels.contains_key(*q)).count();

    println!(
        "bundle: {} docs, {} queries ({} judged), dim={}",
        docs.len(),
        qs.len(),
        n_judged,
        dim
    );
    println!(
        "\n{:<9} {:>8} {:>10} {:>11}  notes",
        "profile", "B/token", "NDCG@10", "reconCos"
    );
    println!("{}", "-".repeat(60));

    // Lossless ceiling: rank the original (un-quantized) embeddings.
    let (flat_float, dindex) = flatten(&docs, dim);
    let float_ndcg = eval_ndcg(&flat_float, &dindex, &qs, &query_ids, &corpus_ids, &qrels);
    println!(
        "{:<9} {:>7}B {:>10.4} {:>11}  lossless ceiling",
        "float",
        dim * 4,
        float_ndcg,
        "1.0000"
    );

    let profiles: Vec<(&str, IndexConfig, usize)> = vec![
        ("4-bit", scalar_cfg(4), dim / 2),
        ("2-bit", scalar_cfg(2), dim / 4),
        ("tern-mass", ternary_cfg(), dim.div_ceil(5)),
        ("tern@.50", ternary_tau_cfg(0.50), dim.div_ceil(5)),
        ("tern@.65*", ternary_tau_cfg(0.65), dim.div_ceil(5)), // * = shipped default
        ("tern@.80", ternary_tau_cfg(0.80), dim.div_ceil(5)),
        ("1-bit", scalar_cfg(1), dim / 8),
    ];

    let ids: Vec<i64> = (0..docs.len() as i64).collect();
    for (label, cfg, bytes) in profiles {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        MmapIndex::create_with_kmeans(&docs, path, &cfg).unwrap();
        let index = MmapIndex::load(path).unwrap();
        let recon = index.reconstruct(&ids).unwrap();
        let cos = mean_recon_cosine(&recon, &docs);
        // `reconstruct` preserves per-doc token counts, so `dindex` is unchanged.
        let (flat_r, _) = flatten(&recon, dim);
        let ndcg = eval_ndcg(&flat_r, &dindex, &qs, &query_ids, &corpus_ids, &qrels);
        let vs_float = ndcg - float_ndcg;
        println!(
            "{:<9} {:>7}B {:>10.4} {:>11.4}  {:+.4} vs float",
            label, bytes, ndcg, cos, vs_float
        );
    }
}
