//! Isolate the *quantization* effect on NDCG@10 with no index in the loop.
//!
//! Computes exhaustive MaxSim (every query against every document) under four
//! precision pairings, mirroring mixedbread's asymmetric-quantization table:
//! float x float, int8 x int8, int8 x binary, binary x binary. Because scoring
//! is exhaustive, the numbers reflect the quantizer alone — not IVF/ANN recall.
//!
//! Usage: `cargo run --release --example quant_schemes -- <bundle_dir>`

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use ndarray::{s, Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::binary::{binarize, quantize_query_int8, signs_pm1};
use next_plaid::maxsim::maxsim_score;
use rayon::prelude::*;

type Qrels = HashMap<String, HashMap<String, i64>>;

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

fn read_f32(p: &Path) -> Array2<f32> {
    Array2::read_npy(File::open(p).unwrap()).unwrap()
}
fn read_i64(p: &Path) -> Array1<i64> {
    Array1::read_npy(File::open(p).unwrap()).unwrap()
}
fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> T {
    serde_json::from_reader(File::open(p).unwrap()).unwrap()
}

fn ndcg_at_k(order: &[usize], corpus_ids: &[String], rels: &HashMap<String, i64>, k: usize) -> f64 {
    let gain = |r: i64| 2f64.powi(r as i32) - 1.0;
    let discount = |i: usize| 1.0 / ((i + 2) as f64).log2();
    let dcg: f64 = order
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &d)| gain(*rels.get(&corpus_ids[d]).unwrap_or(&0)) * discount(i))
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

/// Mean NDCG@10 for one (query docs) precision pairing, scoring exhaustively.
fn eval_scheme(
    queries: &[Array2<f32>],
    docs: &[Array2<f32>],
    corpus_ids: &[String],
    query_ids: &[String],
    qrels: &Qrels,
) -> f64 {
    let (sum, n) = queries
        .par_iter()
        .zip(query_ids)
        .filter_map(|(q, qid)| qrels.get(qid).map(|rels| (q, rels)))
        .map(|(q, rels)| {
            let mut scored: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| (i, maxsim_score(&q.view(), &d.view())))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let order: Vec<usize> = scored.iter().take(10).map(|(i, _)| *i).collect();
            (ndcg_at_k(&order, corpus_ids, rels, 10), 1usize)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    sum / n as f64
}

/// Map each token matrix through a per-token transform (int8 round-trip or ±1 signs).
fn transform(
    items: &[Array2<f32>],
    f: impl Fn(&Array2<f32>) -> Array2<f32> + Sync + Send,
) -> Vec<Array2<f32>> {
    items.par_iter().map(f).collect()
}

fn to_int8(m: &Array2<f32>) -> Array2<f32> {
    quantize_query_int8(&m.view())
}
fn to_signs(m: &Array2<f32>) -> Array2<f32> {
    signs_pm1(&binarize(&m.view()).view(), m.ncols())
}

fn main() {
    let bundle = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: quant_schemes <bundle_dir>"),
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
    println!(
        "docs={} queries={} dim={dim} (exhaustive MaxSim)\n",
        docs.len(),
        queries.len()
    );

    let q_i8 = transform(&queries, to_int8);
    let q_bin = transform(&queries, to_signs);
    let d_i8 = transform(&docs, to_int8);
    let d_bin = transform(&docs, to_signs);

    let ev =
        |q: &[Array2<f32>], d: &[Array2<f32>]| eval_scheme(q, d, &corpus_ids, &query_ids, &qrels);
    let float = ev(&queries, &docs);
    let i8i8 = ev(&q_i8, &d_i8);
    let i8bin = ev(&q_i8, &d_bin);
    let binbin = ev(&q_bin, &d_bin);

    let f32b = dim * 4;
    let row = |name: &str, ndcg: f64, dbytes: usize| {
        println!(
            "{name:<20} {ndcg:>8.4} {:>8.1}% {dbytes:>10} {:>8}x",
            100.0 * ndcg / float,
            f32b / dbytes
        );
    };
    println!(
        "{:<20} {:>8} {:>9} {:>10} {:>9}",
        "scheme", "NDCG@10", "vs float", "doc B/tok", "vs f32"
    );
    row("float x float", float, dim * 4);
    row("int8 x int8", i8i8, dim);
    row("int8 x binary", i8bin, dim / 8);
    row("binary x binary", binbin, dim / 8);
}
