//! End-to-end search latency with and without #169's asymmetric residual LUT,
//! for the 2-bit scalar and ternary (base-3) codecs, on a real ColBERT bundle.
//!
//! For each (codec, rescore-mode) it builds a next-plaid index, warms up, then
//! times `search_batch` over all queries — reporting per-query latency, the
//! real-search NDCG@10 (to confirm asym preserves quality) and the residual
//! store size (bytes/token). This is the "how far can we push latency" view:
//! asym scores over packed codes via a fused int8 LUT instead of decompressing.
//!
//! Run: `cargo run -p next-plaid --release --example search_latency -- <bundle_dir> [reps]`

use ndarray::{s, Array1, Array2};
use ndarray_npy::read_npy;
use next_plaid::index::MmapIndex;
use next_plaid::{IndexConfig, SearchParameters};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

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

fn ndcg_at_k(ranked: &[usize], rels: &HashMap<usize, f32>, k: usize) -> f32 {
    let dcg: f32 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| (2f32.powf(*rels.get(id).unwrap_or(&0.0)) - 1.0) / (i as f32 + 2.0).log2())
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

fn scalar_cfg(nbits: usize) -> IndexConfig {
    IndexConfig {
        nbits,
        seed: Some(42),
        ..Default::default()
    }
}

fn ternary_cfg() -> IndexConfig {
    IndexConfig {
        nbits: 2,
        seed: Some(42),
        ternary: true,
        ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(
        args.get(1)
            .expect("usage: search_latency <bundle_dir> [reps] [n_ivf_probe]"),
    );
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    // Probe depth controls how many centroid cells (and thus candidate docs) reach
    // residual rescoring — the lever that shifts cost from stage-1 into the residual
    // path where #169's asym LUT applies. Sweep it to find the asym crossover.
    let n_ivf_probe: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
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
    let row_of_docid: HashMap<&str, usize> = corpus_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    println!(
        "bundle: {} docs, {} queries, dim={}, reps={}, n_ivf_probe={}\n",
        docs.len(),
        qs.len(),
        dim,
        reps,
        n_ivf_probe
    );
    println!(
        "{:<9} {:>6} {:>8} {:>11} {:>11}  speedup",
        "codec", "B/tok", "rescore", "us/query", "NDCG@10"
    );
    println!("{}", "-".repeat(64));

    for (label, cfg, bytes) in [
        ("2-bit", scalar_cfg(2), dim / 4),
        ("ternary", ternary_cfg(), dim.div_ceil(5)),
    ] {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        MmapIndex::create_with_kmeans(&docs, path, &cfg).unwrap();
        let index = MmapIndex::load(path).unwrap();

        let mut float_us = 0.0f64;
        for (mi, asym) in [false, true].into_iter().enumerate() {
            let params = SearchParameters {
                top_k: 10,
                n_ivf_probe,
                residual_asym: asym,
                ..Default::default()
            };
            // Warm up (mmap faults, thread pool, caches).
            let _ = index.search_batch(&qs, &params, true, None).unwrap();

            let t = Instant::now();
            let mut last = Vec::new();
            for _ in 0..reps {
                last = index.search_batch(&qs, &params, true, None).unwrap();
            }
            let us = t.elapsed().as_micros() as f64 / (reps * qs.len()) as f64;

            let mut ndcg_sum = 0.0f32;
            let mut n = 0u32;
            for (qi, res) in last.iter().enumerate() {
                let Some(rel_map) = qrels.get(&query_ids[qi]) else {
                    continue;
                };
                let rels_row: HashMap<usize, f32> = rel_map
                    .iter()
                    .filter_map(|(d, &r)| row_of_docid.get(d.as_str()).map(|&row| (row, r)))
                    .collect();
                if rels_row.is_empty() {
                    continue;
                }
                let ranked: Vec<usize> = res.passage_ids.iter().map(|&p| p as usize).collect();
                ndcg_sum += ndcg_at_k(&ranked, &rels_row, 10);
                n += 1;
            }
            let ndcg = ndcg_sum / n.max(1) as f32;

            let speedup = if asym {
                format!("{:.2}x", float_us / us)
            } else {
                float_us = us;
                "—".to_string()
            };
            let mode = if asym { "asym-LUT" } else { "float" };
            let b = if mi == 0 {
                format!("{bytes}")
            } else {
                String::new()
            };
            println!(
                "{:<9} {:>6} {:>8} {:>11.1} {:>11.4}  {}",
                if mi == 0 { label } else { "" },
                b,
                mode,
                us,
                ndcg,
                speedup
            );
        }
    }
}
