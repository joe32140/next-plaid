//! End-to-end search latency across the residual codec ladder.
//!
//! For each codec it builds a next-plaid index, warms up, then times
//! `search_batch` over all queries — reporting per-query latency, the residual
//! store size (bytes/token), NDCG@10 where the bundle has qrels, and the ratio
//! against 2-bit at the same probe depth.
//!
//! 2-bit is the baseline because it is the rung ternary is proposed against:
//! ternary is 19 % smaller, and the question this answers is what that costs at
//! query time. Ternary unpacks one byte into five values through a 256-entry
//! table, where the scalar rungs shift and mask per dimension, so it is not
//! obvious a priori which is faster — measure it.
//!
//! Two input modes:
//!
//! * `<bundle_dir>` — a real ColBERT bundle (BEIR export). Absolute numbers on
//!   real token distributions; needs the ~300–600 MB npy files.
//! * `synth` — a seeded clustered corpus (see `common::synth_bundle`), no data
//!   dependency. This is the mode CI runs, because the numbers here are
//!   *latency*, and latency measured on a contended developer machine is a
//!   measurement of the contention. Shapes are env-tunable: `DOCS` (2000),
//!   `TOKENS` (180), `QUERIES` (64), `QTOKENS` (32), `DIM` (128), `CLUSTERS`
//!   (64), `SPREAD` (0.35), `SEED` (0x5EED).
//!
//! Every codec is timed **in one process**, which is load-bearing: the same
//! benchmark moves +/-30 % between CI runners, so cross-codec deltas collected
//! from separate invocations are noise, not measurements.
//!
//! ```text
//! cargo run -p next-plaid --release --example search_latency -- synth 5 1,8,64
//! cargo run -p next-plaid --release --example search_latency -- data/nfcorpus_colbertv2 10
//! ```
//! Args: `<bundle_dir|synth> [reps] [probes]`, where `probes` is a
//! comma-separated list of `n_ivf_probe` depths swept over one index build.
//! Env: `TERNARY_TAU` (0.65).

mod common;

use common::{env_f32, env_usize, synth_bundle};
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
        nbits: 2, // nominal; the ternary codec supersedes it
        seed: Some(42),
        ternary: true,
        // The dead-zone width is a *codebook* choice — it moves quality, not
        // layout or speed — so the latency rows are tau-independent and this
        // only sets what the NDCG column reports.
        ternary_tau: Some(env_f32("TERNARY_TAU", 0.65)),
        ..Default::default()
    }
}

/// Bytes of residual store per token, per codec.
fn bytes_per_token(label: &str, dim: usize) -> usize {
    match label {
        "1-bit" => dim / 8,
        "2-bit" => dim / 4,
        "4-bit" => dim / 2,
        _ => dim.div_ceil(5),
    }
}

/// A bundle, real or synthetic. `qrels` is empty in synthetic mode.
struct Bundle {
    docs: Vec<Array2<f32>>,
    queries: Vec<Array2<f32>>,
    /// Judgements already mapped onto corpus row indices, per query index.
    rels: Vec<Option<HashMap<usize, f32>>>,
    label: String,
}

fn load_real(dir: &Path) -> Bundle {
    let read_json = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
    let corpus: Array2<f32> = read_npy(dir.join("corpus.npy")).unwrap();
    let corpus_lens: Array1<i64> = read_npy(dir.join("corpus_lens.npy")).unwrap();
    let queries: Array2<f32> = read_npy(dir.join("queries.npy")).unwrap();
    let query_lens: Array1<i64> = read_npy(dir.join("query_lens.npy")).unwrap();
    let corpus_ids: Vec<String> = serde_json::from_str(&read_json("corpus_ids.json")).unwrap();
    let query_ids: Vec<String> = serde_json::from_str(&read_json("query_ids.json")).unwrap();
    let qrels: HashMap<String, HashMap<String, f32>> =
        serde_json::from_str(&read_json("qrels.json")).unwrap();

    let row_of_docid: HashMap<&str, usize> = corpus_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let rels = query_ids
        .iter()
        .map(|qid| {
            let m: HashMap<usize, f32> = qrels
                .get(qid)?
                .iter()
                .filter_map(|(d, &r)| row_of_docid.get(d.as_str()).map(|&row| (row, r)))
                .collect();
            (!m.is_empty()).then_some(m)
        })
        .collect();

    Bundle {
        docs: split_rows(&corpus, corpus_lens.as_slice().unwrap()),
        queries: split_rows(&queries, query_lens.as_slice().unwrap()),
        rels,
        label: dir.display().to_string(),
    }
}

fn load_synth() -> Bundle {
    let (docs_n, toks) = (env_usize("DOCS", 2000), env_usize("TOKENS", 180));
    let (nq, qt) = (env_usize("QUERIES", 64), env_usize("QTOKENS", 32));
    let dim = env_usize("DIM", 128);
    let b = synth_bundle(
        docs_n,
        toks,
        nq,
        qt,
        dim,
        env_usize("CLUSTERS", 64),
        env_f32("SPREAD", 0.35),
        env_usize("SEED", 0x5EED) as u64,
    );
    let rels = vec![None; b.queries.len()];
    Bundle {
        docs: b.docs,
        queries: b.queries,
        rels,
        label: "synthetic (seeded, clustered)".to_string(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg1 = args
        .get(1)
        .expect("usage: search_latency <bundle_dir|synth> [reps] [probes]");
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    // Probe depth controls how many centroid cells (and thus candidate docs) reach
    // residual rescoring — the only place the codec choice can show up, so sweep it
    // rather than reporting one arbitrary point.
    // Comma-separated, because every depth reuses one index: k-means over the whole
    // corpus costs far more than the searches being timed, so rebuilding per depth
    // would spend most of the job on a quantity nobody is measuring.
    let probes: Vec<usize> = args
        .get(3)
        .map(|v| v.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![8]);

    let bundle = if arg1 == "synth" {
        load_synth()
    } else {
        load_real(Path::new(arg1))
    };
    let (docs, qs) = (&bundle.docs, &bundle.queries);
    let dim = docs[0].ncols();
    let judged = bundle.rels.iter().filter(|r| r.is_some()).count();

    println!(
        "bundle: {} ({} docs, {} queries, {} judged, dim={})",
        bundle.label,
        docs.len(),
        qs.len(),
        judged,
        dim
    );
    println!(
        "build: {}-{}, reps={}, probes={:?}\n",
        std::env::consts::ARCH,
        std::env::consts::OS,
        reps,
        probes
    );
    // Collect first, print second: the `vs r=2` ratio needs 2-bit's timing at the
    // same probe depth, and 2-bit is not the first rung in the ladder.
    let mut rows: Vec<(&str, usize, usize, f64, String)> = Vec::new();

    for (label, cfg) in [
        ("4-bit", scalar_cfg(4)),
        ("2-bit", scalar_cfg(2)),
        ("ternary", ternary_cfg()),
        ("1-bit", scalar_cfg(1)),
    ] {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        MmapIndex::create_with_kmeans(docs, path, &cfg).unwrap();
        let index = MmapIndex::load(path).unwrap();

        for &n_ivf_probe in &probes {
            let params = SearchParameters {
                top_k: 10,
                n_ivf_probe,
                ..Default::default()
            };
            // Warm up (mmap faults, thread pool, caches).
            let _ = index.search_batch(qs, &params, true, None).unwrap();

            let t = Instant::now();
            let mut last = Vec::new();
            for _ in 0..reps {
                last = index.search_batch(qs, &params, true, None).unwrap();
            }
            let us = t.elapsed().as_micros() as f64 / (reps * qs.len()) as f64;

            let mut ndcg_sum = 0.0f32;
            let mut n = 0u32;
            for (qi, res) in last.iter().enumerate() {
                let Some(rels_row) = bundle.rels[qi].as_ref() else {
                    continue;
                };
                let ranked: Vec<usize> = res.passage_ids.iter().map(|&p| p as usize).collect();
                ndcg_sum += ndcg_at_k(&ranked, rels_row, 10);
                n += 1;
            }
            let ndcg = if n > 0 {
                format!("{:.4}", ndcg_sum / n as f32)
            } else {
                "—".to_string()
            };

            rows.push((label, bytes_per_token(label, dim), n_ivf_probe, us, ndcg));
        }
    }

    let baseline: HashMap<usize, f64> = rows
        .iter()
        .filter(|(label, ..)| *label == "2-bit")
        .map(|&(_, _, probe, us, _)| (probe, us))
        .collect();

    println!(
        "{:<9} {:>6} {:>6} {:>11} {:>9}  vs r=2",
        "codec", "B/tok", "probe", "us/query", "NDCG@10"
    );
    println!("{}", "-".repeat(60));
    let mut shown = "";
    for (label, bytes, probe, us, ndcg) in &rows {
        let ratio = match baseline.get(probe) {
            Some(&b) if *label != "2-bit" => format!("{:.2}x", b / us),
            _ => "—".to_string(),
        };
        println!(
            "{:<9} {:>6} {:>6} {:>11.1} {:>9}  {}",
            if shown == *label { "" } else { label },
            if shown == *label {
                String::new()
            } else {
                format!("{bytes}")
            },
            probe,
            us,
            ndcg,
            ratio
        );
        shown = label;
    }
}
