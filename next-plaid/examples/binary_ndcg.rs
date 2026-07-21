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
///   wide-ANN  = probe 256 IVF cells, no centroid pruning -> a WIDE approximate
///               sweep. This is NOT exhaustive: search() caps exact rescoring
///               at n_full_scores/4 and only probes n_ivf_probe cells, so some
///               documents are never exact-scored. The true quantizer ceiling
///               is the independent numpy brute force (verify_maxsim.py), which
///               scores every query against every document with no ANN.
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

/// Total bytes under a directory (index footprint on disk).
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            total += if p.is_dir() {
                dir_size(&p)
            } else {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
            };
        }
    }
    total
}

/// One quantization config, profiled end to end on a single index build.
struct Profile {
    name: &'static str,
    build_s: f64,
    index_bytes: u64,
    bytes_per_token: usize,
    wide_ndcg: f64,
    dep_ndcg: f64,
    /// Deployed-regime per-query search latency (ms): mean, p50, p95.
    lat_mean: f64,
    lat_p50: f64,
    lat_p95: f64,
    /// Deployed per-query NDCG@10, in judged-query order (for paired bootstrap).
    dep_per_query: Vec<f64>,
    /// Same index re-scored with `residual_asym = true` (NaN/empty for binary).
    asym_ndcg: f64,
    asym_lat_mean: f64,
    asym_lat_p50: f64,
    asym_lat_p95: f64,
    asym_per_query: Vec<f64>,
}

/// NDCG over judged queries; times each search call when `lat` is given.
#[allow(clippy::too_many_arguments)]
fn run_queries(
    index: &MmapIndex,
    params: &SearchParameters,
    corpus_ids: &[String],
    queries: &[Array2<f32>],
    query_ids: &[String],
    qrels: &Qrels,
    mut lat: Option<&mut Vec<f64>>,
) -> (f64, Vec<f64>) {
    let mut per_q = Vec::new();
    for (q, qid) in queries.iter().zip(query_ids) {
        let Some(rels) = qrels.get(qid) else { continue };
        let t = std::time::Instant::now();
        let result = index.search(q, params, None).unwrap();
        if let Some(v) = lat.as_deref_mut() {
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ranked: Vec<String> = result
            .passage_ids
            .iter()
            .map(|&pid| corpus_ids[pid as usize].clone())
            .collect();
        per_q.push(ndcg_at_k(&ranked, rels, 10));
    }
    (per_q.iter().sum::<f64>() / per_q.len() as f64, per_q)
}

/// Build the index once (timed), measure its disk footprint, then run the
/// wide-ANN and deployed regimes on that same index, timing deployed searches.
#[allow(clippy::too_many_arguments)]
fn profile(
    name: &'static str,
    config: &IndexConfig,
    docs: &[Array2<f32>],
    corpus_ids: &[String],
    queries: &[Array2<f32>],
    query_ids: &[String],
    qrels: &Qrels,
    deployed_only: bool,
) -> Profile {
    let dir = std::env::temp_dir().join(format!("np_ndcg_{name}"));
    let _ = std::fs::remove_dir_all(&dir);

    let t = std::time::Instant::now();
    let index = MmapIndex::create_with_kmeans(docs, dir.to_str().unwrap(), config).unwrap();
    let build_s = t.elapsed().as_secs_f64();
    let index_bytes = dir_size(&dir);
    let bytes_per_token = index.mmap_residuals.ncols();

    let wide_ndcg = if deployed_only {
        f64::NAN
    } else {
        run_queries(
            &index,
            &regime_params(false, docs.len()),
            corpus_ids,
            queries,
            query_ids,
            qrels,
            None,
        )
        .0
    };

    // Warm the deployed path once (lazy merges, page cache), then measure.
    let dep_params = regime_params(true, docs.len());
    let _ = index.search(&queries[0], &dep_params, None);
    let mut lat = Vec::new();
    let (dep_ndcg, dep_per_query) = run_queries(
        &index,
        &dep_params,
        corpus_ids,
        queries,
        query_ids,
        qrels,
        Some(&mut lat),
    );
    lat.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
    let lat_mean = lat.iter().sum::<f64>() / lat.len() as f64;

    // A/B on the SAME index: asymmetric int8×LUT residual scoring (skipped
    // for binary indexes, where Stage-2 is already asymmetric).
    let (asym_ndcg, asym_per_query, mut asym_lat) = if config.binary {
        (f64::NAN, Vec::new(), Vec::new())
    } else {
        let asym_params = SearchParameters {
            residual_asym: true,
            ..dep_params.clone()
        };
        let _ = index.search(&queries[0], &asym_params, None);
        let mut alat = Vec::new();
        let (n, pq) = run_queries(
            &index,
            &asym_params,
            corpus_ids,
            queries,
            query_ids,
            qrels,
            Some(&mut alat),
        );
        (n, pq, alat)
    };
    asym_lat.sort_by(|a, b| a.total_cmp(b));
    let apct = |p: f64| {
        if asym_lat.is_empty() {
            f64::NAN
        } else {
            asym_lat[((asym_lat.len() - 1) as f64 * p) as usize]
        }
    };
    let asym_lat_mean = if asym_lat.is_empty() {
        f64::NAN
    } else {
        asym_lat.iter().sum::<f64>() / asym_lat.len() as f64
    };

    let _ = std::fs::remove_dir_all(&dir);
    Profile {
        name,
        build_s,
        index_bytes,
        bytes_per_token,
        wide_ndcg,
        dep_ndcg,
        lat_mean,
        lat_p50: pct(0.5),
        lat_p95: pct(0.95),
        dep_per_query,
        asym_ndcg,
        asym_lat_mean,
        asym_lat_p50: apct(0.5),
        asym_lat_p95: apct(0.95),
        asym_per_query,
    }
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

    let mk = |nbits: usize, binary: bool| IndexConfig {
        nbits,
        binary,
        seed: Some(42),
        ..Default::default()
    };

    // The wide-ANN pass is a coarse recall check, not the ceiling (see
    // regime_params); the numpy brute force is the true ceiling. It is slow on
    // big corpora, so NDCG_DEPLOYED_ONLY=1 skips it and reports deployed only.
    let deployed_only = std::env::var("NDCG_DEPLOYED_ONLY").is_ok();

    let configs: Vec<(&'static str, IndexConfig)> = vec![
        ("residual-nbits4", mk(4, false)),
        ("residual-nbits2", mk(2, false)),
        ("residual-nbits1", mk(1, false)),
        ("binary-int8x1bit", mk(4, true)),
    ];

    let mut rows = Vec::new();
    for (name, config) in &configs {
        eprintln!("profiling {name} ...");
        rows.push(profile(
            name,
            config,
            &docs,
            &corpus_ids,
            &queries,
            &query_ids,
            &qrels,
            deployed_only,
        ));
    }

    let f32_bytes = dim * 4;
    let total_tokens: usize = docs.iter().map(|d| d.nrows()).sum();
    println!(
        "pipeline profile ({} docs, {} doc tokens, {} judged queries, dim={dim}):\n",
        docs.len(),
        total_tokens,
        queries.len()
    );
    println!(
        "{:<18} {:>9} {:>11} {:>9} {:>7} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "scheme",
        "build(s)",
        "index(MB)",
        "B/token",
        "vs f32",
        "wide-ANN",
        "dep NDCG",
        "mean ms",
        "p50 ms",
        "p95 ms"
    );
    for r in &rows {
        println!(
            "{:<18} {:>9.2} {:>11.2} {:>9} {:>6}x {:>10.4} {:>10.4} {:>8.2} {:>8.2} {:>8.2}",
            r.name,
            r.build_s,
            r.index_bytes as f64 / 1e6,
            r.bytes_per_token,
            f32_bytes / r.bytes_per_token,
            r.wide_ndcg,
            r.dep_ndcg,
            r.lat_mean,
            r.lat_p50,
            r.lat_p95
        );
    }

    // A/B lines: float vs asymmetric int8×LUT scoring on the same index.
    for r in rows.iter().filter(|r| r.asym_ndcg.is_finite()) {
        println!(
            "{:<18} asym-LUT: dep NDCG {:.4} (float {:.4}, Δ{:+.4})  mean {:.2} ms vs {:.2} ms ({:.2}x)",
            r.name,
            r.asym_ndcg,
            r.dep_ndcg,
            r.asym_ndcg - r.dep_ndcg,
            r.asym_lat_mean,
            r.lat_mean,
            r.lat_mean / r.asym_lat_mean
        );
    }

    let by_name = |n: &str| rows.iter().find(|r| r.name == n).unwrap();
    let res4 = by_name("residual-nbits4");
    let bin = by_name("binary-int8x1bit");
    println!(
        "\nbinary retains {:.1}% (deployed) of residual-nbits4 NDCG@10 at {:.1}x less doc storage",
        100.0 * bin.dep_ndcg / res4.dep_ndcg,
        res4.bytes_per_token as f64 / bin.bytes_per_token as f64
    );
    if !deployed_only {
        println!(
            "binary retains {:.1}% (wide-ANN) of residual4; deployed keeps {:.1}% (residual4) / {:.1}% (binary) of the wide-ANN sweep (true ceiling = numpy brute force)",
            100.0 * bin.wide_ndcg / res4.wide_ndcg,
            100.0 * res4.dep_ndcg / res4.wide_ndcg,
            100.0 * bin.dep_ndcg / bin.wide_ndcg
        );
    }

    // Machine-readable line (NDCG_JSON=1).
    if std::env::var("NDCG_JSON").is_ok() {
        let json_rows: Vec<String> = rows
            .iter()
            .map(|r| {
                let asym = if r.asym_ndcg.is_finite() {
                    format!(
                        "{{\"ndcg\":{:.6},\"lat_mean_ms\":{:.3},\"lat_p50_ms\":{:.3},\"lat_p95_ms\":{:.3},\"per_query\":[{}]}}",
                        r.asym_ndcg, r.asym_lat_mean, r.asym_lat_p50, r.asym_lat_p95,
                        r.asym_per_query.iter().map(|v| format!("{v:.5}"))
                            .collect::<Vec<_>>().join(",")
                    )
                } else {
                    "null".to_string()
                };
                format!(
                    "{{\"name\":\"{}\",\"build_s\":{:.3},\"index_bytes\":{},\"bytes_per_token\":{},\"wide_ndcg\":{:.6},\"dep_ndcg\":{:.6},\"lat_mean_ms\":{:.3},\"lat_p50_ms\":{:.3},\"lat_p95_ms\":{:.3},\"per_query\":[{}],\"asym\":{}}}",
                    r.name, r.build_s, r.index_bytes, r.bytes_per_token,
                    r.wide_ndcg, r.dep_ndcg, r.lat_mean, r.lat_p50, r.lat_p95,
                    r.dep_per_query.iter().map(|v| format!("{v:.5}"))
                        .collect::<Vec<_>>().join(","),
                    asym
                )
            })
            .collect();
        println!(
            "NDCG_JSON {{\"docs\":{},\"queries\":{},\"dim\":{},\"schemes\":[{}]}}",
            docs.len(),
            queries.len(),
            dim,
            json_rows.join(",")
        );
    }
}
