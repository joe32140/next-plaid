//! MaxSim kernel bench — the stage-2 counterpart of `binary_ndcg`'s
//! end-to-end latency columns, using mixedbread's protocol for direct
//! comparison with their asymmetric-quantization table: one query scored
//! against a fixed candidate list of N docs, median of 9 timed passes after
//! 2 warmups. No IVF, no stage-1 — `exact_score_docs` only, which still
//! includes the per-query prep a real search pays (int8 quantization,
//! fused-LUT build, dense query×centroid matrix).
//!
//! Usage: `maxsim_bench <bundle_dir> [n_docs=1000]`
//!   SYNTH_TOKENS=786  re-chunk the corpus token stream into docs of exactly
//!                     that many tokens (mixedbread's page-image shape) while
//!                     keeping the real token distribution.

use std::fs::File;
use std::path::PathBuf;

use ndarray::{s, Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::index::MmapIndex;
use next_plaid::search::exact_score_docs;
use next_plaid::IndexConfig;

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

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.total_cmp(b));
    times[times.len() / 2]
}

fn main() {
    let bundle = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: maxsim_bench <bundle_dir> [n_docs]"),
    );
    let n_docs: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1000);

    let corpus =
        Array2::<f32>::read_npy(File::open(bundle.join("corpus.npy")).unwrap()).unwrap();
    let lens =
        Array1::<i64>::read_npy(File::open(bundle.join("corpus_lens.npy")).unwrap()).unwrap();
    let queries =
        Array2::<f32>::read_npy(File::open(bundle.join("queries.npy")).unwrap()).unwrap();
    let qlens =
        Array1::<i64>::read_npy(File::open(bundle.join("query_lens.npy")).unwrap()).unwrap();

    // Candidate docs: real per-doc shapes, or fixed-length synthetic chunks of
    // the same token stream (SYNTH_TOKENS=786 = mixedbread's doc shape).
    let docs: Vec<Array2<f32>> = match std::env::var("SYNTH_TOKENS") {
        Ok(t) => {
            let t: usize = t.parse().unwrap();
            assert!(corpus.nrows() >= n_docs * t, "corpus too small for synth shape");
            (0..n_docs)
                .map(|i| corpus.slice(s![i * t..(i + 1) * t, ..]).to_owned())
                .collect()
        }
        Err(_) => unpack(&corpus, &lens).into_iter().take(n_docs).collect(),
    };
    let query = unpack(&queries, &qlens).into_iter().next().unwrap();
    let doc_tokens: usize = docs.iter().map(|d| d.nrows()).sum();
    let dim = docs[0].ncols();
    println!(
        "maxsim_bench: {} docs, {} doc tokens ({:.0}/doc), query {}x{}, median of 9 after 2 warmups",
        docs.len(),
        doc_tokens,
        doc_tokens as f64 / docs.len() as f64,
        query.nrows(),
        dim,
    );

    let ids: Vec<usize> = (0..docs.len()).collect();
    let configs: Vec<(&str, IndexConfig, bool)> = vec![
        // (label, config, residual_asym)
        ("float r4 (decompress+GEMM)", cfg(4, false), false),
        ("asym-LUT r4 (int8xLUT)", cfg(4, false), true),
        ("float r2", cfg(2, false), false),
        ("asym-LUT r2", cfg(2, false), true),
        ("float r1", cfg(1, false), false),
        ("asym-LUT r1", cfg(1, false), true),
        ("binary int8x1bit", cfg(4, true), false),
    ];

    let mut float_ms = f64::NAN;
    // (nbits, binary) -> index cache: float/asym pairs share one build (the
    // A/B is scoring-path-only).
    let mut built: Option<(usize, bool, MmapIndex)> = None;
    for (label, config, asym) in configs {
        let rebuild = !matches!(&built, Some((n, b, _)) if *n == config.nbits && *b == config.binary);
        if rebuild {
            let dir = std::env::temp_dir().join(format!(
                "np_maxsim_bench_{}_{}",
                config.nbits, config.binary
            ));
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("building index nbits={} binary={} ...", config.nbits, config.binary);
            let index =
                MmapIndex::create_with_kmeans(&docs, dir.to_str().unwrap(), &config).unwrap();
            built = Some((config.nbits, config.binary, index));
        }
        let index = &built.as_ref().unwrap().2;

        for _ in 0..2 {
            let _ = exact_score_docs(index, &query, &ids, asym);
        }
        let mut times: Vec<f64> = (0..9)
            .map(|_| {
                let t = std::time::Instant::now();
                let scores = exact_score_docs(index, &query, &ids, asym);
                let el = t.elapsed().as_secs_f64() * 1e3;
                assert_eq!(scores.len(), ids.len());
                el
            })
            .collect();
        let med = median(&mut times);
        if label.starts_with("float r4") {
            float_ms = med;
        }
        println!(
            "{label:<30} {med:8.2} ms /{} docs   {:6.2}x vs float-r4",
            ids.len(),
            float_ms / med,
        );
    }
}

fn cfg(nbits: usize, binary: bool) -> IndexConfig {
    IndexConfig {
        nbits,
        binary,
        seed: Some(42),
        ..Default::default()
    }
}
