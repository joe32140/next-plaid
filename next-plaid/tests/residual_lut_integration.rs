//! End-to-end tests for asymmetric int8×LUT residual scoring: same index,
//! `residual_asym` toggled per search — retrieval must agree with the float
//! path. Both paths apply the identical per-token renormalize (the asym path
//! via cached inverse norms — load-bearing, not optional), so the only
//! remaining difference is int8 quantization of the residual term.

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use next_plaid::index::MmapIndex;
use next_plaid::{IndexConfig, SearchParameters};
use rand::rngs::StdRng;
use tempfile::TempDir;

fn random_docs(num_docs: usize, tokens: usize, dim: usize) -> Vec<Array2<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..num_docs)
        .map(|_| {
            let mut emb: Array2<f32> =
                Array2::random_using((tokens, dim), StandardNormal, &mut rng);
            for mut row in emb.axis_iter_mut(Axis(0)) {
                let norm = row.dot(&row).sqrt().max(1e-12);
                row /= norm;
            }
            emb
        })
        .collect()
}

fn params(asym: bool) -> SearchParameters {
    SearchParameters {
        top_k: 5,
        n_ivf_probe: 16,
        residual_asym: asym,
        ..Default::default()
    }
}

/// Every document must retrieve itself at rank 1 through the LUT path, for
/// every supported nbits.
#[test]
fn lut_path_retrieves_the_query_document() {
    for &nbits in &[1usize, 2, 4] {
        let docs = random_docs(50, 8, 64);
        let dir = TempDir::new().unwrap();
        let config = IndexConfig {
            nbits,
            batch_size: 64,
            seed: Some(42),
            ..Default::default()
        };
        let index =
            MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();
        for (i, doc) in docs.iter().enumerate() {
            let res = index.search(doc, &params(true), None).unwrap();
            assert_eq!(
                res.passage_ids[0], i as i64,
                "nbits={nbits}: doc {i} did not self-retrieve via LUT path"
            );
        }
    }
}

/// The LUT path and the float path must produce near-identical rankings on
/// the same index: identical top-1 and high top-5 overlap for every query.
#[test]
fn lut_path_agrees_with_float_path() {
    let docs = random_docs(80, 8, 64);
    let dir = TempDir::new().unwrap();
    let config = IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        ..Default::default()
    };
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();

    let mut overlap_total = 0usize;
    for doc in docs.iter().take(30) {
        let float = index.search(doc, &params(false), None).unwrap();
        let lut = index.search(doc, &params(true), None).unwrap();
        assert_eq!(
            float.passage_ids[0], lut.passage_ids[0],
            "top-1 disagreement between float and LUT paths"
        );
        overlap_total += lut
            .passage_ids
            .iter()
            .filter(|id| float.passage_ids.contains(id))
            .count();
    }
    // ≥ 4 of 5 average overlap: the paths differ only by int8 rounding of
    // the residual term.
    assert!(
        overlap_total >= 30 * 4,
        "top-5 overlap too low: {overlap_total}/150"
    );
}

/// Scores from the LUT path must approximate the float path's scores: the
/// centroid term is shared exactly and the renormalize is applied
/// identically, so differences come only from int8 residual rounding.
#[test]
fn lut_scores_track_float_scores() {
    let docs = random_docs(60, 8, 64);
    let dir = TempDir::new().unwrap();
    let config = IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        ..Default::default()
    };
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();

    for doc in docs.iter().take(10) {
        let float = index.search(doc, &params(false), None).unwrap();
        let lut = index.search(doc, &params(true), None).unwrap();
        // Compare the top-1 scores (same doc per the agreement test). Docs are
        // 8 tokens of unit vectors → MaxSim ∈ [-8, 8]; the two paths should
        // agree within a few percent of that range.
        let diff = (float.scores[0] - lut.scores[0]).abs();
        assert!(
            diff < 0.4,
            "top-1 score diverged: float {} vs lut {}",
            float.scores[0],
            lut.scores[0]
        );
    }
}

/// Dims that are not a multiple of 8 build no SIMD query planes and score
/// on the scalar kernel through the same public search path. (The codec
/// itself requires `dim·nbits % 8 == 0`, so nbits=1 cannot reach this
/// shape — 2 and 4 can.)
#[test]
fn lut_path_handles_non_byte_aligned_dims() {
    for &nbits in &[2usize, 4] {
        let docs = random_docs(30, 8, 44);
        let dir = TempDir::new().unwrap();
        let config = IndexConfig {
            nbits,
            batch_size: 64,
            seed: Some(42),
            ..Default::default()
        };
        let index =
            MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();
        for (i, doc) in docs.iter().enumerate().step_by(5) {
            let res = index.search(doc, &params(true), None).unwrap();
            assert_eq!(
                res.passage_ids[0], i as i64,
                "nbits={nbits} dim=44: doc {i} did not self-retrieve via scalar LUT path"
            );
        }
    }
}

/// Dims above the fused path's MAX_DIM fall back to the float path:
/// `residual_asym` on such an index must be a no-op, not an error — same
/// ranking, same scores.
#[test]
fn oversize_dim_falls_back_to_float_path() {
    let docs = random_docs(20, 6, 272);
    let dir = TempDir::new().unwrap();
    let config = IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        ..Default::default()
    };
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();
    for (i, doc) in docs.iter().enumerate().step_by(4) {
        let asym = index.search(doc, &params(true), None).unwrap();
        let float = index.search(doc, &params(false), None).unwrap();
        assert_eq!(
            asym.passage_ids[0], i as i64,
            "dim=272: doc {i} did not self-retrieve"
        );
        assert_eq!(
            asym.passage_ids, float.passage_ids,
            "dim=272: rankings diverged"
        );
        assert_eq!(asym.scores, float.scores, "dim=272: fallback scores differ");
    }
}

/// The batched-centroid path (num_centroids > centroid_batch_size) must
/// honor `residual_asym` and agree with the dense path. This is the scale
/// cliff regression test: before the fix, any index with more than
/// `centroid_batch_size` centroids (~67M tokens at the default 100k)
/// silently reverted asym scoring to float decompress+GEMM. Forcing a tiny
/// batch size exercises the batched path on a small index. With every doc
/// in the exact-scored shortlist, rankings must match and scores may differ
/// only by the cdot computation route (full GEMM vs per-centroid dots).
#[test]
fn batched_path_asym_matches_dense_path() {
    for &nbits in &[1usize, 2, 4] {
        let docs = random_docs(80, 8, 64);
        let dir = TempDir::new().unwrap();
        let config = IndexConfig {
            nbits,
            batch_size: 64,
            seed: Some(42),
            ..Default::default()
        };
        let index =
            MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();
        let dense = params(true);
        let batched = SearchParameters {
            centroid_batch_size: 8,
            ..params(true)
        };
        for (i, doc) in docs.iter().enumerate().step_by(9) {
            let rd = index.search(doc, &dense, None).unwrap();
            let rb = index.search(doc, &batched, None).unwrap();
            assert_eq!(
                rd.passage_ids, rb.passage_ids,
                "nbits={nbits} query {i}: batched-asym ranking diverged from dense-asym"
            );
            for (a, b) in rd.scores.iter().zip(&rb.scores) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "nbits={nbits} query {i}: batched-asym score {b} vs dense-asym {a}"
                );
            }
        }
    }
}

/// The parallel batch path must not deadlock on a **cold** index.
///
/// `search_many_mmap(parallel = true)` fans queries across rayon workers, and
/// with `residual_asym` on, every one of them races into the lazily-built
/// inverse-norm cache at once. An initializer that blocks the calling worker
/// in any way that leaves the outer pool free to schedule onto it — a
/// global-pool `par_iter`, or `ThreadPool::install`, which keeps the caller
/// available for its own pool's stealing while it waits — self-deadlocks:
/// the worker steals a sibling query, that query re-enters the initializer on
/// the same thread, and nothing ever completes. This is the first-request
/// shape of a batch-serving deployment, so it must hold from cold.
///
/// The watchdog exists because the failure mode is a hang, not a panic: it
/// turns a stalled CI job into a fast, unambiguous failure.
#[test]
fn parallel_batch_on_cold_index_does_not_deadlock() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let docs = random_docs(400, 32, 64);
    let dir = TempDir::new().unwrap();
    let config = IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        ..Default::default()
    };
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config).unwrap();

    let finished = Arc::new(AtomicBool::new(false));
    {
        let finished = Arc::clone(&finished);
        std::thread::spawn(move || {
            for _ in 0..600 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if finished.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!(
                "DEADLOCK: search_many_mmap(parallel=true) with residual_asym did not \
                 finish within 60s on a cold index"
            );
            std::process::exit(101);
        });
    }

    // Cold: no search has run, so the inverse-norm cache is unbuilt and every
    // worker will reach for it simultaneously.
    let queries: Vec<Array2<f32>> = docs.iter().take(128).cloned().collect();
    let results =
        next_plaid::search::search_many_mmap(&index, &queries, &params(true), true, None).unwrap();
    finished.store(true, Ordering::SeqCst);

    assert_eq!(results.len(), queries.len());
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.passage_ids[0], i as i64,
            "query {i} did not self-retrieve"
        );
    }
}
