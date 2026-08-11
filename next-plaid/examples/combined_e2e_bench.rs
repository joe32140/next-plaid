//! Cross-version end-to-end search driver for the final PR #169 + #170 stack.
//!
//! The same source is compiled against the common PR base and the combined
//! tree. It loads one prebuilt mmap index, replays deterministic queries with
//! real corpus query lengths, warms the process once, and reports two timed
//! passes through the public `search_one_mmap` API.
//!
//! usage: combined_e2e_bench <index_dir> <query_lens.npy> [params_json]

use std::fs::File;

use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::search::search_one_mmap;
use next_plaid::{MmapIndex, SearchParameters};

fn lcg_unit_rows(seed: &mut u64, rows: usize, dim: usize) -> Array2<f32> {
    let mut query = Array2::<f32>::zeros((rows, dim));
    for mut row in query.rows_mut() {
        for value in row.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *value = ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let norm = row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        row.mapv_inplace(|value| value / norm);
    }
    query
}

fn main() {
    let mut args = std::env::args().skip(1);
    let index_dir = args
        .next()
        .expect("usage: combined_e2e_bench <index_dir> <query_lens.npy> [params_json]");
    let lens_path = args.next().expect("need query_lens.npy");
    let params_json = args.next().unwrap_or_else(|| "{}".to_string());

    // Overlay branch-specific fields onto defaults. Unknown fields are
    // ignored by the base tree, so this source remains cross-version.
    let mut base = serde_json::to_value(SearchParameters::default()).unwrap();
    let overlay: serde_json::Value = serde_json::from_str(&params_json).expect("params JSON");
    if let (Some(base), Some(overlay)) = (base.as_object_mut(), overlay.as_object()) {
        for (key, value) in overlay {
            base.insert(key.clone(), value.clone());
        }
    }
    let params: SearchParameters = serde_json::from_value(base).expect("search parameters");
    let index = MmapIndex::load(&index_dir).expect("load index");

    let query_lengths =
        Array1::<i64>::read_npy(File::open(&lens_path).expect("open query lengths"))
            .expect("read query lengths");
    assert!(!query_lengths.is_empty(), "query-length manifest is empty");

    let dim = index.embedding_dim();
    let mut seed = 0x0DD5EED5_u64;
    let queries: Vec<Array2<f32>> = query_lengths
        .iter()
        .take(50)
        .map(|&rows| lcg_unit_rows(&mut seed, rows as usize, dim))
        .collect();

    // Pass zero warms mmap pages and lazy state. Passes one and two are timed.
    let mut times_ms = Vec::with_capacity(queries.len() * 2);
    let mut digests = Vec::with_capacity(queries.len());
    for pass in 0..3 {
        for query in &queries {
            let started = std::time::Instant::now();
            let result = search_one_mmap(&index, query, &params, None).expect("search");
            std::hint::black_box(result.passage_ids.len());
            if pass > 0 {
                times_ms.push(started.elapsed().as_secs_f64() * 1e3);
            } else {
                let mut hash = 1469598103934665603_u64;
                for &doc_id in &result.passage_ids {
                    hash ^= doc_id as u64;
                    hash = hash.wrapping_mul(1099511628211);
                }
                digests.push(hash);
            }
        }
    }

    let digest_text: Vec<String> = digests.iter().map(|hash| format!("{hash:016x}")).collect();
    println!(
        "RESULTS_DIGEST index={index_dir} params={params_json} q={}",
        digest_text.join(",")
    );

    times_ms.sort_by(|a, b| a.total_cmp(b));
    let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let p50_ms = times_ms[times_ms.len() / 2];
    let p95_ms = times_ms[((times_ms.len() as f64 * 0.95) as usize).min(times_ms.len() - 1)];
    println!(
        "E2EBENCH index={index_dir} params={params_json} n={} mean_ms={mean_ms:.3} \
         p50_ms={p50_ms:.3} p95_ms={p95_ms:.3}",
        times_ms.len()
    );
}
