//! Minimal cross-version e2e bench: identical source compiles against both
//! upstream main and the research branch. Loads a prebuilt mmap index, replays
//! deterministic LCG queries shaped by a real query-length manifest, and times
//! `search_one_mmap` wall per query. SearchParameters come from a JSON arg so
//! branch-only fields (e.g. residual_asym) are honored where they exist and
//! ignored (serde default: unknown fields skipped) where they don't.
//!
//! usage: e2e_bench <index_dir> <query_lens.npy> [params_json]

use std::fs::File;

use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::search::search_one_mmap;
use next_plaid::{MmapIndex, SearchParameters};

fn lcg_unit_rows(s: &mut u64, n: usize, dim: usize) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((n, dim));
    for mut row in a.rows_mut() {
        for v in row.iter_mut() {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }
    a
}

fn main() {
    let mut args = std::env::args().skip(1);
    let index_dir = args.next().expect("usage: e2e_bench <index_dir> <query_lens.npy> [params_json]");
    let lens_path = args.next().expect("need query_lens.npy");
    let params_json = args.next().unwrap_or_else(|| "{}".to_string());

    // Overlay the arg JSON onto serialized defaults: not every field carries
    // a serde default, and branch-only keys must pass through harmlessly.
    let mut base = serde_json::to_value(SearchParameters::default()).unwrap();
    let overlay: serde_json::Value = serde_json::from_str(&params_json).expect("params json");
    if let (Some(b), Some(o)) = (base.as_object_mut(), overlay.as_object()) {
        for (k, v) in o {
            b.insert(k.clone(), v.clone());
        }
    }
    let params: SearchParameters = serde_json::from_value(base).expect("params value");
    let index = MmapIndex::load(&index_dir).expect("load index");
    let dim = 128usize;

    let qlens = Array1::<i64>::read_npy(File::open(&lens_path).unwrap()).unwrap();
    let mut seed = 0x0DD5EED5_u64; // same seed as the shape-replay harness
    let queries: Vec<Array2<f32>> = qlens
        .iter()
        .take(50)
        .map(|&l| lcg_unit_rows(&mut seed, l as usize, dim))
        .collect();

    // Pass 0 warms the page cache and lazy caches; passes 1-2 are timed.
    // Pass 0 also digests each query's ranked ids (FNV-1a) so two builds can
    // be compared for result-list identity on the same queries.
    let mut times_ms: Vec<f64> = Vec::new();
    let mut digests: Vec<u64> = Vec::new();
    for pass in 0..3 {
        for q in &queries {
            let t = std::time::Instant::now();
            let r = search_one_mmap(&index, q, &params, None).expect("search");
            std::hint::black_box(r.passage_ids.len());
            if pass > 0 {
                times_ms.push(t.elapsed().as_secs_f64() * 1e3);
            } else {
                let mut h: u64 = 1469598103934665603;
                for &id in &r.passage_ids {
                    h ^= id as u64;
                    h = h.wrapping_mul(1099511628211);
                }
                digests.push(h);
            }
        }
    }
    let hex: Vec<String> = digests.iter().map(|h| format!("{h:016x}")).collect();
    println!("RESULTS_DIGEST index={index_dir} params={params_json} q={}", hex.join(","));
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let p50 = times_ms[times_ms.len() / 2];
    let p95 = times_ms[(times_ms.len() as f64 * 0.95) as usize];
    println!(
        "E2EBENCH index={index_dir} params={params_json} n={} mean_ms={mean:.3} p50_ms={p50:.3} p95_ms={p95:.3}",
        times_ms.len(),
    );
}
