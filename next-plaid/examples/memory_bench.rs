//! Cross-version peak-RSS and latency driver for search-path comparisons.
//!
//! The driver intentionally keeps measurement policy out of the process:
//! CI launches it under the platform's `/usr/bin/time` so every variant is a
//! fresh process and reports OS-observed peak RSS. One sequential query warms
//! mmap pages and lazy index caches, then the requested number of queries run
//! concurrently through the public batch API. The warm-up is excluded from
//! latency, while the process peak still includes every retained allocation.
//!
//! usage: memory_bench <index_dir> <query_lens.npy> [params_json] [batch] [query_rows]

use std::fs::File;

use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::search::{search_many_mmap, search_one_mmap};
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
    let index_dir = args.next().expect(
        "usage: memory_bench <index_dir> <query_lens.npy> [params_json] [batch] [query_rows]",
    );
    let lens_path = args.next().expect("need query_lens.npy");
    let params_json = args.next().unwrap_or_else(|| "{}".to_string());
    let batch = args
        .next()
        .map(|value| value.parse::<usize>().expect("batch must be an integer"))
        .unwrap_or(1);
    assert!(batch > 0, "batch must be positive");
    let fixed_query_rows = args.next().map(|value| {
        value
            .parse::<usize>()
            .expect("query_rows must be an integer")
    });
    if let Some(rows) = fixed_query_rows {
        assert!(rows > 0, "query_rows must be positive");
    }

    // Overlay branch-specific fields onto defaults. Serde ignores unknown
    // fields, so this exact source also compiles and runs against v1.6.5.
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
    let queries: Vec<Array2<f32>> = (0..batch)
        .map(|query_id| {
            let rows =
                fixed_query_rows.unwrap_or(query_lengths[query_id % query_lengths.len()] as usize);
            lcg_unit_rows(&mut seed, rows, dim)
        })
        .collect();

    // Initialize mmap working pages plus either PR's lazy per-token cache
    // before the parallel section. The process peak still includes the cache;
    // warming only prevents initialization time from changing overlap.
    let warm = search_one_mmap(&index, &queries[0], &params, None).expect("warm search");
    std::hint::black_box(warm.passage_ids.len());

    // Repeat to make the concurrent allocation window long enough for OS peak
    // accounting and to average scheduler noise without retaining result
    // vectors between repetitions.
    const REPEATS: usize = 5;
    let started = std::time::Instant::now();
    for _ in 0..REPEATS {
        let results =
            search_many_mmap(&index, &queries, &params, true, None).expect("parallel batch search");
        std::hint::black_box(
            results
                .iter()
                .map(|result| result.passage_ids.len())
                .sum::<usize>(),
        );
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let batch_mean_ms = elapsed_ms / REPEATS as f64;
    let query_mean_ms = elapsed_ms / (REPEATS * batch) as f64;
    let qps = (REPEATS * batch) as f64 / (elapsed_ms / 1e3);

    let tokens = index.doc_offsets.last().copied().unwrap_or(0);
    let centroids = index.num_partitions();
    let query_rows: usize = queries.iter().map(Array2::nrows).sum();
    let padded_query_rows: usize = queries
        .iter()
        .map(|query| query.nrows().div_ceil(16) * 16)
        .sum();
    let cdot_f32_bytes = centroids * query_rows * std::mem::size_of::<f32>();
    let transpose_f32_bytes = cdot_f32_bytes;
    let transpose_q8_bytes = centroids * padded_query_rows;

    println!(
        "MEMBENCH index={index_dir} params={params_json} batch={batch} tokens={tokens} \
         centroids={centroids} query_rows={query_rows} cdot_f32_bytes={cdot_f32_bytes} \
         transpose_f32_bytes={transpose_f32_bytes} transpose_q8_bytes={transpose_q8_bytes} \
         repeats={REPEATS} elapsed_ms={elapsed_ms:.3} batch_mean_ms={batch_mean_ms:.3} \
         query_mean_ms={query_mean_ms:.3} qps={qps:.3}"
    );
}
