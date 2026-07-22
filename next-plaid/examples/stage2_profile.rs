//! Stage-2 phase profile on a real dataset — the pragmatic counterpart of
//! `maxsim_bench`'s fixed-list kernel numbers.
//!
//! For every real query it runs the production Stage-1 (`stage1_shortlist`:
//! dense query×centroid scores, per-token IVF probing, approximate pruning)
//! to get the exact candidate shortlist production would score, then times
//! Stage-2 per scheme from that point to the final MaxSim:
//!
//! * `float rX`  — decompress + GEMM MaxSim, additionally decomposed into a
//!   decompress-only pass and a GEMM-only pass over pre-decompressed docs;
//! * `asym rX`   — fused int8×LUT kernel (prep = int8 quantize + fused-LUT +
//!   planes build, measured via an empty-shortlist call);
//! * `binary`    — fused int8×1-bit kernel.
//!
//! All schemes share Stage-1's query×centroid matrix, as production does.
//! Times are per query: 1 warmup + median of 3, aggregated as mean and p50
//! across queries. Storage note: raw f32 would be 512 B/token; r4/r2/r1
//! pack dim·nbits/8 residual bytes (+8 B code) per token; binary dim/8.
//!
//! Usage: `stage2_profile <bundle_dir|synth> [max_queries]`
//!   `synth` generates a bundle-free corpus for CI (seeded LCG unit-norm
//!   tokens; phase latency depends on shapes, not values): [max_queries]
//!   becomes n_docs (default 2000), SYNTH_TOKENS per-doc tokens (default
//!   180), SYNTH_DIM (default 128), 50 queries x 32 tokens.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{s, Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::index::MmapIndex;
use next_plaid::search::{
    exact_score_docs_prepared, search_one_mmap, stage1_shortlist, SearchParameters,
};
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

fn median3(a: f64, b: f64, c: f64) -> f64 {
    a.max(b).min(a.min(b).max(c))
}

/// Mean and p50 of a set of per-query timings (ms).
fn agg(mut v: Vec<f64>) -> (f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    v.sort_by(|a, b| a.total_cmp(b));
    (mean, v.get(v.len() / 2).copied().unwrap_or(0.0))
}

/// Time one closure with 1 warmup + median of 3, in ms.
fn time3(mut f: impl FnMut()) -> f64 {
    f();
    let mut t = [0.0f64; 3];
    for slot in &mut t {
        let s = Instant::now();
        f();
        *slot = s.elapsed().as_secs_f64() * 1e3;
    }
    median3(t[0], t[1], t[2])
}

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("usage: stage2_profile <bundle_dir|synth> [max_queries]");
    let arg2: Option<usize> = std::env::args().nth(2).map(|s| s.parse().unwrap());

    let (name, docs, queries): (String, Vec<Array2<f32>>, Vec<Array2<f32>>) = if arg1 == "synth" {
        let n_docs = arg2.unwrap_or(2000);
        let t: usize = std::env::var("SYNTH_TOKENS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(180);
        let dim: usize = std::env::var("SYNTH_DIM")
            .map(|v| v.parse().unwrap())
            .unwrap_or(128);
        let mut s = 0x5eed_u64;
        let mut unit_rows = |n: usize| -> Array2<f32> {
            let mut a = Array2::<f32>::zeros((n, dim));
            for mut row in a.rows_mut() {
                for v in row.iter_mut() {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                row.mapv_inplace(|v| v / norm);
            }
            a
        };
        let docs: Vec<Array2<f32>> = (0..n_docs).map(|_| unit_rows(t)).collect();
        let queries: Vec<Array2<f32>> = (0..50).map(|_| unit_rows(32)).collect();
        (format!("synth-{n_docs}x{t}"), docs, queries)
    } else {
        let bundle = PathBuf::from(&arg1);
        let corpus =
            Array2::<f32>::read_npy(File::open(bundle.join("corpus.npy")).unwrap()).unwrap();
        let lens =
            Array1::<i64>::read_npy(File::open(bundle.join("corpus_lens.npy")).unwrap()).unwrap();
        let queries_c =
            Array2::<f32>::read_npy(File::open(bundle.join("queries.npy")).unwrap()).unwrap();
        let qlens =
            Array1::<i64>::read_npy(File::open(bundle.join("query_lens.npy")).unwrap()).unwrap();
        let docs = unpack(&corpus, &lens);
        let queries: Vec<Array2<f32>> = unpack(&queries_c, &qlens)
            .into_iter()
            .take(arg2.unwrap_or(usize::MAX))
            .collect();
        (
            bundle.file_name().unwrap().to_string_lossy().into_owned(),
            docs,
            queries,
        )
    };
    let total_tokens: usize = docs.iter().map(|d| d.nrows()).sum();
    println!(
        "stage2_profile: {} ({} docs, {} doc tokens, {} queries, dim {})",
        name,
        docs.len(),
        total_tokens,
        queries.len(),
        docs[0].ncols(),
    );
    println!("params: SearchParameters::default() (n_ivf_probe=8, n_full_scores=4096)");
    let params = SearchParameters::default();

    // (label, nbits, binary)
    let builds = [("r4", 4usize, false), ("r2", 2, false), ("r1", 1, false), ("binary", 4, true)];
    for (tag, nbits, binary) in builds {
        let dir = std::env::temp_dir().join(format!("np_stage2_profile_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        let t = Instant::now();
        let config = IndexConfig { nbits, binary, seed: Some(42), ..Default::default() };
        let index = MmapIndex::create_with_kmeans(&docs, dir.to_str().unwrap(), &config).unwrap();
        let build_s = t.elapsed().as_secs_f64();

        // Stage-1 once per query: the production shortlist all schemes share.
        let mut stage1_ms = Vec::new();
        let mut shortlists: Vec<(Array2<f32>, Vec<usize>)> = Vec::new();
        let mut cand_docs = 0usize;
        let mut cand_tokens = 0usize;
        for q in &queries {
            let t = Instant::now();
            let (cdot, ids) = stage1_shortlist(&index, q, &params, None).unwrap();
            stage1_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let ids: Vec<usize> = ids.iter().map(|&d| d as usize).collect();
            cand_docs += ids.len();
            cand_tokens += ids
                .iter()
                .map(|&d| index.doc_offsets[d + 1] - index.doc_offsets[d])
                .sum::<usize>();
            shortlists.push((cdot, ids));
        }
        let (s1_mean, s1_p50) = agg(stage1_ms);
        let nq = queries.len() as f64;
        println!(
            "\n=== index {tag} (nbits={nbits}, binary={binary}) — build {build_s:.1}s ===\n\
             stage1 (cdot + probe + approx prune): mean {s1_mean:.2} ms, p50 {s1_p50:.2} ms; \
             shortlist mean {:.0} docs / {:.0} tokens per query",
            cand_docs as f64 / nq,
            cand_tokens as f64 / nq,
        );
        let tok_per_q = cand_tokens as f64 / nq;

        // One-time cost (residual indexes): first asym prep builds the
        // inv-norms cache for the whole index.
        if !binary {
            let (cdot, _) = &shortlists[0];
            let t = Instant::now();
            let _ = exact_score_docs_prepared(&index, &queries[0], cdot, &[], true);
            let first = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let _ = exact_score_docs_prepared(&index, &queries[0], cdot, &[], true);
            let steady = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "one-time inv-norms cache (first asym query): {:.1} ms  \
                 (steady prep {:.3} ms)",
                first - steady,
                steady
            );
        }

        // Schemes on this index. Binary indexes have one path; residual
        // indexes get the float A/B plus its decompress/GEMM decomposition.
        let schemes: &[(&str, bool)] =
            if binary { &[("binary", false)] } else { &[("float", false), ("asym", true)] };
        let mut float_mean = f64::NAN;
        for &(scheme, asym) in schemes {
            let mut prep_ms = Vec::new();
            let mut exact_ms = Vec::new();
            for (q, (cdot, ids)) in queries.iter().zip(&shortlists) {
                prep_ms.push(time3(|| {
                    std::hint::black_box(exact_score_docs_prepared(&index, q, cdot, &[], asym));
                }));
                exact_ms.push(time3(|| {
                    std::hint::black_box(exact_score_docs_prepared(&index, q, cdot, ids, asym));
                }));
            }
            // The whole pipeline through the public API (stage-1 + prep +
            // exact + sort/top-k) — the number a caller of `search` sees.
            // Cross-check: e2e ≈ stage1 + prep + exact.
            let sp = SearchParameters {
                residual_asym: asym,
                ..SearchParameters::default()
            };
            let mut e2e_ms = Vec::new();
            for q in &queries {
                e2e_ms.push(time3(|| {
                    std::hint::black_box(search_one_mmap(&index, q, &sp, None).unwrap());
                }));
            }
            let (p_mean, _) = agg(prep_ms);
            let (e_mean, e_p50) = agg(exact_ms);
            let (t_mean, t_p50) = agg(e2e_ms);
            if scheme == "float" {
                float_mean = e_mean;
            }
            println!(
                "{scheme:<8} prep {:7.0} µs   exact mean {e_mean:7.3} ms  p50 {e_p50:7.3} ms   \
                 {:6.1} ns/token   {:.2}x vs float   | e2e mean {t_mean:7.3} ms  p50 {t_p50:7.3} ms",
                p_mean * 1e3,
                e_mean * 1e6 / tok_per_q,
                float_mean / e_mean,
            );
        }

        // Float decomposition on real shortlists: decompress-only, then pure
        // GEMM MaxSim over pre-decompressed docs (what remains of the float
        // path if decompression were free).
        if !binary {
            use rayon::prelude::*;
            let mut dec_ms = Vec::new();
            let mut gemm_ms = Vec::new();
            for (q, (_, ids)) in queries.iter().zip(&shortlists) {
                dec_ms.push(time3(|| {
                    let s: f32 = ids
                        .par_iter()
                        .map(|&d| index.get_document_embeddings(d).unwrap()[[0, 0]])
                        .sum();
                    std::hint::black_box(s);
                }));
                let dense: Vec<Array2<f32>> =
                    ids.iter().map(|&d| index.get_document_embeddings(d).unwrap()).collect();
                gemm_ms.push(time3(|| {
                    let s: f32 = dense
                        .par_iter()
                        .map(|d| {
                            let sim = q.dot(&d.t());
                            sim.rows()
                                .into_iter()
                                .map(|r| r.fold(f32::NEG_INFINITY, |m, &v| m.max(v)))
                                .sum::<f32>()
                        })
                        .sum();
                    std::hint::black_box(s);
                }));
            }
            let (d_mean, _) = agg(dec_ms);
            let (g_mean, _) = agg(gemm_ms);
            println!(
                "  float = decompress {d_mean:.3} ms ({:.0}%) + gemm+max {g_mean:.3} ms ({:.0}%)   \
                 [sum {:.3} vs measured {float_mean:.3}]",
                100.0 * d_mean / float_mean,
                100.0 * g_mean / float_mean,
                d_mean + g_mean,
            );
        }
    }
}
