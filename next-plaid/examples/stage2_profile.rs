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
//! Usage: `stage2_profile <bundle_dir|synth> [max_queries|n_docs]`
//!        `stage2_profile shapes <lens_dir> [n_docs_cap]`
//!
//! * `<bundle_dir>` — real embeddings (corpus.npy + corpus_lens.npy +
//!   queries.npy + query_lens.npy); [max_queries] caps queries.
//! * `synth` — bundle-free corpus for CI (seeded LCG unit-norm tokens; phase
//!   latency depends on shapes, not values): [n_docs] docs (default 2000),
//!   SYNTH_TOKENS per-doc tokens (default 180), SYNTH_DIM (default 128),
//!   50 queries x 32 tokens.
//! * `shapes` — synthetic values with REAL shape distributions: reads only
//!   corpus_lens.npy + query_lens.npy from <lens_dir> (tiny, committable),
//!   generates unit-norm tokens per real doc/query length. SHAPE_DIM
//!   (default 128) sets dim; QUERIES (default 50) caps query count.
//!   [n_docs_cap] takes the first N docs. Doc and query streams use
//!   independent seeds, so queries are identical whether or not the doc
//!   side is regenerated (matters for INDEX_ROOT cache hits).
//!
//! Env:
//! * `INDEX_ROOT=dir` — persistent index cache: each (name, scheme) index is
//!   built once under dir and mmap-loaded on later runs (any platform — the
//!   on-disk format is little-endian everywhere). A completion marker guards
//!   against half-built dirs from killed runs. Without it, indexes go to a
//!   temp dir and are rebuilt every run. Profiling from a warm cache needs
//!   no corpus in RAM at all — the build's multi-GB k-means footprint is
//!   the reason this exists (a scifact-scale build peaks ~20 GB).
//! * `BUILD_ONLY=1` — ensure all four indexes exist, then exit without
//!   profiling (for a dedicated build job).
//! * `INDEX_TAGS=r4,binary` — restrict which indexes are built/profiled
//!   (corpus-size ladder cells skip r2/r1: the ladder's axis is the
//!   float/asym/binary end-to-end ratio vs corpus size, and r2/r1 would
//!   double the build and artifact cost without informing it).
//! * `NO_BUILD=1` — never build: profile cached indexes, skip missing ones
//!   with a note. Profile jobs consuming a build artifact set this so a
//!   partial artifact (e.g. the tallest ladder rung OOM'd in the build job)
//!   degrades to a skipped cell instead of a doomed multi-GB build on a
//!   small runner.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{s, Array1, Array2};
use ndarray_npy::ReadNpyExt;
use next_plaid::index::MmapIndex;
use next_plaid::search::{
    exact_score_docs_prepared, exact_score_docs_prepared_t, search_one_mmap, stage1_shortlist,
    SearchParameters,
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

/// Seeded LCG unit-norm rows — deterministic across platforms and runs.
fn lcg_unit_rows(s: &mut u64, n: usize, dim: usize) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((n, dim));
    for mut row in a.rows_mut() {
        for v in row.iter_mut() {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }
    a
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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).map(|v| v.parse().unwrap()).unwrap_or(default)
}

/// What the profiler knows about the corpus without materializing it.
/// `gen_docs` is called only when at least one index must be built.
struct Corpus {
    name: String,
    dim: usize,
    doc_lens: Vec<usize>,
    gen_docs: Box<dyn Fn() -> Vec<Array2<f32>>>,
}

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("usage: stage2_profile <bundle_dir|synth> [n] | shapes <lens_dir> [n_docs_cap]");

    let (corpus, queries): (Corpus, Vec<Array2<f32>>) = if arg1 == "shapes" {
        let dir = PathBuf::from(
            std::env::args().nth(2).expect("usage: stage2_profile shapes <lens_dir> [n_docs_cap]"),
        );
        let cap: usize =
            std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
        let dim = env_usize("SHAPE_DIM", 128);
        let n_queries = env_usize("QUERIES", 50);
        let clens =
            Array1::<i64>::read_npy(File::open(dir.join("corpus_lens.npy")).unwrap()).unwrap();
        let qlens =
            Array1::<i64>::read_npy(File::open(dir.join("query_lens.npy")).unwrap()).unwrap();
        let doc_lens: Vec<usize> = clens.iter().take(cap).map(|&l| l as usize).collect();
        let mut qseed = 0x0DD5EED5_u64;
        let queries: Vec<Array2<f32>> = qlens
            .iter()
            .take(n_queries)
            .map(|&l| lcg_unit_rows(&mut qseed, l as usize, dim))
            .collect();
        let name = format!(
            "shapes-{}-{}docs",
            dir.file_name().unwrap().to_string_lossy(),
            doc_lens.len()
        );
        let lens_for_gen = doc_lens.clone();
        let gen_docs = Box::new(move || {
            let mut s = 0x5eed_u64;
            lens_for_gen.iter().map(|&l| lcg_unit_rows(&mut s, l, dim)).collect()
        });
        (Corpus { name, dim, doc_lens, gen_docs }, queries)
    } else if arg1 == "synth" {
        let n_docs: usize =
            std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
        let t = env_usize("SYNTH_TOKENS", 180);
        let dim = env_usize("SYNTH_DIM", 128);
        let mut qseed = 0x0DD5EED5_u64;
        let queries: Vec<Array2<f32>> =
            (0..50).map(|_| lcg_unit_rows(&mut qseed, 32, dim)).collect();
        let gen_docs = Box::new(move || {
            let mut s = 0x5eed_u64;
            (0..n_docs).map(|_| lcg_unit_rows(&mut s, t, dim)).collect()
        });
        (
            Corpus {
                name: format!("synth-{n_docs}x{t}"),
                dim,
                doc_lens: vec![t; n_docs],
                gen_docs,
            },
            queries,
        )
    } else {
        let bundle = PathBuf::from(&arg1);
        let max_queries: usize =
            std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
        let lens =
            Array1::<i64>::read_npy(File::open(bundle.join("corpus_lens.npy")).unwrap()).unwrap();
        let queries_c =
            Array2::<f32>::read_npy(File::open(bundle.join("queries.npy")).unwrap()).unwrap();
        let qlens =
            Array1::<i64>::read_npy(File::open(bundle.join("query_lens.npy")).unwrap()).unwrap();
        let queries: Vec<Array2<f32>> =
            unpack(&queries_c, &qlens).into_iter().take(max_queries).collect();
        let dim = queries_c.ncols();
        let doc_lens: Vec<usize> = lens.iter().map(|&l| l as usize).collect();
        let name = bundle.file_name().unwrap().to_string_lossy().into_owned();
        let bundle_for_gen = bundle.clone();
        let gen_docs = Box::new(move || {
            let corpus =
                Array2::<f32>::read_npy(File::open(bundle_for_gen.join("corpus.npy")).unwrap())
                    .unwrap();
            let lens =
                Array1::<i64>::read_npy(File::open(bundle_for_gen.join("corpus_lens.npy")).unwrap())
                    .unwrap();
            unpack(&corpus, &lens)
        });
        (Corpus { name, dim, doc_lens, gen_docs }, queries)
    };

    let total_tokens: usize = corpus.doc_lens.iter().sum();
    println!(
        "stage2_profile: {} ({} docs, {} doc tokens, {} queries, dim {})",
        corpus.name,
        corpus.doc_lens.len(),
        total_tokens,
        queries.len(),
        corpus.dim,
    );
    println!("params: SearchParameters::default() (n_ivf_probe=8, n_full_scores=4096)");
    // Name the kernel that will actually run, and any ablation in force.
    // A speedup credited to a code path that never executed is the easiest
    // measurement error to make and the hardest to spot afterwards.
    println!(
        "asym kernel: {}   ablation: {:?}",
        next_plaid::residual_lut::active_kernel_name(corpus.dim, true),
        next_plaid::residual_lut::ablation(),
    );
    let params = SearchParameters::default();
    let index_root = std::env::var("INDEX_ROOT").ok();
    let build_only = std::env::var("BUILD_ONLY").is_ok();

    // (label, nbits, binary). INDEX_TAGS=r4,binary restricts the set — used
    // by corpus-size ladder cells where only the float/asym-vs-binary
    // end-to-end ratio matters and r2/r1 would double build + artifact cost.
    // Empty means unset (CI matrix cells pass INDEX_TAGS="" for "all").
    let tag_filter = std::env::var("INDEX_TAGS").ok().filter(|s| !s.is_empty());
    let builds: Vec<(&str, usize, bool)> =
        [("r4", 4usize, false), ("r2", 2, false), ("r1", 1, false), ("binary", 4, true)]
            .into_iter()
            .filter(|(tag, _, _)| {
                tag_filter.as_ref().is_none_or(|f| f.split(',').any(|t| t == *tag))
            })
            .collect();

    // With INDEX_ROOT, an index is reused iff its completion marker exists
    // (a killed build leaves no marker and is rebuilt). Without it, every
    // run rebuilds in a temp dir, as before.
    let dir_for = |tag: &str| -> PathBuf {
        match &index_root {
            Some(r) => Path::new(r).join(format!("{}_{tag}", corpus.name)),
            None => std::env::temp_dir().join(format!("np_stage2_profile_{tag}")),
        }
    };
    let marker = |dir: &Path| dir.join("PROFILE_INDEX_OK");
    let cached = |tag: &str| index_root.is_some() && marker(&dir_for(tag)).exists();

    // Materialize the corpus only if something needs building.
    let no_build = std::env::var("NO_BUILD").is_ok();
    let need_build = !no_build && builds.iter().any(|(tag, _, _)| !cached(tag));
    let docs: Vec<Array2<f32>> = if need_build {
        let t = Instant::now();
        let d = (corpus.gen_docs)();
        println!("corpus materialized in {:.1}s (needed for index build)", t.elapsed().as_secs_f64());
        d
    } else {
        println!("corpus not materialized (no index build will run)");
        Vec::new()
    };

    for (tag, nbits, binary) in builds {
        let dir = dir_for(tag);
        let index = if cached(tag) {
            let t = Instant::now();
            let index = MmapIndex::load(dir.to_str().unwrap()).unwrap();
            println!("\n=== index {tag} (nbits={nbits}, binary={binary}) — loaded from cache in {:.2}s ===", t.elapsed().as_secs_f64());
            index
        } else if no_build {
            println!("\n=== index {tag}: not cached and NO_BUILD set — skipped ===");
            continue;
        } else {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
            let t = Instant::now();
            let config = IndexConfig {
                nbits,
                binary,
                seed: Some(42),
                // Never persist raw f32 embeddings alongside the index: the
                // cache should hold only what search reads.
                start_from_scratch: 0,
                ..Default::default()
            };
            let index =
                MmapIndex::create_with_kmeans(&docs, dir.to_str().unwrap(), &config).unwrap();
            if index_root.is_some() {
                std::fs::write(marker(&dir), b"ok").unwrap();
            }
            println!("\n=== index {tag} (nbits={nbits}, binary={binary}) — build {:.1}s ===", t.elapsed().as_secs_f64());
            index
        };
        if build_only {
            println!("BUILD_ONLY: skipping profile for {tag}");
            continue;
        }

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
            "stage1 (cdot + probe + approx prune): mean {s1_mean:.2} ms, p50 {s1_p50:.2} ms; \
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
                // The prep row includes the [nq, K] -> [K, nq] transpose the
                // asym path pays once per query in production; the exact row
                // must NOT re-pay it (a K-dependent cost the float column
                // never sees), so the exact timing gets the pre-transposed
                // matrix and times scoring only.
                prep_ms.push(time3(|| {
                    std::hint::black_box(exact_score_docs_prepared(&index, q, cdot, &[], asym));
                }));
                let cdot_t = cdot.t().as_standard_layout().into_owned();
                exact_ms.push(time3(|| {
                    std::hint::black_box(exact_score_docs_prepared_t(
                        &index, q, &cdot_t, ids, asym,
                    ));
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
