//! CPU encode configuration matrix.
//!
//! Production indexing runs N single-threaded sessions at batch=1 in input
//! order. That avoids padding waste entirely but keeps every GEMM tiny. This
//! measures whether length-sorted larger batches beat it, and separates the two
//! effects: batching alone (B) vs batching + sorting (C/D).
//!
//! Also gates int8 correctness by embedding SPREAD, not norms — the exported
//! graph L2-normalises, so norms are 1.0 even if int8 has collapsed.
//!
//!   ORT_DYLIB_PATH=... cargo run --release --example cpu_matrix -- \
//!       --model <dir> --corpus <dir> [--docs 600]

use anyhow::{Context, Result};
use ndarray::Array2;
use next_plaid_onnx::{Colbert, ExecutionProvider};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

struct Arm {
    label: &'static str,
    sessions: usize,
    threads: usize,
    batch: usize,
    sorted: bool,
}

fn collect_corpus(root: &Path, want: usize, chunk_lines: usize) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs" || e == "py") {
                let Ok(text) = fs::read_to_string(&path) else {
                    continue;
                };
                let lines: Vec<&str> = text.lines().collect();
                for window in lines.chunks(chunk_lines) {
                    let unit = window.join("\n");
                    if unit.trim().len() > 80 {
                        chunks.push(unit);
                    }
                    if chunks.len() >= want {
                        return Ok(chunks);
                    }
                }
            }
        }
    }
    Ok(chunks)
}

/// Padding cost of a given (order, batch) pairing: padded tokens / real tokens.
fn padding_ratio(lengths: &[usize], batch: usize) -> f64 {
    let real: usize = lengths.iter().sum();
    let padded: usize = lengths
        .chunks(batch)
        .map(|c| c.len() * c.iter().copied().max().unwrap_or(0))
        .sum();
    padded as f64 / real.max(1) as f64
}

fn build(model_dir: &Path, arm: &Arm, quantized: bool) -> Result<Colbert> {
    let mut builder = Colbert::builder(model_dir)
        .with_quantized(quantized)
        .with_batch_size(arm.batch)
        .with_dynamic_batch(false)
        .with_execution_provider(ExecutionProvider::Cpu);
    builder = if arm.sessions > 1 {
        builder.with_parallel(arm.sessions)
    } else {
        builder.with_threads(arm.threads)
    };
    builder.build()
}

/// Mean pairwise cosine between DISTINCT documents' mean-pooled embeddings.
/// Collapsed int8 drives this toward 1.0 while per-row norms stay exactly 1.0.
fn spread(embeddings: &[Array2<f32>]) -> f64 {
    let pooled: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|doc| {
            let mut v = vec![0.0f32; doc.ncols()];
            for row in doc.rows() {
                for (acc, x) in v.iter_mut().zip(row.iter()) {
                    *acc += *x;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect();

    let mut total = 0.0f64;
    let mut count = 0usize;
    for i in 0..pooled.len() {
        for j in (i + 1)..pooled.len() {
            let dot: f32 = pooled[i]
                .iter()
                .zip(pooled[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            total += dot as f64;
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut model_dir = None;
    let mut corpus_dir = None;
    let mut want = 600usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--corpus" => {
                i += 1;
                corpus_dir = Some(PathBuf::from(&args[i]));
            }
            "--docs" => {
                i += 1;
                want = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }
    let model_dir = model_dir.context("--model is required")?;
    let corpus_dir = corpus_dir.context("--corpus is required")?;

    for chunk_lines in [40usize, 8usize] {
        run_matrix(&model_dir, &corpus_dir, want, chunk_lines)?;
    }
    probe_int8(&model_dir, &corpus_dir, want)?;
    Ok(())
}

fn run_matrix(model_dir: &Path, corpus_dir: &Path, want: usize, chunk_lines: usize) -> Result<()> {
    let docs = collect_corpus(corpus_dir, want, chunk_lines)?;
    anyhow::ensure!(
        !docs.is_empty(),
        "no corpus units found under {corpus_dir:?}"
    );

    // Exact per-document token lengths, so padding cost is analytic not guessed.
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let lengths: Vec<usize> = docs
        .iter()
        .map(|d| {
            tokenizer
                .encode(d.as_str(), false)
                .map(|e| e.get_ids().len())
                .unwrap_or(0)
        })
        .collect();

    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let sorted_docs: Vec<String> = order.iter().map(|&i| docs[i].clone()).collect();
    let sorted_lengths: Vec<usize> = order.iter().map(|&i| lengths[i]).collect();

    let mean_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    println!(
        "{} units | tokens mean {:.0} max {} | {} cores\n",
        docs.len(),
        mean_len,
        max_len,
        cores
    );

    let n = cores.min(16);
    let arms = vec![
        Arm {
            label: "PROD  par=N b=1  input ",
            sessions: n,
            threads: 1,
            batch: 1,
            sorted: false,
        },
        Arm {
            label: "      par=N b=8  input ",
            sessions: n,
            threads: 1,
            batch: 8,
            sorted: false,
        },
        Arm {
            label: "      par=N b=8  SORTED",
            sessions: n,
            threads: 1,
            batch: 8,
            sorted: true,
        },
        Arm {
            label: "      par=N b=16 SORTED",
            sessions: n,
            threads: 1,
            batch: 16,
            sorted: true,
        },
        Arm {
            label: "      par=N b=32 SORTED",
            sessions: n,
            threads: 1,
            batch: 32,
            sorted: true,
        },
        Arm {
            label: "CTRL  1sess/Nt  b=32 SRT",
            sessions: 1,
            threads: cores,
            batch: 32,
            sorted: true,
        },
        Arm {
            label: "CTRL  1sess/Nt  b=1  inp",
            sessions: 1,
            threads: cores,
            batch: 1,
            sorted: false,
        },
        Arm {
            label: "CTRL  1sess/4t   b=32 SRT",
            sessions: 1,
            threads: 4,
            batch: 32,
            sorted: true,
        },
        Arm {
            label: "      par=2N b=1  input ",
            sessions: cores * 2,
            threads: 1,
            batch: 1,
            sorted: false,
        },
    ];

    println!(
        "{:<26} {:>10} {:>9} {:>8}",
        format!("arm (int8, {chunk_lines}-line)"),
        "docs/s",
        "vs PROD",
        "pad"
    );
    println!("{}", "-".repeat(56));

    let mut baseline = 0.0f64;
    for (idx, arm) in arms.iter().enumerate() {
        let model = build(model_dir, arm, true)?;
        let (data, lens) = if arm.sorted {
            (&sorted_docs, &sorted_lengths)
        } else {
            (&docs, &lengths)
        };
        let refs: Vec<&str> = data.iter().map(|s| s.as_str()).collect();

        model.encode_documents(&refs[..refs.len().min(64)], None)?; // warmup

        let mut best = f64::MAX;
        for _ in 0..3 {
            let start = Instant::now();
            let out = model.encode_documents(&refs, None)?;
            let secs = start.elapsed().as_secs_f64();
            anyhow::ensure!(
                out.len() == refs.len(),
                "arm returned {} of {}",
                out.len(),
                refs.len()
            );
            best = best.min(secs);
        }
        let rate = refs.len() as f64 / best;
        if idx == 0 {
            baseline = rate;
        }
        println!(
            "{:<26} {:>10.1} {:>8.2}x {:>7.2}x",
            arm.label,
            rate,
            rate / baseline,
            padding_ratio(lens, arm.batch)
        );
    }

    println!();
    Ok(())
}

fn probe_int8(model_dir: &Path, corpus_dir: &Path, want: usize) -> Result<()> {
    let docs = collect_corpus(corpus_dir, want, 40)?;
    println!("int8 vs fp32 on a fixed 48-unit probe");
    let probe: Vec<&str> = docs.iter().take(48).map(|s| s.as_str()).collect();
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let arm = Arm {
        label: "probe",
        sessions: 1,
        threads: cores,
        batch: 8,
        sorted: false,
    };

    let int8 = build(model_dir, &arm, true)?.encode_documents(&probe, None)?;
    let fp32 = build(model_dir, &arm, false)?.encode_documents(&probe, None)?;

    let (s8, s32) = (spread(&int8), spread(&fp32));
    println!("  mean pairwise cosine  int8 {s8:.4}  fp32 {s32:.4}   (collapse -> ~1.0)");

    let mut per_doc = Vec::new();
    for (a, b) in int8.iter().zip(fp32.iter()) {
        if a.shape() != b.shape() {
            continue;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        per_doc.push(dot as f64 / a.nrows() as f64);
    }
    let mean_cos = per_doc.iter().sum::<f64>() / per_doc.len().max(1) as f64;
    let worst = per_doc.iter().cloned().fold(f64::MAX, f64::min);
    println!("  int8-vs-fp32 token cosine  mean {mean_cos:.5}  worst-doc {worst:.5}");

    Ok(())
}
