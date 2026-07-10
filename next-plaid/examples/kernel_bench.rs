//! Focused microbenchmark: what is the fastest way to compute
//! int8-query × 1-bit-document MaxSim?
//!
//! Storage is FIXED at packed sign bits (32×) for every binary variant — the
//! only thing that changes is the COMPUTE-time kernel:
//!
//!   A. per-pair 2P-T      — the fallback path for ragged dims (dim % 8 != 0),
//!                           dims > 256, and CPUs without the fused kernels'
//!                           features: a SIMD dot per (query, doc-token) pair
//!                           on the packed bits.
//!   F/G/H. fused kernels  — the byte-aligned-dim defaults (dim % 8 == 0, up
//!                           to 256): doc-token-outer, each doc token's bits
//!                           expanded once and amortized over all query tokens
//!                           (AVX-512 VNNI / AVX2 masked-SAD / NEON SDOT).
//!   C. decode -> f32 GEMM — maxsim_binary (decode to ±1 then BLAS), reference.
//!
//! Run (dim defaults to 128):
//!   cargo run -p next-plaid --release --example kernel_bench [-- <dim>]
//!   cargo run -p next-plaid --release --features accelerate --example kernel_bench

use std::time::{Duration, Instant};

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;

use next_plaid::binary::{
    binarize, maxsim_binary, maxsim_binary_i8, maxsim_binary_i8_force_avx2_sad,
    maxsim_binary_i8_force_neon, maxsim_binary_i8_force_vnni, maxsim_binary_i8_pairwise,
    quantize_query_i8,
};

const N_DOCS: usize = 4000;
const DOC_TOKENS: usize = 80;
const QUERY_TOKENS: usize = 32;
const REPS: usize = 20;

fn randn(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    Array2::random_using(
        (rows, cols),
        StandardNormal,
        &mut StdRng::seed_from_u64(seed),
    )
}

fn best_of<F: Fn() -> f32>(f: F) -> (Duration, f32) {
    // black_box: the accumulator is discarded at every call site, and without
    // an opaque use LLVM dead-code-eliminates the loop bodies of variants it
    // can fully inline, inflating their numbers by an order of magnitude.
    let mut acc = std::hint::black_box(f()); // warmup
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..REPS {
        let t = Instant::now();
        acc += std::hint::black_box(f());
        best = best.min(t.elapsed());
    }
    (best, acc)
}

fn main() {
    let dim: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("dim must be a positive integer"))
        .unwrap_or(128);
    println!(
        "int8 × 1-bit kernel microbenchmark\n\
         dim={dim} docs={N_DOCS} doc_tokens={DOC_TOKENS} query_tokens={QUERY_TOKENS} reps={REPS}\n\
         (storage fixed at packed bits for A/C and the fused kernels)\n"
    );

    // One query, many docs — this is the Stage-2 rescore inner shape.
    let query = randn(QUERY_TOKENS, dim, 1);
    let docs_f32: Vec<Array2<f32>> = (0..N_DOCS)
        .map(|i| randn(DOC_TOKENS, dim, 100 + i as u64))
        .collect();
    let docs_bits: Vec<Array2<u8>> = docs_f32.iter().map(|d| binarize(&d.view())).collect();

    let q8 = quantize_query_i8(&query.view());
    // Float query for the decode->GEMM reference (int8-rounded to be comparable).
    let qf = {
        let mut m = query.clone();
        for mut row in m.axis_iter_mut(Axis(0)) {
            let ma = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            if ma > 0.0 {
                let s = ma / 127.0;
                row.mapv_inplace(|x| (x / s).round().clamp(-127.0, 127.0) * s);
            }
        }
        m
    };

    let (t_a, _) = best_of(|| {
        docs_bits
            .iter()
            .map(|b| maxsim_binary_i8_pairwise(&q8, &b.view(), dim))
            .sum()
    });
    // F/G: the fused doc-token-outer kernels behind the dispatched entry point.
    let t_f = maxsim_binary_i8_force_vnni(&q8, &docs_bits[0].view()).map(|_| {
        best_of(|| {
            docs_bits
                .iter()
                .map(|b| maxsim_binary_i8_force_vnni(&q8, &b.view()).unwrap())
                .sum()
        })
        .0
    });
    let t_g = maxsim_binary_i8_force_avx2_sad(&q8, &docs_bits[0].view()).map(|_| {
        best_of(|| {
            docs_bits
                .iter()
                .map(|b| maxsim_binary_i8_force_avx2_sad(&q8, &b.view()).unwrap())
                .sum()
        })
        .0
    });
    let t_h = maxsim_binary_i8_force_neon(&q8, &docs_bits[0].view()).map(|_| {
        best_of(|| {
            docs_bits
                .iter()
                .map(|b| maxsim_binary_i8_force_neon(&q8, &b.view()).unwrap())
                .sum()
        })
        .0
    });
    let (t_disp, _) = best_of(|| {
        docs_bits
            .iter()
            .map(|b| maxsim_binary_i8(&q8, &b.view(), dim))
            .sum()
    });
    let (t_c, _) = best_of(|| {
        docs_bits
            .iter()
            .map(|b| maxsim_binary(&qf.view(), &b.view(), dim))
            .sum()
    });

    let us = |d: Duration| d.as_secs_f64() * 1e6 / N_DOCS as f64;
    let rel = |d: Duration| t_a.as_secs_f64() / d.as_secs_f64();
    println!("Per-doc Stage-2 latency (scoring 1 query's {QUERY_TOKENS} tokens vs a {DOC_TOKENS}-token doc):");
    println!(
        "  A. per-pair 2P-T       (32x):  {:>7.3} us/doc   {:.2}x  (fallback)",
        us(t_a),
        rel(t_a)
    );
    if let Some(t) = t_f {
        println!(
            "  F. fused VNNI-512      (32x):  {:>7.3} us/doc   {:.2}x  (vpdpbusd, doc-outer)",
            us(t),
            rel(t)
        );
    }
    if let Some(t) = t_g {
        println!(
            "  G. fused AVX2 SAD      (32x):  {:>7.3} us/doc   {:.2}x  (psadbw, doc-outer)",
            us(t),
            rel(t)
        );
    }
    if let Some(t) = t_h {
        println!(
            "  H. fused NEON SDOT     (32x):  {:>7.3} us/doc   {:.2}x  (sdot, doc-outer)",
            us(t),
            rel(t)
        );
    }
    println!(
        "  *. dispatched default  (32x):  {:>7.3} us/doc   {:.2}x  (maxsim_binary_i8)",
        us(t_disp),
        rel(t_disp)
    );
    println!(
        "  C. decode->f32 GEMM    (32x):  {:>7.3} us/doc   {:.2}x",
        us(t_c),
        rel(t_c)
    );
    println!("\n(baseline = A; higher x = faster)");
}
