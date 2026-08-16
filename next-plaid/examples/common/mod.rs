//! Shared bench scaffolding for the `examples/` harnesses.
//!
//! Everything here is deterministic and dependency-free, so two machines — or
//! two CI runners on different architectures — produce comparable inputs from
//! the same arguments.

#![allow(dead_code)] // each example uses a subset

use ndarray::Array2;

/// Deterministic pseudo-random floats in `[-1, 1)`; no `rand` dependency and
/// the same numbers on every platform, so two machines' outputs compare.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// Sum of 4 uniforms — approximately normal, which matters for synthetic
    /// residuals: the codecs' cutoffs are quantiles, so a uniform residual
    /// distribution would flatter the dead-zone codecs relative to real data.
    pub fn next_gauss(&mut self) -> f32 {
        (0..4).map(|_| self.next_f32()).sum::<f32>() * 0.5
    }

    pub fn array(&mut self, rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| self.next_f32() * scale)
    }
}

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn l2_normalize(row: &mut [f32]) {
    let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for v in row.iter_mut() {
        *v /= n;
    }
}

/// A seeded stand-in for a ColBERT bundle, for the benches that only need
/// *shapes and statistics* rather than real text.
///
/// Uniform noise would be the wrong generator here: k-means would find no
/// structure, every centroid would sit at roughly the same distance from every
/// token, and stage-1 pruning would keep a wildly unrealistic candidate set —
/// which is exactly the quantity that decides how much work reaches the
/// residual path being measured. So tokens are drawn around `n_clusters`
/// shared centers with Gaussian jitter and L2-normalized, the way real
/// late-interaction token embeddings sit: clustered on the unit sphere, with
/// residuals whose magnitude sets the codec's job.
///
/// `spread` is the jitter relative to the center; larger means bigger
/// residuals and therefore more quantization-sensitive.
pub struct SynthBundle {
    pub docs: Vec<Array2<f32>>,
    pub queries: Vec<Array2<f32>>,
}

pub fn synth_bundle(
    n_docs: usize,
    tokens_per_doc: usize,
    n_queries: usize,
    query_tokens: usize,
    dim: usize,
    n_clusters: usize,
    spread: f32,
    seed: u64,
) -> SynthBundle {
    let mut rng = Lcg::new(seed);

    let mut centers = Array2::<f32>::zeros((n_clusters, dim));
    for mut c in centers.rows_mut() {
        let s = c.as_slice_mut().unwrap();
        for v in s.iter_mut() {
            *v = rng.next_gauss();
        }
        l2_normalize(s);
    }

    let group = |n_groups: usize, len: usize, rng: &mut Lcg| -> Vec<Array2<f32>> {
        (0..n_groups)
            .map(|_| {
                // One dominant center per group (a document is topically
                // coherent), with each token free to drift to a neighbour.
                let home = (rng.0 as usize).wrapping_mul(2654435761) % n_clusters;
                let mut m = Array2::<f32>::zeros((len, dim));
                for mut row in m.rows_mut() {
                    let c = if rng.next_f32() < 0.6 {
                        home
                    } else {
                        (rng.0 as usize).wrapping_mul(40503) % n_clusters
                    };
                    let s = row.as_slice_mut().unwrap();
                    for (v, &cv) in s.iter_mut().zip(centers.row(c)) {
                        *v = cv + spread * rng.next_gauss();
                    }
                    l2_normalize(s);
                }
                m
            })
            .collect()
    };

    let docs = group(n_docs, tokens_per_doc, &mut rng);
    let queries = group(n_queries, query_tokens, &mut rng);
    SynthBundle { docs, queries }
}

/// Read a `usize` from the environment, or fall back.
pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read an `f32` from the environment, or fall back.
pub fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
