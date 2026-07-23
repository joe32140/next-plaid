//! Search functionality for PLAID

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use ndarray::Array1;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::codec::CentroidStore;
use crate::error::Result;
use crate::maxsim;

/// Per-token top-k heaps and per-centroid max scores from a batch of centroids.
type ProbePartial = (
    Vec<BinaryHeap<(Reverse<OrdF32>, usize)>>,
    HashMap<usize, f32>,
);

/// Maximum number of documents to decompress concurrently during exact scoring.
/// This limits peak memory usage from parallel decompression.
/// With 128 docs × ~300KB per doc = ~40MB max concurrent decompression memory.
const DECOMPRESS_CHUNK_SIZE: usize = 128;

/// Search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParameters {
    /// Number of queries per batch
    pub batch_size: usize,
    /// Number of documents to re-rank with exact scores
    pub n_full_scores: usize,
    /// Number of final results to return per query
    pub top_k: usize,
    /// Number of IVF cells to probe during search
    pub n_ivf_probe: usize,
    /// Batch size for centroid scoring during IVF probing (0 = exhaustive).
    /// Lower values use less memory but are slower. Default 100_000.
    /// Only used when num_centroids > centroid_batch_size.
    #[serde(default = "default_centroid_batch_size")]
    pub centroid_batch_size: usize,
    /// Centroid score threshold (t_cs) for centroid pruning.
    /// A centroid is only included if its maximum score across all query tokens
    /// meets or exceeds this threshold. Set to None to disable pruning.
    /// Default: Some(0.4)
    #[serde(default = "default_centroid_score_threshold")]
    pub centroid_score_threshold: Option<f32>,
    /// Score residual candidates asymmetrically — int8 query × int8 LUT over
    /// the stored codes plus the centroid term from the IVF probe matrix —
    /// instead of decompress→f32 MaxSim. Compute-only: same index, same
    /// storage, so the two modes can be A/B'd per search. Honored on both
    /// the dense and batched-centroid search paths (the batched path packs
    /// its sparse centroid scores into a compact matrix); ignored for
    /// binary indexes and for dims the fused kernels don't support
    /// (> 256), which fall back to the float path. Default off.
    #[serde(default)]
    pub residual_asym: bool,
}

fn default_centroid_batch_size() -> usize {
    100_000
}

fn default_centroid_score_threshold() -> Option<f32> {
    Some(0.4)
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k: 10,
            n_ivf_probe: 8,
            centroid_batch_size: default_centroid_batch_size(),
            centroid_score_threshold: default_centroid_score_threshold(),
            residual_asym: false,
        }
    }
}

/// Result of a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query ID
    pub query_id: usize,
    /// Retrieved document IDs (ranked by relevance)
    pub passage_ids: Vec<i64>,
    /// Relevance scores for each document
    pub scores: Vec<f32>,
}

/// ColBERT-style MaxSim scoring: for each query token, find the max similarity
/// with any document token, then sum across query tokens.
///
/// Always uses the CPU implementation (BLAS GEMM + SIMD max reduction), which
/// benchmarks show is faster than CUDA for per-document scoring due to GPU
/// transfer overhead dominating at typical query/document sizes.
fn colbert_score(query: &ArrayView2<f32>, doc: &ArrayView2<f32>) -> f32 {
    maxsim::maxsim_score(query, doc)
}

/// The query prepared once for Stage-2 scoring, in the representation the index
/// requires. Building this once (rather than per document) hoists query-side
/// work — int8 quantization for binary indexes — out of the per-candidate loop.
enum ScoreQuery<'a> {
    /// Binary index: query kept as int8 codes for the asymmetric int8 x 1-bit
    /// MaxSim kernel (the `q·b = 2·Σ_{b=+1} q − Σ q` identity, integer adds only).
    Binary(crate::binary::QueryI8),
    /// Float / residual index: the caller's full-precision query, borrowed —
    /// standard ColBERT MaxSim needs no per-query preparation.
    Float(&'a Array2<f32>),
    /// Residual index scored asymmetrically: int8 query codes plus the fused
    /// byte→int8-weights LUT, scoring the stored codes directly (centroid
    /// term from the dense query×centroid matrix; no decompression). The
    /// plane-ordered query copy feeds the SIMD kernels' expand layout.
    ResidualLut {
        q8: crate::binary::QueryI8,
        lut: crate::residual_lut::ResidualLut,
        /// `None` for non-byte-aligned dims, which score on the scalar path.
        planes: Option<crate::residual_lut::QueryPlanes>,
    },
}

/// Prepare the query for the index's Stage-2 scoring path, once per search.
fn prepare_score_query<'a>(
    index: &crate::index::MmapIndex,
    query: &'a Array2<f32>,
    residual_asym: bool,
) -> ScoreQuery<'a> {
    if index.metadata.binary {
        return ScoreQuery::Binary(crate::binary::quantize_query_i8(&query.view()));
    }
    if residual_asym && index.codec.embedding_dim() <= crate::residual_lut::MAX_DIM {
        if let Some(lut) = crate::residual_lut::quantize_lut(&index.codec) {
            let dim = index.codec.embedding_dim();
            let q8 = crate::binary::quantize_query_i8(&query.view());
            let planes = dim
                .is_multiple_of(8)
                .then(|| crate::residual_lut::build_query_planes(&q8, &lut, dim));
            return ScoreQuery::ResidualLut { q8, lut, planes };
        }
    }
    ScoreQuery::Float(query)
}

/// Exact MaxSim of the prepared query against document `doc_id`.
///
/// For binary indexes this scores the int8 query directly against the document's
/// stored 1-bit signs via the asymmetric `2P − T` kernel — no decompression to
/// float. Otherwise the residual codes are decompressed and scored with
/// full-precision ColBERT MaxSim.
fn exact_doc_score(
    index: &crate::index::MmapIndex,
    query: &ScoreQuery,
    cdot_t: Option<&Array2<f32>>,
    doc_id: usize,
) -> Option<f32> {
    match query {
        ScoreQuery::Binary(q8) => {
            let start = index.doc_offsets[doc_id];
            let end = index.doc_offsets[doc_id + 1];
            let doc_bits = index.mmap_residuals.slice_rows(start, end);
            Some(crate::binary::maxsim_binary_i8(
                q8,
                &doc_bits,
                index.codec.embedding_dim(),
            ))
        }
        ScoreQuery::Float(q) => {
            let doc = index.get_document_embeddings(doc_id).ok()?;
            Some(colbert_score(&q.view(), &doc.view()))
        }
        ScoreQuery::ResidualLut { q8, lut, planes } => {
            // Needs the query×centroid matrix for the centroid term, in the
            // kernels' centroid-major [K, nq] layout; prepare_score_query
            // never builds this arm on paths without it.
            let cdot_t = cdot_t?;
            let inv_norms = index.residual_inv_norms()?;
            let start = index.doc_offsets[doc_id];
            let end = index.doc_offsets[doc_id + 1];
            let packed = index.mmap_residuals.slice_rows(start, end);
            let codes = index.mmap_codes.slice(start, end);
            Some(crate::residual_lut::maxsim_residual_lut_i8(
                q8,
                planes.as_ref(),
                &packed,
                &codes,
                &cdot_t.view(),
                lut,
                &inv_norms[start..end],
                index.codec.embedding_dim(),
            ))
        }
    }
}

/// Exact asym score of one doc against a *compact* query×centroid matrix.
///
/// The batched search path never materializes the dense `[K, nq]` matrix the
/// [`ScoreQuery::ResidualLut`] arm of [`exact_doc_score`] expects — that is
/// the point of batching. Instead it packs the distinct centroid scores it
/// already computed sparsely for approximate scoring into `compact_cd_t`
/// (centroid-major: one row of `nq` scores per distinct centroid), with
/// `remap` translating real centroid ids to compact rows. The doc's codes
/// are remapped once per doc outside the SIMD kernel, which then runs
/// unchanged: the kernel only ever indexes `cd[cid * nq + qi]` and is
/// agnostic to what the ids mean.
fn exact_doc_score_asym_compact(
    index: &crate::index::MmapIndex,
    q8: &crate::binary::QueryI8,
    lut: &crate::residual_lut::ResidualLut,
    planes: Option<&crate::residual_lut::QueryPlanes>,
    compact_cd_t: &Array2<f32>,
    remap: &HashMap<i64, i64>,
    doc_id: usize,
) -> Option<f32> {
    let inv_norms = index.residual_inv_norms()?;
    let start = index.doc_offsets[doc_id];
    let end = index.doc_offsets[doc_id + 1];
    let packed = index.mmap_residuals.slice_rows(start, end);
    let codes = index.mmap_codes.slice(start, end);
    let mut remapped = Vec::with_capacity(codes.len());
    for c in codes.iter() {
        // Every shortlisted doc is a candidate, and the compact matrix covers
        // every centroid any candidate references — a miss is a logic bug.
        debug_assert!(remap.contains_key(c), "shortlist code {c} missing from centroid union");
        remapped.push(*remap.get(c)?);
    }
    Some(crate::residual_lut::maxsim_residual_lut_i8(
        q8,
        planes,
        &packed,
        &remapped,
        &compact_cd_t.view(),
        lut,
        &inv_norms[start..end],
        index.codec.embedding_dim(),
    ))
}

/// Bench-only stage-2 entry: exact MaxSim of `query` against `doc_ids`
/// through the same prepared-query paths `search` uses (float decompression,
/// asymmetric int8×LUT, or binary int8×1-bit). Includes the per-query
/// preparation a real search pays (quantization, fused-LUT build, dense
/// query×centroid matrix) but no IVF and no stage-1 — so one timed call is
/// the full per-query cost of the exact scoring stage over a fixed
/// candidate list. Not part of the search API.
#[doc(hidden)]
pub fn exact_score_docs(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    doc_ids: &[usize],
    residual_asym: bool,
) -> Vec<f32> {
    let sq = prepare_score_query(index, query, residual_asym);
    let cdot_t = if matches!(&sq, ScoreQuery::ResidualLut { .. }) {
        let _ = index.residual_inv_norms();
        // Built directly centroid-major (C × Qᵀ): same GEMM cost as the
        // [nq, K] orientation, already in the kernels' fold-friendly layout.
        Some(index.codec.centroids_view().dot(&query.t()))
    } else {
        None
    };
    // Parallel over docs like the production stage-2 candidate loop above.
    doc_ids
        .par_iter()
        .map(|&d| exact_doc_score(index, &sq, cdot_t.as_ref(), d).unwrap_or(f32::NEG_INFINITY))
        .collect()
}

/// Transpose stage-1's `[nq, K]` centroid scores into the kernels'
/// centroid-major `[K, nq]` layout, in cache-blocked passes.
///
/// `ndarray`'s `.t().as_standard_layout()` walks the destination in order,
/// so every element read jumps a full row of the source — at K = 16k that
/// is a ~64 KB stride, i.e. a cache **and** TLB miss per element, measured
/// at 2–16 ms per query on x86 (enough to eat half the fused kernels' win
/// at the end-to-end altitude). Blocking over centroids makes both sides
/// sequential: for each strip of `BLK` centroids we read `nq` runs of
/// `BLK` contiguous floats and write `BLK` runs of `nq` contiguous floats,
/// so each cache line is used fully instead of once.
fn transpose_cdot(cdot: &Array2<f32>) -> Array2<f32> {
    use crate::residual_lut::{ablation, Ablation};
    match ablation() {
        // Hand the kernels stage-1's matrix as-is; they gather from it.
        Ablation::RowMajor => return cdot.clone(),
        // The naive element-wise transpose this function replaced.
        Ablation::NaiveTranspose => return cdot.t().as_standard_layout().into_owned(),
        _ => {}
    }
    const BLK: usize = 64;
    let (nq, k) = (cdot.nrows(), cdot.ncols());
    let src = cdot.as_standard_layout();
    let src = src.as_slice().expect("cdot must be contiguous");
    let mut out = Array2::<f32>::zeros((k, nq));
    let dst = out.as_slice_mut().expect("fresh array is contiguous");
    let mut c0 = 0usize;
    while c0 < k {
        let c1 = (c0 + BLK).min(k);
        for qi in 0..nq {
            let row = &src[qi * k + c0..qi * k + c1];
            for (j, &v) in row.iter().enumerate() {
                dst[(c0 + j) * nq + qi] = v;
            }
        }
        c0 = c1;
    }
    out
}

/// Like [`exact_score_docs`], but with the dense query×centroid matrix
/// supplied by the caller in stage-1's `[nq, K]` layout — the production
/// situation, where Stage-1 already computed it for IVF probing. Used by the
/// stage-2 profiler so the LUT path is not charged twice for the centroid
/// term; the one-pass transpose to the kernels' centroid-major layout
/// happens here, the same per-query cost `search_one_mmap` pays. An empty
/// `doc_ids` makes this a pure measurement of per-query preparation (int8
/// quantization, fused-LUT + planes build, and on first call the index's
/// inv-norms cache).
#[doc(hidden)]
pub fn exact_score_docs_prepared(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    cdot: &Array2<f32>,
    doc_ids: &[usize],
    residual_asym: bool,
) -> Vec<f32> {
    let sq = prepare_score_query(index, query, residual_asym);
    let cdot_t = if matches!(&sq, ScoreQuery::ResidualLut { .. }) {
        let _ = index.residual_inv_norms();
        Some(transpose_cdot(cdot))
    } else {
        None
    };
    doc_ids
        .par_iter()
        .map(|&d| exact_doc_score(index, &sq, cdot_t.as_ref(), d).unwrap_or(f32::NEG_INFINITY))
        .collect()
}

/// Like [`exact_score_docs_prepared`], but taking the query×centroid matrix
/// already in the kernels' centroid-major `[K, nq]` layout, so the timed
/// call contains *only* scoring work. This is the profiler's exact-phase
/// entry: the `[nq, K]` → `[K, nq]` transpose is per-query preparation (the
/// production search pays it once per query, and the profiler's prep phase
/// accounts for it); leaving it inside the exact timing would add a
/// K-dependent cost to the asym column that the float column never pays.
#[doc(hidden)]
pub fn exact_score_docs_prepared_t(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    cdot_t: &Array2<f32>,
    doc_ids: &[usize],
    residual_asym: bool,
) -> Vec<f32> {
    let sq = prepare_score_query(index, query, residual_asym);
    if matches!(&sq, ScoreQuery::ResidualLut { .. }) {
        let _ = index.residual_inv_norms();
    }
    doc_ids
        .par_iter()
        .map(|&d| exact_doc_score(index, &sq, Some(cdot_t), d).unwrap_or(f32::NEG_INFINITY))
        .collect()
}

/// Wrapper for f32 to use with BinaryHeap (implements Ord)
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_score_ascending(self.0, other.0)
    }
}

fn cmp_score_ascending(a: f32, b: f32) -> Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => a.total_cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => Ordering::Equal,
    }
}

fn cmp_score_descending(a: f32, b: f32) -> Ordering {
    cmp_score_ascending(b, a)
}

fn is_score_better(candidate: f32, current: f32) -> bool {
    cmp_score_ascending(candidate, current).is_gt()
}

fn max_score(a: f32, b: f32) -> f32 {
    if is_score_better(b, a) {
        b
    } else {
        a
    }
}

/// Batched IVF probing for memory-efficient centroid scoring.
///
/// Processes centroids in chunks, keeping only top-k scores per query token in a heap.
/// Returns the union of top centroids across all query tokens.
/// If a threshold is provided, filters out centroids where max score < threshold.
fn ivf_probe_batched(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    n_probe: usize,
    batch_size: usize,
    centroid_score_threshold: Option<f32>,
) -> Vec<usize> {
    let num_centroids = centroids.nrows();
    let num_tokens = query.nrows();

    // Build batch ranges for parallel processing
    let batch_ranges: Vec<(usize, usize)> = (0..num_centroids)
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(num_centroids)))
        .collect();

    // Process centroid batches in parallel. Each rayon thread computes a GEMM
    // (with single-threaded BLAS via OPENBLAS_NUM_THREADS=1) and maintains local
    // per-token top-k heaps. Memory is bounded: rayon's thread pool ensures at most
    // num_cpus batch_scores matrices (each batch_size × num_tokens × 4 bytes) exist
    // simultaneously, same as the sequential approach where num_cpus queries each
    // process one batch at a time.
    let local_results: Vec<ProbePartial> = batch_ranges
        .par_iter()
        .map(|&(batch_start, batch_end)| {
            let mut heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
                .map(|_| BinaryHeap::with_capacity(n_probe + 1))
                .collect();
            let mut max_scores: HashMap<usize, f32> = HashMap::new();

            // Get batch view (zero-copy from mmap)
            let batch_centroids = centroids.slice_rows(batch_start, batch_end);

            // Compute scores: [num_tokens, batch_size] — single-threaded BLAS
            let batch_scores = query.dot(&batch_centroids.t());

            // Update local heaps with this batch's scores
            for (q_idx, heap) in heaps.iter_mut().enumerate() {
                for (local_c, &score) in batch_scores.row(q_idx).iter().enumerate() {
                    let global_c = batch_start + local_c;
                    let entry = (Reverse(OrdF32(score)), global_c);

                    if heap.len() < n_probe {
                        heap.push(entry);
                        max_scores
                            .entry(global_c)
                            .and_modify(|s| *s = max_score(*s, score))
                            .or_insert(score);
                    } else if let Some(&(Reverse(OrdF32(min_score)), _)) = heap.peek() {
                        if is_score_better(score, min_score) {
                            heap.pop();
                            heap.push(entry);
                            max_scores
                                .entry(global_c)
                                .and_modify(|s| *s = max_score(*s, score))
                                .or_insert(score);
                        }
                    }
                }
            }

            (heaps, max_scores)
        })
        .collect();

    // Merge local heaps into final result (lightweight: each heap has at most
    // n_probe entries, and there are num_batches heaps per token to merge)
    let mut final_heaps: Vec<BinaryHeap<(Reverse<OrdF32>, usize)>> = (0..num_tokens)
        .map(|_| BinaryHeap::with_capacity(n_probe + 1))
        .collect();
    let mut final_max_scores: HashMap<usize, f32> = HashMap::new();

    for (local_heaps, local_max_scores) in local_results {
        for (q_idx, local_heap) in local_heaps.into_iter().enumerate() {
            for entry in local_heap {
                let (Reverse(OrdF32(score)), _) = entry;
                if final_heaps[q_idx].len() < n_probe {
                    final_heaps[q_idx].push(entry);
                } else if let Some(&(Reverse(OrdF32(min_score)), _)) = final_heaps[q_idx].peek() {
                    if is_score_better(score, min_score) {
                        final_heaps[q_idx].pop();
                        final_heaps[q_idx].push(entry);
                    }
                }
            }
        }
        for (c, score) in local_max_scores {
            final_max_scores
                .entry(c)
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }
    }

    // Union top centroids across all query tokens
    let mut selected: HashSet<usize> = HashSet::new();
    for heap in final_heaps {
        for (_, c) in heap {
            selected.insert(c);
        }
    }

    // Apply centroid score threshold if set
    if let Some(threshold) = centroid_score_threshold {
        selected.retain(|c| {
            final_max_scores
                .get(c)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                >= threshold
        });
    }

    selected.into_iter().collect()
}

/// Build sparse centroid scores for a set of centroid IDs.
///
/// Returns a HashMap mapping centroid_id -> query scores array.
fn build_sparse_centroid_scores(
    query: &Array2<f32>,
    centroids: &CentroidStore,
    centroid_ids: &HashSet<usize>,
) -> HashMap<usize, Array1<f32>> {
    centroid_ids
        .iter()
        .map(|&c| {
            let centroid = centroids.row(c);
            let scores: Array1<f32> = query.dot(&centroid);
            (c, scores)
        })
        .collect()
}

/// Compute approximate scores using sparse centroid score lookup.
fn approximate_score_sparse(
    sparse_scores: &HashMap<usize, Array1<f32>>,
    doc_codes: &[usize],
    num_query_tokens: usize,
) -> f32 {
    let mut score = 0.0;

    // For each query token
    for q_idx in 0..num_query_tokens {
        let mut max_score = f32::NEG_INFINITY;

        // For each document token's code
        for &code in doc_codes.iter() {
            if let Some(centroid_scores) = sparse_scores.get(&code) {
                let centroid_score = centroid_scores[q_idx];
                if centroid_score > max_score {
                    max_score = centroid_score;
                }
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
        }
    }

    score
}

/// Compute approximate scores for mmap index using code lookups.
fn approximate_score_mmap(query_centroid_scores: &Array2<f32>, doc_codes: &[i64]) -> f32 {
    let mut score = 0.0;

    for q_idx in 0..query_centroid_scores.nrows() {
        let mut max_score = f32::NEG_INFINITY;

        for &code in doc_codes.iter() {
            let centroid_score = query_centroid_scores[[q_idx, code as usize]];
            if centroid_score > max_score {
                max_score = centroid_score;
            }
        }

        if max_score > f32::NEG_INFINITY {
            score += max_score;
        }
    }

    score
}

/// Search a memory-mapped index for a single query.
pub fn search_one_mmap(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_centroids = index.codec.num_centroids();

    // Decide whether to use batched mode for memory efficiency
    let use_batched = params.centroid_batch_size > 0 && num_centroids > params.centroid_batch_size;

    if use_batched {
        // Batched path: memory-efficient IVF probing for large centroid counts
        return search_one_mmap_batched(index, query, params, subset);
    }

    let (query_centroid_scores, to_decompress) = stage1_shortlist(index, query, params, subset)?;

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores. Binary indexes score against an int8 query; the
    // full-precision query is used for the float (residual) path.
    // Chunked processing limits concurrent memory from parallel decompression.
    let exact_query = prepare_score_query(index, query, params.residual_asym);
    let cdot_t = if matches!(&exact_query, ScoreQuery::ResidualLut { .. }) {
        // Build the per-token norm cache once, outside the parallel loop.
        let _ = index.residual_inv_norms();
        // One transpose pass per query: stage-1 needs [nq, K] row-major for
        // per-token probing, the exact kernels want centroid-major [K, nq]
        // so a token's scores across query rows are one contiguous strip
        // (vectorized fold + no K-strided gather per row).
        Some(transpose_cdot(&query_centroid_scores))
    } else {
        None
    };
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_chunks(DECOMPRESS_CHUNK_SIZE)
        .flat_map(|chunk| {
            chunk
                .iter()
                .filter_map(|&doc_id| {
                    let score =
                        exact_doc_score(index, &exact_query, cdot_t.as_ref(), doc_id as usize)?;
                    Some((doc_id, score))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Stage 1 of the standard (non-batched) search: dense query×centroid scores,
/// per-token IVF cell selection, candidate gathering, approximate codes-only
/// scoring, and pruning down to the exact-scoring shortlist. Everything a
/// query pays *before* the Stage-2 kernels take over.
///
/// Returns the dense query×centroid matrix (reused by Stage-2 for the LUT
/// path's centroid term) and the pruned candidate list, which is empty when
/// nothing survives probing/filtering.
///
/// This is the production path — `search_one_mmap` calls it — exposed
/// (hidden) so the stage-2 profiler measures the identical shortlist.
#[doc(hidden)]
pub fn stage1_shortlist(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<(Array2<f32>, Vec<i64>)> {
    let num_centroids = index.codec.num_centroids();
    let num_query_tokens = query.nrows();

    // Standard path: compute full query-centroid scores upfront
    let query_centroid_scores = query.dot(&index.codec.centroids_view().t());

    // When subset is provided, pre-compute eligible centroids: only those containing
    // at least one embedding from a subset document. Centroids without subset docs
    // can't contribute candidates, so skipping them is a pure optimization.
    let eligible_centroids: Option<HashSet<usize>> = subset.map(|subset_docs| {
        let mut centroids = HashSet::new();
        for &doc_id in subset_docs {
            let doc_idx = doc_id as usize;
            if doc_idx < index.doc_lengths.len() {
                let start = index.doc_offsets[doc_idx];
                let end = index.doc_offsets[doc_idx + 1];
                let codes = index.mmap_codes.slice(start, end);
                for &c in codes.iter() {
                    centroids.insert(c as usize);
                }
            }
        }
        centroids
    });

    // When pre-filtering, scale n_ivf_probe by the document ratio to compensate
    // for candidates lost to filtering. If 50% of docs are filtered out, we probe
    // ~2x more centroids to find enough relevant candidates.
    // No filter: n_ivf_probe unchanged.
    let effective_n_ivf_probe = match (&eligible_centroids, subset) {
        (Some(eligible), Some(subset_docs)) if !eligible.is_empty() => {
            let num_docs = index.doc_lengths.len();
            let subset_len = subset_docs.len();
            let scaled = if subset_len > 0 {
                (params.n_ivf_probe as u64 * num_docs as u64 / subset_len as u64) as usize
            } else {
                params.n_ivf_probe
            };
            scaled.max(params.n_ivf_probe).min(eligible.len())
        }
        _ => params.n_ivf_probe,
    };

    // Find top IVF cells to probe using per-token top-k selection.
    // When pre-filtering, only score eligible centroids (same selection logic,
    // smaller pool). This can only improve recall for subset docs since
    // ineligible centroids would have wasted probe slots.
    let cells_to_probe: Vec<usize> = {
        let mut selected_centroids = HashSet::new();

        for q_idx in 0..num_query_tokens {
            let mut centroid_scores: Vec<(usize, f32)> = match &eligible_centroids {
                Some(eligible) => eligible
                    .iter()
                    .map(|&c| (c, query_centroid_scores[[q_idx, c]]))
                    .collect(),
                None => (0..num_centroids)
                    .map(|c| (c, query_centroid_scores[[q_idx, c]]))
                    .collect(),
            };

            // Partial selection: O(K) average instead of O(K log K) for full sort
            // After this, the top n elements are in positions 0..n
            // (but not sorted among themselves - which is fine since we use a HashSet)
            let n_probe = effective_n_ivf_probe.min(centroid_scores.len());
            if centroid_scores.len() > n_probe {
                centroid_scores
                    .select_nth_unstable_by(n_probe - 1, |a, b| cmp_score_descending(a.1, b.1));
            }

            for (c, _) in centroid_scores.iter().take(n_probe) {
                selected_centroids.insert(*c);
            }
        }

        // Apply centroid score threshold: filter out centroids where max score < threshold
        if let Some(threshold) = params.centroid_score_threshold {
            selected_centroids.retain(|&c| {
                let max_score: f32 = (0..num_query_tokens)
                    .map(|q_idx| query_centroid_scores[[q_idx, c]])
                    .max_by(|a, b| cmp_score_ascending(*a, *b))
                    .unwrap_or(f32::NEG_INFINITY);
                max_score >= threshold
            });
        }

        selected_centroids.into_iter().collect()
    };

    // Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok((query_centroid_scores, vec![]));
    }

    // Compute approximate scores
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let score = approximate_score_mmap(&query_centroid_scores, &codes);
            (doc_id, score)
        })
        .collect();

    // Sort by approximate score and take top candidates
    approx_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

    Ok((query_centroid_scores, to_decompress))
}

/// Memory-efficient batched search for MmapIndex with large centroid counts.
///
/// Uses batched IVF probing and sparse centroid scoring to minimize memory usage.
fn search_one_mmap_batched(
    index: &crate::index::MmapIndex,
    query: &Array2<f32>,
    params: &SearchParameters,
    subset: Option<&[i64]>,
) -> Result<QueryResult> {
    let num_query_tokens = query.nrows();

    // Step 1: Batched IVF probing
    let cells_to_probe = ivf_probe_batched(
        query,
        &index.codec.centroids,
        params.n_ivf_probe,
        params.centroid_batch_size,
        params.centroid_score_threshold,
    );

    // Step 2: Get candidate documents from IVF
    let mut candidates = index.get_candidates(&cells_to_probe);

    // Filter by subset if provided
    if let Some(subset_docs) = subset {
        let subset_set: HashSet<i64> = subset_docs.iter().copied().collect();
        candidates.retain(|&c| subset_set.contains(&c));
    }

    if candidates.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Step 3: Collect unique centroids from all candidate documents
    let mut unique_centroids: HashSet<usize> = HashSet::new();
    for &doc_id in &candidates {
        let start = index.doc_offsets[doc_id as usize];
        let end = index.doc_offsets[doc_id as usize + 1];
        let codes = index.mmap_codes.slice(start, end);
        for &code in codes.iter() {
            unique_centroids.insert(code as usize);
        }
    }

    // Step 4: Build sparse centroid scores
    let sparse_scores =
        build_sparse_centroid_scores(query, &index.codec.centroids, &unique_centroids);

    // Step 5: Compute approximate scores using sparse lookup
    let mut approx_scores: Vec<(i64, f32)> = candidates
        .par_iter()
        .map(|&doc_id| {
            let start = index.doc_offsets[doc_id as usize];
            let end = index.doc_offsets[doc_id as usize + 1];
            let codes = index.mmap_codes.slice(start, end);
            let doc_codes: Vec<usize> = codes.iter().map(|&c| c as usize).collect();
            let score = approximate_score_sparse(&sparse_scores, &doc_codes, num_query_tokens);
            (doc_id, score)
        })
        .collect();

    // Sort by approximate score and take top candidates
    approx_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));
    let top_candidates: Vec<i64> = approx_scores
        .iter()
        .take(params.n_full_scores)
        .map(|(id, _)| *id)
        .collect();

    // Further reduce for full decompression
    let n_decompress = (params.n_full_scores / 4).max(params.top_k);
    let to_decompress: Vec<i64> = top_candidates.into_iter().take(n_decompress).collect();

    if to_decompress.is_empty() {
        return Ok(QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    // Compute exact scores. Binary indexes score against an int8 query; the
    // full-precision query is used for the float (residual) path.
    // Chunked processing limits concurrent memory from parallel decompression.
    //
    // The asymmetric residual arm needs query×centroid scores for its
    // centroid term. This path deliberately never builds the dense matrix —
    // that is the point of batching — but the sparse centroid scores
    // computed for approximate scoring already cover every centroid any
    // candidate references, a superset of the shortlist's. Packing them
    // into a compact centroid-major [distinct, nq] matrix (one contiguous
    // row per centroid, the kernels' fold layout) plus a per-doc code remap
    // feeds the same fused kernels the dense path uses.
    let exact_query = prepare_score_query(index, query, params.residual_asym);
    let asym_compact = if matches!(&exact_query, ScoreQuery::ResidualLut { .. }) {
        // Build the per-token norm cache once, outside the parallel loop.
        let _ = index.residual_inv_norms();
        let mut ids: Vec<usize> = sparse_scores.keys().copied().collect();
        ids.sort_unstable();
        let remap: HashMap<i64, i64> = ids
            .iter()
            .enumerate()
            .map(|(row, &c)| (c as i64, row as i64))
            .collect();
        // Centroid-major, matching the dense path — except under the
        // row-major ablation, which reproduces the pre-work layout on both
        // paths so the two stay comparable.
        let compact = if crate::residual_lut::ablation()
            == crate::residual_lut::Ablation::RowMajor
        {
            let mut c = Array2::<f32>::zeros((num_query_tokens, ids.len()));
            for (col, &cid) in ids.iter().enumerate() {
                c.column_mut(col).assign(&sparse_scores[&cid]);
            }
            c
        } else {
            let mut c = Array2::<f32>::zeros((ids.len(), num_query_tokens));
            for (row, &cid) in ids.iter().enumerate() {
                c.row_mut(row).assign(&sparse_scores[&cid]);
            }
            c
        };
        Some((compact, remap))
    } else {
        None
    };
    let mut exact_scores: Vec<(i64, f32)> = to_decompress
        .par_chunks(DECOMPRESS_CHUNK_SIZE)
        .flat_map(|chunk| {
            chunk
                .iter()
                .filter_map(|&doc_id| {
                    let score = match (&exact_query, &asym_compact) {
                        (ScoreQuery::ResidualLut { q8, lut, planes }, Some((cd, remap))) => {
                            exact_doc_score_asym_compact(
                                index,
                                q8,
                                lut,
                                planes.as_ref(),
                                cd,
                                remap,
                                doc_id as usize,
                            )
                        }
                        _ => exact_doc_score(index, &exact_query, None, doc_id as usize),
                    }?;
                    Some((doc_id, score))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sort by exact score
    exact_scores.sort_by(|a, b| cmp_score_descending(a.1, b.1));

    // Return top-k results
    let result_count = params.top_k.min(exact_scores.len());
    let passage_ids: Vec<i64> = exact_scores
        .iter()
        .take(result_count)
        .map(|(id, _)| *id)
        .collect();
    let scores: Vec<f32> = exact_scores
        .iter()
        .take(result_count)
        .map(|(_, s)| *s)
        .collect();

    Ok(QueryResult {
        query_id: 0,
        passage_ids,
        scores,
    })
}

/// Search a memory-mapped index for multiple queries.
pub fn search_many_mmap(
    index: &crate::index::MmapIndex,
    queries: &[Array2<f32>],
    params: &SearchParameters,
    parallel: bool,
    subset: Option<&[i64]>,
) -> Result<Vec<QueryResult>> {
    if parallel {
        let results: Vec<QueryResult> = queries
            .par_iter()
            .enumerate()
            .map(|(i, query)| {
                let mut result =
                    search_one_mmap(index, query, params, subset).unwrap_or_else(|_| QueryResult {
                        query_id: i,
                        passage_ids: vec![],
                        scores: vec![],
                    });
                result.query_id = i;
                result
            })
            .collect();
        Ok(results)
    } else {
        let mut results = Vec::with_capacity(queries.len());
        for (i, query) in queries.iter().enumerate() {
            let mut result = search_one_mmap(index, query, params, subset)?;
            result.query_id = i;
            results.push(result);
        }
        Ok(results)
    }
}

/// Alias type for search result (for API compatibility)
pub type SearchResult = QueryResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colbert_score() {
        // Query with 2 tokens, dim 4
        let query =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

        // Document with 3 tokens
        let doc = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.5, 0.5, 0.0, 0.0, // sim with q0: 0.5, sim with q1: 0.5
                0.8, 0.2, 0.0, 0.0, // sim with q0: 0.8, sim with q1: 0.2
                0.0, 0.9, 0.1, 0.0, // sim with q0: 0.0, sim with q1: 0.9
            ],
        )
        .unwrap();

        let score = colbert_score(&query.view(), &doc.view());
        // q0 max: 0.8 (from token 1), q1 max: 0.9 (from token 2)
        // Total: 0.8 + 0.9 = 1.7
        assert!((score - 1.7).abs() < 1e-5);
    }

    #[test]
    fn test_search_params_default() {
        let params = SearchParameters::default();
        assert_eq!(params.batch_size, 2000);
        assert_eq!(params.n_full_scores, 4096);
        assert_eq!(params.top_k, 10);
        assert_eq!(params.n_ivf_probe, 8);
        assert_eq!(params.centroid_score_threshold, Some(0.4));
    }

    #[test]
    fn test_cmp_score_descending_places_non_finite_scores_last() {
        let mut scores = [1.0f32, f32::INFINITY, 0.5, f32::NAN];
        scores.sort_by(|a, b| cmp_score_descending(*a, *b));

        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 0.5);
        assert!(!scores[2].is_finite());
        assert!(!scores[3].is_finite());
    }

    #[test]
    fn test_score_replacement_treats_finite_values_as_better_than_non_finite() {
        assert!(is_score_better(1.0, f32::NAN));
        assert!(is_score_better(1.0, f32::INFINITY));
        assert!(!is_score_better(f32::NAN, 1.0));
        assert!(!is_score_better(f32::INFINITY, 1.0));
    }

    #[test]
    fn test_max_score_keeps_finite_value_over_non_finite_value() {
        assert_eq!(max_score(f32::NAN, 1.0), 1.0);
        assert_eq!(max_score(1.0, f32::NAN), 1.0);
        assert_eq!(max_score(f32::INFINITY, 1.0), 1.0);
        assert_eq!(max_score(1.0, f32::INFINITY), 1.0);
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::*;

    /// The blocked transpose must equal the naive one for every shape — in
    /// particular for K spanning many blocks (the production case; the
    /// search integration tests all use toy indexes whose K never exceeds
    /// one block) and for K not a multiple of the block size.
    #[test]
    fn blocked_transpose_matches_naive() {
        // `row_major` deliberately hands the matrix through untransposed,
        // and `naive_transpose` selects the implementation this test is the
        // reference for; both make the assertion below meaningless.
        if crate::residual_lut::ablation() != crate::residual_lut::Ablation::Off {
            return;
        }
        for &nq in &[1usize, 3, 32] {
            for &k in &[1usize, 7, 64, 65, 200, 4096] {
                let a = Array2::<f32>::from_shape_fn((nq, k), |(q, c)| (q * 7919 + c) as f32);
                let got = transpose_cdot(&a);
                assert_eq!(got.dim(), (k, nq), "nq={nq} k={k}");
                for q in 0..nq {
                    for c in 0..k {
                        assert_eq!(got[[c, q]], a[[q, c]], "nq={nq} k={k} at ({q},{c})");
                    }
                }
            }
        }
    }
}
