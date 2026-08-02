# Upstream PR body — READY (rebased on v1.6.5, 2026-08-01)

Branch: `joe32140:feat/asymmetric-residual-lut` → `lightonai:main`.
OPENED 2026-08-01: https://github.com/lightonai/next-plaid/pull/169
#155 squash-merged as dd3ab5c; the CR is rebased onto origin/main
76092e1 (Release 1.6.5) with zero conflicts. Gates re-run on the rebased
tree 2026-08-01: clippy native ✓ (0), clippy x86_64-apple-darwin ✓ (0),
`RUSTDOCFLAGS="-D warnings" cargo doc` ✓, full test suite ✓ (146 lib +
integration, arm64 native). Stage-1 perf follow-up is stacked as
`perf/stage1-pipeline` (opens as draft referencing this PR).

Open with (the PR body is everything below the `---`):

```bash
gh pr create --repo lightonai/next-plaid --base main --head joe32140:feat/asymmetric-residual-lut --title "feat: asymmetric residual scoring — int8 query × fused LUT MaxSim, 5–8× faster rescoring, identical NDCG@10" --body-file <(sed '1,/^---$/d' /Users/joe/next-plaid-lut/notes/lut-cr-pr-body.md)
```

---

## What

Optional asymmetric scoring for **residual** indexes
(`SearchParameters::residual_asym`, default off): Stage-2 scores the
*stored* codes directly — int8 query × a fused byte→int8-weights table,
plus the centroid term Stage-1 already computed — instead of
decompressing every candidate token to f32 and running a BLAS MaxSim.
Compute-only: same index, same storage, so the two modes can be A/B'd
per search. The residual-codec counterpart of #155's int8 × binary
scoring. The stage-1 half of the same campaign is stacked on top of this
PR as #170.

## How

Scoring splits the dot product exactly:

```
q · token = q · centroid[cid]                    (from the IVF probe matrix)
          + Σ_d q_d · bucket_weights[code_d]     (int8 × int8, integer MACs)
```

then applies the float path's own per-token renormalize via a cached
`1/‖centroid + residual‖` (computed once per index; skipping it measures
up to −0.17 NDCG@10 at nbits=1, which is why it exists).

For byte-aligned dims ≤ 256, fused doc-token-outer kernels expand each
token's packed bytes once in registers — the 256-entry byte→weights
table provably factors into per-key-position 16-entry nibble tables
(verified over all 256 byte values at build, scalar fallback if the
packing ever changes), the shape NEON `tbl` / SSE `pshufb` consume —
amortized over all query rows:

- **NEON** `tbl` expand + SDOT, epilogue folds 4 query rows per
  `vmaxq_f32` with a `vpaddq` transpose-reduce
- **AVX2** `pshufb` expand + `maddubs`/`madd`, 8-wide fold
- **AVX-512 VNNI** `vpdpbusd` (sign carried via `movepi8_mask` +
  `mask_sub_epi8`; exact because both operands clamp to ±127), 16-wide
  fold

The epilogue reads the centroid term from a **centroid-major** `[K, nq]`
matrix — Stage-1's `[nq, K]` matrix transposed once per query with a
cache-blocked pass — so one doc token touches one contiguous strip
instead of gathering `nq` floats a row apart. (A controlled per-component
ablation showed this layout, not the SIMD epilogue, is the dominant win
on every CPU tested.)

All paths compute the identical integer accumulator and the identical
float epilogue expression: the parity suite asserts **bit-equality** with
the scalar reference across nq × nbits × dim — each arch kernel called
directly, plus the dispatcher — and a semantics test pins normalized
scoring to the `decompress` reference. The batched-centroid path
(num_centroids > centroid_batch_size) packs its sparse centroid scores
into a compact centroid-major matrix with a per-doc code remap, so asym
scoring survives large-K indexes unchanged (regression-tested against the
dense path).

## Measured

- **Quality — NDCG@10 is unchanged**: |Δ| ≤ 0.002 vs the float path on
  identical codes, and identical to three decimals at deployed settings —
  3 ColBERT checkpoints × 3 BEIR corpora × nbits 4/2/1, incl. long-query
  ArguAna. The int8 error lands only on the residual correction; the
  dominant centroid term stays float. Spot-checked again on real GTE
  embeddings at deployed settings (seed-42 builds, float vs asym on
  identical indexes):

  | dataset | r=4 | r=2 | r=1 |
  |---|---|---|---|
  | NFCorpus | 0.3809 / 0.3811 | 0.3779 / 0.3779 | 0.3701 / 0.3705 |
  | SciFact | 0.7609 / 0.7607 | 0.7507 / 0.7507 | 0.7427 / 0.7427 |

  Max |Δ| across the six cells: 0.0004.
- **Latency** — the rescore stage (exact scoring of ~1024-doc
  shortlists, real corpus shapes), fused asym vs the float
  decompress+GEMM path on identical indexes. The CI rows come from one
  interleaved run of the same tree with `residual_asym` off vs on —
  same stage-1, same shortlists, so the ratio isolates this PR's
  scheme change. 9 cells per platform (3 datasets × nbits 4/2/1);
  geomean, with the per-cell range in parentheses:

  | platform | fused vs float rescore | decompress share of the float rescore |
  |---|---|---|
  | x86 AVX2 (GitHub CI) | **6.1×** (5.2–7.8×) | 74–83% |
  | Neoverse (GitHub CI) | **6.3×** (5.3–8.2×) | 67–76% |
  | macOS arm64 (GitHub CI) | **5.1×** (4.3–5.7×) | 71–80% |
  | Apple M4 (native, 0.5–7M-token corpus ladder) | **5.2×** (4.7–5.8×) | 48–72% |

  The right column is the mechanism: the fused kernels win by skipping
  decompression, not by out-multiplying GEMM — and it also bounds the
  win by whatever share decompression holds in a given deployment (the
  share falls as corpora grow: 72% → 48% across the M4 ladder).
- **End-to-end**: this PR alone moves whole-query latency 1.4–5.2×
  depending on corpus size — stage-1 dominates at scale, and that is
  the stacked follow-up's job. Combined with the stage-1 rework
  (#170), the same interleaved A/B against v1.6.5 measures **5.6–7.1×
  end-to-end geomean** (same 9 cells per platform) on x86 AVX2,
  Neoverse, and macOS arm64 CI, and **6.3×** on a native M4 — the
  combined number belongs to the pair, not to this PR alone.
- **AVX-512 honesty note**: the VNNI kernel is written, feature-gated,
  and covered by the parity suite *on VNNI hardware*, but GitHub's
  standard runners don't have VNNI — correctness-validated, no perf
  claim.

Measurement details, per-component ablations, and the bench harness live
on the research branch `feat/asymmetric-lut-residual`.

## Prior art

The mechanism — scoring quantized codes in registers instead of
decompressing them — is the asymmetric-distance-computation lineage:
FAISS FastScan / [André et al., VLDB 2016](http://www.vldb.org/pvldb/vol9/p288-andre.pdf)
established the SIMD-register nibble-LUT (`pshufb`) form for PQ, and
[WARP (SIGIR 2025)](https://arxiv.org/abs/2501.17788) applied "implicit
decompression" to late-interaction retrieval on CPU for XTR's PQ-style
codec. This PR is the counterpart for **PLAID's per-dimension scalar
residual codec**: rather than a precomputed-distance table (which fits
PQ subspace codebooks), it expands codes to int8 *weights* in registers
and runs an explicit int8 dot (SDOT / `maddubs` / `vpdpbusd`), which is
what lets it reuse Stage-1's centroid term exactly and stay bit-compatible
with the float path's renormalized semantics.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
