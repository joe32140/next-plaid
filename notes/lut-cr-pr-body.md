# Upstream PR body — ready for when #155 merges

Branch: `joe32140:feat/asymmetric-residual-lut` → `lightonai:main`
(rebase onto main once #155 lands; the branch is stacked directly on it,
so the rebase should be clean or empty).

Open with (the PR body is everything below the `---`; extract it to a
temp file first, or paste it into the GitHub UI):

```bash
cd /Users/joe/next-plaid-cr && git fetch origin && git rebase origin/main && git push --force-with-lease fork feat/asymmetric-residual-lut && gh pr create --repo lightonai/next-plaid --base main --head joe32140:feat/asymmetric-residual-lut --title "feat: asymmetric residual scoring — int8 query × fused LUT with SIMD MaxSim kernels" --body-file <(sed '1,/^---$/d' /Users/joe/next-plaid-lut/notes/lut-cr-pr-body.md)
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
scoring.

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

- **Quality**: |ΔNDCG@10| ≤ 0.002 vs the float path on identical codes —
  3 ColBERT checkpoints × 3 BEIR corpora × nbits 4/2/1, incl. long-query
  ArguAna. The int8 error lands only on the residual correction; the
  dominant centroid term stays float.
- **Latency** (exact-scoring stage, 1024-doc shortlists, real corpus
  shapes, x86 AVX2 / Neoverse / Apple M4): 4.7–8.4× vs decompress+GEMM
  at the kernel level; 1.4–5.2× end-to-end depending on corpus size
  (Stage-1 dominates at scale). Decompression is 48–84% of the float
  exact path (platform-dependent) — the fused kernels win by skipping
  it, not by out-multiplying GEMM.
- **AVX-512 honesty note**: the VNNI kernel is written, feature-gated,
  and covered by the parity suite *on VNNI hardware*, but GitHub's
  standard runners don't have VNNI — correctness-validated, no perf
  claim.

Measurement details, per-component ablations, and the bench harness live
on the research branch `feat/asymmetric-lut-residual`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
