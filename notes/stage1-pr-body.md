# Draft PR body — stage-1 pipeline (stacked on the asym-LUT PR)

Branch: `joe32140:perf/stage1-pipeline` → `lightonai:main`, opened as
**draft**. Replace `#ASYM_PR` with the real PR number before opening.

Title: `perf: rework the stage-1 shortlist pipeline; per-doc parallel exact scoring`

---

> **Stacked on #ASYM_PR** — the first two commits are that PR; only the
> last two commits (`perf(search): rework the stage-1 shortlist pipeline`
> and `perf(search): per-doc parallel exact scoring`) are under review
> here. Once #ASYM_PR merges this rebases to a `search.rs` + `index.rs`
> change only (+514/−82).

## What

Stage 1 — everything a query pays before exact scoring — rebuilt phase by
phase, plus per-doc parallelism in the exact-scoring loop. Default-on: no
new parameters, no storage or API change. Every phase change ships with an
equivalence test against the path it replaces.

- **centroid GEMM**: column-block-parallel; the dim-long reduction stays
  inside one block, so the output is **bit-identical** to the single
  `dot` call (tested).
- **IVF probe**: running-threshold chunk scan replaces a K-length buffer
  fill + `select_nth` per query token — one sequential read per row, and
  chunks that can't beat the current k-th value are skipped on one
  predictable branch. Same top-k set by value (tested under ties).
- **candidate gather**: doc ids are dense, so a word bitmap dedups in
  O(postings) and scanning its set bits emits the same sorted list the
  old sort+dedup produced, without the sort.
- **approximate flood**: centroid scores are quantized to u8 in a fused
  transpose+quantize pass (PLAID's own centroid-interaction trick) and
  scored chunk-outer with a register-resident 16-lane max accumulator.
  The quantization is monotone per lane (u8 max = f32 max); the flood
  only ranks a 4096-deep prune cut. Gated on quality, not just kernel
  math: deployed nDCG@10 identical **to four decimals** vs exact
  flooding on real-embedding corpora (NFCorpus + SciFact, r4/r2/r1 and
  binary).
- **prune**: partial-select the surviving `n_full_scores` before sorting
  only that prefix; identical top list and order.
- **exact scoring**: plain per-doc `par_iter` replaces fixed 128-doc
  chunking. The chunking's memory rationale no longer holds (in-flight
  decompressed docs are bounded by the thread count either way) and 8
  chunks at n_decompress = 1024 underfilled the pool — 6.1 effective
  threads measured on a 10-core M4; 9.4 after.

When the asym arm (#ASYM_PR) is active, the flood's fused pass also emits
the f32 centroid-major matrix stage 2 needs, so one traversal serves both
stages.

## Measured

- Stage 1, Apple M4 native, fiqa 52k docs / 7M tokens: **6.70 → 1.66 ms**
  per query (4.0×). Replicated on GitHub CI: stage-1 sums drop 6–12× on
  x86 AVX2, Neoverse, and macOS arm64 — server parts pay more for the
  scatter this removes than Apple's L2 does, so the M4 number is the
  conservative one.
- End-to-end vs v1.6.5 (interleaved same-box A/B, identical prebuilt
  indexes, 3 datasets × nbits 4/2/1 per platform): **1.2–1.5× geomean
  with float stage-2** (this PR alone); **5.6–7.1× geomean with
  `residual_asym` on** (this PR + #ASYM_PR), 6.3× on native M4.
- The biggest per-phase ratio (probe, 9.5×) is deliberately not the
  headline: phase speedups compose by time share, and the flood + GEMM
  dominated. Per-phase tables and the full exploration ledger (including
  the ideas measured and killed) live on the research branch
  `feat/asymmetric-lut-residual`.

## Notes for review

- The replaced row-major flood scorers are retained under `#[cfg(test)]`
  as the reference implementations for the bit-identity tests.
- NaN behavior in the probe scan matches `cmp_score_descending`'s
  NaN-last order (a NaN never wins a `v > thr` comparison).
- The u32 code side-array costs 4 B/token of RAM, built lazily on first
  search — same pattern as the existing `residual_inv_norms` cache.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
