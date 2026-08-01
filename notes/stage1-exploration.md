# Stage-1 exploration night — 2026-07-31

Goal (Joe, before sleep): further stage-1 performance via principled,
cross-platform changes suitable as defaults. Log every idea: hypothesis →
measurement → verdict. Finalize the winning combination, clean up, review.

Protocol: cached CI indexes (`~/beir-data/quant_grid/ci_indexes/flat`),
arm64-verified binaries, idle M4, per-query S1PHASE medians with first 3
queries dropped (page-cache warmup). Cross-platform confirmation on fork CI
(x86 + Neoverse) before anything becomes a default — M4 is the
scatter-forgiving platform.

## Baseline after the q8-flood + mechanics commits (a38c318, 4ba7b62, f8ffc93)

M4, median per query:

| cell | total | cdot | probe | gather | approx | sort |
|---|---|---|---|---|---|---|
| scifact r4 (K=16384) | 3.20 ms | 1.32 (41%) | 0.60 (19%) | 0.18 (6%) | 1.10 (34%) | 0.07 (2%) |
| scifact binary | 3.11 ms | 1.27 (41%) | 0.60 (19%) | 0.17 (6%) | 1.11 (36%) | 0.07 (2%) |

(fiqa 15k/52k rerunning — first attempt lost to zsh non-word-splitting.)

The flood is no longer dominant. The cdot GEMM (`query.dot(centroids.t())`,
single-threaded matrixmultiply) is now the top phase, probe second.

## Idea ledger

### E3 — dedup per-doc codes before flood: KILLED (measured)
Hypothesis: clustering maps repeated subwords to the same centroid, so
per-doc code lists contain duplicates; max is idempotent, so a deduped
side array cuts flood tokens losslessly.
Measurement: dup rate = **1.0%** scifact, **4.1%** fiqa-52k.
Verdict: not worth a second 4 B/token side array for ≤4% fewer flood
iterations. Killed before writing any Rust. (Untested residue: sorting
codes per doc for cdot_t locality — only relevant if CI shows the flood
scatter-bound on Neoverse.)

### E2 — parallelize the cdot GEMM over centroid-column blocks: IN PROGRESS
Hypothesis: cdot is 41% of stage-1 and single-threaded; `[nq,dim]·[dim,K]`
splits perfectly over K-column blocks into disjoint output slices; rayon is
already a dependency and stage-1's flood already uses intra-query
parallelism (upstream's own design), so this is consistent with the
existing contract. The 128-long reduction happens within one block, so
per-element summation order — and therefore every output bit — is
unchanged.
Expect: 3–6× on the phase on 8 idle cores; less under server saturation
(work-stealing degrades gracefully).

### E1 — probe top-k via chunk-max skip scan: PLANNED
Hypothesis: per token we currently fill a K-length index buffer (128 KB
writes at K=32k) and `select_nth_unstable_by` with indirect comparisons —
0.6 ms/query at K=16k. A running-threshold scan (track the current 8th-best
value; per 16-wide chunk compute chunk-max — autovectorizes to a max
reduction — and skip the chunk unless chunk_max beats the threshold) reads
each row once sequentially and almost never enters the scalar path.
Same top-k set by value; tie choice arbitrary, as it already is with
`select_nth_unstable`.

### E4 — sort cells_to_probe before gather: PLANNED
`selected_centroids` is a HashSet; iteration order is arbitrary, so
`get_candidates` walks IVF postings in random cell order. Sorting ~100–200
cell ids is free and makes the postings reads sequential in the ivf array.

### E5 — parallelize transpose_quantize blocks: PLANNED (if E2 lands)
Same disjoint-block argument as E2; each c-block owns contiguous dst
ranges in both outputs.

## Round 1 measurements (E1 + E2 + E4 together), M4

| cell | total | cdot | probe | gather | approx (tq/flood) | vs baseline |
|---|---|---|---|---|---|---|
| scifact r4 | 1.89 ms | 0.43 | 0.09 | 0.19 | 1.09 (0.17/0.92) | **1.69×** |
| fiqa52k r4 | 3.90 ms | 0.66 | 0.11 | 0.42 | 2.60 (0.32/2.29) | **1.72×** |
| fiqa52k binary | 3.60 ms | 0.58 | 0.11 | 0.39 | 2.44 (0.28/2.16) | 1.68× |

- **E2 (parallel cdot): KEEP** — 3.1–3.8× on the phase, bit-identical
  (tested). Diminishing vs core count as expected at this matrix size.
- **E1 (probe scan): KEEP** — 6.7–9.5× on the phase (0.6→0.09, 1.05→0.11).
  The chunk-skip almost never enters its scalar path, as hypothesized.
  Equivalence test: top-k value multiset identical to select_nth under ties.
- **E4 (sorted cells): NEUTRAL on M4** (gather 0.39→0.42, noise). M4's L2
  absorbs the hash-order scatter; keep it pending the Neoverse read —
  cost is a sort of ~200 ids.
- New picture: the flood proper is now 58–68% of stage-1. tq (transpose+
  quantize prep) is only 0.15–0.32 ms — not worth parallelizing yet (E5
  deferred).

### E7 — register-resident flood accumulator (chunk-outer loop): IN PROGRESS
Hypothesis: the q8 flood spends most of its per-token work outside the two
`umax`es — acc load+store every token and a dynamic-length vector loop's
prologue/epilogue (nq is a runtime value, so LLVM keeps trip checks).
Restructure: pad the quantized matrix rows to a 16-multiple stride (pad
bytes quantize as 0 = lo, max-neutral, and add 0 to the final sum — both
invariants free), then loop chunks-outer / tokens-inner with a `[u8; 16]`
accumulator that LLVM promotes to one vector register. Per token per chunk:
load code, load 16 B row segment, one `umax` — no acc traffic, no tail
handling. Codes are re-read once per chunk (≤2 passes at nq ≤ 32, L1-hot).
Cross-platform safe Rust; autovectorizes to `umax.16b` / `pmaxub`.

## Round 2 measurement (E7), M4

| cell | total | cdot | probe | gather | tq | flood | vs baseline |
|---|---|---|---|---|---|---|---|
| scifact r4 | 1.21 ms | 0.41 | 0.09 | 0.18 | 0.19 | 0.23 | **2.6×** |
| fiqa52k r4 | 2.24 ms | 0.67 | 0.11 | 0.41 | 0.35 | 0.63 | **3.0×** |
| fiqa52k binary | 2.00 ms | 0.59 | 0.11 | 0.39 | 0.28 | 0.54 | 3.0× |

**E7: KEEP — the night's biggest single win.** Flood 2.29 → 0.63 ms (3.6×)
at fiqa-52k. Disassembly confirms exactly one `umax.16b` in the token loop
(the register accumulator): the old per-token acc load/store + dynamic-trip
epilogue was ~2/3 of the flood's cost, not the maxes themselves.

Profile now flat: cdot 30%, tq+flood 44%, gather 18%. Round 3 adds
E8 (bitmap gather — emits the identical sorted, deduped candidate list
without sorting the concatenated postings) and E5 (parallel transpose+
quantize prep, same disjoint-block argument as E2).

## Round 3 measurement (E5 + E8), M4 — the night's resting point

| cell | total | cdot | probe | gather | tq | flood | sort | vs baseline |
|---|---|---|---|---|---|---|---|---|
| scifact r4 | 0.96 ms | 0.41 | 0.09 | 0.05 | 0.13 | 0.21 | 0.08 | **3.3×** |
| scifact binary | 0.93 ms | 0.39 | 0.09 | 0.04 | 0.12 | 0.22 | 0.07 | 3.3× |
| fiqa52k r4 | 1.66 ms | 0.64 | 0.11 | 0.10 | 0.20 | 0.53 | 0.12 | **4.0×** |
| fiqa52k binary | 1.55 ms | 0.59 | 0.11 | 0.09 | 0.17 | 0.53 | 0.11 | 3.9× |

- **E8: KEEP** — gather 0.41 → 0.10 ms (4×) at 52k; identical output.
- **E5: KEEP** — tq 0.35 → 0.20 ms; modest, disjoint-block parallel,
  bit-identical.
- e2e fiqa-52k r4-asym: **5.01 ms mean** (was ~8.5 with the previous
  stage-1; float+old-stage-1 comparison lands after CI).
- Remaining profile is balanced (cdot 39–42% the largest). Stopping the
  M4 exploration here: every further idea we costed (finer cdot blocking,
  sort micro-opts) is < 0.1 ms on the table and platform-fragile.
  Committed as 4a2a917.

## Scope note — batched-centroid path
All of tonight's changes live in `stage1_shortlist` (the dense path,
K ≤ centroid_batch_size = 100k, i.e. corpora up to ~335k docs) plus
`get_candidates` (shared). The batched path (>100k centroids) still runs
its own stage-1 loop and gets only the E8 gather win. Porting E1/E2/E5/E7
there is mechanical but untested — deliberately out of tonight's scope;
flagged for the CR conversation.

## Quality gate (blocker for q8-as-default, independent of tonight's ideas)
q8 flood quantizes shortlist scores and was never nDCG-validated.
binary_ndcg A/B (default q8 vs NP_S1_ABLATE=f32) queued after the timed
runs finish — timed and quality runs must not share the machine.
Runs: {nfcorpus_gte, scifact_gte} × {default q8, NP_S1_ABLATE=f32} ×
{residual-nbits4, binary-int8x1bit}, NDCG_DEPLOYED_ONLY=1, seed-42 builds
(deterministic, so the two modes search identical indexes).
