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

## Self-review of the final diff (9d1f16d..4a7e600)

Line-by-line pass over every hunk, hunting inverted conditions, bounds,
ties, and empty-input edges. Findings:
- `probe_top_k_scan`: a NaN score can occupy a slot during the fill phase
  (first n values) and is never evicted (`v > thr` is false for NaN, and
  argmin skips NaN). Reachable only with corrupt embeddings — the old
  `select_nth` comparator had its own NaN-last convention, so behavior
  differs only on corrupt data. Documented, not guarded.
- `probe_top_k_scan` insert cost is O(n) per insert (argmin rescan); fine
  at n_ivf_probe ≤ 64, degrades gracefully for exotic configs.
- `get_candidates` now panics on an out-of-range ivf doc id where the old
  code silently propagated it into `doc_offsets` (which then panicked
  anyway). Strictly earlier failure, same class.
- Empty-input edges (no candidates, empty rows, nq=0) all fall through to
  the same results as before; sum overflow impossible (255·stride ≪ u32).
Everything else: exact-equivalence arguments hold (bit-identity tests for
par_cdot and the fused emit; value-multiset test for the scan; identical
output construction for the bitmap gather).

## Cross-platform validation (fork CI run 30690316753 vs 30073526835)

fiqa-52k, default rung, per-phase medians. Cross-run comparison on shared
runners — machine instances differ between runs, so treat ratios as
directional; they are far above runner noise. All parity gates green on
all three platforms.

| platform | before total | after total | ratio | probe | gather | cdot | approx |
|---|---|---|---|---|---|---|---|
| x86 (ubuntu-latest), r4 | 15.62 | 5.52 | **2.8×** | 2.14→0.26 | 0.55→0.10 | 5.39→2.02 | 7.44→3.06 |
| Neoverse (24.04-arm), r4 | 12.63 | 3.81 | **3.3×** | 2.23→0.17 | 0.50→0.12 | 3.93→1.12 | 5.68→2.29 |
| macOS VM, r4 | 16.98 | 4.84 | **3.5×** | 1.74→0.16 | 0.52→0.15 | 2.44→1.56 | 11.00→2.60 |
| M4 local (clean same-box A/B) | 6.70 | 1.66 | **4.0×** | | | | |

The M4 was the *least* favorable platform for these changes, confirming
the working rule: scatter-elimination wins grow on server parts (the M4's
L2 was absorbing what Neoverse/x86 pay for). Probe scan is 8–13× off-M4.

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

### Results (real GTE embeddings, deployed regime)

| bundle | scheme | q8 nDCG@10 | f32 nDCG@10 | Δ |
|---|---|---|---|---|
| nfcorpus | residual-nbits4 | 0.3809 | 0.3809 | **0.0000** |
| nfcorpus | binary-int8x1bit | 0.2875 | 0.2875 | **0.0000** |
| nfcorpus | r4 + asym-LUT | 0.3811 | 0.3811 | **0.0000** |
| scifact | residual-nbits4 | 0.7609 | 0.7609 | **0.0000** |
| scifact | binary-int8x1bit | 0.6865 | 0.6865 | **0.0000** |
| scifact | r4 + asym-LUT | 0.7607 | 0.7607 | **0.0000** |

**GATE PASSED.** q8 flood is quality-free at nDCG@10 to four decimals on
both datasets, both schemes, and under the stage-2 LUT overlay. q8 stays
the default.

## Night's conclusion

### What worked (final combination, all default-on)
1. **E2 par_cdot** — column-block-parallel GEMM, bit-identical. cdot
   2.7–3.8× everywhere.
2. **E1 probe_top_k_scan** — running-threshold chunk scan. 6.7× (M4) to
   13× (Neoverse) on the probe; the biggest per-phase ratio of the night.
3. **E7 register-accumulator flood** — stride-padded q8 matrix,
   chunk-outer loop, `[u8;16]` in a vector register. Flood 3.6× on M4;
   the single biggest absolute win (2.29 → 0.53 ms at 52k).
4. **E8 bitmap gather** — O(postings) dedup emitting the sorted list
   directly. 4–5.5× on the phase.
5. **E5 parallel transpose+quantize** — bit-identical block parallelism;
   modest (~1.7×) but free.
6. **E4 sorted probe cells** — neutral on M4, kept for server parts where
   sequential postings reads matter; cost ≈ 0.

Stage-1 fiqa-52k end state: M4 6.70→1.66 ms (4.0×), Neoverse 12.63→3.81
(3.3×), x86 15.62→5.52 (2.8×), macOS VM 16.98→4.84 (3.5×). e2e fiqa-52k
r4+LUT on M4: 5.01 ms mean. Quality: unchanged to 4 decimals (gate above).

### What didn't work / what we learned
- **E3 code dedup: killed by a 10-minute measurement** (dup rate 1–4%) —
  the cheapest kill of the night; measure before writing kernels.
- The flood's cost was never the `umax`s — it was accumulator memory
  traffic and dynamic-trip-count loop overhead. Same lesson as the last
  round's `(*a).max(v)` story, one level up: **shape the loop so the hot
  state lives in a register and the trip counts are static.**
- The M4 understates every scatter win (probe 6.7× local vs 13× Neoverse;
  gather neutral-looking E4). Cross-platform CI remains mandatory before
  claiming a layout change is worthless.
- Intra-query parallelism was already stage-1's contract (the flood used
  par_iter upstream); extending it to the GEMM and transpose is
  consistency, not a new policy.

## Stage-2 round (same night, after Joe's go-ahead)

Applied the same method to stage-2. New NP_S2_PHASES instrumentation
splits search_one_mmap into s1 / prep / score / sortemit / glue walls plus
pooled kernel CPU time; NP_S2_PRETOUCH adds a timed cache-line walk of the
residual bytes to separate memory wait from compute (the mmap views are
lazy, so a naive load timer measures nothing).

### Decomposition (M4, fiqa-52k, per-scheme medians)

| scheme | total | s1 | prep | score wall | sortemit | glue | kern CPU |
|---|---|---|---|---|---|---|---|
| asym r4 | 4.49 | 1.59 | 0.01 | 2.86 | 0.02 | **0.01** | 17.35 |
| float r4 | 16.09 | 1.49 | 0.00 | 14.64 | 0.02 | −0.06 | 92.25 |
| binary | 2.56 | 1.50 | 0.00 | 1.06 | 0.00 | −0.01 | 6.13 |

Hypotheses killed by the numbers, before any code:
- **"Missing 0.85 ms of glue": DEAD.** Glue is ~0.01 ms. The earlier gap
  was an artifact of comparing the harness's isolated-exact loop with e2e.
- **Memory wait: ~nil on the warm M4.** Pretouch walk = 0.6 ms CPU across
  the whole pool; kernel time unchanged. Id-sorted scoring and
  RAM-resident residuals are non-events here (CI-cold-cache questions at
  most).
- Prep and sortemit: both ≈ 0. Nothing to do.

What survived: **parallel shape.** score wall 2.86 ms vs 17.35 ms kernel
CPU = 6.1 effective threads on a 10-core M4, because the loop split 1024
docs into exactly 8 fixed chunks (DECOMPRESS_CHUNK_SIZE = 128): two cores
never get work and one long-doc chunk straggles. The chunking's memory
rationale is vestigial — in-flight decompressed docs are bounded by thread
count either way (confirming the par_chunks study). Fix: plain par_iter
with adaptive splitting, both dense and batched paths.

### Fix measured (M4, fiqa-52k)

| scheme | score wall | total | effective threads |
|---|---|---|---|
| asym r4 | 2.86 → **2.19 ms** (1.31×) | 4.49 → **3.77 ms** | 6.1 → 9.4 |
| binary | 1.06 → **0.82 ms** (1.29×) | 2.56 → **2.29 ms** | |
| float r4 | 14.64 → 12.56 ms (1.17×) | 16.09 → 14.43 ms | |

Pooled kernel CPU rose (E-cores now participate and are slower per op) —
wall is what matters, and the pool is full. With scheduling fixed, the
remaining stage-2 time IS the kernel, which nano-plaid already
characterized near its roofline. The only exact-mechanics idea left is
doc-batching to amortize query tile loads — invasive, uncertain payoff,
deliberately not started tonight. Everything else on the stage-2 idea list
was killed by the decomposition before a line of code was written, which
is the method doing its job.

Combined night effect at fiqa-52k on M4: **asym e2e 5.01 → 3.77 ms**
(stage-1 round) → and the stage-1 wins carry every scheme.

### Cross-platform read of the par_iter fix + the granularity floor

Cross-run CI (30692851529 → 30708616379 → 30709931168), and a
methodological faceplant worth keeping:

1. Eyeballing the fiqa-52k rows of the par_iter run, I read "x86 asym
   consistently 0.95–0.97×", diagnosed steal overhead at ~17 µs/doc
   tasks, and committed a `with_min_len(8)` floor (fbe6f0c).
2. The *geomean over all 32 cells* told the truth: pure per-doc on x86
   was **1.006×** — neutral. The 0.95s were instance noise concentrated
   in the rows I happened to look at. The floor run then measured 0.914×
   on a *different* x86 instance — the same noise class, now pointing the
   other way. Neoverse: 1.028× (per-doc) vs 1.057× (floor), also within
   the band.
3. Theory agrees the two shapes are near-identical: rayon splits ranges
   adaptively on steal demand — it never creates per-element tasks — so
   a manual floor defends against a machine that doesn't exist. Both M4
   measurements concur (2.19 vs 2.25 ms, noise).

**Reverted the floor; pure per-doc par_iter is the final shape.**
Lesson recorded next to the thermal-drift protocol: cross-run CI deltas
are per-instance; never diagnose from a row subset when the harness gives
you 32 cells to geomean — and when a "fix" only helps under a theory the
scheduler doesn't implement, re-derive before committing.

Net cross-platform read of the stage-2 round (vs pre-round, all cells):
M4 asym e2e 4.49 → 3.77 ms (1.19×, clean same-box A/B); Neoverse
+3–6% geomean; x86 neutral; macOS VM unreadable. The M4/Neoverse win is
real and core-count-dependent, exactly as adaptive splitting predicts.

### Follow-ups deliberately left
- Port the same mechanics to the batched-centroid path (>~335k docs).
- Recall-coupled ideas (bound pruning, adaptive probing) — quality-gated
  chapter, needs the WARP read first.
- The stage-1 work now merits its own CR stacked on the stage-2 LUT CR;
  curation night pattern applies (strip NP_S1_PHASES, keep NP_S1_ABLATE?
  — decide at curation).

The u8 quantization of flood scores is invisible at nDCG@10 to four
decimals on nfcorpus — the 4096-deep prune cut absorbs sub-LSB rank
perturbations exactly as hypothesized. These runs also exercise the entire
new stage-1 stack (E1/E2/E4/E5/E7/E8 active in both arms; only the flood
mode differs), and reconfirm stage-2 LUT quality through the new stage-1
(Δ vs float +0.0002 / −0.0001 on fresh seed-42 builds).
