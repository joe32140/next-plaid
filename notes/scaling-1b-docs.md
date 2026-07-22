# Scaling next-plaid to 1B documents — code-grounded analysis

Branch: `feat/asymmetric-lut-residual`, crate `next-plaid/`. All file:line references are into
`next-plaid/src/` unless prefixed. Every number is tagged:

- **[M]** measured (this week's CI/bench numbers, taken as given)
- **[C]** computed extrapolation (formula evaluated from code + the 1B-scale assumptions)
- **[J]** judgment

Scale assumptions throughout: **N = 10^9 docs × 200 tokens/doc → T = 2×10^11 tokens, dim d = 128,
32 query tokens.**

---

## 1. Scale picture at 1B docs

### 1.1 Derived core quantities

| Quantity | Formula (code) | Value at 1B docs |
|---|---|---|
| Total tokens T | N × 200 | 2×10^11 |
| Centroids K | `2^floor(log2(16·sqrt(T)))` — kmeans.rs:304-308 | 16·√(2×10^11) = 7.16M → **K = 2^22 = 4,194,304** [C] |
| Avg docs/IVF cell | N · E[distinct cells/doc] / K ≈ 10^9·~190/4.19M | ≈ **43,000 docs/cell** [C] |
| IVF postings | Σ_docs distinct cells/doc (dedup at index.rs:906-914) | ≈ 1.9×10^11 entries [C] |
| cdot GEMM / query | 2 · 32 · d · K FLOP (search.rs:580 or batched search.rs:329) | **34.4 GFLOP/query** [C] |
| k-means sample docs | `1+16·sqrt(120·N)` — kmeans.rs:273-276 | 5.54M docs = 1.11×10^9 sample tokens [C] |

### 1.2 On-disk index size per scheme (merged files, single directory)

Per-token bytes at d=128 [M, confirmed by code: `packed_dim` binary.rs:1086, `dim·nbits/8`
index.rs:307-311, code = 8 B i64]:

| Scheme | residual/sign B/tok | code B/tok | merged_residuals.npy | merged_codes.npy | Total payload |
|---|---|---|---|---|---|
| f32 (raw) | 512 | — | — | — | 102.4 TB [C] |
| r4 (nbits=4) | 64 | 8 | 12.8 TB | 1.6 TB | 14.4 TB [C] |
| r2 | 32 | 8 | 6.4 TB | 1.6 TB | 8.0 TB [C] |
| r1 | 16 | 8 | 3.2 TB | 1.6 TB | 4.8 TB [C] |
| binary | 16 | 8 | 3.2 TB | 1.6 TB | 4.8 TB [C] |

Plus, for every scheme:

- `ivf.npy`: ~1.9×10^11 × 8 B = **1.5 TB** [C]
- `centroids.npy`: 4.19M × 128 × 4 B = **2.15 GB** [C]
- `doclens.{i}.json`: 20,000 files (N / batch_size 50,000 — index.rs:731), ~4 GB of JSON [C]
- **×2 disk duplication**: merge_codes_chunks / merge_residuals_chunks (mmap.rs:1266, 1483) write
  merged copies but never delete the `{i}.codes.npy` / `{i}.residuals.npy` chunk files → real disk
  = chunks + merged ≈ **29 TB for r4, 9.6 TB for binary** [C].

### 1.3 RAM-resident structures in `MmapIndex` (index.rs:1037-1062) at 1B docs

What's mmap'd vs RAM, straight from `MmapIndex::load` (index.rs:1072-1186):

| Structure | Type / where loaded | Formula | Size @1B | Verdict |
|---|---|---|---|---|
| `mmap_codes` | mmap (index.rs:1168-1171) | T × 8 B on disk | 1.6 TB disk | OK (paged) |
| `mmap_residuals` | mmap (index.rs:1168-1171) | T × packed | 3.2–12.8 TB disk | OK (paged) |
| `codec.centroids` | mmap (codec.rs:548-554, `CentroidStore::Mmap`) | K·d·4 | 2.15 GB disk | OK, but scanned fully per query |
| **`ivf`** | **RAM** `Array1<i64>` (index.rs:1121-1126) | postings × 8 | **≈1.5 TB RAM** | **hard blocker** |
| `ivf_lengths` | RAM `Array1<i32>` (index.rs:1128-1133) | K × 4 | 16.8 MB | fine |
| `ivf_offsets` | RAM `Array1<i64>` (index.rs:1136-1140) | (K+1) × 8 | 33.6 MB | fine |
| **`doc_lengths`** | RAM `Array1<i64>` from 20k JSON files (index.rs:1143-1150) | N × 8 | **8 GB RAM** (+ JSON parse of 10^9 ints on every load) | blocker-adjacent |
| **`doc_offsets`** | RAM `Array1<usize>` (index.rs:1153-1156) | (N+1) × 8 | **8 GB RAM** | blocker-adjacent |
| bucket cutoffs/weights, avg_residual | RAM (codec.rs:564-590) | O(2^nbits + d) | KB | fine |
| **`residual_inv_norms`** | RAM `OnceLock<Vec<f32>>`, built lazily on first asym query (index.rs:1375-1387) | T × 4 | **800 GB RAM** + transient **1.6 TB** `Vec<i64>` from `mmap_codes.slice(0, n)` (mmap.rs:829-838 copies element-wise) | **hard blocker** |

Total RAM to serve one r4 index at 1B docs with the current code: **≈1.5 TB before the first asym
query, ≈2.3 TB+ after** [C]. Even at the near-term 10M-doc target the same formulas give: ivf 15.2 GB,
doc_lengths+offsets 320 MB, inv_norms 8 GB + 16 GB transient — already past a 16 GB dev box [C].

### 1.4 Per-query cost at scale (stage-1 dominated)

[M] Stage-1 = 84–92% of binary e2e at 2M tokens; cdot measured 6.5 ms @ K=8,192 and 18.5 ms @
K=16,384 (4 vCPU x86). Linear-in-K extrapolation of the measured 18.5 ms [C]:

| Docs | Tokens | K | cdot GFLOP/q | Stage-1 cdot extrapolated (4 vCPU) |
|---|---|---|---|---|
| 10k | 2×10^6 | 16,384 | 0.13 | 18.5 ms [M] |
| 336k | 6.7×10^7 | 131,072 | 1.07 | ~150 ms [C] |
| 10M | 2×10^9 | 524,288 | 4.3 | ~590 ms [C] |
| 100M | 2×10^10 | 2,097,152 | 17.2 | ~2.4 s [C] |
| 1B | 2×10^11 | 4,194,304 | 34.4 | **~4.7 s** [C] |

And stage-1.5 (candidate gather + approximate scoring) grows even faster [C]:

- Candidates/query ≈ 256 probed cells × 43k docs/cell ≈ **11M docs** (get_candidates
  sort+dedup of 11M i64 — index.rs:1189-1203, ~1 s alone).
- Approximate scoring touches *every code of every candidate*: 11M docs × 200 codes × 32 query
  tokens ≈ **7×10^10 lookups/query** (approximate_score_mmap, search.rs:460-479; the batched
  variant does the same through a per-code `HashMap` lookup, search.rs:429-457, 758-767 — worse).
  At ~10–30 ns/lookup this is **10–30 min/query**. This term, not cdot, is the true 1B-scale
  killer [C].
- Stage-2 stays fixed at ~1024 docs (`n_full_scores/4`, search.rs:699-700) [M] — candidate I/O is
  1024 × 200 × 72 B ≈ 15 MB of random reads, fine on NVMe [C].

### 1.5 Build cost extrapolation

[M] create_with_kmeans peak RSS: 8 GB @ 0.5M tokens, 12.8 GB @ 2M, 15.7 GB @ 7M (4 threads).

- The build API takes the whole corpus as `&[Array2<f32>]` (create_index_files, index.rs:577;
  create_index_with_kmeans_files, index.rs:969) → raw embeddings alone at 1B docs = **102 TB RAM**.
  Everything else is moot until this is streaming [C].
- k-means: samples_tensor = 1.11×10^9 tokens × 512 B = **567 GB RAM** (kmeans.rs:291-301);
  training assignment ≈ samples × K × d × 2 ≈ **1.2×10^18 FLOP per iteration** × 4 iters
  (kmeans_niters, index.rs:87) — ~2 weeks per iteration at 1 TFLOP/s [C].
- Full-corpus assignment (compress_into_codes_cpu, codec.rs:297-343): T × K × d × 2 =
  **2.1×10^20 FLOP** — ~25 days even at a sustained 100 TFLOP/s [C]. The 1 GB score-matrix budget
  (codec.rs:11) caps memory but not compute.
- IVF construction holds `all_codes: Vec<usize>` (T × 8 = **1.6 TB**, index.rs:744) plus
  `code_to_docs: BTreeMap<usize, Vec<i64>>` (≈3–5× postings ≈ **5–8 TB**, index.rs:891-914) in RAM [C].

---

## 2. Hard blockers (break outright at 1B docs)

Ordered by severity; each with the exact code and the formula that breaks.

1. **IVF loaded into RAM** — index.rs:1121-1126 (`Array1::read_npy` of `ivf.npy`).
   Size = postings × 8 B ≈ 1.5 TB @1B; already 15 GB @10M docs. The access pattern is
   contiguous ranges per probed cell (index.rs:1193-1197) — the single most mmap-friendly
   structure in the system, and it's the one fully materialized. [C]

2. **Approximate scoring is O(candidates × doclen × nq)** — search.rs:460-479 / 758-767, driven
   by candidates ≈ probes × N·190/K ∝ N/√N·... ≈ 11M docs @1B. 7×10^10 lookups/query. No code
   path scores candidates from cell-level information; every candidate's full code list is
   re-read (one heap-allocating `Vec<i64>` per doc per query — mmap.rs:829-838). [C]

3. **Dense full-K centroid scan per query** — two sites assume it:
   - stage1_shortlist computes `query.dot(centroids.t())` over all K (search.rs:580) and a
     per-token O(K) scan (search.rs:632-643); the dense matrix is 32×4.19M×4 = 537 MB/query @1B [C].
   - ivf_probe_batched covers 0..K in batches (search.rs:306-309) — memory-bounded but still
     O(K) compute: 34 GFLOP → ~4.7 s/query on 4 vCPU [C].
   Downstream consumers of the dense matrix: approximate_score_mmap row indexing
   (search.rs:467) and the LUT path's centroid term `cdot[[qi, cid]]`
   (residual_lut.rs:259, exact_doc_score search.rs:172-191). Any pruning replacement must
   provide (a) per-token top-n cells, (b) per-centroid max for the 0.4 threshold
   (search.rs:652-659), (c) on-demand scores for arbitrary candidate codes — (c) already exists
   as build_sparse_centroid_scores (search.rs:414-427).

4. **`residual_inv_norms` OnceLock** — index.rs:1375-1387 + residual_lut.rs:176-206.
   T × 4 B = 800 GB resident, plus `mmap_codes.slice(0, n)` materializes a T-element
   `Vec<i64>` = 1.6 TB *before* computing. Build time: doc comment says "~seconds per million
   tokens" [M] → 2×10^5 s ≈ **55 hours inside the first query's `get_or_init`**, during which
   every other asym query blocks on the OnceLock [C]. Must become a build-time artifact.

5. **Build = whole corpus in RAM** — create_index_files signature (index.rs:577) and every
   caller (index.rs:969, MmapIndex::create_with_kmeans index.rs:1467); k-means sample tensor
   567 GB (kmeans.rs:291); IVF build tables 5–8 TB (index.rs:744, 891). 102 TB of raw f32 input. [C]

6. **k-means flat training + flat assignment** — kmeans.rs:319-342 trains a flat K=4.19M
   codebook (10^18 FLOP/iter); compress_into_codes is exact brute-force N×K (10^20 FLOP total).
   No hierarchical/sampled-K structure anywhere. [C]

7. **Updates and deletes are O(index), not O(delta)** —
   - update_index rewrites the *entire* IVF per batch (update.rs:1021-1090: loads old ivf,
     merges, rewrites → 1.5 TB read+write per update @1B; 15 GB @10M).
   - It then clears merged files (update.rs:1129, mmap.rs:1714-1743), so the next load re-merges
     **the whole payload** (14.4 TB rewrite for r4 @1B; ~144 GB @10M docs) [C].
   - Centroid expansion runs find_outliers = O(new_tokens × K × d) (update.rs:490-608 called at
     update.rs:662) — 10k new docs @1B-scale K costs 2×10^6·4.19M·128 ≈ 10^15 FLOP [C].
   - delete_from_index also rewrites the full IVF (delete.rs:200-234).
   Verdict: streaming builds via `MmapIndex::update` are usable to ~100k docs, degrade badly
   past ~1M, unusable at 1B [J].

8. **Merge machinery itself** — merge_codes_chunks writes i64s one `write_all` at a time
   (mmap.rs:1391-1397: 2×10^11 calls); the chunk-scan reads *entire* chunk arrays just to learn
   row counts (mmap.rs:1325-1328, 1543-1546 `Array1/Array2::read_npy` for `.len()`), so a cold
   load without a valid manifest reads all 14 TB twice. Manifest fast-path (mmap.rs:1280-1300)
   avoids this only while `metadata.json` mtime is unchanged. [C]

Integer-width audit (mostly OK on 64-bit):
- `ivf_lengths: i32` (index.rs:515-521, 904-911; update.rs:1074 `as i32`): per-cell count is a
  deduped doc count ≤ N, so it overflows only past 2.1B *docs* — safe at 1B, silent wrap beyond [C].
- Codes stored as i64 (index.rs:854, EncodedIndexChunk index.rs:180) — wasteful (K=4.19M fits u32;
  codes file could be 800 GB instead of 1.6 TB) but not an overflow [C].
- doc ids i64 everywhere (ivf, search results) — fine to 9.2×10^18.
- `doc_offsets: usize`, npy shapes parsed as usize (mmap.rs:720-749) — fine on 64-bit; the whole
  crate is silently 64-bit-only (12.8 TB mmaps need 47-bit address space — fine on x86-64/aarch64) [J].
- Legacy converter `convert_i64_to_i32_npy` truncates with `as i32` (mmap.rs:581-584) — legacy
  fast-plaid path only [J].

---

## 3. Concrete near-term improvements (single node, ≤10M docs), ranked

At 10M docs: K = 524,288, cdot ≈ 590 ms/q [C], ivf = 15 GB RAM, inv_norms first-call ≈ 8 GB + 16 GB
transient. Validation hooks: **quality grid** = exp/quant-grid CI + `examples/binary_ndcg.rs`
(criterion: ≤0.002 NDCG@10 delta), **phase profiler** = `examples/stage2_profile.rs`
(INDEX_ROOT cache, INDEX_TAGS, shapes replay), **corpus ladder** = fiqa 4k/15k/52k CI ladder +
`examples/shapes/{fiqa,nfcorpus,scifact}` manifests.

1. **Re-enable `residual_asym` on the batched path (it silently dies at ~336k docs).**
   `use_batched` triggers when K > centroid_batch_size = 100,000 (search.rs:492, default
   search.rs:58-60), i.e. at T ≥ 6.7×10^7 tokens ≈ **336k docs** [C] — and
   search_one_mmap_batched hard-codes `prepare_score_query(index, query, false)`
   (search.rs:790-796) because the LUT arm needs the dense cdot matrix. But the centroid term
   only needs scores for *codes that appear in shortlisted docs*, and the batched path already
   builds exactly that sparse map (unique_centroids + build_sparse_centroid_scores,
   search.rs:742-755). Change: pass a sparse/gathered cdot (e.g. a compact
   `[nq × |unique_centroids|]` matrix + code→column remap) into the LUT arm.
   Effect: the branch's flagship stage-2 win stops being capped at small corpora.
   Validate: quality grid asym-vs-float delta ≤0.002 NDCG on fiqa52k (which is *above* the
   336k-token threshold? fiqa52k ≈ 52k docs — also add a ladder rung >336k docs to cover the
   batched path at all); phase profiler stage-2 times must match the non-batched asym numbers.

2. **mmap the IVF and store postings as u32.**
   index.rs:1121-1126 → keep `ivf.npy` on disk (access is contiguous per cell,
   index.rs:1193-1197: perfect mmap locality); write postings as u32 (valid to 4.3B docs/segment).
   Effect: −15 GB RAM @10M docs, −50% ivf disk, no result change (bit-identical).
   Validate: corpus ladder RSS measurement; assert identical passage_ids/scores on fiqa52k.

3. **Persist `residual_inv_norms` as a build artifact (`inv_norms.npy`), mmap on load.**
   Kill the lazy OnceLock build (index.rs:1375-1387) or keep it only as a fallback for old
   indexes. Also fix the transient T×8 B `Vec` by iterating the mmap directly instead of
   `mmap_codes.slice(0, n)` (mmap.rs:829-838). At build time the norms are nearly free (residuals
   are already in cache during encode, index.rs:816-821). Effect: first-asym-query stall
   (minutes @10M docs) → 0; RAM becomes OS-paged. Validate: stage2_profile "prep" phase before/
   after; bit-identical scores.

4. **Two-level centroid scan (stage-1 ANN) — the biggest latency lever above ~1M docs.**
   Stage-1 is 84–92% of e2e [M] and pure O(K). Cluster the K centroids into √K meta-centroids
   at build time (a second tiny k-means over `centroids.npy`, ~4M×128 input @1B, 524k×128 @10M —
   seconds); per query token scan meta-centroids (√K ≈ 724 @10M) and expand only the top-m
   meta-cells (m ≈ 16–64) before the exact per-token top-n_probe selection. Cost drops from
   O(K) to O(√K + m·√K) per token ≈ 20–30× less GEMM [C]. The three consumers listed in
   blocker #3 are all satisfiable: per-token top-cells (directly), threshold max (from scanned
   subset — slightly conservative), candidate-code scores (sparse gather, improvement #1's
   machinery). Effect @10M: stage-1 ~590 ms → ~30–60 ms [C, to be measured].
   Validate: quality grid ≤0.002 NDCG at m sweep (m is the falsifiable knob: report
   NDCG-vs-m curve; if no m gives ≤0.002 at ≥5× speedup, the design is falsified);
   corpus-ladder e2e times.

5. **Zero-copy, u32 codes.**
   (a) `MmapNpyArray1I64::slice` returns an element-by-element `Vec<i64>` (mmap.rs:829-838) —
   the npy data offset is 64-byte aligned (header layout mmap.rs:1177-1182), so return
   `&[i64]` via an aligned cast; kills one heap alloc + copy per candidate doc per query
   (≈930k allocs/query @10M [C]).
   (b) Store codes as u32 on disk (halve merged_codes.npy; halve bytes streamed by approximate
   scoring and inv-norms build). Requires touching write sites (index.rs:854, update.rs:959)
   and the mmap reader; keep i64 read-compat.
   Validate: bit-identical results; phase profiler approx-scoring segment.

6. **Cell-level approximate scoring (restructure stage-1.5).**
   Instead of re-reading all codes of every candidate (search.rs:679-687), accumulate per-doc
   partial MaxSim directly from the probe: for each probed cell c with per-token scores s[q][c],
   for each doc in cell c, `best[doc][q] = max(best[doc][q], s[q][c])`. Cost = postings-touched ×
   nq ≈ 930k×32 @10M vs 930k×200×32 today — **~200× fewer ops**, and zero mmap_codes traffic in
   stage 1.5 [C]. This is a *different estimator* (scores only probed cells, i.e. a lower bound —
   closer to the original PLAID paper's centroid-interaction pruning) so it is quality-relevant:
   gate behind a SearchParameters flag and run the full quality grid; falsified if NDCG delta
   >0.002 at n_full_scores=4096. Memory: candidates×nq f32 (~120 MB @10M) — cap with u16
   quantized scores or a two-pass over cells [J].

7. **Doclens: JSON → npy, u32.**
   index.rs:1143-1150 parses 200 JSON files of 10^7 ints on every load (seconds); doc_lengths
   never exceeds token-per-doc bounds (u16 would do; u32 is safe). Write `doclens.npy` once at
   merge time next to the manifest. Effect: load time and 240 MB RAM @10M. Validate: load-time
   timing in stage2_profile INDEX_ROOT warm path.

8. **Merge I/O hygiene.**
   Read chunk shapes from npy headers (parse_npy_header, mmap.rs:660-717) instead of
   full `read_npy` (mmap.rs:1325-1328, 1543-1546); write codes via buffered block copies not
   per-value write_all (mmap.rs:1391-1397); optionally delete chunk files after a verified
   merge (or better: mmap chunks directly through a small chunk-table indirection and drop the
   merge entirely — this also fixes the post-update full re-merge, update.rs:1129).
   Effect @10M: halves disk (2×72 GB → 72 GB r4), update-then-first-search stall shrinks from
   full-payload rewrite to zero. Validate: ladder build wall-clock + disk usage; integration
   tests (tests/binary_integration.rs, residual_lut_integration.rs) unchanged.

9. **get_candidates dedup via bitmap.**
   sort+dedup of ~1M i64s per query (index.rs:1200-1202) → a num_docs-bit visited bitmap
   (1.25 MB @10M) reset per query, or reuse a per-thread scratch. ~O(candidates) instead of
   O(candidates log candidates). Minor but free. Validate: bit-identical results.

10. **Batch cdot across concurrent queries.**
    search_many_mmap parallelizes per query (search.rs:842-855) but each query does its own
    32×K GEMM; stacking Q queries into one (32·Q)×K GEMM amortizes the centroid stream
    (2.15 GB @1B is re-streamed per query today). Throughput-only change; latency-neutral.
    Validate: QPS benchmark via search_many_mmap parallel mode on ladder indexes.

---

## 4. Architecture for 1B

What the system must become, grounded in which seams exist vs. what's missing.

### 4.1 Segments (the load-bearing change)

One flat index cannot work: K ∝ √T forces both flat k-means (blocker #6) and O(K) scans
(blocker #3), and IVF/doc arrays are global RAM (blocker #1). The standard fix is
Lucene-style **segments**: self-contained sub-indexes of ~5–20M docs, each with its own
centroids (K ≈ 2^19), IVF, codes/residuals, doclens, inv-norms. 1B docs = 50–200 segments.
Per-segment everything fits today's measured envelope (that's precisely the ≤10M near-term
regime section 3 optimizes) [J].

Existing seams that support this:
- Chunk files with per-chunk `metadata.json` incl. `embedding_offset` (index.rs:169-176,
  829-842) — a physical sharding of the payload already exists; what's global is only IVF,
  centroids, and the doc-id space.
- Scores are absolute MaxSim sums (search.rs:97-99) — comparable across segments with the same
  query, so multi-segment top-k is a trivial k-way merge of `QueryResult`s. No IDF-style global
  statistics anywhere in scoring [C].
- The API server already manages a registry of independent `MmapIndex`es with lock-free swap
  (next-plaid-api/src/state.rs:21-57, 221-231 `ArcSwap<MmapIndex>` slots) — a segment router can
  reuse this shape.
- `subset` search (search.rs:585-617) gives per-segment doc filtering for free.
- Binary/residual scheme choice is per-index metadata (index.rs:133-137) — segments can even mix
  schemes during migration.

Missing entirely:
- A `Segment`/`Shard` type and a doc-id remap (global i64 ↔ (segment, local u32)). Nothing in
  the crate composes two MmapIndexes.
- Segment lifecycle: sealed segments + a small mutable "buffer" segment (the existing
  buffer/start_from_scratch machinery, index.rs:1531-1677, is per-index and rebuild-oriented,
  but its *shape* — small mutable head + big immutable body — is exactly right to generalize) [J].
- Tombstones. delete currently rewrites payload+IVF (delete.rs:66-234); at segment scale
  deletes become a bitmap consulted at scoring time + periodic segment rewrite.

### 4.2 Centroid index (sub-linear stage-1)

Per segment K ≈ 524k still needs improvement #4 (two-level scan) to hit interactive latency;
at the top level, fan-out replaces the impossible flat K=4.2M. The two-level scan is the
minimal ANN; if per-token probing quality demands more, an HNSW over centroids is the
standard next step — but it must serve the three consumers in blocker #3, and the threshold
(search.rs:652-659) becomes approximate. Start with two-level; it's testable against the
quality grid [J].

### 4.3 Hierarchical / sampled k-means build

Per segment, today's pipeline nearly works (measured 15.7 GB @7M tokens [M]); the corpus must
stream, though:
- Two-pass build: pass 1 reservoir-samples tokens for k-means (the sampling heuristic
  kmeans.rs:273-276 already caps samples; it just materializes them from an in-RAM corpus);
  pass 2 streams docs from disk in 50k-doc chunks through encode_index_chunk (index.rs:300)
  — which is *already* chunk-shaped — writing chunks + accumulating IVF postings to a spillable
  sorter instead of `Vec<usize>` + BTreeMap (index.rs:744, 891).
- For K ≥ 2^19, train hierarchically: coarse k-means (√K), then per-coarse-cell sub-k-means;
  assignment via coarse-then-fine is O(√K) per token, which also gives the two-level search
  structure for free (same tree) [J].

### 4.4 Disk layout

- Per-segment directory = current index directory format (keeps mmap loaders intact), minus the
  merged-file duplication (improvement #8), plus `doclens.npy`, `inv_norms.npy`, and a
  `centroid_tree.npy`.
- Candidate gather at query time is ~15 MB of random reads over the payload [C] — NVMe-friendly;
  mmap remains adequate per segment (payload/segment ≈ 150–300 GB r4 — page cache holds the hot
  IVF+codes, residuals stream on demand). io_uring batching is an optimization, not a
  requirement [J].

### 4.5 Distributed query fan-out

Nothing distributed exists in-repo (single-process API, state.rs). Needed: a thin router that
fans a query to node-local segment sets (search_batch already exists per index,
index.rs:1335-1343), merges top-k, and owns the global doc-id map. Because segment scoring is
embarrassingly parallel and scores are comparable, this is orchestration work, not algorithm
work [J]. Rough sizing: 1B docs r4 ≈ 15 TB payload → 8–15 nodes with 2 TB NVMe each; per-query
work per node = a handful of per-segment searches of section-3-optimized cost (~tens of ms each)
[C, J].

---

## 5. Ordered roadmap (each step falsifiable)

| # | Step | Falsification criterion |
|---|---|---|
| 1 | mmap + u32 IVF (impr. #2) | RSS on fiqa52k ladder does not drop by ≈ivf size, or any result differs bit-wise |
| 2 | Persist inv_norms + kill transient codes Vec (impr. #3) | stage2_profile asym "prep" phase not ~eliminated on warm INDEX_ROOT; NDCG delta ≠ 0 |
| 3 | Sparse cdot → asym on batched path (impr. #1) + a >336k-doc ladder rung | asym vs float NDCG delta >0.002 on the new rung, or stage-2 speedup < the non-batched asym ratio |
| 4 | Zero-copy u32 codes + doclens.npy + merge hygiene (impr. #5/#7/#8) | approx-scoring phase not faster; load time not faster; disk not halved |
| 5 | Two-level centroid scan behind a SearchParameters flag (impr. #4) | no operating point with NDCG delta ≤0.002 *and* ≥5× stage-1 speedup at 10M-scale (shapes-replay ladder) |
| 6 | Cell-level approximate scoring behind a flag (impr. #6) | NDCG delta >0.002 at n_full_scores=4096 on the quality grid |
| 7 | Segment abstraction: `Vec<MmapIndex>` + id remap + k-way merge, exposed via the API registry | multi-segment result set ≠ single-index result set on an identically-partitioned corpus (must be near-identical; only IVF boundary effects allowed, measured ≤0.002 NDCG) |
| 8 | Streaming two-pass build with spill-sort IVF | peak RSS grows with corpus size (must be flat in corpus, linear in segment) |
| 9 | Hierarchical k-means for K ≥ 2^18 (shared tree with step 5) | NDCG delta >0.002 vs flat k-means at 52k-doc scale, or build wall-clock not ≥3× better at 1M-doc scale |
| 10 | Tombstone deletes + segment compaction | delete cost not O(deleted); search-after-delete results wrong |
| 11 | Multi-node router (fan-out + merge) over segment nodes | e2e p50 at 1B synthetic (shapes-replay) exceeds budget with section-3 per-segment numbers — i.e. the whole extrapolation chain is re-measured, not assumed |

Steps 1–4 are pure wins (no quality surface, bit-identical or better), land on the current
branch, and matter *today* at fiqa52k–10M scale. Steps 5–6 open the latency ceiling and carry
the quality-grid gate. Steps 7–11 are the 1B architecture; nothing before step 7 requires
breaking the on-disk format, and step 7 deliberately reuses it per segment.
