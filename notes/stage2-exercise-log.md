# Stage-2 quantization exercise — worklog

Running log of the closing phase of the stage-2 exercise: what was done, in
order, with the evidence for each step. Companion docs:
[scaling-1b-docs.md](scaling-1b-docs.md) (1B-doc analysis),
nano-plaid `docs/class4.html` (the kernel story + port epilogue).

## State when this log opened (2026-07-22)

Measurement program complete, CI-verified on x86 AVX2 / Neoverse / Apple M1:

- **Quality**: fused int8 (asym) scoring ≈ float on identical codes — 27
  cells (3 models × 3 datasets × r4/r2/r1), |ΔNDCG@10| ≤ 0.0021, ArguAna
  falsifier passed. Binary retention model-dependent: lateon 83–99%,
  gte 73–90%, edge17m 20–37%.
- **Phase decomposition** (real-shape corpora, cached-index CI profiler):
  decompression is 65–84% of the float exact path — the fused kernels win
  by skipping it, not by out-multiplying GEMM.
- **Corpus ladder** (fiqa shapes, 0.53M / 2.0M / 7.0M tokens): binary e2e
  advantage 7.0→3.8→2.35× (x86), 4.9→2.2→1.57× (Neoverse),
  6.2→3.5→2.0× (M1). Stage-1 is 89–98% of binary's query at 7M tokens.
  The 52k prediction (~2.4×/1.6×/2–2.5×) made in advance and confirmed.
- **Baseline relativity**: the same binary kernel is 13–22× vs
  decompress+GEMM, ~3–4× vs raw-float+vendor-BLAS (reproduces mixedbread's
  3.82×), ~1× vs Apple AMX raw-float. Claims must name their baseline.
- **Repeatability**: two independent CI passes; absolutes drift 8–20% on
  shared VMs, structure stable → quote multipliers as ranges.
- **Traps caught** (each would have silently corrupted a claim): rustup
  x86-default-under-Rosetta, ORT ≤1.20 int8-first-session, OpenBLAS thread
  oversubscription, jetsam SIGKILL with clean-looking logs, `.gitignore`
  eating the shapes manifests, allocator floor-raising across sequential
  builds in one process.

## Must-fix queue (agreed 2026-07-22)

1. **#34 asym scale cliff** — `search.rs:492` routes K > 100,000 to the
   batched stage-1, which hard-codes `residual_asym = false` (~line 796):
   above ~67M tokens (~335k docs) the flagship feature silently reverts to
   float. Fix: after the shortlist is known, gather the distinct centroid
   ids its tokens reference, run the small dense Q × centroids[distinct]
   GEMM, remap per-doc codes to compact columns, reuse the existing kernels
   unchanged. Parity test: force batched mode on a small index (lower
   `centroid_batch_size`) and require bit-equality with the dense path.
2. **#30 dim-48 SIMD expand** — r2/r1 packed rows (12/6 B) fall below the
   16-byte SIMD chunk at dim 48 (edge17m) and run scalar; pad into a
   16-byte scratch so `tbl`/`pshufb` stay engaged. Validate with the
   existing bit-exactness parity suite; re-bench one edge17m cell.
3. **#23 bootstrap CIs** — `scripts/m4_bootstrap_cis.py` over the 9
   completed sweep logs in `~/beir-data/quant_grid/m4_results/`; add scope
   claims to the nano-plaid bench README.
4. **#28 native-M4 columns + final table** — overnight idle-machine run
   (arm64-verified binary, `INDEX_ROOT` cache): mixedbread-protocol
   786-token row + real-shape profile; assemble the final three-altitude
   table (three named baselines, CIs, caveats); re-measure nano-plaid's
   Rosetta-tainted README M4 rows.

## On hold (deliberately, so we don't forget)

- **#32 vfold port** — vectorize the asym epilogue fold (nano-plaid
  measured ~2.1×/rung; closes binary-vs-r1 from ~3× to ~1.2×). Final table
  ships with "asym columns are conservative; known ~2× headroom".
- **#33 stage-1 optimization** — the sequel. Instrument stage-1 phases
  first; then cdot int8 GEMM, cdot transpose + vectorized approx scorer,
  centroid-scan pruning, cell-level approx scoring. Reprioritized by the
  1B analysis: the candidate-flood approx scorer, not the cdot GEMM, is
  the dominant term at scale. Ladder = progress metric.
- **Quantization class** ("Quantization for PLAID, measured") — standalone
  page, next-plaid as main implementation; four-chapter spine (quality /
  where float time goes / altitude & Amdahl / name-your-baseline) + the
  measurement-traps sidebar. Awaiting go on the proposed shape.
- **Chunk-size retune** (spawned session, task_fcd170f0) — par_chunks(128)
  → per-doc/smaller chunks in exact scoring; gated on this branch landing.
- **inv-norms persistence** — first-query spike measured 1.35–3.7 s at 7M
  tokens, linear growth; persist at build time (scaling-notes roadmap
  step; becomes necessary ≥50M tokens).
- **HF discussion #3 ORT note** — drafted, user-gated, do not post without
  explicit approval.
- **nano-plaid README M4 rows** — Rosetta-tainted, re-measure (folds into
  #28's idle-machine window).

## Log

### 2026-07-22 — log opened; starting #34 (asym batched-path cliff)

### 2026-07-22 — #34 fixed: asym now survives the batched path

Test-first: added `batched_path_asym_matches_dense_path`
(tests/residual_lut_integration.rs) — forces the batched path on a small
index via `centroid_batch_size: 8`, requires identical rankings and scores
within 1e-4 of the dense asym path, for nbits 1/2/4. **Confirmed failing
against the unfixed code** (the old behavior scored float, ~1e-3 off).

Fix (search.rs): the batched path already collects the distinct centroid
ids of all candidate docs and scores them sparsely for approximate scoring
(`build_sparse_centroid_scores`) — a superset of anything the shortlist
references. New `exact_doc_score_asym_compact` packs those into a compact
`[nq, distinct]` matrix + cid→column remap; each doc's codes are remapped
once outside the kernel, and `maxsim_residual_lut_i8` runs **unchanged**
(it only indexes `cd[qi*ncent + cid]`, agnostic to id meaning).
`search_one_mmap_batched` now passes `params.residual_asym` through instead
of hard-coding `false`. Scores differ from the dense path only by the cdot
computation route (full GEMM vs per-centroid dots — last-ulp), hence the
1e-4 test tolerance rather than bit-equality.

Result: 196/196 tests pass locally (native arm64). The feature now works
above the ~67M-token / ~335k-doc cliff; the compact matrix costs
O(distinct-centroids) instead of O(K), which is also the right shape for
the 1B regime.
