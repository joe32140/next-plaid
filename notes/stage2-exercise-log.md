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

- ~~**#32 vfold port**~~ — un-held 2026-07-22 on user go; see the #32 log
  entry below. The "asym conservative, ~2× headroom" footnote retires if
  the dataset-scale measurements confirm the port.
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

### 2026-07-22 — #23a: bootstrap CIs on the quality grid (9 cells, 10k paired resamples)

`scripts/m4_bootstrap_cis.py` over the sweep logs; full table saved to
`~/beir-data/quant_grid/m4_results/bootstrap_cis.txt`. One recovery first:
the aborted repeatability rerun had left a 61-byte stub over
`scifact_gte.log`; the overnight original was intact in `.run1` and was
restored (stub kept as `.rerun-aborted`).

**Asym − float (acceptance test), 27 contrasts:** all point estimates
within ±0.0021; CIs essentially inside ±0.005 (two upper bounds graze
+0.0052, in asym's favor). Three contrasts exclude zero — all POSITIVE
(asym slightly better than float): arguana_lateon r1 +0.0009,
nfcorpus_edge17m r4 +0.0009, scifact_edge17m r4 +0.0021. Quality
neutrality is now an error-barred claim, and where it isn't neutral it
favors the int8 path.

**Binary − r1 (same 24 B/token):** model-dependent with tight CIs —
edge17m −0.23..−0.45* (catastrophic), gte −0.056..−0.083*, lateon mixed
(nfcorpus/scifact CIs cross zero; arguana −0.045*, the query-budget
falsifier biting). Notable: nano-plaid's SciFact finding (binary > r1)
does NOT generalize across models — on lateon they tie, on weaker models
binary loses badly. "Binary is a per-model bet" now has numbers.

**r4 − r1:** +0.007..+0.028, significant in 7/9 cells — residual bits buy
measurable quality everywhere.

### 2026-07-22 — #23b: scope claims in nano-plaid README (pushed 71353b8)

The "a production port should expect 2–3×" paragraph now cites the
measured outcome (2.2–6.3× vs compiled decompress+GEMM, 3 CPUs, decompress
65–84%), the 9-cell CI-backed quality result, the renormalization
subtlety, and the name-your-baseline discipline (13–22× / 3–4× / ~1× for
one kernel against three float baselines).

### 2026-07-22 — #28 prep: native M4 columns without local builds

The M4 must never build indexes (16 GB jetsam rule), so tonight's run
mmaps CI-built artifacts end to end:

- `maxsim_bench` now honors `INDEX_ROOT`, sharing `stage2_profile`'s cache
  format — same builder, same seed, same LCG doc stream, so CI's
  BUILD_ONLY output is bit-identical to what the bench would build.
  Verified: run1 builds+caches, run2 loads all four.
- New `build-synth786` CI leg builds the mixedbread-shape indexes (1000
  docs × 786 tokens, one process per index, always-saved cache) and ships
  them as an artifact (pushed b3144a4).
- Overnight script `~/beir-data/quant_grid/m4_overnight_28.sh` scheduled
  for 23:30 (idle window): downloads the latest green run's artifacts,
  verifies the binary is arm64, then runs (a) 786×1000 mixedbread protocol
  vs matrixmultiply baseline (raw-float anchor row included), (b) same vs
  Accelerate/AMX baseline, (c) shape-replay phase profiles for all cells +
  the full 4k/15k/52k ladder — all mmap-only. Output:
  `m4_results/overnight_28.log`. Remaining after that: assemble the final
  three-altitude table; nano-plaid README M4 row re-measurement still
  requires its own venv run (manual follow-up in the same window pattern).

### 2026-07-22 — #32: vfold port (user go: "let's give it a try as long as we are noted")

Prediction registered before measuring (structure matters as much as the
numbers): kernel-level 1.6–2.1× ordered r1 > r2 > r4 and dim 48 > dim 128;
binary-vs-r1 gap ~3× → ~1.2–1.5×; e2e ~1.3–1.4× at nfcorpus scale,
~1.05× at 52k (Amdahl). Falsifier: flat-across-rungs gains would mean the
fold-share model of the kernel is wrong.

Port shape (two changes, deliberately landed together):

1. **cdot transposed to centroid-major** `[K, nq]` end to end — the layout
   nano-plaid had from the start (`cdot_t`). One centroid's scores across
   query rows are now one contiguous strip, so the fold loads them as one
   vector; previously each (row, token) gathered `cd[qi*K + cid]` — at
   dataset K (16–65k) that is 32 loads scattered 64–256 KB apart per token.
   Stage-1 keeps its row-major `[nq, K]` (per-token probing wants row
   scans); the dense path transposes once per query
   (`.t().as_standard_layout()`), the batched path's compact matrix simply
   became row-per-centroid (its sparse scores were already `Array1<f32>`
   per cid — the transposed build is *less* code), and `exact_score_docs`
   builds C×Qᵀ directly at no extra cost.
2. **fold_block** (NEON 4-wide, AVX2 8-wide): per doc token the SDOT/maddubs
   accumulators land in an `accs[nq]` scratch, then one vectorized pass does
   convert→scale-mul→cdot-add→inv-mul→max. Bit-identical to the scalar
   epilogue by construction: cvt is the same round-to-nearest as `as f32`,
   mul/add kept separate (never fused), `inv` multiplied last, and
   vector max equals the scalar `if s > best` select for finite scores —
   `simd_kernel_matches_scalar_bitwise` still asserts `to_bits` equality,
   no tolerance change needed. 196/196 tests pass, incl. the #34
   batched-vs-dense parity test.

First A/B (M4 native arm64, synth 500×180 K=4096, NON-idle machine, bench
includes per-query prep + cdot GEMM): asym r4 2.26→1.79 ms (1.26×),
r2 2.24→1.88 (1.19×), r1 2.14→1.74 (1.23×); binary 0.50→0.44 (no code
change — that pair is the ±12% noise gauge). Real but below the predicted
1.6–2.1× on this cell, and NOT ordered r1 > r4 — partial hit on the
falsifier: on M4 at small K the scalar fold was evidently cheaper than the
nano-derived model assumed (out-of-order hides it behind SDOT latency).
The layout half of the change barely bites at K=4096 (512 KB matrix,
cache-resident); the dataset-scale CI cells (K 16–65k) and tonight's idle
M4 run are the decisive measurements. nano's remaining rung (`tr`,
transpose-reduce of the per-row horizontal `vaddvq`) stays unported.

### 2026-07-22 — session interruption note

The 23:30 overnight scheduler task and a running test command were both
SIGKILLed at ~20:30 by a session restart (not jetsam: 74% memory free, no
JetsamEvent reports, the scheduler died mid-sleep, well before firing).
Overnight #28 re-armed afterwards; it now measures the post-vfold tree,
which is what the final table should carry anyway (CI run 29960245502
remains the pre-vfold reference point for the asym columns).

### 2026-07-22 — #30: dim-48 SIMD tail — fixed, with an honest negative

The narrow-dim expand fix landed: sub-16-byte packed tails now pad into a
16-byte scratch, expand with the same `tbl`/`pshufb`, and copy out only the
valid lanes (a direct store would clobber the next plane's low bytes).
Bit-exact by construction — the nibble tables are verified against the
fused table over all 256 byte values — and the parity suite (dims incl. 40
and 48 × nbits 1/2/4, `to_bits` equality) passes on native NEON; AVX2
validates on CI.

**Measured effect is small**: quick M4 sanity at dim 48 (300 docs, non-idle
machine): asym r4 0.90→0.81 ms, r2 0.91→0.84, r1 flat (0.91→0.95 — within
the ±15% noise floor; binary moved 0.15→0.18 with no code change). This is
Class 04 chapter 09's amortization law: expand is charged once per doc
token while the score core is charged per query row (~32×), so even an
all-scalar expand was a minor share. The real dim-48 cost driver for
edge17m is the per-(row,token) float fold + cdot gather epilogue — i.e.
the on-hold vfold port (#32), not the expand. Kept the fix anyway: it is
strictly correct, removes the "falls to scalar tail" asterisk from the
final table, and the r4/r2 gain is real if modest.
