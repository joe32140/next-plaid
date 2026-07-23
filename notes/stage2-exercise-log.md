# Stage-2 quantization exercise — worklog

Running log of the closing phase of the stage-2 exercise: what was done, in
order, with the evidence for each step. Companion docs:
[stage2-final-summary.md](stage2-final-summary.md) (the results doc),
[scaling-1b-docs.md](scaling-1b-docs.md) (1B-doc analysis),
nano-plaid `docs/class4.html` (the kernel story + port epilogue).

### 2026-07-23 (night) — the clean CR, and what its CI gates caught

Overnight goal: fold the ablation results into class 4, and shape the LUT
work into a clean CR stacked on #155. This branch
(`feat/asymmetric-lut-residual`) stays frozen as the research record —
notes/, scripts/, the bench workflow, the ablation harness, all of it. The
CR is a new branch built from it.

**The CR: `feat/asymmetric-residual-lut`** (fork, stacked directly on
`feat/asymmetric-binary-quant` = #155, which it is 0 commits behind).
Two commits, 6 files, +1,814/−59 — down from the research branch's 62
commits and +5,959:

- **In**: `residual_lut.rs` (scalar + NEON/AVX2/AVX-512 kernels + parity
  and semantics tests), `search.rs` integration (`residual_asym` flag,
  blocked transpose, batched-path compact matrix), `index.rs` inv-norms
  cache, `lib.rs`/`binary.rs` glue, the 4 integration tests.
- **Out**: examples (stage2_profile, maxsim_bench, binary_ndcg), shape
  manifests, maxsim-bench.yml, notes/, scripts/ — measurement harness,
  not product. Precedent: #155 itself dropped its kernel_bench example
  before review.
- **Out, deliberately: the whole `NP_ASYM_ABLATE` machinery.** Reasons:
  (1) the `row_major` mode makes a *public function's matrix orientation*
  depend on an env var — a library footgun no reviewer should accept;
  (2) its testing job is done better without it — the parity suite now
  calls each arch kernel *directly* (so AVX2 is covered even on VNNI
  hardware, where the old ForceAvx2 escape hatch was needed) plus the
  dispatcher; (3) attribution is a research question, answered, archived
  here. The ablation stays fully reproducible on this branch.
- Kept: `active_kernel_name()` (the print-what-you-dispatch lesson,
  20 lines), the AVX-512 kernel (feature-gated, honest not-executed-on-CI
  status; #155 ships AVX-512 too), all hard asserts.
- Also fixed while porting: `transpose_cdot`'s doc comment still carried
  the *retracted* "2–16 ms, half the kernel win" transpose claim — now
  states the controlled numbers (~0.3–0.4 ms x86, ~40 µs M4). A retracted
  number almost shipped in a doc comment; grep your docs when you retract.
- `stage1_shortlist` demoted pub → pub(crate) (its `pub` only served the
  profiler); `exact_score_docs{,_prepared,_prepared_t}` and
  `cdot_to_kernel_layout` exist only for harnesses → not in the CR.

**What the CI gates caught (the night's real finding).** `ci.yml` only
triggers on PRs targeting main — fork-branch pushes never ran it. So all
week, "CI green" meant the test + bench workflows, *never* the clippy/doc
gates the upstream PR will face. Exercised them three ways: clippy on
native arm64, clippy on `--target x86_64-apple-darwin` (the runner's
view), and a fork-internal draft PR (joe32140#3, base=fork main) that
runs the real ci.yml. Caught and fixed, all of which would have failed
the upstream PR at `-D warnings`:

1. `clippy::large_enum_variant` on `ScoreQuery` (392 B) — boxed the LUT.
2. 4× `needless_range_loop` in the avx2/avx512 expand/init loops +
   1 in neon (x86 clippy sees the modules local clippy can't) —
   iterator forms.
3. `too_many_arguments` on the safe dispatcher — allowed, as the kernels
   already do.
4. 2× `rustdoc::private_intra_doc_links` (public docs linking private
   `derive_nibble_lut`, `padded_stride`) — caught only by the draft PR's
   Documentation job; local repro added via `RUSTDOCFLAGS="-D warnings"
   cargo doc --no-deps`.

Validation: 146 lib + 4 integration tests pass natively (arm64-verified
binary); clippy clean on both targets; rustdoc clean; fork PR #3 runs the
full matrix. The upstream PR against lightonai:main is *not* opened —
stacked on an unmerged #155 it would show both diffs; the body is ready
in [lut-cr-pr-body.md](lut-cr-pr-body.md) for when #155 lands (Joe's
call).

**Class 4 updated** (nano-plaid 5b537f6): the port epilogue gains "The
ablation: who actually earned the speedup" — the per-CPU attribution
table, the three findings (layout dominant everywhere; the fold's sign
flips with microarchitecture and tr's real job is repairing the M4
regression; chapter 09's 2.1× did not transfer to the runtime-dim shape —
"an optimization's value moves to a new kernel shape as a hypothesis, not
as a number"), and the two harness lessons (route benchmarks through the
measured layout policy; an inert ablation is a free noise floor).

### 2026-07-23 — ablation: what each component actually contributed (M4)

Built `NP_ASYM_ABLATE`, which switches off exactly one component per run in
the *same binary* against the *same* cached indexes, and made the parity
suite run under every mode (an ablation that changed results would make its
own timing meaningless). Native M4, machine idle, r4, 20 queries, exact
kernel ms:

| ablation | scifact | fiqa-15k | component this row adds |
|---|---:|---:|---|
| `row_major` (pre-work kernel) | 3.499 | 3.283 | — baseline |
| `no_vfold` | 2.863 | 2.605 | **centroid-major cdot layout → 1.22× / 1.26×** |
| `no_tr` | 3.583 | 3.150 | vectorized fold → **0.80× / 0.83× (regression!)** |
| *(production)* | 2.730 | 2.679 | transpose-reduce → 1.31× / 1.18× |
| total | | | **1.28× / 1.23×** |

Three things this says, none of which I would have got right by reasoning:

1. **The layout change did the work.** Making the centroid term contiguous
   per token — not any epilogue vectorization — is where the M4 gain lives.
2. **vfold on its own is a regression here** (0.80×): writing every row's
   accumulator to a scratch and re-reading it costs more than folding four
   rows at once saves. It only pays once `tr` removes the round-trip by
   keeping the accumulators in registers.
3. **vfold + tr together ≈ scalar fold + good layout** on M4 (2.730 vs
   2.863 on scifact = 1.05×; 2.679 vs 2.605 on fiqa = 0.97×, i.e. a wash).
   The two epilogue rungs I spent the evening on are, on Apple silicon,
   worth roughly nothing over simply fixing the memory layout — which is
   the same lesson the earlier misses were pointing at and I kept
   attributing to the wrong cause.

Blocked vs naive transpose on M4: prep 146→184 µs (scifact), 157→195 µs
(fiqa) — ~40 µs/query at K=16,384. On x86 the same naive transpose measured
2–16 ms. Same code, ~100× difference in penalty: Apple's memory system
absorbs the strided access that x86's TLB does not.

Harness bug found while doing this: the profiler hard-coded the transpose
outside the timed region, bypassing the ablation switch, so `row_major`
handed the kernel a `[K, nq]` matrix and tripped its own assert — losing
exactly the row that turned out to matter most. Fixed by exporting
`cdot_to_kernel_layout` so benchmarks apply the production layout policy.

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
- ~~**tr rung (transpose-reduce)**~~ — taken same evening (NEON only),
  prediction falsified (~3–5%, not 1.15–1.35×), kept for the consistent
  small win; see log entry. Successor candidate, stated precisely (the
  bench IS dim 128 — the issue is code shape, not data shape): with
  `dim` a runtime value the compiler can neither unroll `while k < dim`
  nor pin the expanded weights in registers, so `w` sits in a stack
  buffer re-loaded 8×32 = 256 times per token; nano's fixed-128 kernel
  loads it into 8 registers once (8 loads) and straight-lines the SDOTs.
  A runtime `dim == 128` fast path (same dispatch pattern as the SIMD
  gate) removes both. Mechanism-based prediction: weight-load traffic
  drops ~32×; combined with **doc-token blocking** (binary's 4 tokens
  per query-row pass, quartering query-row re-streaming) this is the
  load-traffic story that explains binary's residual ~3.5× edge better
  than any epilogue accounting did — both epilogue rungs (vfold, tr)
  shaved instructions while the load stream stayed untouched, which is
  exactly why they underdelivered. AVX2 tr remains gated on a
  reference/measurement.
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

### 2026-07-23 — CI verdict on vfold+tr: the original prediction WAS right, at dataset scale

Compared run 29960353782 (pre-vfold, clean profiler) against 29980662977
(vfold + tr + centroid-major cdot, clean profiler). **Binary is the
control**: its code is untouched and it measures 0.97–1.03× across every
cell — so the asym deltas below are signal, not runner drift.

Exact-kernel speedup, asym (pre → post):

| cell            | x86 AVX2            | Neoverse            |
|-----------------|---------------------|---------------------|
| nfcorpus (0.86M)| 10.0 → 6.5 (1.53×)  | 5.6 → 3.7 (1.52×)   |
| scifact (1.19M) | 17.2 → 9.4 (1.83×)  | 14.4 → 7.4 (1.96×)  |
| fiqa-15k (2.0M) | 23.3 → 8.2 (2.84×)  | 14.0 → 6.3 (2.22×)  |
| fiqa-4k (0.53M) | 12.0 → 7.4 (1.62×)  | 8.3 → 4.6 (1.82×)   |
| fiqa-52k (7.0M) | 13.6 → 8.4 (1.61×)  | 14.0 → 7.2 (1.93×)  |

1.5–2.8×, i.e. **the 1.6–2.1× vfold prediction landed** — the M4's 1.2×
was the outlier, exactly as flagged at the time: at K=4096 the cdot is
cache-resident so the layout half of the change is worth nothing, while
at dataset K (8k–32k) the old per-(row,token) strided gather was the
dominant cost. Asym vs float at the kernel is now **6.1–9.5× on x86**
(was 2.2–6.3×). Lesson for the class: a microbenchmark at the wrong
working-set size can hide the entire effect being measured.

### 2026-07-23 — the transpose bill: cache-blocked, after CI showed it eating half the win

The same comparison exposed the cost the layout change introduced.
Per-query `prep` (which now legitimately carries the [nq,K]→[K,nq]
transpose) went from ~15 µs to **2–16 ms on x86** (0.07–0.44 ms on
Neoverse), and it shows up in production e2e: fiqa-15k r4 saved 15.1 ms
of exact time but only 7.4 ms of e2e; scifact r4 saved 7.8 ms of exact
and 1.2 ms of e2e. Roughly half the kernel win was being handed back.

Cause: `ndarray`'s `.t().as_standard_layout()` walks the destination in
order, so each element read jumps a full source row — at K=16k that is a
~64 KB stride, a cache *and* TLB miss per element, ~500k of them.

Fix: `transpose_cdot`, blocked 64 centroids at a time, so both sides are
sequential (read nq runs of 64 contiguous floats, write 64 runs of nq).
Unit-tested against the naive transpose across multi-block and ragged K
— the search integration tests all use toy indexes whose K fits in one
block, so they could never have caught a blocking bug. Verdict pending
on the next CI run + the M4 overnight.

Note the shape of this: the transpose was invisible while it sat inside
the timed exact region (it just inflated one column); moving it to prep
for *fairness* is what made it measurable. Harness honesty found a real
performance bug.

### 2026-07-22 — opportunity #1 (tr rung, NEON): done — prediction falsified, kept anyway

Ported nano's transpose-reduce to the NEON kernel (AVX2 untouched — no
reference exists, gate stands): query rows in blocks of 4 keep full
accumulator vectors; a `vpaddq` pairwise tree lands the four horizontal
sums in one register (integer adds — lane-identical to the per-row
`vaddvq`), folded by `fold4`; nq%4 tail rows take the vfold path.
Bit-exact across all 105 parity cells (nq 3/7/8/9/32 covers all-tail,
mixed, and vector-only shapes); 196/196.

Predicted 1.15–1.35× with the flat line dropping uniformly. **Measured:
1.79/1.83/1.80 → 1.74/1.76/1.71 ms (~3–5%), binary gauge stable.** The
registered falsifier fired: the per-row reduce was NOT the M4 floor.
Combined with the vfold miss, the revised model: nano's rung gains lived
in fixed-dim-128 straight-line kernels; next-plaid's generic-dim kernel
spends its floor in the `while k < dim` per-iteration branch + loop
machinery, and in re-streaming the query rows once per doc token
(binary's 4-token blocking pays that once per 4). Rung lessons do not
transfer proportionally across kernel shapes — a Class-04-worthy lesson.

Kept: consistent improvement on all rungs, strictly less work, exactness
proven. New evidence-driven candidates for the ledger (not tonight):
dim-128-specialized straight-line inner loop, and doc-token blocking
(opportunity #3) — the two structural differences from the kernels whose
numbers we imported.

### 2026-07-22 — opportunity #2 (alloc/sqw hoist): done, honest negative on M4

Post-review opportunity list, item 2 executed: `sqw` (query-constant
`scales[qi]·lut.scale`) moved into `QueryPlanes`, built once per query
instead of per doc call; kernel `best`/`accs` scratch moved to a
per-rayon-thread `thread_local` reused across the ~1024 per-candidate
calls (kernels size-and-initialize on entry; parity test passes fresh
empty Vecs to prove no state carries).

Predicted 3–8% from alloc-cost arithmetic; **measured: no change on M4**
(asym 1.79/1.83/1.80 vs 1.76/1.77/1.77 baseline, binary noise gauge ±6%).
The model missed because macOS's thread-cached allocator makes repeated
same-size per-doc alloc/free nearly free. Kept anyway: correct, tested
(196/196), removes real churn, and the CI glibc cells (different
allocator economics) are the remaining test of the prediction.

**Opportunity #1 (tr rung) noted, not taken** — see on-hold ledger.

### 2026-07-22 — pre-close implementation review (user: "check the implementation before we close our LUT direction")

Three parallel review angles (kernel line-scan, integration line-scan,
measurement-harness validity; a conventions pass returned clean), each
candidate verified against source. 13 findings, all confirmed. Fixed
immediately (same evening, before the overnight run):

1. **inv-norms OnceLock deadlock** (index.rs) — the initializer ran a
   global-pool `par_iter` inside `get_or_init` while being reachable from
   inside rayon workers; work-stealing during its joins could re-enter the
   OnceLock on the initializing thread → permanent hang under concurrent
   API load on a cold index. Fix: the compute now runs on a dedicated
   one-shot pool, so it depends on no global-pool worker.
2. **Kernel API soundness** (residual_lut.rs) — the vfold commit had
   downgraded out-of-range centroid ids from a clean panic (slice index)
   to raw-pointer OOB reads (UB) from a safe pub fn, and shape mismatches
   (packed width, planes size, scales len, old [nq,K] orientation, dim >
   256) were unchecked or debug-only. Fix: one hard-assert validation
   block in the dispatcher (a pass over ~130 codes/doc — noise), scalar
   path's debug_asserts promoted.
3. **AVX2 fold had zero bit-exact coverage** — parity tests used nq=7/6,
   so the 8-wide vector fold body never executed under `to_bits` equality
   (NEON's 4-wide did). Fix: the parity test now sweeps
   nq ∈ {3,7,8,9,32}, covering both vector bodies and both tails.
4. **Profiler exact-phase contamination** (stage2_profile) — since the
   vfold commit, the timed exact region re-paid the [nq,K]→[K,nq]
   transpose (K-dependent, ~4 MB/call at 52k) already counted in prep,
   inflating asym exact columns and skewing the in-flight pre/post-vfold
   A/B *against* vfold at tall rungs. Fix: new `exact_score_docs_prepared_t`
   takes the pre-transposed matrix; the profiler transposes per query
   outside the timer (prep row still charges it — production pays it).
5. **Bundle/synth cache collision** (maxsim_bench) — INDEX_ROOT dirs were
   named `synth-{n}x{t}_{tag}` even for real-embedding bundles, so the
   planned real-data 786 row would have silently mmap'd synthetic-LCG
   indexes. Fix: bundle mode now prefixes `bundle-{stem}-`; synth naming
   untouched (CI-artifact bit-identity depends on it).
6. **Stale `residual_asym` doc** — still claimed "ignored on the
   batched-centroid path" post-#34. Fixed.

Deferred, deliberately (documented so they aren't forgotten):
- **CI cache-key hardening** — keys carry no builder-code fingerprint
  (`stage2-idx-v1-*`; synth786 has no content hash at all), and partial /
  split-job banking can freeze sibling indexes at different builder
  generations. Real poisoning risk, but adding source hashes tonight
  would invalidate every cache mid-exercise and force multi-hour
  rebuilds; do it as part of upstream-PR polish, with the v-bump
  discipline holding until then (builder untouched since the caches were
  banked — verified).
- **Query length disclosure** — the "mixedbread protocol" rows use a
  32-token query vs their 33×128 (~3% flattering on absolutes, ratios
  unaffected). Not changing mid-exercise (would break comparability with
  every prior run); the final table gets a query-length caveat instead.

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
