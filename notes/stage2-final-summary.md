# Stage-2 quantization for PLAID — final summary

What this exercise set out to answer, and what the measurements say. Every
number here comes from a committed harness; the companion
[worklog](stage2-exercise-log.md) records the order things happened in,
including the predictions that were wrong.

**Scope of the claim.** Quality: 3 checkpoints × 3 datasets × 4 schemes,
real embeddings, bootstrap CIs. Latency: shape-replay corpora (real token
length distributions, synthetic values — latency depends on shapes, not
values) on 4 CPUs: x86 AVX2, x86 AVX-512 VNNI, Neoverse (Graviton-class),
Apple M-series. Shared CI runners drift 8–20% run to run, so **ratios are
the signal and absolutes are context**.

---

## 1. The scheme table

| scheme | B/token (dim 128) | vs f32 | quality |
|---|---:|---:|---|
| raw f32 | 512 | 1× | reference |
| float r4 (decompress + GEMM) | 72 | 7.1× | the deployed baseline |
| **fused int8 on r4 / r2 / r1** | 72 / 40 / 24 | 7.1 / 12.8 / 21× | \|Δ\| ≤ 0.0021 NDCG@10 vs float on identical codes |
| **binary (int8 × 1-bit)** | 24 | 21× | model-dependent: 83–99% / 73–90% / 20–37% retention |

The fused int8 path is **compute-only**: same index, same bytes on disk, so
it can be A/B'd per search with `residual_asym`.

### Quality, with error bars (9 cells, 10k paired bootstrap resamples)

| contrast | result |
|---|---|
| asym − float (27 contrasts) | all \|Δ\| ≤ 0.0021; 3 significant, **all favouring asym** |
| binary − r1 (same 24 B/token) | lateon ~tie, gte −0.056..−0.083\*, edge17m −0.23..−0.45\* |
| r4 − r1 | +0.007..+0.028, significant in 7/9 |

The headline: **asym is free quality-wise; binary is a per-model bet.**
nano-plaid's "binary beats r1 on SciFact" does *not* generalise across
checkpoints — that finding is why the grid exists.

---

## 2. Where float time actually goes

Decompression, not arithmetic, is the cost the fused kernels delete:

| phase (real shapes, float r4 exact path) | x86 | Apple M4 |
|---|---|---|
| decompress to f32 | **65–84%** | **48–72%** |
| GEMM + max | 16–35% | 21–36% |

This is why "we beat BLAS" would be the wrong claim. We don't out-multiply
BLAS — we skip the step before it.

The share is platform- and corpus-dependent, and it *falls* as the corpus
grows (72% → 48% across the M4 ladder), because larger corpora bring longer
candidate documents and the GEMM scales worse than the byte-unpacking. That
decay is the leading indicator of the fused kernels' ceiling: the win is
bounded by whatever fraction decompression represents.

---

## 3. Name your baseline

The single most misleading thing you can do with these kernels is quote a
speedup without saying what it is against. Native M4, mixedbread's protocol
(1000 docs × 786 tokens, one 32-token query, median of 9):

| | vs float r4 (decompress+GEMM) | vs raw f32, same GEMM |
|---|---:|---:|
| **matrixmultiply build** | | |
| float r4 (decompress + GEMM) | 57.75 ms — 1.00× | — |
| raw f32, never compressed | 15.29 ms | 1.00× |
| asym r4 (int8×LUT) | 21.62 ms — **2.67×** | **0.71×** |
| binary (int8×1-bit) | 7.06 ms — **8.17×** | **2.17×** |
| **Accelerate/AMX build** | | |
| float r4 (decompress + GEMM) | 29.21 ms — 1.00× | — |
| raw f32, never compressed | 8.11 ms | 1.00× |
| asym r4 | 13.75 ms — **2.12×** | **0.59×** |
| binary | 6.15 ms — **4.75×** | **1.32×** |

Read the two columns together. Against the path a compressed deployment
actually runs, asym is 2.1–2.7× and binary 4.8–8.2×. Against a system that
never compressed at all and has Apple's matrix unit, **asym is slower than
float** (0.59–0.71×) and binary's edge shrinks to 1.3×.

Both are true. The fused kernels buy their speed by not decompressing, so
they win exactly to the extent that decompression is in your baseline —
which is why the phase decomposition in §2 is the load-bearing measurement,
not the speedup number.

---

### Native M4, real-shape corpora (production tree, idle machine)

Exact stage-2 kernel, and the whole query through the public API:

| cell (tokens) | float exact | asym exact | binary exact | float e2e | asym e2e | binary e2e |
|---|---:|---:|---:|---:|---:|---:|
| nfcorpus (0.86M) | 9.98 | **1.77** (5.6×) | **0.67** (14.9×) | 15.1 | 4.5 | 3.0 |
| scifact (1.19M) | 12.83 | **2.72** (4.7×) | **0.96** (13.4×) | 21.1 | 8.7 | 6.2 |
| fiqa-15k (2.0M) | 14.31 | **2.78** (5.1×) | **1.22** (11.7×) | 24.0 | 10.2 | 10.2 |
| fiqa-4k (0.53M) | 19.68 | **3.98** (4.9×) | **1.16** (17.0×) | 25.8 | 8.6 | 5.5 |
| fiqa-52k (7.0M) | 25.63 | **4.42** (5.8×) | **1.26** (20.3×) | 47.6 | 25.9 | 21.0 |

All times ms/query. The e2e columns are where Amdahl shows up: at fiqa-52k
the kernel is 5.8×/20× faster but the query is only 1.8×/2.3× faster,
because stage-1 is most of it.

---

## 4. Amdahl: the ratio decays, the saving does not

Stage-2 is bounded by stage-1's share of the query. Binary's end-to-end
advantage over float, measured across a corpus ladder of one dataset:

| tokens (K centroids) | x86 | Neoverse | M-series | stage-1 share of binary's query |
|---|---:|---:|---:|---:|
| 0.53M (8,192) | 7.0× | 4.9× | 6.2× | ~65% |
| 2.0M (16,384) | 3.8× | 2.2× | 3.5× | ~84–92% |
| 7.0M (32,768) | **2.35×** | **1.57×** | **2.00×** | **89–98%** |

Predicted in advance from K = 2^floor(log2(16·√n)) and confirmed. The
*absolute* saving (~60 ms/query on x86) is scale-invariant; only the ratio
shrinks, because stage-1 grows in both numerator and denominator. **The
next real win is stage-1, not stage-2.**

---

## 5. Component attribution

`NP_ASYM_ABLATE` switches off exactly one component per run — same binary,
same cached indexes, same queries — and the bit-exactness suite runs under
every mode, so an ablation cannot quietly change the computation it is
timing. Each row below adds one component to the row above it.

### Apple M4 (native arm64, idle machine, r4, exact-kernel ms)

| ablation | scifact | fiqa-15k | this row adds | gain |
|---|---:|---:|---|---|
| `row_major` | 3.499 | 3.283 | *(the kernel before this work)* | baseline |
| `no_vfold` | 2.863 | 2.605 | centroid-major cdot layout | **1.22× / 1.26×** |
| `no_tr` | 3.583 | 3.150 | vectorized fold | **0.80× / 0.83×** ⚠ |
| *(production)* | 2.730 | 2.679 | transpose-reduce | 1.31× / 1.18× |
| | | | **total** | **1.28× / 1.23×** |

Three findings, none of which I predicted correctly:

1. **The memory layout did the work.** Making a token's centroid scores
   contiguous across query rows — not any epilogue vectorisation — is where
   the gain lives.
2. **The vectorised fold alone is a *regression* on M4** (0.80×, reproduced
   in 2 further repeats: 2.69/2.99 scalar vs 3.43/3.69 vfold). Writing each
   row's accumulator to a scratch and reading it back costs more than
   folding four rows at once saves.
3. **`vfold` + `tr` together ≈ scalar fold + good layout** (2.73 vs 2.86,
   and 2.68 vs 2.61 — a wash). The two epilogue rungs are worth roughly
   nothing on Apple silicon once the layout is right.

The honest reading: I imported two optimisations from a sibling project
where they were measured to pay, and on this kernel shape they did not —
while the unglamorous change I made *in passing* to enable them (the
layout) was the entire win.

### Transpose implementation

Making the centroid term contiguous requires transposing stage-1's matrix
once per query. How that transpose is written matters enormously, and
differently per platform:

| per-query prep | M4, K=16k | M4, K=32k | x86, K=16k |
|---|---:|---:|---|
| blocked (production) | 146–157 µs | 191 µs | (see CI) |
| naive `as_standard_layout` | 184–195 µs | 238 µs | **0.7–15.8 ms** |
| penalty | ~40 µs | ~47 µs | **milliseconds** |

Same code, wildly different penalty. Apple's memory system absorbs the
strided access almost entirely (~50 µs to transpose 4 MB); on x86 the same
access pattern costs milliseconds, because every element read is both a
cache and a TLB miss. Had this been benchmarked only on the Mac, the
transpose would have looked free and shipped as-is — and x86 users would
have paid half the kernel win back without anyone seeing it.

---

## 6. Platform coverage

| CPU | binary kernel | asym kernel | status |
|---|---|---|---|
| x86 AVX2 | AVX2 SAD | `pshufb` + `maddubs` | shipped, CI-verified |
| x86 AVX-512 VNNI | `vpdpbusd` | **`vpdpbusd` + 16-wide fold (new)** | see §6.1 |
| aarch64 (Neoverse) | SDOT | `tbl` + SDOT + tr | shipped, CI-verified |
| Apple M-series | SDOT | `tbl` + SDOT + tr | shipped, verified natively |

### 6.1 AVX-512 bridge

The residual-LUT path had no AVX-512 kernel (the binary path already had
one). The new kernel keeps the expand at 128-bit `pshufb` — it is charged
once per doc token and amortised over every query row — and moves the part
charged per *(query row, token)*, ~32× more often, to 64-lane `vpdpbusd`
plus a 16-wide fold.

`vpdpbusd` multiplies unsigned × signed, so it gets `|w|` against
`sign(w)·q`. There is no 512-bit `vpsignb`, so the sign is applied with a
mask (`movepi8_mask` + `mask_sub_epi8`). Two things make this exact: lanes
where `w == 0` need no handling because `|w| = 0` zeroes the product
regardless, and `-128` never occurs on either side (both clamp to ±127 at
quantisation), so the negation cannot overflow.

<!--AVX512-STATUS-->

---

## 7. Bugs this exercise found

Found by review and by harness discipline, not by tests failing:

| # | bug | how it would have hurt |
|---|---|---|
| 1 | `residual_asym` silently ignored above ~335k docs (batched path hard-coded `false`) | the flagship feature reverts to float at exactly the scale it matters |
| 2 | inv-norms `OnceLock` initialised with a rayon parallel loop, reachable from inside rayon workers | permanent deadlock on a cold index under concurrent load |
| 3 | kernel took centroid ids as raw pointer offsets with no bounds check | out-of-bounds reads (UB) from a safe public function |
| 4 | AVX2 8-wide fold had zero bit-exact coverage (parity test used nq=7) | a wrong x86 fold would pass the whole suite |
| 5 | cdot transpose was eating ~half the kernel win at e2e | we would have shipped a kernel gain the user never sees |
| 6 | dim-48 packed rows fell off the SIMD path | narrow-dim models silently slower |

Plus measurement traps caught before they corrupted a claim: rustup
defaulting to x86-under-Rosetta on Apple silicon (bit us again in this
session — a fresh worktree had no override and silently benchmarked the
scalar path); ORT ≤1.20 collapsing int8 embeddings unless an fp32 session
runs first; OpenBLAS oversubscribing against rayon; jetsam killing builds
with clean-looking logs; `.gitignore` swallowing the shape manifests; a
profiler double-counting prep inside the timed region.

---

## 7.1 What this exercise taught about measuring

Five lessons, each paid for with a wrong prediction:

1. **Benchmark at the working-set size you deploy at.** The vectorised-fold
   change measured 1.2× on a local synthetic cell (K = 4,096, centroid
   matrix cache-resident) and 1.5–2.8× on real dataset cells (K = 8k–32k).
   Same code. The small benchmark hid the entire effect, because the effect
   *was* a memory-access change.
2. **Port measured results, not measured conclusions.** Two optimisations
   imported from a sibling project — both genuinely measured there — did
   not reproduce here, because that kernel is fixed-dim and fully unrolled
   and ours is not. The rung ladder transferred; the numbers did not.
3. **Keep a control in every comparison.** The binary kernel was untouched
   all session, so its 0.97–1.03× across every cell is what licenses
   reading the asym deltas as signal rather than runner drift.
4. **A fair harness finds real bugs.** The cdot transpose was invisible
   while it sat inside the timed exact region — it just inflated one
   column. Moving it out *for fairness* is what exposed it as a genuine
   per-query cost eating half the win.
5. **Ablations must be proven inert.** Every `NP_ASYM_ABLATE` mode runs the
   full bit-exactness suite in CI. An ablation that quietly changed the
   computation would produce a confident, meaningless number.

---

## 8. What is deliberately not done

| item | why it is parked |
|---|---|
| stage-1 optimisation | the honest next lever (84–98% of the query at scale); the candidate-flood approximate scorer dominates, not the cdot GEMM |
| dim-128 specialised kernel | with `dim` a runtime value the compiler can neither unroll the dot loop nor pin the expanded weights in registers, so `w` is re-loaded 8×nq times per token instead of 8 |
| doc-token blocking | binary scores 4 tokens per query-row pass; asym re-streams query rows per token — the structural half of binary's remaining edge |
| CI cache keys carry no builder fingerprint | a builder change would silently serve stale indexes; the v-bump discipline holds meanwhile |
| inv-norms persistence | first-query spike 0.04–3.7 s, grows linearly; becomes necessary ≥50M tokens |

---

## 9. Reproducing

```bash
# quality grid (Modal, GPU embeds + 3x3x4 sweep)
python scripts/modal_quant_grid.py

# per-platform latency, from cached indexes (no local index builds)
NO_BUILD=1 INDEX_ROOT=indexes \
  cargo run --release --example stage2_profile -- shapes next-plaid/examples/shapes/scifact

# component attribution: one switch off at a time, same binary
NP_ASYM_ABLATE=row_major|no_vfold|no_tr|naive_transpose|force_avx2 <same command>

# every ablation must stay bit-exact
NP_ASYM_ABLATE=<mode> cargo test --release -p next-plaid --lib residual_lut
```
