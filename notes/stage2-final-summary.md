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

| phase (real shapes) | share of the float exact path |
|---|---|
| decompress to f32 | **65–84%** |
| GEMM + max | 16–35% |

This is why "we beat BLAS" would be the wrong claim. We don't out-multiply
BLAS — we skip the step before it.

---

## 3. Name your baseline

The same binary kernel, three honest headlines:

| float baseline | speedup | note |
|---|---:|---|
| decompress + matrixmultiply (our deployed path) | 13–28× | the cost actually being removed |
| raw f32 + vendor BLAS (never compressed) | ~3–4.4× | reproduces mixedbread's published 3.82× |
| raw f32 + Apple AMX | ~1.1–1.8× | Apple's matrix unit ≈ our kernels on raw floats |

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

| | M4 (prep µs) | x86 (prep) |
|---|---:|---|
| blocked (production) | 146–157 | µs-scale |
| naive `as_standard_layout` | 184–195 | **2–16 ms** |
| penalty | ~40 µs | up to ~100× worse |

Same code, wildly different penalty: Apple's memory system absorbs the
strided access that x86's TLB does not. A benchmark on one machine would
have mis-ranked this change completely.

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
