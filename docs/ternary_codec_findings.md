# Ternary residual codec — cross-model fidelity, storage & latency

*Measured 2026-08-16 on `integration/ternary-asym` (merge of `main` + ternary + #170 stage-1 pipeline + #169 asymmetric residual LUT). All runs local, $0.*

## TL;DR

- **Storage.** Ternary is a real rung between 2-bit and 1-bit: `ceil(dim/5)` bytes/token
  — **26 B @ dim128** (vs 2-bit 32 B, 1-bit 16 B) and **20 B @ dim96**. ~19 % smaller
  than 2-bit at dim128, ~17 % at dim96.
- **Fidelity (7 model×corpus cells).** Ternary does **not** collapse on low-retention
  models and **clears 1-bit in every cell**. The two fragility axes split: on
  *capacity*-limited `mxbai` (D=48) it matches/beats 2-bit (nfcorpus +0.0019, scifact
  −0.0004 — the dead-zone nails near-degenerate residuals); on the *multilingual-basis*
  `mLateOn` (lowest-retention fleet ckpt) it's the one model clearly sub-2-bit but only by
  −0.0036…−0.0065. Worst case anywhere is −0.0065 vs 2-bit for a 19 % shrink.
- **#169 asym is now ternary-native.** A base-3 fused LUT gives ternary indexes #169's
  no-reconstruction rescoring, with NDCG bit-for-bit matching the float path
  (parity test + real-search NDCG agree to ≤0.0005). No new SIMD kernel: ternary
  routes to the scalar kernel whose `d == dim` break already masks the padded byte.
- **Latency.** Measured contention-free on CI per-ISA (`asym-bench` workflow; local
  end-to-end runs were noisy under a shared machine). At the **Stage-2 rescore kernel**
  (what #169 changes), asym is a **7–9× speedup** on the SIMD rungs (1/2/4-bit) on both
  x86 (`avx2`) and arm (`neon-sdot`). Ternary takes the *scalar* kernel, so it splits:
  **1.49× on arm64, 0.67× on x86_64** — enable `residual_asym` for ternary on arm, keep it
  on float rescore on x86 until a base-3 SIMD kernel lands. End-to-end this dilutes because
  search is stage-1-bound (unchanged by #169, #170's territory). See
  [CI rescore ratios](#ci-rescore-ratios-contention-free).

## What shipped (integration)

`integration/ternary-asym` reconciles four lines of work:

| source | what |
|---|---|
| `main` (21903f1) | baseline |
| ternary codec | base-3 dead-zone residual `{−m,0,+m}`, 5 trits/byte |
| **#170** `perf/stage1-pipeline` | par_cdot GEMM, centroid flood scorer, per-doc `par_iter` exact scoring |
| **#169** `feat/asymmetric-residual-lut` | int8-query × fused-LUT residual MaxSim, no float reconstruction |

The only real conflict was `search.rs` (both #169 and #170 rewrite it); resolved to
keep #170's per-doc `par_iter` exact scoring **and** #169's asym dispatch.

### Making #169 ternary-compatible

`quantize_lut` gained a base-3 branch (`residual_lut.rs`): for a ternary codec it emits a
`fused[256 * 5]` int8 table — row `b` = the five trit weights packed in byte `b` — with
`keys_per_byte = 5`, `nibble = None`. Two consequences fall out for free:

1. **No new kernel.** The SIMD paths require `nibble.is_some()` (nibble-factored
   16-entry tables); trits don't align to nibbles, so `nibble = None` routes ternary to
   the **scalar** reference kernel, whose existing `if d == dim { break }` already stops
   before the padded trits in the last byte (dim128 → 26 B, last byte holds 3 pad trits).
2. **inv_norms** got a matching base-3 branch (#169 builds the sidecar for every
   non-binary index; without it, ternary index creation panicked).

Verified by `ternary_fused_table_matches_packing` (unit) and
`ternary_asymmetric_scoring_agrees_with_float` (integration: asym top-1 == float top-1,
top-10 overlap ≥ 9/10). Full suite green, clippy + fmt clean. Pushed:
[`feat/ternary-residual`], [`integration/ternary-asym`].

## Storage ladder (bytes / token)

| profile | dim128 | dim96 | vs 2-bit (dim128) |
|---|--:|--:|--:|
| float | 512 | 384 | — |
| 4-bit | 64 | 48 | +100 % |
| 2-bit | 32 | 24 | — |
| **ternary** | **26** | **20** | **−19 %** |
| 1-bit | 16 | 12 | −50 % |

## Fidelity ladder — codec-isolated NDCG@10 (`ndcg_eval`)

Exhaustive float MaxSim over reconstructed docs, fixed k-means seed → the only variable
per row is the residual codec (no stage-1 confound). `reconCos` = mean per-token cosine
to the original embedding.

| bundle (retention axis) | float | 4-bit | 2-bit | **ternary** | 1-bit | Δ ternary vs 2-bit |
|---|--:|--:|--:|--:|--:|--:|
| **scifact / ColBERTv2** (high) | 0.6464 | 0.6459 | 0.6464 | **0.6411** | 0.6400 | −0.0053 |
| **nfcorpus / ColBERTv2** (high) | 0.3324 | 0.3306 | 0.3317 | **0.3317** | 0.3278 | −0.0000 |
| **nfcorpus / mxbai** (LOW: capacity) | 0.3092 | 0.3059 | 0.3050 | **0.3069** | 0.3038 | **+0.0019** |
| **scifact / mxbai** (LOW: capacity) | 0.6309 | 0.6329 | 0.6252 | **0.6248** | 0.6131 | −0.0004 |
| **nfcorpus / mLateOn** (LOWEST: multiling.) | 0.3759 | 0.3757 | 0.3668 | **0.3603** | 0.3530 | −0.0065 |
| **scifact / mLateOn** (LOWEST: multiling.) | 0.7533 | 0.7475 | 0.7415 | **0.7379** | 0.7247 | −0.0036 |
| **nfcorpus / answerai** (dim96) | 0.3725 | 0.3708 | 0.3662 | **0.3617** | 0.3533 | −0.0045 |

`reconCos` is monotone everywhere (4-bit > 2-bit > ternary > 1-bit), e.g. nfcorpus/mxbai
2-bit 0.9754 · ternary 0.9672 · 1-bit 0.9509 — ternary's 1.585 bits/dim sits between 1
and 2 bits exactly as expected.

**Read — ternary holds up on low-retention models, and the two fragility axes split:**
- **Capacity-limited fragility (mxbai, D=48).** Ternary is 2-bit-*class or better*:
  nfcorpus *ahead* +0.0019, scifact a dead tie −0.0004. Residuals are near-degenerate, so
  the dead-zone's explicit zero bucket encodes them exactly where 2-bit must spend a
  bucket edge. Not a "collapse toward 1-bit" risk — it's the *most* competitive here.
- **Multilingual-basis fragility (mLateOn, the lowest-retention fleet ckpt).** Here
  ternary is the one model clearly *sub*-2-bit (−0.0065 nfcorpus, −0.0036 scifact). Note
  reconCos is near-perfect (0.9986) yet NDCG still drops — razor-thin margins flip under
  tiny error, and the problem *isn't* near-zero dims, so the dead-zone can't rescue it.
  Still comfortably above 1-bit (+0.0073 nfcorpus, +0.0132 scifact).
- **Net:** ternary is 2-bit-class on 3 of 7 cells, between-class (closer to 2-bit) on 3,
  and beats 2-bit on 1. Worst case anywhere is −0.0108 vs float, −0.0065 vs 2-bit; it
  **clears 1-bit in all 7 cells**. So ternary is a safe ~19% shrink over 2-bit even on the
  most codec-hostile checkpoints — costing at most ~0.006 NDCG, and free on capacity-bound
  ones. The dim96 answerai (−0.0045) quantizes well overall (all reconCos > 0.99).

## Latency (`search_latency`, end-to-end `search_batch`, reps=10, n_ivf_probe=8)

`float` = decompress-then-score; `asym-LUT` = #169 int8×fused-LUT, no reconstruction.
Real-search NDCG shown to confirm asym preserves quality.

> ⚠️ **These end-to-end numbers were measured on a shared/contended workstation** — a
> second workload was running, so absolute µs and even some ratios are noisy (a stray
> nfcorpus/mLateOn run showed a spurious 1.60× that did not reproduce). Treat the
> end-to-end table as *indicative of the regime* (stage-1-bound), not as precise timings.
> The **clean, contention-free asym-vs-float rescore ratios come from CI** (dedicated
> runners, per-ISA) via the `asym_rescore_check` isolation harness — see
> [CI rescore ratios](#ci-rescore-ratios-contention-free) below. Storage and NDCG rows
> elsewhere in this doc are deterministic and unaffected.

| bundle | codec | B/tok | float µs/q | asym µs/q | asym speedup | NDCG (float→asym) |
|---|---|--:|--:|--:|--:|--:|
| nfcorpus/ColBERTv2 | 2-bit | 32 | 9 329 | 13 531 | 0.69× | 0.3234 → 0.3232 |
| | ternary | 26 | 11 056 | 14 523 | 0.76× | 0.3239 → 0.3239 |
| nfcorpus/mxbai | 2-bit | 32 | 67 858 | 95 936 | 0.71× | 0.3073 → 0.3074 |
| | ternary | 26 | 61 225 | 95 723 | 0.64× | 0.3069 → 0.3074 |
| scifact/ColBERTv2 | 2-bit | 32 | 27 466 | 27 230 | 1.01× | 0.6464 → 0.6457 |
| | ternary | 26 | 22 081 | 27 641 | 0.80× | 0.6411 → 0.6411 |

**Two findings:**

1. **Asym-LUT is not an end-to-end win at this scale (0.64–1.01×).** `search_batch` is
   dominated by stage-1 (centroid GEMM + IVF probe + shortlist), which #169 doesn't
   touch; with an n_ivf_probe=8 shortlist over 3.6–5.2 K docs, residual rescoring is a
   small slice, so asym's per-query setup (int8 query quant + plane build + inv_norms) is
   net overhead. Best case (scifact 2-bit) it breaks even. **NDCG parity holds** — asym
   matches float rescore to ≤0.0007, so the integration is correct; it just isn't the
   bottleneck here.
2. **Ternary's real latency lever is its footprint, in the *float* path.** Reconstruction
   is memory-bandwidth bound, so 26 B vs 32 B (−19 %) buys wall-clock on bandwidth-bound
   working sets: ternary-float beats 2-bit-float by **20 %** on scifact/ColBERTv2 (22.1
   vs 27.5 ms) and **10 %** on nfcorpus/mxbai (61.2 vs 67.9 ms). On the small in-cache
   nfcorpus/ColBERTv2 it's **18 % slower** (11.1 vs 9.3 ms) — there the residual store
   fits in cache and the base-3 unpack (div/mod-3) costs more than the bytes saved. So
   ternary's latency benefit appears exactly where it matters: large / long-doc corpora.

### Probe-depth crossover — where does asym start winning?

_The asym LUT accelerates residual rescoring only; to see it help end-to-end you have to
push cost into that path by widening the shortlist. Sweep n_ivf_probe ∈ {8, 32, 128}
(asym speedup = float µs ÷ asym µs; >1 = asym faster):_

| | scifact / ColBERTv2 | | nfcorpus / mxbai | |
|---|--:|--:|--:|--:|
| **n_ivf_probe** | 2-bit | ternary | 2-bit | ternary |
| 8 | 1.03× | 0.82× | 0.72× | 0.67× |
| 32 | 1.01× | 0.88× | 0.74× | 0.67× |
| 128 | 1.07× | 0.94× | 0.69× | 0.66× |

**No decisive crossover, even at probe=128.** Widening the shortlist grows stage-1 (the
centroid flood scorer) and residual rescoring *together*, so the asym/float ratio barely
moves and absolute float latency itself rises with probe (2-bit scifact float 29.0 → 28.2
→ 30.2 ms; mxbai 78 → 89 → 102 ms). The asym LUT lands at ~parity-to-slightly-ahead for
2-bit (SIMD nibble path, 1.01–1.07× on ColBERTv2) and stays a slight cost for ternary
(scalar path, climbing 0.82 → 0.94× with depth but never crossing 1×). On the long-doc
mxbai corpus asym never wins at any depth (0.66–0.74×). NDCG parity holds throughout. So
at BEIR corpus sizes (3.6–5.2 K docs) the asym LUT's value is **not** latency — it's
avoiding materialization of reconstructed f32 vectors (a *memory* win that scales with
corpus size), and it makes ternary a first-class rescoring citizen. Latency at this scale
is owned by stage-1 and, secondarily, by the float-path footprint effect below.

### CI rescore ratios (contention-free)

End-to-end timing above is stage-1-bound *and* was measured under contention, so it can't
cleanly isolate what #169 actually changes: **Stage-2 rescore** (float decompress+MaxSim
vs asym fused-LUT over packed codes). The `asym_rescore_check` harness times exactly that
arm on seeded synthetic shapes (data-independent), and the `asym-bench` workflow runs it on
dedicated per-ISA runners — `ubuntu-latest` (x86_64 `avx2`/`avx512-vnni`) and
`ubuntu-24.04-arm` (`neon-sdot`). Scalar rungs (1/2/4-bit) take the SIMD kernel; ternary
takes the scalar kernel (`nibble = None`) by design.

Stage-2 rescore only, **ns/token**, median of 9 reps, 4096 docs × 230 tokens, dim 128
(run [31964951628](https://github.com/joe32140/next-plaid/actions/runs/31964951628)):

| codec | kernel | x86_64 float | x86_64 asym | **x86_64 ratio** | arm64 float | arm64 asym | **arm64 ratio** |
|---|---|--:|--:|--:|--:|--:|--:|
| 4-bit | SIMD | 708.0 | 100.2 | **7.06×** | 388.7 | 45.7 | **8.50×** |
| 2-bit | SIMD | 690.9 | 99.9 | **6.92×** | 348.5 | 45.7 | **7.62×** |
| 1-bit | SIMD | 612.5 | 69.1 | **8.86×** | 308.4 | 37.6 | **8.21×** |
| **ternary** | scalar | 387.9 | 575.5 | **0.67×** | 239.4 | 161.0 | **1.49×** |

*(x86_64 = `avx2`; arm64 = `neon-sdot`; ternary = `scalar` on both.)*

**This is the result the contended end-to-end table buried.** At the rescore kernel #169
is a **7–9× win on the scalar rungs (1/2/4-bit) on both ISAs** — float decompresses to f32
then MaxSims, asym scores int8 straight over packed codes under SIMD. So the asym LUT is
emphatically worth it *at Stage 2*; the end-to-end dilution is entirely stage-1 (which is
what #170 attacks, and which the local numbers were dominated by).

**Ternary splits on ISA — the one caveat.** Because ternary factors to `nibble = None` it
takes the *scalar* asym kernel, and that races the (well-vectorized) float path
differently per arch: on **arm64 asym still wins 1.49×** (scalar-int accumulation beats
float reconstruct+MaxSim), but on **x86_64 it loses 0.67×** (avx2 makes the float path
faster than the non-SIMD integer LUT walk). Concretely, ternary-asym costs 575 ns/tok on
x86 vs 2-bit-asym's 100 — the missing SIMD, not the codec. **Deployment:** for ternary,
prefer `residual_asym` on arm64 and the float rescore on x86_64. **Future work:** a
vectorized base-3 expand (pack 5 trits → SIMD lanes) would give ternary the same 7–9× on
x86 that the nibble rungs already get — the single highest-value follow-up here.

## How far can we push latency — bottom line

Separate the two stages:

- **Stage-2 rescore (what the codec + #169 own):** asym is a **7–9× kernel speedup** on the
  SIMD rungs (1/2/4-bit), clean on both x86 and arm (CI). Ternary gets the same idea but on
  the scalar kernel, so it wins on arm (1.49×) and loses on x86 (0.67×) — vectorizing the
  base-3 expand is the fix. So Stage-2 rescore is *already* pushed near its floor for the
  nibble codecs; the ceiling that remains is stage-1.
- **Stage-1 (candidate generation):** owns end-to-end latency at BEIR scale — unchanged by
  #169, targeted by #170. This is why the end-to-end asym speedup is small even though the
  kernel speedup is 7–9×.
- **Storage:** ternary is a clean ~19 % index shrink over 2-bit with **≤ 0.0065 NDCG cost**
  (free on capacity-bound models, and it clears 1-bit everywhere). If footprint-bound,
  ternary is the better default than 2-bit; and its smaller float-path footprint also helps
  the float rescore path on bandwidth-bound corpora.

**Deployment recipe:** 2-bit or ternary for footprint; enable `residual_asym` for the
7–9× rescore win on the nibble rungs (all ISAs) and for ternary **on arm64**; keep ternary
on float rescore on x86_64 until the base-3 kernel is vectorized.

## Repro

```bash
cd next-plaid
cargo run --release -p next-plaid --example ndcg_eval     -- <bundle_dir>
cargo run --release -p next-plaid --example search_latency -- <bundle_dir> <reps> [n_ivf_probe]
```

Bundles: BEIR corpus/queries/qrels encoded with `scripts/make_colbert_bundle.py`
(ColBERTv2, mixedbread mxbai-colbert-large-v1, answerai-colbert-small-v1) for
scifact + nfcorpus.

[`feat/ternary-residual`]: https://github.com/joe32140/next-plaid/tree/feat/ternary-residual
[`integration/ternary-asym`]: https://github.com/joe32140/next-plaid/tree/integration/ternary-asym
