# Ternary residual codec — cross-model fidelity, storage & latency

*Measured 2026-08-16/17 on `integration/ternary-asym` (merge of `main` + ternary + #170 stage-1 pipeline + #169 asymmetric residual LUT). Quality local ($0); latency on CI, per-ISA.*

## TL;DR

- **Storage.** Ternary is a real rung between 2-bit and 1-bit: `ceil(dim/5)` bytes/token
  — **26 B @ dim128** (vs 2-bit 32 B, 1-bit 16 B) and **20 B @ dim96**. ~19 % smaller
  than 2-bit at dim128, ~17 % at dim96.
- **The dead-zone width is the whole ballgame.** `create_with_kmeans` used to split at the
  1/3 and 2/3 residual quantiles — equal-mass buckets, ~⅓ of dims dead by construction,
  which is *not* where a residual distribution wants its dead zone. Setting it explicitly
  (`ternary_tau`, `|r| < τ·σ` stores 0) at **τ = 0.65** turns ternary from *losing* to 2-bit
  in 4 of 5 measured cells into **beating it in 5 of 5** — and lands it at **4-bit's mean
  retention for 40 % of 4-bit's bytes**. This was the single largest effect in the study.
- **Fidelity.** At τ = 0.65, mean NDCG retention over the 5 re-measured cells is **99.56 %**
  against 4-bit's 99.56 % and 2-bit's 98.86 %, at 26 B against 64 B and 32 B. Ternary clears
  1-bit in every cell measured, at any τ.
- **#169 asym is ternary-native, SIMD included, in one hop.** A base-3 fused LUT gives
  ternary indexes #169's no-reconstruction rescoring, bit-for-bit against the scalar
  reference. Base-3 cannot nibble-factor (243 values, and every SIMD byte-shuffle is
  16-entry), so ternary expands via a **256-entry byte→weights table copied straight into
  the kernel's weight buffer** — see [Making #169 ternary-compatible](#making-169-ternary-compatible).
- **Latency.** `residual_asym` is a **2.7–5.3× rescore win for every codec on both ISAs**.
  Ternary costs a little more than 2-bit to expand, and *how much* is strongly
  microarchitecture-dependent: end-to-end at probe 8, **+1.7 % on arm (Neoverse-N2)** and
  **+7.2 % on x86 (EPYC 7763, avx2)**. Without asym, ternary is the **fastest** of the four
  codecs *and* the second smallest. Enable `residual_asym` everywhere regardless.

## r=1 vs r=2 vs ternary — head to head

Storage is exact. NDCG is codec-isolated (fixed k-means seed, exhaustive float MaxSim over
reconstructions, so no stage-1 confound). Latency is CI, all four codecs **in one process**
— see the [measurement note](#ci-rescore-ratios-contention-free) for why that matters.

| | **r=1** (1-bit) | **ternary** @ τ=0.65 | **r=2** (2-bit) | r=4 (4-bit) |
|---|--:|--:|--:|--:|
| bits / dim | 1.000 | 1.585 | 2.000 | 4.000 |
| **B/token @ dim128** | **16** | **26** | **32** | 64 |
| B/token @ dim96 | 12 | 20 | 24 | 48 |
| vs r=2 storage | −50 % | **−19 %** | — | +100 % |
| **mean NDCG retention** (5 cells) | 96.93 % | **99.56 %** | 98.86 % | 99.56 % |
| worst cell | 93.91 % | **98.71 %** | 97.58 % | 98.93 % |
| cells where it beats r=2 | 0 / 5 | **5 / 5** | — | 4 / 5 |
| rescore ns/tok — x86 `avx2` | 93.8 | 108.1 | 93.8 | 96.4 |
| rescore ns/tok — arm `neon-sdot` | 55.9 | 56.3 | 52.4 | 56.6 |
| asym vs float — x86 / arm | 4.47× / 3.61× | 2.68× / 3.24× | 4.74× / 4.26× | 5.27× / 4.12× |

**NDCG@10 per cell, and Δ vs 2-bit** (the number that decides whether 19 % fewer bytes is
free). Positive = ternary wins at less storage:

| bundle (fragility axis) | dim | float | r=2 | tern (default τ) | **tern @ τ=0.65** | Δ vs r=2 |
|---|--:|--:|--:|--:|--:|--:|
| nfcorpus / mLateOn (basis) | 128 | .3759 | .3668 | .3603 | **.3723** | **+0.0055** |
| nfcorpus / mxbai (capacity) | 128 | .3092 | .3050 | .3069 | **.3093** | **+0.0043** |
| nfcorpus / answerai | 96 | .3725 | .3662 | .3617 | **.3677** | **+0.0015** |
| nfcorpus / ColBERTv2 | 128 | .3324 | .3317 | .3317 | **.3323** | **+0.0006** |
| scifact / ColBERTv2 | 128 | .6464 | .6464 | .6411 | **.6466** | **+0.0002** |
| *mean Δ vs r=2* | | | — | *−0.0029* | | ***+0.0024*** |

**Reading it:**

1. **τ = 0.65 is the setting.** Mean Δ vs 2-bit by τ: default **−0.0029**, τ=0.50 −0.0002,
   **τ=0.65 +0.0024**, τ=0.80 +0.0021. τ=0.80 has a comparable mean but goes negative in 2
   of 5 cells; **τ=0.65 is the only setting positive in all five**, which is why it is the
   pick rather than the marginally-higher-mean alternative.
2. **The default was the problem, not the codec.** Every earlier "ternary loses to r=2"
   statement in this document was measured at the equal-mass split. It is retracted.
3. **Ternary beats r=1 everywhere** — never a reason to prefer r=1 on quality, only on size.
4. **The fragility axes stopped mattering once τ was tuned.** The two cells that most needed
   help — *basis*-fragile mLateOn and *capacity*-limited mxbai — are now the two biggest
   wins (+0.0055, +0.0043). At the bad default they were the extremes in both directions.

> **Scope.** τ was swept on 5 of the 7 model×corpus cells (`scifact/mxbai` and
> `scifact/mLateOn` were not re-run, so every mean above is over the same 5 cells for every
> codec, not over 7). The two unmeasured cells were mid-pack at the default τ, so they are
> unlikely to overturn the ranking — but they have not been checked, and the τ default
> should not be considered settled on 7 cells until they are.

**Pick:** **ternary at τ=0.65** as the default rung — it is 19 % smaller than r=2 and beat it
in every cell measured. Use r=4 when quality is the only constraint and bytes are free; r=1
only when 16 B/token is a hard requirement.

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
`keys_per_byte = 5`, `nibble = None`. Three consequences:

1. **Correct scoring for free.** The fused table is consumed by the **scalar** reference
   kernel, whose existing `if d == dim { break }` already stops before the padded trits in
   the last byte (dim128 → 26 B, last byte holds 3 pad trits).
2. **inv_norms** got a matching base-3 branch (#169 builds the sidecar for every
   non-binary index; without it, ternary index creation panicked).
3. **SIMD, in one hop.** The fused SIMD paths require `nibble.is_some()`, and trits
   genuinely don't align to nibbles: a byte carries 243 values and every SIMD byte-shuffle
   (`tbl`, `pshufb`) is 16-entry, while any packing that *does* factor costs ≥2 bits/dim —
   which is 2-bit exactly, storage win gone. So a scalar pre-pass is the price of sub-2-bit
   packing, not a defect. Ternary therefore builds a [`TernaryDirect`] table: each stored
   byte's five weights, padded to eight, copied straight into the kernel's weight buffer at
   `5i`. The copies overlap by three bytes but every one is a *pure* store — the next
   iteration overwrites the slack — so there is no read-modify-write. Output is natural dim
   order, so ternary queries need no permutation at all.

   *This replaced a two-hop route* (repack base-3 into a 2-bit stream, then run the nibble
   `tbl`). Measured in isolation by `examples/ternary_expand_bench`, one process, dim 128:
   two hops **20.62** ns/token on x86 and **15.32** on arm, against one hop's **12.20** and
   **10.79** — −40.8 % and −29.6 %. End-to-end on arm (the ISA where the runner CPU held
   constant across runs, so the comparison is sound) the ternary-vs-2-bit gap went from
   **+7.1…+9.0 %** to **+0.3…+1.9 %** across probe depths 1/8/32/128.

Verified by `ternary_fused_table_matches_packing` and `ternary_direct_table_matches_fused`
(unit), `ternary_simd_matches_scalar_bitwise` (SIMD accumulator bit-equal to the scalar
base-3 reference across dims 5…256 — multiples of neither 8 nor 5, and walked ascending
*then descending* with alternating document lengths, so a short row lands on a longer one's
leftovers in the reused weight buffer; run natively on aarch64 with `dotprod`), and
`ternary_asymmetric_scoring_agrees_with_float` (integration: asym top-1 == float top-1,
top-10 overlap ≥ 9/10). Full suite green on both targets, clippy + fmt clean. Pushed:
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

> ⛔ **Superseded — kept only as a record of how the measurement went wrong.** These runs
> predate both the one-hop expansion and the CI e2e ladder. A later run of this same
> harness on the same box reported ternary asym at **0.80×** — i.e. *slower* than float,
> which the contention-free CI ladder puts at 2.4–3.3×. Use
> [CI rescore ratios](#ci-rescore-ratios-contention-free) and the CI e2e ladder instead.
>
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

> **Measurement note — read before comparing anything here.** Two rules, both learned the
> hard way in this study:
>
> 1. **Only compare numbers produced inside one process.** The same benchmark on
>    *unchanged* code moves ±30 % between CI runners. An earlier version of this harness
>    ran one codec per invocation, and its cross-codec deltas were noise — on one run it
>    reported ternary *faster* than 2-bit, which cannot happen, since ternary does
>    everything 2-bit does and expands from a wider alphabet. `asym_rescore_check … all`
>    now runs every codec interleaved in one process.
> 2. **Check the runner CPU before comparing across runs.** GitHub's `ubuntu-latest` served
>    an Intel Xeon 8573C, an AMD EPYC 9V74 (`avx512-vnni`) and an AMD EPYC 7763 (`avx2`) on
>    three consecutive runs of this workflow. `ubuntu-24.04-arm` has been Neoverse-N2
>    throughout, so arm is the axis on which run-over-run comparison is currently sound.

Stage-2 rescore only, **ns/token**, median of 5 interleaved reps, 4096 docs × 230 tokens,
dim 128, all four codecs in one process
(run [31978074468](https://github.com/joe32140/next-plaid/actions/runs/31978074468)):

| codec | x86 float | x86 asym | **x86 ratio** | arm float | arm asym | **arm ratio** |
|---|--:|--:|--:|--:|--:|--:|
| 4-bit | 508.2 | 96.4 | **5.27×** | 233.1 | 56.6 | **4.12×** |
| 2-bit | 444.6 | 93.8 | **4.74×** | 223.1 | 52.4 | **4.26×** |
| **ternary** | 289.2 | **108.1** | **2.68×** | 182.4 | **56.3** | **3.24×** |
| 1-bit | 419.4 | 93.8 | **4.47×** | 201.8 | 55.9 | **3.61×** |

*(x86 = EPYC 7763 / `avx2`; arm = Neoverse-N2 / `neon-sdot`.)*

**Asym is a 2.7–5.3× win for every codec on both ISAs.** Two things about the ternary row:

- Its low *ratio* is not weakness — it is a fast float baseline (289 / 182 ns/tok, the
  fastest of the four) divided into a normal endpoint. Ternary's byte→5-values decode is
  cheaper than scalar bit-unpacking, so **without asym ternary is both the smallest useful
  rung and the fastest**.
- Its asym cost over 2-bit is the expansion, and nothing else: after expansion both occupy
  the same lane count and run an identical dot. That cost is **+3.86 ns/token on arm
  (+7.4 %) and +14.29 on x86 (+15.2 %)**.

End-to-end (`search_latency`, synthetic 4000×180, probe 8) the same gap reads **+1.7 % arm,
+7.2 % x86** — smaller, because stage 1 is common to both and unchanged by #169.

**Why x86 pays more.** The one-hop expansion writes 8 bytes to place 5 weights, so it issues
26 stores/token at dim 128 against the nibble path's 8 wide ones. On a store-port-limited
core that shows up directly; on a wider one it hides under the dot. This is a hypothesis
consistent with the arm/x86 split and with the local Apple-silicon reading (+2.2 %), **not a
verified cause** — confirming it means either a PMU counter or a variant that stores 15
weights at a time, and neither has been done.

## How far can we push latency — bottom line

Separate the two stages:

- **Stage-2 rescore (what the codec + #169 own):** asym is a **2.7–5.3× kernel speedup for
  every codec**, clean on both x86 and arm (CI), ternary included. Stage-2 is therefore
  pushed near its floor for *all* codecs; what remains is stage-1.
- **Stage-1 (candidate generation):** owns end-to-end latency at BEIR scale — unchanged by
  #169, targeted by #170. This is why the end-to-end asym speedup is well below the kernel
  speedup. Probe depth 1→128 moves e2e asym latency ~25–29 %, so it is the lever that
  decides how much of a search the codec choice can touch at all.
- **Storage and quality together:** at τ=0.65 ternary is a **~19 % index shrink over 2-bit
  that also scores better than 2-bit in every cell measured** — the trade that used to exist
  was an artifact of the equal-mass default. Its smaller footprint also makes it the fastest
  codec on the *float* rescore path.

**Deployment recipe:** **ternary at `ternary_tau = 0.65`** for footprint-bound deployments —
smaller than 2-bit and better than it on quality. Enable `residual_asym` unconditionally;
every codec reaches a fused SIMD kernel on both ISAs and gains 2.7–5.3× at rescore with NDCG
parity. Budget ternary ~+2 % (arm) to ~+7 % (x86) e2e over 2-bit for that 19 %; if e2e
latency is the binding constraint on x86 specifically, 2-bit remains the safer pick.

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
