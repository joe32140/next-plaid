# Ternary residual codec

A base-3 residual codec for PLAID indexes: each dimension is stored as one of
`{−m, 0, +m}`, five trits packed per byte (3⁵ = 243 ≤ 256), for ~1.585 bits/dim
and `ceil(dim/5)` bytes per token. It reconstructs `centroid + weight` and scores
with float MaxSim exactly like the scalar codec, so it drops into the existing
search path — it supersedes `nbits` rather than extending it.

It exists to fill a real gap in the storage ladder. Between 1-bit (16 B/token at
dim 128) and 2-bit (32 B) there was nothing, and that is a 2× step in the
dominant on-disk cost of a late-interaction index.

## Storage

| | 1-bit | **ternary** | 2-bit | 4-bit |
|---|--:|--:|--:|--:|
| bits / dim | 1.000 | **1.585** | 2.000 | 4.000 |
| B/token @ dim 128 | 16 | **26** | 32 | 64 |
| B/token @ dim 96 | 12 | **20** | 24 | 48 |
| B/token @ dim 48 | 6 | **10** | 12 | 24 |
| vs 2-bit | −50 % | **−19 %** | — | +100 % |

## The dead-zone width is the setting, not a detail

Ternary's one real degree of freedom is how wide the zero bucket is. The obvious
construction — reuse the scalar path's equal-mass quantiles, cutting at the 1/3
and 2/3 residual quantiles — zeroes exactly a third of the dimensions no matter
how the residuals are actually shaped, and it is the codec's *worst* setting.

`IndexConfig::ternary_tau` sets it explicitly: a dimension stores `0` when
`|r| < τ·σ`, and `±E[|r| : live]` otherwise. It defaults to `Some(0.65)`;
`None` restores the equal-mass split.

On roughly Gaussian residuals τ = 0.65 zeroes ~48 % of dimensions rather than
33 %, and spends the levels it frees on a larger magnitude for the survivors.
That trade is worth about 0.006 NDCG@10:

| | mean Δ NDCG@10 vs 2-bit |
|---|--:|
| equal-mass (`ternary_tau: None`) | −0.0029 |
| τ = 0.50 | −0.0002 |
| **τ = 0.65** (default) | **+0.0028** |
| τ = 0.80 | +0.0027 |

So the choice is not "19 % smaller, slightly worse" — at τ = 0.65 ternary is
smaller than 2-bit *and* better than it.

## Evidence

Codec-isolated NDCG@10: fixed k-means seed, exhaustive float MaxSim over
reconstructions, so the residual codec is the only variable per row (no stage-1
confound). Every profile within a cell shares the queries, the residuals and the
centroids, which makes each Δ a *paired* quantity.

Pooled over the 8 cells with enough judged queries to resolve the effect —
**4,240 queries, 6 corpora, 4 encoder families, three dims** — τ = 0.65 beats
2-bit by **+0.0028 and is positive in 8 of 8**:

| corpus / model | dim | docs | q | float | 2-bit | equal-mass | **τ=0.65** | Δ vs 2-bit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| POJ-104 / LateOn-Code-edge | 48 | 3,965 | 1000 | .3120 | .3102 | .3113 | **.3138** | **+0.0036** |
| POJ-104 / LFM2.5-ColBERT | 128 | 3,965 | 1000 | .3933 | .3848 | .3838 | **.3875** | **+0.0027** |
| FiQA / LFM2.5-ColBERT | 128 | 57,638 | 648 | .4862 | .4846 | .4794 | **.4872** | **+0.0026** |
| nfcorpus / mLateOn | 128 | 3,633 | 323 | .3759 | .3668 | .3603 | **.3723** | **+0.0055** |
| nfcorpus / answerai | 96 | 3,633 | 323 | .3725 | .3662 | .3617 | **.3677** | **+0.0015** |
| nfcorpus / mxbai | 128 | 3,633 | 323 | .3092 | .3050 | .3069 | **.3093** | **+0.0043** |
| nfcorpus / ColBERTv2 | 128 | 3,633 | 323 | .3324 | .3317 | .3317 | **.3323** | **+0.0006** |
| scifact / ColBERTv2 | 128 | 5,183 | 300 | .6464 | .6464 | .6411 | **.6466** | **+0.0002** |

Ternary at τ = 0.65 reaches **mean NDCG retention equal to 4-bit's** at 26 B/token
against 4-bit's 64, and **clears 1-bit in every cell measured, at every τ**.

τ was originally tuned on nfcorpus + scifact — both biomedical, all BERT-family
encoders — so the first three rows are the out-of-distribution check: finance and
code, on a dim-48 edge model and a non-BERT hybrid. The margin reproduced at the
same size it was fitted at.

The FiQA row also settles corpus scale, the axis every other cell shares: at
57,638 documents there are an order of magnitude more competitors than anywhere
else here and ranking margins are correspondingly tighter — the regime where a
quantizer's error should start flipping results. It returns +0.0026, in line with
the pooled mean rather than below it.

**τ = 0.80 is not a demonstrated improvement.** Head-to-head against 0.65 across
these 8 cells it is +0.0005 (+0.0011 weighted by judged queries), positive in 5 of
8 — while against 2-bit it wins 6 of 8 where τ = 0.65 wins 8 of 8. Worth stating
honestly: the three highest-power cells all favour 0.80 by +0.0016 to +0.0023, so
the direction is not random and more evidence could move it. The default is the
setting that never loses, not the higher mean.

## Query-time cost

**This section is being re-measured against 1.7.0 and the numbers below are not
yet updated.** They were taken before #169 landed, when every codec shared the
float decode path. Asymmetric rescoring is now the default and it dispatches
*per codec*, so the comparison is no longer like-for-like — see
"How ternary interacts with asymmetric scoring" below.

## How ternary interacts with asymmetric scoring

Since 1.7.0, asymmetric rescoring — int8 query against a fused byte→weights
table, no float decompression — is the default residual path. Ternary engages
it: `quantize_lut` builds the fused table straight from the base-3 trit table,
because a trit value *is* its weight index.

What ternary cannot have is the in-register expansion. NEON `tbl` and AVX2
`pshufb` index a **16-entry** table with a nibble, and that is why base 2 is so
comfortable: a 2-bit or 4-bit byte factors into nibbles, each nibble indexes the
shuffle, and the decode never leaves the vector unit. A base-3 byte carries 243
values and does not factor — there is no way to split 243 into two 16-entry
lookups. And any packing that *does* factor costs at least 2 bits per dimension,
which is the 2-bit codec with the storage win gone.

**So a scalar expansion pass is the price of sub-2-bit packing, not a defect in
the implementation.** Ternary takes `maxsim_residual_lut_scalar`, whose
`d == dim` break already masks the padded tail byte. It keeps the substance of
asymmetric scoring and gives up the vector unit.

The practical consequence is that any end-to-end latency comparison against
2-bit is now a **codec + kernel** result, not a codec result: SIMD-asym 2-bit
against scalar-asym ternary. Set `NEXT_PLAID_REPORT_KERNEL=1` to see which
kernel each index dispatched to before reading a ladder.

A one-hop 256×5 byte→weights expansion that rides the existing SIMD kernels
does exist and closes most of this gap; it is a follow-up, not part of the
codec.

## Reading a codec ladder without fooling yourself

Four things cost real time to learn while measuring this, and they generalize to
any quantizer comparison:

1. **A cell can be unreadable, and the ladder detects it for free.** 1-bit is
   strictly lossier than float, so *a cell reporting 1-bit as better than float is
   reporting noise* — no extra computation, the row is already there. The gate does
   bias its survivors upward (it selects cells whose noise happened to align with
   the true ordering), so use it to discard cells, never to rescue them.
2. **It catches two different pathologies, and only one is fixable with money.**
   *Too few judged queries* — four of twelve NanoBEIR cells at 50 queries — is
   cheap to fix, since queries cost one forward pass each and never touch the
   corpus encode. *Too low a float ceiling* is not fixable at any budget: a code
   encoder run over financial prose failed this gate with **648** judged queries
   and all seven lossy profiles above float, because at a float NDCG@10 of 0.2492
   the ranking is too weakly determined for quantization noise to be asymmetric.
   Screen a candidate cell on its float NDCG before paying to encode it — judged
   queries are necessary, not sufficient.
3. **Buying queries can flip a sign, not just shrink an error bar.** One cell put
   τ = 0.65 *behind* 2-bit at −0.0008 with 130 queries; the same corpus and model
   at 1,000 queries gives +0.0027.
4. **Reconstruction fidelity picks the wrong τ.** `reconCos` prefers 0.65 over
   0.80 in 15 of 17 cells while NDCG's mean prefers 0.80. It averages over
   millions of tokens instead of hundreds of queries and costs nothing extra —
   exactly the cheap proxy one would reach for — and it disagrees with the ranking
   metric on the knob being tuned. It is also the more *robust* of the two: in the
   cell that failed the gate above, `reconCos` stayed perfectly ordered while NDCG
   was pure noise. Robust and wrong. Tune τ on NDCG.

## Usage

```rust
use next_plaid::IndexConfig;

let config = IndexConfig {
    ternary: true,          // supersedes `nbits`; mutually exclusive with `binary`
    ..Default::default()    // ternary_tau defaults to Some(0.65)
};
```

`ternary_tau: None` selects the equal-mass split, which is retained for
reproducing the pre-default behaviour. It is not recommended.

## Repro

```bash
cargo run --release -p next-plaid --example ndcg_eval -- <bundle-dir>
```

prints the full ladder (float / 4-bit / 2-bit / equal-mass / τ ∈ {.50, .65, .80}
/ 1-bit) with bytes per token, NDCG@10 and mean per-token reconstruction cosine.
A bundle is `corpus.npy`, `corpus_lens.npy`, `queries.npy`, `query_lens.npy`,
`corpus_ids.json`, `query_ids.json`, `qrels.json`.

Correctness is covered by `next-plaid/tests/ternary_integration.rs` — on-disk size
lands between the 1-bit and 2-bit rungs, search retrieves through the
reconstruct-then-MaxSim path, updates and deletes stay ternary-aware, the dead
zone is wider than equal-mass and deterministic across rebuilds, and a config
written before `ternary_tau` existed deserializes to the shipped default.
