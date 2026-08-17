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
| FiQA / LFM2.5-ColBERT | 128 | 20,000 | 648 | .5847 | .5830 | .5793 | **.5867** | **+0.0037** |
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

**τ = 0.80 is a dead tie**, not an improvement: head-to-head against 0.65 across
these 8 cells it is −0.0001, positive in 4 of 8. The default is the incumbent,
which also reconstructs marginally better.

## Reading a codec ladder without fooling yourself

Three things cost real time to learn while measuring this, and they generalize to
any quantizer comparison:

1. **A cell with ~50 judged queries cannot resolve a 0.002 effect, and the ladder
   detects that for free.** 1-bit is strictly lossier than float, so *a cell
   reporting 1-bit as better than float is reporting noise* — no extra
   computation, the row is already there. Four of twelve NanoBEIR cells failed
   that check. The gate does bias its survivors upward (it selects cells whose
   noise happened to align with the true ordering), so use it to discard cells,
   never to rescue them.
2. **Buying queries can flip a sign, not just shrink an error bar.** One cell put
   τ = 0.65 *behind* 2-bit at −0.0008 with 130 queries; the same corpus and model
   at 1,000 queries gives +0.0027.
3. **Reconstruction fidelity picks the wrong τ.** `reconCos` prefers 0.65 over
   0.80 in 15 of 17 cells while NDCG's mean prefers 0.80. It averages over
   millions of tokens instead of hundreds of queries and costs nothing extra —
   exactly the cheap proxy one would reach for — and it disagrees with the
   ranking metric on the knob being tuned. Tune τ on NDCG.

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
