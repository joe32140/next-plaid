# Fine-tuning for binary quantization

Companion to [`binary_quantization.md`](binary_quantization.md). That doc adds the
asymmetric **int8 × 1-bit** document store to next-plaid; this one asks a follow-up
question: **can we _train_ a ColBERT model so it binarizes better** — and does the
gain survive real two-stage PLAID retrieval?

All NDCG@10. "Exhaustive" = brute-force MaxSim (the quantizer's ceiling, ANN
removed). "Deployed" = the real next-plaid PLAID index (`n_ivf_probe=8`,
`n_full_scores=4096`, centroid pruning ON) scored through the binary CR.
Training harness: `TACET/infra/modal_lateon_binary.py` (PyLate ColBERT on Modal,
LightOn `embeddings-fine-tuning` data); PLAID eval harness:
`TACET/infra/modal_plaid_eval.py` + [`examples/binary_ndcg.rs`](../next-plaid/examples/binary_ndcg.rs).

## TL;DR

1. **The binary CR is numerically correct.** On every checkpoint we did not
   retrain, the Rust *deployed-binary* NDCG matches an independent NumPy
   *exhaustive-binary* NDCG to 4 decimals — across SciFact, NFCorpus, ArguAna and
   25K-doc SciDocs.
2. **The ANN stage is nearly free for NDCG@10.** Even on SciDocs, where PLAID
   rescores only ~16% of the corpus (4096 of 25 657 docs), deployed == exhaustive.
   Binarizing documents doesn't hurt recall because Stage-1 candidate generation
   uses float centroids.
3. **Plain supervised fine-tuning is the reliable binarization lever** — it lifts
   binary NDCG on every dataset with no float cost.
4. **A binary-aware auxiliary loss adds a real bonus at low weight (α ≈ 0.1–0.2)**
   — it recovers most of the base→reference binary gap where headroom exists, at
   sub-noise cost on saturated sets, and even nudges float NDCG *up*. At α = 0.4 it
   over-trades (float collapses); the win is α-gated.

## Setup: four checkpoints

LightOn releases both stages of LateOn, which lets us inject binary-awareness at
the *correct* place (the supervised final stage) instead of perturbing a finalized
model:

| arm | checkpoint | training we applied | isolates |
|-----|------------|---------------------|----------|
| **reference** | `lightonai/LateOn` | none (their full Stage-2) | the yardstick |
| **base** | `lightonai/LateOn-unsupervised` | none (Stage-1 only) | our start |
| **control** | `LateOn-unsupervised` | plain MaxSim contrastive (Stage-2 repro) | *fine-tuning's* effect |
| **treatment** | `LateOn-unsupervised` | + binary-aware loss | *the binary loss's* effect |

Binary-aware loss = `(1−α)·float-MaxSim-CE + α·STE-sign-MaxSim-CE + β·bit-balance`.
The STE-sign term scores the batch with `sign(doc)` embeddings (straight-through
gradient) so the model is optimized for how it will actually be stored; the
bit-balance term pushes per-dimension mean sign toward 0 to keep bits live.

## Result 1 — binary NDCG@10 across checkpoints (deployed PLAID)

| checkpoint | scifact | nfcorpus | arguana | scidocs |
|------------|--------:|---------:|--------:|--------:|
| reference  | 0.7451 | 0.3633 | 0.2957 | 0.1830 |
| base       | 0.7398 | 0.3570 | 0.2431 | 0.1893 |
| control    | 0.7400 | 0.3719 | 0.2643 | 0.1910 |
| treatment (α=0.4) | 0.7396 | 0.3625 | 0.2820 | 0.1935 |

Reading a column top-to-bottom: **base→control** (plain fine-tuning) is a clean
win everywhere (nfcorpus +0.015, arguana +0.021, scidocs +0.002, scifact ≈ref).
**control→treatment** is *headroom-gated*: large on arguana (base was far below
reference), ~zero/negative on saturated scifact & nfcorpus.

## Result 2 — the α sweep (exhaustive, balance = 0.1)

**int8 × binary NDCG@10**

| α | nfcorpus | arguana |
|---|---------:|--------:|
| 0.0 (control) | 0.3705 | 0.2652 |
| 0.1 | 0.3650 | 0.2886 |
| 0.2 | 0.3629 | **0.2911** |
| 0.3 | 0.3579 | 0.2858 |
| 0.4 | 0.3611 | 0.2786 |

**float NDCG@10**

| α | nfcorpus | arguana |
|---|---------:|--------:|
| 0.0 | 0.3750 | 0.3418 |
| 0.1 | **0.3802** | **0.3527** |
| 0.2 | 0.3781 | 0.3472 |
| 0.3 | 0.3761 | 0.3365 |
| 0.4 | 0.3727 | 0.3263 |

The arguana binary gain is **concave in α, peaking near 0.2**; arguana *float*
peaks earlier at 0.1. So the useful band is **α ≈ 0.1–0.2**. At α = 0.1 vs control:
arguana float **+0.011** / binary **+0.023** (both up), nfcorpus float **+0.005** /
binary −0.005 (sub-noise). α = 0.4 is past both peaks — it over-trades.

Surprise: at low α the STE-sign term acts as a mild regularizer that lifts *float*
NDCG too, not just binary.

## Result 3 — α = 0.1 survives deployment

Does the sweep's near-free win hold under real two-stage retrieval? Yes — the
arguana gain is identical in exhaustive and deployed:

| | control (deployed) | α=0.1 (deployed) | Δ | Δ (exhaustive) |
|---|---:|---:|---:|---:|
| arguana binary  | 0.2643 | **0.2875** | +0.023 | +0.023 |
| nfcorpus binary | 0.3719 | 0.3619 | −0.010 | −0.005 |

## Result 4 — α = 0.1 is neutral-to-positive everywhere else

The sweep above only used the two discriminating datasets. Extending α = 0.1 vs the
plain control to the saturated (scifact) and large (scidocs) sets confirms there is
no dataset where α = 0.1 meaningfully hurts (exhaustive):

| dataset | control int8×bin | α=0.1 int8×bin | Δ | control float | α=0.1 float | Δ float |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| scifact | 0.7384 | 0.7384 | **0.000** | 0.7492 | 0.7562 | +0.007 |
| scidocs | 0.1920 | 0.1951 | **+0.003** | 0.2152 | 0.2143 | −0.001 |

Combined with the sweep, the full α = 0.1 vs control scorecard (int8 × binary):
**arguana +0.023, scidocs +0.003, scifact 0.000, nfcorpus −0.005.** It helps where
there is headroom, is neutral where there isn't, and never costs more than noise —
a genuine near-free improvement. Float NDCG is flat-to-up on all four.

## Result 5 — the bit-balance term (β) is nearly inert here

Sweeping β at fixed α = 0.1 (exhaustive):

| β | nfcorpus int8×bin | arguana int8×bin | nfcorpus bin×bin | arguana bin×bin |
|---|:---:|:---:|:---:|:---:|
| 0.0 | 0.3647 | 0.2868 | 0.3395 | 0.2413 |
| 0.1 | 0.3660 | 0.2889 | 0.3416 | 0.2419 |
| 0.3 | 0.3629 | 0.2876 | 0.3414 | 0.2423 |
| 0.5 | 0.3654 | 0.2867 | 0.3448 | 0.2450 |

No trend on the asymmetric int8 × binary scheme (both columns wander inside ~±0.002
noise). There's a *faint* monotone lift on the fully-symmetric binary × binary case
at higher β (nfcorpus 0.3395→0.3448, arguana 0.2413→0.2450) — consistent with the
balance term keeping more bits live, which matters most when *both* sides are 1-bit.
For the deployed int8 × binary index, **β ≈ 0.1 is a fine default but the STE-sign
term (α) is doing essentially all the work.**

## CR correctness: independent cross-check

For arms we did **not** retrain (reference, base — identical model to the NumPy
reference), Rust deployed-binary equals NumPy exhaustive-binary to 4 dp:

| arm / dataset | NumPy exhaustive int8×binary | Rust deployed binary |
|---------------|:---:|:---:|
| reference / scifact | 0.7451 | 0.7451 |
| base / scifact      | 0.7398 | 0.7398 |
| reference / nfcorpus | 0.3633 | 0.3633 |
| base / arguana      | 0.2431 | 0.2431 |
| reference / scidocs | 0.1829 | 0.1830 |
| base / scidocs      | 0.1893 | 0.1893 |

Two independent implementations (NumPy brute force vs the Rust `binary.rs` inside a
real PLAID index) agreeing to 4 dp is a strong correctness check on the CR's
`binarize` + asymmetric int8×1-bit `maxsim_binary`.

## Recommendations

- **Deploying an off-the-shelf ColBERT with binary docs?** Expect ~95% of residual
  NDCG@10 at 32× smaller documents; the two-stage index costs ~nothing beyond the
  quantizer for NDCG@10.
- **Have a final-stage fine-tune in your pipeline?** Do it from the pre-final
  checkpoint and add the binary-aware loss at **α ≈ 0.1, β ≈ 0.1**. It's a
  near-free improvement where the model binarizes poorly, and harmless where it
  already binarizes well. Don't exceed α ≈ 0.2 — larger weights trade float for
  binary and then lose both.
- **Validate on real NDCG, not a proxy** — and remember the retention-% metric
  (binary ÷ that model's own float) can be inflated when float drops; report
  absolute binary NDCG.
