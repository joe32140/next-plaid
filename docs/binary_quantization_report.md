# What Governs Binary Quantization in ColBERT — and What Doesn't

*A first-principles report on asymmetric int8×1-bit document quantization for
late-interaction retrievers.*

Companion code: quantizer + CR in [`next-plaid`](../next-plaid/src) (`binary.rs`,
`examples/binary_ndcg.rs`); training / probing harness in
`TACET/infra/modal_lateon_binary.py`; deployed-PLAID eval in
`TACET/infra/modal_plaid_eval.py`. All NDCG@10.

> **Status.** Well-controlled. The claims below are backed by 9 mixed-provenance
> checkpoints plus eight targeted ablations (NE1–NE8): every competing explanation we
> could name — scale, projection dim, backbone size, spectral isotropy, token
> magnitude, query int8, teacher distribution, and post-hoc threshold/rotation fixes —
> was tested and ruled out, leaving the **training objective** as the one clean driver
> of saturation. The load-bearing result (§6, dead bits are a predictor you cannot
> fix) is robust and generalizes across eval sets (NE6).

---

## Abstract

Asymmetric binary quantization stores each ColBERT document token as the **sign**
of its embedding (1 bit/dim) while keeping the query in int8 — a 32× shrink of the
document store that is *exact* to compute. Yet across off-the-shelf ColBERT models
its quality ranges from near-lossless to catastrophic: on NFCorpus, **24% to 97%**
of a model's own int8×int8 NDCG@10 (as low as **18%** on ArguAna). We ask what
governs this and whether training can improve it.

**Binarizability is governed by per-axis sign saturation** — the fraction of
dimensions whose sign is near-constant across the corpus ("dead bits"). Across nine
checkpoints of every provenance, low saturation robustly predicts near-lossless
binarization **regardless of the training objective** (two knowledge-distilled models
have 0% dead bits and retain 84–97%). dead% is the best simple predictor and nothing
cheaper sits underneath it: spectral isotropy, token magnitude, and the query-int8
step are all measured and none explains the variation (§5, NE1/NE4). Saturation is a
**symptom, not a lever**: four interventions that remove dead bits (per-dim
centering, a balance penalty, an ITQ rotation, and a spectral-isotropy variable)
fail to improve quality, and centering/ITQ *destroy* it — no post-hoc transform
recovers binarizability (§6, NE5). You reduce saturation only by (re)training, and the
one clean driver is the **final-stage objective**: contrastive *actively de-saturates*
— to ~0% dead at every projection dim (NE7) and every backbone size (NE8), replicated
over seeds (NE2) — while distillation is *inert*, leaving whatever default the
pretrained backbone had, **regardless of teacher distribution** (a bimodal reranker
teacher saturates no more than a narrow one, NE3). The backbone supplies a default
anisotropy that varies by architecture but neither dim nor size predicts it (NE7/NE8),
and it does not matter much because contrastive overrides it. So edge models are
saturated because they met a **KD** final stage that left their backbone's default
alone (a fresh *contrastive* head on the same Ettin-17M backbone reaches 1% dead), not
because they are small; ColBERTv2/mxbai-large are fine because their backbone default
was already low; LateOn is fine because contrastive de-saturated it.
A binary-*aware* loss adds only a small headroom-gated bonus (§7.3). **Practical
upshot:** check dead% before deploying binary docs; prefer a well-conditioned
checkpoint (any recipe) or briefly contrastively fine-tune; no post-hoc trick or
binary-specific loss rescues a saturated one.

---

## 1. The question

Two-stage late-interaction retrieval (ColBERT [Khattab & Zaharia 2020]; PLAID
[Santhanam et al. 2022]) stores one vector **per token**, so the document store
dominates memory. Asymmetric quantization [mixedbread 2024] keeps the query in int8
and stores each document token as `sign(x) ∈ {−1,+1}^d`. MaxSim is then **exact and
cheap** via `q · b = 2·Σ_{i: bᵢ=+1} qᵢ − Σᵢ qᵢ` — no float multiplies on the doc
side, 32× smaller docs. The catch is quality (24%→97% on NFCorpus). **(Q1)** what
governs it; **(Q2)** can we train for it.

---

## 2. Mechanics you need

- **Asymmetric int8×1-bit.** Query → int8 (per-token max-abs scale); doc token →
  `sign(x)`. The asymmetry (int8 query, not binary query) separates int8×binary from
  the coarser binary×binary floor — and NE4 shows the int8-query step is *lossless*,
  so all binary loss is doc-signing (§5.4).
- **MaxSim** ranks by *relative* score across documents.
- **Sign-at-zero code (orthant pattern).** `sign(x) ∈ {−1,+1}^d` — which coordinates
  are positive vs negative. Binary MaxSim ranks entirely by this code; binarizability
  = the degree to which it carries the float ranking signal.
- **Dead bit.** A dimension with `|mean_t sign(x_{t,d})| > 0.9`; it stores the same
  bit for every doc and cancels from the ranking. "dead%" = fraction dead.
- **Two-stage PLAID.** Stage-1 float centroids drive candidate recall; binarizing the
  doc store barely touches NDCG@10 (§4).
- **Exhaustive vs deployed.** Exhaustive = GPU brute-force MaxSim; deployed = the real
  next-plaid PLAID index. §4 shows they agree.
- **Metrics.** NDCG@10, exponential gain. **Absolute binary NDCG** is primary;
  "retention %" = int8×binary ÷ int8×int8 throughout (§7.4 shows retention-% falling
  while binary NDCG rises).

---

## 3. Thesis

> **Binarizability is governed by per-axis sign saturation of the sign-at-zero
> code.** Low dead% ⇒ the orthant pattern carries the ranking signal ⇒ near-lossless
> binarization, **across objectives and backbones**. Dead bits are a **symptom, not a
> lever**: no post-hoc transform (threshold, balance, rotation) recovers
> binarizability — threshold 0 is where training set the boundary (§6, firm). You
> reduce saturation only by (re)training: it **originates in the backbone** and the
> **final-stage objective decides whether it is fixed** — contrastive de-saturates,
> distillation is inert.

---

## 4. The measurement scaffold is sound

**4.1 The binary CR is numerically correct** — Rust *deployed-binary* == NumPy
*exhaustive-binary* to 4 dp on SciFact/NFCorpus/ArguAna/25K-doc SciDocs (e.g.
reference/SciFact 0.7451=0.7451; base/SciDocs 0.1893=0.1893). **4.2 The ANN stage is
nearly free** — deployed == exhaustive even on SciDocs (~16% rescored). So quality
lives in the **embedding geometry**, not the index.

---

## 5. What governs binarizability

**5.1 Not scale, not dimensionality.** A 353M model binarizes *worse* than a 150M one
at the same dim (LFM2.5 74% vs LateOn-unsup 96%); non-monotonic in params. Scale
alone is not it.

**5.2 Per-axis sign saturation (dead%) predicts — across nine mixed-provenance
models.** To break the recipe/quality collinearity of a small sample, we probed four
public checkpoints alongside the original five (untrained, NFCorpus; retention =
int8×binary ÷ int8×int8):

| model | recipe | dim | dead% | iso_u | retention |
|---|---|---:|---:|---:|---:|
| ColBERTv2 | **KD / distill** | 128 | **0%** | 0.118 | **97.4%** |
| LateOn-unsup | contrastive | 128 | 12% | 0.009 | 96% |
| LateOn | contrastive | 128 | 12% | 0.009 | ~95% |
| answerai-colbert-small | KD / distill | 96 | 47% | 0.015 | 84.6% |
| mxbai-colbert-large | **KD / distill** | 128 | **0%** | 0.106 | 83.8% |
| LFM2.5 | KD | 128 | 57% | 0.010 | 74% |
| GTE-ModernColBERT | contrastive/distill | 128 | 64% | 0.010 | 72.1% |
| mxbai-edge-32m | KD | 64 | 55% | 0.019 | 61% |
| mxbai-edge-17m | KD | 48 | 65% | 0.024 | 24% |

*(jina-colbert-v2 failed to load — missing `einops`.)*

- **dead% is a real predictor, not a recipe proxy.** Low dead% (≤~15%) → 84–97% for
  **both KD and contrastive** models; the two lowest-dead% models are *distilled*.
- **One-sided and noisy.** ≤~15% ⇒ good reliably; above that it is loose (answerai
  47%→85% vs 17M 65%→24%), and 0% dead is not sufficient (mxbai-large 0%→84% vs
  ColBERTv2 0%→97%). Residual variation at fixed dead% is unexplained by any
  unsupervised scalar (§5.4) — it is the *supervised* discrimination quality of the
  sign pattern.

**5.3 Not spectral isotropy.** An earlier draft's "all ColBERT is ~rank-2
anisotropic" was **false** — ColBERTv2 has effective rank ~56/128. Isotropy varies
and the two most-isotropic models are 0%-dead, but it is not monotonic (LateOn has
the *lowest* iso_u yet 95% retention). Not a law.

**5.4 Not token magnitude, and not query-int8 (NE4).** Two clean negatives from
decomposing all nine:
- **The query-int8 step is lossless.** `float-query × binary-doc` retention (docret)
  ≈ `int8-query × binary-doc` retention for every model (17M 24.0 vs 24.2; mxbai-large
  83.3 vs 83.8; LateOn 94.6 vs 94.7). **All binary loss is document signing.**
- **Magnitude uniformity does not predict.** `sign_cos` = cos(token, its sign) is
  ~0.78–0.82 across most models regardless of retention (17M 0.808→24%, ColBERTv2
  0.800→97%, mxbai-large 0.803→84%), and LateOn is a low outlier (0.434) yet retains
  95%. Reconstruction fidelity of the sign map is *not* what matters — only whether
  the sign pattern discriminates.

**5.5 Not projection dim — capacity-as-dim is refuted (NE7).** Fresh ColBERT heads on
a fixed backbone (ModernBERT-base) + contrastive recipe at dims 32/64/128 are **all
0% dead**. Retention rises with dim (NFCorpus 51.6→63.1→72.8%) but purely because
higher-dim models are stronger retrievers with more redundancy under signing — *not*
via saturation. Low dim does not cause dead bits.

**5.6 The clean driver is the objective; the backbone supplies a default it doesn't
control.** We tried to pin the origin on backbone *size* (NE8: fresh ColBERT heads at
fixed dim on the Ettin family, 17m→400m). It **failed as a size law**: under KD,
dead% is non-monotonic in size (46/39/53/29/36%), and under **contrastive it collapses
to ~0% at every size** (17m 1%, 32m/68m/150m/400m 0%). So — with dim (5.5), magnitude
(5.4), and teacher distribution (§7.2) already ruled out — the one clean, reproducible,
controllable variable is the **training objective**: contrastive de-saturates whatever
it starts with; KD is inert and leaves the backbone's *default* saturation. That
default varies by backbone (Ettin-tiny anisotropic → ~50% under KD; BERT-large
isotropic → 0%) but *not* by a tidy size trend, and it is under-characterized (an
architecture/pretraining property). Practically it matters little: whatever the
backbone's default, contrastive removes it. So the released edge models are saturated
because of **KD** (a fresh *contrastive* head on the same Ettin-17M backbone reaches
1% dead), not because they are small.

---

## 6. A dead bit is a lever you cannot pull — the firm result

If dead bits were the *cause*, removing them would restore quality. Four
interventions remove them; none helps, and two are catastrophic. Each is a
within-model intervention.

**6.1 Per-dim threshold centering (free, identity-preserving) — collapses NDCG.**
Store `sign(x − μ)` with a per-dim corpus mean `μ`. This **revives 100% of dead bits
(→0%)** yet **not one of eight cells improves** (median-centering identical, omitted):

| model | dead@0 | NF thr@0 → centered | ArguAna thr@0 → centered |
|---|---:|---:|---:|
| mxbai-17m | 65% | 0.0876 → **0.0131** | 0.0593 → **0.0015** |
| mxbai-32m | 55% | 0.2198 → **0.0188** | 0.1719 → **0.0007** |
| LFM2.5 | 57% | 0.2810 → **0.0551** | 0.1711 → **0.0180** |
| LateOn | 12% | 0.3633 → **0.2229** | 0.2955 → **0.0089** |

The skeptic's confound — "μ is in-sample" — is *inverted* (μ is fit on the scored
corpus, the transductive best case), so deployed centering is only worse. The LateOn
row (12% dead, still collapses) isolates "moving the boundary scrambles *live* bits."
It is not an ArguAna artifact: on **FiQA** (multi-relevant, non-symmetric; NE6) all
four models collapse — mxbai-17m 0.0355 → 0.0006, LFM2.5 0.2604 → 0.0413, the
13%-dead LateOn 0.4615 → 0.0671, and even the 0%-dead mxbai-large drops (0.2469 →
0.1790).

**6.2 A during-training balance penalty (β) — inert.** Drives dead bits 7→5→3 yet
leaves binary NDCG flat.

**6.3 An ITQ rotation (NE5) — also fails; no post-hoc transform recovers.** ITQ [Gong
& Lazebnik 2011] centers, then fits the orthonormal rotation minimizing binary
quantization error, and rotates query+doc together (inner products preserved in
float). It is the decisive fork test: help ⇒ the collapse was "centering without
re-basing"; hurt ⇒ intrinsic to sign-after-shift. **It hurts** (int8×binary vs
sign@0):

| model | dead@0 | sign@0 | mean-center | ITQ |
|---|---:|---:|---:|---:|
| mxbai-17m | 65% | 0.0876 | 0.0131 | **0.0079** |
| LFM2.5 | 57% | 0.2810 | 0.0551 | **0.1059** |
| LateOn | 12% | 0.3633 | 0.2229 | **0.2034** |
| mxbai-large | 0% | 0.2581 | 0.2501 | **0.2706** |

ITQ is far below sign@0 on every saturated model (only marginally above plain
centering) and roughly neutral on the already-0%-dead model. So the collapse is
**intrinsic to sign-after-shift**, and the last free lever is closed:
**no post-hoc transform — threshold shift or rotation — recovers binarizability.**
(Standardization ÷std is a no-op for a sign code, so it is not run.)

**6.4 Mechanism.** A dead bit is a constant plus noise; at threshold 0 the constant
cancels from the ranking. Any move off 0 (centering, ITQ's centering step) replaces
it with the sign of noise and scrambles the trained boundary — which is why even the
12%-dead LateOn collapses and why the optimal rotation cannot save it. **The fix is
(re)training, not post-processing.**

---

## 7. What actually moves binarizability

**7.1 Contrastive recovers a dead checkpoint; distillation cannot (NE9).** The
operational question: given a dead binary checkpoint and a distillation pipeline, can
you keep distilling, or must you switch to contrastive? We fine-tuned the collapsed 17M
edge model (4k FiQA, 1 epoch, no binary-specific loss) both ways, same data:

| fine-tune of dead 17M | dead% | NF int8×binary | NF retention | ArguAna ret | eff-rank |
|---|---:|---:|---:|---:|---:|
| *(untrained)* | 65% | 0.0876 | 24.2% | 18.3% | 1.6 |
| contrastive | **15%** | **0.2420** | **71.4%** | 78.6% | 8.0 |
| distillation (KD) | 67% | 0.0885 | 24.4% | 17.6% | 1.6 |

Contrastive revives it (65%→15% dead, 24%→71%) and re-conditions the geometry
(eff-rank 1.6→8.0). **KD is inert even from a 65%-dead base** — dead% 65%→67%, retention
flat, eff-rank unmoved. Distillation cannot un-kill a saturated checkpoint; contrastive
is the lever.

**7.2 Objective ablation — contrastive de-saturates, KD is inert regardless of
teacher.** From the same base (LateOn-unsupervised), same mined docs, varying only
the loss.

*Replicated (NE2, 3 seeds):* contrastive dead **7.8 ± 0.0%**, NF 0.3687 ± 0.0012,
ArguAna 0.2686 ± 0.0026; KD dead **11.2 ± 0.4%**, NF 0.3594 ± 0.0003, ArguAna 0.2416
± 0.0007. Δ(contr−kd) = **+0.009 ± 0.001** (NF) and **+0.027 ± 0.003** (ArguAna), both
clearing 2σ. Contrastive de-saturates and lifts binary NDCG; weak-teacher KD does not.

*Teacher distribution does not matter (NE3):* re-running KD with a **cross-encoder
reranker** teacher (bimodal-saturated: logits [−11.5, 10.8], sd **4.6**) vs the narrow
GTE teacher (sd **0.045**) gives the **same 12% dead** either way (kd_gte 12% /
kd_rerank 12%, vs contrastive 8%). So a saturated teacher does *not* drive saturation
— **KD is simply inert on saturation, leaving whatever the base had.** This refutes
"distillation is inherently saturating" (H8) and, with NE1's 0%-dead KD models,
completes the account: KD leaves the backbone's conditioning as-is; contrastive
changes it.

**7.3 A binary-aware loss is a marginal, headroom-gated top-up.** STE-sign at α≈0.1
vs the plain contrastive control (exhaustive int8×binary): ArguAna +0.023, SciDocs
+0.003, SciFact 0.000, NFCorpus −0.005; survives deployment (α=0.1 deployed ArguAna
0.2875 vs 0.2643 control). Beyond α≈0.2 it over-trades; β inert. Binary-*specific*
optimization is a minor add-on; the lever is generic contrastive conditioning.

**7.4 Report absolute binary NDCG, not retention-%.** In §7.2 contrastive's ArguAna
retention-% *falls* (84.4→77.9%) while binary NDCG *rises* (0.243→0.264): fine-tuning
lifted the int8 denominator more. Lead with absolutes.

---

## 8. The unifying account

Binary MaxSim ranks by `q · sign(x)`, so it preserves the float ranking to the extent
a document's **orthant pattern** carries its ranking variation — i.e. to the extent
its sign code is **unsaturated and discriminative**. Everything reduces to that one
quantity:

- **Where saturation comes from — a backbone default the objective overrides.**
  Pretrained encoders carry a default anisotropy/saturation (representation
  degeneration [Gao 2019]) that varies by architecture — but *not* by a clean size or
  dim law: projection dim is not a cause (NE7), and backbone size is non-monotonic
  under KD while contrastive de-saturates every size (NE8). Magnitude and query-int8
  are not factors either (NE4). So the backbone's default is real but secondary; it is
  the objective that determines the outcome.
- **Whether it is fixed — the final-stage objective.** Contrastive InfoNCE's
  *uniformity* pressure [Wang & Isola 2020] spreads negatives and de-saturates as a
  by-product of ranking (measured, §7.1–7.2). Distillation reproduces a teacher's
  *relative* scores — invariant to per-coordinate offset — and is **inert on
  saturation whatever the teacher's shape** (NE3). So a KD model's saturation is its
  backbone's saturation, unmodified; a contrastive model's is reduced.
- **Why it cannot be undone post-hoc.** The discriminative content lives at the
  trained zero boundary; centering, β, and ITQ all move or scramble it and cannot
  manufacture discrimination that was never encoded (§6).

Two loose ends the account leaves open: the residual variation at fixed dead% (0%-dead
mxbai-large 84% vs ColBERTv2 97%) is a *supervised* property (does the sign pattern
separate relevant from irrelevant?) that no unsupervised scalar we tried captures; and
capacity-as-*backbone-size* (vs dim, ruled out) is inferred from the released models,
not yet trained-controlled.

### 8.1 The theory, sharpened (2026-07-06): capacity, provenance, two ledgers

A 9-model × 4-corpus sweep (NFCorpus/SciFact/ArguAna/SciDocs, 36 cells) plus five new
targeted experiments (NE10–NE16) upgrade §8 from an account to a theory with confirmed
mechanisms. Three laws:

**Law 1 — Capacity: the unit is the live axis, not the dead fraction.** `sign(x)@0` is
SimHash in the model's own basis; an axis carries one bit iff the corpus straddles zero
there. Writing a token as `x = μ + z` (corpus-shared offset + token-specific signal),
an axis is dead iff `|μ_i|` outweighs the spread of `z_i`. The right predictor is the
surviving channel width — **live axes = dim × (1 − dead%)**: pooled over all 36 cells,
`log(live)` → retention has **r = +0.885** (+0.86…+0.92 per-corpus) vs −0.73 for dead%,
and it dissolves the §5.2 mid-band anomalies (answerai "47% dead yet fine" = 51 live
axes of 96; edge-32m "55% dead yet bad" = 29 of 64; the edge-17m collapse = ~15 live
axes, and no 15-bit code separates a corpus). dead% is corpus-stable (±6 pts across
datasets) — a model property. Corpus size only tightens margins (SciDocs uniformly
hardest, ordering never changes).

**Law 2 — Provenance: the footprint is the backbone's default unless the final stage
contains repulsion.** A shared doc-side offset shifts every document's MaxSim score
equally (`Σ_i q_i·μ`), so KL-distillation *and* InfoNCE are loss-level invariant to it;
KD's inertness is structural (no term ever pushes two doc embeddings apart), while
contrastive training removes shared components as uniformity pressure (they inflate all
doc–doc similarities) and reclaims wasted norm budget. Confirmed causally two ways:

- **NE13 (footprint trajectory).** Training the dead 17m one epoch, probing 5
  checkpoints: contrastive drives dead axes 31→23→8→7 while the shared-offset norm
  fraction falls 0.967→0.911→0.785→0.766 **in lockstep**; under pure KD both are flat
  to three decimals (31→32; 0.967→0.968). The per-axis offset-to-spread ratio is
  exactly the quantity the objective does or does not move. (Caveat: contrastive
  reduces the footprint's *axis coverage*, not necessarily its spectral mass — LateOn
  keeps a rank-1.5 dominant direction concentrated on ~16 axes; `mu_norm_frac` is
  still 0.77 after recovery. The axis-level statement is what replicates.)
- **NE15 (the repulsion test — decisive).** The dichotomy is NOT "KD vs contrastive";
  it is the presence of a repulsion term. ColBERTv2's paper (§3.2) states its loss
  verbatim: KL-distillation **plus in-batch negatives cross-entropy** ("against all
  passages corresponding to other queries in the same batch") — the 0%-dead "distilled"
  model had repulsion all along (and its native codec is 1–2 bits/dim; the checkpoint
  was developed under a near-1-bit regime). Test: from the dead 17m, alternate KD and
  in-batch-contrastive batches (half data each): dead 65% → **29%**, retention 24% →
  **68.8%** NF / 68.5% ArguAna — nearly the pure-contrastive recovery (15% / 71%) where
  pure KD stayed frozen (67% / 24%). Practical upshot: **add in-batch negatives to a KD
  stage and the checkpoint binarizes** — which is ColBERTv2's recipe.

**Law 3 — Two ledgers: given capacity, the residual is where the signal is stored, and
no unsupervised scalar can see it.** Within live axes, ranking signal splits between
the sign pattern (code ledger) and within-orthant magnitude (magnitude ledger); how a
model splits is a *supervised* quantity — a correlation between discarded magnitudes
and relevance — invisible to unsupervised geometry. This is why the ColBERTv2 /
mxbai-large "twin paradox" had to happen (identical on every doc scalar — 0% dead,
eff-rank 56 vs 47, sign_cos 0.80 — yet 97% vs 84% retention and ~0% vs 19% query-sign
penalty), and why the §5.3–5.4 scalar hunt had to fail. ColBERTv2 is the code-dominant
extreme — a de facto learned Hamming machine (signing *denoises* it: binary ≥ float on
SciFact/SciDocs). The query side: we first conjectured a geometric mechanism (signing
rescales a token's contribution by `√d/‖q‖₁` → importance redistribution toward spiky
tokens) and **NE10 refuted it** — mxbai-large has the most uniform queries in the fleet
(within-query spread variance 0.00004, 6× below ColBERTv2's) and still pays 18–25%.
What NE10 found instead: every model flips 36–56% of its token argmaxes under query
signing; **ColBERTv2 flips 36% and loses nothing**. Query robustness is
*value-stability under flips*, not flip avoidance — consistent with ColBERTv2's own
§3.3 cluster hypothesis (tight token clusters, the property its 1–2-bit residual codec
exploits): a within-cluster argmax hop is value-preserving; in a non-clustered space
the hop lands far away (answerai: 53% penalty at 45% flips). Direct test (proposed
NE19): per-flip score delta.

Open after the theory: why ColBERTv2 is code-dominant (era/backbone/serving-loop
accident vs trainable property); the precise mechanism converting repulsion into an
*axis-aligned* footprint reduction; residual mid-band scatter (LFM2.5 under-performs
its 55 live axes on ArguAna).

---

## 9. Hypotheses and their fate (the honest ledger)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Set by scale/dim | **Refuted** | 353M<150M (§5.1); dim-32 is 0%-dead (NE7/§5.5) |
| H2 | dead% predicts across recipes | **Supported (one-sided)** | 9-model dissociation; noisy above ~15% (§5.2) |
| H3 | A spectral-isotropy scalar is the law | **Refuted** | not monotonic (§5.3) |
| H4 | A threshold shift recovers dead-bit signal | **Refuted** | 8/8 collapse; in-sample μ best-case (§6.1) |
| H5 | A balance penalty improves binarization | **Refuted (inert)** | dead 7→3, flat (§6.2) |
| H6 | An ITQ rotation recovers it | **Refuted** | hurts saturated, neutral on good (NE5/§6.3) |
| H7 | Contrastive fine-tuning re-conditions a broken model | **Supported** | 17M 24→72% (§7.1) |
| H8 | Same-base contrastive de-saturates > (weak) KD | **Supported (replicated)** | Δ clears 3-seed noise (NE2/§7.2) |
| H9 | Distillation is inherently saturating / bad | **Refuted** | 0%-dead KD models; bimodal teacher ≠ more saturation (NE1, NE3) |
| H10 | Token magnitude explains residual variation | **Refuted** | sign_cos uncorrelated w/ retention (NE4/§5.4) |
| H11 | Query-int8 is a meaningful loss source | **Refuted** | docret ≈ ret everywhere (NE4/§5.4) |
| H12 | A saturated teacher drives sign saturation | **Refuted** | reranker (sd 4.6) = GTE (sd 0.045), 12% (NE3) |
| H13 | Saturation originates in backbone **size** | **Refuted** | fresh-head KD across Ettin 17m–400m is non-monotonic (46/39/53/29/36%); contrastive de-saturates every size to ~0% (NE8) |
| H14 | The **objective** de-saturates independent of dim & backbone size | **Supported** | contrastive→0–1% dead at every dim (NE7) and every Ettin size (NE8); KD→30–53%, same backbones (NE8) |
| H15 | A binary-aware loss is the key | **Refuted (marginal)** | α≈0.1 top-up only (§7.3) |
| H16 | Live-axis **count** (not dead fraction) is the capacity law | **Supported** | log(live)→retention r=+0.885 pooled, beats dead% on all 4 corpora; fixes mid-band inversions (§8.1) |
| H17 | The operative recipe variable is a **repulsion term**, not KD-vs-contrastive | **Supported** | KD+IB rescues the dead 17m (65→29% dead, 24→69% ret) where pure KD is frozen; ColBERTv2's loss = KL + in-batch CE verbatim (NE15/§8.1) |
| H18 | The per-axis footprint \|μ_i\|/σ_i is what the objective moves | **Supported** | lockstep fall under contrastive, flat to 3 decimals under KD, 5 checkpoints (NE13/§8.1) |
| H19 | Query-signing loss = importance redistribution by token spread | **Refuted** | mxbai-large most-uniform queries, still pays 18–25%; flips are universal (36–56%) (NE10/§8.1) |
| H20 | Query robustness = value-stability under argmax flips (cluster redundancy) | **Refuted (inverted)** | v2 has the LARGEST per-flip Δ (0.193; doc loss 4.5%) & pays 0; answerai smallest (0.013; 0.4%) & pays 53% (NE19) |
| H28 | Carrier = doc-level margins vs perturbation | **Supported** | v2 m(1,2)=8–21% of top score ≫ its 4.5% perturbation; answerai top-10 packs in 0.69%; Spearman ρ (i8 vs signed q, top-10) exactly monotone with penalty both corpora (NE25) |
| H29 | The query-penalty split is the ModernBERT architecture | **Refuted** | at 8k rows ModernBERT-base reaches effR 54, q-pen 3.0–8.2% (BERT-class); Ettin, same arch/recipe/data, stays at effR 31.8, 18–33% (NE22b) |
| H30 | Backbone effects = pretraining lineage (weights), not arch class | **Supported** | robust cluster = {bert-base, its distillate}; RoBERTa pays 16.3% at float MATCHED to bert-base (0.182 vs 0.185); ELECTRA (BERT arch) pays 18.6%; bert-large fine → mxbai-large's fragility is pipeline (NE22b/NE23) |
| H31 | Doc-side ledger placement is also backbone lineage | **Supported** | fresh heads all ~0% dead yet doc retention spans 47–96% (Ettin 47–85 vs MB-base ~90 flat); fresh-Ettin = manufactured capacity-curve outlier (NE22b) |
| H32 | Pre-norm streams collapse mid-depth; post-LN stays clean | **Supported (shape corrected)** | MB-family mid-layers effR/H 0.001–0.02, ‖μ‖ 67–90% of token norm; BERT ≥0.07, μ≈0.50; footprint peaks EARLY then rotates off-axis; profile does NOT predict head trainability (NE24) |
| H21 | Dead-axis magnitudes carry no ranking signal | **Supported** | selective centering: no gain anywhere (NE11) |
| H22 | Centering damage = boundary-move on live axes | **Refuted** | dead-axes-only centering collapses (LateOn 16/128 axes: arg 0.296→0.008); argmax **selects** injected noise (NE11) |
| H23 | mxbai-large's residual ledger is axis-level weighting | **Refuted** | weighted Hamming moves it 0.2585→0.2579; token-level, unrecoverable (NE14) |
| H24 | Any anti-concentration pressure substitutes for repulsion | **Refuted (3 arms)** | KD+logdet: 0% dead/effR 24 yet binary ≈0.12 (float broken 0.363→0.272); KD+balance: geometry unmoved, ≈0.12; KD+‖μ‖²-only (surgical null-space arm): 0% dead/effR 20 yet float breaks worse, binary 0.085 — vs KD+IB/contrastive ≈0.24 (NE17/NE17b) |
| H25 | A stage-aware codec (centroid + per-cluster-λ sign residual) rescues saturated models without retraining | **Supported (corpus caveat)** | NF: LFM 0.281→0.358 (95% ret), GTE 0.275→0.364 (95%), 17m 24→79%; residual dead%=0 everywhere; hurts LateOn/17m on long-query ArguAna → per-checkpoint eval decides (NE20) |
| H26 | The backbone controls two-sided sign robustness at fixed objective | **Supported (n=1/arch)** | fresh contrastive heads, all else fixed: bert-base → effR 55.1, q-pen 2.2%/0.9% (reproduces ColBERTv2's 55.9/0.800 geometry); ettin-150m → effR 42.7, q-pen 29.2%/12.6%, worse doc retention at equal ~0% dead (NE22) |
| H27 | The released ModernBERT-family rank-1.5 spectrum is the arch default | **Refuted** | fresh contrastive Ettin-150m head → effR 42.7; the rank-1.5 signature is lineage, not architecture (NE22) |
| H33 | The fleet's dead% ordering is decided by *documented* final-stage repulsion presence | **Supported (documentary)** | JaColBERTv2.5 removed IB after a float-only ablation (0.581→0.580 / 0.681→0.682); its KD-only descendants are the dead cluster (answerai 47%, GTE-MC 64%, edge 55–65%); LateOn card = no KD, pure contrastive → 12%; v2 = KD+IB → 0% (NE27) |
| H34 | Edge-model saturation is partly *imported* by Stella-style L2 embedding distillation (‖y−ŷ‖² is offset-sensitive, unlike score-KD) | **Supported (mechanism live, class-general)** | StellaV5 μ-frac 0.644 / 6% dead; bge-base 0.794 / 19% — offset-heavy is the single-vector norm; L2 imports it, downstream KD-only provably can't remove it; source attribution (Stella vs Ettin default, NE24) open (NE29) |
| H35 | Training *through* the codec (α-STE inside the contrastive CE — the BPR pairing) improves saturated-checkpoint **rescue** beyond repulsion alone at matched budget | **Supported (small, uniform; replicated)** | NE28: best arm on both subjects carries STE (+2–4 pts); side flips with headroom (two-sided at 48d; one-sided at 128d; the 48d one-sided harm was WEIGHT not side — α=0.1 ≈ control, NE30a). Under the FULL recipe (mixture+self-KD), doc-STE still wins 11/12 cells vs the no-STE control and replicates at seed 43 within ±0.007 (NE30b/c/d) |
| H36 | The KD anchor's float preservation is limited by anchor-data **domain coverage**, not KD weight | **Supported** | doubling KD:IB from 1:2 to 1:1 leaves SciFact float drift unchanged (−0.025 → −0.028) while gates pass wherever the mixture covers the domain; the seven LightOn subsets contain no scientific text (NE30b′) |
| H37 | Code-aware training transfers **across codecs** (robustness is a property of where signal sits, not of the specific code trained through) | **Supported (n=1 direction, FDE→sign)** | LateOn-regularized, trained only through MUVERA/SMVE SimHash projections, posts fleet-best axis-aligned sign robustness (NF i8×b 0.3742, 97.0% ret) it was never trained for; reciprocal sign→FDE direction untested (NE33) |

---

## 10. Limitations and open questions

- **The backbone default is under-characterized.** NE8 (fresh-head Ettin 17m–400m
  sweep) *refuted* backbone size as the law — KD-residual dead% is non-monotonic and
  contrastive de-saturates every size. What architectural property sets a backbone's
  default saturation (isotropic BERT-large vs anisotropic Ettin-tiny) is open, but it
  matters little in practice since contrastive overrides it. Caveat: NE8's fresh-head,
  2-epoch models are under-trained, so its *retention* numbers are noise (150m showed
  131%); only its dead% and the clean objective contrast (contrastive ~0% vs KD
  30–53%) are trustworthy.
- **Residual variation at fixed dead% is unexplained** by any unsupervised scalar
  (mxbai-large 84% vs ColBERTv2 97%, both 0% dead) — likely a supervised
  discrimination property; a float-vs-binary score-fidelity metric on held-out
  (query, doc) pairs would test it.
- **Single base for the objective ablation** (LateOn-unsup); teacher-distribution and
  seed variance are now controlled (NE2/NE3), but only one base.
- **ArguAna over-reliance** — a symmetric 1-relevant-doc BEIR outlier supplying the
  most extreme numbers; NE6 (FiQA-test/Touché) would de-risk §6.1.

### Experiments (NE1–NE16)

- ✅ **NE1** mixed-provenance dissociation — dead% predicts across recipes; KD ≠ bad.
- ✅ **NE2** seeds — contrastive>KD same-base gap clears noise.
- ✅ **NE3** teacher-distribution KD — saturated teacher ≠ more saturation; KD inert.
- ✅ **NE4** magnitude/query-int8 — both ruled out; all loss is doc-signing.
- ✅ **NE5** ITQ — no post-hoc transform recovers binarizability.
- ✅ **NE7** projection-dim sweep — dim does not cause saturation.
- ✅ **NE8** backbone-size sweep — size is *not* the law (refuted H13); the objective
  cleanly de-saturates every size (contrastive ~0% vs KD 30–53%).
- ✅ **NE6** non-ArguAna eval — the §6.1 centering collapse holds on FiQA.
- ✅ **NE9** recovery — continuing distillation leaves a dead checkpoint dead (65→67%,
  ret 24→24%); switching to contrastive recovers it (15% dead, 71–79% ret).
- ✅ **NE13** footprint trajectory — |μ_i|/σ_i falls in lockstep with dead% under
  contrastive, flat under KD (5 checkpoints) (§8.1).
- ✅ **NE15** repulsion test — KD + in-batch negatives rescues the dead 17m
  (65→29% dead, 24→69% ret); ColBERTv2's recipe verified as KL + IB (§8.1).
- ✅ **NE10** query-side geometry — REFUTED H19: query uniformity doesn't order the
  penalty (mxbai-large most uniform, pays 18–25%); flips universal (36–56%); ColBERTv2
  flips 36% losing 0 → value-stability/cluster-redundancy is the surviving account (H20).
- ✅ **NE11** selective centering — "no gain" confirmed (dead-axis magnitude is
  ranking-noise, H21); "no collapse" REFUTED: even 16/128 centered axes collapse
  (LateOn arg 0.296→0.008) — MaxSim's argmax *selects* injected noise; the trained-0
  threshold is load-bearing because it keeps dead axes constant (H22).
- ✅ **NE14** weighted Hamming — mxbai-large unmoved (token-level ledger, H23 refuted);
  side-win: saturated mid-band models gain +0.02–0.03 free (answerai +0.023, GTE
  +0.021, LFM +0.032 arg) from a per-axis query rescale at serving time.
- ✅ **NE16** two-sided STE — NEGATIVE: α=0.1 query+doc STE leaves binaryxbinary at the
  doc-only treatment level (nf 0.3458 vs 0.3459); two-sided code dominance is not a
  bolt-on (larger α/data untested).
- ✅ **NE17** KD + entropy regularizer — REFUTED H24 with a clean dissociation: balance
  (γ=0.3) leaves geometry unmoved (60% dead, float intact); logdet (γ=0.1) fully fixes
  geometry (0% dead, effR 23.9) but breaks float (0.363→0.272); BOTH land at absolute
  binary ≈0.12 vs ≈0.24 for KD+IB/contrastive. Capacity is necessary, not sufficient —
  task-aligned repulsion migrates signal onto freed axes; unsupervised entropy does not.
- ✅ **NE17b** KD + ‖μ‖²-only suppression — REFUTED our own "float intact" prediction:
  fully de-saturates (0% dead, effR 20.3) but float breaks *worse* (nf ~0.23, arg
  0.186; binary 0.085). With ‖μ‖ ≈ 0.97 of token norm the offset is structural — the
  token cone is built around it — so KD's loss-level invariance does not make removal
  parameterizable in isolation. Net of 3 arms: *you cannot impose the geometry with a
  regularizer; the task must earn it through contrast.* (Caveat: single γ per arm,
  single seed, 17m only.)
- ✅ **NE18** mid-band recovery — HALF the gap, ON the capacity curve: LFM 350M with 4k
  contrastive rows (α=0.1 arm best): nf i8xb 0.2810→0.3240 (float held 0.3735), arg
  0.1711→0.2649. dead only →30–33% (350M conditions slower than 17m); at ~89 live axes
  the capacity curve predicts ~87–90% retention — measured 86.6%, i.e. the shortfall vs
  LateOn (0.3633) is under-conditioning at this budget, not a proven ledger wall. The
  α=0.1 STE term finally shows its regime from a saturated start: better absolute
  binary AND better float preservation than α=0 (H15 stands as "marginal" but its
  designed-regime value is now demonstrated).
- ✅ **NE20** stage-aware codec (eval-only): `q·c + (q∘λ)·sign(d−c)`, k-means K=4096
  (below PLAID convention → conservative). NF: LFM 95% ret / GTE 95% / 17m 79% with
  ZERO retraining — beats the NE18 fine-tune; residual dead%=0 for all 9 (centroids
  absorb the common mode); centroid-only alone scores 0.21–0.34. Contrast with NE11:
  centering *discards* the offset (noise into an argmax), the codec *keeps* it in float
  per cluster. Caveat: hurts LateOn (0.296→0.259) and degenerate 17m on long-query
  ArguAna — deployment = per-checkpoint decompose, pick pure-sign vs codec. ~17.5
  B/token vs 16 pure-sign.
- ✅ **NE22** backbone test — fresh contrastive heads, all else fixed: bert-base
  reproduces ColBERTv2's geometry (effR 55.1 vs 55.9, sign-cos 0.800, q-pen 2.2%/0.9%);
  ettin-150m (ModernBERT arch): effR 42.7 but q-pen 29.2%/12.6% and worse doc retention
  at equal ~0% dead. The v2 anomaly is largely the backbone; two-sided robustness is
  purchasable by choosing BERT-base. Also refutes "rank-1.5 is the ModernBERT default"
  (lineage, not arch). Open: which component (post-LN? positions? tokenizer? data) —
  needs a surgical ablation; and why BERT-large has the doc geometry but not the query
  robustness.
- ✅ **NE19** per-flip value delta — REFUTED H20 by inversion: v2's flips lose the MOST
  token value (Δ 0.193, doc loss 4.5% — fleet maxima) at ~0% penalty; answerai's are
  near-neutral (0.013, 0.4%) at 53%. No token-level quantity carries query robustness.
- ✅ **NE25** margin census — H28 SUPPORTED: v2 top-1→2 margin 8.0% NF / 21.2% Arg of
  top score (≫ its perturbation); answerai's whole top-10 within 0.69%; Spearman ρ of
  top-10 under query signing exactly monotone with penalty on both corpora
  (0.751→0.312 NF). Margins widen with query length → penalty is a short-query
  phenomenon. Residual: LateOn vs mxbai-large equal margins, different penalties —
  perturbation differential unmeasured.
- ✅ **NE22b** backbone at 2× data × 4 corpora — architecture attribution DIES:
  ModernBERT-base reaches effR 54 / q-pen 3.0–8.2% (BERT-class) while Ettin (same
  arch, recipe, data) stays effR 31.8 / 18–33% / 47–85% doc-ret; bert-large ≈ base
  (mxbai-large's fragility = pipeline); ModernBERT-large fresh heads fail to bootstrap
  (float ≈0.01, excluded). Lineage, not arch class. Doc-side: at ~0% dead everywhere,
  retention spans 47–96% by lineage = Law 3 under experimental control.
- ✅ **NE23** component census — post-LN not sufficient (RoBERTa 16.3% NF q-pen at
  float MATCHED to bert-base), BERT-arch not sufficient (ELECTRA 18.6%), rank doesn't
  carry it (ELECTRA effR 50.5), maturity insufficient (the RoBERTa row); survivor =
  bert-base weights or their distillate (DistilBERT 3.5%/1.2%, best float 0.233).
  DeBERTa-v3 failed to load (tiktoken dep).
- ✅ **NE24** depth probe — MB-family raw streams collapse mid-depth (effR/H 0.001–0.02,
  ‖μ‖ 67–90%), BERT stays clean (≥0.07, μ≈0.50); footprint peaks early (L3–8) then
  rotates off-axis; profile does NOT predict which lineages recover.
- ✅ **NE26** doc-side margin census (twin of NE25) — ρ_doc = Spearman(float vs
  binary-doc scores over the float top-10) is exactly monotone with doc retention on
  both corpora: v2 ρ 0.747 / top-1 survives 77% (float margins m(1,2) 10.03% NF,
  23.70% Arg) … 17m ρ 0.222 / 28%. LateOn is the second currency made visible: 95%
  retention with tiny 0.73% margins — near-zero perturbation instead of wide margins.
  Two ways to survive the sign: perturb little (repulsion-trained code carries the
  signal) or margin wide (lineage). ☐ perturbation-differential probe (LateOn vs
  mxbai-large residual). ☐ why BERT-lineage weights produce wide-margin heads.
- ☐ **Open** what architectural property sets a backbone's default saturation; why
  ColBERTv2 is code-dominant (Law 3); mechanism of repulsion → axis-aligned footprint.

---

## 11. NE27 — recipe receipts, production-codec audit, and the binary-aware-training literature (2026-07-06)

### 11.1 Production-codec audit (mixedbread API, live)

Probed `api.mixedbread.com/v1/embeddings` (user-supplied key) with multi-format
responses (`float`,`int8`,`binary`,`ubinary`) on `mxbai-embed-large-v1`, 8 varied
sentences:

- **binary/ubinary = `packbits(float > 0)`, little-endian.** Bit-for-bit match 100.0%
  (big-endian unpack reads as ~50% — a red herring worth documenting). **Threshold
  zero, no rotation, no calibration** — the production codec is exactly the codec
  studied here, so by the iron rule (§6) any robustness they report is trained-in.
- **int8 = round(float × s) with one global scale s ≈ 446** (per-dim affine
  regression: slope median 446.1, IQR 440.6–450.4, intercept ≈ 0, R² 0.999). A fixed
  calibration constant — not per-vector max-abs (would be 611 on the probe vector),
  not per-dim min/max.
- **Wholembed v3 — the "trained with this tradeoff in mind" model — is a
  *late-interaction* model and is closed.** Not addressable via the embeddings
  endpoint under any plausible ID (8 tried → `model_not_found`); the OpenAPI enum
  lists only the three single-vector models; the blog has no code snippet; it serves
  only through their managed search product. **No geometry census is possible.** Their
  reported ladder (internal bench avg): float 90.26 / i8×i8 90.27 / i8×b 89.65
  (−0.61) / b×b 83.06 (−7.20).

### 11.2 Recipe receipts — the fleet's dead% has public documentation

| model | year | final-stage loss (public source) | dead% | NF ret. |
|---|---|---|---|---|
| ColBERTv2 | 2021 | KL-div KD **+ in-batch CE** (paper §3.2) | 0% | 97% |
| LateOn | 2025 | **contrastive only** — card: KD phase *not applied*; 1.4B-pair unsup + supervised hard-neg | 12% | 95% |
| answerai-colbert-small | 2024 | JaColBERTv2.5 recipe = KL-div **only** (IB explicitly removed) | 47% | 85% |
| GTE-ModernColBERT | 2025 | "small KD step" on a dense base (LightOn) | 64% | 72% |
| mxbai-edge-32m/17m | 2025 | KL-div-only ColBERT stage (bs128, 16-way), downstream of **Stella-style L2 *embedding* distillation** from StellaV5-1.5B (arXiv 2510.14880) | 55/65% | 61/24% |
| Wholembed v3 | 2026 | closed — "trained with this tradeoff in mind" | ? | −0.61 pts (theirs) |

**The deletion event.** JaColBERTv2.5 (arXiv 2407.20750, Table 3) ablated in-batch
negatives on float NDCG@10: JQaRA 0.581→0.580, MIRACL-ja 0.681→0.682 —
indistinguishable — and wrote *"Following the results of this experiment, we remove
the use of in-batch negatives from our training recipe."* The KD-only recipe became
the community standard; its descendants are the dead-bit cluster. This corroborates
H17 **from the field's own documents**, with no dependence on our probes. The ablation
was sound on the metric watched; the sign channel was on no dashboard.

**H34 (new, untested).** The edge models' Stella-stage loss is ‖y−ŷ‖² on embeddings —
*not* offset-blind (score-KD provably is): it copies the teacher's mean vector into
the student. Their saturation may be partly *imported* from StellaV5, then frozen by
the KD-only ColBERT stage. Probe (NE29, $0): cosine(μ_edge, projected μ_StellaV5).

### 11.3 Literature map — training for doc-side binarizability

1. **BPR** (Yamada et al., ACL 2021; arXiv 2106.00882) — the canonical
   train-through-the-codec result, single-vector: a `sign()` hash layer on DPR with a
   straight-through estimator and a **two-task loss** — Hamming-based candidate
   generation + reranking with a **float query against binary documents**. That is
   the asymmetric scheme, trained end-to-end, in 2021: NQ/TriviaQA index 65GB→2GB
   (~32×) at answer-parity. Follow-up: learning-to-hash domain adaptation for
   zero-shot/BEIR (arXiv 2205.11498).
2. **Hashing classics (2011–2019)** — ITQ (rotation), HashNet (tanh annealing),
   GreedyHash (STE), spectral-hashing balance/independence constraints. Consistent
   lesson: the constraints work only **jointly with the task loss**, never as
   bolt-ons — which is what NE17 re-established at modern scale (balance/logdet/‖μ‖²
   all break or stay inert without contrast).
3. **Production single-vector QAT claims (recipes undisclosed):** Cohere embed-v3
   "compression-aware training" (2024-04); mixedbread binary-MRL (claims 90.76%
   binary retention @512d) and Wholembed v3; **Qwen3-VL-Embedding (Jan 2026, arXiv
   2601.04720)** — integrates MRL **+ QAT** in the training pipeline, the most recent
   mainstream adoption. All single-vector; none publish the loss.
4. **Theory (single-vector):** arXiv 2605.17524 (Xiao, May 2026) — covariance
   structure (off-diagonals contribute 30–50% of signal) sets binary ranking
   fidelity; **coordinate heterogeneity** sets the marginal value of each bit and
   whether random rotation helps or hurts; InfoNCE embeddings, 9 families × 18
   datasets. Convergent with the capacity law at the single-vector level; no
   multi-vector, no MaxSim, no provenance intervention. Also arXiv 2505.18113 (STE
   sample-complexity theory for 1-bit).
5. **The gap.** No public binary-aware training recipe exists for late interaction —
   the setting where MaxSim's argmax amplifies quantization noise. KD+IB with
   measured sign retention (NE15) appears to be the only public one.

### 11.4 Proposed follow-ups

- ✅ **NE28** (2026-07-06, ~$5) — *the BPR pairing at rescue scale*. **Correction
  recorded:** our α-STE term in `_make_loss` already IS the BPR pairing —
  `(1−α)·CE(float MaxSim) + α·CE(MaxSim(q, sign_STE(D)))` — STE inside the
  *contrastive* CE, never KD-coupled. All arms 10k FiQA queries × 2 epochs (~2.5×
  prior volume), matched-budget α=0 controls. Results (NF / Arg, i8×b abs and ret):

  | subject | α=0 control | α=0.3 doc-side | α=0.3 two-sided |
  |---|---|---|---|
  | 17m (48d, dead pole) | 0.2584 / 76.9% · 0.2597 / 79.2% (dead 65→6%) | 0.2502 / 74.7% · 0.2255 / 70.6% ✗ | **0.2653 / 80.4% · 0.2766 / 82.2%** (dead 0%) |
  | LFM-350M (128d) | 0.3156 / 88.1% · 0.2201 / 78.0% (dead 57→27%) | **0.3265 / 88.5% · 0.2643 / 79.1%** | 0.3220 / 86.4% · 0.2379 / 76.0% (dead 32%) |

  Findings: (1) **volume alone was the biggest lever** — the α=0 controls beat every
  prior rescue (17m 24→77–79%; LFM 0.281→0.3156 NF), and each arm lands ON the
  capacity curve at its live-axis count (45–48 live → 77–82%; 87–93 live → 86–88%) —
  the 17m's residual gap to ColBERTv2 is now the 48-dim channel, not training.
  (2) **The best arm on both subjects carries an STE term, but the side flips**:
  two-sided wins on the low-dim dead model (+3.5/+3.0 ret pts, 0% dead); doc-side
  wins on the 128-d model and *protects float against fine-tuning drift* (Arg float
  0.3341 vs control's collapsed 0.2826; NF float 0.3684 vs 0.3584) — the control's
  FiQA-shift forgetting is the practical hazard the STE term damps. One-sided HURTS
  the 17m (−2/−9 ret pts): BPR's rerank task does not port naively to MaxSim at the
  capacity ceiling. (3) My registered prediction (α ≈ control; ts edge on b×b) was
  wrong twice: the STE edge is real (2–4 pts, both corpora, both subjects) and it
  shows on i8×b, not b×b. Single seed per arm — orderings consistent across two
  corpora within subject.
- ✅ **NE29** (2026-07-06, ~$0.3) — H34 teacher-offset probe. StellaV5-1.5B (the edge
  L2-distill teacher), NFCorpus n=512, trained 1024-d head: **μ-fraction 0.644**,
  6% dead axes, mean |mean-sign| 0.45, p90 0.85, effR 59.6. Reference probe
  bge-base-en-v1.5 (top-tier pure-contrastive single-vector): μ-fraction **0.794**,
  19% dead — i.e. offset-heavy is the *class norm* for single-vector sentence
  spaces, which survive it by Law 1 (600–960 live axes ≫ knee). Verdict: the import
  mechanism is live and *general* — any embedding-L2 distill from a standard
  single-vector teacher imports a large offset; the edge pipeline then finishes with
  offset-blind KD only. This also unifies GTE-ModernColBERT (dense contrastive base
  + small KD) with the edge models without extra machinery. Attribution between
  imported-from-Stella and Ettin's own mid-depth collapse (NE24) stays open — needs
  their intermediate checkpoints.

---

## 11.5 NE30 — from mechanism to a shippable variant (2026-07-07)

Goal shift: build "edge-32m-binary" under the release constraint **float must not
drop** (gate: variant float within 0.010 of base on every eval corpus).

- **NE30a** (32m rescue smoke, FiQA 10k×2ep): control 91.6% NF / 83.4% Arg ret
  (dead 55→2%) — **beats the 85–87% capacity forecast**; no H31 Ettin discount for
  fine-tuned heads. At 64d neither STE side wins on capacity (both ≈ +0.006 abs,
  all drift-mediated); two-sided starts *costing* geometry (8% dead, effR 18.9→11.8).
  Plus the 17m confound closure: one-sided α=0.1 ≈ control (77.1/77.8%) → NE28's
  one-sided harm at 48d was **weight, not side**.
- **NE30b/c** (recipe vs no-STE control; mixture 20k = msmarco:8000 + nq:2500 +
  hotpotqa:2500 + fever:2000 + squadv2:1500 + trivia:1500 + fiqa:2000; **self-KD**
  anchor = 10k tuples scored by the base checkpoint, KD⇄IB round-robin; 2ep bs16):

  | corpus | base float / i8×b (ret) | +STE (NE30b) floatΔ / i8×b (ret) |
  |---|---|---|
  | NFCorpus | 0.3621 / 0.2198 (60%) | +0.005 ✓ / 0.3397 (92.2%) |
  | SciFact | 0.7447 / 0.5583 (75%) | −0.025 ✗ / 0.6882 (95.6%) |
  | ArguAna | 0.3327 / 0.1719 (52%) | −0.005 ✓ / 0.2864 (87.5%) |
  | SciDocs | 0.1646 / 0.0832 (51%) | −0.007 ✓ / 0.1437 (90.9%) |

  Dead axes **0/64**; sign_cos **0.8007 — ColBERTv2's exact signature**,
  manufactured on an Ettin-lineage 32M model. STE attribution under the full
  recipe: +STE beats no-STE on **11/12 cells** (float 4/4, i8×b 3/4, b×b 4/4),
  gates 3/4 vs 2/4 → STE stays.
- **NE30d** (seed 43): replicates within ±0.007 i8×b / ±1 pt ret everywhere;
  sign_cos 0.8009. SciFact drift reproducible (−0.025 both seeds); ArguAna's gate
  flips with seed noise (−0.005 → −0.014) — the 0.010 gate ≈ seed σ there.
- **NE30b′** (KD:IB doubled to 1:1): SciFact float **unchanged** (−0.028), binary a
  wash → the drift is **data-limited, not weight-limited** (H36). Recipe frozen at
  1:2.
- **Frozen recipe:** 20k seven-subset mixture + self-KD (1:2) + in-batch negatives
  + α=0.3 doc-side STE, lr 1e-5, 2 epochs, bs16. Known trade on the 32m, disclosed:
  SciFact float −0.025 (96.6% of base) against +23% absolute binary NDCG there.
- **LFM-350M transfer (H100, recipe applied blind): 4/4 float gates PASS** —
  NF −0.009 / i8×b 0.3487 (93.9% ret, base 73%); SciFact **+0.010** / 0.7204
  (95.5%, base 79%); ArguAna −0.008 / 0.2985 (87.0%, base 49%); SciDocs −0.007 /
  0.1523 (86.1%, base 61%). b×b: NF 0.192→0.339, SciFact 0.476→0.711. Geometry:
  dead 5/128 (3.9%), sign_cos 0.804 (v2 signature again), effR 19.5. NF i8×b
  0.3487 = 96% of LateOn's 0.3633 at 0.371 float. Notably SciFact float *improved*
  on LFM where the 32m failed its gate → the H36 domain-coverage limit is
  backbone-dependent (a stronger backbone holds uncovered domains through the same
  anchor). **The recipe transfers across architecture and size unchanged** — the
  program's closing result.
- **Caveat for release:** all NE30 runs were eval-only (`save_strategy="no"`); the
  validated recipe is frozen but the variant *weights* were not persisted. A
  save+push rerun costs ~$1 (32m) / ~$4 (LFM). (Fixed: `save_to` now persists
  weights to the volume before eval; the v2 ship-run uses it.)
- **v2 ship-run (bs64, lr 2e-5, GTE-teacher KD, 50k mixture; H100, weights SAVED
  to `volume:/cache/variants/edge-32m-binary-v2`):** float **above base on 2/4**
  (NF +0.0047, **ArguAna +0.0062** — the corpus where base is weakest), SciDocs
  −0.0053 PASS, SciFact −0.0117 (narrow FAIL, but the external teacher **halved**
  v1's −0.025 drift). i8×b 0.3270/0.6888/0.2957/0.1433 (ret 87–94%); geometry
  dead 0/64, sign_cos 0.803 (fourth checkpoint to converge on the v2 signature).
  vs v1 (bs16/self-KD/20k): v2 wins float everywhere and ArguAna binary
  (+0.009), loses NF binary (0.3270 vs 0.3397) — teacher/batch/volume changed
  together (the disentangling bs16-v2 arm OOM'd and was not rerun for budget), so
  the NF-binary dip is unattributed. Registered "float>base on ≥3/4" prediction:
  MISSED (2/4); magnitudes half of predicted. "Outperform its own" verdict:
  **partially achieved, teacher-limited** — next lever is a stronger teacher
  (cross-encoder rerank of the same tuples), not more batch.

## 11.6 NE33 — audit of `lightonai/LateOn-regularized`: cross-codec transfer in the wild (2026-07-07)

LightOn released a regularized LateOn trained for **MUVERA/SMVE** robustness
([blog](https://huggingface.co/blog/lightonai/lateon-regularization)): loss =
`(1−α)·MaxSim contrastive + α·contrastive on the projected (FDE) representations`,
STE through the projection, fixed random projections, cheap supervised stage.
Structurally, that is exactly our α-recipe with the codec swapped: **sign() →
SimHash-bucketed FDE**. Their reported result: MUVERA rk=0 NDCG 2.89 → 40.80 with
PLAID held (55.28 → 55.72); geometry MORE anisotropic (corpus stable rank −26%,
top-eigenvalue share 21.3 → 27.0%) — "the model learned which dimensions to
concentrate into."

Our fleet probe of the released checkpoint (nfcorpus/arguana, exhaustive MaxSim):

| | float | i8×i8 | i8×b | b×b | ret (÷i8×i8) |
|---|--:|--:|--:|--:|--:|
| NFCorpus | 0.3836 | 0.3857 | **0.3742** | 0.3593 | **97.0%** |
| ArguAna | 0.3531 | **0.3280** | 0.2946 | 0.2322 | 89.8% (83.4% vs float) |

sign-health: dead 18/128 (14%), mean|balance| 0.506, **effR 1.40**, sign_cos 0.431.

Findings:
1. **Cross-codec transfer (new — they did not measure sign quantization):**
   training through the SimHash/FDE code also bought axis-aligned **sign**
   robustness — 97.0% NF retention at LateOn-level float, the fleet-best absolute
   i8×b we have measured (0.3742 > v2's 0.3229, > our best rescue 0.3487-LFM).
   Trained code-robustness generalizes across codecs (direction tested: FDE→sign).
2. **Law 3 in the wild:** effR 1.40 is the most collapsed spectrum we have ever
   measured — below raw ModernBERT streams — on the best binarizer in the fleet.
   Unsupervised spectral scalars do not decide binary fate (H3, H24 vindicated at
   production scale); their "learned which dimensions to concentrate into" is the
   two-ledgers insight, independently derived.
3. **New issue for them: int8 QUERY quantization regresses on ArguAna** —
   i8×i8 0.3280 vs float 0.3531 (−7.1%), unique in the fleet (int8 lossless
   everywhere else, all sessions). Their blog doesn't cover quantization; worth
   reporting upstream before anyone runs it behind an int8-query pipeline.
4. **The recipe family closes:** BPR 2021 (STE×contrastive through sign,
   single-vector) ≡ our α-STE / NE28-NE30 (sign, multi-vector) ≡ LightOn
   MUVERA-STE (FDE, multi-vector). "Train through the code you serve" now has
   three independent instances — and the robustness it buys is not codec-local.
   Proposed reciprocal test (H37): evaluate our sign-trained NE30 variants under
   MUVERA — does sign→FDE transfer hold in the other direction?
- **Infra note:** three run-pairs died to client-side cancellations during a local
  network flake (03:2x–04:2x UTC); cancellations do NOT trigger Modal `retries`.
  Now standard: `modal run --detach`, RESULT dict logged remotely, base-eval +
  self-KD cached to the volume and committed immediately.

---

## References (informal)

Khattab & Zaharia 2020 (ColBERT); Santhanam et al. 2022 (ColBERTv2/PLAID — a KD recipe
that binarizes losslessly here); mixedbread 2024 (asymmetric quant; Wholembed "trained
for quantization robustness"); Gong & Lazebnik 2011 (ITQ); Timkey & van Schijndel 2021
(rogue dimensions; standardization); Gao et al. 2019 (representation degeneration);
Ethayarajh 2019 (anisotropy); Wang & Isola 2020 (alignment & uniformity — the §8
contrastive mechanism); Hofstätter et al. 2020 (margin-MSE / cross-encoder KD); mxbai-
edge tech report (arXiv 2510.14880; reranker teachers, saturated targets; KL-div-only
ColBERT stage + Stella L2 embedding distillation); LFM2 tech report (arXiv
2511.23404); LateOn blog (contrastive + NV-Retriever mining, no KD); Yamada et al.
2021 (BPR, arXiv 2106.00882 — STE through sign + asymmetric float-q×binary-d loss);
Clavié 2024 (JaColBERTv2.5, arXiv 2407.20750 — the in-batch-negatives removal);
Xiao 2026 (arXiv 2605.17524 — covariance/heterogeneity theory, single-vector);
Qwen3-VL-Embedding 2026 (arXiv 2601.04720 — MRL+QAT in production training); Cohere
2024 (compression-aware training).

---

## Appendix — methods and full tables

**Models.** Original five: `mixedbread-ai/mxbai-edge-colbert-v0-{17m,32m}` (48/64),
`LiquidAI/LFM2.5-ColBERT-350M` (128), `lightonai/LateOn` & `LateOn-unsupervised`
(128). NE1 public set: `colbert-ir/colbertv2.0`,
`answerdotai/answerai-colbert-small-v1` (96), `lightonai/GTE-ModernColBERT-v1`,
`mixedbread-ai/mxbai-colbert-large-v1` (128). NE7 backbone:
`answerdotai/ModernBERT-base`. NE3 reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Training.** PyLate ColBERT on Modal A10G. Data: `lightonai/embeddings-fine-tuning`
FiQA slice, 4000 queries, 1 pos + 3 hard negatives (rank ≥15), batch 16, LR 1e-5,
bf16 (1 epoch except NE7 = 3). Losses: `Contrastive` (InfoNCE), STE-sign variant,
`Distillation` (KL on teacher scores).

**Evaluation.** BEIR NFCorpus/ArguAna/SciFact/SciDocs. Exhaustive = GPU brute-force
MaxSim (validated vs NumPy to 3.8e-6), decomposed f32×f32 / int8×int8 / int8×binary /
float×binary / binary×binary, plus per-dim-centered, ITQ, and health scalars (dead%,
isotropy PR/eff-rank, sign_cos). Deployed = next-plaid PLAID. Retention = int8×binary
÷ int8×int8. Dead-bit metric uses `np.sign` (ties→0), the bit stream uses `x>0`
(ties→−1); exact zeros are measure-zero.

**Harness** (`TACET/infra/modal_lateon_binary.py`): `experiment`, `sweep`,
`probe`/`probe_center`, `kd_experiment` + `kd_seed`/`kd_seeds_main`, `geom_probe`/
`geom_main` (dead%/isotropy/magnitude/retention), `itq_probe`/`itq_main`,
`dim_one`/`dim_main`, `kd_teacher`/`kd_teacher_main`. Deployed PLAID:
`modal_plaid_eval.py` + `examples/binary_ndcg.rs`.

**Deployed-PLAID 4-arm binary NDCG@10** (LateOn family; treatment is α=0.4):

| checkpoint | SciFact | NFCorpus | ArguAna | SciDocs |
|---|---:|---:|---:|---:|
| reference (LateOn) | 0.7451 | 0.3633 | 0.2957 | 0.1830 |
| base (LateOn-unsup) | 0.7398 | 0.3570 | 0.2431 | 0.1893 |
| control (+contrastive) | 0.7400 | 0.3719 | 0.2643 | 0.1910 |
| treatment (+STE α=0.4) | 0.7396 | 0.3625 | 0.2820 | 0.1935 |

## 12. NE31 — SciFact × mxbai-edge-colbert-v0-17m: end-to-end pipeline profile (2026-07-07)

First full-pipeline measurement on real BEIR data with a low-dim edge model,
run on the next-plaid index itself (not the python decomposition). Setup:
SciFact (5,183 docs, 1.38M doc tokens, 300 judged queries), embedded once
with `mixedbread-ai/mxbai-edge-colbert-v0-17m` (dim=48) via
`scripts/embed_beir_colbert.py`; three index configs profiled by
`examples/binary_ndcg.rs` on an Apple M4 (native aarch64 + Accelerate).
Build = `create_with_kmeans` wall clock; latency = per-query `search` in the
deployed regime (top_k=10, PLAID defaults), after one warmup query.

| scheme | build (s) | index (MB) | B/token | vs f32 | iso NDCG@10 | dep NDCG@10 | mean ms | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| residual nbits=4 | 29.1 | 99.0 | 24 | 8× | 0.3818 | 0.3819 | 17.7 | 17.6 | 18.9 |
| residual nbits=2 | 30.3 | 65.8 | 12 | 16× | 0.3859 | 0.3859 | 17.6 | 17.6 | 19.0 |
| binary int8×1-bit | 30.7 | 49.2 | 6 | 32× | 0.0308 | 0.0322 | 19.8 | 19.8 | 21.3 |

Independent verification (pure numpy, exhaustive brute force over all docs,
no next-plaid code): float MaxSim NDCG@10 = 0.3675; sign-binarized docs =
0.0233 → **6.3% retention**. The rust pipeline tracks both ends faithfully
(residual ≈ float ceiling at ~104% of exhaustive due to int8-query rounding
noise being negligible; binary matches the numpy collapse), so the failure is
the quantizer on this model, not the search machinery.

Reading through the capacity ledger of §8.1: dim=48 gives each token a
48-bit budget — 2.7× less than the dim=128 models where the synthetic
scaffold and the LateOn fleet sit — and this checkpoint was not trained with
any binarization pressure. The collapse (6% retention vs ~99% synthetic at
dim=128) is the strongest datapoint yet that binarizability is a property of
the checkpoint × dimension, not of the codec: nbits=2 on the same embeddings
retains 101% (0.386 vs 0.382 at nbits=4), so even 2-bit magnitude
information rescues what signs alone cannot carry at this dimension.

Practical takeaways: (1) do not default `binary: true` for dim<64 edge
models without a binarization-aware checkpoint (§11.3's training recipes are
the remedy to test — NE30's STE treatment is the natural candidate);
(2) residual nbits=2 is the sweet spot for this model — 16× compression,
retention ≥ nbits=4, same latency; (3) latency is flat across schemes here
because dim=48 keeps the fused dim=128 kernels out of play and Stage-2 cost
is dominated by the shared candidate pipeline — the binary row pays a small
penalty on the per-pair fallback path, another argument for generalizing the
fused kernels beyond dim=128 if edge models become a target.

Repro: `python scripts/embed_beir_colbert.py --data <scifact> --out <bundle>
--model mixedbread-ai/mxbai-edge-colbert-v0-17m` then
`cargo run --release --features accelerate --example binary_ndcg -- <bundle>`
(add `--target aarch64-apple-darwin` on Apple Silicon with an x86 toolchain).

## 13. NE32 — same pipeline, LateOn-regularized (dim=128): the counterpoint (2026-07-07)

Identical dataset and harness as NE31 (SciFact, 5,183 docs, 300 judged
queries), only the checkpoint changed: `lightonai/LateOn-regularized`
(dim=128, 1.19M doc tokens), embedded via pylate 1.4. Apple M4, native
aarch64 + Accelerate.

| scheme | build (s) | index (MB) | B/token | vs f32 | iso NDCG@10 | dep NDCG@10 | mean ms | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| residual nbits=4 | 45.1 | 184.8 | 64 | 8× | 0.7615 | 0.7615 | 21.1 | 20.7 | 25.2 |
| residual nbits=2 | 50.2 | 108.5 | 32 | 16× | 0.7605 | 0.7605 | 18.8 | 18.2 | 22.2 |
| binary int8×1-bit | 48.2 | 70.4 | 16 | 32× | 0.7513 | 0.7513 | **5.6** | 5.3 | 8.9 |

Independent numpy brute force: float exhaustive 0.7629, sign-binarized docs
0.7512 → **98.5% retention**, matching the rust pipeline (98.7%) to within
rounding — the deployed two-stage keeps 100% of the isolated ceiling for
every scheme on this corpus.

Side-by-side with NE31, this is the thesis in one table: the identical codec
and identical search machinery go from **6% retention / no speedup**
(edge-17m, dim=48, no binarization pressure) to **98.7% retention / 3.8×
end-to-end speedup** (LateOn-regularized, dim=128). At dim=128 the fused
NEON SDOT kernel engages and Stage-2 drops from decompress+GEMM (21.1 ms) to
integer 2P−T on the stored bits (5.6 ms), while the index shrinks 2.6×
against nbits=4. Binarizability is a property of checkpoint × dimension;
given a binarization-aware 128-dim checkpoint, `binary: true` dominates the
residual codec on every axis measured here — quality within 1.3%, 3.8×
faster queries, 2.6× smaller index, same build cost.

## 14. NE33 — base LateOn ablation: what the regularization buys (2026-07-07)

Same harness, `lightonai/LateOn` (no regularization, dim=128):

| scheme | index (MB) | B/token | iso NDCG@10 | dep NDCG@10 | mean ms |
|---|---:|---:|---:|---:|---:|
| residual nbits=4 | 185.0 | 64 | 0.7639 | 0.7639 | 38.3* |
| residual nbits=2 | 108.7 | 32 | 0.7595 | 0.7595 | 19.3 |
| binary int8×1-bit | 70.6 | 16 | 0.7451 | 0.7451 | 5.7 |

\*residual-nbits4 profiled concurrently with the numpy verification job;
its latency is contaminated by core contention — NE32's ~21 ms is the clean
number (the binary and nbits=2 rows ran after contention ended and match
NE32 within noise). Numpy brute force: float 0.7626, binary-doc 0.7449 →
97.7% retention, again matching the rust pipeline (97.5%).

The ablation, LateOn base → LateOn-regularized (identical float ceiling,
0.7626 vs 0.7629):

| | base | regularized | delta |
|---|---:|---:|---:|
| binary NDCG@10 (deployed) | 0.7451 | 0.7513 | +0.62pt |
| retention vs residual4 | 97.5% | 98.7% | +1.2pt |

Two readings. First, the base 128-dim LateOn checkpoint is *already* highly
binarizable — 97.5% retention with no binarization-aware training — so the
NE31 collapse (6%) is attributable to the checkpoint family × dim=48, not to
missing regularization per se; capacity comes first, training pressure
second (consistent with §8.1's two-ledger account). Second, the
regularization closes ~1/3 of the remaining gap to the float ceiling for
free — float NDCG is unchanged (0.7626 → 0.7629) — so it is a strict
improvement for anyone intending to deploy `binary: true`, just not the
difference between working and broken at this dimension.
