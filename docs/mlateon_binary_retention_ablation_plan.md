# mLateOn Asymmetric Binary-Retention Study

**Status:** Experimental plan ready for implementation

**Audience:** An engineer or research agent with no prior context

**Primary hardware:** One NVIDIA RTX 4090 (24 GB)

**Primary training budget:** 100,000 examples per ablation

**Handoff instruction:** Read this document completely before implementing or launching anything. Begin with the Stage 0 correctness gates, then run only the R0 reference. Do not queue the full matrix until R0 passes Gate A.

## 1. Objective

Determine why `multi-vector-encoder/mLateOn-medical` retains almost all of its float retrieval quality when document token embeddings are reduced to one-bit signs, while the closely related `lightonai/mLateOn-unsupervised` and `lightonai/mLateOn` checkpoints lose much more quality.

The immediate scientific questions are:

1. Is the result caused by a real change in representation geometry, or does the MIRIAD evaluation simply make binary errors unusually harmless?
2. Is the change caused by the MIRIAD data, the contrastive objective and temperature, the trained model components, or a PyLate-versus-Sentence-Transformers implementation difference?
3. Can the behavior be reproduced in 100,000 training examples?
4. Can the useful behavior be transferred to other datasets without sacrificing float retrieval quality?
5. If ordinary float training is insufficient, can asymmetric binary-aware training or an orthogonal output rotation produce reliable retention?

The study must distinguish three superficially similar outcomes:

- **Binary-friendly geometry:** float vectors are represented in a coordinate basis whose signs preserve useful dot products and MaxSim decisions.
- **Large retrieval margins:** quantization introduces substantial score error, but the task is easy enough that rankings do not change.
- **Evaluation or scorer artifact:** an implementation, masking, precision, or normalization difference makes one route appear better without changing the underlying model.

## 2. Minimal technical background

mLateOn is a ColBERT-style multi-vector retrieval model. Each query and document becomes a sequence of normalized 128-dimensional token vectors. A query-document score is computed with MaxSim:

```text
score(q, d) = sum_i max_j dot(q_i, d_j)
```

For asymmetric binary retrieval, query vectors remain floating point or are quantized to INT8, while each document coordinate is represented by its sign. The deployment target is therefore approximately:

```text
FP16 or per-token INT8 query  ×  one-bit document
```

This is different from symmetric binary retrieval, where both queries and documents are reduced to signs. Symmetric binary results should be measured as a diagnostic, but they are not the primary endpoint.

The experiment must use the repository's existing asymmetric binary route consistently. Do not silently change sign conventions, zero handling, token normalization, per-token scaling, packing, or score correction between runs. Record those details in every result artifact.

## 3. Model lineage

The relevant model lineage is:

```text
mmBERT-base
  -> PyLate multilingual contrastive pretraining on approximately 2.8B pairs
     -> lightonai/mLateOn-unsupervised
        -> PyLate hard-negative and teacher-distillation training
           -> lightonai/mLateOn
        -> Sentence Transformers v6 training on 1M MIRIAD pairs
           -> multi-vector-encoder/mLateOn-medical
```

Important consequence: `mLateOn-medical` starts from `mLateOn-unsupervised`, not from the finished `mLateOn` checkpoint. The finished general-purpose model and the medical model are separate descendants of the same unsupervised parent.

The parent contains a transformer followed by three residual/projection dense modules:

```text
Transformer
Dense 768 -> 1536, residual
Dense 1536 -> 768, residual
Dense 768 -> 128
```

The Sentence Transformers compatibility loader preserves these weights and adds explicit document masking and token normalization modules. The published medical artifact contains:

```text
Transformer
Dense 768 -> 1536, residual
Dense 1536 -> 768, residual
Dense 768 -> 128
MultiVectorMask, document-side punctuation skiplist
Normalize, token_embeddings
```

## 4. Facts already established

Treat the following as prior results. Do not rerun them unless validating a new implementation.

### 4.1 Published 200k-corpus reproduction

On the MIRIAD evaluation with approximately 1,000 queries and 200,000 documents:

| Route | NDCG@10 |
|---|---:|
| FP16 query x FP16 document | 0.9160437584 |
| FP16 query x binary document | 0.9057900906 |
| Global INT8 query x binary document | 0.9059209228 |
| Per-token INT8 query x binary document | 0.9055010080 |

Raw document storage was approximately 44.92 GB in FP16 and 2.8075 GB in one-bit signs, a 16x reduction. The published quality claim is therefore reproducible.

### 4.2 Same-pipeline controlled evaluation

On the same 20,000-document MIRIAD slice, with tokenization, punctuation masking, document counts, and evaluation code held fixed:

| Model | Float NDCG@10 | FP query x binary document | Retention |
|---|---:|---:|---:|
| `mLateOn-unsupervised` | 0.9407018423 | 0.7250466943 | 77.08% |
| `mLateOn-medical` | 0.9780931473 | 0.9771437049 | 99.90% |

The corresponding binary-query x binary-document scores were approximately 0.4578 and 0.9721. This large difference survives a controlled evaluator and therefore comes from learned weights or their interaction with the dataset, not document counts or punctuation configuration.

### 4.3 Finished mLateOn is not naturally binary-friendly

The finished `lightonai/mLateOn` checkpoint produced approximately:

```text
float NDCG@10:                 0.9564522505
FP query x binary document:   0.7434540391
retention:                    77.73%
```

Ordinary PyLate supervised training with hard negatives and distillation does not automatically produce the medical model's retention.

### 4.4 Binary quality is coordinate-basis sensitive

Prior TACET experiments found:

| Model/transform | SciFact | NFCorpus |
|---|---:|---:|
| Raw mLateOn float | 0.7654 | 0.3813 |
| Raw mLateOn binary | 0.5779 | 0.2267 |
| Householder-rotated binary | 0.7236 | 0.3356 |

Applying the same orthogonal transform to queries and documents preserves float dot products and MaxSim scores, while changing sign quantization behavior. This means binary retention is not solely a semantic-quality property; the coordinate basis matters substantially.

### 4.5 FP32 casting is not the cause

The medical recipe loads FP32 master weights and trains with BF16 autocast. PyLate training does the same in the relevant path. Both Sentence Transformers trainers replace the loss's model reference with Accelerate's prepared model. The wrapped forward pass runs under autocast and recursively returns BF16/FP16 outputs as FP32 to the external loss.

Consequently, in the actual training routes:

| Operation | PyLate | Sentence Transformers v6 |
|---|---|---|
| Transformer/projection forward | BF16 autocast | BF16 autocast |
| Tensor returned to loss | FP32 | FP32 |
| Token normalization | FP32 | FP32 |
| External MaxSim scorer | FP32 | FP32 |
| Cross-entropy | FP32 | FP32 |

Sentence Transformers v6 also calls `.float()` after the document-token maximum and before the query-token sum. That protects direct BF16 inference, but it is a no-op on the already-FP32 training scores. Do not spend a 100k training run on this cast.

### 4.6 One real scoring difference remains

PyLate masks document padding or skipped tokens by multiplying their token similarities by zero before the document-token maximum. Sentence Transformers v6 fills them with the minimum representable value before taking the maximum.

The v6 behavior is mathematically preferable: a masked zero should not defeat a legitimate negative dot product. Existing parity tests pass because real tokens normally dominate. This difference is worth one controlled ablation, but it is not currently considered capable of explaining the entire 77% to 99.9% retention change.

## 5. Published medical training recipe

The reference experiment must reproduce this recipe as closely as possible:

| Setting | Value |
|---|---|
| Initial checkpoint | `lightonai/mLateOn-unsupervised` |
| Training data | `tomaarsen/miriad-4.4M-split` |
| Examples for this study | first 100,000 rows used by the published ordering, saved as a fixed manifest |
| Epochs | 1 |
| Effective batch size | 128 |
| GradCache encoding mini-batch | 16 |
| Loss | `CachedMultiVectorMultipleNegativesRankingLoss` |
| Scale / temperature | scale 1.0 / temperature 1.0 |
| Learning rate | `1e-4` |
| Warmup | 5% |
| Weight decay | 0 |
| Optimizer | fused AdamW when available |
| Master/model load dtype | FP32 |
| Mixed precision | BF16 |
| Query prompt | `[Q] ` |
| Document prompt | `[D] ` |
| Maximum tokenizer length | 8192 |
| Per-task query/document caps | unset |
| Batch sampler | no duplicates |
| Document skiplist | `string.punctuation` |
| Query skiplist | none |
| Trainable parameters | full model |

The original 1M training reportedly took roughly 13-14.5 hours on an RTX 3090. A 100k run should take approximately 45-70 minutes of training on a 4090, or roughly 1-1.5 hours including reduced evaluation and checkpoint overhead.

## 6. Leading hypotheses

Ranked roughly by current plausibility:

### H1: Objective temperature changes the geometry

The medical run uses scale 1, equivalent to temperature 1. The unsupervised PyLate pretraining used temperature 0.02, equivalent to scale 50. This is a very large difference in softmax sharpness. A softer objective may produce broader, more sign-stable token alignments instead of relying on a few precise floating-point coordinate magnitudes.

### H2: MIRIAD's pair construction produces forgiving lexical token alignment

MIRIAD questions are generated from source passages, and the passages average approximately 941 tokens. Query/source pairs can have strong lexical or localized semantic correspondence. Float and binary scores may differ substantially while the positive document remains separated from distractors by a large margin.

### H3: The output projection learns a favorable coordinate basis

Because sign quantization is basis sensitive and the final representation is only 128 dimensions, the three-layer projection stack may rotate or reshape token vectors into a sign-friendly basis during medical training.

### H4: Full-model, relatively high-learning-rate adaptation matters

The medical run updates the entire model at `1e-4`, much larger than typical supervised fine-tuning rates used by the finished mLateOn branch. Large parameter movement could reorganize both contextual token features and the final projection.

### H5: Masking or framework semantics make a smaller contribution

The zero-versus-negative-infinity document mask is a real semantic difference. Other framework-level explanations have largely been ruled out through source review and parity tests.

## 7. Experimental principles

1. Change one named variable at a time unless running an explicitly declared factorial experiment.
2. Use the same fixed training row IDs and fixed evaluation corpora across comparable runs.
3. Record package versions, model revisions, git commit, seed, GPU, CUDA version, and the complete configuration.
4. Save checkpoints by examples processed, not merely epochs, so batch-size experiments remain interpretable.
5. Never judge binary retention without reporting absolute float and binary quality.
6. Use the same quantization and retrieval implementation for every model in a table.
7. Use FP32 scoring as the canonical correctness path. Optimized lower-precision scoring can be benchmarked separately.
8. Do not run all experiments blindly. Execute in waves and use the decision gates below.

## 8. Stage 0: correctness gates before training

These checks are cheaper and more valuable than a failed 100k run.

### G0.1: Frozen PyLate/MVE parity

Using the same parent checkpoint and the same small batch, compare:

- exact input IDs and attention masks;
- placement and token identity of `[Q]` and `[D]`;
- document skiplist mask;
- token embeddings before and after normalization;
- MaxSim score matrix;
- cross-entropy loss;
- gradient norm and gradient cosine similarity by module.

Run once with v6 masking and once with PyLate zero masking. The pure-framework comparison should use the same masking implementation.

### G0.2: Dtype trace

Install temporary hooks or logging to record the dtype at:

- transformer output;
- every projection output;
- normalized token embeddings;
- MaxSim dot-product tensor;
- MaxSim maximum;
- final score matrix;
- loss.

Expected result: model internals use BF16 where autocast permits it, while the tensors passed to and produced by the external scoring loss are FP32.

### G0.3: GradCache replay check

Run the same first update using encoding mini-batches 8, 16, and 32. Compare loss and parameter delta. Chunk size should only affect memory and runtime. If results differ materially, fix replay determinism before training.

### G0.4: Basis-sensitivity distribution

For `mLateOn-unsupervised`, `mLateOn`, and `mLateOn-medical`:

1. Evaluate the identity basis.
2. Generate at least 20 seeded random orthogonal 128x128 transforms.
3. Apply each transform to both query and document token vectors.
4. Confirm float scores are unchanged within tolerance.
5. Measure asymmetric binary NDCG for each transform.
6. Include the existing optimized Householder transform if available.

This distinguishes a generally binary-friendly representation from a lucky identity basis. It requires no retraining.

## 9. Reference 100k run and checkpoint gate

Run the exact reference configuration as `R0`. Save at:

| Examples seen | Approximate step at batch 128 |
|---:|---:|
| 0 | 0 |
| 10,000 | 78 |
| 25,000 | 195 |
| 50,000 | 391 |
| 100,000 | 782 |

Use a smaller fixed evaluation corpus at intermediate checkpoints and the full 200k MIRIAD corpus only at the final checkpoint.

The reference subset must be frozen before training. Save its dataset revision, split, row indices, stable IDs when available, and content hashes. If an existing reproduction script used a different 100k selection, preserve that selection and document the difference rather than silently replacing it.

### Gate A

Proceed with the full 100k ablation matrix only if R0 produces a clear movement in asymmetric binary retention or the geometry diagnostics. If 100k shows little change, continue R0 to 250k before interpreting any null result. The published behavior may require more examples than the proposed screening budget.

## 10. Wave 1: causal ablations

All runs start from the same parent revision and use the same 100k row IDs unless the table explicitly says otherwise.

| ID | Single change from R0 | Question answered | Priority |
|---|---|---|---|
| R0 | None | Reference trajectory | Required |
| R1 | PyLate training loop with shared v6 scorer/mask and otherwise identical settings | Is there a pure framework effect? | Required |
| R2 | PyLate zero-mask semantics inside MVE | Does scorer masking alter learning? | High |
| R3 | Scale 10 instead of 1 | Is retention temperature-sensitive? | High |
| R4 | Scale 50 / temperature 0.02 | Does parent-like sharpness prevent retention? | Required |
| R5 | Learning rate `3e-5` | Does update magnitude drive the change? | High |
| R6 | Start from finished `lightonai/mLateOn` | Does supervised lineage prevent adaptation? | High |
| R7 | Freeze transformer; train all three dense modules | Is the projection stack sufficient? | Required |
| R8 | Freeze all dense modules; train transformer | Is the backbone sufficient? | Required |
| R9 | Remove document punctuation skiplist | Does punctuation masking affect geometry? | Medium |

Run these once with a common seed. Replicate only R0 and the most informative variants after screening.

### Wave 1 interpretation

- `R0 ~= R1`: clear the framework implementation as a cause.
- `R1` differs but `R2` matches PyLate: masking is responsible for the framework difference.
- `R4` loses retention while float quality remains good: temperature is a leading causal factor.
- `R7` reproduces retention: the projection basis is the main learned mechanism.
- `R8` reproduces retention: the backbone's contextual token geometry is the main mechanism.
- Neither R7 nor R8 reproduces retention: coordinated backbone-plus-projection adaptation is required.
- R6 adapts poorly: the finished checkpoint occupies a less plastic or less suitable basin, consistent with the published 25k float-quality ablation.

## 11. Wave 2: data x temperature factorial

Construct three 100k datasets:

1. High lexical-overlap MIRIAD query/passage pairs.
2. Low lexical-overlap MIRIAD query/passage pairs.
3. General retrieval pairs sampled from the parent's training distribution.

Match language, query-token length, document-token length, and sample count as closely as practical. Define lexical overlap before looking at binary results. A simple baseline is query-token set overlap after normalization and stopword handling, but retain the raw measure and the exact code.

Train each dataset at scale 1 and scale 50:

| ID | Data | Scale |
|---|---|---:|
| D1 | High-overlap MIRIAD | 1 |
| D2 | High-overlap MIRIAD | 50 |
| D3 | Low-overlap MIRIAD | 1 |
| D4 | Low-overlap MIRIAD | 50 |
| D5 | Length/language-matched general pairs | 1 |
| D6 | Length/language-matched general pairs | 50 |

This factorial is important because data and temperature may interact. One-factor-at-a-time runs would miss a result such as scale 1 helping only on source-passage question pairs.

### Wave 2 interpretation

- Only high-overlap MIRIAD works: the result is primarily a data/task-margin phenomenon.
- All three datasets work at scale 1 and fail at scale 50: the objective scale is central.
- MIRIAD works at both scales but general pairs do not: domain or pair construction dominates.
- Retention improves on out-of-domain evaluation as well: training changed global representation geometry.
- Retention improves only on MIRIAD: training probably enlarged in-domain margins rather than producing a universally binary-friendly model.

## 12. Wave 3: negative and distillation structure

These experiments explain why the finished mLateOn training branch does not retain binary quality.

| ID | Change from R0 |
|---|---|
| O1 | Add one mined hard negative per query |
| O2 | Add seven mined hard negatives per query |
| O3 | Add hard negatives plus teacher-score KL distillation |
| O4 | Effective batch 32 instead of 128 |
| O5 | Effective batch 512 instead of 128, if memory permits |

Run these at scale 1 first. Do not combine scale 50, hard negatives, distillation, and a different batch in one run; that would recreate a broad recipe difference without identifying its cause.

## 13. Wave 4: binary-aware engineering

Start this wave only after the ordinary-training mechanism is reasonably understood. The deployment objective is asymmetric, so the quantized loss must keep queries floating point or fake-quantized to the intended INT8 route while binarizing documents.

Let:

```text
L_float = contrastive loss using float query and float document tokens
L_bin   = contrastive loss using float query and STE-sign document tokens
```

Recommended runs:

| ID | Objective/change |
|---|---|
| Q1 | `L_float + 0.25 * L_bin` |
| Q2 | `L_float + 1.0 * L_bin` |
| Q3 | `L_bin` only |
| Q4 | `L_float` plus KL consistency between float and quantized score matrices |
| Q5 | STE-sign documents, FP16 queries |
| Q6 | STE-sign documents, fake per-token INT8 queries |
| Q7 | Freeze base model; learn only an orthogonal output rotation against the asymmetric binary objective |

Q7 is particularly attractive. Applying an orthogonal transform to both sides preserves float dot products exactly, so it offers a constrained way to improve binary behavior without damaging the float model. Parameterize it in a way that remains orthogonal, such as products of Householder reflections; do not use an unconstrained dense matrix and call it a rotation.

Generic sign-balance regularization is lower priority. Prior TACET evidence shows that basis repair helps, but it does not establish that marginal sign balance alone is sufficient.

## 14. Evaluation datasets

Every final checkpoint should be evaluated on:

1. The held-out MIRIAD benchmark with approximately 1,000 queries and 200,000 passages.
2. The same MIRIAD benchmark divided into lexical-overlap, document-length, and float-margin difficulty buckets.
3. SciFact.
4. NFCorpus.
5. At least one non-medical general retrieval dataset with enough candidates to expose ranking errors.

For fast checkpoint evaluation, use a fixed MIRIAD probe such as 1,000 queries against 50,000 documents. Do not change the probe between runs. The final 100k checkpoint receives the full 200k evaluation.

Ensure the training and evaluation passages are disjoint according to the dataset's stable IDs or content hashes. Record the deduplication policy.

## 15. Required metrics

### 15.1 Retrieval metrics

For every dataset and checkpoint report:

- float-query x float-document NDCG@10;
- float-query x binary-document NDCG@10;
- per-token-INT8-query x binary-document NDCG@10;
- binary-query x binary-document NDCG@10 as a diagnostic;
- absolute binary drop;
- retention ratio;
- Recall@10 and MRR@10 where labels support them;
- float/binary top-10 intersection;
- paired bootstrap 95% confidence intervals over queries for final comparisons.

Define the primary retention values explicitly:

```text
absolute_drop = NDCG_binary - NDCG_float
retention     = NDCG_binary / NDCG_float
```

Never present retention without both underlying NDCG values. A weak float model can have deceptively high retention.

### 15.2 Score and MaxSim diagnostics

On a fixed probe set report:

- Pearson and Spearman correlation between float and binary document scores;
- absolute score-error distribution;
- positive-versus-hardest-negative margin before and after quantization;
- fraction of margins whose sign flips;
- fraction of query tokens whose MaxSim-winning document token changes after binarization;
- score-error and winner-stability results by lexical-overlap and document-length bucket.

These measurements determine whether high NDCG retention comes from accurate binary scores or merely large task margins.

### 15.3 Representation diagnostics

For normalized document token vectors report:

- cosine similarity between each float token and its appropriately scaled sign vector;
- relative dot-product error against a fixed query-token probe;
- fraction of coordinates near zero at several declared thresholds;
- mean absolute coordinate magnitude;
- coefficient of variation of coordinate magnitudes;
- per-dimension positive-sign frequency and entropy;
- covariance spectrum or anisotropy summary;
- binary retention across random orthogonal bases;
- parameter displacement from the initial checkpoint by model module;
- gradient norm by transformer and each dense module during training.

## 16. Reproducibility and artifact layout

Each run should produce a self-contained directory resembling:

```text
experiments/mlateon_binary_100k/<run_id>/
  config.json
  environment.json
  data_manifest.json
  train_metrics.jsonl
  checkpoints/
  eval/
    miriad_50k.json
    miriad_200k.json
    scifact.json
    nfcorpus.json
  geometry/
    token_statistics.json
    score_error.json
    rotation_sweep.json
  summary.md
```

At minimum, `config.json` must include:

- run ID and parent run ID;
- the single intended change;
- seed;
- exact model name and revision;
- exact dataset name, revision, row IDs, and content hash;
- all optimizer, scheduler, loss, batch, length, prompt, masking, and dtype settings;
- quantization-route configuration;
- code git commit and dirty-worktree state;
- package versions;
- GPU model, driver, CUDA, and PyTorch versions.

Use seed 42 for screening unless an existing reproduction requires another seed. Replicate R0 and the top three scientifically informative or best-performing variants with three total seeds.

## 17. Runtime plan on an RTX 4090

Expected 100k training time is approximately 45-70 minutes. Budget 1-1.5 hours per run with reduced evaluation.

| Wave | Runs | Estimated training | Estimated with evaluation |
|---|---:|---:|---:|
| Correctness gates | no full runs | less than one run | workload-dependent |
| Wave 1 | 10 | 8-12 h | 10-15 h |
| Wave 2 | 6 | 5-7 h | 6-9 h |
| Wave 3 | 5 | 4-6 h | 5-8 h |
| Wave 4 | 7 | 5-8 h | 7-11 h |

Do not schedule all 28 training runs at once. Recommended order:

1. Complete correctness gates.
2. Run R0 and validate Gate A.
3. Run R1, R2, R4, R7, and R8 first because they offer the highest causal value.
4. Complete the remaining Wave 1 screen.
5. Choose Wave 2 sampling thresholds without looking at binary outcomes.
6. Run the Wave 2 factorial.
7. Run Wave 3 only if hard-negative/distillation structure remains unresolved.
8. Run Wave 4 after selecting the best ordinary-training reference.
9. Replicate R0 and the top three variants with two additional seeds each.

Benchmark the first 50-100 optimizer steps of R0 and update the ETA from measured examples per second. Monitor thermal throttling and host-side tokenization stalls. Pretokenization is recommended.

## 18. Decision rules

Use these rules when writing the conclusion:

| Observation | Conclusion supported |
|---|---|
| R0 and pure-framework R1 agree within uncertainty | Framework choice is not causal |
| R2 explains an R0/R1 gap | Mask semantics cause the framework difference |
| Scale 50 reduces binary retention without reducing float quality | Soft objective scale is a major geometry driver |
| Projection-only R7 reproduces the effect | Output basis/head is sufficient |
| Backbone-only R8 reproduces the effect | Contextual token geometry is sufficient |
| Only high-overlap MIRIAD retains quality | Pair construction or task margin dominates |
| Retention rises on MIRIAD and out-of-domain datasets | Global geometry changed |
| Identity basis is exceptional among random rotations | The checkpoint is favorably oriented rather than universally sign-friendly |
| Most rotations retain quality | The representation is broadly quantization-friendly |
| Q7 succeeds while float metrics remain invariant | Orthogonal basis optimization is the safest engineering route |
| Binary NDCG is stable but score errors and winner changes are large | Evaluation margins hide quantization error |

Avoid claiming a cause from retention alone. A causal conclusion should combine retrieval metrics, score/margin diagnostics, representation statistics, and the relevant controlled ablation.

For screening, treat an improvement as practically interesting when it materially reduces the absolute binary NDCG loss without obtaining the result by lowering float quality. For the final engineering model, the provisional target is a MIRIAD asymmetric-binary drop no larger than 0.01 NDCG@10, a float-quality regression no larger than 0.005, and evidence that the gain survives at least two out-of-domain evaluations. Use confidence intervals and three-seed results for the final judgment rather than treating these thresholds as statistical tests.

## 19. Definition of done

The study is complete when it delivers:

1. A verified R0 100k trajectory and a determination of whether 100k is an adequate screening budget.
2. A pure PyLate-versus-MVE comparison with matched tokens, masks, scorer, data, and hyperparameters.
3. Temperature, projection-only, and backbone-only ablations.
4. A controlled data x temperature result.
5. In-domain and out-of-domain float and asymmetric-binary evaluation.
6. Score-margin, MaxSim-winner, geometry, and random-rotation diagnostics.
7. Three-seed confirmation for R0 and the three most informative variants.
8. A final conclusion separating framework effects, learned geometry, basis orientation, and dataset difficulty.
9. If ordinary training does not generalize, at least one asymmetric binary-aware or orthogonal-rotation experiment.
10. A concise table suitable for the technical blog that states exactly what was held fixed and avoids implying unsupported causality.

## 20. Known documentation issue

The published Sentence Transformers blog snippet refers to `model[2]` as the `MultiVectorMask` module when modifying its skiplist. In the released mLateOn-compatible architecture, modules 1, 2, and 3 are dense layers and the mask is module 4. The final artifact contains the correct punctuation mask, so this appears to be a documentation index error rather than a bad released model.

Do not hard-code a module number. Locate the `MultiVectorMask` by type.

## 21. Primary references

- Medical training recipe: <https://huggingface.co/blog/train-multi-vector-encoder>
- Medical model: <https://huggingface.co/multi-vector-encoder/mLateOn-medical>
- Unsupervised parent: <https://huggingface.co/lightonai/mLateOn-unsupervised>
- Finished mLateOn: <https://huggingface.co/lightonai/mLateOn>
- LightOn training repository: <https://github.com/lightonai/mdenseon-mlateon>
- PyLate 1.3.4 model implementation: <https://github.com/lightonai/pylate/blob/1.3.4/pylate/models/colbert.py>
- PyLate cached contrastive loss: <https://github.com/lightonai/pylate/blob/1.3.4/pylate/losses/cached_contrastive.py>
- PyLate MaxSim scorer: <https://github.com/lightonai/pylate/blob/1.3.4/pylate/scores/scores.py>
- Sentence Transformers v6 compatibility loader: <https://github.com/huggingface/sentence-transformers/blob/v6.0.0/sentence_transformers/multi_vector_encoder/model.py>
- Sentence Transformers v6 MaxSim implementation: <https://github.com/huggingface/sentence-transformers/blob/v6.0.0/sentence_transformers/util/similarity.py>
- Sentence Transformers v6 cached multi-vector loss: <https://github.com/huggingface/sentence-transformers/blob/v6.0.0/sentence_transformers/multi_vector_encoder/losses/cached_multiple_negatives_ranking.py>
- PyTorch automatic mixed precision documentation: <https://docs.pytorch.org/docs/stable/amp.html>
