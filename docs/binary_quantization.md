# Asymmetric binary quantization

NextPlaid can store documents as **1-bit signs** instead of residual (`nbits`)
codes, and score them with an **asymmetric MaxSim**: the query stays at higher
precision (int8) while each document dimension collapses to a single sign bit.
This is the multi-vector analogue of mixedbread's
[asymmetric quantization](https://www.mixedbread.com/blog/asymmetric-quant)
"Int8 × Binary" scheme.

Unlike the residual codec — which reconstructs `centroid + bucket_weight` back to
`f32` before scoring — binary documents are **never decompressed to a learned
float**. Scoring uses the exact identity that a dot product against a sign vector
`s ∈ {−1,+1}^d` is `q · s = Σ_d q_d · s_d`, so document precision drops to one bit
per dimension while ranking stays close to full precision.

## Usage

```rust
use next_plaid::{IndexConfig, MmapIndex};

let config = IndexConfig { binary: true, ..Default::default() };
let index = MmapIndex::create_with_kmeans(&doc_embeddings, path, &config)?;
let results = index.search(&query, &params, None)?; // asymmetric int8 × 1-bit
```

The centroid/IVF first stage is unchanged — only the document store and the
Stage-2 exact rescore differ. Incremental `update()` of a binary index is not
supported yet (rebuild instead); `delete()` works.

## What it costs and saves

Document bytes/token drop from `dim * nbits / 8` to `ceil(dim / 8)` — a 1-bit
store is `nbits×` smaller than the residual store and **32× smaller than raw
`f32`**.

## Reproduction (SciFact, ColBERT `answerai-colbert-small-v1`, dim 96)

```bash
python scripts/embed_beir_colbert.py --data <beir_dir>/scifact --out /tmp/scifact_colbert
cargo run --release --example binary_ndcg -- /tmp/scifact_colbert
```

| scheme                 | NDCG@10 | doc bytes/tok | vs f32 |
|------------------------|---------|---------------|--------|
| residual (nbits=4)     | 0.7354  | 48            | 8×     |
| binary (int8 × 1-bit)  | 0.7017  | 12            | 32×    |

**Binary retains 95.4% of residual NDCG@10 at 4× smaller documents (32× vs
`f32`)** — matching the blog's finding that document vectors tolerate 1-bit
precision when the query keeps higher precision.

## Implementation

- [`next-plaid/src/binary.rs`](../next-plaid/src/binary.rs) — `binarize`,
  `signs_pm1`, `maxsim_binary`, `quantize_query_int8` (unit-tested).
- `IndexConfig::binary` / `Metadata::binary` — the on-disk flag; create paths
  pack signs into the document byte store.
- `search::exact_doc_score` — Stage-2 branch to asymmetric binary scoring.
