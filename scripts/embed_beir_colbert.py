#!/usr/bin/env python
"""Embed a BEIR-layout dataset with a ColBERT model into a compact, Rust-readable
bundle for the `binary_ndcg` example.

Outputs (in --out):
  corpus.npy      [total_doc_tokens, dim]  f32   (concatenated token vectors)
  corpus_lens.npy [num_docs]               i64   (tokens per document)
  corpus_ids.json [num_docs]               str   (BEIR corpus ids, row order)
  queries.npy     [total_query_tokens, dim]f32
  query_lens.npy  [num_queries]            i64
  query_ids.json  [num_queries]            str
  qrels.json      {query_id: {doc_id: score}}

Ragged per-item arrays are recovered in Rust by walking the *_lens files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pylate import models


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(path):
    ids, texts = [], []
    for row in read_jsonl(path):
        ids.append(str(row["_id"]))
        title, text = row.get("title", ""), row.get("text", "")
        texts.append((title + " " + text).strip())
    return ids, texts


def load_queries(path):
    ids, texts = [], []
    for row in read_jsonl(path):
        ids.append(str(row["_id"]))
        texts.append(row["text"])
    return ids, texts


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def pack(embeddings, dim):
    """List of [n_i, dim] -> (concat [sum n_i, dim] f32, lens [N] i64)."""
    lens = np.array([e.shape[0] for e in embeddings], dtype=np.int64)
    concat = np.concatenate(embeddings, axis=0).astype(np.float32)
    assert concat.shape[1] == dim
    return concat, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BEIR dataset dir")
    ap.add_argument("--out", required=True, help="output bundle dir")
    ap.add_argument("--model", default="answerdotai/answerai-colbert-small-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(data / "qrels" / f"{args.split}.tsv")
    eval_qids = set(qrels)

    corpus_ids, corpus_texts = load_corpus(data / "corpus.jsonl")
    all_qids, all_qtexts = load_queries(data / "queries.jsonl")
    # Keep only queries that have judgments in this split.
    query_ids, query_texts = zip(
        *[(q, t) for q, t in zip(all_qids, all_qtexts) if q in eval_qids]
    )
    query_ids, query_texts = list(query_ids), list(query_texts)

    print(f"docs={len(corpus_ids)} queries={len(query_ids)} model={args.model}")
    model = models.ColBERT(model_name_or_path=args.model)

    doc_emb = model.encode(
        corpus_texts, is_query=False, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )
    q_emb = model.encode(
        query_texts, is_query=True, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )
    dim = int(doc_emb[0].shape[1])

    corpus, corpus_lens = pack(doc_emb, dim)
    queries, query_lens = pack(q_emb, dim)

    np.save(out / "corpus.npy", corpus)
    np.save(out / "corpus_lens.npy", corpus_lens)
    np.save(out / "queries.npy", queries)
    np.save(out / "query_lens.npy", query_lens)
    (out / "corpus_ids.json").write_text(json.dumps(corpus_ids))
    (out / "query_ids.json").write_text(json.dumps(query_ids))
    (out / "qrels.json").write_text(json.dumps(qrels))
    print(f"wrote bundle to {out} (dim={dim})")


if __name__ == "__main__":
    main()
