"""Modal app: quant-grid — 3 checkpoints x 4 BEIR datasets x 4 codecs, plus a
corpus-size scale strip, for the binary-vs-residual robustness claims.

Design (encode-once):
  * embed      (GPU, once per model x dataset) -> bundle on the tacet-bundles
               volume under /bundles/quant_grid/{dataset}_{tag}/
  * ceiling    (CPU, once per bundle) -> full per-query x per-doc f32 MaxSim
               score matrix (f32_scores.npy). The exact-f32 ceiling for ANY
               corpus subset is then free post-processing (subset columns).
  * eval_cell  (CPU, per grid/scale cell) -> forms the doc subset in-process
               (all qrels-judged docs pinned, nested seed-permuted distractor
               prefix; embeddings are never re-encoded), runs the Rust
               binary_ndcg harness (residual-4/2/1 + binary-int8x1bit), and
               writes a result JSON next to the bundles.

Provenance rules (binary-quantization report SS8.1/SS8.5 postmortems):
  * pylate is version-pinned; pylate/torch/transformers versions are logged
    into every bundle's meta.json (the pylate<=1.3.2 multi-Dense loader bug
    silently randomizes the mxbai heads).
  * eval_cell logs machine arch + the harness SHA; the Rust build is native
    x86_64 in-image (no cross-arch trap possible).
  * NDCG here uses the harness's exact convention: gain 2^rel - 1, discount
    1/log2(i+2), k=10.

Usage (from the exp/quant-grid worktree root):
  modal run --detach scripts/modal_quant_grid.py::embed_all
  modal run scripts/modal_quant_grid.py::ceiling_all
  modal run --detach scripts/modal_quant_grid.py::grid
  modal run --detach scripts/modal_quant_grid.py::scale_strip
  modal run scripts/modal_quant_grid.py::report
"""

import json
import os
import subprocess
import time

import modal

_T0 = time.time()


def _log(msg):
    print(f"[+{time.time() - _T0:7.1f}s] {msg}", flush=True)


app = modal.App("quant-grid")

CACHE_DIR = "/cache"
BUNDLE_DIR = "/bundles"
ROOT = f"{BUNDLE_DIR}/quant_grid"
hf_cache = modal.Volume.from_name("tacet-hf-cache", create_if_missing=True)
bundles_vol = modal.Volume.from_name("tacet-bundles", create_if_missing=True)
HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# Phenotype spread per the report's recovery table: near-ceiling / saturated
# 128d / saturated low-dim.
MODELS = {
    "lateon_reg": "lightonai/LateOn-regularized",
    "gte": "lightonai/GTE-ModernColBERT-v1",
    "edge17m": "mixedbread-ai/mxbai-edge-colbert-v0-17m",
}
DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
SCALE_DATASET = "fiqa"  # 57.6K docs; the only corpus big enough for a curve
SCALE_SIZES = [4000, 7000, 14000, 28000, 0]  # 0 = full corpus

# Encoder-precision strip: mixedbread ships end-to-end ONNX exports of the 17m
# checkpoint (per onnx_config.json: prefixes, punctuation skiplist, no query
# expansion, dim=48, projection head in-graph). Both precisions go through the
# SAME embed_onnx code path, so fp32-vs-int8 encoder is the only variable —
# deliberately NOT in MODELS so the pylate entrypoints don't pick them up.
ONNX_TAGS = {
    # tag: (model_id, onnx_file, requantize_in_container)
    "edge17m_onnxf32": ("mixedbread-ai/mxbai-edge-colbert-v0-17m", "model.onnx", False),
    # Vendor int8 export: measured anisotropy collapse (tokens vs global mean
    # dir cos 0.9995 vs fp32's 0.9646; NDCG 0.001). Kept as the artifact row.
    "edge17m_q8": ("mixedbread-ai/mxbai-edge-colbert-v0-17m", "model_int8.onnx", False),
    # Our own per-channel dynamic quant of model.onnx: discriminates broken
    # export from inherent int8 fragility of this checkpoint.
    "edge17m_q8pc": ("mixedbread-ai/mxbai-edge-colbert-v0-17m", "model.onnx", True),
}

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pylate==1.6.0",  # >=1.6 required: multi-Dense loader bug in <=1.3.2
        "sentence-transformers>=3.4",
        "transformers>=4.48",
        "torch>=2.2",
        "numpy>=1.24",
        "hf_transfer>=0.1.6",
    )
    .env({
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
    })
)

onnx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime==1.20.1",
        "onnx>=1.16",  # required by onnxruntime.quantization
        "transformers>=4.48",
        "numpy>=1.24",
        "huggingface_hub>=0.23",
        "hf_transfer>=0.1.6",
    )
    .env({
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
    })
)

# Rust toolchain + the exp/quant-grid harness, built natively in-image. Source
# layers last so edits rebuild only the cargo step.
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
        " | sh -s -- -y --profile minimal --default-toolchain stable"
    )
    .pip_install("numpy>=1.24")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "/src",
        copy=True,
        ignore=["target", "**/target", ".git", "**/*.npy", "docs"],
    )
    .run_commands(
        "cd /src && /root/.cargo/bin/cargo build --release -p next-plaid"
        " --example binary_ndcg"
        " && test -x /src/target/release/examples/binary_ndcg"
    )
)


def _load_beir(name):
    """BEIR zip -> (corpus_ids, corpus_texts, query_ids, query_texts, qrels).

    test split; judged-only queries; title+' '+text corpus convention (matches
    scripts/embed_beir_colbert.py and every report bundle).
    """
    import io
    import urllib.request
    import zipfile

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    zpath = f"{CACHE_DIR}/beir_zips/{name}.zip"
    if os.path.exists(zpath):
        buf = io.BytesIO(open(zpath, "rb").read())
        _log(f"_load_beir[{name}]: volume-cached zip ({buf.getbuffer().nbytes / 1e6:.1f} MB)")
    else:
        _log(f"_load_beir[{name}]: downloading {url}")
        data = urllib.request.urlopen(url, timeout=300).read()
        os.makedirs(f"{CACHE_DIR}/beir_zips", exist_ok=True)
        with open(zpath, "wb") as f:
            f.write(data)
        hf_cache.commit()
        buf = io.BytesIO(data)
    zf = zipfile.ZipFile(buf)

    def jsonl(path):
        with zf.open(f"{name}/{path}") as f:
            for line in io.TextIOWrapper(f, "utf-8"):
                if line.strip():
                    yield json.loads(line)

    corpus_ids, corpus_texts = [], []
    for row in jsonl("corpus.jsonl"):
        corpus_ids.append(str(row["_id"]))
        corpus_texts.append((row.get("title", "") + " " + row.get("text", "")).strip())

    qtext = {str(r["_id"]): r["text"] for r in jsonl("queries.jsonl")}
    qrels = {}
    with zf.open(f"{name}/qrels/test.tsv") as f:
        it = io.TextIOWrapper(f, "utf-8")
        next(it)
        for line in it:
            q, d, s = line.split()
            qrels.setdefault(q, {})[d] = int(s)

    query_ids = [q for q in qtext if q in qrels]
    query_texts = [qtext[q] for q in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts, qrels


@app.function(
    image=embed_image, gpu="a10g", memory=49152, timeout=7200,
    secrets=[HF_SECRET], volumes={CACHE_DIR: hf_cache, BUNDLE_DIR: bundles_vol},
)
def embed(tag: str, dataset: str, force: bool = False):
    """Encode one (model, dataset) into a Rust-readable bundle. Idempotent."""
    import numpy as np

    out = f"{ROOT}/{dataset}_{tag}"
    if os.path.exists(f"{out}/meta.json") and not force:
        _log(f"embed[{dataset}/{tag}]: bundle exists, skipping")
        return {"bundle": out, "skipped": True}

    import pylate
    import torch
    import transformers
    from pylate import models

    corpus_ids, corpus_texts, query_ids, query_texts, qrels = _load_beir(dataset)
    _log(f"embed[{dataset}/{tag}]: docs={len(corpus_ids)} queries={len(query_ids)} "
         f"model={MODELS[tag]} pylate={pylate.__version__}")

    model = models.ColBERT(model_name_or_path=MODELS[tag])
    doc_emb = model.encode(corpus_texts, is_query=False, batch_size=64,
                           show_progress_bar=True, convert_to_numpy=True)
    q_emb = model.encode(query_texts, is_query=True, batch_size=64,
                         show_progress_bar=True, convert_to_numpy=True)
    dim = int(doc_emb[0].shape[1])

    def pack(embs):
        lens = np.array([e.shape[0] for e in embs], dtype=np.int64)
        return np.concatenate(embs, axis=0).astype(np.float32), lens

    corpus, corpus_lens = pack(doc_emb)
    queries, query_lens = pack(q_emb)

    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/corpus.npy", corpus)
    np.save(f"{out}/corpus_lens.npy", corpus_lens)
    np.save(f"{out}/queries.npy", queries)
    np.save(f"{out}/query_lens.npy", query_lens)
    for fname, obj in [("corpus_ids.json", corpus_ids), ("query_ids.json", query_ids),
                       ("qrels.json", qrels)]:
        with open(f"{out}/{fname}", "w") as f:
            json.dump(obj, f)
    meta = {
        "model_id": MODELS[tag], "tag": tag, "dataset": dataset, "dim": dim,
        "docs": len(corpus_ids), "queries": len(query_ids),
        "doc_tokens": int(corpus_lens.sum()), "query_tokens": int(query_lens.sum()),
        "pylate": pylate.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    bundles_vol.commit()
    _log(f"embed[{dataset}/{tag}]: wrote {out} dim={dim} tokens={meta['doc_tokens']}")
    return {"bundle": out, "meta": meta}


@app.function(
    image=onnx_image, cpu=16, memory=32768, timeout=14400,
    secrets=[HF_SECRET], volumes={CACHE_DIR: hf_cache, BUNDLE_DIR: bundles_vol},
)
def embed_onnx(tag: str, dataset: str, force: bool = False):
    """Encode with mixedbread's end-to-end ONNX export (fp32 or int8 weights).

    Same bundle format as embed(). Preprocessing follows onnx_config.json:
    string prefixes '[Q] '/'[D] ', doc-side punctuation skiplist, no query
    expansion. Embeddings are L2-normalized post-hoc; pre-normalization token
    norm stats go into meta.json (encoder-precision norm-drift evidence — if
    the graph already normalizes, the stats read ~1.0 and the renorm is a
    no-op)."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    out = f"{ROOT}/{dataset}_{tag}"
    if os.path.exists(f"{out}/meta.json") and not force:
        _log(f"embed_onnx[{dataset}/{tag}]: bundle exists, skipping")
        return {"bundle": out, "skipped": True}

    model_id, onnx_file, requantize = ONNX_TAGS[tag]
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = _load_beir(dataset)
    cfg = json.load(open(hf_hub_download(model_id, "onnx_config.json")))
    mpath = hf_hub_download(model_id, onnx_file)
    if requantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qpath = "/tmp/model_q8pc.onnx"
        quantize_dynamic(mpath, qpath, per_channel=True,
                         weight_type=QuantType.QInt8)
        _log(f"embed_onnx[{dataset}/{tag}]: per-channel dynamic quant "
             f"{os.path.getsize(mpath) / 1e6:.1f} -> "
             f"{os.path.getsize(qpath) / 1e6:.1f} MB")
        mpath = qpath
    tok = AutoTokenizer.from_pretrained(model_id)
    if requantize or onnx_file != "model.onnx":
        # ORT 1.20.1 CPU bug (bisected via trajectory_probe): a dynamic-quant
        # int8 session created as the FIRST session in the process emits
        # collapsed embeddings (tokens ~cos 0.9995 to the mean direction,
        # retrieval NDCG ~0) while norms stay unit. Any fp32 session run
        # first flips the same int8 session healthy. Warm one before
        # creating a quantized session.
        f = ort.InferenceSession(hf_hub_download(model_id, "model.onnx"),
                                 providers=["CPUExecutionProvider"])
        warm = tok(["[D] warmup"], return_tensors="np")
        f.run(None, {i.name: warm[i.name].astype(np.int64)
                     for i in f.get_inputs() if i.name in warm})
        del f
        _log(f"embed_onnx[{dataset}/{tag}]: fp32 warm session run "
             f"(ORT int8-first-session workaround)")

    opts = ort.SessionOptions()
    sess = ort.InferenceSession(mpath, opts, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    _log(f"embed_onnx[{dataset}/{tag}]: {onnx_file} inputs={in_names} "
         f"docs={len(corpus_ids)} queries={len(query_ids)}")

    skip_ids = {
        i for w in cfg["skiplist_words"]
        if (i := tok.convert_tokens_to_ids(w)) not in (None, tok.unk_token_id)
    }

    def encode(texts, prefix, maxlen, filter_skiplist, label):
        res = [None] * len(texts)
        norms = []
        order = np.argsort([-len(t) for t in texts])  # length-bucketed batches
        B = 32
        for s in range(0, len(order), B):
            bidx = order[s:s + B]
            enc = tok([prefix + texts[i] for i in bidx], padding=True,
                      truncation=True, max_length=maxlen, return_tensors="np")
            feed = {n: enc[n].astype(np.int64) for n in in_names if n in enc}
            emb = sess.run(None, feed)[0]  # [b, seq, dim]
            for r, i in enumerate(bidx):
                keep = enc["attention_mask"][r].astype(bool)
                if filter_skiplist:
                    keep &= ~np.isin(enc["input_ids"][r], list(skip_ids))
                e = emb[r][keep].astype(np.float32)
                n = np.linalg.norm(e, axis=1)
                norms.append(n)
                res[i] = e / np.maximum(n, 1e-12)[:, None]
            if (s // B) % 20 == 0:
                _log(f"embed_onnx[{dataset}/{tag}]: {label} {s + len(bidx)}/{len(texts)}")
        alln = np.concatenate(norms)
        stats = {"mean": float(alln.mean()), "std": float(alln.std()),
                 "min": float(alln.min()), "max": float(alln.max())}
        _log(f"embed_onnx[{dataset}/{tag}]: {label} pre-norm token norms {stats}")
        return res, stats

    doc_emb, doc_norms = encode(corpus_texts, cfg["document_prefix"],
                                cfg["document_length"], True, "docs")
    q_emb, q_norms = encode(query_texts, cfg["query_prefix"],
                            cfg["query_length"], False, "queries")
    dim = int(doc_emb[0].shape[1])

    def pack(embs):
        lens = np.array([e.shape[0] for e in embs], dtype=np.int64)
        return np.concatenate(embs, axis=0).astype(np.float32), lens

    corpus, corpus_lens = pack(doc_emb)
    queries, query_lens = pack(q_emb)
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/corpus.npy", corpus)
    np.save(f"{out}/corpus_lens.npy", corpus_lens)
    np.save(f"{out}/queries.npy", queries)
    np.save(f"{out}/query_lens.npy", query_lens)
    for fname, obj in [("corpus_ids.json", corpus_ids), ("query_ids.json", query_ids),
                       ("qrels.json", qrels)]:
        with open(f"{out}/{fname}", "w") as f:
            json.dump(obj, f)
    meta = {
        "model_id": model_id, "tag": tag, "dataset": dataset, "dim": dim,
        "onnx_file": onnx_file, "requantized_per_channel": requantize,
        "encoder": "onnxruntime-cpu",
        "onnxruntime": ort.__version__,
        "doc_prenorm_token_norms": doc_norms, "query_prenorm_token_norms": q_norms,
        "docs": len(corpus_ids), "queries": len(query_ids),
        "doc_tokens": int(corpus_lens.sum()), "query_tokens": int(query_lens.sum()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    bundles_vol.commit()
    _log(f"embed_onnx[{dataset}/{tag}]: wrote {out} dim={dim} "
         f"tokens={meta['doc_tokens']}")
    return {"bundle": out, "meta": meta}


@app.function(
    image=onnx_image, cpu=16, memory=16384, timeout=3600,
    secrets=[HF_SECRET], volumes={CACHE_DIR: hf_cache},
)
def batch_probe(n: int = 64, longest: bool = False, use_opts: bool = False):
    """Discriminate weight-recipe vs runtime cause of the int8 collapse.

    Both the vendor int8 export and our per-channel requant collapse under
    the batch-32 encode (per-token cos to fp32 ~0.89 but all tokens cos
    0.9995 to the global mean direction -> NDCG ~0). Dynamic-quant
    ACTIVATION scales are per-call max-abs over the whole [batch*seq,
    hidden] tensor; padding rows and cross-doc outliers exist only in the
    batched call. Encode the same SciFact docs with model_int8.onnx at
    batch=1 vs batch=32 against the fp32 reference and compare token
    spread."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_id = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    cfg = json.load(open(hf_hub_download(model_id, "onnx_config.json")))
    tok = AutoTokenizer.from_pretrained(model_id)
    _, corpus_texts, *_ = _load_beir("scifact")
    if longest:
        # The n longest docs by char length — reproduces the first batches of
        # embed_onnx's global length-sorted order (512-token padded batches).
        corpus_texts = sorted(corpus_texts, key=len, reverse=True)
    texts = [cfg["document_prefix"] + t for t in corpus_texts[:n]]
    _log(f"batch_probe: n={n} longest={longest} "
         f"char lens {min(map(len, texts))}..{max(map(len, texts))}")

    def run(sess, batch):
        order = (np.argsort([-len(t) for t in texts]) if batch > 1
                 else np.arange(len(texts)))
        out = [None] * len(texts)
        for s in range(0, len(texts), batch):
            bidx = order[s:s + batch]
            enc = tok([texts[i] for i in bidx], padding=True, truncation=True,
                      max_length=cfg["document_length"], return_tensors="np")
            feed = {i.name: enc[i.name].astype(np.int64)
                    for i in sess.get_inputs() if i.name in enc}
            emb = sess.run(None, feed)[0]
            for r, i in enumerate(bidx):
                keep = enc["attention_mask"][r].astype(bool)
                e = emb[r][keep].astype(np.float32)
                out[i] = e / np.maximum(
                    np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        return np.concatenate(out)

    def spread(e):
        mu = e.mean(0)
        mu /= np.linalg.norm(mu)
        s = e @ mu
        return s.mean(), s.std()

    fp32 = ort.InferenceSession(hf_hub_download(model_id, "model.onnx"),
                                providers=["CPUExecutionProvider"])
    ref = run(fp32, 1)
    m, s = spread(ref)
    _log(f"fp32 B=1   : spread(cos to mean dir) mean={m:.4f} std={s:.4f}")
    del fp32

    sess_opts = ort.SessionOptions() if use_opts else None
    _log(f"int8 session: use_opts={use_opts}")
    int8 = ort.InferenceSession(hf_hub_download(model_id, "model_int8.onnx"),
                                sess_opts, providers=["CPUExecutionProvider"])
    results = {"fp32_spread": [float(m), float(s)]}
    for label, b in [("int8 B=1", 1), ("int8 B=32", 32)]:
        e = run(int8, b)
        cos = (e * ref).sum(1)
        m, s = spread(e)
        _log(f"{label:<11}: cos(fp32) mean={cos.mean():.4f} p5={np.percentile(cos, 5):.4f}"
             f"  spread mean={m:.4f} std={s:.4f}")
        results[label] = {"cos_fp32": float(cos.mean()),
                          "spread": [float(m), float(s)]}
    return results


@app.local_entrypoint()
def probe(n: int = 64, longest: bool = False, use_opts: bool = False):
    _log(f"batch_probe: {json.dumps(batch_probe.remote(n, longest, use_opts))}")


@app.function(
    image=onnx_image, cpu=16, memory=32768, timeout=7200,
    secrets=[HF_SECRET], volumes={CACHE_DIR: hf_cache},
)
def trajectory_probe(warm_fp32: bool = False, max_batches: int = 0):
    """Full-corpus int8 encode replicating embed_onnx exactly (cpu=16,
    SessionOptions(), global length sort, B=32), logging within-batch token
    spread per batch. Healthy spread ~0.02; collapsed ~0.007. A mid-run drop
    means cumulative session-state corruption (ORT dynamic-quant + shrinking
    shapes); uniform collapse from batch 0 means an environment delta vs
    batch_probe."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_id = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    cfg = json.load(open(hf_hub_download(model_id, "onnx_config.json")))
    tok = AutoTokenizer.from_pretrained(model_id)
    _, corpus_texts, *_ = _load_beir("scifact")
    texts = [cfg["document_prefix"] + t for t in corpus_texts]

    if warm_fp32:
        # The one variable separating collapsed runs from healthy probes:
        # healthy probes ran an fp32 session in-process before creating the
        # int8 session. Replicate that with a single tiny fp32 call.
        f = ort.InferenceSession(hf_hub_download(model_id, "model.onnx"),
                                 providers=["CPUExecutionProvider"])
        warm = tok([texts[0][:200]], return_tensors="np")
        f.run(None, {i.name: warm[i.name].astype(np.int64)
                     for i in f.get_inputs() if i.name in warm})
        del f
        _log("warmed fp32 session before int8 session creation")

    opts = ort.SessionOptions()
    sess = ort.InferenceSession(hf_hub_download(model_id, "model_int8.onnx"),
                                opts, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    order = np.argsort([-len(t) for t in texts])
    B = 32
    spreads = []
    for s in range(0, len(order), B):
        if max_batches and s // B >= max_batches:
            break
        bidx = order[s:s + B]
        enc = tok([texts[i] for i in bidx], padding=True, truncation=True,
                  max_length=cfg["document_length"], return_tensors="np")
        feed = {n: enc[n].astype(np.int64) for n in in_names if n in enc}
        emb = sess.run(None, feed)[0]
        keep = enc["attention_mask"].astype(bool)
        e = emb[keep].astype(np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        mu = e.mean(0)
        mu /= np.linalg.norm(mu)
        spreads.append(float((e @ mu).std()))
        bi = s // B
        if bi % 10 == 0 or spreads[-1] < 0.012:
            _log(f"batch {bi:3d} seqlen={enc['input_ids'].shape[1]:3d} "
                 f"spread={spreads[-1]:.4f}")
    arr = np.array(spreads)
    _log(f"trajectory: first10={arr[:10].mean():.4f} last10={arr[-10:].mean():.4f} "
         f"min={arr.min():.4f} n_collapsed(<0.012)={int((arr < 0.012).sum())}/{len(arr)}")
    return {"first10": float(arr[:10].mean()), "last10": float(arr[-10:].mean()),
            "min": float(arr.min()), "spreads": [round(v, 5) for v in spreads]}


@app.local_entrypoint()
def trajectory(warm_fp32: bool = False, max_batches: int = 0):
    r = trajectory_probe.remote(warm_fp32, max_batches)
    _log(f"trajectory_probe(warm_fp32={warm_fp32}): "
         f"first10={r['first10']} last10={r['last10']} min={r['min']}")


onnx_latest_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime",  # unpinned: latest — ORT version probe for the
                        # int8-first-session bug (embed image pins 1.20.1)
        "transformers>=4.48",
        "numpy>=1.24",
        "huggingface_hub>=0.23",
        "hf_transfer>=0.1.6",
    )
    .env({
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
    })
)


@app.function(
    image=onnx_latest_image, cpu=16, memory=16384, timeout=3600,
    secrets=[HF_SECRET], volumes={CACHE_DIR: hf_cache},
)
def version_probe(warm_fp32: bool = False):
    """int8-first-session bug check on the LATEST onnxruntime (x86).
    Two batches of the 32 longest SciFact docs; healthy spread ~0.019,
    collapsed ~0.006. Run once per order in SEPARATE modal runs — the bug
    state is process-global."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_id = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    cfg = json.load(open(hf_hub_download(model_id, "onnx_config.json")))
    tok = AutoTokenizer.from_pretrained(model_id)
    _, corpus_texts, *_ = _load_beir("scifact")
    texts = sorted((cfg["document_prefix"] + t for t in corpus_texts),
                   key=len, reverse=True)[:64]

    if warm_fp32:
        f = ort.InferenceSession(hf_hub_download(model_id, "model.onnx"),
                                 providers=["CPUExecutionProvider"])
        w = tok(["[D] warmup"], return_tensors="np")
        f.run(None, {i.name: w[i.name].astype(np.int64)
                     for i in f.get_inputs() if i.name in w})
        del f

    sess = ort.InferenceSession(hf_hub_download(model_id, "model_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    spreads = []
    for s in range(0, 64, 32):
        enc = tok(texts[s:s + 32], padding=True, truncation=True,
                  max_length=cfg["document_length"], return_tensors="np")
        emb = sess.run(None, {i.name: enc[i.name].astype(np.int64)
                              for i in sess.get_inputs() if i.name in enc})[0]
        e = emb[enc["attention_mask"].astype(bool)].astype(np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        mu = e.mean(0)
        mu /= np.linalg.norm(mu)
        spreads.append(float((e @ mu).std()))
    _log(f"version_probe: ort={ort.__version__} warm_fp32={warm_fp32} "
         f"spreads={[round(v, 4) for v in spreads]}")
    return {"ort": ort.__version__, "warm_fp32": warm_fp32, "spreads": spreads}


@app.local_entrypoint()
def ort_version(warm_fp32: bool = False):
    _log(f"version_probe: {json.dumps(version_probe.remote(warm_fp32))}")


def _ndcg10(ranked_rels, all_rels):
    """Harness-exact NDCG@10: gain 2^r - 1, discount 1/log2(i+2)."""
    import math

    dcg = sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(ranked_rels[:10]))
    ideal = sorted(all_rels, reverse=True)
    idcg = sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(ideal[:10]))
    return dcg / idcg if idcg > 0 else 0.0


@app.function(
    image=eval_image, cpu=16, memory=65536, timeout=7200,
    volumes={BUNDLE_DIR: bundles_vol},
)
def ceiling(tag: str, dataset: str, force: bool = False):
    """Full per-query x per-doc exact-f32 MaxSim score matrix -> f32_scores.npy.

    The f32 ceiling of ANY doc subset is then a column-select + argsort."""
    import numpy as np

    bundle = f"{ROOT}/{dataset}_{tag}"
    out = f"{bundle}/f32_scores.npy"
    if os.path.exists(out) and not force:
        _log(f"ceiling[{dataset}/{tag}]: exists, skipping")
        return {"scores": out, "skipped": True}

    corpus = np.load(f"{bundle}/corpus.npy")
    clens = np.load(f"{bundle}/corpus_lens.npy")
    queries = np.load(f"{bundle}/queries.npy")
    qlens = np.load(f"{bundle}/query_lens.npy")
    doc_off = np.concatenate([[0], np.cumsum(clens)])
    q_off = np.concatenate([[0], np.cumsum(qlens)])
    n_docs, n_q = len(clens), len(qlens)
    scores = np.zeros((n_q, n_docs), np.float32)

    BLOCK = 2048  # docs per GEMM block; bounds sim matrix memory
    QGRP = 16     # queries per group (ArguAna queries are long — bounds sim rows)
    for bs in range(0, n_docs, BLOCK):
        be = min(bs + BLOCK, n_docs)
        tok = corpus[doc_off[bs]:doc_off[be]]
        rel_off = (doc_off[bs:be] - doc_off[bs]).astype(np.intp)
        for qs in range(0, n_q, QGRP):
            qe = min(qs + QGRP, n_q)
            qtok = queries[q_off[qs]:q_off[qe]]
            sim = qtok @ tok.T                                   # [qtok, blocktok]
            tokmax = np.maximum.reduceat(sim, rel_off, axis=1)   # [qtok, docs]
            qrel = (q_off[qs:qe] - q_off[qs]).astype(np.intp)
            scores[qs:qe, bs:be] = np.add.reduceat(tokmax, qrel, axis=0)
        _log(f"ceiling[{dataset}/{tag}]: docs {be}/{n_docs}")

    np.save(out, scores)
    bundles_vol.commit()
    _log(f"ceiling[{dataset}/{tag}]: wrote {out} shape={scores.shape}")
    return {"scores": out, "shape": list(scores.shape)}


@app.function(
    image=eval_image, cpu=16, memory=65536, timeout=10800,
    volumes={BUNDLE_DIR: bundles_vol},
)
def eval_cell(tag: str, dataset: str, size: int = 0, seed: int = 0,
              harness_sha: str = "unknown", force: bool = False):
    """One grid/scale cell: subset -> Rust binary_ndcg (4 schemes) + f32 ceiling."""
    import platform
    import shutil

    import numpy as np

    bundle = f"{ROOT}/{dataset}_{tag}"
    rname = f"{ROOT}/results/{dataset}_{tag}_n{size or 'full'}_s{seed}.json"
    if os.path.exists(rname) and not force:
        _log(f"eval[{dataset}/{tag} n={size} s={seed}]: result exists, skipping")
        return json.load(open(rname))

    corpus_ids = json.load(open(f"{bundle}/corpus_ids.json"))
    qrels = json.load(open(f"{bundle}/qrels.json"))
    clens = np.load(f"{bundle}/corpus_lens.npy")
    n_docs = len(clens)

    # -- doc subset: judged docs pinned, nested distractor prefix ------------
    if size and size < n_docs:
        judged = {d for rels in qrels.values() for d in rels}
        idx_j = [i for i, c in enumerate(corpus_ids) if c in judged]
        idx_r = [i for i, c in enumerate(corpus_ids) if c not in judged]
        perm = np.random.default_rng(seed).permutation(len(idx_r))
        need = max(0, size - len(idx_j))
        keep = np.array(sorted(idx_j + [idx_r[j] for j in perm[:need]]))
    else:
        keep = np.arange(n_docs)
        size = 0

    work = "/tmp/cell"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    corpus = np.load(f"{bundle}/corpus.npy")
    doc_off = np.concatenate([[0], np.cumsum(clens)])
    np.save(f"{work}/corpus.npy",
            np.concatenate([corpus[doc_off[i]:doc_off[i + 1]] for i in keep]))
    np.save(f"{work}/corpus_lens.npy", clens[keep])
    with open(f"{work}/corpus_ids.json", "w") as f:
        json.dump([corpus_ids[i] for i in keep], f)
    for fname in ("queries.npy", "query_lens.npy", "query_ids.json", "qrels.json"):
        shutil.copy(f"{bundle}/{fname}", f"{work}/{fname}")
    del corpus

    # -- exact-f32 ceiling on this subset from the precomputed score matrix --
    query_ids = json.load(open(f"{bundle}/query_ids.json"))
    f32_ndcg, f32_per_query = None, None
    if os.path.exists(f"{bundle}/f32_scores.npy"):
        scores = np.load(f"{bundle}/f32_scores.npy", mmap_mode="r")[:, keep]
        kept_ids = [corpus_ids[i] for i in keep]
        vals = []
        for qi, qid in enumerate(query_ids):
            rels = qrels.get(qid, {})
            if not rels:
                continue
            order = np.argsort(-scores[qi])[:10]
            vals.append(_ndcg10([rels.get(kept_ids[j], 0) for j in order],
                                list(rels.values())))
        f32_ndcg = float(np.mean(vals))
        f32_per_query = [round(v, 5) for v in vals]

    # -- Rust harness (residual-4/2/1 + binary-int8x1bit) --------------------
    env = dict(os.environ, NDCG_JSON="1")
    if (size or n_docs) >= 20000:
        env["NDCG_DEPLOYED_ONLY"] = "1"  # skip the slow wide-ANN pass at scale
    _log(f"eval[{dataset}/{tag} n={size or n_docs} s={seed}]: arch={platform.machine()} "
         f"sha={harness_sha} running harness ...")
    t = time.time()
    proc = subprocess.run(
        ["/src/target/release/examples/binary_ndcg", work],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"harness failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("NDCG_JSON"))
    harness = json.loads(line[len("NDCG_JSON "):])

    result = {
        "dataset": dataset, "tag": tag,
        "model_id": MODELS.get(tag) or ONNX_TAGS[tag][0],
        "size": size or n_docs, "n_docs_kept": int(len(keep)), "seed": seed,
        "f32_ndcg": f32_ndcg, "f32_per_query": f32_per_query, "harness": harness,
        "harness_sha": harness_sha, "arch": platform.machine(),
        "harness_wall_s": round(time.time() - t, 1),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(rname, "w") as f:
        json.dump(result, f, indent=1)
    bundles_vol.commit()
    for r in harness["schemes"]:
        ret = f" ret={r['dep_ndcg'] / f32_ndcg:6.1%}" if f32_ndcg else ""
        _log(f"  {r['name']:<18} dep={r['dep_ndcg']:.4f}{ret}")
    return result


def _sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                              ).stdout.strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def embed_all(force: bool = False):
    """All 12 (model x dataset) embeds, parallel, idempotent."""
    pairs = [(t, d, force) for t in MODELS for d in DATASETS]
    for r in embed.starmap(pairs):
        _log(f"done: {r['bundle']}" + (" (skipped)" if r.get("skipped") else ""))


@app.local_entrypoint()
def ceiling_all(force: bool = False):
    pairs = [(t, d, force) for t in MODELS for d in DATASETS]
    for r in ceiling.starmap(pairs):
        _log(f"done: {r['scores']}" + (" (skipped)" if r.get("skipped") else ""))


@app.local_entrypoint()
def grid(force: bool = False):
    """Figure A: 3 models x {scifact, nfcorpus, arguana}, full corpus."""
    sha = _sha()
    cells = [(t, d, 0, 0, sha, force) for t in MODELS
             for d in ("scifact", "nfcorpus", "arguana")]
    for r in eval_cell.starmap(cells):
        _log(f"done: {r['dataset']}/{r['tag']}")


@app.local_entrypoint()
def scale_strip(force: bool = False):
    """Figure B: FiQA scale curve x 3 models, + 3-seed variance probe @14K/17m."""
    sha = _sha()
    cells = [(t, SCALE_DATASET, n, 0, sha, force) for t in MODELS for n in SCALE_SIZES]
    cells += [("edge17m", SCALE_DATASET, 14000, s, sha, force) for s in (1, 2)]
    for r in eval_cell.starmap(cells):
        _log(f"done: {r['dataset']}/{r['tag']} n={r['size']} s={r['seed']}")


@app.local_entrypoint()
def onnx_pair(dataset: str = "scifact", force: bool = False):
    """Encoder-precision strip: fp32-ONNX vs int8-ONNX 17m encoder, same
    preprocessing, full pipeline (embed -> ceiling -> eval_cell). SciFact
    default: the 17m checkpoint's largest absolute binary loss with room
    above the floor to detect further degradation."""
    sha = _sha()
    for r in embed_onnx.starmap([(t, dataset, force) for t in ONNX_TAGS]):
        _log(f"embedded: {r['bundle']}" + (" (skipped)" if r.get("skipped") else ""))
    for r in ceiling.starmap([(t, dataset, force) for t in ONNX_TAGS]):
        _log(f"ceiling: {r['scores']}" + (" (skipped)" if r.get("skipped") else ""))
    for r in eval_cell.starmap([(t, dataset, 0, 0, sha, force) for t in ONNX_TAGS]):
        _log(f"done: {r['dataset']}/{r['tag']}")


@app.local_entrypoint()
def onnx_fix(dataset: str = "scifact"):
    """Force-redo only the quantized-encoder cells with the fp32-warm
    workaround; the onnxf32 cell is unaffected and kept."""
    sha = _sha()
    tags = ["edge17m_q8", "edge17m_q8pc"]
    for r in embed_onnx.starmap([(t, dataset, True) for t in tags]):
        _log(f"embedded: {r['bundle']}")
    for r in ceiling.starmap([(t, dataset, True) for t in tags]):
        _log(f"ceiling: {r['scores']}")
    for r in eval_cell.starmap([(t, dataset, 0, 0, sha, True) for t in tags]):
        _log(f"done: {r['dataset']}/{r['tag']}")


@app.local_entrypoint()
def report():
    """Print every stored result as a flat table (download results first:
    modal volume get tacet-bundles quant_grid/results ./results --force)."""
    import glob as g

    rows = []
    for p in sorted(g.glob("results/*.json")):
        r = json.load(open(p))
        for s in r["harness"]["schemes"]:
            ret = s["dep_ndcg"] / r["f32_ndcg"] if r.get("f32_ndcg") else float("nan")
            rows.append((r["dataset"], r["tag"], r["size"], r["seed"], s["name"],
                         s["dep_ndcg"], r.get("f32_ndcg"), ret))
    if rows:
        print(f"{'dataset':<10}{'model':<12}{'size':>7}{'seed':>5}{'scheme':<20}"
              f"{'dep':>8}{'f32':>8}{'retain':>8}")
        for d, t, n, s, name, dep, f32, ret in rows:
            f32s = f"{f32:8.4f}" if f32 else "     n/a"
            print(f"{d:<10}{t:<12}{n:>7}{s:>5}{name:<20}{dep:>8.4f}{f32s}{ret:>7.1%}")
