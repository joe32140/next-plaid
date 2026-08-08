"""Cross-CPU benchmark of ONNX export variants for ColBERT/ModernBERT.

Every export "improvement" measured so far LOST on arm64. ORT dispatches to
completely different MLAS kernels on x86 (AVX2 / AVX-512 / VNNI) than on ARM
(NEON), so an arm64-only verdict cannot be generalised. This runs the same
matrix wherever it lands and prints one comparable table.

Deterministic synthetic corpus so numbers are comparable across machines.

  python export_matrix.py [--model ID] [--json out.json] [--docs 300] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colbert_export.export import (  # noqa: E402
    ColBERTForONNX,
    _legacy_exporter_kwargs,
    detect_model_architecture,
)
from colbert_export.modernbert_ops import attn_implementation, rotary_contrib_op  # noqa: E402

DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-17m"

FUSED_NAMES = (
    "Attention", "MultiHeadAttention", "SkipLayerNormalization", "LayerNormalization",
    "SimplifiedLayerNormalization", "FastGelu", "BiasGelu", "Gelu", "QuickGelu",
    "EmbedLayerNormalization", "RotaryEmbedding", "QAttention",
    "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QLinearMatMul",
)

KEYWORDS = ["fn", "let", "match", "impl", "struct", "pub", "async", "await", "return",
            "if", "else", "for", "while", "use", "mod", "trait", "where", "const"]
IDENTS = ["config", "buffer", "index", "session", "encoder", "tokenizer", "document",
          "embedding", "matrix", "cluster", "residual", "centroid", "query", "batch"]


def build_corpus(count: int, seed: int = 20260807) -> list[str]:
    """Deterministic code-shaped units with a realistic length spread."""
    rng = random.Random(seed)
    units = []
    for _ in range(count):
        n_lines = rng.choice([3, 4, 5, 6, 8, 10, 12, 16])
        lines = []
        for _ in range(n_lines):
            width = rng.randint(4, 12)
            parts = [rng.choice(KEYWORDS) if rng.random() < 0.35 else rng.choice(IDENTS)
                     for _ in range(width)]
            lines.append("    " + " ".join(parts))
        units.append("[D] " + "\n".join(lines))
    return units


def cpu_identity() -> dict:
    info = {"machine": platform.machine(), "system": platform.system(),
            "processor": platform.processor() or "", "python": platform.python_version()}
    try:
        if platform.system() == "Darwin":
            info["brand"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif platform.system() == "Linux":
            flags, brand = set(), ""
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name") and not brand:
                    brand = line.split(":", 1)[1].strip()
                if line.startswith(("flags", "Features")):
                    flags.update(line.split(":", 1)[1].split())
            info["brand"] = brand
            info["isa"] = sorted(f for f in flags if f in {
                "avx2", "avx512f", "avx512bw", "avx512vnni", "avx_vnni", "amx_int8",
                "asimd", "asimddp", "sve", "i8mm", "bf16"})
    except Exception:
        pass
    try:
        info["cores"] = len(__import__("os").sched_getaffinity(0))
    except Exception:
        info["cores"] = __import__("os").cpu_count()
    return info


def census(path: Path):
    model = onnx.load(str(path), load_external_data=False)
    import collections
    return collections.Counter(n.op_type for n in model.graph.node)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", default=None)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fp32 only, fewer variants")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    out = Path(args.outdir or "export_matrix_out")
    out.mkdir(parents=True, exist_ok=True)

    identity = cpu_identity()
    print(f"host: {identity.get('brand', identity['machine'])} "
          f"[{identity['machine']}] cores={identity.get('cores')}")
    if identity.get("isa"):
        print(f"isa : {' '.join(identity['isa'])}")
    print(f"torch {torch.__version__} | onnx {onnx.__version__} | ort {ort.__version__}\n")

    from pylate import models as pylate_models

    # Fail with something actionable rather than a bare TypeError deep in the
    # constructor: pip can resolve a very old pylate on platforms with thin
    # wheel coverage, and the exporter depends on this kwarg.
    import inspect

    if "do_query_expansion" not in inspect.signature(pylate_models.ColBERT.__init__).parameters:
        raise SystemExit(
            "Installed pylate is too old: ColBERT.__init__ has no "
            "`do_query_expansion` (needs >=1.3.3, verified on 1.6.0). "
            "Pin it explicitly -- pip backtracks to 1.2.0 on some platforms."
        )

    pylate_model = pylate_models.ColBERT(model_name_or_path=args.model, device="cpu",
                                         do_query_expansion=False)
    arch = detect_model_architecture(pylate_model)
    backbone = pylate_model[0].auto_model
    is_modernbert = "ModernBert" in arch["model_class"]
    print(f"{arch['model_class']} | out_dim {arch['output_dim']} | "
          f"modernbert={is_modernbert}\n")

    wrapper = ColBERTForONNX(pylate_model, uses_token_type_ids=arch["uses_token_type_ids"])
    wrapper.eval()
    tokenizer = pylate_model[0].tokenizer
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    corpus = build_corpus(args.docs)
    probe = corpus[:4]
    enc = tokenizer(probe, return_tensors="pt", padding=True, truncation=True, max_length=256)
    feed_t = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if arch["uses_token_type_ids"]:
        feed_t["token_type_ids"] = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
    with torch.no_grad():
        reference = wrapper(**feed_t).numpy()
    probe_np = {k: v.numpy() for k, v in feed_t.items()}

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    feeds = []
    for unit in corpus:
        e = tok.encode(unit, add_special_tokens=True)
        item = {"input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        if arch["uses_token_type_ids"]:
            item["token_type_ids"] = np.zeros_like(item["input_ids"])
        feeds.append(item)
    tokens = sum(f["input_ids"].shape[1] for f in feeds)
    print(f"corpus: {len(feeds)} units, {tokens} tokens, mean {tokens/len(feeds):.0f}\n")

    names = list(feed_t.keys())
    axes = {n: {0: "batch_size", 1: "sequence_length"} for n in names}
    axes["output"] = {0: "batch_size", 1: "sequence_length"}
    example = tuple(feed_t[n] for n in names)

    def do_export(path, opset, rope, impl):
        with attn_implementation(backbone, impl):
            with rotary_contrib_op() if rope else _null():
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, example, str(path), input_names=names,
                        output_names=["output"], dynamic_axes=axes,
                        opset_version=opset, do_constant_folding=True,
                        **_legacy_exporter_kwargs(),
                    )

    import contextlib

    @contextlib.contextmanager
    def _null():
        yield False

    variants = [("op14", 14, False, "sdpa")]
    if not args.quick:
        variants += [
            ("op17", 17, False, "sdpa"),
            ("op14-eager", 14, False, "eager"),
        ]
    if is_modernbert:
        variants.append(("op14-rope", 14, True, "sdpa"))
        if not args.quick:
            variants.append(("op17-rope", 17, True, "sdpa"))

    built = []
    for tag, opset, rope, impl in variants:
        path = out / f"{tag}.onnx"
        try:
            do_export(path, opset, rope, impl)
            built.append((tag, path))
        except Exception as exc:
            print(f"  export {tag} FAILED: {str(exc)[:100]}")

    # ORT's offline transformer fusion, applied to the baseline
    if not args.quick and built:
        try:
            from onnxruntime.transformers.optimizer import optimize_model
            src = out / "op14.onnx"
            if src.exists():
                fused = optimize_model(str(src), model_type="bert", opt_level=1,
                                       use_gpu=False,
                                       num_heads=getattr(backbone.config, "num_attention_heads", 0),
                                       hidden_size=getattr(backbone.config, "hidden_size", 0))
                dst = out / "op14-ortfuse.onnx"
                fused.save_model_to_file(str(dst))
                built.append(("op14-ortfuse", dst))
        except Exception as exc:
            print(f"  ortfuse FAILED: {str(exc)[:100]}")

    # int8 siblings (default scheme for every export variant)
    if not args.quick:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for tag, path in list(built):
            q = out / f"{tag}-int8.onnx"
            try:
                # Mirror colbert_export.quantize.quantize_model exactly, so the
                # gated arms measure what we actually ship. Plain QInt8 is kept
                # below as a labelled diagnostic.
                quantize_dynamic(model_input=str(path), model_output=str(q),
                                 weight_type=QuantType.QInt8, reduce_range=True)
                built.append((f"{tag}-int8", q))
            except Exception as exc:
                print(f"  quantize {tag} FAILED: {str(exc)[:80]}")

        # Quantization-scheme sweep on the baseline export. ORT recommends
        # QUInt8 weights (the u8s8 path) and reduce_range on pre-VNNI x86;
        # on ARM the signed path is the fast one. This is exactly the knob
        # expected to invert between architectures.
        # Diagnostics that justify the shipped default. `plain` is the old
        # default and is expected to fail the cosine floor on x86 (0.911) --
        # that is the evidence for reduce_range, not a regression.
        schemes = [
            ("plain", dict(weight_type=QuantType.QInt8)),
            ("u8", dict(weight_type=QuantType.QUInt8)),
            ("s8pc", dict(weight_type=QuantType.QInt8, per_channel=True)),
        ]
        baseline = out / "op14.onnx"
        if baseline.exists():
            for name, kwargs in schemes:
                q = out / f"op14-int8-{name}.onnx"
                try:
                    quantize_dynamic(model_input=str(baseline), model_output=str(q), **kwargs)
                    built.append((f"op14-int8-{name}", q))
                except Exception as exc:
                    print(f"  quantize op14/{name} FAILED: {str(exc)[:80]}")

    def bench(path, tag):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opt_path = out / f"_opt_{tag}.onnx"
        opts.optimized_model_filepath = str(opt_path)

        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        session_ms = (time.perf_counter() - t0) * 1e3

        got = sess.run(None, probe_np)[0]
        delta = float(np.abs(got - reference).max())
        cos = float((got * reference).sum(-1).mean())

        for f in feeds[: min(20, len(feeds))]:
            sess.run(None, f)
        best = float("inf")
        for _ in range(args.reps):
            start = time.perf_counter()
            for f in feeds:
                sess.run(None, f)
            best = min(best, time.perf_counter() - start)

        counter = census(opt_path)
        opt_path.unlink(missing_ok=True)
        return {
            "tag": tag, "docs_per_s": len(feeds) / best, "tokens_per_s": tokens / best,
            "session_ms": session_ms, "nodes": sum(counter.values()),
            "max_abs_delta": delta, "cosine": cos,
            "fused": sorted(k for k in counter if k in FUSED_NAMES),
        }

    print(f"{'variant':<20} {'docs/s':>9} {'tok/s':>9} {'vs base':>8} {'nodes':>6} "
          f"{'sess ms':>8} {'max|d|':>10}")
    print("-" * 80)
    rows, base = [], {}
    for tag, path in built:
        try:
            row = bench(path, tag)
        except Exception as exc:
            print(f"{tag:<20} RUN FAILED: {str(exc)[:60]}")
            continue
        kind = "int8" if "int8" in tag else "fp32"
        if tag in ("op14", "op14-int8"):
            base[kind] = row["docs_per_s"]
        row["vs_base"] = row["docs_per_s"] / base.get(kind, row["docs_per_s"])
        rows.append(row)
        print(f"{tag:<20} {row['docs_per_s']:>9.1f} {row['tokens_per_s']:>9.0f} "
              f"{row['vs_base']:>7.3f}x {row['nodes']:>6} {row['session_ms']:>8.1f} "
              f"{row['max_abs_delta']:>10.2e}")

    if rows:
        best_fp32 = max((r for r in rows if "int8" not in r["tag"]),
                        key=lambda r: r["docs_per_s"])
        print(f"\nfastest fp32: {best_fp32['tag']} ({best_fp32['docs_per_s']:.1f} docs/s)")
        int8_rows = [r for r in rows if "int8" in r["tag"]]
        if int8_rows:
            best_int8 = max(int8_rows, key=lambda r: r["docs_per_s"])
            print(f"fastest int8: {best_int8['tag']} ({best_int8['docs_per_s']:.1f} docs/s) "
                  f"= {best_int8['docs_per_s']/best_fp32['docs_per_s']:.3f}x of best fp32")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"host": identity, "model": args.model,
             "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                          "onnxruntime": ort.__version__},
             "corpus": {"units": len(feeds), "tokens": tokens},
             "results": rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
