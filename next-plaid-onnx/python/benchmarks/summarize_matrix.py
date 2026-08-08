"""Merge export_matrix.py JSON results from several CPUs into one markdown table.

  python summarize_matrix.py results/*.json >> $GITHUB_STEP_SUMMARY
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Minimum acceptable int8 vs fp32 embedding cosine. The shipped default
# QInt8 scheme measured 0.911 on x86_64, which is real ranking damage.
INT8_COS_MIN = 0.99

# Scheme-sweep arms exist to justify the shipped default, not to be shipped.
# They are reported but not gated, or CI would sit red on a diagnostic.
DIAGNOSTIC_SUFFIXES = ("-plain", "-u8", "-s8pc")


def load(paths):
    runs = []
    for p in paths:
        try:
            runs.append(json.loads(Path(p).read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"<!-- skipped {p}: {exc} -->")
    return runs


def label(run):
    host = run.get("host", {})
    brand = host.get("brand") or host.get("processor") or "?"
    brand = brand.replace("(R)", "").replace("(TM)", "").strip() or "?"
    # The model belongs in the label: the same host is benchmarked once per
    # model, and int8 can win on one size and lose on the other.
    model = str(run.get("model", "")).split("/")[-1]
    return f"{brand[:26]} ({host.get('machine', '?')}) / {model[:22]}"


def main() -> int:
    runs = load(sys.argv[1:])
    if not runs:
        print("no results")
        return 1

    tags = []
    for run in runs:
        for row in run["results"]:
            if row["tag"] not in tags:
                tags.append(row["tag"])

    print("## ONNX export matrix\n")
    for run in runs:
        host = run.get("host", {})
        isa = " ".join(host.get("isa", [])) or "n/a"
        print(f"- **{label(run)}** — cores {host.get('cores')}, isa `{isa}`, "
              f"ort {run['versions']['onnxruntime']}, torch {run['versions']['torch']}")
    print()

    # docs/s, each variant relative to that host's own same-precision baseline.
    # Ratios are recomputed here rather than read from `vs_base` so older result
    # files stay comparable.
    def baseline(run, tag):
        kind = "int8" if "int8" in tag else "fp32"
        want = "op14-int8" if kind == "int8" else "op14"
        row = next((r for r in run["results"] if r["tag"] == want), None)
        return row["docs_per_s"] if row else None

    print("| variant | " + " | ".join(label(r) for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for tag in tags:
        cells = []
        for run in runs:
            row = next((r for r in run["results"] if r["tag"] == tag), None)
            if row is None:
                cells.append("—")
            else:
                base = baseline(run, tag)
                ratio = f" ({row['docs_per_s']/base:.3f}×)" if base else ""
                cells.append(f"{row['docs_per_s']:.0f}{ratio}")
        print(f"| `{tag}` | " + " | ".join(cells) + " |")

    print("\n### Best per host\n")
    print("| host | fastest fp32 | fastest int8 | int8 vs fp32 |")
    print("|---|---|---|---|")
    for run in runs:
        rows = run["results"]
        fp32 = [r for r in rows if "int8" not in r["tag"]]
        i8 = [r for r in rows if "int8" in r["tag"]]
        if not fp32:
            continue
        bf = max(fp32, key=lambda r: r["docs_per_s"])
        bi = max(i8, key=lambda r: r["docs_per_s"]) if i8 else None
        ratio = f"{bi['docs_per_s']/bf['docs_per_s']:.3f}×" if bi else "—"
        print(f"| {label(run)} | `{bf['tag']}` {bf['docs_per_s']:.0f} | "
              f"{'`'+bi['tag']+'` '+format(bi['docs_per_s'], '.0f') if bi else '—'} | {ratio} |")

    # ---- gates -----------------------------------------------------------
    # fp32 must be numerically equivalent to torch.
    print("\n### Parity\n")
    worst = 0.0
    for run in runs:
        for row in run["results"]:
            if "int8" not in row["tag"]:
                worst = max(worst, row["max_abs_delta"])
    print(f"worst fp32 `max|Δ|`: **{worst:.2e}**\n")

    # int8 quality is architecture-dependent and must be gated per host, not
    # globally: the SAME quantized file measured cosine 0.99975 on arm64 and
    # 0.91056 on x86_64 (pre-VNNI u8s8 saturation). A global average hides that.
    print("| host | variant | int8 cosine |")
    print("|---|---|---|")
    int8_fail = []
    for run in runs:
        for row in run["results"]:
            if "int8" not in row["tag"]:
                continue
            diagnostic = row["tag"].endswith(DIAGNOSTIC_SUFFIXES)
            low = row["cosine"] < INT8_COS_MIN
            if low and not diagnostic:
                int8_fail.append((label(run), row["tag"], row["cosine"]))
            mark = " (diagnostic)" if diagnostic else (" ⚠️" if low else "")
            print(f"| {label(run)} | `{row['tag']}` | {row['cosine']:.5f}{mark} |")

    failed = False
    if worst > 1e-4:
        print("\n> **FAIL** — an fp32 export variant diverged from the torch reference.")
        failed = True
    if int8_fail:
        print(f"\n> **FAIL** — int8 cosine below {INT8_COS_MIN} on:")
        for host, tag, cos in int8_fail:
            print(f"> - {host} `{tag}` = {cos:.5f}")
        print("> Try `reduce_range=True` — it fixes pre-VNNI x86 saturation at no throughput cost.")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
