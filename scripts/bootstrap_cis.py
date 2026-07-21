#!/usr/bin/env python3
"""Paired per-query bootstrap CIs for the quant-grid results.

Reads the result JSONs (download first:
  modal volume ls tacet-bundles quant_grid/results  # then modal volume get each
), and for every cell with per-query NDCG dumps computes 95% bootstrap CIs for:
  * each scheme's retention (dep/f32), paired against the f32 ceiling
  * the binary - residual-1 delta (the decision-rule contrast)
  * the residual-4 - residual-1 delta (the bits-vs-quality contrast)

Paired design: one resample of query indices is applied to BOTH sides of every
contrast, so run-to-run query difficulty cancels (report SS2.C: nothing under
~0.02 NDCG is a result without this).

Usage: python scripts/bootstrap_cis.py [results_dir] [n_boot]
"""
import glob
import json
import sys

import numpy as np

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
rng = np.random.default_rng(0)


def ci(deltas):
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo, hi


def boot_mean(vals, idx):
    return vals[idx].mean(axis=1)


rows = []
for p in sorted(glob.glob(f"{RESULTS}/*.json")):
    r = json.load(open(p))
    if "harness" not in r or not r.get("f32_per_query"):
        continue
    schemes = {s["name"]: s for s in r["harness"]["schemes"]}
    if not all("per_query" in s for s in schemes.values()):
        continue
    f32 = np.asarray(r["f32_per_query"])
    nq = len(f32)
    per = {n: np.asarray(s["per_query"]) for n, s in schemes.items()}
    if any(len(v) != nq for v in per.values()):
        print(f"skip {p}: per-query length mismatch")
        continue

    cell = f"{r['dataset']}/{r['tag']}" + (f"@{r['size']}" if r.get("size") else "")
    idx = rng.integers(0, nq, (N_BOOT, nq))
    f32_b = boot_mean(f32, idx)

    for name, v in per.items():
        ret = boot_mean(v, idx) / f32_b
        lo, hi = ci(ret)
        rows.append((cell, f"retention {name}", v.mean() / f32.mean(), lo, hi, nq))

    for a, b, label in [
        ("binary-int8x1bit", "residual-nbits1", "binary - r1"),
        ("residual-nbits4", "residual-nbits1", "r4 - r1"),
    ]:
        if a in per and b in per:
            d = boot_mean(per[a], idx) - boot_mean(per[b], idx)
            lo, hi = ci(d)
            sig = "" if lo <= 0 <= hi else "  *"
            rows.append((cell, f"delta {label}", per[a].mean() - per[b].mean(),
                         lo, hi, nq, sig))

print(f"{N_BOOT} paired bootstrap resamples; * = 95% CI excludes 0\n")
print(f"{'cell':<26}{'quantity':<30}{'point':>9}{'ci95_lo':>9}{'ci95_hi':>9}{'n':>6}")
for row in rows:
    cell, what, pt, lo, hi, nq = row[:6]
    sig = row[6] if len(row) > 6 else ""
    print(f"{cell:<26}{what:<30}{pt:>9.4f}{lo:>9.4f}{hi:>9.4f}{nq:>6}{sig}")
