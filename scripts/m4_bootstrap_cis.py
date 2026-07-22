#!/usr/bin/env python3
"""Paired per-query bootstrap CIs for the local M4 sweep logs.

Reads the NDCG_JSON line from each cell log written by m4_latency_sweep.sh
and computes 95% bootstrap CIs for:
  * asym-LUT - float delta per residual scheme (the normalization acceptance
    test: the CI should sit inside +/-0.005 for every route)
  * binary - residual-1 delta (the decision-rule contrast, local re-check)
  * residual-4 - residual-1 delta (bits-vs-quality contrast)

Paired design: one resample of query indices is applied to BOTH sides of
every contrast, so query difficulty cancels (report SS2.C).

Usage: python scripts/m4_bootstrap_cis.py [logs_dir] [n_boot]
"""
import glob
import json
import os
import sys

import numpy as np

LOGS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/beir-data/quant_grid/m4_results")
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
rng = np.random.default_rng(0)


def ci(deltas):
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo, hi


def boot_mean(vals, idx):
    return vals[idx].mean(axis=1)


rows = []
for p in sorted(glob.glob(f"{LOGS}/*.log")):
    line = next((l for l in open(p) if l.startswith("NDCG_JSON ")), None)
    if line is None:
        print(f"skip {os.path.basename(p)}: no NDCG_JSON line")
        continue
    r = json.loads(line[len("NDCG_JSON "):])
    cell = os.path.basename(p)[:-len(".log")]
    per = {s["name"]: np.asarray(s["per_query"]) for s in r["schemes"]}
    asym = {s["name"]: np.asarray(s["asym"]["per_query"])
            for s in r["schemes"] if s.get("asym")}
    nq = r["queries"]
    if any(len(v) != nq for v in per.values()):
        print(f"skip {cell}: per-query length mismatch")
        continue
    idx = rng.integers(0, nq, (N_BOOT, nq))

    for name, a in sorted(asym.items()):
        d = boot_mean(a, idx) - boot_mean(per[name], idx)
        lo, hi = ci(d)
        sig = "" if lo <= 0 <= hi else "  *"
        rows.append((cell, f"asym - float {name}", a.mean() - per[name].mean(),
                     lo, hi, nq, sig))

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
print(f"{'cell':<22}{'quantity':<32}{'point':>9}{'ci95_lo':>9}{'ci95_hi':>9}{'n':>6}")
for cell, what, pt, lo, hi, nq, sig in rows:
    print(f"{cell:<22}{what:<32}{pt:>9.4f}{lo:>9.4f}{hi:>9.4f}{nq:>6}{sig}")
