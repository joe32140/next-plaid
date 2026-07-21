#!/bin/bash
# Idle-M4 latency pass: all 9 grid bundles through the arm64 harness
# (deployed regime, float + binary + asym-LUT columns). Run on a quiet
# machine; outputs one log per cell plus the NDCG_JSON line for parsing.
set -u
BIN="$(dirname "$0")/../target/aarch64-apple-darwin/release/examples/binary_ndcg"
BUNDLES=~/beir-data/quant_grid
OUT=${1:-~/beir-data/quant_grid/m4_results}
mkdir -p "$OUT"
file "$BIN" | grep -q arm64 || { echo "FATAL: harness is not an arm64 binary (Rosetta trap)"; exit 1; }
for d in scifact nfcorpus arguana; do
  for t in lateon_reg gte edge17m; do
    cell="${d}_${t}"
    [ -f "$OUT/$cell.log" ] && { echo "skip $cell (done)"; continue; }
    echo "=== $cell $(date +%H:%M:%S)"
    NDCG_DEPLOYED_ONLY=1 NDCG_JSON=1 "$BIN" "$BUNDLES/$cell" >"$OUT/$cell.log" 2>&1
    grep -E "asym-LUT|retains" "$OUT/$cell.log" | head -5
  done
done
echo "M4 sweep complete: $OUT"
