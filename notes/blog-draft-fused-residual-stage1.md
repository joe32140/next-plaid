# The rescore that never decompresses (and the stage-1 it exposed)

*A two-part optimization story from next-plaid's residual route: first we taught
stage 2 to score compressed codes directly, then we discovered that the fastest
kernel's real product is a new bottleneck — and chased it upstream through five
more phases. Every number below is measured; the ones from shared CI runners are
quoted as ratios, never absolutes.*

<!-- STATUS: draft. Two slots pending tonight's runs:
     [CI-TABLE]   3-platform e2e verdict from e2e-ab run
     [SCIFACT-PARITY] r2/r1 nDCG rows -->

## The uncomfortable starting point

PLAID-style retrieval engines store every document token as a compressed code:
a centroid id plus a few bits of residual correction per dimension. That makes
the index 8–32× smaller than float16 — and then, at query time, the engine
decompresses every candidate token back to floats so a BLAS GEMM can multiply
them. Measured inside next-plaid's float rescore path, **decompression is
61–83% of stage-2 wall time** depending on platform and dataset. The
compression win is repaid, with interest, on every query.

We knew this pattern from a side project — [nano-plaid](https://github.com/joe32140/nano-plaid),
a numpy-first teaching engine where the same decompress-then-GEMM route made
the *best-quality* scheme 5.5× slower than brute-force float search. There we
fixed it with a trick old enough to have gray hair (FAISS's fast-scan and
llama.cpp's Q4 kernels are built on it): a 4-bit code indexes a **16-entry
table of int8 weights, and a 16-byte table fits in a SIMD register**, so
"decode" is one `tbl` (NEON) or `pshufb` (AVX2) instruction. The dot product
splits along the codec's own anatomy:

```
q · token  =  q · centroid[cid]   +   Σ_d  q_d · weights[code_d]
              ^ already computed      ^ int8 dot on looked-up bytes
                by stage 1's
                centroid GEMM
```

The centroid term is a lookup into a matrix stage 1 built anyway for candidate
pruning. The residual term is an integer dot between the int8-quantized query
and bytes that never leave the register file. Nothing is ever decompressed;
float appears once, at the final scale-and-add.

## Porting it into a production engine

nano-plaid's kernel was compiled for dim = 128 and one packing layout. The
port into next-plaid (`residual_asym` on the search parameters) had to survive
the production codec — any nbits, any dim, an LSB-first bit-reversal in the
packing — and it taught us three things the toy repo couldn't:

**Generalize for the engine; re-specialize for the register.** The honest
general decoder is a 256-entry byte→weights table. It is also unSIMDable —
`tbl`/`pshufb` address 16 entries — and our first "fast" port measured **0.46×**:
slower than the GEMM it replaced. The fix factors the byte table into
per-key-position 16-entry nibble tables, **verifies the factorization over all
256 byte values at build time**, and falls back to scalar if the check ever
fails. The equivalence is a tested invariant, not a hope — and the bit-identity
suite runs on x86 AVX2, Apple NEON, and Neoverse in CI, so no platform's kernel
can drift from the scalar spec.

**Attribution beats vibes.** The port carried three optimizations, so we built
an ablation switch that disables exactly one per run — same binary, same cached
indexes, same queries, parity suite green under every mode. The result
surprised us: the *data layout* (transposing stage 1's centroid-score matrix
to centroid-major so a token's scores sit contiguous) did most of the work on
every CPU, worth up to 1.8× alone on Neoverse. The fold vectorization we were
proudest of turned out to be the smaller half everywhere — and a measured
*regression* on the M4 until a transpose-reduce repaired it. An optimization's
value is a property of the kernel shape it was measured in; it moves to a new
shape as a hypothesis, not as a number.

**Quality gates before speed claims.** Across 3 embedding models × 3 datasets
× r4/r2/r1, the fused int8 path moved nDCG@10 by **|Δ| ≤ 0.0021** against
decompress-then-GEMM. On real GTE embeddings with deployed settings, float vs
fused agree to the third decimal place (table at the end). The int8 error
lands only on the residual; the centroid term — most of the magnitude — stays
float.

## The bottleneck walks upstream

Then the trap we should have seen coming: with stage 2 fast, **stage 1 was
suddenly most of the query**. On an M4 at fiqa-52k scale, stage 1 cost 6.70 ms
against a ~1.7 ms fused rescore. So we profiled stage 1 into its five phases —
centroid GEMM (cdot), IVF probe, candidate gather, approximate "flood" scoring,
prune — and worked the list biggest-slice-first, re-profiling after every win.
One night of profile-driven changes, each tied to a hardware principle:

- **Flood: registers beat memory.** The flood walks every candidate's centroid
  codes and takes per-query-token maxes. We quantized the centroid-score matrix
  to int8 once per query, padded its stride so each token's 16 scores load as
  one aligned vector, and kept the running max in a register — one `umax.16b`
  per token, verified in the disassembly. The accumulator never touches memory
  until the final store.
- **cdot: free parallelism, bit for bit.** The centroid GEMM parallelizes over
  column blocks — reductions stay inside one element, so the result is
  bit-identical to the serial product (a property we test, not assume).
- **Probe: a top-k that barely looks.** Selecting the clusters to probe now
  scans with a running threshold and a per-chunk max pre-filter: a chunk whose
  max can't beat the current k-th value is skipped with one predictable branch.
- **Gather: the address space is the sort.** Candidate ids are dense, so a
  word bitmap dedups in O(n) and — as a free side effect — emits them sorted:
  the address space *is* the sort order.
- **Two kills, cheaply.** A dedup side-array died in ten minutes when counting
  showed a 1–4% duplicate rate; a task-granularity "fix" died when geomean over
  all 32 CI cells showed the regression it defended against didn't exist. The
  cheapest optimization is the one you kill before writing it — and the way to
  kill it cheaply is to measure the thing it depends on first.

Stage 1 went **6.70 → 1.66 ms** on the M4 (4.0×), with the quantized flood
gated on quality: identical nDCG@10 to four decimals against exact flooding on
both test corpora. The per-phase speedups are much bigger than 4× — the probe
alone is 9.5× — and that gap *is* Amdahl's law: total speedup is
1/Σ(share_i/speedup_i), so the biggest ratio is worth exactly as much as the
slice it applies to. Chase slices, not ratios.

## The verdict, on three platforms plus the M4

Single-stream, same prebuilt indexes, same length-realistic queries,
interleaved runs so thermal drift hits both sides alike. On the idle M4
(trustworthy absolutes), end-to-end at fiqa-15k, residual-4:

| | mainline v1.6.5 | this branch (fused, asym) | speedup |
|---|---|---|---|
| e2e / query | 18.9 ms | 3.0 ms | **6.3×** |
| stage-2 rescore | 14.25 ms (71% of it decompression) | 2.08 ms, no decompress phase | 6.9× |
| stage 1 | 4.68 ms | 0.93 ms | 5.0× |

The same A/B through GitHub's shared runners (4-vCPU VMs — ratios are the
signal, absolutes are not):

| platform (9-cell geomean: 3 datasets × r4/r2/r1) | branch, asym off — stage 1 alone | branch, asym on — full campaign |
|---|---|---|
| x86 AVX2 (GitHub CI) | 1.20× | **5.9×** |
| Neoverse N2 (GitHub CI) | 1.47× | **7.1×** |
| macOS arm64 (GitHub CI) | 1.34× | **5.6×** |

Two honest footnotes. First, the gap between the columns is a lesson in
itself: on 4-vCPU shared VMs the float rescore is so memory-starved that it
dominates end-to-end, so stage 1 alone only moves the total 1.2–1.5× there —
an e2e benchmark measures the *configuration*, not the branch, and you should
always say which switches are on. Second, the M4 *understates* the stage-1
win: its huge L2 absorbs scatter that server cores pay for — the probe rewrite
that measured 6.7× locally measured 13× on Neoverse. Never trust one
microarchitecture.

## Quality is the constant, speed is the variable

nDCG@10 on real GTE embeddings, deployed configuration, seed-42 builds —
float scoring vs the fused int8 path on identical indexes:

| dataset | r | float | fused | Δ |
|---|---|---|---|---|
| NFCorpus | 4 | 0.3809 | 0.3811 | +0.0002 |
| NFCorpus | 2 | 0.3779 | 0.3779 | 0.0000 |
| NFCorpus | 1 | 0.3701 | 0.3705 | +0.0004 |
| SciFact | 4 | 0.7609 | 0.7607 | −0.0002 |
[SCIFACT-PARITY]

The stage-1 quantized flood is held to the same standard: Δ = 0.0000 on both
corpora against exact flooding.

## What we'd tell you to steal

1. **Score the codes, not the reconstruction.** If your decoder fits in a
   register, decompression is a design mistake, not a cost.
2. **The dot product distributes over your codec.** Split it along the
   storage format's own anatomy and half the work is already done elsewhere.
3. **Profile in phases; attack the biggest slice; re-profile.** Amdahl is not
   a footnote, it's the schedule.
4. **Interleave your A/Bs, geomean all cells, and verify the binary's
   architecture** before believing any number.
5. **Gate every speedup on a quality metric** measured on real embeddings —
   and keep the negative results in the ledger.

The branch is `feat/asymmetric-lut-residual`; the teaching version of every
kernel here — with the interactive lessons that derive them — is
[nano-plaid's SIMD school](https://github.com/joe32140/nano-plaid).
