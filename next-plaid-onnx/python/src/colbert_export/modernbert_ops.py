"""ModernBERT-specific ONNX export specialisations.

ModernBERT is the dominant ColBERT backbone (GTE-ModernColBERT, LateOn,
mxbai-edge, Reason-ModernColBERT), so architecture-specific export work
amortises across most of the ecosystem.

ORT's generic fusion does not help it: passing `num_heads`/`hidden_size` hints
to `onnxruntime.transformers.optimizer` changes nothing, because that pass
matches BERT's absolute-position attention and ModernBERT uses RoPE with
alternating local/global windows.

`rotary_contrib_op()` swaps ModernBERT's `apply_rotary_pos_emb` for a version
whose eager math is unchanged (bit-identical) but which exports to a single
`com.microsoft.RotaryEmbedding` node per tensor instead of ~25 glue nodes.

MEASURED (ORT 1.28, 2026-08-07, single-threaded, docs/s relative to the
op14 baseline on the same host+model):

                        x86/17M   x86/150M   arm/17M   arm/150M
    rope contrib op      1.069x    1.016x     0.966x    0.976x
    + int8 (combined)    1.580x    1.797x     0.855x    1.215x

The effect is determined by CPU ARCHITECTURE, consistently across model size:
a real win on x86_64, a real loss on arm64. ORT's CPU RotaryEmbedding kernel
has an x86 path that ARM does not benefit from. It cuts the graph 557 -> 406
nodes (17M) / 1587 -> 1109 (150M) on both.

Correctness is not in question: eager parity is exactly 0.000e+00 (forward is
the stock math) and ONNX parity ~4e-06.

This stays an explicit opt-in because there is no single correct default.
`attn_implementation` is NOT a safe blanket default either -- it wins on three
of the four host/model combinations (1.034-1.046x) but LOSES on x86/150M
(0.971x).

`benchmarks/export_matrix.py` reproduces the table on any host; the
`ONNX export matrix` workflow runs it across four ISAs and two model sizes.
"""

from __future__ import annotations

import contextlib

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _RotaryEmbeddingONNX(torch.autograd.Function):
    """forward == stock math (parity by construction); symbolic == contrib op."""

    @staticmethod
    def forward(ctx, x, cos, sin, cos_cache, sin_cache, position_ids):
        return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))

    @staticmethod
    def symbolic(g, x, cos, sin, cos_cache, sin_cache, position_ids):
        # interleaved=0 is the rotate_half (GPT-NeoX) convention ModernBERT uses.
        return g.op(
            "com.microsoft::RotaryEmbedding",
            x, position_ids, cos_cache, sin_cache,
            interleaved_i=0,
        )


def _patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # ModernBERT builds cos/sin as cat(freqs, freqs) with shape
    # [batch, seq, head_dim], so the first half IS the [seq, head_dim/2] cache
    # ORT expects.
    half = cos.shape[-1] // 2
    cos_cache = cos[0, :, :half].contiguous()
    sin_cache = sin[0, :, :half].contiguous()
    # Rank-1 position_ids is ORT's "start offset" form; using the (batch, seq)
    # form would bake the traced batch size into the graph.
    position_ids = torch.zeros(1, dtype=torch.int64, device=cos.device)
    return (
        _RotaryEmbeddingONNX.apply(q, cos, sin, cos_cache, sin_cache, position_ids),
        _RotaryEmbeddingONNX.apply(k, cos, sin, cos_cache, sin_cache, position_ids),
    )


@contextlib.contextmanager
def rotary_contrib_op():
    """Emit com.microsoft.RotaryEmbedding for ModernBERT RoPE during export."""
    try:
        from transformers.models.modernbert import modeling_modernbert as mb
    except ImportError:  # not a ModernBERT install; nothing to patch
        yield False
        return

    original = mb.apply_rotary_pos_emb
    mb.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
    try:
        yield True
    finally:
        mb.apply_rotary_pos_emb = original


@contextlib.contextmanager
def attn_implementation(model, impl: str):
    """Trace under a specific attention implementation ('eager' or 'sdpa').

    sdpa traces into a decomposed pattern; eager produces the explicit
    matmul/softmax shape that ORT's attention fusion is written against, so the
    two can optimise very differently.
    """
    config = model.config
    previous = getattr(config, "_attn_implementation", None)
    try:
        config._attn_implementation = impl
        yield
    finally:
        if previous is not None:
            config._attn_implementation = previous
