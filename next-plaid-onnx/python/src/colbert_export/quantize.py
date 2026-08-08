"""Quantize ONNX models to INT8 for faster CPU inference.

This module applies dynamic quantization to reduce model size and speed up inference.
INT8 quantization typically provides:
- 3-4x reduction in model size
- 1.5-2x speedup in inference
- >0.99 cosine similarity preserved
"""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(model_dir: Path, verbose: bool = True) -> Path:
    """Quantize an ONNX model to INT8.

    Args:
        model_dir: Directory containing model.onnx
        verbose: Whether to print progress messages

    Returns:
        Path to quantized model (model_int8.onnx)
    """
    model_dir = Path(model_dir)
    input_path = model_dir / "model.onnx"
    output_path = model_dir / "model_int8.onnx"

    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")

    if verbose:
        print(f"Quantizing {input_path}...")
        print(f"  Input size: {input_path.stat().st_size / 1e6:.1f} MB")

    # Apply dynamic INT8 quantization
    # reduce_range=True quantizes weights to 7 bits. It is NOT strictly better
    # everywhere -- it is chosen because it raises the WORST case. Measured
    # 2026-08-07, ORT 1.28, embedding cosine vs the torch reference:
    #
    #                        x86/17M   x86/150M   arm/17M   arm/150M
    #   QInt8 (was default)   0.91056   0.98776   0.99975   0.99215
    #   QInt8 reduce_range    0.99423   0.98528   0.99906   0.99118
    #
    # Plain QInt8 collapses to 0.911 on pre-VNNI x86 for the small model
    # (u8s8 MatMulInteger saturation); reduce_range fixes that outright while
    # costing <=0.003 cosine elsewhere, so the floor goes 0.911 -> 0.985.
    # Throughput is unchanged on every host measured.
    #
    # Do NOT switch to per_channel=True: it is the best scheme on arm64
    # (0.99463) and CATASTROPHIC on x86 for the 150M model (cos 0.053).
    # Re-measure with benchmarks/export_matrix.py before changing any of this.
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        reduce_range=True,
    )

    if verbose:
        print(f"  Output size: {output_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Compression: {input_path.stat().st_size / output_path.stat().st_size:.1f}x")
        print(f"  Saved to: {output_path}")

    return output_path
