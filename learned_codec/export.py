#!/usr/bin/env python3
"""
Export trained SRA model to ONNX for inference.

Exports the decoder only (for CPU decode at serve time).
The encoder runs in PyTorch on GPU at encode time.

Usage:
  python export.py --checkpoint checkpoints/sra_small_1e3/best.pt --output models/residual_codec.onnx
  python export.py --checkpoint best.pt --output codec.onnx --quantize  # INT8
"""

import argparse
import torch
import torch.onnx
import numpy as np

from model import SparseResidualAutoencoder, count_params, model_size_kb


def export_decoder(checkpoint_path: str, output_path: str, quantize: bool = False):
    """Export just the decoder to ONNX."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    variant = config["variant"]

    if variant == "unet":
        print("UNet variant requires full model export (encoder+decoder). Not supported yet.")
        return

    model = SparseResidualAutoencoder(variant)
    model.load_state_dict(ckpt["model"])
    model.eval()

    decoder = model.decoder

    # Determine latent dimensions from config
    configs = {
        "tiny": {"latent_channels": 16},
        "small": {"latent_channels": 32},
        "medium": {"latent_channels": 32},
    }
    latent_ch = configs[variant]["latent_channels"]

    # Dummy input: [1, C_latent, 64, 64] for 1024x1024 output
    dummy_latent = torch.randn(1, latent_ch, 64, 64)

    print(f"Exporting SRA-{variant} decoder:")
    print(f"  Decoder params: {count_params(decoder):,} ({model_size_kb(decoder):.1f} KB)")
    print(f"  Input: [1, {latent_ch}, 64, 64]")

    with torch.no_grad():
        out = decoder(dummy_latent)
    print(f"  Output: {out.shape}")

    torch.onnx.export(
        decoder,
        dummy_latent,
        output_path,
        opset_version=17,
        input_names=["latent"],
        output_names=["residual"],
        dynamic_axes={
            "latent": {0: "batch", 2: "height", 3: "width"},
            "residual": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"  Saved: {output_path}")

    # File size
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ONNX size: {size_kb:.1f} KB")

    if quantize:
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quant_path = output_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(output_path, quant_path, weight_type=QuantType.QInt8)
            quant_size = os.path.getsize(quant_path) / 1024
            print(f"  INT8 quantized: {quant_path} ({quant_size:.1f} KB)")
        except ImportError:
            print("  Quantization requires: pip install onnxruntime onnx")

    # Benchmark metrics from checkpoint
    metrics = ckpt.get("metrics", {})
    if metrics:
        print(f"\n  Checkpoint metrics:")
        print(f"    BPP:  {metrics.get('val_bpp', '?'):.4f}")
        print(f"    PSNR: {metrics.get('val_psnr', '?'):.2f} dB")
        print(f"    Size: {metrics.get('val_kb', '?'):.1f} KB")

    baselines = ckpt.get("jpeg_baselines", {})
    if baselines and metrics:
        print(f"\n  vs JPEG baselines:")
        for q, jm in sorted(baselines.items(), key=lambda x: int(x[0])):
            savings = (1 - metrics["val_bpp"] / jm["bpp"]) * 100
            psnr_diff = metrics["val_psnr"] - jm["psnr"]
            print(f"    vs Q{q}: {savings:+.1f}% size, {psnr_diff:+.2f} dB PSNR")


def benchmark_decode(onnx_path: str, n_runs: int = 100):
    """Benchmark ONNX decode speed on CPU."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Benchmarking requires: pip install onnxruntime")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = [1, sess.get_inputs()[0].shape[1], 64, 64]

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        sess.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    print(f"\nDecode benchmark ({n_runs} runs, 1024x1024 output):")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(times, 99):.2f} ms")


if __name__ == "__main__":
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt")
    ap.add_argument("--output", required=True, help="Output ONNX path")
    ap.add_argument("--quantize", action="store_true", help="Also export INT8 quantized")
    ap.add_argument("--benchmark", action="store_true", help="Benchmark decode speed")
    args = ap.parse_args()

    export_decoder(args.checkpoint, args.output, args.quantize)

    if args.benchmark:
        benchmark_decode(args.output)
