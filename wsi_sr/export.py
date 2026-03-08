#!/usr/bin/env python3
"""
Export trained model: collapse branches → ONNX → optional INT8 quantization.

Usage:
  python export.py --checkpoint checkpoints/best.pt --output model_collapsed.onnx
  python export.py --checkpoint checkpoints/best.pt --output model_int8.onnx --quantize
"""

import argparse
import time

import torch
import torch.nn as nn

from model import WSISRX4, WSIEnhanceNet, collapse_model, count_params, model_size_kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Training checkpoint .pt")
    ap.add_argument("--output", default="model.onnx", help="Output ONNX path")
    ap.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    ap.add_argument("--benchmark", action="store_true", help="Run CPU inference benchmark")
    args = ap.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    mode = config["mode"]
    channels = config["channels"]
    blocks = config["blocks"]

    if mode == "sr":
        model = WSISRX4(channels=channels, num_blocks=blocks)
        input_shape = (1, 3, 256, 256)
    else:
        model = WSIEnhanceNet(channels=channels, num_blocks=blocks)
        input_shape = (1, 3, 1024, 1024)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Mode: {mode}")
    print(f"Training model: {count_params(model):,} params, {model_size_kb(model):.1f} KB")

    # Collapse
    collapsed = collapse_model(model)
    print(f"Collapsed model: {count_params(collapsed):,} params, {model_size_kb(collapsed):.1f} KB")

    # Verify collapse
    x = torch.randn(*input_shape)
    with torch.no_grad():
        y1 = model(x)
        y2 = collapsed(x)
        diff = (y1 - y2).abs().max().item()
    print(f"Collapse verification — max diff: {diff:.2e}")

    # Export to ONNX
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        collapsed,
        x,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"},
                      "output": {2: "out_height", 3: "out_width"}},
        opset_version=17,
    )
    import os
    onnx_size = os.path.getsize(args.output)
    print(f"ONNX file: {onnx_size / 1024:.1f} KB")

    # Optional INT8 quantization
    if args.quantize:
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = args.output.replace(".onnx", "_int8.onnx")
            quantize_dynamic(args.output, quant_path, weight_type=QuantType.QInt8)
            quant_size = os.path.getsize(quant_path)
            print(f"INT8 quantized: {quant_size / 1024:.1f} KB → {quant_path}")
        except ImportError:
            print("Install onnxruntime to enable INT8 quantization: pip install onnxruntime")

    # Optional benchmark
    if args.benchmark:
        print("\nCPU inference benchmark (collapsed model, float32)...")
        collapsed = collapsed.cpu()
        x = torch.randn(*input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                collapsed(x)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                collapsed(x)
                times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        print(f"  Input: {input_shape}")
        print(f"  P50: {p50:.1f} ms")
        print(f"  P95: {p95:.1f} ms")
        print(f"  Mean: {sum(times)/len(times):.1f} ms")


if __name__ == "__main__":
    main()
