#!/usr/bin/env python3
"""
Export trained model: collapse branches → ONNX → optional INT8 quantization.

Auto-detects architecture from checkpoint state dict keys:
  - y_head.* → WSISRX4Dual
  - conv1.* (no body.*) → ESPCNR
  - head.*/body.* → WSISRX4

Usage:
  python export.py --checkpoint models/dual_best.pt --output models/dual.onnx --quantize
  python export.py --checkpoint models/wsisrx4_v2_best.pt --output models/wsisrx4.onnx --quantize
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from model import (WSISRX4, WSISRX4Dual, ESPCNR, WSIEnhanceNet,
                    collapse_model, count_params, model_size_kb)


def detect_architecture(state_dict, config):
    """Auto-detect model architecture from state dict keys."""
    keys = set(state_dict.keys())
    if any(k.startswith("y_head.") for k in keys):
        return "dual"
    if any(k.startswith("conv1.") for k in keys) and not any(k.startswith("body.") for k in keys):
        return "espcnr"
    return "wsisrx4"


def create_model(arch, config):
    """Instantiate model from architecture name and config."""
    channels = config.get("channels", 16)
    blocks = config.get("blocks", 5)

    if arch == "dual":
        return WSISRX4Dual(y_channels=channels, y_blocks=blocks)
    elif arch == "espcnr":
        return ESPCNR(upscale_factor=4)
    elif config.get("mode") == "enhance":
        return WSIEnhanceNet(channels=channels, num_blocks=blocks)
    else:
        return WSISRX4(channels=channels, num_blocks=blocks)


def embed_external_data(onnx_path):
    """If ONNX export created an external .data file, embed it into a single file."""
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        import onnx
        model = onnx.load(onnx_path)
        onnx.save_model(model, onnx_path, save_as_external_data=False)
        os.remove(data_path)
        print(f"  Embedded external data into single file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Training checkpoint .pt")
    ap.add_argument("--output", default="model.onnx", help="Output ONNX path")
    ap.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    ap.add_argument("--benchmark", action="store_true", help="Run CPU inference benchmark")
    args = ap.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = ckpt["model"]

    # Auto-detect architecture
    arch = detect_architecture(state_dict, config)
    model = create_model(arch, config)
    input_shape = (1, 3, 256, 256) if config.get("mode") != "enhance" else (1, 3, 1024, 1024)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Architecture: {arch} ({model.__class__.__name__})")
    print(f"Training model: {count_params(model):,} params, {model_size_kb(model):.1f} KB")

    # Collapse (WSISRX4/Dual have collapsible blocks; ESPCNR does not)
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

    # Embed external data if present (new PyTorch creates .data files)
    embed_external_data(args.output)

    onnx_size = os.path.getsize(args.output)
    print(f"ONNX file: {onnx_size / 1024:.1f} KB")

    # Optional INT8 quantization
    if args.quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = args.output.replace(".onnx", "_int8.onnx")
            quantize_dynamic(args.output, quant_path, weight_type=QuantType.QInt8)
            quant_size = os.path.getsize(quant_path)
            print(f"INT8 quantized: {quant_size / 1024:.1f} KB → {quant_path}")
        except ImportError:
            print("Install onnxruntime to enable INT8 quantization: pip install onnxruntime")

    # Optional benchmark
    if args.benchmark:
        print(f"\nCPU inference benchmark ({arch}, collapsed, float32)...")
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
