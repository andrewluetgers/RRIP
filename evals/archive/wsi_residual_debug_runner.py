#!/usr/bin/env python3
"""
wsi_residual_debug_runner.py

Debug runner that uses the production wsi_residual_tool.py code with debug instrumentation.
This script focuses ONLY on capturing intermediate states during encoding.
Analysis (metrics, manifests, PAC files) should be done separately.

Usage:
  python wsi_residual_debug_runner.py --image paper/L0-1024.jpg --tile 256 --resq 75

This script:
1. Creates a temporary pyramid from the input image
2. Runs the production encode() function with debug callbacks
3. Saves all intermediate images and data for later analysis
"""

import argparse
import pathlib
import numpy as np
from PIL import Image
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import pickle

# Import production code
from wsi_residual_tool import (
    encode,
    DebugContext,
    rgb_to_ycbcr_bt601,
    ycbcr_to_rgb_bt601,
    load_rgb,
    save_gray_jpeg
)


class DebugCapture:
    """Captures intermediate states during production encoding for later analysis."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata about the capture session
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {},
            "capture_events": []
        }

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "arrays").mkdir(exist_ok=True)
        (self.output_dir / "residuals").mkdir(exist_ok=True)

    def save_array(self, arr: np.ndarray, name: str, as_image: bool = True) -> Dict[str, Any]:
        """Save a numpy array, optionally as an image."""
        # Save raw array data for analysis
        array_path = self.output_dir / "arrays" / f"{name}.npy"
        np.save(array_path, arr)

        info = {
            "name": name,
            "array_path": str(array_path),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "range": [float(arr.min()), float(arr.max())],
            "mean": float(arr.mean()),
            "std": float(arr.std())
        }

        # Also save as image for quick visualization
        if as_image:
            if arr.ndim == 3 and arr.shape[2] == 3:
                # RGB image
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            else:
                # Grayscale - save both raw and normalized versions
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

                # Save normalized version for residuals
                if "residual" in name.lower():
                    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255)
                    img_norm = Image.fromarray(arr_norm.astype(np.uint8), mode="L")
                    img_norm.save(self.output_dir / "images" / f"{name}_normalized.png")

            image_path = self.output_dir / "images" / f"{name}.png"
            img.save(image_path)
            info["image_path"] = str(image_path)

        return info

    def setup_callbacks(self, debug_ctx: DebugContext):
        """Register all debug callbacks to capture intermediate states."""

        def on_init(**kwargs):
            self.metadata["configuration"] = {
                "tile_size": kwargs.get('tile_size'),
                "quantization_levels": kwargs.get('resq'),
                "L0": kwargs.get('L0'),
                "L1": kwargs.get('L1'),
                "L2": kwargs.get('L2')
            }

        def on_l2_loaded(**kwargs):
            x2, y2, tile = kwargs['x2'], kwargs['y2'], kwargs['tile']

            # Save the tile and its components
            self.save_array(tile, f"L2_{x2}_{y2}_original")

            # Save YCbCr channels
            Y, Cb, Cr = rgb_to_ycbcr_bt601(tile)
            self.save_array(Y, f"L2_{x2}_{y2}_luma")
            self.save_array(Cb, f"L2_{x2}_{y2}_chroma_cb")
            self.save_array(Cr, f"L2_{x2}_{y2}_chroma_cr")

            self.metadata["capture_events"].append({
                "event": "l2_loaded",
                "x2": x2, "y2": y2
            })

        def on_l1_prediction_mosaic(**kwargs):
            x2, y2, pred = kwargs['x2'], kwargs['y2'], kwargs['prediction']
            self.save_array(pred, f"L1_mosaic_{x2}_{y2}_prediction")

        def on_l1_tile_start(**kwargs):
            x1, y1, gt, pred = kwargs['x1'], kwargs['y1'], kwargs['ground_truth'], kwargs['prediction']
            # Just capture the data
            self.save_array(gt, f"L1_{x1}_{y1}_original")
            self.save_array(pred, f"L1_{x1}_{y1}_prediction")

        def on_l1_ycbcr(**kwargs):
            x1, y1 = kwargs['x1'], kwargs['y1']
            self.save_array(kwargs['Y_gt'], f"L1_{x1}_{y1}_Y_gt", as_image=False)
            self.save_array(kwargs['Y_pred'], f"L1_{x1}_{y1}_Y_pred", as_image=False)
            self.save_array(kwargs['Cb_pred'], f"L1_{x1}_{y1}_Cb_pred", as_image=False)
            self.save_array(kwargs['Cr_pred'], f"L1_{x1}_{y1}_Cr_pred", as_image=False)

        def on_l1_residual_raw(**kwargs):
            x1, y1, residual = kwargs['x1'], kwargs['y1'], kwargs['residual']
            self.save_array(residual, f"L1_{x1}_{y1}_residual_raw")

        def on_l1_residual_encoded(**kwargs):
            x1, y1, encoded = kwargs['x1'], kwargs['y1'], kwargs['encoded']
            self.save_array(encoded, f"L1_{x1}_{y1}_residual_encoded")

        def on_l1_reconstructed(**kwargs):
            x1, y1 = kwargs['x1'], kwargs['y1']
            self.save_array(kwargs['Y_recon'], f"L1_{x1}_{y1}_Y_recon", as_image=False)
            self.save_array(kwargs['rgb_recon'], f"L1_{x1}_{y1}_reconstructed")

        def on_l1_mosaic_complete(**kwargs):
            x2, y2, mosaic = kwargs['x2'], kwargs['y2'], kwargs['mosaic']
            self.save_array(mosaic, f"L1_mosaic_{x2}_{y2}_complete")

        def on_l0_prediction_mosaic(**kwargs):
            x2, y2, pred = kwargs['x2'], kwargs['y2'], kwargs['prediction']
            self.save_array(pred, f"L0_mosaic_{x2}_{y2}_prediction")

        def on_l0_tile_start(**kwargs):
            x0, y0, gt, pred = kwargs['x0'], kwargs['y0'], kwargs['ground_truth'], kwargs['prediction']
            self.save_array(gt, f"L0_{x0}_{y0}_original")
            self.save_array(pred, f"L0_{x0}_{y0}_prediction")

        def on_l0_ycbcr(**kwargs):
            x0, y0 = kwargs['x0'], kwargs['y0']
            self.save_array(kwargs['Y_gt'], f"L0_{x0}_{y0}_Y_gt", as_image=False)
            self.save_array(kwargs['Y_pred'], f"L0_{x0}_{y0}_Y_pred", as_image=False)

        def on_l0_residual_raw(**kwargs):
            x0, y0, residual = kwargs['x0'], kwargs['y0'], kwargs['residual']
            self.save_array(residual, f"L0_{x0}_{y0}_residual_raw")

        def on_l0_residual_encoded(**kwargs):
            x0, y0, encoded = kwargs['x0'], kwargs['y0'], kwargs['encoded']
            self.save_array(encoded, f"L0_{x0}_{y0}_residual_encoded")

        def on_encoding_complete(**kwargs):
            self.metadata["statistics"] = kwargs['summary']

        # Register all callbacks
        debug_ctx.register_callback('init', on_init)
        debug_ctx.register_callback('l2_loaded', on_l2_loaded)
        debug_ctx.register_callback('l1_prediction_mosaic', on_l1_prediction_mosaic)
        debug_ctx.register_callback('l1_tile_start', on_l1_tile_start)
        debug_ctx.register_callback('l1_ycbcr', on_l1_ycbcr)
        debug_ctx.register_callback('l1_residual_raw', on_l1_residual_raw)
        debug_ctx.register_callback('l1_residual_encoded', on_l1_residual_encoded)
        debug_ctx.register_callback('l1_reconstructed', on_l1_reconstructed)
        debug_ctx.register_callback('l1_mosaic_complete', on_l1_mosaic_complete)
        debug_ctx.register_callback('l0_prediction_mosaic', on_l0_prediction_mosaic)
        debug_ctx.register_callback('l0_tile_start', on_l0_tile_start)
        debug_ctx.register_callback('l0_ycbcr', on_l0_ycbcr)
        debug_ctx.register_callback('l0_residual_raw', on_l0_residual_raw)
        debug_ctx.register_callback('l0_residual_encoded', on_l0_residual_encoded)
        debug_ctx.register_callback('encoding_complete', on_encoding_complete)

    def save_metadata(self):
        """Save the capture metadata."""
        metadata_path = self.output_dir / "capture_metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2))
        return metadata_path


def create_pyramid_from_image(image_path: pathlib.Path, output_dir: pathlib.Path, tile_size: int = 256):
    """Create a simple pyramid structure from a single image for testing."""
    img = Image.open(image_path)
    if img.size != (1024, 1024):
        raise ValueError(f"Expected 1024x1024 image, got {img.size}")

    # Create pyramid structure
    pyramid_dir = output_dir / "test_pyramid_files"
    pyramid_dir.mkdir(parents=True, exist_ok=True)

    # L0 level (4x4 tiles of 256x256)
    l0_dir = pyramid_dir / "12"  # Assuming level 12 is L0
    l0_dir.mkdir(exist_ok=True)
    for y in range(4):
        for x in range(4):
            tile = img.crop((x*256, y*256, (x+1)*256, (y+1)*256))
            tile.save(l0_dir / f"{x}_{y}.jpg", quality=95)

    # L1 level (2x2 tiles of 256x256, each from 512x512 region)
    l1_dir = pyramid_dir / "11"
    l1_dir.mkdir(exist_ok=True)
    for y in range(2):
        for x in range(2):
            region = img.crop((x*512, y*512, (x+1)*512, (y+1)*512))
            tile = region.resize((256, 256), Image.Resampling.LANCZOS)
            tile.save(l1_dir / f"{x}_{y}.jpg", quality=95)

    # L2 level (1 tile of 256x256 from entire image)
    l2_dir = pyramid_dir / "10"
    l2_dir.mkdir(exist_ok=True)
    tile = img.resize((256, 256), Image.Resampling.LANCZOS)
    tile.save(l2_dir / "0_0.jpg", quality=95)

    # Create DZI file
    dzi_path = output_dir / "test_pyramid.dzi"
    dzi_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="0"
  TileSize="{tile_size}">
  <Size Width="1024" Height="1024"/>
</Image>"""
    dzi_path.write_text(dzi_content)

    return output_dir / "test_pyramid"


def main():
    parser = argparse.ArgumentParser(description="Debug capture for ORIGAMI compression")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--out", help="Output directory (auto-generated if not specified)")
    parser.add_argument("--tile", type=int, default=256, help="Tile size")
    parser.add_argument("--resq", type=int, default=75, help="JPEG quality for residuals")

    args = parser.parse_args()

    # Generate output directory name if not specified
    if args.out is None:
        image_path = pathlib.Path(args.image)
        image_name = image_path.stem.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_parts = ["debug", image_name, f"j{args.resq}", timestamp]
        args.out = "paper/" + "_".join(dir_parts)
        print(f"Output directory: {args.out}")

    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for pyramid
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create pyramid from input image
        print(f"Creating pyramid from {args.image}...")
        pyramid_prefix = create_pyramid_from_image(
            pathlib.Path(args.image),
            temp_path,
            args.tile
        )

        # Setup debug capture
        capture = DebugCapture(output_dir)

        # Add input parameters to metadata
        capture.metadata["input"] = {
            "image": str(args.image),
            "tile_size": args.tile,
            "jpeg_quality": args.resq
        }

        # Create debug context
        debug_ctx = DebugContext(enabled=True, output_dir=output_dir)

        # Register callbacks
        capture.setup_callbacks(debug_ctx)

        # Run production encode with debug instrumentation
        print("Running production encode with debug capture...")
        encode(
            pyramid_prefix=pyramid_prefix,
            out_dir=output_dir,
            tile_size=args.tile,
            resq=args.resq,  # JPEG quality for residuals
            debug_ctx=debug_ctx
        )

        # Save capture metadata
        metadata_path = capture.save_metadata()

        print(f"\n✓ Debug capture complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Images: {output_dir}/images/")
        print(f"  Raw arrays: {output_dir}/arrays/")
        print(f"\nUse wsi_residual_analyze.py to generate metrics and manifests from this capture.")


if __name__ == "__main__":
    main()