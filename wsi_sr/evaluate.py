#!/usr/bin/env python3
"""
Evaluate WSI SR model against bilinear and lanczos3 baselines.

For each 1024x1024 ground-truth tile:
  1. Downsample 4x to 256x256 via Lanczos (optionally JPEG-compress)
  2. Reconstruct 1024x1024 via: SR model, bilinear upsample, lanczos3 upsample
  3. Compute full visual metrics vs ground truth
  4. Measure residual sizes (JPEG-compressed luma residual at various qualities)

Metrics computed:
  - PSNR (dB) — peak signal-to-noise ratio
  - MSE — mean squared error (lower = better)
  - SSIM — structural similarity index (higher = better)
  - MS-SSIM — multi-scale SSIM (higher = better, often better correlates with perception)
  - Delta E (CIE2000) — perceptual color difference (lower = better, <2 = imperceptible)
  - LPIPS (AlexNet) — learned perceptual similarity (lower = better)
  - VIF — visual information fidelity (higher = better)
  - SSIMULACRA2 — perceptual quality metric tuned for compression (higher = better)
  - Residual size — JPEG-compressed luma residual in bytes at q50 (our pipeline default)

Usage:
  python evaluate.py --tiles /path/to/1024_tiles --checkpoint checkpoints/best.pt
  python evaluate.py --tiles /path/to/1024_tiles --checkpoint best.pt --save-samples 10
  python evaluate.py --tiles /path/to/1024_tiles --checkpoint best.pt --no-jpeg
  python evaluate.py --tiles /path/to/1024_tiles --checkpoint best.pt --resq 40,50,60
"""

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model import WSISRX4, WSIEnhanceNet, collapse_model, count_params, model_size_kb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR in dB. Images are uint8 [0,255]."""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error. Images are uint8 [0,255]."""
    return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM via scikit-image. Images are uint8 HxWxC."""
    from skimage.metrics import structural_similarity
    return structural_similarity(target, pred, channel_axis=2, data_range=255)


def compute_ms_ssim(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Multi-scale SSIM. Requires images >= 176x176. Returns None if too small."""
    try:
        from pytorch_msssim import ms_ssim
        # uint8 HWC -> float BCHW [0,1]
        def to_t(img):
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        val = ms_ssim(to_t(pred), to_t(target), data_range=1.0)
        return float(val.item())
    except ImportError:
        return None
    except Exception:
        return None


def compute_delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean CIE2000 Delta E. Images are uint8 HxWxC (RGB).
    Matches compute_metrics.py: normalize to float64 [0,1] before rgb2lab."""
    from skimage.color import rgb2lab, deltaE_ciede2000
    lab_pred = rgb2lab(pred.astype(np.float64) / 255.0)
    lab_target = rgb2lab(target.astype(np.float64) / 255.0)
    de = deltaE_ciede2000(lab_target, lab_pred)
    return float(np.mean(de))


def compute_vif(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Visual Information Fidelity via sewar. Images are uint8 HxWxC."""
    try:
        from sewar.full_ref import vifp
        return float(vifp(target, pred))
    except ImportError:
        return None


def compute_ssimulacra2(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """SSIMULACRA2 score. Tries Python packages first, falls back to CLI binary.
    Score interpretation: >90 excellent, >70 good, >50 ok, <30 bad."""
    import shutil
    import subprocess
    import tempfile

    # Try Python packages first
    for pkg_name in ["ssimulacra2", "ssimulacra2_py"]:
        try:
            pkg = __import__(pkg_name)
            if hasattr(pkg, "compute"):
                return float(pkg.compute(Image.fromarray(target), Image.fromarray(pred)))
            elif hasattr(pkg, "ssimulacra2"):
                return float(pkg.ssimulacra2(target, pred))
        except (ImportError, Exception):
            continue

    # Fall back to CLI binary (matching compute_metrics.py approach)
    if shutil.which("ssimulacra2"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ref_path = os.path.join(tmp, "ref.png")
                test_path = os.path.join(tmp, "test.png")
                Image.fromarray(target).save(ref_path)
                Image.fromarray(pred).save(test_path)
                result = subprocess.run(
                    ["ssimulacra2", ref_path, test_path],
                    capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        line = line.strip()
                        try:
                            return float(line)
                        except ValueError:
                            # Try extracting number from line
                            import re
                            m = re.search(r"[-+]?\d*\.?\d+", line)
                            if m:
                                return float(m.group())
        except Exception:
            pass

    return None


class LPIPSMetric:
    """Lazy-loaded LPIPS wrapper."""

    def __init__(self, device: str):
        self.device = device
        self._net = None

    def _load(self):
        import lpips
        self._net = lpips.LPIPS(net="alex").to(self.device).eval()

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        if self._net is None:
            self._load()
        # uint8 HWC -> float BCHW in [-1, 1]
        def to_tensor(img):
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return t * 2.0 - 1.0
        with torch.no_grad():
            d = self._net(to_tensor(pred).to(self.device),
                          to_tensor(target).to(self.device))
        return float(d.item())


# ---------------------------------------------------------------------------
# Residual size measurement
# ---------------------------------------------------------------------------

def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to Y (luma) channel via BT.601. Returns uint8 HxW."""
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def compute_residual_size(pred: np.ndarray, target: np.ndarray,
                          jpeg_quality: int = 50) -> int:
    """Compute JPEG-compressed luma residual size in bytes.

    This mirrors the ORIGAMI residual pipeline:
    1. Compute luma of both prediction and ground truth
    2. residual = (gt_y - pred_y + 128), clamped to [0, 255]
    3. JPEG-compress the residual at the given quality
    4. Return the compressed size in bytes

    A better prediction → smaller residual → fewer bytes.
    """
    pred_y = rgb_to_y(pred)
    gt_y = rgb_to_y(target)

    # Compute centered residual (matching ORIGAMI's residual.rs)
    residual = gt_y.astype(np.int16) - pred_y.astype(np.int16) + 128
    residual = np.clip(residual, 0, 255).astype(np.uint8)

    # JPEG compress and measure size
    residual_img = Image.fromarray(residual, mode='L')
    buf = io.BytesIO()
    residual_img.save(buf, format='JPEG', quality=jpeg_quality)
    return buf.tell()


def compute_residual_deviation(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute per-pixel deviation statistics for the luma residual.

    Catches outlier pixels that compress well in JPEG but represent
    large local errors — important for pathology where a few wrong
    pixels on a cell boundary could mislead diagnosis.
    """
    pred_y = rgb_to_y(pred).astype(np.float32)
    gt_y = rgb_to_y(target).astype(np.float32)
    abs_dev = np.abs(gt_y - pred_y)
    n_pixels = abs_dev.size

    return {
        "max_dev": float(np.max(abs_dev)),
        "p99_dev": float(np.percentile(abs_dev, 99)),
        "p95_dev": float(np.percentile(abs_dev, 95)),
        "mean_dev": float(np.mean(abs_dev)),
        "pct_over_10": float(np.sum(abs_dev > 10) / n_pixels * 100),
        "pct_over_20": float(np.sum(abs_dev > 20) / n_pixels * 100),
        "pct_over_30": float(np.sum(abs_dev > 30) / n_pixels * 100),
    }


def compute_residual_sizes(pred: np.ndarray, target: np.ndarray,
                           qualities: List[int]) -> Dict[str, float]:
    """Compute residual sizes at multiple JPEG quality levels + deviation stats."""
    result = {}
    for q in qualities:
        result[f"res_q{q}"] = compute_residual_size(pred, target, q)
    # Add deviation stats
    result.update(compute_residual_deviation(pred, target))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_tiles(tile_dir: str) -> List[str]:
    paths = sorted([
        os.path.join(tile_dir, f)
        for f in os.listdir(tile_dir)
        if f.lower().endswith(EXTS)
    ])
    if not paths:
        raise ValueError(f"No images found in {tile_dir}")
    return paths


def downsample_4x(img: Image.Image, jpeg_quality: Optional[int]) -> Image.Image:
    """Downsample 4x via Lanczos, optionally JPEG compress."""
    w, h = img.size
    small = img.resize((w // 4, h // 4), Image.LANCZOS)
    if jpeg_quality is not None:
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
        buf.seek(0)
        small = Image.open(buf).convert("RGB")
    return small


def upsample_pil(small: Image.Image, size: tuple, method) -> np.ndarray:
    """Upsample PIL image and return uint8 numpy array."""
    return np.array(small.resize(size, method))


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """BCHW float [0,1] tensor -> HWC uint8 numpy."""
    img = t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


@dataclass
class TileResult:
    name: str
    sr: Dict[str, float] = field(default_factory=dict)
    bilinear: Dict[str, float] = field(default_factory=dict)
    lanczos3: Dict[str, float] = field(default_factory=dict)


def format_value(metric: str, v: float) -> str:
    """Format a metric value for display."""
    if "res_q" in metric:
        return f"{v / 1024:.1f}K"
    elif metric == "MSE":
        return f"{v:.2f}"
    elif metric in ("PSNR", "DeltaE"):
        return f"{v:.2f}"
    elif metric in ("SSIM", "MS-SSIM", "VIF"):
        return f"{v:.4f}"
    elif metric == "LPIPS":
        return f"{v:.4f}"
    elif metric == "SSIMUL2":
        return f"{v:.1f}"
    elif metric in ("max_dev", "p99_dev", "p95_dev", "mean_dev"):
        return f"{v:.1f}"
    elif metric.startswith("pct_over_"):
        return f"{v:.2f}%"
    else:
        return f"{v:.4f}"


def higher_is_better(metric: str) -> bool:
    """Return True if higher values are better for this metric."""
    return metric in ("PSNR", "SSIM", "MS-SSIM", "VIF", "SSIMUL2")


def print_table(results: List[TileResult], metric_names: List[str]):
    """Print a formatted results table."""
    methods = ["bilinear", "lanczos3", "sr"]
    labels = {"bilinear": "Bilinear", "lanczos3": "Lanczos3", "sr": "SR Model"}

    # Aggregate stats
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)

    for m in methods:
        print(f"\n  {labels[m]}:")
        for metric in metric_names:
            vals = [getattr(r, m).get(metric, float("nan")) for r in results]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            arr = np.array(vals)
            print(f"    {metric:<10}  mean={format_value(metric, np.mean(arr)):>8}  "
                  f"median={format_value(metric, np.median(arr)):>8}  "
                  f"std={format_value(metric, np.std(arr)):>8}  "
                  f"min={format_value(metric, np.min(arr)):>8}  "
                  f"max={format_value(metric, np.max(arr)):>8}")

    # Summary comparison table
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON (mean values)")
    print("=" * 100)

    # Column widths
    mw = 10  # metric name
    vw = 12  # value column

    print(f"  {'Metric':<{mw}}", end="")
    for m in methods:
        print(f"  {labels[m]:>{vw}}", end="")
    print(f"  {'SR vs Bilin':>{vw}}  {'SR vs Lanc':>{vw}}  {'Direction':>10}")
    print("  " + "-" * (mw + (vw + 2) * 5 + 12))

    for metric in metric_names:
        vals_by_method = {}
        for m in methods:
            v = [getattr(r, m).get(metric, float("nan")) for r in results]
            v = [x for x in v if not np.isnan(x)]
            vals_by_method[m] = np.mean(v) if v else float("nan")

        print(f"  {metric:<{mw}}", end="")
        for m in methods:
            print(f"  {format_value(metric, vals_by_method[m]):>{vw}}", end="")

        # Deltas vs baselines
        sr_val = vals_by_method["sr"]
        for baseline in ["bilinear", "lanczos3"]:
            delta = sr_val - vals_by_method[baseline]
            sign = "+" if delta >= 0 else ""
            print(f"  {sign}{format_value(metric, delta):>{vw-1}}", end="")

        # Direction indicator
        hib = higher_is_better(metric)
        sr_vs_lanc = sr_val - vals_by_method["lanczos3"]
        if hib:
            indicator = "BETTER" if sr_vs_lanc > 0 else "worse"
        else:
            indicator = "BETTER" if sr_vs_lanc < 0 else "worse"
        print(f"  {indicator:>10}")

    print()


def save_comparison(gt: np.ndarray, bilinear: np.ndarray, lanczos: np.ndarray,
                    sr: np.ndarray, path: str, crop_size: int = 256):
    """Save a side-by-side comparison image with a detail crop."""
    h, w = gt.shape[:2]
    cy, cx = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    crops = [
        img[cy:cy + crop_size, cx:cx + crop_size]
        for img in [gt, bilinear, lanczos, sr]
    ]

    # Top row: full images resized to 512x512
    thumbs = [
        np.array(Image.fromarray(img).resize((512, 512), Image.LANCZOS))
        for img in [gt, bilinear, lanczos, sr]
    ]
    top = np.concatenate(thumbs, axis=1)

    # Bottom row: detail crops scaled up 2x
    crop_views = [
        np.array(Image.fromarray(c).resize((crop_size * 2, crop_size * 2), Image.NEAREST))
        for c in crops
    ]
    bottom = np.concatenate(crop_views, axis=1)

    # Match widths
    if bottom.shape[1] < top.shape[1]:
        pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
        bottom = np.concatenate([bottom, pad], axis=1)
    elif bottom.shape[1] > top.shape[1]:
        pad = np.zeros((top.shape[0], bottom.shape[1] - top.shape[1], 3), dtype=np.uint8)
        top = np.concatenate([top, pad], axis=1)

    combined = np.concatenate([top, bottom], axis=0)
    Image.fromarray(combined).save(path, quality=95)


def save_residual_comparison(gt: np.ndarray, bilinear: np.ndarray, lanczos: np.ndarray,
                             sr: np.ndarray, path: str):
    """Save residual visualizations side-by-side (amplified for visibility)."""
    gt_y = rgb_to_y(gt)

    residuals = []
    for recon in [bilinear, lanczos, sr]:
        pred_y = rgb_to_y(recon)
        # Centered residual
        res = gt_y.astype(np.int16) - pred_y.astype(np.int16) + 128
        res = np.clip(res, 0, 255).astype(np.uint8)
        # Amplify for visibility: stretch [96, 160] to [0, 255]
        amplified = np.clip((res.astype(np.float32) - 96) * (255.0 / 64.0), 0, 255).astype(np.uint8)
        # Convert to RGB for display
        residuals.append(np.stack([amplified] * 3, axis=2))

    row = np.concatenate(residuals, axis=1)
    Image.fromarray(row).save(path, quality=95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate WSI SR model vs bilinear/lanczos3 baselines")
    ap.add_argument("--tiles", required=True,
                    help="Directory of 1024x1024 ground-truth tiles")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt",
                    help="Model checkpoint path (default: checkpoints/best.pt)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG quality for simulated base (default: 95)")
    ap.add_argument("--no-jpeg", action="store_true",
                    help="Skip JPEG compression on downsampled input")
    ap.add_argument("--device", default=None,
                    help="Device: cuda or cpu (default: auto-detect)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size for SR model inference (default: 4)")
    ap.add_argument("--save-samples", type=int, default=0,
                    help="Save N comparison images (default: 0)")
    ap.add_argument("--sample-dir", default="eval_samples",
                    help="Directory for comparison images (default: eval_samples)")
    ap.add_argument("--no-lpips", action="store_true",
                    help="Skip LPIPS computation")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Limit number of tiles to evaluate")
    ap.add_argument("--resq", default="60,80,90",
                    help="Residual JPEG quality levels to measure, comma-separated (default: 60,80,90)")
    ap.add_argument("--json", default=None,
                    help="Save full results as JSON to this path")
    args = ap.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    jpeg_quality = None if args.no_jpeg else args.jpeg_quality
    res_qualities = [int(q.strip()) for q in args.resq.split(",")]

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    mode = config["mode"]
    channels = config["channels"]
    blocks = config["blocks"]

    if mode == "sr":
        model = WSISRX4(channels=channels, num_blocks=blocks)
    else:
        model = WSIEnhanceNet(channels=channels, num_blocks=blocks)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Collapse for fast inference
    collapsed = collapse_model(model)
    collapsed = collapsed.to(device).eval()

    print(f"Model: {mode} mode, {count_params(collapsed):,} params, "
          f"{model_size_kb(collapsed):.1f} KB (collapsed)")
    print(f"Device: {device}")
    print(f"JPEG quality: {jpeg_quality if jpeg_quality else 'none'}")
    print(f"Residual quality levels: {res_qualities}")

    # --- Discover tiles ---
    tile_paths = list_tiles(args.tiles)
    if args.max_tiles:
        tile_paths = tile_paths[:args.max_tiles]
    print(f"Tiles: {len(tile_paths)}")

    # --- Probe available metrics ---
    lpips_metric = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_metric = LPIPSMetric(device)
            print("LPIPS: enabled (alex)")
        except ImportError:
            print("LPIPS: skipped (pip install lpips)")

    has_vif = False
    try:
        from sewar.full_ref import vifp
        has_vif = True
        print("VIF: enabled")
    except ImportError:
        print("VIF: skipped (pip install sewar)")

    has_ms_ssim = False
    try:
        from pytorch_msssim import ms_ssim
        has_ms_ssim = True
        print("MS-SSIM: enabled")
    except ImportError:
        print("MS-SSIM: skipped (pip install pytorch-msssim)")

    has_ssimulacra2 = False
    try:
        import ssimulacra2
        has_ssimulacra2 = True
        print("SSIMULACRA2: enabled")
    except ImportError:
        try:
            import ssimulacra2_py
            has_ssimulacra2 = True
            print("SSIMULACRA2: enabled (ssimulacra2_py)")
        except ImportError:
            print("SSIMULACRA2: skipped (pip install ssimulacra2 or ssimulacra2-py)")

    # --- Prepare sample saving ---
    if args.save_samples > 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # --- Build metric computation function ---
    def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        d = {}
        d["PSNR"] = compute_psnr(pred, target)
        d["MSE"] = compute_mse(pred, target)
        d["SSIM"] = compute_ssim(pred, target)
        d["DeltaE"] = compute_delta_e(pred, target)

        if has_ms_ssim:
            val = compute_ms_ssim(pred, target)
            if val is not None:
                d["MS-SSIM"] = val

        if has_vif:
            val = compute_vif(pred, target)
            if val is not None:
                d["VIF"] = val

        if has_ssimulacra2:
            val = compute_ssimulacra2(pred, target)
            if val is not None:
                d["SSIMUL2"] = val

        if lpips_metric is not None:
            d["LPIPS"] = lpips_metric(pred, target)

        # Residual sizes
        res_sizes = compute_residual_sizes(pred, target, res_qualities)
        d.update(res_sizes)

        return d

    # --- Evaluate ---
    results: List[TileResult] = []
    sr_times: List[float] = []

    batch_inputs: List[torch.Tensor] = []
    batch_gt: List[np.ndarray] = []
    batch_small: List[Image.Image] = []
    batch_names: List[str] = []

    def flush_batch():
        nonlocal batch_inputs, batch_gt, batch_small, batch_names
        if not batch_inputs:
            return

        input_tensor = torch.stack(batch_inputs).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            sr_output = collapsed(input_tensor).clamp(0, 1)
        elapsed = time.perf_counter() - t0
        sr_times.append(elapsed / len(batch_inputs))

        for i in range(len(batch_inputs)):
            name = batch_names[i]
            gt = batch_gt[i]
            small = batch_small[i]
            sr_img = tensor_to_uint8(sr_output[i:i+1])
            target_size = (gt.shape[1], gt.shape[0])  # (W, H)

            # Baselines
            bilinear_img = upsample_pil(small, target_size, Image.BILINEAR)
            lanczos_img = upsample_pil(small, target_size, Image.LANCZOS)

            # Compute all metrics for each method
            result = TileResult(name=name)
            for method_name, recon in [("sr", sr_img),
                                        ("bilinear", bilinear_img),
                                        ("lanczos3", lanczos_img)]:
                result_dict = compute_all_metrics(recon, gt)
                setattr(result, method_name, result_dict)

            results.append(result)

            # Save comparison images
            if args.save_samples > 0 and len(results) <= args.save_samples:
                out_path = os.path.join(args.sample_dir, f"{name}_compare.jpg")
                save_comparison(gt, bilinear_img, lanczos_img, sr_img, out_path)
                res_path = os.path.join(args.sample_dir, f"{name}_residuals.jpg")
                save_residual_comparison(gt, bilinear_img, lanczos_img, sr_img, res_path)
                print(f"  Saved: {out_path}")

        batch_inputs.clear()
        batch_gt.clear()
        batch_small.clear()
        batch_names.clear()

    print(f"\nEvaluating {len(tile_paths)} tiles...")
    print("-" * 80)

    for idx, path in enumerate(tile_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        gt_pil = Image.open(path).convert("RGB")
        gt = np.array(gt_pil)

        small = downsample_4x(gt_pil, jpeg_quality)

        if mode == "sr":
            input_tensor = TF.to_tensor(small)
        else:
            upsampled = small.resize(gt_pil.size, Image.LANCZOS)
            input_tensor = TF.to_tensor(upsampled)

        batch_inputs.append(input_tensor)
        batch_gt.append(gt)
        batch_small.append(small)
        batch_names.append(name)

        if len(batch_inputs) >= args.batch_size:
            flush_batch()
            r = results[-1]
            res_key = f"res_q{res_qualities[0]}"
            sr_res = r.sr.get(res_key, 0) / 1024
            bl_res = r.bilinear.get(res_key, 0) / 1024
            lc_res = r.lanczos3.get(res_key, 0) / 1024
            print(f"  [{idx+1}/{len(tile_paths)}] {r.name}  "
                  f"SR: PSNR={r.sr['PSNR']:.2f} SSIM={r.sr['SSIM']:.4f} res={sr_res:.1f}K  "
                  f"Bilinear: PSNR={r.bilinear['PSNR']:.2f} res={bl_res:.1f}K  "
                  f"Lanczos3: PSNR={r.lanczos3['PSNR']:.2f} res={lc_res:.1f}K")

    # Flush remaining
    flush_batch()

    if not results:
        print("No results. Check that tile directory contains images.")
        return

    # Determine which metrics were computed
    metric_names = ["PSNR", "MSE", "SSIM"]
    if has_ms_ssim and "MS-SSIM" in results[0].sr:
        metric_names.append("MS-SSIM")
    metric_names.append("DeltaE")
    if has_vif and "VIF" in results[0].sr:
        metric_names.append("VIF")
    if has_ssimulacra2 and "SSIMUL2" in results[0].sr:
        metric_names.append("SSIMUL2")
    if lpips_metric is not None and "LPIPS" in results[0].sr:
        metric_names.append("LPIPS")
    # Add residual size metrics
    for q in res_qualities:
        metric_names.append(f"res_q{q}")
    # Add deviation stats
    metric_names.extend(["max_dev", "p99_dev", "p95_dev", "mean_dev",
                         "pct_over_10", "pct_over_20", "pct_over_30"])

    print_table(results, metric_names)

    # Timing stats
    if sr_times:
        arr = np.array(sr_times) * 1000
        print(f"SR Model Inference: mean={np.mean(arr):.1f} ms/tile, "
              f"median={np.median(arr):.1f} ms/tile ({device})")

    if args.save_samples > 0:
        print(f"\nComparison images saved to: {args.sample_dir}/")
        print("  *_compare.jpg: GT | Bilinear | Lanczos3 | SR (top: overview, bottom: center crop)")
        print("  *_residuals.jpg: Bilinear | Lanczos3 | SR luma residuals (amplified)")

    # Save JSON results
    if args.json:
        json_data = {
            "config": {
                "checkpoint": args.checkpoint,
                "mode": mode,
                "channels": channels,
                "blocks": blocks,
                "params": count_params(collapsed),
                "model_kb": model_size_kb(collapsed),
                "jpeg_quality": jpeg_quality,
                "res_qualities": res_qualities,
                "device": device,
                "n_tiles": len(results),
            },
            "per_tile": [
                {
                    "name": r.name,
                    "sr": r.sr,
                    "bilinear": r.bilinear,
                    "lanczos3": r.lanczos3,
                }
                for r in results
            ],
            "aggregate": {},
        }
        methods = ["bilinear", "lanczos3", "sr"]
        for m in methods:
            agg = {}
            for metric in metric_names:
                vals = [getattr(r, m).get(metric, float("nan")) for r in results]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    agg[metric] = {
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }
            json_data["aggregate"][m] = agg

        # Per-tile difficulty ranking — ALL tiles, sorted by difficulty (worst first)
        # Composite difficulty score: weighted combination of deviation and residual size
        res_key = f"res_q{res_qualities[-1]}"
        for r in results:
            max_d = r.sr.get("max_dev", 0)
            p99_d = r.sr.get("p99_dev", 0)
            res_sz = r.sr.get(res_key, 0)
            # Normalize each dimension and combine
            r.sr["_difficulty"] = max_d + p99_d * 2 + res_sz / 1024

        ranked = sorted(results, key=lambda r: r.sr.get("_difficulty", 0), reverse=True)
        json_data["tile_difficulty_ranking"] = [
            {
                "rank": i + 1,
                "name": r.name,
                "difficulty": round(r.sr["_difficulty"], 2),
                "max_dev": r.sr.get("max_dev", 0),
                "p99_dev": r.sr.get("p99_dev", 0),
                "pct_over_20": r.sr.get("pct_over_20", 0),
                "res_size": r.sr.get(res_key, 0),
                "psnr": r.sr.get("PSNR", 0),
                "ssim": r.sr.get("SSIM", 0),
                "delta_e": r.sr.get("DeltaE", 0),
            }
            for i, r in enumerate(ranked)
        ]
        # Clean up temp field
        for r in results:
            r.sr.pop("_difficulty", None)

        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nFull results saved to: {args.json}")

        # Also write per-tile CSV for easy analysis and filtering
        csv_path = args.json.replace(".json", "_tiles.csv")
        with open(csv_path, "w") as f:
            cols = ["name", "difficulty", "max_dev", "p99_dev", "pct_over_20",
                    "res_size", "psnr", "ssim", "delta_e"]
            f.write(",".join(cols) + "\n")
            for entry in json_data["tile_difficulty_ranking"]:
                f.write(",".join(str(entry.get(c, "")) for c in cols) + "\n")
        print(f"Per-tile CSV: {csv_path}")

        # Print summary: distribution of difficulty across all tiles
        difficulties = [e["difficulty"] for e in json_data["tile_difficulty_ranking"]]
        n = len(difficulties)
        print(f"\nDifficulty distribution ({n} tiles):")
        print(f"  worst:  {difficulties[0]:.1f}  (tile: {ranked[0].name})")
        print(f"  p99:    {difficulties[int(n*0.01)]:.1f}")
        print(f"  p95:    {difficulties[int(n*0.05)]:.1f}")
        print(f"  median: {difficulties[n//2]:.1f}")
        print(f"  easiest:{difficulties[-1]:.1f}  (tile: {ranked[-1].name})")

        # Top 10 hardest
        print(f"\nTop 10 hardest tiles:")
        print(f"  {'Tile':<30} {'diff':>8} {'max_dev':>8} {'p99':>8} {'>20px':>8} {'res_KB':>8}")
        for e in json_data["tile_difficulty_ranking"][:10]:
            print(f"  {e['name']:<30} {e['difficulty']:>8.1f} {e['max_dev']:>8.1f} "
                  f"{e['p99_dev']:>8.1f} {e['pct_over_20']:>7.2f}% "
                  f"{e['res_size']/1024:>7.1f}")


if __name__ == "__main__":
    main()
