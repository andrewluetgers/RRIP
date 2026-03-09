#!/usr/bin/env python3
"""
Train a tiny 4x SR / enhancement model for WSI tiles.

Two training modes:
  --mode sr:        Input is 256x256 (raw L2), model does 4x upscale to 1024x1024
  --mode enhance:   Input is 1024x1024 (pre-upscaled via lanczos3), model refines in-place

Usage:
  # SR mode: model learns to upsample 256→1024
  python train.py --tiles /path/to/1024x1024_tiles --mode sr --epochs 200

  # Enhance mode: model refines pre-upscaled images (same resolution in/out)
  python train.py --tiles /path/to/1024x1024_tiles --mode enhance --epochs 200

  # With JPEG quality simulation on input
  python train.py --tiles /path/to/1024x1024_tiles --mode sr --jpeg-quality 95

  # Resume from checkpoint
  python train.py --tiles /path/to/tiles --resume checkpoints/best.pt
"""

import os
import sys
import time
import json
import argparse
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from model import WSISRX4, WSISRX4Dual, WSIEnhanceNet, ESPCN, ESPCNR, collapse_model, count_params, model_size_kb
from dataset import WSISRDataset


class PerceptualLoss(nn.Module):
    """Lightweight perceptual loss using VGG16 features (first 2 blocks only)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:9].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = vgg
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.features(pred_n), self.features(target_n))


class FFTLoss(nn.Module):
    """Frequency-domain loss that emphasizes high-frequency detail.

    Transforms pred and target to frequency domain via 2D FFT, then computes
    L1 loss on the magnitude spectrum. High-frequency components (edges, texture,
    cell boundaries) contribute more because they have smaller magnitudes and
    errors there are proportionally larger.

    Based on: "Fourier Space Losses for Efficient Perceptual Image SR" (ICCV 2021)
    and "Focal Frequency Loss for Image Reconstruction" (ICCV 2021).
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 2D FFT on spatial dimensions
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        # L1 on real+imaginary components (better gradient flow than magnitude-only)
        pred_ri = torch.view_as_real(pred_fft)     # [..., 2]
        target_ri = torch.view_as_real(target_fft)

        return self.weight * F.l1_loss(pred_ri, target_ri)


def _json_safe(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _cloud_upload(local_path: str, remote_path: str, timeout: int = 30) -> bool:
    """Upload a file to GCS or S3. Tries gsutil first, falls back to aws s3."""
    import subprocess
    for cmd in [
        ["gsutil", "-q", "cp", local_path, remote_path],
        ["aws", "s3", "cp", local_path, remote_path.replace("gs://", "s3://wsi-sr-training-results/").lstrip("s3://wsi-sr-training-results/") if remote_path.startswith("gs://") else remote_path],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if r.returncode == 0:
                return True
        except Exception:
            continue
    return False


def write_gcs_status(gcs_bucket: str, filename: str, data: dict):
    """Write a JSON status file to cloud storage for live monitoring."""
    if not gcs_bucket:
        return
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = f.name
        # Try GCS, then S3
        gcs_path = f"{gcs_bucket}/status/{filename}"
        s3_path = f"s3://wsi-sr-training-results/{gcs_bucket.split('/')[-1]}/status/{filename}" if "gs://" in gcs_bucket else f"{gcs_bucket}/status/{filename}"
        import subprocess
        ok = False
        for cmd in [
            ["gsutil", "-q", "cp", tmp, gcs_path],
            ["aws", "s3", "cp", tmp, s3_path],
        ]:
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=15)
                if r.returncode == 0:
                    ok = True
                    break
            except Exception:
                continue
        os.unlink(tmp)
    except Exception:
        pass


def upload_checkpoint_gcs(gcs_bucket: str, local_path: str, name: str, metrics: dict):
    """Upload a checkpoint + its metadata to cloud storage (GCS or S3)."""
    if not gcs_bucket:
        return
    import subprocess, tempfile
    try:
        # Determine paths
        stage = gcs_bucket.rstrip("/").split("/")[-1]  # e.g. "stage1"
        gcs_pt = f"{gcs_bucket}/checkpoints/{name}.pt"
        s3_pt = f"s3://wsi-sr-training-results/{stage}/checkpoints/{name}.pt"
        gcs_json = f"{gcs_bucket}/checkpoints/{name}.json"
        s3_json = f"s3://wsi-sr-training-results/{stage}/checkpoints/{name}.json"

        # Upload .pt
        for cmd in [["gsutil", "-q", "cp", local_path, gcs_pt], ["aws", "s3", "cp", local_path, s3_pt]]:
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=120)
                if r.returncode == 0:
                    break
            except Exception:
                continue

        # Upload metadata JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2)
            tmp = f.name
        for cmd in [["gsutil", "-q", "cp", tmp, gcs_json], ["aws", "s3", "cp", tmp, s3_json]]:
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=15)
                if r.returncode == 0:
                    break
            except Exception:
                continue
        os.unlink(tmp)
        print(f"  Uploaded checkpoint: {name}")
    except Exception as e:
        print(f"  Checkpoint upload failed: {e}")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 99.0
    return 10 * np.log10(1.0 / mse)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """BCHW float [0,1] → HWC uint8."""
    return (t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)


def compute_ssimulacra2(pred_u8: np.ndarray, target_u8: np.ndarray):
    """SSIMULACRA2 score. Tries Python packages first, falls back to CLI binary.
    Score interpretation: >90 excellent, >70 good, >50 ok, <30 bad."""
    from PIL import Image

    for pkg_name in ["ssimulacra2", "ssimulacra2_py"]:
        try:
            pkg = __import__(pkg_name)
            if hasattr(pkg, "compute"):
                return float(pkg.compute(Image.fromarray(target_u8), Image.fromarray(pred_u8)))
            elif hasattr(pkg, "ssimulacra2"):
                return float(pkg.ssimulacra2(target_u8, pred_u8))
        except (ImportError, Exception):
            continue

    import shutil, subprocess, tempfile
    if shutil.which("ssimulacra2"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ref_path = os.path.join(tmp, "ref.png")
                test_path = os.path.join(tmp, "test.png")
                Image.fromarray(target_u8).save(ref_path)
                Image.fromarray(pred_u8).save(test_path)
                result = subprocess.run(
                    ["ssimulacra2", ref_path, test_path],
                    capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    import re
                    for line in result.stdout.strip().split("\n"):
                        m = re.search(r"[-+]?\d*\.?\d+", line.strip())
                        if m:
                            return float(m.group())
        except Exception:
            pass
    return None


def compute_val_metrics(pred_u8: np.ndarray, target_u8: np.ndarray,
                        has_ssim=True, has_delta_e=True,
                        has_ssimulacra2=False) -> dict:
    """Compute visual quality metrics on uint8 HWC images.
    Lightweight subset suitable for running every 5 epochs during training."""
    metrics = {}

    # PSNR
    mse = float(np.mean((pred_u8.astype(np.float64) - target_u8.astype(np.float64)) ** 2))
    metrics["mse"] = mse
    metrics["psnr"] = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10))

    # SSIM
    if has_ssim:
        try:
            from skimage.metrics import structural_similarity
            metrics["ssim"] = float(structural_similarity(
                target_u8, pred_u8, channel_axis=2, data_range=255))
        except Exception:
            pass

    # Delta E (CIE2000)
    if has_delta_e:
        try:
            from skimage.color import rgb2lab, deltaE_ciede2000
            lab_pred = rgb2lab(pred_u8.astype(np.float64) / 255.0)
            lab_target = rgb2lab(target_u8.astype(np.float64) / 255.0)
            metrics["delta_e"] = float(np.mean(deltaE_ciede2000(lab_target, lab_pred)))
        except Exception:
            pass

    # SSIMULACRA2
    if has_ssimulacra2:
        val = compute_ssimulacra2(pred_u8, target_u8)
        if val is not None:
            metrics["ssimulacra2"] = val

    return metrics


@dataclass
class ResidualStats:
    """Statistics about prediction residuals."""
    size_bytes: float      # Mean JPEG-compressed residual size
    max_dev: float         # Max absolute pixel deviation (0-255 scale)
    p99_dev: float         # 99th percentile absolute deviation
    p95_dev: float         # 95th percentile absolute deviation
    mean_dev: float        # Mean absolute deviation
    pct_over_10: float     # % of pixels with |error| > 10
    pct_over_20: float     # % of pixels with |error| > 20
    pct_over_30: float     # % of pixels with |error| > 30


def compute_residual_stats(pred: torch.Tensor, target: torch.Tensor,
                           jpeg_quality: int = 80) -> ResidualStats:
    """Compute residual size and deviation statistics across a batch.

    Mirrors the ORIGAMI residual pipeline:
      residual = gt_Y - pred_Y + 128, JPEG-compressed at given quality.
    Also tracks outlier deviations that JPEG compression might hide.
    """
    import io
    from PIL import Image

    # pred/target are BCHW float [0,1]
    pred_np = (pred.detach().cpu().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    target_np = (target.detach().cpu() * 255).byte().permute(0, 2, 3, 1).numpy()

    total_bytes = 0
    batch_size = pred_np.shape[0]
    all_abs_devs = []

    for i in range(batch_size):
        p = pred_np[i].astype(np.float32)
        t = target_np[i].astype(np.float32)

        # RGB to Y (BT.601)
        pred_y = 0.299 * p[:,:,0] + 0.587 * p[:,:,1] + 0.114 * p[:,:,2]
        gt_y = 0.299 * t[:,:,0] + 0.587 * t[:,:,1] + 0.114 * t[:,:,2]

        # Raw deviation (before centering/clamping)
        deviation = gt_y - pred_y  # float, can be negative
        abs_dev = np.abs(deviation)
        all_abs_devs.append(abs_dev.ravel())

        # Centered residual for JPEG size measurement
        residual = np.clip(deviation + 128, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(residual, mode='L').save(buf, format='JPEG', quality=jpeg_quality)
        total_bytes += buf.tell()

    # Aggregate deviation stats across entire batch
    all_devs = np.concatenate(all_abs_devs)
    n_pixels = len(all_devs)

    return ResidualStats(
        size_bytes=total_bytes / batch_size,
        max_dev=float(np.max(all_devs)),
        p99_dev=float(np.percentile(all_devs, 99)),
        p95_dev=float(np.percentile(all_devs, 95)),
        mean_dev=float(np.mean(all_devs)),
        pct_over_10=float(np.sum(all_devs > 10) / n_pixels * 100),
        pct_over_20=float(np.sum(all_devs > 20) / n_pixels * 100),
        pct_over_30=float(np.sum(all_devs > 30) / n_pixels * 100),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="Directory of 1024x1024 target tiles")
    ap.add_argument("--mode", choices=["sr", "enhance"], default="sr",
                    help="sr: 256→1024 upscale. enhance: 1024→1024 refinement.")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="Simulate JPEG compression on input (e.g., 95)")
    ap.add_argument("--channels", type=int, default=16, help="Model channels")
    ap.add_argument("--blocks", type=int, default=5, help="Number of collapsible blocks")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256,
                    help="Random crop size from target tile for training (0=full tile)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--perceptual-weight", type=float, default=0.01,
                    help="Weight for perceptual loss (0=disable)")
    ap.add_argument("--fft-weight", type=float, default=0.0,
                    help="Weight for FFT frequency loss (0=disable, 0.1=recommended). "
                         "Emphasizes high-frequency detail: edges, texture, cell boundaries.")
    ap.add_argument("--arch", choices=["wsisrx4", "wsisrx4dual", "espcn", "espcnr"], default="wsisrx4",
                    help="Model architecture: wsisrx4 (RGB), wsisrx4dual (Y-heavy+CbCr-light), "
                         "espcn (baseline), espcnr (baseline+residual)")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint")
    ap.add_argument("--hard-mining", type=float, default=0.0,
                    help="Hard example mining factor (0=off, 2=2x oversample hard tiles). "
                         "Hard tiles get random re-crops and rotations, not identical repeats.")
    ap.add_argument("--run-id", default=None, help="Run ID for DuckDB tracking (default: auto)")
    ap.add_argument("--db", default=None, help="DuckDB path for tracking (default: <outdir>/wsi_sr.duckdb)")
    ap.add_argument("--gcs-status", default=None,
                    help="GCS bucket for live status (e.g. gs://bucket). Writes train_progress.json + checkpoints.")
    ap.add_argument("--patience", type=int, default=0,
                    help="Plateau patience: after N val epochs with no PSNR improvement, "
                         "switch to exploration mode (evaluate new tiles instead of training). "
                         "0=disabled, training runs all epochs.")
    ap.add_argument("--explore-tiles", default=None,
                    help="Directory of tiles for exploration mode eval (separate from training tiles)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # DuckDB tracking (optional — gracefully degrades if duckdb not installed)
    wsdb = None
    try:
        from db import WSISRDB
        db_path = args.db or os.path.join(args.outdir, "wsi_sr.duckdb")
        wsdb = WSISRDB(db_path)
        run_id = args.run_id or f"run_{int(time.time())}"
        print(f"DuckDB: {db_path} (run_id: {run_id})")
    except ImportError:
        run_id = args.run_id or f"run_{int(time.time())}"
        print("DuckDB: not installed, logging to JSONL only (pip install duckdb to enable)")

    crop = args.crop if args.crop > 0 else None

    # Dataset
    dataset = WSISRDataset(
        tile_dir=args.tiles,
        jpeg_quality=args.jpeg_quality,
        crop_size=crop,
        augment=True,
        mode=args.mode,
    )
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Disable augmentation for validation
    # (random_split shares the underlying dataset, so we validate with augmentation off
    #  by using a separate dataset instance)
    val_dataset = WSISRDataset(
        tile_dir=args.tiles,
        jpeg_quality=args.jpeg_quality,
        crop_size=None,  # full tile for validation
        augment=False,
        mode=args.mode,
    )
    # Use the same indices as val split
    val_indices = val_ds.indices if hasattr(val_ds, 'indices') else list(range(n_train, len(dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Hard example mining: track per-tile difficulty, oversample hard tiles
    hard_mining = args.hard_mining > 0
    tile_weights = np.ones(len(dataset), dtype=np.float64)  # uniform initially
    train_indices = train_ds.indices if hasattr(train_ds, 'indices') else list(range(n_train))

    # Hard mining epoch size is fixed and known upfront:
    #   base = len(train_indices)  (every tile once)
    #   max_extra = base * hard_mining_factor / 2  (bounded extra samples)
    #   total = base + max_extra
    # Example: 1000 tiles, --hard-mining 2.0 → 1000 + 1000 = 2000 samples/epoch max
    hard_max_samples = len(train_indices) + int(len(train_indices) * args.hard_mining / 2) if hard_mining else len(train_indices)

    def make_train_loader(weights=None):
        if weights is not None and hard_mining:
            # Additive oversampling: every tile keeps baseline frequency,
            # hard tiles get extra samples. Epoch size is capped at hard_max_samples.
            train_w = np.array([weights[i] for i in train_indices])
            sampler = WeightedRandomSampler(train_w, num_samples=hard_max_samples, replacement=True)
            return DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
        return DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader = make_train_loader()
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"Dataset: {len(dataset)} tiles ({n_train} train, {n_val} val)")
    print(f"Mode: {args.mode}")
    if hard_mining:
        n_extra = hard_max_samples - len(train_indices)
        print(f"Hard example mining: {hard_max_samples} samples/epoch "
              f"({n_train} base + {n_extra} max extra for hard tiles)")
    batches_per_epoch = hard_max_samples // args.batch
    print(f"Batches/epoch: {batches_per_epoch}, "
          f"Total batches: {batches_per_epoch * args.epochs}")
    print(f"Crop: {crop}, Batch: {args.batch}")

    # Exploration mode: separate tile pool for plateau evaluation
    explore_loader = None
    explore_idx = 0  # tracks position in explore set
    if args.explore_tiles and os.path.isdir(args.explore_tiles):
        explore_ds = WSISRDataset(
            tile_dir=args.explore_tiles,
            jpeg_quality=args.jpeg_quality,
            crop_size=None,  # full tile for eval
            augment=False,
            mode=args.mode,
        )
        explore_loader = DataLoader(explore_ds, batch_size=1, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
        print(f"Exploration tiles: {len(explore_ds)} (from {args.explore_tiles})")
        if args.patience > 0:
            print(f"Plateau patience: {args.patience} val epochs → explore mode")
    elif args.patience > 0:
        print(f"Plateau patience: {args.patience} val epochs (no explore tiles, will just stop training)")

    # Model
    if args.mode == "sr":
        if args.arch == "espcn":
            model = ESPCN(upscale_factor=4).to(device)
        elif args.arch == "espcnr":
            model = ESPCNR(upscale_factor=4).to(device)
        elif args.arch == "wsisrx4dual":
            model = WSISRX4Dual(y_channels=args.channels, y_blocks=args.blocks,
                                c_channels=max(args.channels // 2, 4), c_blocks=2).to(device)
        else:
            model = WSISRX4(channels=args.channels, num_blocks=args.blocks).to(device)
    else:
        model = WSIEnhanceNet(channels=args.channels, num_blocks=args.blocks).to(device)

    print(f"Model ({args.arch}): {count_params(model):,} params, {model_size_kb(model):.1f} KB (float32)")

    # Loss
    l1_loss = nn.L1Loss()
    perceptual_loss = None
    if args.perceptual_weight > 0:
        perceptual_loss = PerceptualLoss().to(device)
        print(f"Perceptual loss weight: {args.perceptual_weight}")
    fft_loss = None
    if args.fft_weight > 0:
        fft_loss = FFTLoss(weight=args.fft_weight).to(device)
        print(f"FFT frequency loss weight: {args.fft_weight}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_psnr = 0.0
    best_epoch = 0
    epochs_since_best = 0
    exploring = False  # True when plateau detected, shifts to eval mode

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Detect SSIMULACRA2 availability
    _has_ssimulacra2 = False
    for _pkg in ["ssimulacra2", "ssimulacra2_py"]:
        try:
            __import__(_pkg)
            _has_ssimulacra2 = True
            print(f"SSIMULACRA2: enabled ({_pkg})")
            break
        except ImportError:
            continue
    if not _has_ssimulacra2:
        import shutil
        if shutil.which("ssimulacra2"):
            _has_ssimulacra2 = True
            print("SSIMULACRA2: enabled (CLI binary)")
    if not _has_ssimulacra2:
        print("SSIMULACRA2: skipped (pip install ssimulacra2 or ssimulacra2-py)")

    # Compute baseline residual stats + visual metrics (bilinear/bicubic) for reference
    # Sample up to 20 val tiles to keep this fast (SSIM/Delta E are CPU-intensive)
    max_baseline_samples = min(20, len(val_loader))
    print(f"\nBaseline stats (q80, {max_baseline_samples} val samples):")
    baseline_stats = {"bilinear": [], "bicubic": []}
    baseline_metrics = {"bilinear": [], "bicubic": []}
    with torch.no_grad():
        for bi, (lr_img, hr_img, _) in enumerate(val_loader):
            if bi >= max_baseline_samples:
                break
            hr_img = hr_img.to("cpu")
            bilinear_up = F.interpolate(lr_img, scale_factor=4, mode="bilinear", align_corners=False)
            baseline_stats["bilinear"].append(compute_residual_stats(bilinear_up, hr_img, jpeg_quality=80))
            baseline_metrics["bilinear"].append(compute_val_metrics(
                tensor_to_uint8(bilinear_up), tensor_to_uint8(hr_img),
                has_ssimulacra2=_has_ssimulacra2))
            bicubic_up = F.interpolate(lr_img, scale_factor=4, mode="bicubic", align_corners=False)
            baseline_stats["bicubic"].append(compute_residual_stats(bicubic_up, hr_img, jpeg_quality=80))
            baseline_metrics["bicubic"].append(compute_val_metrics(
                tensor_to_uint8(bicubic_up), tensor_to_uint8(hr_img),
                has_ssimulacra2=_has_ssimulacra2))
    baseline_summary = {}
    for method in baseline_stats:
        stats = baseline_stats[method]
        bm = baseline_metrics[method]
        summary = {
            "size_kb": float(np.mean([s.size_bytes for s in stats]) / 1024),
            "max_dev": float(np.mean([s.max_dev for s in stats])),
            "p99_dev": float(np.mean([s.p99_dev for s in stats])),
            "p95_dev": float(np.mean([s.p95_dev for s in stats])),
            "mean_dev": float(np.mean([s.mean_dev for s in stats])),
            "pct_over_10": float(np.mean([s.pct_over_10 for s in stats])),
            "pct_over_20": float(np.mean([s.pct_over_20 for s in stats])),
            "pct_over_30": float(np.mean([s.pct_over_30 for s in stats])),
            "psnr": float(np.mean([m["psnr"] for m in bm])),
            "mse": float(np.mean([m["mse"] for m in bm])),
            "ssim": float(np.mean([m["ssim"] for m in bm if "ssim" in m])) if any("ssim" in m for m in bm) else None,
            "delta_e": float(np.mean([m["delta_e"] for m in bm if "delta_e" in m])) if any("delta_e" in m for m in bm) else None,
            "ssimulacra2": float(np.mean([m["ssimulacra2"] for m in bm if "ssimulacra2" in m])) if any("ssimulacra2" in m for m in bm) else None,
        }
        baseline_summary[method] = summary
        ssim_str = f"  ssim={summary['ssim']:.4f}" if summary['ssim'] is not None else ""
        de_str = f"  dE={summary['delta_e']:.2f}" if summary['delta_e'] is not None else ""
        s2_str = f"  s2={summary['ssimulacra2']:.1f}" if summary['ssimulacra2'] is not None else ""
        print(f"  {method:>8}: psnr={summary['psnr']:.2f}  mse={summary['mse']:.1f}{ssim_str}{de_str}{s2_str}  "
              f"size={summary['size_kb']:.1f}KB  p99={summary['p99_dev']:.1f}  >20px={summary['pct_over_20']:.2f}%")
    print()

    # Training log — JSON lines file for plotting
    log_path = os.path.join(args.outdir, "training_log.jsonl")
    log_entries = []

    # Write header entry with config and baselines
    header_entry = {
        "type": "header",
        "config": {
            "mode": args.mode, "channels": args.channels, "blocks": args.blocks,
            "epochs": args.epochs, "batch": args.batch, "crop": crop,
            "lr": args.lr, "perceptual_weight": args.perceptual_weight,
            "jpeg_quality": args.jpeg_quality, "seed": args.seed,
            "n_train": n_train, "n_val": n_val,
        },
        "baselines": baseline_summary,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(header_entry) + "\n")
    print(f"Training log: {log_path}")

    # Store run config and baselines in DuckDB
    if wsdb:
        wsdb.create_run(run_id, header_entry["config"])
        wsdb.log_baselines(run_id, baseline_summary)

    # Training loop
    explore_iter = iter(explore_loader) if explore_loader else None
    n_explored = 0

    for epoch in range(start_epoch, args.epochs):
        # Rescan for new tiles (from tile_watcher streaming in slides)
        n_new = dataset.rescan()
        if n_new > 0:
            # Expand tile_weights array for new tiles
            tile_weights = np.append(tile_weights, np.ones(n_new))
            # Update train indices — new tiles go into training
            new_start = len(train_indices) + n_val
            train_indices.extend(range(new_start, new_start + n_new))
            if hard_mining:
                hard_max_samples = len(train_indices) + int(len(train_indices) * args.hard_mining / 2)
            train_loader = make_train_loader()
            print(f"  +{n_new} new tiles (total: {len(dataset)} tiles, {len(train_indices)} training)")

        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        t0 = time.time()

        for batch_idx, (lr_img, hr_img, tile_idx) in enumerate(train_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            pred = model(lr_img)

            # Clamp to valid range for loss computation
            pred_clamped = pred.clamp(0, 1)

            loss = l1_loss(pred_clamped, hr_img)
            if perceptual_loss is not None:
                loss = loss + args.perceptual_weight * perceptual_loss(pred_clamped, hr_img)
            if fft_loss is not None:
                loss = loss + fft_loss(pred_clamped, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_psnr += psnr(pred_clamped.detach(), hr_img)

            # Track per-tile loss for hard example mining
            if hard_mining:
                # Exponential moving average of per-tile loss
                with torch.no_grad():
                    per_sample_loss = F.l1_loss(
                        pred_clamped, hr_img, reduction='none').mean(dim=[1, 2, 3])
                for i, idx in enumerate(tile_idx.tolist()):
                    tile_weights[idx] = 0.8 * tile_weights[idx] + 0.2 * per_sample_loss[i].item()

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_psnr = epoch_psnr / n_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_psnr = 0.0
        val_res_stats = []
        val_metrics_list = []
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                for lr_img, hr_img, _ in val_loader:
                    lr_img = lr_img.to(device)
                    hr_img = hr_img.to(device)
                    pred = model(lr_img).clamp(0, 1)
                    val_psnr += psnr(pred, hr_img)
                    val_res_stats.append(compute_residual_stats(pred, hr_img, jpeg_quality=80))
                    val_metrics_list.append(compute_val_metrics(
                        tensor_to_uint8(pred), tensor_to_uint8(hr_img),
                        has_ssimulacra2=_has_ssimulacra2))
            val_psnr /= len(val_loader)

            # Aggregate residual stats
            val_res_kb = np.mean([s.size_bytes for s in val_res_stats]) / 1024
            val_max_dev = np.mean([s.max_dev for s in val_res_stats])
            val_p99_dev = np.mean([s.p99_dev for s in val_res_stats])
            val_pct20 = np.mean([s.pct_over_20 for s in val_res_stats])

            # Aggregate visual metrics
            val_mse = float(np.mean([m["mse"] for m in val_metrics_list]))
            val_ssim = float(np.mean([m["ssim"] for m in val_metrics_list if "ssim" in m])) if any("ssim" in m for m in val_metrics_list) else None
            val_delta_e = float(np.mean([m["delta_e"] for m in val_metrics_list if "delta_e" in m])) if any("delta_e" in m for m in val_metrics_list) else None
            val_ssimulacra2 = float(np.mean([m["ssimulacra2"] for m in val_metrics_list if "ssimulacra2" in m])) if any("ssimulacra2" in m for m in val_metrics_list) else None

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                best_epoch = epoch + 1
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            # Plateau detection → begin exploration alongside training
            if (args.patience > 0 and epochs_since_best >= args.patience
                    and not exploring):
                exploring = True
                print(f"\n*** PLATEAU detected: no val improvement for "
                      f"{epochs_since_best} val epochs (best={best_psnr:.2f} at epoch {best_epoch})")
                if explore_loader:
                    print(f"*** Beginning exploration: will evaluate {len(explore_loader.dataset)} "
                          f"new tiles during val epochs (training continues)")
                else:
                    print(f"*** No explore tiles provided, training continues normally")

            res_stats_dict = {
                "size_kb": val_res_kb,
                "max_dev": val_max_dev,
                "p99_dev": val_p99_dev,
                "p95_dev": float(np.mean([s.p95_dev for s in val_res_stats])),
                "mean_dev": float(np.mean([s.mean_dev for s in val_res_stats])),
                "pct_over_10": float(np.mean([s.pct_over_10 for s in val_res_stats])),
                "pct_over_20": val_pct20,
                "pct_over_30": float(np.mean([s.pct_over_30 for s in val_res_stats])),
            }

            ssim_str = f"  ssim={val_ssim:.4f}" if val_ssim is not None else ""
            de_str = f"  dE={val_delta_e:.2f}" if val_delta_e is not None else ""
            s2_str = f"  s2={val_ssimulacra2:.1f}" if val_ssimulacra2 is not None else ""
            print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"train_psnr={avg_psnr:.2f}  val_psnr={val_psnr:.2f}  "
                  f"mse={val_mse:.1f}{ssim_str}{de_str}{s2_str}  "
                  f"res={val_res_kb:.1f}KB  max={val_max_dev:.0f}  "
                  f"p99={val_p99_dev:.1f}  >20={val_pct20:.2f}%  "
                  f"lr={lr:.2e}  {elapsed:.1f}s"
                  f"{'  *** BEST ***' if is_best else ''}")

            # Log entry for plotting
            log_entry = {
                "type": "val",
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_psnr": avg_psnr,
                "val_psnr": val_psnr,
                "val_mse": val_mse,
                "val_ssim": val_ssim,
                "val_delta_e": val_delta_e,
                "val_ssimulacra2": val_ssimulacra2,
                "lr": lr,
                "elapsed_s": elapsed,
                "is_best": bool(is_best),
                "residual": res_stats_dict,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(_json_safe(log_entry)) + "\n")
            if wsdb:
                wsdb.log_epoch(run_id, log_entry)

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "residual_stats": res_stats_dict,
                "config": {
                    "mode": args.mode,
                    "channels": args.channels,
                    "blocks": args.blocks,
                },
            }
            torch.save(ckpt, os.path.join(args.outdir, "latest.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
                # Upload best checkpoint to GCS
                if args.gcs_status:
                    upload_checkpoint_gcs(args.gcs_status,
                        os.path.join(args.outdir, "best.pt"),
                        f"best_ep{epoch+1}",
                        {"epoch": epoch+1, "val_psnr": val_psnr,
                         "val_ssim": val_ssim, "val_delta_e": val_delta_e,
                         "residual_kb": float(val_res_kb), "run_id": run_id})

            # Write live training status to GCS
            total_elapsed = sum(e.get("elapsed_s", 0) for e in [log_entry])
            write_gcs_status(args.gcs_status, "train_progress.json", {
                "stage": "train",
                "run_id": run_id,
                "epoch": epoch + 1,
                "total_epochs": args.epochs,
                "loss": avg_loss,
                "train_psnr": avg_psnr,
                "val_psnr": val_psnr,
                "val_mse": val_mse,
                "val_ssim": val_ssim,
                "val_delta_e": val_delta_e,
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
                "residual_kb": float(val_res_kb),
                "exploring": exploring,
                "explored_tiles": n_explored,
                "elapsed_min": round(elapsed / 60, 1),
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

            # Rebuild sampler with updated tile weights for hard example mining
            if hard_mining:
                # Scale weights: hard tiles get up to hard_mining × more samples
                # Additive: all tiles keep baseline frequency, hard tiles get extra on top
                w = tile_weights.copy()
                w_min, w_max = w[train_indices].min(), w[train_indices].max()
                if w_max > w_min:
                    # Normalize to [1, 1+factor], so hardest tile is factor× more likely
                    w_norm = (w - w_min) / (w_max - w_min)  # 0..1
                    w = 1.0 + w_norm * args.hard_mining
                train_loader = make_train_loader(w)
                n_extra = hard_max_samples - len(train_indices)
                print(f"  Hard mining: {hard_max_samples} samples/epoch "
                      f"(+{n_extra} extra for hard tiles, "
                      f"hardest {args.hard_mining + 1:.1f}x more likely)")

            # Exploration: evaluate a batch of new tiles when plateaued
            # Training continues normally — this is additive, not a replacement
            if exploring and explore_loader and n_explored < len(explore_loader.dataset):
                model.eval()
                explore_batch_size = min(50, len(explore_loader.dataset) - n_explored)
                explore_metrics = []
                with torch.no_grad():
                    for _ in range(explore_batch_size):
                        try:
                            e_lr, e_hr, e_idx = next(explore_iter)
                        except StopIteration:
                            break
                        e_lr = e_lr.to(device)
                        e_hr = e_hr.to(device)
                        e_pred = model(e_lr).clamp(0, 1)
                        em = compute_val_metrics(
                            tensor_to_uint8(e_pred), tensor_to_uint8(e_hr),
                            has_ssimulacra2=_has_ssimulacra2)
                        ers = compute_residual_stats(e_pred, e_hr, jpeg_quality=80)
                        em["residual_size_kb"] = ers.size_bytes / 1024
                        em["max_dev"] = ers.max_dev
                        em["p99_dev"] = ers.p99_dev
                        em["tile_idx"] = e_idx.item()
                        explore_metrics.append(em)
                        n_explored += 1

                if explore_metrics and wsdb:
                    for em in explore_metrics:
                        tile_path = explore_loader.dataset.paths[em["tile_idx"]]
                        wsdb.conn.execute("""
                            INSERT INTO eval_tiles (run_id, eval_name, tile_name, method,
                                psnr, mse, ssim, delta_e, ssimulacra2,
                                max_dev, p99_dev, difficulty)
                            VALUES (?, 'explore', ?, 'sr', ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [run_id, os.path.basename(tile_path),
                              em.get("psnr"), em.get("mse"), em.get("ssim"),
                              em.get("delta_e"), em.get("ssimulacra2"),
                              em.get("max_dev"), em.get("p99_dev"),
                              (em.get("max_dev", 0) + em.get("p99_dev", 0) * 2 +
                               em.get("residual_size_kb", 0))])

                if explore_metrics:
                    avg_psnr_e = np.mean([em["psnr"] for em in explore_metrics])

                    # Feed hard tiles back into current training
                    # Threshold: tiles harder than the median training tile get added
                    if hard_mining:
                        train_median_weight = float(np.median(tile_weights[train_indices]))
                        hard_paths = []
                        for em in explore_metrics:
                            difficulty = (em.get("max_dev", 0) + em.get("p99_dev", 0) * 2 +
                                          em.get("residual_size_kb", 0))
                            # Compare difficulty to training loss scale
                            # Use MAE loss as proxy: higher loss = harder
                            if em.get("mse", 0) > 0:
                                proxy_loss = em["mse"] ** 0.5 / 255.0  # rough MAE proxy
                                if proxy_loss > train_median_weight:
                                    tile_path = explore_loader.dataset.paths[em["tile_idx"]]
                                    hard_paths.append(tile_path)

                        if hard_paths:
                            new_indices = dataset.add_tiles(hard_paths)
                            # Expand tile_weights array and set high initial weight
                            old_len = len(tile_weights)
                            tile_weights = np.append(tile_weights,
                                np.full(len(new_indices), tile_weights[train_indices].max()))
                            train_indices.extend(new_indices)
                            # Rebuild sampler with expanded pool
                            w = tile_weights.copy()
                            w_min, w_max = w[train_indices].min(), w[train_indices].max()
                            if w_max > w_min:
                                w_norm = (w - w_min) / (w_max - w_min)
                                w = 1.0 + w_norm * args.hard_mining
                            # Update max samples for expanded pool
                            hard_max_samples = len(train_indices) + int(len(train_indices) * args.hard_mining / 2)
                            train_loader = make_train_loader(w)
                            print(f"  Fed {len(hard_paths)} hard tiles back into training "
                                  f"(pool: {len(train_indices)} tiles)")

                    print(f"  Explored {n_explored}/{len(explore_loader.dataset)} new tiles  "
                          f"avg_psnr={avg_psnr_e:.2f}")
        else:
            print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"train_psnr={avg_psnr:.2f}  lr={lr:.2e}  {elapsed:.1f}s")

            # Log training-only epochs too
            log_entry = {
                "type": "train",
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_psnr": avg_psnr,
                "lr": lr,
                "elapsed_s": elapsed,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(_json_safe(log_entry)) + "\n")
            if wsdb:
                wsdb.log_epoch(run_id, log_entry)

    # Finalize
    if wsdb:
        wsdb.finish_run(run_id, best_psnr, best_epoch, args.epochs)
        wsdb.close()

    print(f"\nDone. Best val PSNR: {best_psnr:.2f}")
    print(f"Checkpoints saved to: {args.outdir}/")
    print(f"Training log: {log_path}")
    if wsdb:
        print(f"DuckDB: {wsdb.path}")
    print(f"Generate plots: python plot_training.py --log {log_path}")


if __name__ == "__main__":
    main()
