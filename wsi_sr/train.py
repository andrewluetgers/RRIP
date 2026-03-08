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
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import WSISRX4, WSIEnhanceNet, collapse_model, count_params, model_size_kb
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


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 99.0
    return 10 * np.log10(1.0 / mse)


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
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

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

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"Dataset: {len(dataset)} tiles ({n_train} train, {n_val} val)")
    print(f"Mode: {args.mode}")
    print(f"Crop: {crop}, Batch: {args.batch}")

    # Model
    if args.mode == "sr":
        model = WSISRX4(channels=args.channels, num_blocks=args.blocks).to(device)
    else:
        model = WSIEnhanceNet(channels=args.channels, num_blocks=args.blocks).to(device)

    print(f"Model: {count_params(model):,} params, {model_size_kb(model):.1f} KB (float32)")

    # Loss
    l1_loss = nn.L1Loss()
    perceptual_loss = None
    if args.perceptual_weight > 0:
        perceptual_loss = PerceptualLoss().to(device)
        print(f"Perceptual loss weight: {args.perceptual_weight}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_psnr = 0.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        t0 = time.time()

        for batch_idx, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            pred = model(lr_img)

            # Clamp to valid range for loss computation
            pred_clamped = pred.clamp(0, 1)

            loss = l1_loss(pred_clamped, hr_img)
            if perceptual_loss is not None:
                loss = loss + args.perceptual_weight * perceptual_loss(pred_clamped, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += psnr(pred_clamped.detach(), hr_img)

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_psnr = epoch_psnr / n_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_psnr = 0.0
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img = lr_img.to(device)
                    hr_img = hr_img.to(device)
                    pred = model(lr_img).clamp(0, 1)
                    val_psnr += psnr(pred, hr_img)
            val_psnr /= len(val_loader)

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr

            print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"train_psnr={avg_psnr:.2f}  val_psnr={val_psnr:.2f}  "
                  f"lr={lr:.2e}  {elapsed:.1f}s"
                  f"{'  *** BEST ***' if is_best else ''}")

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "config": {
                    "mode": args.mode,
                    "channels": args.channels,
                    "blocks": args.blocks,
                },
            }
            torch.save(ckpt, os.path.join(args.outdir, "latest.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
        else:
            print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"train_psnr={avg_psnr:.2f}  lr={lr:.2e}  {elapsed:.1f}s")

    print(f"\nDone. Best val PSNR: {best_psnr:.2f}")
    print(f"Checkpoints saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
