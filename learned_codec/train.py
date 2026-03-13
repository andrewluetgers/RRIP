#!/usr/bin/env python3
"""
Train a Sparse Residual Autoencoder (SRA) for learned residual compression.

Loss = distortion + lambda * rate
     = MSE(residual, reconstruction) + lambda * bits(latent)

Usage:
  python train.py --variant small --lambda 1e-3 --data /path/to/residuals --run-id sra_small_1e3
  python train.py --variant tiny --lambda 1e-3 --epochs 100 --batch 16
"""

import os
import sys
import time
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from model import SparseResidualAutoencoder, count_params, model_size_kb


class ResidualDataset(Dataset):
    """Dataset of grayscale residual PNGs for codec training.

    Returns residual images as float32 tensors in [0, 1].
    Supports random crops for training and full-size for evaluation.
    """

    EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(self, data_dir: str, crop_size: int = None, augment: bool = True):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.augment = augment
        self.paths = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files
            if Path(f).suffix.lower() in self.EXTS
        ])
        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(self.paths[idx]).convert("L")  # grayscale

        # Random crop
        if self.crop_size and self.crop_size < min(img.size):
            w, h = img.size
            cs = self.crop_size
            x = random.randint(0, w - cs)
            y = random.randint(0, h - cs)
            img = img.crop((x, y, x + cs, y + cs))

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
            k = random.randint(0, 3)
            if k > 0:
                img = TF.rotate(img, k * 90)

        tensor = TF.to_tensor(img)  # [1, H, W] float32 [0, 1]
        return tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Directory of residual PNGs")
    ap.add_argument("--variant", choices=["tiny", "small", "medium", "unet"],
                    default="small", help="Model variant")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="Rate-distortion lambda (higher = smaller bitstream)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256,
                    help="Random crop size for training (0=full size)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.run_id:
        args.outdir = os.path.join(args.outdir, args.run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    crop = args.crop if args.crop > 0 else None

    # Dataset
    dataset = ResidualDataset(args.data, crop_size=crop, augment=True)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    val_dataset = ResidualDataset(args.data, crop_size=None, augment=False)
    val_indices = val_ds.indices if hasattr(val_ds, 'indices') else list(range(n_train, len(dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"Dataset: {len(dataset)} residuals ({n_train} train, {n_val} val)")
    print(f"Crop: {crop}, Batch: {args.batch}")

    # Model
    model = SparseResidualAutoencoder(args.variant).to(device)
    enc_params = count_params(model.encoder)
    dec_params = count_params(model.decoder)
    ent_params = count_params(model.entropy_model)
    print(f"Model SRA-{args.variant}: enc={enc_params:,} dec={dec_params:,} ent={ent_params:,} "
          f"total={count_params(model):,} ({model_size_kb(model):.1f} KB)")
    print(f"Lambda: {args.lam}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # JPEG baseline for comparison
    print("\nJPEG baseline (held-out residuals):")
    jpeg_baselines = {}
    with torch.no_grad():
        import io
        from PIL import Image as PILImage
        sample_count = min(20, len(val_loader))
        for q in [40, 60, 80, 90]:
            total_bytes = 0
            total_mse = 0
            for bi, residual in enumerate(val_loader):
                if bi >= sample_count:
                    break
                res_np = (residual[0, 0].numpy() * 255).astype(np.uint8)
                buf = io.BytesIO()
                PILImage.fromarray(res_np, mode='L').save(buf, format='JPEG', quality=q)
                total_bytes += buf.tell()

                buf.seek(0)
                decoded = np.array(PILImage.open(buf).convert('L'), dtype=np.float32)
                total_mse += np.mean((res_np.astype(np.float32) - decoded) ** 2)

            avg_bytes = total_bytes / sample_count
            avg_bpp = avg_bytes * 8 / (residual.shape[2] * residual.shape[3])
            avg_mse = total_mse / sample_count
            avg_psnr = 10 * np.log10(255**2 / max(avg_mse, 1e-10))
            jpeg_baselines[q] = {"bpp": avg_bpp, "psnr": avg_psnr, "bytes": avg_bytes}
            print(f"  JPEG Q{q:>2}: {avg_bpp:.4f} bpp, PSNR={avg_psnr:.2f} dB, {avg_bytes/1024:.1f} KB")

    # Training log
    log_path = os.path.join(args.outdir, "training_log.jsonl")
    print(f"\nTraining log: {log_path}")

    best_rd = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        epoch_loss = 0
        epoch_rate = 0
        epoch_dist = 0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)

            # Loss = distortion + lambda * rate
            # Normalize by number of pixels for stable training
            n_pixels = batch.shape[2] * batch.shape[3]
            distortion = out["distortion"].mean() / n_pixels  # MSE per pixel
            rate = out["rate"].mean() / n_pixels  # bits per pixel

            loss = distortion + args.lam * rate

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rate += rate.item()
            epoch_dist += distortion.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_rate = epoch_rate / n_batches
        avg_dist = epoch_dist / n_batches
        avg_psnr = 10 * np.log10(1.0 / max(avg_dist, 1e-10))  # PSNR in [0,1] range

        # Validation every 5 epochs
        val_str = ""
        is_best = False
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_rate = 0
            val_dist = 0
            val_count = 0

            with torch.no_grad():
                for residual in val_loader:
                    residual = residual.to(device)
                    out = model(residual)
                    n_pix = residual.shape[2] * residual.shape[3]
                    val_rate += out["rate"].sum().item() / n_pix
                    val_dist += out["distortion"].sum().item() / n_pix
                    val_count += 1

            val_bpp = val_rate / val_count
            val_mse = val_dist / val_count
            val_psnr = 10 * np.log10(1.0 / max(val_mse, 1e-10))

            # Estimate equivalent JPEG size for comparison
            val_bytes = val_bpp * residual.shape[2] * residual.shape[3] / 8
            val_kb = val_bytes / 1024

            # RD metric for best model selection
            rd_score = val_mse + args.lam * val_bpp
            is_best = rd_score < best_rd
            if is_best:
                best_rd = rd_score
                best_epoch = epoch + 1

                # Save checkpoint
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_rd": best_rd,
                    "config": {
                        "variant": args.variant,
                        "lam": args.lam,
                        "crop": crop,
                        "lr": args.lr,
                    },
                    "metrics": {
                        "val_bpp": val_bpp,
                        "val_psnr": val_psnr,
                        "val_mse": val_mse,
                        "val_kb": val_kb,
                    },
                    "jpeg_baselines": jpeg_baselines,
                }, os.path.join(args.outdir, "best.pt"))

            val_str = (f"  val_bpp={val_bpp:.4f}  val_psnr={val_psnr:.2f}  "
                       f"val_kb={val_kb:.1f}  ")
            if is_best:
                val_str += "*** BEST ***"

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "rate_bpp": avg_rate,
            "distortion_mse": avg_dist,
            "train_psnr": avg_psnr,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": elapsed,
        }
        if val_str:
            log_entry.update({
                "val_bpp": val_bpp,
                "val_psnr": val_psnr,
                "val_mse": val_mse,
                "val_kb": val_kb,
                "is_best": is_best,
            })
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

        line = (f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.5f}  "
                f"rate={avg_rate:.4f}bpp  dist={avg_dist:.6f}  "
                f"psnr={avg_psnr:.2f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"{elapsed:.1f}s")
        if val_str:
            line += val_str
        print(line)
        sys.stdout.flush()

    # Final summary
    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Best checkpoint: {os.path.join(args.outdir, 'best.pt')}")

    # Compare vs JPEG
    if best_rd < float("inf"):
        ckpt = torch.load(os.path.join(args.outdir, "best.pt"), map_location="cpu",
                          weights_only=False)
        m = ckpt["metrics"]
        print(f"\nBest model: {m['val_bpp']:.4f} bpp, PSNR={m['val_psnr']:.2f}, {m['val_kb']:.1f} KB")
        print("\nJPEG comparison:")
        for q, jm in sorted(jpeg_baselines.items()):
            savings = (1 - m["val_bpp"] / jm["bpp"]) * 100 if jm["bpp"] > 0 else 0
            psnr_diff = m["val_psnr"] - jm["psnr"]
            print(f"  vs Q{q}: {savings:+.1f}% size, {psnr_diff:+.2f} dB PSNR")


if __name__ == "__main__":
    main()
