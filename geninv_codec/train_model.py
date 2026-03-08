#!/usr/bin/env python3
"""
Train a tiny toy model (E and G) for "latent predicts luma residual" and save to a portable checkpoint.

This trainer supports:
- --data PATH to a single image file OR a directory of images
- Base codec proxy: JPEG quality=Q applied to each training image to create Y_base
- Residual target: R = Y_orig - Y_base

Usage examples:
  python train_model.py --data L0-1024.jpg --out model.pt
  python train_model.py --data ./tiles/ --steps 5000 --patch 64 --zdim 16 --ch 64 --out model_tiles.pt

Notes:
- For WSI, you'll likely want to train on a directory of pre-cropped tiles.
- Training on one image only is just to prove the mechanism.
"""

from __future__ import annotations

import os
import argparse
import random
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from src.utils import rgb_to_y, jpeg_encode_decode_rgb
from src.models import TinyEncoder, TinyGenerator


def list_images(path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    if os.path.isdir(path):
        out = []
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(exts):
                    out.append(os.path.join(root, fn))
        return sorted(out)
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Image file or directory of images")
    ap.add_argument("--out", default="model.pt", help="Output checkpoint path")
    ap.add_argument("--base_quality", type=int, default=50, help="JPEG quality for base codec proxy")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"No images found in {args.data}")

    # Load all images into memory (prototype). For large datasets, stream from disk.
    # Precompute Y_base and R_true for each image to speed up training.
    dataset = []
    for p in paths:
        rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        base = jpeg_encode_decode_rgb(rgb, quality=args.base_quality)
        Y_orig = rgb_to_y(rgb)
        Y_base = rgb_to_y(base)
        R_true = (Y_orig - Y_base).astype(np.float32)
        dataset.append((Y_base, R_true))

    # Models
    E = TinyEncoder(zdim=args.zdim, ch=args.ch).to(device)
    G = TinyGenerator(zdim=args.zdim, ch=args.ch).to(device)

    opt = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=args.lr)

    # Train
    for step in range(args.steps):
        # pick random image
        Y_base, R_true = random.choice(dataset)
        H, W = Y_base.shape
        ps = args.patch
        if H < ps or W < ps:
            continue

        # sample batch of random patches
        ys = np.random.randint(0, H - ps + 1, size=(args.batch,))
        xs = np.random.randint(0, W - ps + 1, size=(args.batch,))

        B_batch = np.stack([Y_base[y:y+ps, x:x+ps] for y, x in zip(ys, xs)], axis=0)
        R_batch = np.stack([R_true[y:y+ps, x:x+ps] for y, x in zip(ys, xs)], axis=0)

        B = torch.from_numpy(B_batch).to(device=device, dtype=torch.float32).unsqueeze(1)
        R = torch.from_numpy(R_batch).to(device=device, dtype=torch.float32).unsqueeze(1)

        z0 = E(B, R)
        R_hat = G(B, z0)

        loss = F.l1_loss(R_hat, R)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 250 == 0:
            print(f"step {step+1}/{args.steps}  loss={loss.item():.6f}")

    # Save portable checkpoint
    ckpt = {
        "config": {
            "patch": int(args.patch),
            "zdim": int(args.zdim),
            "ch": int(args.ch),
            "base_quality": int(args.base_quality),
        },
        "E_state": E.state_dict(),
        "G_state": G.state_dict(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
