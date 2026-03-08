"""
Dataset for WSI 4x super-resolution / enhancement training.

Two modes:
  sr:      Returns (256x256 input, 1024x1024 target) — model does 4x upscale
  enhance: Returns (1024x1024 lanczos3-upsampled input, 1024x1024 target) — model refines

In both cases, the input is generated from the target tile by downsampling 4x
then optionally applying JPEG/JXL compression to simulate the base codec.

Just dump 1024x1024 L0 tiles into a directory and point the dataset at it.
"""

import io
import os
import random
from typing import Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class WSISRDataset(Dataset):
    """Dataset that loads 1024x1024 target tiles and generates inputs.

    Supports growing tile directories — call rescan() to pick up new tiles
    added by the tile_watcher during training.

    Args:
        tile_dir: Directory of 1024x1024 PNG/JPG tiles (searched recursively)
        jpeg_quality: If set, apply JPEG compression to 256x256 (simulates base codec)
        crop_size: Random crop from target for faster training. None = full tile.
                   For SR mode, crops are taken at target resolution then input is downsampled.
        augment: Apply random flips/rotations
        mode: "sr" (256→1024) or "enhance" (1024→1024)
    """

    EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

    def __init__(
        self,
        tile_dir: str,
        jpeg_quality: Optional[int] = None,
        crop_size: Optional[int] = None,
        augment: bool = True,
        mode: str = "sr",
    ):
        self.tile_dir = tile_dir
        self.jpeg_quality = jpeg_quality
        self.crop_size = crop_size
        self.augment = augment
        self.mode = mode
        self._known_paths = set()
        self.paths = []
        self.rescan()
        if not self.paths:
            raise ValueError(f"No images found in {tile_dir}")

    def rescan(self) -> int:
        """Scan for new tiles. Returns count of newly added tiles."""
        new_paths = []
        for root, _, files in os.walk(self.tile_dir):
            for f in files:
                if f.lower().endswith(self.EXTS):
                    p = os.path.join(root, f)
                    if p not in self._known_paths:
                        new_paths.append(p)
                        self._known_paths.add(p)
        if new_paths:
            new_paths.sort()
            self.paths.extend(new_paths)
        return len(new_paths)

    def __len__(self) -> int:
        return len(self.paths)

    def add_tiles(self, paths: list) -> list:
        """Dynamically add tiles to the dataset. Returns the new indices.
        Used by exploration mode to feed hard tiles back into training."""
        start_idx = len(self.paths)
        new_paths = [p for p in paths if p not in set(self.paths)]
        self.paths.extend(new_paths)
        return list(range(start_idx, start_idx + len(new_paths)))

    def _simulate_base(self, target: Image.Image) -> Image.Image:
        """Simulate ORIGAMI base codec: downsample 4x, optionally JPEG compress."""
        tw, th = target.size
        small = target.resize((tw // 4, th // 4), Image.LANCZOS)

        # Optional JPEG compression of the 256x256 base
        if self.jpeg_quality is not None:
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=self.jpeg_quality, subsampling=0)
            buf.seek(0)
            small = Image.open(buf).convert("RGB")

        return small

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        target = Image.open(self.paths[idx]).convert("RGB")

        # Random crop from target if specified (must be divisible by 4)
        if self.crop_size is not None and self.crop_size < min(target.size):
            w, h = target.size
            cs = self.crop_size
            x = random.randint(0, w - cs)
            y = random.randint(0, h - cs)
            target = target.crop((x, y, x + cs, y + cs))

        # Create base (256x256 or crop_size/4)
        small = self._simulate_base(target)

        if self.mode == "sr":
            # SR mode: input is 256x256, target is 1024x1024
            input_img = small
        else:
            # Enhance mode: input is lanczos3 upsampled back to target size
            input_img = small.resize(target.size, Image.LANCZOS)

        # Augmentation (same transform to both)
        if self.augment:
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                target = TF.hflip(target)
            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                target = TF.vflip(target)
            k = random.randint(0, 3)
            if k > 0:
                input_img = TF.rotate(input_img, k * 90)
                target = TF.rotate(target, k * 90)

        return TF.to_tensor(input_img), TF.to_tensor(target), idx
