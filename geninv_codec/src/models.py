"""
Tiny conditional generator + encoder used for the generator-inversion demo.

- Encoder E(B, R) -> z0  (amortized inversion initializer)
- Generator G(B, z) -> R_hat

Where:
- B is luma patch (1 channel)
- R is true residual patch (1 channel)
- z is small latent vector (zdim)

This is a toy architecture intended to prove the end-to-end "latent predicts residual" idea.
Train on many patches (ideally from many WSI tiles), then reuse the saved checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, channels: int, zdim: int):
        super().__init__()
        self.to_gb = nn.Linear(zdim, channels * 2)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(z)
        g, b = gb.chunk(2, dim=1)
        g = g[:, :, None, None]
        b = b[:, :, None, None]
        return x * (1.0 + g) + b


class TinyGenerator(nn.Module):
    """G(B, z) -> R_hat (signed)."""
    def __init__(self, zdim: int = 8, ch: int = 64):
        super().__init__()
        self.in1 = nn.Conv2d(1, ch, 3, padding=1)
        self.in2 = nn.Conv2d(ch, ch, 3, padding=1)

        self.f1 = FiLM(ch, zdim)
        self.f2 = FiLM(ch, zdim)
        self.f3 = FiLM(ch, zdim)

        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)

        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, B: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in1(B))
        x = F.relu(self.in2(x))

        x = F.relu(self.f1(self.c1(x), z))
        x = F.relu(self.f2(self.c2(x), z))
        x = F.relu(self.f3(self.c3(x), z))

        return self.out(x)


class TinyEncoder(nn.Module):
    """E(B, R) -> z0."""
    def __init__(self, zdim: int = 8, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(ch, zdim)

    def forward(self, B: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x = torch.cat([B, R], dim=1)
        x = self.net(x).flatten(1)
        return self.fc(x)
