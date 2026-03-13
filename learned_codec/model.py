"""
Sparse Residual Autoencoder (SRA) for learned residual compression.

Encodes 1024x1024 grayscale residuals into compact latent representations
with learned entropy coding. Decoder must run <10ms on CPU via ONNX Runtime.

Variants:
  SRA-Tiny:   16ch encoder/decoder, 16ch latent, ~15K decoder params
  SRA-Small:  32ch encoder/decoder, 32ch latent, ~50K decoder params
  SRA-Medium: 64ch encoder/decoder, 32ch latent, ~150K decoder params
  SRA-UNet:   32ch encoder/decoder, 32ch latent, skip connections, ~80K decoder params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from entropy import FactorizedEntropy


class SRAEncoder(nn.Module):
    """Strided conv encoder: 1024x1024 → 64x64 latent.

    4 stride-2 convolutions with GeLU activation.
    Runs on GPU at encode time — speed is secondary to quality.
    """

    def __init__(self, channels: int = 32, latent_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),       # 1024 → 512
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), # 512 → 256
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), # 256 → 128
            nn.GELU(),
            nn.Conv2d(channels, latent_channels, 3, stride=2, padding=1),  # 128 → 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SRADecoder(nn.Module):
    """Transposed conv decoder: 64x64 latent → 1024x1024 residual.

    4 transposed convolutions with GeLU activation.
    Must run <10ms on CPU — keep it minimal.
    """

    def __init__(self, channels: int = 32, latent_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, channels, 4, stride=2, padding=1),  # 64 → 128
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),  # 128 → 256
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),  # 256 → 512
            nn.GELU(),
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1),  # 512 → 1024
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SRAUNetDecoder(nn.Module):
    """U-Net style decoder with skip connections from encoder features.

    Receives encoder intermediate features at matching spatial scales.
    Concatenates encoder features with decoder features at each level.
    """

    def __init__(self, channels: int = 32, latent_channels: int = 32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_channels, channels, 4, stride=2, padding=1)   # 64→128
        self.conv1 = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.GELU())
        self.up2 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)           # 128→256
        self.conv2 = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.GELU())
        self.up3 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)           # 256→512
        self.conv3 = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.GELU())
        self.up4 = nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1)                  # 512→1024

    def forward(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        """skips: [enc_128, enc_256, enc_512] from encoder intermediate outputs."""
        x = F.gelu(self.up1(z))
        x = self.conv1(torch.cat([x, skips[0]], dim=1))
        x = F.gelu(self.up2(x))
        x = self.conv2(torch.cat([x, skips[1]], dim=1))
        x = F.gelu(self.up3(x))
        x = self.conv3(torch.cat([x, skips[2]], dim=1))
        x = self.up4(x)
        return x


class SRAUNetEncoder(nn.Module):
    """Encoder that also returns intermediate features for U-Net skip connections."""

    def __init__(self, channels: int = 32, latent_channels: int = 32):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.GELU())       # 1024→512
        self.down2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.GELU()) # 512→256
        self.down3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.GELU()) # 256→128
        self.down4 = nn.Conv2d(channels, latent_channels, 3, stride=2, padding=1)  # 128→64

    def forward(self, x: torch.Tensor) -> tuple:
        s512 = self.down1(x)   # [B, C, 512, 512]
        s256 = self.down2(s512) # [B, C, 256, 256]
        s128 = self.down3(s256) # [B, C, 128, 128]
        z = self.down4(s128)    # [B, C_latent, 64, 64]
        return z, [s128, s256, s512]


class SparseResidualAutoencoder(nn.Module):
    """Full autoencoder: encoder + entropy model + decoder.

    Training: encoder → add noise → estimate rate → decoder → MSE loss
    Inference: encoder → quantize → ANS encode → bitstream
               bitstream → ANS decode → dequantize → decoder → residual
    """

    def __init__(self, variant: str = "small"):
        super().__init__()
        configs = {
            "tiny":   {"channels": 16, "latent_channels": 16},
            "small":  {"channels": 32, "latent_channels": 32},
            "medium": {"channels": 64, "latent_channels": 32},
            "unet":   {"channels": 32, "latent_channels": 32},
        }
        if variant not in configs:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

        cfg = configs[variant]
        self.variant = variant

        if variant == "unet":
            self.encoder = SRAUNetEncoder(**cfg)
            self.decoder = SRAUNetDecoder(**cfg)
        else:
            self.encoder = SRAEncoder(**cfg)
            self.decoder = SRADecoder(**cfg)

        self.entropy_model = FactorizedEntropy(cfg["latent_channels"])

    def forward(self, x: torch.Tensor) -> dict:
        """Training forward pass.

        Args:
            x: residual images [B, 1, H, W] float32, values in [0, 1]
        Returns:
            dict with: x_hat (reconstruction), rate (bits), distortion (MSE)
        """
        if self.variant == "unet":
            y, skips = self.encoder(x)
        else:
            y = self.encoder(x)

        y_hat, rate = self.entropy_model(y)

        if self.variant == "unet":
            x_hat = self.decoder(y_hat, skips)
        else:
            x_hat = self.decoder(y_hat)

        # MSE distortion
        distortion = F.mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2, 3))  # [B]

        return {
            "x_hat": x_hat,
            "rate": rate,          # bits per sample [B]
            "distortion": distortion,  # MSE sum per sample [B]
            "y": y,                # latent (for analysis)
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        """Compress a residual to quantized latent + CDF tables.

        Args:
            x: [1, 1, H, W] residual
        Returns:
            dict with y_q (quantized latent), cdfs, offsets
        """
        if self.variant == "unet":
            y, _ = self.encoder(x)
        else:
            y = self.encoder(x)

        y_q, cdfs, offsets = self.entropy_model.compress(y)
        return {"y_q": y_q, "cdfs": cdfs, "offsets": offsets}

    @torch.no_grad()
    def decompress(self, y_q: torch.Tensor) -> torch.Tensor:
        """Decode from quantized latent.

        Args:
            y_q: [1, C, H, W] quantized latent (int32 or float32)
        Returns:
            x_hat: [1, 1, H, W] reconstructed residual
        """
        z = y_q.to(torch.float32)
        if self.variant == "unet":
            # No skip connections available at decode time for UNet
            # Fall back to plain decoder (skip connections are zero)
            raise NotImplementedError(
                "UNet decoder requires skip connections from encoder. "
                "For inference, export the full model (encoder+decoder) to ONNX.")
        return self.decoder(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_kb(model: nn.Module) -> float:
    return count_params(model) * 4 / 1024


if __name__ == "__main__":
    for variant in ["tiny", "small", "medium", "unet"]:
        print(f"\n{'=' * 50}")
        print(f"SRA-{variant.upper()}")
        print(f"{'=' * 50}")
        model = SparseResidualAutoencoder(variant)
        enc_params = count_params(model.encoder)
        dec_params = count_params(model.decoder)
        ent_params = count_params(model.entropy_model)
        total = count_params(model)
        print(f"  Encoder:  {enc_params:>8,} params ({model_size_kb(model.encoder):.1f} KB)")
        print(f"  Decoder:  {dec_params:>8,} params ({model_size_kb(model.decoder):.1f} KB)")
        print(f"  Entropy:  {ent_params:>8,} params ({model_size_kb(model.entropy_model):.1f} KB)")
        print(f"  Total:    {total:>8,} params ({model_size_kb(model):.1f} KB)")

        x = torch.randn(2, 1, 256, 256)  # test with 256x256 crop
        out = model(x)
        print(f"  Input:  {x.shape}")
        print(f"  Output: {out['x_hat'].shape}")
        print(f"  Rate:   {out['rate'].mean().item():.0f} bits/sample")
        print(f"  MSE:    {out['distortion'].mean().item():.4f}")

        bpp = out['rate'].mean().item() / (256 * 256)
        print(f"  BPP:    {bpp:.4f}")
