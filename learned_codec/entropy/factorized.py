"""
Factorized entropy model for learned image compression.

Based on Ballé et al. 2017 "End-to-end Optimized Image Compression".
Each latent channel has a learned piecewise-linear CDF.
No autoregressive context — keeps decode fast.

During training: rate estimated as -log2(P(y_hat)) using the learned density.
During inference: actual ANS entropy coding via constriction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FactorizedEntropy(nn.Module):
    """Learned factorized prior for entropy estimation.

    Each channel has an independent learned density modeled as a flexible
    monotonic transform of a standard logistic distribution.

    The density is parameterized by a matrix H that transforms cumulative
    logistic values through softplus-activated layers to produce CDFs.
    """

    def __init__(self, channels: int, num_filters: int = 3, init_scale: float = 10.0):
        super().__init__()
        self.channels = channels
        self.init_scale = init_scale

        # Build the monotonic transform: logistic CDF → learned CDF
        # Each "filter" is a 1x1 conv equivalent applied per-channel
        filters = [1, num_filters, num_filters, num_filters, 1]
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        for i in range(len(filters) - 1):
            init = np.log(np.expm1(1.0 / init_scale / filters[i + 1]))
            matrix = nn.Parameter(torch.full(
                (channels, filters[i + 1], filters[i]), init, dtype=torch.float32))
            bias = nn.Parameter(torch.zeros(channels, filters[i + 1], 1))
            factor = nn.Parameter(torch.zeros(channels, filters[i + 1], 1))

            self._matrices.append(matrix)
            self._biases.append(bias)
            self._factors.append(factor)

    def _logits_cumulative(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the cumulative density (as logits) at given values.

        Args:
            inputs: [C, 1, N] tensor of values to evaluate
        Returns:
            logits of CDF values, same shape
        """
        logits = inputs
        for i, (matrix, bias, factor) in enumerate(
                zip(self._matrices, self._biases, self._factors)):
            # matrix: [C, out, in], logits: [C, in, N]
            weight = F.softplus(matrix)  # ensure positive
            logits = torch.bmm(weight, logits)  # [C, out, N]
            logits = logits + bias
            if i < len(self._matrices) - 1:
                # GDN-like gating
                gate = torch.tanh(factor) * logits
                logits = logits + gate
        return logits

    def forward(self, y: torch.Tensor) -> tuple:
        """Compute rate estimate and add uniform noise for training.

        Args:
            y: latent tensor [B, C, H, W] (continuous, pre-quantization)
        Returns:
            y_hat: y + uniform noise (simulates quantization)
            rate: bits per element estimate [B]
        """
        # Add uniform noise to simulate quantization
        half = 0.5
        noise = torch.empty_like(y).uniform_(-half, half)
        y_hat = y + noise

        # Estimate rate: -log2(P(y_hat))
        # P(y_hat) ≈ CDF(y_hat + 0.5) - CDF(y_hat - 0.5)
        # Reshape for per-channel processing
        B, C, H, W = y_hat.shape
        values = y_hat.permute(1, 0, 2, 3).reshape(C, 1, -1)  # [C, 1, B*H*W]

        upper = self._logits_cumulative(values + half)
        lower = self._logits_cumulative(values - half)

        # Probability = sigmoid(upper) - sigmoid(lower)
        # Use log-domain for numerical stability
        sign = -torch.sign(values).detach()
        upper = upper * sign
        lower = lower * sign
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        likelihood = likelihood.clamp(min=1e-9)

        # Rate in bits
        rate = -torch.log2(likelihood)  # [C, 1, B*H*W]
        rate = rate.reshape(C, B, H, W).permute(1, 0, 2, 3)  # [B, C, H, W]
        rate = rate.sum(dim=(1, 2, 3))  # [B] total bits per sample

        return y_hat, rate

    @torch.no_grad()
    def compress(self, y: torch.Tensor) -> tuple:
        """Quantize and compute CDFs for entropy coding.

        Args:
            y: latent tensor [1, C, H, W]
        Returns:
            y_q: quantized latent (int32) [1, C, H, W]
            cdfs: per-channel CDF tables for ANS coding
            cdf_sizes: number of CDF entries per channel
            offsets: minimum quantized value per channel
        """
        y_q = torch.round(y).to(torch.int32)

        # Build CDF tables for each channel
        # Range of quantized values
        min_val = int(y_q.min().item()) - 2
        max_val = int(y_q.max().item()) + 2
        symbols = torch.arange(min_val, max_val + 1, dtype=torch.float32,
                               device=y.device)

        C = y.shape[1]
        num_symbols = len(symbols)
        cdfs = torch.zeros(C, num_symbols + 1, device=y.device)

        for sym_idx, sym_val in enumerate(symbols):
            val = sym_val.reshape(1, 1, 1).expand(C, 1, 1)
            logits = self._logits_cumulative(val + 0.5)
            cdfs[:, sym_idx + 1] = torch.sigmoid(logits.squeeze())

        # Ensure monotonicity and proper CDF bounds
        cdfs[:, 0] = 0.0
        cdfs = torch.clamp(cdfs, 0.0, 1.0)
        for i in range(1, cdfs.shape[1]):
            cdfs[:, i] = torch.max(cdfs[:, i], cdfs[:, i - 1])
        cdfs[:, -1] = 1.0

        offsets = torch.full((C,), min_val, dtype=torch.int32)

        return y_q, cdfs, offsets

    @torch.no_grad()
    def decompress(self, y_q: torch.Tensor) -> torch.Tensor:
        """Dequantize (just cast to float)."""
        return y_q.to(torch.float32)
