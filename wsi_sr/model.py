"""
Reparameterizable models for WSI tile enhancement.

Two architectures:
  WSISRX4:      4x super-resolution (256x256 → 1024x1024)
  WSIEnhanceNet: Same-resolution refinement (1024x1024 → 1024x1024)

Both use collapsible linear blocks:
  Training mode:  multi-branch blocks (3x3 + 1x1 + identity + BN) for better learning
  Inference mode: collapsed to plain 3x3 conv + ReLU stack for maximum speed

Based on: "Collapsible Linear Blocks for Super-Efficient Super Resolution" (MLSys 2022)
https://github.com/ARM-software/sesr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CollapsibleBlock(nn.Module):
    """Multi-branch block that collapses to a single 3x3 conv at inference.

    Training: 3x3_conv + BN + 1x1_conv + BN + identity → sum → ReLU
    Inference: single 3x3_conv → ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn_id = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x)) + self.bn_id(x)
        return F.relu(out, inplace=True)


class CollapsedBlock(nn.Module):
    """Plain 3x3 conv + ReLU. Result of collapsing a CollapsibleBlock."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x), inplace=True)


def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fuse conv + BN into a single conv with bias."""
    assert bn.running_mean is not None and bn.running_var is not None
    w = conv.weight
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    std = torch.sqrt(var + eps)
    scale = gamma / std

    fused_w = w * scale.reshape(-1, 1, 1, 1)
    fused_b = beta - mean * scale
    return fused_w, fused_b


def _pad_1x1_to_3x3(w: torch.Tensor) -> torch.Tensor:
    """Pad a 1x1 conv weight to 3x3."""
    return F.pad(w, [1, 1, 1, 1])


def collapse_block(block: CollapsibleBlock) -> CollapsedBlock:
    """Collapse a multi-branch block into a plain 3x3 conv."""
    channels = block.conv3.weight.shape[0]

    w3, b3 = _fuse_bn(block.conv3, block.bn3)
    w1, b1 = _fuse_bn(block.conv1, block.bn1)

    identity_w = torch.zeros_like(block.conv1.weight)
    for i in range(channels):
        identity_w[i, i, 0, 0] = 1.0
    id_conv = nn.Conv2d(channels, channels, 1, bias=False)
    id_conv.weight = nn.Parameter(identity_w)
    id_conv = id_conv.to(block.conv3.weight.device)
    w_id, b_id = _fuse_bn(id_conv, block.bn_id)

    w1_padded = _pad_1x1_to_3x3(w1)
    w_id_padded = _pad_1x1_to_3x3(w_id)

    merged_w = w3 + w1_padded + w_id_padded
    merged_b = b3 + b1 + b_id

    out = CollapsedBlock(channels)
    out.conv.weight = nn.Parameter(merged_w)
    out.conv.bias = nn.Parameter(merged_b)
    return out


# ---------------------------------------------------------------------------
# SR model: 256x256 → 1024x1024
# ---------------------------------------------------------------------------

class WSISRX4(nn.Module):
    """4x super-resolution for WSI tiles.

    Input:  256x256x3 (L2 base tile)
    Output: 1024x1024x3 (enhanced L0 tile)

    The model operates at low resolution (256x256) then upscales via pixel shuffle.
    Global residual: bilinear upsample of input is added to output.
    Compute: all convolutions at 256x256, very fast.
    """

    def __init__(self, channels: int = 16, num_blocks: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[CollapsibleBlock(channels) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 3 * 16, 3, padding=1)  # 3 * 4^2
        self.shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        h = self.head(x)
        h = self.body(h)
        h = self.tail(h)
        h = self.shuffle(h)
        return base + h


# ---------------------------------------------------------------------------
# Enhance model: 1024x1024 → 1024x1024
# ---------------------------------------------------------------------------

class WSIEnhanceNet(nn.Module):
    """Same-resolution enhancement for WSI tiles.

    Input:  1024x1024x3 (lanczos3-upsampled L2, i.e. the current ORIGAMI prediction)
    Output: 1024x1024x3 (enhanced, closer to original L0)

    The model refines the pre-upsampled image in place. It learns to:
    - Remove upsample blur
    - Sharpen edges and cellular detail
    - Correct chroma shifts from the base codec
    - Restore texture patterns typical of H&E/IHC staining

    Global residual: input is added to model output (model predicts the correction).
    Compute: all convolutions at 1024x1024, ~16x more compute than SR mode.
             Still fast with only 16 channels (~5-15ms on server CPU for collapsed model).
    """

    def __init__(self, channels: int = 16, num_blocks: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[CollapsibleBlock(channels) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h)
        h = self.tail(h)
        return x + h  # global residual: predict the correction


# ---------------------------------------------------------------------------
# Collapse + export utilities
# ---------------------------------------------------------------------------

def collapse_model(model: nn.Module) -> nn.Module:
    """Collapse all multi-branch blocks to plain convs for fast inference."""
    collapsed = copy.deepcopy(model)
    new_body = nn.Sequential()
    for block in collapsed.body:
        if isinstance(block, CollapsibleBlock):
            new_body.append(collapse_block(block))
        else:
            new_body.append(block)
    collapsed.body = new_body
    return collapsed


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_kb(model: nn.Module) -> float:
    """Model size in KB (float32)."""
    return count_params(model) * 4 / 1024


if __name__ == "__main__":
    print("=" * 60)
    print("SR Model (256→1024)")
    print("=" * 60)
    sr = WSISRX4(channels=16, num_blocks=5)
    print(f"  Training: {count_params(sr):,} params, {model_size_kb(sr):.1f} KB")
    x = torch.randn(1, 3, 256, 256)
    y = sr(x)
    print(f"  Input: {x.shape} → Output: {y.shape}")
    sr.eval()
    sr_c = collapse_model(sr)
    print(f"  Collapsed: {count_params(sr_c):,} params, {model_size_kb(sr_c):.1f} KB")
    with torch.no_grad():
        diff = (sr(x) - sr_c(x)).abs().max().item()
    print(f"  Collapse error: {diff:.2e}")

    print()
    print("=" * 60)
    print("Enhance Model (1024→1024)")
    print("=" * 60)
    enh = WSIEnhanceNet(channels=16, num_blocks=5)
    print(f"  Training: {count_params(enh):,} params, {model_size_kb(enh):.1f} KB")
    x2 = torch.randn(1, 3, 1024, 1024)
    y2 = enh(x2)
    print(f"  Input: {x2.shape} → Output: {y2.shape}")
    enh.eval()
    enh_c = collapse_model(enh)
    print(f"  Collapsed: {count_params(enh_c):,} params, {model_size_kb(enh_c):.1f} KB")
    with torch.no_grad():
        diff2 = (enh(x2) - enh_c(x2)).abs().max().item()
    print(f"  Collapse error: {diff2:.2e}")

    print()
    print("=" * 60)
    print("Tradeoff: SR vs Enhance")
    print("=" * 60)
    print("  SR mode:      convolutions at 256x256 (fast), pixel shuffle to 1024x1024")
    print("  Enhance mode: convolutions at 1024x1024 (16x more compute, but safer)")
    print("  SR is faster but must hallucinate spatial detail.")
    print("  Enhance is slower but only refines existing structure — safer for pathology.")
