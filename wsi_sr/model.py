"""
Reparameterizable models for WSI tile enhancement.

Architectures (all 4x SR: 256x256 → 1024x1024):
  WSISRX4:         RGB, single branch (19K params collapsed)
  WSISRX4Dual:     Y-heavy + CbCr-light dual branch (18K params collapsed)
  WSISRX4WideDeep: Scaled dual branch, 32ch Y/10blk + 16ch CbCr/4blk (120K collapsed)
  WSISRX4Large:    Large dual branch, 48ch Y/12blk + 24ch CbCr/6blk (290K collapsed)
  WSIEnhanceNet:   Same-resolution refinement (1024x1024 → 1024x1024)

All use collapsible linear blocks:
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
# Dual-branch SR: Y-heavy + lightweight chroma
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] float tensor (BCHW) → YCbCr [0,1]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """YCbCr [0,1] float tensor (BCHW) → RGB [0,1]."""
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


class WSISRX4Dual(nn.Module):
    """Dual-branch 4x SR: heavy Y branch + lightweight CbCr branch.

    Allocates most capacity to luma (where residual quality matters)
    while still learning chroma upsampling (where Delta E comes from).

    Y branch:    1→y_ch channels, y_blocks CollapsibleBlocks → pixel_shuffle(4)
    CbCr branch: 2→c_ch channels, c_blocks CollapsibleBlocks → pixel_shuffle(4)

    Both branches use global residual (bilinear upsample of input added to output).
    Input/output are RGB — YCbCr conversion is internal.
    """

    def __init__(self, y_channels: int = 16, y_blocks: int = 5,
                 c_channels: int = 8, c_blocks: int = 2):
        super().__init__()
        # Y branch (heavy)
        self.y_head = nn.Sequential(
            nn.Conv2d(1, y_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.y_body = nn.Sequential(
            *[CollapsibleBlock(y_channels) for _ in range(y_blocks)]
        )
        self.y_tail = nn.Conv2d(y_channels, 1 * 16, 3, padding=1)  # 1 * 4^2

        # CbCr branch (light)
        self.c_head = nn.Sequential(
            nn.Conv2d(2, c_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c_body = nn.Sequential(
            *[CollapsibleBlock(c_channels) for _ in range(c_blocks)]
        )
        self.c_tail = nn.Conv2d(c_channels, 2 * 16, 3, padding=1)  # 2 * 4^2

        self.shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB → YCbCr
        ycbcr = rgb_to_ycbcr(x)
        y_in = ycbcr[:, 0:1]
        cbcr_in = ycbcr[:, 1:3]

        # Bilinear upsample as base
        y_base = F.interpolate(y_in, scale_factor=4, mode="bilinear", align_corners=False)
        cbcr_base = F.interpolate(cbcr_in, scale_factor=4, mode="bilinear", align_corners=False)

        # Y branch
        hy = self.y_head(y_in)
        hy = self.y_body(hy)
        hy = self.shuffle(self.y_tail(hy))
        y_out = y_base + hy

        # CbCr branch
        hc = self.c_head(cbcr_in)
        hc = self.c_body(hc)
        hc = self.shuffle(self.c_tail(hc))
        cbcr_out = cbcr_base + hc

        # Recombine and convert back to RGB
        ycbcr_out = torch.cat([y_out, cbcr_out], dim=1)
        return ycbcr_to_rgb(ycbcr_out)


# ---------------------------------------------------------------------------
# Scaled dual-branch variants
# ---------------------------------------------------------------------------

class WSISRX4WideDeep(nn.Module):
    """Wide+Deep dual-branch 4x SR.

    Scaled-up WSISRX4Dual with more channels and deeper networks.
    ~120K collapsed params, 469 KB float32. Fits in L2 cache.

    Y branch:    1→32 channels, 10 CollapsibleBlocks → pixel_shuffle(4)
    CbCr branch: 2→16 channels, 4 CollapsibleBlocks → pixel_shuffle(4)

    Receptive field: 25×25 (Y branch), captures nuclei + surrounding context.
    Estimated inference: ~25ms on AVX2 (within 48ms lanczos3 budget).
    """

    def __init__(self, y_channels: int = 32, y_blocks: int = 10,
                 c_channels: int = 16, c_blocks: int = 4):
        super().__init__()
        # Y branch (wide + deep)
        self.y_head = nn.Sequential(
            nn.Conv2d(1, y_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.y_body = nn.Sequential(
            *[CollapsibleBlock(y_channels) for _ in range(y_blocks)]
        )
        self.y_tail = nn.Conv2d(y_channels, 1 * 16, 3, padding=1)  # 1 * 4^2

        # CbCr branch (moderate)
        self.c_head = nn.Sequential(
            nn.Conv2d(2, c_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c_body = nn.Sequential(
            *[CollapsibleBlock(c_channels) for _ in range(c_blocks)]
        )
        self.c_tail = nn.Conv2d(c_channels, 2 * 16, 3, padding=1)  # 2 * 4^2

        self.shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ycbcr = rgb_to_ycbcr(x)
        y_in = ycbcr[:, 0:1]
        cbcr_in = ycbcr[:, 1:3]

        y_base = F.interpolate(y_in, scale_factor=4, mode="bilinear", align_corners=False)
        cbcr_base = F.interpolate(cbcr_in, scale_factor=4, mode="bilinear", align_corners=False)

        hy = self.y_head(y_in)
        hy = self.y_body(hy)
        hy = self.shuffle(self.y_tail(hy))
        y_out = y_base + hy

        hc = self.c_head(cbcr_in)
        hc = self.c_body(hc)
        hc = self.shuffle(self.c_tail(hc))
        cbcr_out = cbcr_base + hc

        ycbcr_out = torch.cat([y_out, cbcr_out], dim=1)
        return ycbcr_to_rgb(ycbcr_out)


class WSISRX4Large(nn.Module):
    """Large dual-branch 4x SR.

    Maximum capacity within the lanczos3 decode latency budget.
    ~290K collapsed params, 1.1 MB float32. Fits in L2/L3 cache.

    Y branch:    1→48 channels, 12 CollapsibleBlocks → pixel_shuffle(4)
    CbCr branch: 2→24 channels, 6 CollapsibleBlocks → pixel_shuffle(4)

    Receptive field: 29×29 (Y branch), captures small tissue structures.
    Estimated inference: ~50ms on AVX2 (matches lanczos3 budget).
    """

    def __init__(self, y_channels: int = 48, y_blocks: int = 12,
                 c_channels: int = 24, c_blocks: int = 6):
        super().__init__()
        # Y branch (large)
        self.y_head = nn.Sequential(
            nn.Conv2d(1, y_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.y_body = nn.Sequential(
            *[CollapsibleBlock(y_channels) for _ in range(y_blocks)]
        )
        self.y_tail = nn.Conv2d(y_channels, 1 * 16, 3, padding=1)

        # CbCr branch (moderate-large)
        self.c_head = nn.Sequential(
            nn.Conv2d(2, c_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c_body = nn.Sequential(
            *[CollapsibleBlock(c_channels) for _ in range(c_blocks)]
        )
        self.c_tail = nn.Conv2d(c_channels, 2 * 16, 3, padding=1)

        self.shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ycbcr = rgb_to_ycbcr(x)
        y_in = ycbcr[:, 0:1]
        cbcr_in = ycbcr[:, 1:3]

        y_base = F.interpolate(y_in, scale_factor=4, mode="bilinear", align_corners=False)
        cbcr_base = F.interpolate(cbcr_in, scale_factor=4, mode="bilinear", align_corners=False)

        hy = self.y_head(y_in)
        hy = self.y_body(hy)
        hy = self.shuffle(self.y_tail(hy))
        y_out = y_base + hy

        hc = self.c_head(cbcr_in)
        hc = self.c_body(hc)
        hc = self.shuffle(self.c_tail(hc))
        cbcr_out = cbcr_base + hc

        ycbcr_out = torch.cat([y_out, cbcr_out], dim=1)
        return ycbcr_to_rgb(ycbcr_out)


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

    def __init__(self, channels: int = 16, num_blocks: int = 5, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[CollapsibleBlock(channels) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When using extra input channels (e.g. edge map), the RGB portion
        # is x[:, :3] for the global residual skip connection
        rgb = x[:, :3] if self.in_channels > 3 else x
        h = self.head(x)
        h = self.body(h)
        h = self.tail(h)
        return rgb + h  # global residual: predict the correction


# ---------------------------------------------------------------------------
# ESPCN baseline: classic lightweight SR for comparison
# ---------------------------------------------------------------------------

class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (Shi et al., CVPR 2016).

    Classic lightweight SR baseline. 3 convs in LR space + pixel shuffle.
    ~24K params at default settings — close to our WSISRX4 collapsed size.

    Architecture:
      conv(3→64, 5x5) → tanh → conv(64→32, 3x3) → tanh → conv(32→3*r², 3x3) → pixel_shuffle(r)

    No global residual — the model predicts the full HR output directly.
    This is the standard formulation from the paper.
    """

    def __init__(self, upscale_factor: int = 4, channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, channels * (upscale_factor ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = self.conv3(h)
        return self.shuffle(h)


class ESPCNR(nn.Module):
    """ESPCN with global residual connection (our modification).

    Same as ESPCN but adds bilinear-upsampled input to the output,
    so the model only needs to predict the residual correction.
    This makes it a fair comparison with WSISRX4 which also uses global residual.
    """

    def __init__(self, upscale_factor: int = 4, channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, channels * (upscale_factor ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = self.conv3(h)
        return base + self.shuffle(h)


# ---------------------------------------------------------------------------
# Collapse + export utilities
# ---------------------------------------------------------------------------

def _collapse_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Collapse all CollapsibleBlocks in a Sequential."""
    new_seq = nn.Sequential()
    for block in seq:
        if isinstance(block, CollapsibleBlock):
            new_seq.append(collapse_block(block))
        else:
            new_seq.append(block)
    return new_seq


def collapse_model(model: nn.Module) -> nn.Module:
    """Collapse all multi-branch blocks to plain convs for fast inference."""
    collapsed = copy.deepcopy(model)
    if isinstance(model, (WSISRX4Dual, WSISRX4WideDeep, WSISRX4Large)):
        collapsed.y_body = _collapse_sequential(collapsed.y_body)
        collapsed.c_body = _collapse_sequential(collapsed.c_body)
    elif hasattr(collapsed, 'body'):
        collapsed.body = _collapse_sequential(collapsed.body)
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
    print("ESPCN Baseline (256→1024)")
    print("=" * 60)
    espcn = ESPCN(upscale_factor=4)
    print(f"  ESPCN:  {count_params(espcn):,} params, {model_size_kb(espcn):.1f} KB")
    y_espcn = espcn(x)
    print(f"  Input: {x.shape} → Output: {y_espcn.shape}")
    espcnr = ESPCNR(upscale_factor=4)
    print(f"  ESPCNR: {count_params(espcnr):,} params, {model_size_kb(espcnr):.1f} KB (with global residual)")
    y_espcnr = espcnr(x)
    print(f"  Input: {x.shape} → Output: {y_espcnr.shape}")

    print()
    print("=" * 60)
    print("Dual-Branch SR (256→1024, Y-heavy + CbCr-light)")
    print("=" * 60)
    dual = WSISRX4Dual(y_channels=16, y_blocks=5, c_channels=8, c_blocks=2)
    print(f"  Training: {count_params(dual):,} params, {model_size_kb(dual):.1f} KB")
    y_dual = dual(x)
    print(f"  Input: {x.shape} → Output: {y_dual.shape}")
    dual.eval()
    dual_c = collapse_model(dual)
    print(f"  Collapsed: {count_params(dual_c):,} params, {model_size_kb(dual_c):.1f} KB")
    with torch.no_grad():
        diff_dual = (dual(x) - dual_c(x)).abs().max().item()
    print(f"  Collapse error: {diff_dual:.2e}")

    print()
    print("=" * 60)
    print("Wide+Deep Dual-Branch SR (256→1024)")
    print("=" * 60)
    wd = WSISRX4WideDeep()
    print(f"  Training: {count_params(wd):,} params, {model_size_kb(wd):.1f} KB")
    y_wd = wd(x)
    print(f"  Input: {x.shape} → Output: {y_wd.shape}")
    wd.eval()
    wd_c = collapse_model(wd)
    print(f"  Collapsed: {count_params(wd_c):,} params, {model_size_kb(wd_c):.1f} KB")
    with torch.no_grad():
        diff_wd = (wd(x) - wd_c(x)).abs().max().item()
    print(f"  Collapse error: {diff_wd:.2e}")

    print()
    print("=" * 60)
    print("Large Dual-Branch SR (256→1024)")
    print("=" * 60)
    lg = WSISRX4Large()
    print(f"  Training: {count_params(lg):,} params, {model_size_kb(lg):.1f} KB")
    y_lg = lg(x)
    print(f"  Input: {x.shape} → Output: {y_lg.shape}")
    lg.eval()
    lg_c = collapse_model(lg)
    print(f"  Collapsed: {count_params(lg_c):,} params, {model_size_kb(lg_c):.1f} KB")
    with torch.no_grad():
        diff_lg = (lg(x) - lg_c(x)).abs().max().item()
    print(f"  Collapse error: {diff_lg:.2e}")

    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  WSISRX4 (collapsed):     {count_params(sr_c):>8,} params, {model_size_kb(sr_c):>7.1f} KB — RGB, 5 blocks, 16ch")
    print(f"  WSISRX4Dual (collapsed): {count_params(dual_c):>8,} params, {model_size_kb(dual_c):>7.1f} KB — Y(5blk,16ch) + CbCr(2blk,8ch)")
    print(f"  WideDeep (collapsed):    {count_params(wd_c):>8,} params, {model_size_kb(wd_c):>7.1f} KB — Y(10blk,32ch) + CbCr(4blk,16ch)")
    print(f"  Large (collapsed):       {count_params(lg_c):>8,} params, {model_size_kb(lg_c):>7.1f} KB — Y(12blk,48ch) + CbCr(6blk,24ch)")
    print(f"  ESPCN:                   {count_params(espcn):>8,} params, {model_size_kb(espcn):>7.1f} KB — classic baseline (tanh)")
    print(f"  ESPCNR:                  {count_params(espcnr):>8,} params, {model_size_kb(espcnr):>7.1f} KB — baseline + global residual")
