# geninv_full_report.py
# Hybrid experiment: base codec (JPEG q50 stand-in) + true residual vs latent-generated residual (+ optional correction).
# Luminance-only (Rec.601). Residual/correction stored as centered uint8 and encoded as JPEG90/95.
#
# Also trains tiny E/G and saves portable model checkpoint, and can reuse it later.
#
# Requirements: pillow, numpy, torch, matplotlib

import os
import io
import json
import math
import zipfile
import random
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------
def rgb_to_y(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

def jpeg_encode_decode_rgb(rgb_u8: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=0, optimize=True
    )
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)

def save_gray01_png(path: str, arr01: np.ndarray):
    u8 = (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)

def save_rgb_png(path: str, rgb_u8: np.ndarray):
    Image.fromarray(rgb_u8, mode="RGB").save(path)

def psnr01(a01: np.ndarray, b01: np.ndarray) -> float:
    mse = float(np.mean((a01 - b01) ** 2, dtype=np.float64))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def mae01(a01: np.ndarray, b01: np.ndarray) -> float:
    return float(np.mean(np.abs(a01 - b01), dtype=np.float64))

def quantize_residual_int8(residual_float: np.ndarray) -> np.ndarray:
    q = np.round(residual_float * 255.0).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int16)
    return q

def dequantize_residual_int8(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 255.0).astype(np.float32)

def signed_to_u8_centered(q: np.ndarray) -> np.ndarray:
    u = (q.astype(np.int16) + 128)
    return np.clip(u, 0, 255).astype(np.uint8)

def u8_centered_to_signed(u: np.ndarray) -> np.ndarray:
    q = (u.astype(np.int16) - 128)
    return np.clip(q, -127, 127).astype(np.int16)

def save_centered_int8_as_jpeg(path: str, q_int8: np.ndarray, quality: int):
    u8 = signed_to_u8_centered(q_int8)
    Image.fromarray(u8, mode="L").save(path, quality=int(quality), subsampling=0, optimize=True)

def load_centered_jpeg_as_int8(path: str) -> np.ndarray:
    u8 = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return u8_centered_to_signed(u8)

def residual_visual(res: np.ndarray, ref_scale: float) -> np.ndarray:
    s = float(ref_scale) if ref_scale > 1e-12 else 1e-12
    return np.clip((res / (2.0 * s)) + 0.5, 0.0, 1.0).astype(np.float32)

def error_vis_centered(err: np.ndarray, gain: float = 6.0) -> np.ndarray:
    # visualize signed error centered at 128
    q = np.clip(np.round(err * 255.0 * gain), -127, 127).astype(np.int16)
    return signed_to_u8_centered(q)

def to_torch_gray(y01: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(y01).to(device=device, dtype=torch.float32)
    return t.unsqueeze(0).unsqueeze(0)

def from_torch_gray(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()[0, 0].astype(np.float32)

def pack_latents_int8(z_list: np.ndarray, scale: float, patch: int, H: int, W: int) -> bytes:
    # z_list: (num_patches, zdim) float32
    # global scale used; store int8 values
    zq = np.clip(np.round(z_list / scale), -127, 127).astype(np.int8)
    header = {
        "patch": patch,
        "H": H,
        "W": W,
        "zdim": int(z_list.shape[1]),
        "scale": float(scale),
        "dtype": "int8",
        "order": "row-major patches",
    }
    hb = json.dumps(header).encode("utf-8")
    return len(hb).to_bytes(4, "little") + hb + zq.tobytes()

def bytes_len(path: str) -> int:
    return os.path.getsize(path)


# -------------------------
# Tiny toy E/G
# -------------------------
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
    def __init__(self, zdim: int = 8, ch: int = 32):
        super().__init__()
        self.in1 = nn.Conv2d(1, ch, 3, padding=1)
        self.in2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.f1 = FiLM(ch, zdim)
        self.f2 = FiLM(ch, zdim)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, B: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in1(B))
        x = F.relu(self.in2(x))
        x = F.relu(self.f1(self.c1(x), z))
        x = F.relu(self.f2(self.c2(x), z))
        return self.out(x)  # residual can be signed

class TinyEncoder(nn.Module):
    def __init__(self, zdim: int = 8, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(ch, zdim)

    def forward(self, B: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x = torch.cat([B, R], dim=1)
        x = self.net(x).flatten(1)
        return self.fc(x)


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    input_path: str = "L0-1024.jpg"
    out_dir: str = "geninv_full_report"
    zip_path: str = "geninv_full_report.zip"

    # base codec proxy
    base_jpeg_quality: int = 50

    # luminance patching
    patch: int = 64
    zdim: int = 8

    # training (toy)
    seed: int = 0
    train_steps: int = 400     # increase if you want better demo quality
    batch: int = 64
    lr: float = 2e-3

    # model persistence
    model_path: str = "model.pt"
    reuse_model_if_exists: bool = True

    # residual/correction JPEG qualities
    residual_jpeg_qualities = (90, 95)
    correction_jpeg_qualities = (90, 95)

    # visuals
    err_gain: float = 6.0


# -------------------------
# Train or load model
# -------------------------
def save_model(path: str, cfg: Config, E: nn.Module, G: nn.Module):
    ckpt = {
        "config": {"patch": cfg.patch, "zdim": cfg.zdim},
        "E_state": E.state_dict(),
        "G_state": G.state_dict(),
    }
    torch.save(ckpt, path)

def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    zdim = int(ckpt["config"]["zdim"])
    E = TinyEncoder(zdim=zdim).to(device)
    G = TinyGenerator(zdim=zdim).to(device)
    E.load_state_dict(ckpt["E_state"])
    G.load_state_dict(ckpt["G_state"])
    E.eval(); G.eval()
    return E, G, ckpt

def train_toy_model(cfg: Config, Y_base: np.ndarray, R_true: np.ndarray, device: str):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    H, W = Y_base.shape
    ps = cfg.patch

    tB = torch.from_numpy(Y_base).to(device=device, dtype=torch.float32)
    tR = torch.from_numpy(R_true).to(device=device, dtype=torch.float32)

    E = TinyEncoder(zdim=cfg.zdim).to(device)
    G = TinyGenerator(zdim=cfg.zdim).to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=cfg.lr)

    E.train(); G.train()
    for step in range(cfg.train_steps):
        ys = torch.randint(0, H - ps + 1, (cfg.batch,), device=device)
        xs = torch.randint(0, W - ps + 1, (cfg.batch,), device=device)

        B_batch = torch.stack([tB[y:y+ps, x:x+ps] for y, x in zip(ys, xs)], dim=0).unsqueeze(1)
        R_batch = torch.stack([tR[y:y+ps, x:x+ps] for y, x in zip(ys, xs)], dim=0).unsqueeze(1)

        z0 = E(B_batch, R_batch)
        R_hat = G(B_batch, z0)
        loss = F.l1_loss(R_hat, R_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    E.eval(); G.eval()
    return E, G


# -------------------------
# Full-frame latent prediction
# -------------------------
@torch.no_grad()
def predict_full_residual_and_latents(cfg: Config, E: nn.Module, G: nn.Module,
                                      Y_base: np.ndarray, R_true: np.ndarray, device: str):
    H, W = Y_base.shape
    ps = cfg.patch

    tB = torch.from_numpy(Y_base).to(device=device, dtype=torch.float32)
    tR = torch.from_numpy(R_true).to(device=device, dtype=torch.float32)

    R_hat = torch.zeros((H, W), device=device, dtype=torch.float32)
    z_list = []

    for yy in range(0, H, ps):
        for xx in range(0, W, ps):
            Bp = tB[yy:yy+ps, xx:xx+ps].unsqueeze(0).unsqueeze(0)
            Rp = tR[yy:yy+ps, xx:xx+ps].unsqueeze(0).unsqueeze(0)
            z0 = E(Bp, Rp)          # encode-time latent (amortized)
            rhat = G(Bp, z0)[0, 0]  # (ps,ps)
            R_hat[yy:yy+ps, xx:xx+ps] = rhat
            z_list.append(z0[0].detach().cpu().numpy())

    return from_torch_gray(R_hat.unsqueeze(0).unsqueeze(0)), np.stack(z_list, axis=0).astype(np.float32)


# -------------------------
# Main report
# -------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original and create base
    orig_rgb = np.array(Image.open(cfg.input_path).convert("RGB"), dtype=np.uint8)
    H, W = orig_rgb.shape[:2]

    base_rgb = jpeg_encode_decode_rgb(orig_rgb, cfg.base_jpeg_quality)

    # Save originals
    save_rgb_png(os.path.join(cfg.out_dir, "orig_rgb.png"), orig_rgb)
    save_rgb_png(os.path.join(cfg.out_dir, "base_rgb.png"), base_rgb)

    Y_orig = rgb_to_y(orig_rgb)
    Y_base = rgb_to_y(base_rgb)
    R_true = (Y_orig - Y_base).astype(np.float32)

    # Save Y images + base error vis
    save_gray01_png(os.path.join(cfg.out_dir, "Y_orig.png"), Y_orig)
    save_gray01_png(os.path.join(cfg.out_dir, "Y_base.png"), Y_base)
    save_gray01_png(os.path.join(cfg.out_dir, "err_base_vis.png"), (error_vis_centered(Y_orig - Y_base, cfg.err_gain) / 255.0))

    # Save true residual visualization
    full_scale = float(np.max(np.abs(R_true)) + 1e-9)
    save_gray01_png(os.path.join(cfg.out_dir, "true_residual_visual.png"), residual_visual(R_true, full_scale))

    # Train or load model
    model_full_path = os.path.join(cfg.out_dir, cfg.model_path)
    if cfg.reuse_model_if_exists and os.path.exists(model_full_path):
        E, G, ckpt = load_model(model_full_path, device=device)
        model_note = f"Loaded model from {model_full_path}"
    else:
        E, G = train_toy_model(cfg, Y_base, R_true, device=device)
        save_model(model_full_path, cfg, E, G)
        model_note = f"Trained toy model and saved to {model_full_path}"

    # Predict residual via latent model
    R_hat, z_list = predict_full_residual_and_latents(cfg, E, G, Y_base, R_true, device=device)
    Y_latent_only = np.clip(Y_base + R_hat, 0.0, 1.0).astype(np.float32)

    # Pack & size latent payload (int8 + global scale)
    z_max = float(np.max(np.abs(z_list)) + 1e-9)
    z_scale = z_max / 127.0
    latent_payload = pack_latents_int8(z_list, z_scale, cfg.patch, H, W)
    latent_path = os.path.join(cfg.out_dir, "latent_payload.bin")
    with open(latent_path, "wb") as f:
        f.write(latent_payload)

    # Save predicted residual visualization + latent-only recon + error map
    save_gray01_png(os.path.join(cfg.out_dir, "pred_residual_visual.png"), residual_visual(R_hat, full_scale))
    save_gray01_png(os.path.join(cfg.out_dir, "Y_latent_only.png"), Y_latent_only)
    save_gray01_png(os.path.join(cfg.out_dir, "err_latent_only_vis.png"),
                    (error_vis_centered(Y_orig - Y_latent_only, cfg.err_gain) / 255.0))

    # -------- Variant A: True residual stored as JPEG90/95 then applied --------
    rows = []
    base_jpeg_path = os.path.join(cfg.out_dir, f"base_codec_q{cfg.base_jpeg_quality}.jpg")
    Image.fromarray(orig_rgb).save(base_jpeg_path, quality=cfg.base_jpeg_quality, subsampling=0, optimize=True)
    base_bytes = bytes_len(base_jpeg_path)

    rows.append({
        "method": f"Base only (JPEG q{cfg.base_jpeg_quality})",
        "base_bytes": base_bytes,
        "extra_bytes": 0,
        "total_bytes": base_bytes,
        "Y_PSNR": psnr01(Y_base, Y_orig),
        "Y_MAE": mae01(Y_base, Y_orig),
    })

    Rq = quantize_residual_int8(R_true)

    for q in cfg.residual_jpeg_qualities:
        res_path = os.path.join(cfg.out_dir, f"true_residual_centered_JPEG{q}.jpg")
        save_centered_int8_as_jpeg(res_path, Rq, quality=q)

        # decode and apply
        Rq_dec = load_centered_jpeg_as_int8(res_path)
        R_dec = dequantize_residual_int8(Rq_dec)
        Y_rec = np.clip(Y_base + R_dec, 0.0, 1.0).astype(np.float32)

        save_gray01_png(os.path.join(cfg.out_dir, f"Y_true_residual_applied_JPEG{q}.png"), Y_rec)
        save_gray01_png(os.path.join(cfg.out_dir, f"err_true_residual_JPEG{q}_vis.png"),
                        (error_vis_centered(Y_orig - Y_rec, cfg.err_gain) / 255.0))

        extra = bytes_len(res_path)
        rows.append({
            "method": f"True residual JPEG{q} + apply",
            "base_bytes": base_bytes,
            "extra_bytes": extra,
            "total_bytes": base_bytes + extra,
            "Y_PSNR": psnr01(Y_rec, Y_orig),
            "Y_MAE": mae01(Y_rec, Y_orig),
        })

    # -------- Variant B: Latent-only, and Latent + correction JPEG90/95 --------
    latent_bytes = bytes_len(latent_path)
    rows.append({
        "method": f"Latent-only (zdim={cfg.zdim}, patch={cfg.patch}) + apply",
        "base_bytes": base_bytes,
        "extra_bytes": latent_bytes,
        "total_bytes": base_bytes + latent_bytes,
        "Y_PSNR": psnr01(Y_latent_only, Y_orig),
        "Y_MAE": mae01(Y_latent_only, Y_orig),
    })

    Corr = (R_true - R_hat).astype(np.float32)
    Corrq = quantize_residual_int8(Corr)

    # Save correction visualization
    corr_scale = float(np.max(np.abs(Corr)) + 1e-9)
    save_gray01_png(os.path.join(cfg.out_dir, "correction_visual.png"), residual_visual(Corr, corr_scale))

    for q in cfg.correction_jpeg_qualities:
        corr_path = os.path.join(cfg.out_dir, f"correction_centered_JPEG{q}.jpg")
        save_centered_int8_as_jpeg(corr_path, Corrq, quality=q)

        Corrq_dec = load_centered_jpeg_as_int8(corr_path)
        Corr_dec = dequantize_residual_int8(Corrq_dec)

        Y_lat_corr = np.clip(Y_base + R_hat + Corr_dec, 0.0, 1.0).astype(np.float32)

        save_gray01_png(os.path.join(cfg.out_dir, f"Y_latent_plus_corr_JPEG{q}.png"), Y_lat_corr)
        save_gray01_png(os.path.join(cfg.out_dir, f"err_latent_plus_corr_JPEG{q}_vis.png"),
                        (error_vis_centered(Y_orig - Y_lat_corr, cfg.err_gain) / 255.0))

        extra = latent_bytes + bytes_len(corr_path)
        rows.append({
            "method": f"Latent + correction JPEG{q} + apply",
            "base_bytes": base_bytes,
            "extra_bytes": extra,
            "total_bytes": base_bytes + extra,
            "Y_PSNR": psnr01(Y_lat_corr, Y_orig),
            "Y_MAE": mae01(Y_lat_corr, Y_orig),
        })

    # Write results.csv
    import csv
    results_csv = os.path.join(cfg.out_dir, "results.csv")
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # metrics.json for quick glance
    metrics = {
        "config": asdict(cfg),
        "model": model_note,
        "base_bytes": base_bytes,
        "latent_bytes": latent_bytes,
        "rows": rows,
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Montage (compact view)
    def show(ax, path, title):
        ax.imshow(Image.open(path), cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    montage = os.path.join(cfg.out_dir, "montage.png")
    fig = plt.figure(figsize=(14, 10))
    axs = fig.subplots(3, 4).flatten()

    axs[0].imshow(Image.open(os.path.join(cfg.out_dir, "orig_rgb.png")))
    axs[0].set_title("orig_rgb"); axs[0].axis("off")
    axs[1].imshow(Image.open(os.path.join(cfg.out_dir, "base_rgb.png")))
    axs[1].set_title(f"base_rgb (JPEG q{cfg.base_jpeg_quality})"); axs[1].axis("off")

    show(axs[2], os.path.join(cfg.out_dir, "Y_orig.png"), "Y_orig")
    show(axs[3], os.path.join(cfg.out_dir, "Y_base.png"), "Y_base")

    show(axs[4], os.path.join(cfg.out_dir, "true_residual_visual.png"), "true residual (vis)")
    show(axs[5], os.path.join(cfg.out_dir, "pred_residual_visual.png"), "pred residual (vis)")
    show(axs[6], os.path.join(cfg.out_dir, "correction_visual.png"), "correction (vis)")
    show(axs[7], os.path.join(cfg.out_dir, "err_base_vis.png"), "err base (vis)")

    show(axs[8], os.path.join(cfg.out_dir, "Y_latent_only.png"), "Y latent-only")
    show(axs[9], os.path.join(cfg.out_dir, "err_latent_only_vis.png"), "err latent-only (vis)")

    # include one true residual and one corrected version if exists
    tr95 = os.path.join(cfg.out_dir, "Y_true_residual_applied_JPEG95.png")
    lc95 = os.path.join(cfg.out_dir, "Y_latent_plus_corr_JPEG95.png")
    if os.path.exists(tr95):
        show(axs[10], tr95, "Y true residual JPEG95")
    else:
        axs[10].axis("off")
    if os.path.exists(lc95):
        show(axs[11], lc95, "Y latent+corr JPEG95")
    else:
        axs[11].axis("off")

    fig.tight_layout()
    fig.savefig(montage, dpi=180)
    plt.close(fig)

    # Zip report folder
    zip_out = os.path.join(cfg.out_dir, cfg.zip_path)
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(cfg.out_dir):
            for fn in files:
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, cfg.out_dir)
                z.write(p, arcname=os.path.join(cfg.out_dir, arc))

    print("Done.")
    print("Report folder:", cfg.out_dir)
    print("Model checkpoint:", model_full_path)
    print("Results CSV:", results_csv)
    print("Zip:", zip_out)


if __name__ == "__main__":
    main()