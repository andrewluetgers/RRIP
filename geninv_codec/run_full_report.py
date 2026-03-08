#!/usr/bin/env python3
"""
Run the full apples-to-apples report on a single image:

Compare:
A) True luma residual stored as JPEG90/95 and applied to base
B) Latent-predicted residual stored as compact int8 latent payload
   - latent-only reconstruction
   - correction map (true - predicted residual) stored as JPEG90/95 and applied

Also produces:
- images (Y views, residual/correction maps, error visualizations)
- results.csv + metrics.json
- montage.png
- zip of the report folder

Usage:
  python run_full_report.py --image L0-1024.jpg --model model.pt --outdir report_out

Notes:
- Base codec proxy is JPEG quality=50 by default (stand-in for JPEG XL).
- Latents are int8 with a single global scale (maxabs/127).
"""

from __future__ import annotations

import os
import json
import csv
import zipfile
import argparse

import numpy as np
from PIL import Image

import torch

from src.utils import (
    rgb_to_y, jpeg_encode_decode_rgb,
    quantize_residual_int8, dequantize_residual_int8,
    save_gray01_png, save_rgb_png,
    save_centered_int8_as_jpeg, load_centered_jpeg_as_int8,
    residual_visual, error_vis_centered,
    psnr01, mae01,
    pack_latents_int8, zlib_size
)
from src.models import TinyEncoder, TinyGenerator


@torch.no_grad()
def predict_full_residual_and_latents(E, G, Y_base, R_true, patch, device):
    H, W = Y_base.shape
    ps = patch
    tB = torch.from_numpy(Y_base).to(device=device, dtype=torch.float32)
    tR = torch.from_numpy(R_true).to(device=device, dtype=torch.float32)

    R_hat = torch.zeros((H, W), device=device, dtype=torch.float32)
    z_list = []

    for yy in range(0, H, ps):
        for xx in range(0, W, ps):
            Bp = tB[yy:yy+ps, xx:xx+ps].unsqueeze(0).unsqueeze(0)
            Rp = tR[yy:yy+ps, xx:xx+ps].unsqueeze(0).unsqueeze(0)
            z0 = E(Bp, Rp)
            rhat = G(Bp, z0)[0, 0]
            R_hat[yy:yy+ps, xx:xx+ps] = rhat
            z_list.append(z0[0].detach().cpu().numpy())

    R_hat_np = R_hat.detach().cpu().numpy().astype(np.float32)
    z_np = np.stack(z_list, axis=0).astype(np.float32)
    return R_hat_np, z_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input RGB image")
    ap.add_argument("--model", required=True, help="Model checkpoint .pt from train_model.py")
    ap.add_argument("--outdir", default="geninv_report", help="Output directory")
    ap.add_argument("--base_quality", type=int, default=50, help="JPEG quality for base codec proxy")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--err_gain", type=float, default=6.0)
    ap.add_argument("--residual_jpeg", type=int, nargs="+", default=[90, 95], help="Qualities for true residual JPEG")
    ap.add_argument("--corr_jpeg", type=int, nargs="+", default=[90, 95], help="Qualities for correction JPEG")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.model, map_location=device)
    # Prefer checkpoint config unless user overrides explicitly
    zdim = int(ckpt.get("config", {}).get("zdim", args.zdim))
    ch = int(ckpt.get("config", {}).get("ch", args.ch))

    E = TinyEncoder(zdim=zdim, ch=ch).to(device)
    G = TinyGenerator(zdim=zdim, ch=ch).to(device)
    E.load_state_dict(ckpt["E_state"])
    G.load_state_dict(ckpt["G_state"])
    E.eval(); G.eval()

    # Load image + base proxy
    orig_rgb = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    base_rgb = jpeg_encode_decode_rgb(orig_rgb, quality=args.base_quality)

    save_rgb_png(os.path.join(args.outdir, "orig_rgb.png"), orig_rgb)
    save_rgb_png(os.path.join(args.outdir, "base_rgb.png"), base_rgb)

    # Save base codec file to account bytes
    base_codec_path = os.path.join(args.outdir, f"base_codec_q{args.base_quality}.jpg")
    Image.fromarray(orig_rgb).save(base_codec_path, quality=int(args.base_quality), subsampling=0, optimize=True)
    base_bytes = os.path.getsize(base_codec_path)

    Y_orig = rgb_to_y(orig_rgb)
    Y_base = rgb_to_y(base_rgb)
    R_true = (Y_orig - Y_base).astype(np.float32)

    H, W = Y_orig.shape
    save_gray01_png(os.path.join(args.outdir, "Y_orig.png"), Y_orig)
    save_gray01_png(os.path.join(args.outdir, "Y_base.png"), Y_base)

    # Visuals
    r_scale = float(np.max(np.abs(R_true)) + 1e-9)
    save_gray01_png(os.path.join(args.outdir, "true_residual_visual.png"), residual_visual(R_true, r_scale))
    err_base_u8 = error_vis_centered(Y_orig - Y_base, gain=args.err_gain)
    Image.fromarray(err_base_u8, mode="L").save(os.path.join(args.outdir, "err_base_vis.png"))

    # ---- A) True residual stored as JPEG90/95 and applied ----
    results = []
    results.append({
        "method": f"Base only (JPEG q{args.base_quality})",
        "base_bytes": base_bytes,
        "extra_bytes": 0,
        "total_bytes": base_bytes,
        "Y_PSNR": psnr01(Y_base, Y_orig),
        "Y_MAE": mae01(Y_base, Y_orig),
    })

    Rq = quantize_residual_int8(R_true)
    for q in args.residual_jpeg:
        res_path = os.path.join(args.outdir, f"true_residual_centered_JPEG{q}.jpg")
        save_centered_int8_as_jpeg(res_path, Rq, quality=q)
        extra = os.path.getsize(res_path)

        Rq_dec = load_centered_jpeg_as_int8(res_path)
        R_dec = dequantize_residual_int8(Rq_dec)
        Y_rec = np.clip(Y_base + R_dec, 0.0, 1.0).astype(np.float32)

        save_gray01_png(os.path.join(args.outdir, f"Y_true_residual_applied_JPEG{q}.png"), Y_rec)
        err_u8 = error_vis_centered(Y_orig - Y_rec, gain=args.err_gain)
        Image.fromarray(err_u8, mode="L").save(os.path.join(args.outdir, f"err_true_residual_JPEG{q}_vis.png"))

        results.append({
            "method": f"True residual JPEG{q} + apply",
            "base_bytes": base_bytes,
            "extra_bytes": extra,
            "total_bytes": base_bytes + extra,
            "Y_PSNR": psnr01(Y_rec, Y_orig),
            "Y_MAE": mae01(Y_rec, Y_orig),
        })

    # ---- B) Latent predicted residual (amortized) ----
    R_hat, z_list = predict_full_residual_and_latents(E, G, Y_base, R_true, patch=args.patch, device=device)
    Y_latent_only = np.clip(Y_base + R_hat, 0.0, 1.0).astype(np.float32)

    save_gray01_png(os.path.join(args.outdir, "pred_residual_visual.png"), residual_visual(R_hat, r_scale))
    save_gray01_png(os.path.join(args.outdir, "Y_latent_only.png"), Y_latent_only)
    err_u8 = error_vis_centered(Y_orig - Y_latent_only, gain=args.err_gain)
    Image.fromarray(err_u8, mode="L").save(os.path.join(args.outdir, "err_latent_only_vis.png"))

    # Pack latent payload (raw + zlib size accounting)
    z_max = float(np.max(np.abs(z_list)) + 1e-9)
    z_scale = z_max / 127.0
    latent_payload = pack_latents_int8(z_list, z_scale, args.patch, H, W)
    latent_raw_path = os.path.join(args.outdir, "latent_payload_raw.bin")
    with open(latent_raw_path, "wb") as f:
        f.write(latent_payload)

    latent_raw_bytes = os.path.getsize(latent_raw_path)
    latent_zlib_bytes = zlib_size(latent_payload, level=9)

    # For convenience, also write the zlib-compressed file
    import zlib
    latent_zlib_path = os.path.join(args.outdir, "latent_payload_zlib.bin")
    with open(latent_zlib_path, "wb") as f:
        f.write(zlib.compress(latent_payload, level=9))

    # Latent-only result (counting zlib bytes as "what you'd transmit")
    results.append({
        "method": f"Latent-only + apply (zdim={zdim}, patch={args.patch})",
        "base_bytes": base_bytes,
        "extra_bytes": latent_zlib_bytes,
        "total_bytes": base_bytes + latent_zlib_bytes,
        "Y_PSNR": psnr01(Y_latent_only, Y_orig),
        "Y_MAE": mae01(Y_latent_only, Y_orig),
    })

    # ---- Correction layer: Corr = true - predicted residual ----
    Corr = (R_true - R_hat).astype(np.float32)
    Corrq = quantize_residual_int8(Corr)
    corr_scale = float(np.max(np.abs(Corr)) + 1e-9)
    save_gray01_png(os.path.join(args.outdir, "correction_visual.png"), residual_visual(Corr, corr_scale))

    for q in args.corr_jpeg:
        corr_path = os.path.join(args.outdir, f"correction_centered_JPEG{q}.jpg")
        save_centered_int8_as_jpeg(corr_path, Corrq, quality=q)
        corr_bytes = os.path.getsize(corr_path)

        Corrq_dec = load_centered_jpeg_as_int8(corr_path)
        Corr_dec = dequantize_residual_int8(Corrq_dec)

        Y_lat_corr = np.clip(Y_base + R_hat + Corr_dec, 0.0, 1.0).astype(np.float32)
        save_gray01_png(os.path.join(args.outdir, f"Y_latent_plus_corr_JPEG{q}.png"), Y_lat_corr)
        err_u8 = error_vis_centered(Y_orig - Y_lat_corr, gain=args.err_gain)
        Image.fromarray(err_u8, mode="L").save(os.path.join(args.outdir, f"err_latent_plus_corr_JPEG{q}_vis.png"))

        results.append({
            "method": f"Latent + correction JPEG{q} + apply",
            "base_bytes": base_bytes,
            "extra_bytes": latent_zlib_bytes + corr_bytes,
            "total_bytes": base_bytes + latent_zlib_bytes + corr_bytes,
            "Y_PSNR": psnr01(Y_lat_corr, Y_orig),
            "Y_MAE": mae01(Y_lat_corr, Y_orig),
        })

    # ---- Write results.csv + metrics.json ----
    results_csv = os.path.join(args.outdir, "results.csv")
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    metrics = {
        "image": args.image,
        "base_quality": args.base_quality,
        "patch": args.patch,
        "zdim": zdim,
        "ch": ch,
        "latent_raw_bytes": latent_raw_bytes,
        "latent_zlib_bytes": latent_zlib_bytes,
        "latent_scale": z_scale,
        "note": "Encode-time z is computed using true residual (amortized inversion). To simulate a codec, you would transmit the packed latents and decode to regenerate residual.",
        "results": results,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Montage ----
    try:
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        def show(ax, p, t):
            ax.imshow(PILImage.open(p), cmap="gray")
            ax.set_title(t, fontsize=8)
            ax.axis("off")

        fig = plt.figure(figsize=(14, 10))
        axs = fig.subplots(3, 4).flatten()

        axs[0].imshow(PILImage.open(os.path.join(args.outdir, "orig_rgb.png"))); axs[0].set_title("orig_rgb"); axs[0].axis("off")
        axs[1].imshow(PILImage.open(os.path.join(args.outdir, "base_rgb.png"))); axs[1].set_title(f"base_rgb q{args.base_quality}"); axs[1].axis("off")
        show(axs[2], os.path.join(args.outdir, "Y_orig.png"), "Y_orig")
        show(axs[3], os.path.join(args.outdir, "Y_base.png"), "Y_base")

        show(axs[4], os.path.join(args.outdir, "true_residual_visual.png"), "true residual (vis)")
        show(axs[5], os.path.join(args.outdir, "pred_residual_visual.png"), "pred residual (vis)")
        show(axs[6], os.path.join(args.outdir, "correction_visual.png"), "correction (vis)")
        show(axs[7], os.path.join(args.outdir, "err_base_vis.png"), "err base (vis)")

        show(axs[8], os.path.join(args.outdir, "Y_latent_only.png"), "Y latent-only")
        show(axs[9], os.path.join(args.outdir, "err_latent_only_vis.png"), "err latent-only (vis)")

        # Add the best quality examples if present
        tr = os.path.join(args.outdir, "Y_true_residual_applied_JPEG95.png")
        lc = os.path.join(args.outdir, "Y_latent_plus_corr_JPEG95.png")
        if os.path.exists(tr): show(axs[10], tr, "Y true residual JPEG95")
        else: axs[10].axis("off")
        if os.path.exists(lc): show(axs[11], lc, "Y latent+corr JPEG95")
        else: axs[11].axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "montage.png"), dpi=180)
        plt.close(fig)
    except Exception as e:
        print("montage skipped:", e)

    # ---- Zip report ----
    zip_path = os.path.join(args.outdir, "report.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(args.outdir):
            for fn in files:
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, args.outdir)
                z.write(p, arcname=os.path.join("report", arc))

    print("done.")
    print("outdir:", args.outdir)
    print("report zip:", zip_path)


if __name__ == "__main__":
    main()
