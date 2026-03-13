import os, json, zlib, math
import numpy as np
from PIL import Image

def rgb_to_y(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    return (0.299*r + 0.587*g + 0.114*b).astype(np.float32)

def dct_matrix(N):
    M = np.zeros((N, N), dtype=np.float32)
    factor = math.pi / N
    for k in range(N):
        alpha = math.sqrt(1.0/N) if k == 0 else math.sqrt(2.0/N)
        for n in range(N):
            M[k, n] = alpha * math.cos((n + 0.5) * k * factor)
    return M

def dct2(patch, M):
    return M @ patch @ M.T

def idct2(coeff, M):
    return M.T @ coeff @ M

def save_gray(path, arr01):
    u8 = (np.clip(arr01,0,1)*255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)

def main():
    in_path = "L0-1024.jpg"
    work_dir = "geninv_demo"
    out_dir = os.path.join(work_dir, "patch_demo")
    os.makedirs(out_dir, exist_ok=True)

    # Load orig and base
    orig_rgb = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8)
    base_rgb = np.array(Image.open(os.path.join(work_dir, "base_q50.jpg")).convert("RGB"), dtype=np.uint8)

    Y_orig = rgb_to_y(orig_rgb)
    Y_base = rgb_to_y(base_rgb)
    R = (Y_orig - Y_base).astype(np.float32)

    # pick a patch location (hard-coded like the demo)
    patch = 64
    y0, x0 = 256, 256
    rp = R[y0:y0+patch, x0:x0+patch]

    # latent = top-left 8x8 DCT coeffs (64 numbers)
    Nkeep = 8
    M = dct_matrix(patch)
    C = dct2(rp, M)
    z = C[:Nkeep, :Nkeep].copy()

    # quantize to int8 with global scale from this patch (demo simple)
    m = float(np.max(np.abs(z)) + 1e-9)
    scale = m / 127.0
    zq = np.clip(np.round(z/scale), -127, 127).astype(np.int8)
    z_deq = zq.astype(np.float32) * scale

    # reconstruct residual patch
    C2 = np.zeros((patch, patch), dtype=np.float32)
    C2[:Nkeep, :Nkeep] = z_deq
    rhat = idct2(C2, M).astype(np.float32)

    # reconstructed Y patch
    yb = Y_base[y0:y0+patch, x0:x0+patch]
    yrec = np.clip(yb + rhat, 0.0, 1.0)

    # write images
    save_gray(os.path.join(out_dir, "Y_base_patch.png"), yb)
    save_gray(os.path.join(out_dir, "Y_orig_patch.png"), Y_orig[y0:y0+patch, x0:x0+patch])
    # visualize residual
    save_gray(os.path.join(out_dir, "R_true_vis.png"), (rp / (2*(np.max(np.abs(rp))+1e-9))) + 0.5)
    save_gray(os.path.join(out_dir, "R_hat_vis.png"), (rhat / (2*(np.max(np.abs(rp))+1e-9))) + 0.5)
    save_gray(os.path.join(out_dir, "Y_recon_patch.png"), yrec)

    # pack latent payload
    header = {"patch": patch, "Nkeep": Nkeep, "scale": scale, "y0": y0, "x0": x0, "dtype": "int8"}
    hb = json.dumps(header).encode("utf-8")
    payload = len(hb).to_bytes(4, "little") + hb + zq.tobytes()
    with open(os.path.join(out_dir, "latent_raw.bin"), "wb") as f:
        f.write(payload)
    with open(os.path.join(out_dir, "latent_zlib.bin"), "wb") as f:
        f.write(zlib.compress(payload, level=9))

    print("wrote:", out_dir)

if __name__ == "__main__":
    main()