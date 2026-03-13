import os, math
import numpy as np
from PIL import Image

def rgb_to_y(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    return (0.299*r + 0.587*g + 0.114*b).astype(np.float32)

def quantize_residual_int8(residual_float):
    q = np.round(residual_float * 255.0).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int16)
    return q

def signed_to_u8_centered(q):
    u = (q.astype(np.int16) + 128)
    return np.clip(u, 0, 255).astype(np.uint8)

def save_gray(path, arr01):
    u8 = (np.clip(arr01,0,1)*255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)

def save_gray_u8(path, u8):
    Image.fromarray(u8, mode="L").save(path)

def error_vis_u8(err, gain=6.0):
    e = np.clip(np.round(err * 255.0 * gain), -127, 127).astype(np.int16)
    return signed_to_u8_centered(e)

def main():
    in_path = "L0-1024.jpg"
    out_dir = "geninv_demo"
    os.makedirs(out_dir, exist_ok=True)

    orig_rgb = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8)
    Y_orig = rgb_to_y(orig_rgb)

    # Base compression proxy (JPEG q=50)
    base_path = os.path.join(out_dir, "base_q50.jpg")
    Image.fromarray(orig_rgb).save(base_path, quality=50, subsampling=0, optimize=True)
    base_rgb = np.array(Image.open(base_path).convert("RGB"), dtype=np.uint8)
    Y_base = rgb_to_y(base_rgb)

    # Residual in Y
    R = (Y_orig - Y_base).astype(np.float32)
    Rq = quantize_residual_int8(R)
    R_u8 = signed_to_u8_centered(Rq)

    # Save some views
    save_gray(os.path.join(out_dir, "Y_orig.png"), Y_orig)
    save_gray(os.path.join(out_dir, "Y_base.png"), Y_base)
    save_gray_u8(os.path.join(out_dir, "err_base_vis.png"), error_vis_u8(Y_orig - Y_base))
    save_gray_u8(os.path.join(out_dir, "residual_Y_int8_centered.png"), R_u8)

    # “float residual” visualization (purely for viewing)
    # normalize by max abs for visibility
    m = float(np.max(np.abs(R)) + 1e-9)
    vis = (R / (2*m)) + 0.5
    save_gray(os.path.join(out_dir, "residual_Y_float_vis.png"), vis)

    print("wrote:", out_dir)

if __name__ == "__main__":
    main()