\
#!/usr/bin/env python3
"""
wsi_residual_tool.py

Build a DeepZoom pyramid with pyvips, then encode luma residual JPEGs for the two highest-res levels (L0/L1)
conditioned on covering L2 tiles. Keeps all levels L2+ unchanged.

Example:
  python wsi_residual_tool.py build --slide /path/to/slide.svs --out out --tile 256 --q 90
  python wsi_residual_tool.py encode --pyramid out/baseline_pyramid --out out --tile 256 --resq 32 --max-parents 200
  python wsi_residual_tool.py pack --residuals out/residuals_q32 --out out/residual_packs

Requires:
  pip install pyvips Pillow numpy scikit-image requests tqdm openslide-python
and system libs:
  libvips (+ openslide support), openslide
"""
import argparse, pathlib, shutil, re, json, random, math
import numpy as np
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("Warning: lz4 not installed. Pack files will not be compressed. Install with: pip install lz4")
from PIL import Image
from typing import Optional, Dict, Callable, Any
import sys, os
# Add evals/scripts to path for jpeg_encoder module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evals', 'scripts'))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_file, parse_encoder_arg


class DebugContext:
    """Debug instrumentation context with zero overhead when disabled."""

    def __init__(self, enabled: bool = False, output_dir: Optional[pathlib.Path] = None):
        self.enabled = enabled
        self.output_dir = output_dir
        self.callbacks: Dict[str, Callable] = {}
        self.data: Dict[str, Any] = {}

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a specific debug event."""
        if self.enabled:
            self.callbacks[event] = callback

    def emit(self, event: str, **kwargs):
        """Emit a debug event with data. Zero-cost when disabled."""
        if not self.enabled:
            return
        if event in self.callbacks:
            self.callbacks[event](**kwargs)

    def capture(self, key: str, value: Any):
        """Capture data for later use. Zero-cost when disabled."""
        if not self.enabled:
            return
        self.data[key] = value

def import_pyvips():
    import pyvips
    return pyvips

def dzsave(slide_path: str, out_prefix: pathlib.Path, tile_size=256, q=90):
    pyvips = import_pyvips()
    img = pyvips.Image.openslideload(slide_path) if hasattr(pyvips.Image, "openslideload") else pyvips.Image.new_from_file(slide_path, access="sequential")
    # cleanup
    for p in [out_prefix.with_suffix(".dzi"), out_prefix.parent / (out_prefix.name + "_files")]:
        if p.exists():
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
    img.dzsave(str(out_prefix), tile_size=tile_size, overlap=0, suffix=f".jpg[Q={q}]", depth="onepixel", layout="dz")

def parse_levels(files_dir: pathlib.Path):
    levels = sorted([int(p.name) for p in files_dir.iterdir() if p.is_dir()])
    max_level = max(levels)
    return levels, max_level, max_level, max_level-1, max_level-2  # L0,L1,L2 in DZ numbering

def tile_path(files_dir: pathlib.Path, level:int, x:int, y:int):
    return files_dir / str(level) / f"{x}_{y}.jpg"

def load_rgb(p: pathlib.Path):
    return np.array(Image.open(p).convert("RGB"))

def save_gray_jpeg(arr_u8: np.ndarray, p: pathlib.Path, q:int, encoder=JpegEncoder.LIBJPEG_TURBO):
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr_u8.astype(np.uint8), mode="L")
    encode_jpeg_to_file(img, p, int(q), encoder)

def rgb_to_ycbcr_bt601(rgb_u8):
    rgb = rgb_u8.astype(np.float32)
    R,G,B = rgb[...,0], rgb[...,1], rgb[...,2]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128.0
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128.0
    return Y, Cb, Cr

def ycbcr_to_rgb_bt601(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb
    return np.clip(np.stack([R,G,B], axis=-1), 0, 255).astype(np.uint8)

def encode(pyramid_prefix: pathlib.Path, out_dir: pathlib.Path, tile_size=256, resq=32, max_parents=None, sample_prob=0.0, debug_ctx: Optional[DebugContext] = None, encoder=JpegEncoder.LIBJPEG_TURBO):
    # Default to disabled debug context if not provided
    if debug_ctx is None:
        debug_ctx = DebugContext(enabled=False)

    files_dir = pyramid_prefix.parent / (pyramid_prefix.name + "_files")
    levels, max_level, L0, L1, L2 = parse_levels(files_dir)
    if max_level < 2:
        raise ValueError("Need at least 3 DeepZoom levels (L0/L1/L2). Rebuild with depth='onepixel' or 'onetile'.")
    out_res = out_dir / f"residuals_j{resq}"
    if out_res.exists(): shutil.rmtree(out_res)
    out_res.mkdir(parents=True, exist_ok=True)

    # Emit initialization event
    debug_ctx.emit('init', levels=levels, L0=L0, L1=L1, L2=L2, tile_size=tile_size, resq=resq)

    baseline_bytes = sum(f.stat().st_size for lv in levels for f in (files_dir/str(lv)).glob("*.jpg"))
    retained_bytes = sum(f.stat().st_size for lv in levels if lv <= L2 for f in (files_dir/str(lv)).glob("*.jpg"))

    l2_tiles=[]
    for f in (files_dir/str(L2)).glob("*.jpg"):
        m=re.match(r"(\d+)_(\d+)\.jpg$", f.name)
        if m: l2_tiles.append((int(m.group(1)), int(m.group(2))))
    if max_parents:
        random.shuffle(l2_tiles)
        l2_tiles = l2_tiles[:max_parents]

    UPSAMPLE = Image.Resampling.BILINEAR
    residual_bytes_L1=0
    residual_bytes_L0=0

    for (x2,y2) in l2_tiles:
        p2=tile_path(files_dir, L2, x2, y2)
        if not p2.exists(): continue
        l2=load_rgb(p2)

        # Debug: L2 tile loaded
        debug_ctx.emit('l2_loaded', x2=x2, y2=y2, tile=l2, path=p2)

        l1_pred = np.array(Image.fromarray(l2).resize((tile_size*2, tile_size*2), resample=UPSAMPLE))

        # Debug: L1 prediction mosaic created
        debug_ctx.emit('l1_prediction_mosaic', x2=x2, y2=y2, prediction=l1_pred)

        recon_l1=[[None,None],[None,None]]
        for dy in range(2):
            for dx in range(2):
                x1=x2*2+dx; y1=y2*2+dy
                p1=tile_path(files_dir, L1, x1, y1)
                if not p1.exists(): continue
                gt=load_rgb(p1)
                pred=l1_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]

                # Debug: L1 tile processing
                debug_ctx.emit('l1_tile_start', x1=x1, y1=y1, ground_truth=gt, prediction=pred)

                Yg,Cbg,Crg=rgb_to_ycbcr_bt601(gt)
                Yp,Cbp,Crp=rgb_to_ycbcr_bt601(pred)

                # Debug: YCbCr conversion
                debug_ctx.emit('l1_ycbcr', x1=x1, y1=y1, Y_gt=Yg, Y_pred=Yp, Cb_pred=Cbp, Cr_pred=Crp)

                Ry=Yg-Yp

                # Debug: Raw residual
                debug_ctx.emit('l1_residual_raw', x1=x1, y1=y1, residual=Ry)

                enc=np.clip(np.round(Ry+128.0),0,255).astype(np.uint8)

                # Debug: Encoded residual
                debug_ctx.emit('l1_residual_encoded', x1=x1, y1=y1, encoded=enc)

                rp=out_res/"L1"/f"{x2}_{y2}"/f"{x1}_{y1}.jpg"
                save_gray_jpeg(enc,rp,resq,encoder)
                residual_bytes_L1 += rp.stat().st_size

                # Debug: Saved residual
                debug_ctx.emit('l1_residual_saved', x1=x1, y1=y1, path=rp, size=rp.stat().st_size)

                r_dec=np.array(Image.open(rp).convert("L")).astype(np.float32)-128.0
                Yhat=np.clip(Yp+r_dec,0,255)
                recon_l1[dy][dx]=ycbcr_to_rgb_bt601(Yhat,Cbp,Crp)

                # Debug: L1 reconstruction
                debug_ctx.emit('l1_reconstructed', x1=x1, y1=y1, Y_recon=Yhat, rgb_recon=recon_l1[dy][dx])
        if any(recon_l1[dy][dx] is None for dy in range(2) for dx in range(2)):
            continue
        l1_mosaic=np.concatenate([np.concatenate(row,axis=1) for row in recon_l1], axis=0)

        # Debug: L1 mosaic complete
        debug_ctx.emit('l1_mosaic_complete', x2=x2, y2=y2, mosaic=l1_mosaic)

        l0_pred=np.array(Image.fromarray(l1_mosaic).resize((tile_size*4,tile_size*4), resample=UPSAMPLE))

        # Debug: L0 prediction mosaic
        debug_ctx.emit('l0_prediction_mosaic', x2=x2, y2=y2, prediction=l0_pred)

        for dy in range(4):
            for dx in range(4):
                x0=x2*4+dx; y0=y2*4+dy
                p0=tile_path(files_dir, L0, x0, y0)
                if not p0.exists(): continue
                gt=load_rgb(p0)
                pred=l0_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]

                # Debug: L0 tile processing
                debug_ctx.emit('l0_tile_start', x0=x0, y0=y0, ground_truth=gt, prediction=pred)

                Yg,Cbg,Crg=rgb_to_ycbcr_bt601(gt)
                Yp,Cbp,Crp=rgb_to_ycbcr_bt601(pred)

                # Debug: L0 YCbCr
                debug_ctx.emit('l0_ycbcr', x0=x0, y0=y0, Y_gt=Yg, Y_pred=Yp, Cb_pred=Cbp, Cr_pred=Crp)

                Ry=Yg-Yp

                # Debug: L0 raw residual
                debug_ctx.emit('l0_residual_raw', x0=x0, y0=y0, residual=Ry)

                enc=np.clip(np.round(Ry+128.0),0,255).astype(np.uint8)

                # Debug: L0 encoded residual
                debug_ctx.emit('l0_residual_encoded', x0=x0, y0=y0, encoded=enc)

                rp=out_res/"L0"/f"{x2}_{y2}"/f"{x0}_{y0}.jpg"
                save_gray_jpeg(enc,rp,resq,encoder)
                residual_bytes_L0 += rp.stat().st_size

                # Debug: L0 saved residual
                debug_ctx.emit('l0_residual_saved', x0=x0, y0=y0, path=rp, size=rp.stat().st_size)

    proposed_bytes = retained_bytes + residual_bytes_L1 + residual_bytes_L0
    summary = {
        "tile_size": tile_size,
        "residual_jpeg_q_L": resq,
        "dz_levels": {"L0": L0, "L1": L1, "L2": L2},
        "baseline_bytes_all_levels": baseline_bytes,
        "retained_bytes_L2plus": retained_bytes,
        "residual_bytes_L1": residual_bytes_L1,
        "residual_bytes_L0": residual_bytes_L0,
        "proposed_bytes": proposed_bytes,
        "compression_ratio": baseline_bytes/proposed_bytes,
        "savings_pct": (1-proposed_bytes/baseline_bytes)*100.0
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Debug: Encoding complete
    debug_ctx.emit('encoding_complete', summary=summary)

def pack_residuals(residuals_dir: pathlib.Path, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    l1_dir = residuals_dir / "L1"
    l0_dir = residuals_dir / "L0"
    if not l1_dir.exists() or not l0_dir.exists():
        raise FileNotFoundError("Expected residuals/L1 and residuals/L0 directories")

    parents = sorted({p.name for p in l1_dir.iterdir() if p.is_dir()} | {p.name for p in l0_dir.iterdir() if p.is_dir()})
    magic = b"RRIP"
    version = 1
    tile_size = 256

    for parent in parents:
        try:
            x2, y2 = [int(v) for v in parent.split("_")]
        except Exception as exc:
            raise ValueError(f"Invalid parent folder name: {parent}") from exc
        l1_parent = l1_dir / parent
        l0_parent = l0_dir / parent
        entries = []

        # L1: 4 entries
        for dy in range(2):
            for dx in range(2):
                x1 = x2 * 2 + dx
                y1 = y2 * 2 + dy
                p = l1_parent / f"{x1}_{y1}.jpg"
                if p.exists():
                    entries.append((1, dy * 2 + dx, p.read_bytes()))

        # L0: 16 entries
        for dy in range(4):
            for dx in range(4):
                x0 = x2 * 4 + dx
                y0 = y2 * 4 + dy
                p = l0_parent / f"{x0}_{y0}.jpg"
                if p.exists():
                    entries.append((0, dy * 4 + dx, p.read_bytes()))

        if not entries:
            continue

        index_offset = 24
        entry_size = 16
        index_size = len(entries) * entry_size
        data_offset = index_offset + index_size
        data = b"".join([e[2] for e in entries])

        header = [
            magic,
            version.to_bytes(2, "little"),
            tile_size.to_bytes(2, "little"),
            len(entries).to_bytes(4, "little"),
            index_offset.to_bytes(4, "little"),
            data_offset.to_bytes(4, "little"),
            (0).to_bytes(4, "little"),
        ]

        index = []
        cursor = 0
        for level_kind, idx, blob in entries:
            index.append(bytes([level_kind, idx, 0, 0]))
            index.append(cursor.to_bytes(4, "little"))
            index.append(len(blob).to_bytes(4, "little"))
            index.append((0).to_bytes(4, "little"))
            cursor += len(blob)

        # Assemble the uncompressed pack file
        pack_data = b"".join(header) + b"".join(index) + data

        # Always compress with LZ4 (it's required now)
        if not HAS_LZ4:
            raise ImportError("LZ4 is required. Install with: pip install lz4")

        # Use lz4.block for compatibility with Rust lz4_flex
        import lz4.block
        # Prepend size for lz4_flex::decompress_size_prepended
        compressed_data = len(pack_data).to_bytes(4, 'little') + lz4.block.compress(pack_data, mode='fast', compression=0, store_size=False)
        compression_ratio = len(pack_data) / len(compressed_data)
        savings = 100 * (1 - len(compressed_data) / len(pack_data))
        print(f"  {parent}.pack: {len(pack_data)//1024}KB → {len(compressed_data)//1024}KB (ratio: {compression_ratio:.2f}x, savings: {savings:.1f}%)")

        out_path = out_dir / f"{parent}.pack"
        out_path.write_bytes(compressed_data)

def main():
    ap=argparse.ArgumentParser()
    sp=ap.add_subparsers(dest="cmd", required=True)

    ap_b=sp.add_parser("build")
    ap_b.add_argument("--slide", required=True)
    ap_b.add_argument("--out", required=True)
    ap_b.add_argument("--tile", type=int, default=256)
    ap_b.add_argument("--q", type=int, default=90)

    ap_e=sp.add_parser("encode")
    ap_e.add_argument("--pyramid", required=True, help="Path prefix (without _files) e.g. out/baseline_pyramid")
    ap_e.add_argument("--out", required=True)
    ap_e.add_argument("--tile", type=int, default=256)
    ap_e.add_argument("--resq", type=int, default=32)
    ap_e.add_argument("--max-parents", type=int, default=None)
    ap_e.add_argument("--encoder", default="libjpeg-turbo",
                     help="JPEG encoder: libjpeg-turbo (default) or jpegli")

    ap_p=sp.add_parser("pack")
    ap_p.add_argument("--residuals", required=True, help="Path to residuals_qXX folder")
    ap_p.add_argument("--out", required=True, help="Output folder for packfiles")

    args=ap.parse_args()
    out=pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd=="build":
        dzsave(args.slide, out/"baseline_pyramid", tile_size=args.tile, q=args.q)
    elif args.cmd=="encode":
        enc = parse_encoder_arg(args.encoder)
        encode(pathlib.Path(args.pyramid), out, tile_size=args.tile, resq=args.resq, max_parents=args.max_parents, encoder=enc)
    elif args.cmd=="pack":
        pack_residuals(pathlib.Path(args.residuals), pathlib.Path(args.out))

if __name__=="__main__":
    main()
