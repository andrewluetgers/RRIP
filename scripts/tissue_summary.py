#!/usr/bin/env python3
"""
Generate an HTML summary page for WSI tissue detection results.

Stitches a low-res thumbnail for each slide, overlays tissue contours
from the .tissue.json, and produces a single self-contained HTML file.

Usage:
    uv run python scripts/tissue_summary.py --dzi-dir ~/dev/data/WSI/dzi --out /tmp/tissue_summary.html
"""

import argparse
import base64
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


def stitch_level(files_dir: str, level: int) -> Image.Image:
    """Stitch all tiles at a given level into one image."""
    level_dir = os.path.join(files_dir, str(level))
    tiles = {}
    max_tx = max_ty = 0
    for f in os.listdir(level_dir):
        if not f.endswith('.jpeg'):
            continue
        tx, ty = f.replace('.jpeg', '').split('_')
        tx, ty = int(tx), int(ty)
        tiles[(tx, ty)] = os.path.join(level_dir, f)
        max_tx = max(max_tx, tx)
        max_ty = max(max_ty, ty)

    tile_size = 256
    canvas = Image.new('RGB', ((max_tx + 1) * tile_size, (max_ty + 1) * tile_size),
                        (255, 255, 255))
    for (tx, ty), path in tiles.items():
        img = Image.open(path)
        canvas.paste(img, (tx * tile_size, ty * tile_size))

    return canvas


def img_to_data_uri(img: Image.Image, fmt: str = 'JPEG', quality: int = 85) -> str:
    """Convert PIL image to a base64 data URI."""
    buf = BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
    return f'data:{mime};base64,{b64}'


def build_svg_overlay(meta: dict, thumb_w: int, thumb_h: int) -> str:
    """Build an SVG string scaled to the thumbnail dimensions.

    Contours are in normalized (0–1) coordinates, so we just multiply
    by the thumbnail pixel dimensions for perfect alignment.
    """
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{thumb_w}" height="{thumb_h}" '
        f'viewBox="0 0 {thumb_w} {thumb_h}" '
        f'style="position:absolute;top:0;left:0">')

    # Bounding polygons (light blue, dashed — outermost)
    for contour in meta.get('bound_contours_norm', []):
        pts = ' '.join(f'{x * thumb_w:.1f},{y * thumb_h:.1f}' for x, y in contour)
        parts.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(60,140,220,0.08)" stroke="#3c8cdc" stroke-width="1.5" '
            f'stroke-dasharray="4,2"/>')

    # Margin contours
    for contour in meta.get('margin_contours_norm', []):
        pts = ' '.join(f'{x * thumb_w:.1f},{y * thumb_h:.1f}' for x, y in contour)
        parts.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(0,180,0,0.12)" stroke="#00aa00" stroke-width="1.5"/>')

    # Small contours (yellow — below min_area threshold)
    for contour in meta.get('small_contours_norm', []):
        pts = ' '.join(f'{x * thumb_w:.1f},{y * thumb_h:.1f}' for x, y in contour)
        parts.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(220,200,0,0.3)" stroke="#ccaa00" stroke-width="1"/>')

    # Tissue contours
    for contour in meta.get('tissue_contours_norm', []):
        pts = ' '.join(f'{x * thumb_w:.1f},{y * thumb_h:.1f}' for x, y in contour)
        parts.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(220,40,40,0.18)" stroke="#cc0000" stroke-width="2"/>')

    parts.append('</svg>')
    return '\n'.join(parts)


def generate_html(dzi_dir: str, out_path: str, thumb_level: int = 11):
    """Generate the HTML summary page."""
    slides = sorted(
        f.replace('.dzi', '')
        for f in os.listdir(dzi_dir)
        if f.endswith('.dzi')
    )

    cards_html = []
    for slide_name in slides:
        tissue_path = os.path.join(dzi_dir, f'{slide_name}.tissue.json')
        if not os.path.exists(tissue_path):
            continue

        with open(tissue_path) as f:
            meta = json.load(f)

        files_dir = os.path.join(dzi_dir, f'{slide_name}_files')

        # Use the same level that detection was run on so contours align
        detect_level = meta['detect_level']
        thumb = stitch_level(files_dir, detect_level)
        tw, th = thumb.size
        # Scale down if too large
        max_thumb = 600
        if max(tw, th) > max_thumb:
            scale = max_thumb / max(tw, th)
            thumb = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
            tw, th = thumb.size

        data_uri = img_to_data_uri(thumb)
        svg_overlay = build_svg_overlay(meta, tw, th)

        # Load debug images
        debug_imgs = {}
        # Large images (JPEG) — resize to thumbnail size
        for img_name in ('gray', 'threshold', 'mask', 'margin_mask'):
            img_path = os.path.join(dzi_dir, f'{slide_name}.tissue_{img_name}.jpg')
            if os.path.exists(img_path):
                dbg = Image.open(img_path)
                dbg = dbg.resize((tw, th), Image.LANCZOS)
                debug_imgs[img_name] = img_to_data_uri(dbg)
        # Small pixel-exact images (PNG) — resize with NEAREST to preserve pixels
        for img_name in ('tissue_bitmap', 'margin_bitmap'):
            img_path = os.path.join(dzi_dir, f'{slide_name}.tissue_{img_name}.png')
            if os.path.exists(img_path):
                dbg = Image.open(img_path)
                dbg = dbg.resize((tw, th), Image.NEAREST)
                debug_imgs[img_name] = img_to_data_uri(dbg)
        # Color grid — resize with bilinear to show smooth gradients
        cg_path = os.path.join(dzi_dir, f'{slide_name}.tissue_color_grid.png')
        if os.path.exists(cg_path):
            dbg = Image.open(cg_path)
            dbg = dbg.resize((tw, th), Image.BILINEAR)
            debug_imgs['color_grid'] = img_to_data_uri(dbg)

        # Short name for display
        short_name = slide_name[:30] + '...' if len(slide_name) > 30 else slide_name

        # Stats
        max_level = meta['max_level']
        l0_total = meta['l0_total_tiles']
        l0_included = meta['l0_included_tiles']
        l0_skipped = meta['l0_skipped_tiles']
        skip_pct = meta['l0_skip_pct']
        tissue_pct = round(100 * meta['tissue_detect_tiles'] / meta['total_detect_tiles'], 1)
        threshold = meta['threshold_std']
        px_thresh_blank = meta.get('pixel_threshold_blank', '?')
        px_thresh_center = meta.get('pixel_threshold_center', '?')
        px_thresh_artifact = meta.get('pixel_threshold_artifact', '?')
        has_dark = meta.get('has_dark_peak', False)
        n_contours = len(meta.get('tissue_contours_norm', []))
        grid_w, grid_h = meta['detect_grid']
        slide_w = meta['slide_w_px']
        slide_h = meta['slide_h_px']

        # Build debug image strip (smaller versions)
        debug_strip = ''
        dbg_scale = 0.5
        dbg_w, dbg_h = int(tw * dbg_scale), int(th * dbg_scale)
        labels = {
            'gray': 'Grayscale', 'threshold': 'Thresholds',
            'mask': 'Tissue Mask', 'margin_mask': 'Margin Mask',
            'tissue_bitmap': 'Tissue Bitmap', 'margin_bitmap': 'Margin Bitmap',
            'color_grid': 'Blank Colors',
        }
        for img_name in ('gray', 'threshold', 'mask', 'margin_mask',
                         'tissue_bitmap', 'margin_bitmap', 'color_grid'):
            if img_name in debug_imgs:
                debug_strip += (
                    f'<div class="debug-img">'
                    f'<img src="{debug_imgs[img_name]}" width="{dbg_w}" height="{dbg_h}">'
                    f'<span>{labels[img_name]}</span>'
                    f'</div>')

        card = f"""
    <div class="card">
      <h3 title="{slide_name}">{short_name}</h3>
      <div class="thumb-container" style="width:{tw}px;height:{th}px">
        <img src="{data_uri}" width="{tw}" height="{th}">
        {svg_overlay}
      </div>
      <div class="debug-strip">{debug_strip}</div>
      <table>
        <tr><td>Dimensions</td><td>{slide_w:,} x {slide_h:,} px</td></tr>
        <tr><td>L0 tiles</td><td>{l0_total:,}</td></tr>
        <tr><td>Tissue tiles</td><td>{l0_included:,} ({100 - skip_pct:.1f}%)</td></tr>
        <tr><td>Blank tiles</td><td>{l0_skipped:,} ({skip_pct}%)</td></tr>
        <tr><td>Tissue regions</td><td>{n_contours}</td></tr>
        <tr><td>Detection</td><td>L{meta['detect_level']} ({grid_w}x{grid_h}), tile std &gt; {threshold}</td></tr>
        <tr><td>Pixel thresh</td><td>blank&ge;{px_thresh_blank} center={px_thresh_center} artifact&le;{px_thresh_artifact}{' (dark)' if has_dark else ''}</td></tr>
        <tr><td>Margin</td><td>{meta['margin']} L0 px</td></tr>
      </table>
    </div>"""
        cards_html.append(card)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>WSI Tissue Detection Summary</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      padding: 24px;
    }}
    h1 {{
      text-align: center;
      margin-bottom: 8px;
      font-size: 24px;
      color: #fff;
    }}
    .subtitle {{
      text-align: center;
      color: #888;
      margin-bottom: 24px;
      font-size: 14px;
    }}
    .legend {{
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-bottom: 24px;
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 16px;
      height: 16px;
      border-radius: 3px;
    }}
    .grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      justify-content: center;
    }}
    .card {{
      background: #16213e;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 16px;
      max-width: 640px;
    }}
    .card h3 {{
      font-size: 13px;
      font-family: monospace;
      color: #aaa;
      margin-bottom: 10px;
      word-break: break-all;
    }}
    .thumb-container {{
      position: relative;
      margin-bottom: 12px;
      border: 1px solid #333;
      border-radius: 4px;
      overflow: hidden;
    }}
    .thumb-container img {{
      display: block;
    }}
    .debug-strip {{
      display: flex;
      gap: 4px;
      margin-bottom: 10px;
      overflow-x: auto;
    }}
    .debug-img {{
      position: relative;
      flex-shrink: 0;
    }}
    .debug-img img {{
      display: block;
      border-radius: 3px;
      border: 1px solid #333;
    }}
    .debug-img span {{
      position: absolute;
      bottom: 2px;
      left: 4px;
      font-size: 10px;
      color: #fff;
      background: rgba(0,0,0,0.6);
      padding: 1px 4px;
      border-radius: 2px;
    }}
    table {{
      width: 100%;
      font-size: 13px;
      border-collapse: collapse;
    }}
    td {{
      padding: 3px 8px;
      border-bottom: 1px solid #2a2a4a;
    }}
    td:first-child {{
      color: #888;
      width: 130px;
    }}
    td:last-child {{
      font-family: monospace;
      color: #ccc;
    }}
  </style>
</head>
<body>
  <h1>WSI Tissue Detection Summary</h1>
  <div class="subtitle">{len(cards_html)} slides from {dzi_dir}</div>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(220,40,40,0.4);border:2px solid #cc0000"></div>
      Tissue
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(0,180,0,0.2);border:2px solid #00aa00"></div>
      Margin
    </div>
  </div>
  <div class="grid">
    {''.join(cards_html)}
  </div>
</body>
</html>"""

    with open(out_path, 'w') as f:
        f.write(html)
    print(f'Written: {out_path} ({len(cards_html)} slides)')


def main():
    ap = argparse.ArgumentParser(
        description='Generate HTML summary of WSI tissue detection')
    ap.add_argument('--dzi-dir', required=True,
                    help='Directory containing .dzi and .tissue.json files')
    ap.add_argument('--out', default='/tmp/tissue_summary.html',
                    help='Output HTML path')
    ap.add_argument('--thumb-level', type=int, default=11,
                    help='DZI level for thumbnails (default: 11)')
    args = ap.parse_args()

    generate_html(args.dzi_dir, args.out, args.thumb_level)


if __name__ == '__main__':
    main()
