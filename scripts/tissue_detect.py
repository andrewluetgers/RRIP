#!/usr/bin/env python3
"""
Detect tissue regions in DZI pyramids and generate tile inclusion masks.

For each slide, analyzes a low-resolution level to classify tiles as tissue
or blank using per-tile grayscale standard deviation with an Otsu-derived
threshold (computed per slide to adapt to staining/scanner differences).

Outputs a .tissue.json metadata file per slide containing:
  - The detected threshold and detection parameters
  - A tile inclusion bitmask at the detection level
  - Functions to query any (level, tx, ty) for inclusion

Usage:
    # Detect tissue for all slides in a DZI directory
    uv run python scripts/tissue_detect.py --dzi-dir ~/dev/data/WSI/dzi

    # Detect for a single slide
    uv run python scripts/tissue_detect.py --dzi-dir ~/dev/data/WSI/dzi --slide 3DHISTECH-1

    # Custom margin (in detection-level tiles)
    uv run python scripts/tissue_detect.py --dzi-dir ~/dev/data/WSI/dzi --margin 3
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image


def detect_tissue(files_dir: str, max_level: int, levels: list[int],
                  margin: int = 512, detect_offset: int = 3) -> dict:
    """Detect tissue tiles in a DZI pyramid.

    Args:
        files_dir: Path to the {slide}_files directory.
        max_level: Highest-resolution level number (L0).
        levels: Sorted list of available level numbers.
        margin: Dilation radius in L0 pixels (default 512 = 2 L0 tiles).
        detect_offset: How many levels below max_level to detect at.

    Returns:
        Metadata dict with detection results.
    """
    # Pick detection level
    detect_level = max_level - detect_offset
    while detect_level not in levels and detect_offset > 0:
        detect_offset -= 1
        detect_level = max_level - detect_offset
    if detect_level not in levels:
        detect_level = levels[0]

    level_dir = os.path.join(files_dir, str(detect_level))
    tile_size = 256

    # Read all tiles at detection level, compute per-tile stats
    tile_stats = {}
    max_tx = max_ty = 0
    for f in os.listdir(level_dir):
        if not f.endswith('.jpeg'):
            continue
        tx, ty = f.replace('.jpeg', '').split('_')
        tx, ty = int(tx), int(ty)
        max_tx = max(max_tx, tx)
        max_ty = max(max_ty, ty)

        path = os.path.join(level_dir, f)
        img = np.array(Image.open(path))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        tile_stats[(tx, ty)] = {
            'std': float(gray.std()),
            'mean': float(gray.mean()),
        }

    grid_w = max_tx + 1
    grid_h = max_ty + 1
    n_tiles = len(tile_stats)

    # Compute Otsu threshold on the std distribution
    stds = np.array([s['std'] for s in tile_stats.values()])
    # Scale to u8 for Otsu (multiply by 10, cap at 255)
    std_u8 = np.clip(stds * 10, 0, 255).astype(np.uint8)
    otsu_raw, _ = cv2.threshold(std_u8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = otsu_raw / 10.0

    # Build binary tissue mask at detection level
    tissue_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for (tx, ty), s in tile_stats.items():
        if s['std'] > threshold:
            tissue_mask[ty, tx] = 1

    tissue_count = int(tissue_mask.sum())

    # Compute L0 tile coverage
    scale = 2 ** (max_level - detect_level)

    # Dilate tile mask for margin (convert L0 px margin to detect-level tiles)
    margin_tiles = max(1, int(math.ceil(margin / (tile_size * scale))))
    if margin_tiles > 0:
        k = 2 * margin_tiles + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_with_margin = cv2.dilate(tissue_mask, kernel, iterations=1)
    else:
        mask_with_margin = tissue_mask.copy()

    margin_count = int(mask_with_margin.sum())

    # Read DZI manifest to get actual dimensions
    # (infer from L0 tile count)
    l0_dir = os.path.join(files_dir, str(max_level))
    l0_tiles = set()
    for f in os.listdir(l0_dir):
        if not f.endswith('.jpeg'):
            continue
        x, y = f.replace('.jpeg', '').split('_')
        l0_tiles.add((int(x), int(y)))

    l0_max_x = max(x for x, y in l0_tiles) if l0_tiles else 0
    l0_max_y = max(y for x, y in l0_tiles) if l0_tiles else 0
    l0_grid_w = l0_max_x + 1
    l0_grid_h = l0_max_y + 1
    l0_total = len(l0_tiles)

    # Count L0 tiles included
    l0_included = 0
    for ty in range(grid_h):
        for tx in range(grid_w):
            if mask_with_margin[ty, tx]:
                for dy in range(scale):
                    for dx in range(scale):
                        l0x = tx * scale + dx
                        l0y = ty * scale + dy
                        if (l0x, l0y) in l0_tiles:
                            l0_included += 1

    # Encode included tiles as list
    included_tiles = []
    for ty in range(grid_h):
        for tx in range(grid_w):
            if mask_with_margin[ty, tx]:
                included_tiles.append([tx, ty])

    # --- Pixel-level contour extraction ---
    # Stitch detection level into RGB canvas, capped at ~4K max dimension.
    full_h = grid_h * tile_size
    full_w = grid_w * tile_size
    max_canvas = 4096
    if max(full_w, full_h) > max_canvas:
        ds = max_canvas / max(full_w, full_h)
        canvas_w_target = int(full_w * ds)
        canvas_h_target = int(full_h * ds)
        # Compute per-tile target size
        tile_ds = ds
    else:
        canvas_w_target = full_w
        canvas_h_target = full_h
        tile_ds = 1.0

    canvas_rgb = np.full((canvas_h_target, canvas_w_target, 3), 255,
                         dtype=np.uint8)
    for f in os.listdir(level_dir):
        if not f.endswith('.jpeg'):
            continue
        tx, ty = f.replace('.jpeg', '').split('_')
        tx, ty = int(tx), int(ty)
        path = os.path.join(level_dir, f)
        img = np.array(Image.open(path))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if tile_ds < 1.0:
            new_w = max(1, int(img.shape[1] * tile_ds))
            new_h = max(1, int(img.shape[0] * tile_ds))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0 = int(ty * tile_size * tile_ds)
        x0 = int(tx * tile_size * tile_ds)
        h, w = img.shape[:2]
        # Clamp to canvas bounds
        h = min(h, canvas_h_target - y0)
        w = min(w, canvas_w_target - x0)
        if h > 0 and w > 0:
            canvas_rgb[y0:y0 + h, x0:x0 + w] = img[:h, :w]

    # Convert to grayscale
    gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
    canvas_h, canvas_w = gray.shape

    # --- Multi-threshold histogram analysis ---
    # Exclude pure white (255) pixels — these are tile padding that
    # extends beyond the actual slide area, not slide content.
    hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    # Zero out the padding bin so it doesn't influence thresholds
    hist = hist_full.copy()
    hist[255] = 0
    total_px = int(hist.sum())  # only count slide pixels

    # 1. Find the WHITE/BLANK peak and the inflection point before it.
    #    The blank peak is the slide's actual background (e.g. 232-238),
    #    NOT the tile padding at 255.
    smooth_hist = np.convolve(hist, np.ones(7) / 7, mode='same')
    white_peak = int(np.argmax(smooth_hist[150:])) + 150
    # Compute the gradient (rate of change) of the smoothed histogram
    grad = np.diff(smooth_hist)
    # Walk left from the peak. The blank floor is where the gradient
    # drops below 10% of the max gradient in the peak region.
    # This finds the inflection point — where the steep rise begins.
    peak_region_grad = grad[max(white_peak - 15, 200):white_peak]
    if len(peak_region_grad) > 0:
        max_grad = peak_region_grad.max()
        grad_threshold = max_grad * 0.1
        blank_floor = white_peak
        for i in range(white_peak - 1, 150, -1):
            if grad[i] < grad_threshold:
                blank_floor = i + 1  # +1: the steep rise starts at i+1
                break
    else:
        blank_floor = white_peak - 5
    blank_floor = max(blank_floor, 200)  # safety: never below 200

    # 2. Find the DARK/ARTIFACT peak: check if there's significant mass
    #    near 0. Walk up from 0, accumulate until we pass any dark cluster.
    dark_mass = hist[:30].sum() / total_px  # fraction of pixels in 0-29
    has_dark_peak = dark_mass > 0.001  # more than 0.1% is a dark cluster
    if has_dark_peak:
        # Find where the dark cluster ends using the same gradient approach
        # Walk right from 0 until the gradient flattens out
        artifact_ceil = 10
        dark_grad = grad[:50]
        if len(dark_grad) > 0:
            # Find where gradient stops being strongly negative (cluster ending)
            min_grad = dark_grad.min()
            if min_grad < 0:
                for i in range(len(dark_grad)):
                    if dark_grad[i] > min_grad * 0.1:
                        artifact_ceil = i
                        break
        artifact_ceil = min(artifact_ceil, 50)  # safety cap
    else:
        artifact_ceil = 0  # no dark peak, don't clip anything

    # 3. Center threshold: Otsu on just the tissue range (between
    #    artifact_ceil and blank_floor) to separate tissue from
    #    near-white glass/background.
    tissue_range = gray[(gray > artifact_ceil) & (gray < blank_floor)]
    if len(tissue_range) > 100:
        otsu_val, _ = cv2.threshold(
            tissue_range, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        center_threshold = int(otsu_val)
    else:
        center_threshold = (artifact_ceil + blank_floor) // 2

    # --- Build tissue mask ---
    # Main tissue: pixels between artifact_ceil and blank_floor
    tissue_mask_main = np.zeros_like(gray, dtype=np.uint8)
    tissue_mask_main[(gray > artifact_ceil) & (gray < blank_floor)] = 255

    # --- Handle dark regions spatially ---
    # Dark pixels (below artifact_ceil) might be:
    #   a) Scanner edge artifacts (on the margins) → exclude
    #   b) Marker lines / dark tissue (in the interior) → include
    dark_mask = np.zeros_like(gray, dtype=np.uint8)
    if has_dark_peak and artifact_ceil > 0:
        dark_mask[gray <= artifact_ceil] = 255

        # Define margin zone: pixels within 5% of image edges
        margin_w = max(int(canvas_w * 0.05), 20)
        margin_h = max(int(canvas_h * 0.05), 20)
        interior_mask = np.zeros_like(gray, dtype=np.uint8)
        interior_mask[margin_h:-margin_h, margin_w:-margin_w] = 255

        # Find connected components in the dark mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8)

        interior_dark = np.zeros_like(gray, dtype=np.uint8)
        for label_id in range(1, n_labels):  # skip background (0)
            component = (labels == label_id).astype(np.uint8)
            # Check what fraction of this component is in the interior
            interior_pixels = cv2.bitwise_and(component, interior_mask).sum()
            total_component = component.sum()
            if total_component > 0 and interior_pixels / total_component > 0.5:
                # Mostly interior → include (marker lines, dark tissue)
                interior_dark[labels == label_id] = 255

        # Merge interior dark regions into the tissue mask
        tissue_mask_main = cv2.bitwise_or(tissue_mask_main, interior_dark)

    # --- Morphological cleanup ---
    # Scale kernel sizes relative to the canvas (kernels were tuned for ~256px tiles)
    k_scale = max(1, int(tile_size * tile_ds / 20))
    close_k = max(3, k_scale * 3) | 1  # ensure odd
    open_k = max(3, k_scale * 6) | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    pixel_mask = cv2.morphologyEx(tissue_mask_main, cv2.MORPH_CLOSE,
                                  close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_OPEN, open_kernel)

    # Store threshold values used
    pixel_threshold_gray = int(blank_floor)
    pixel_threshold_artifact = int(artifact_ceil)
    pixel_threshold_center = int(center_threshold)

    # Find tissue contours on pixel mask
    contours_raw, _ = cv2.findContours(
        pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store contours in normalized coordinates (0–1) relative to the
    # detection-level stitched image so they align with any thumbnail
    # regardless of which level it was stitched from.
    canvas_h, canvas_w = pixel_mask.shape
    min_contour_area = (tile_size * tile_ds * 0.3) ** 2

    # Compute HSV saturation for color-based filtering of small contours
    hsv = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    contours_norm = []
    small_contours_norm = []  # below area threshold AND no tissue color
    for c in sorted(contours_raw, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        keep = False
        if area >= min_contour_area:
            keep = True
        else:
            # Small contour — check if it has tissue color (saturation)
            # Create a mask for just this contour's pixels
            cmask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            cv2.drawContours(cmask, [c], 0, 255, cv2.FILLED)
            # Mean saturation inside this contour
            mean_sat = cv2.mean(saturation, mask=cmask)[0]
            # H&E tissue typically has saturation > 15-20
            if mean_sat > 15:
                keep = True

        epsilon = 0.001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] /= canvas_w
        pts[:, 1] /= canvas_h
        norm_pts = [[round(x, 6), round(y, 6)] for x, y in pts.tolist()]
        if keep:
            contours_norm.append(norm_pts)
        else:
            small_contours_norm.append(norm_pts)

    # Margin contours: dilate by margin L0 pixels converted to canvas pixels
    margin_det_px = max(1, int(margin / scale * tile_ds))
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_det_px + 1, 2 * margin_det_px + 1))
    margin_mask = cv2.dilate(pixel_mask, margin_kernel, iterations=1)
    margin_contours_raw, _ = cv2.findContours(
        margin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin_contours_norm = []
    for c in sorted(margin_contours_raw, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_contour_area:
            continue
        epsilon = 0.001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] /= canvas_w
        pts[:, 1] /= canvas_h
        margin_contours_norm.append([[round(x, 6), round(y, 6)] for x, y in pts.tolist()])

    # --- Bounding polygons: simplified, smoothed envelopes around tissue groups ---
    # Build a mask from only the kept tissue contours (no noise)
    tissue_only_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for c in contours_raw:
        area = cv2.contourArea(c)
        if area >= min_contour_area:
            cv2.drawContours(tissue_only_mask, [c], 0, 255, cv2.FILLED)
        elif area > 0:
            # Check saturation for small ones (same logic as above)
            cmask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            cv2.drawContours(cmask, [c], 0, 255, cv2.FILLED)
            mean_sat = cv2.mean(saturation, mask=cmask)[0]
            if mean_sat > 15:
                cv2.drawContours(tissue_only_mask, [c], 0, 255, cv2.FILLED)

    # Dilate by 3x margin to create wide bounding regions that group nearby tissue
    bound_det_px = max(3, int(margin * 3 / scale * tile_ds))
    bound_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * bound_det_px + 1, 2 * bound_det_px + 1))
    bound_mask = cv2.dilate(tissue_only_mask, bound_kernel, iterations=1)
    # Smooth the boundary with a large Gaussian blur + threshold
    bound_mask = cv2.GaussianBlur(bound_mask, (bound_det_px | 1, bound_det_px | 1), 0)
    _, bound_mask = cv2.threshold(bound_mask, 127, 255, cv2.THRESH_BINARY)

    bound_contours_raw, _ = cv2.findContours(
        bound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bound_contours_norm = []
    for c in sorted(bound_contours_raw, key=cv2.contourArea, reverse=True):
        # Simplify aggressively for smooth shapes (0.5% of perimeter)
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] /= canvas_w
        pts[:, 1] /= canvas_h
        bound_contours_norm.append([[round(x, 6), round(y, 6)] for x, y in pts.tolist()])

    meta = {
        'detect_level': detect_level,
        'detect_grid': [grid_w, grid_h],
        'max_level': max_level,
        'scale': scale,
        'threshold_std': round(threshold, 2),
        'pixel_threshold_blank': pixel_threshold_gray,
        'pixel_threshold_center': pixel_threshold_center,
        'pixel_threshold_artifact': pixel_threshold_artifact,
        'has_dark_peak': bool(has_dark_peak),
        'margin': margin,
        'tile_size': tile_size,
        'total_detect_tiles': n_tiles,
        'tissue_detect_tiles': tissue_count,
        'tissue_with_margin_tiles': margin_count,
        'l0_grid': [l0_grid_w, l0_grid_h],
        'l0_total_tiles': l0_total,
        'l0_included_tiles': l0_included,
        'l0_skipped_tiles': l0_total - l0_included,
        'l0_skip_pct': round(100.0 * (l0_total - l0_included) / l0_total, 1)
                       if l0_total > 0 else 0,
        'included_detect_tiles': included_tiles,
        'tissue_contours_norm': contours_norm,
        'small_contours_norm': small_contours_norm,
        'margin_contours_norm': margin_contours_norm,
        'bound_contours_norm': bound_contours_norm,
        'slide_w_px': l0_grid_w * tile_size,
        'slide_h_px': l0_grid_h * tile_size,
        'detect_canvas_w': canvas_w,
        'detect_canvas_h': canvas_h,
    }

    # Build a threshold visualization: color-code the different regions
    thresh_vis = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # White/blank regions: light gray
    thresh_vis[gray >= blank_floor] = [180, 180, 180]
    # Tissue range: green
    tissue_zone = (gray > artifact_ceil) & (gray < blank_floor)
    thresh_vis[tissue_zone] = [0, 180, 0]
    # Above center threshold but below blank: light green (near-white tissue)
    light_tissue = tissue_zone & (gray >= center_threshold)
    thresh_vis[light_tissue] = [100, 220, 100]
    # Dark artifacts: red (excluded), interior dark: blue (included)
    if has_dark_peak and artifact_ceil > 0:
        try:
            dark_edge = (gray <= artifact_ceil) & (interior_dark == 0)
            thresh_vis[dark_edge] = [180, 0, 0]
            thresh_vis[interior_dark > 0] = [0, 80, 200]
        except NameError:
            thresh_vis[gray <= artifact_ceil] = [180, 0, 0]

    # --- Blank color grid (10x10) ---
    grid_cols, grid_rows = 10, 10
    blank_pixels = (pixel_mask == 0) & (gray < 253)
    cell_h = canvas_h / grid_rows
    cell_w = canvas_w / grid_cols
    blank_color_grid = []
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, y1 = int(gy * cell_h), int((gy + 1) * cell_h)
            x0, x1 = int(gx * cell_w), int((gx + 1) * cell_w)
            cell_blank = blank_pixels[y0:y1, x0:x1]
            cell_rgb = canvas_rgb[y0:y1, x0:x1]
            if cell_blank.any():
                r = int(np.median(cell_rgb[cell_blank, 0]))
                gv = int(np.median(cell_rgb[cell_blank, 1]))
                b = int(np.median(cell_rgb[cell_blank, 2]))
                blank_color_grid.append([r, gv, b])
            else:
                blank_color_grid.append(None)

    # Fill None cells with nearest neighbor
    for i in range(len(blank_color_grid)):
        if blank_color_grid[i] is not None:
            continue
        gy_i, gx_i = divmod(i, grid_cols)
        best_dist, best_color = float('inf'), [232, 232, 232]
        for j in range(len(blank_color_grid)):
            if blank_color_grid[j] is None:
                continue
            jy, jx = divmod(j, grid_cols)
            dist = abs(gy_i - jy) + abs(gx_i - jx)
            if dist < best_dist:
                best_dist, best_color = dist, blank_color_grid[j]
        blank_color_grid[i] = best_color

    meta['blank_color_grid'] = {
        'cols': grid_cols,
        'rows': grid_rows,
        'rgb': blank_color_grid,
    }

    # --- Build debug images ---
    # Blank color grid visualization (10x10 RGB image)
    color_grid_img = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
    for idx, rgb in enumerate(blank_color_grid):
        gy_idx, gx_idx = divmod(idx, grid_cols)
        color_grid_img[gy_idx, gx_idx] = rgb

    # Tissue bitmap visualization (1px per L0 tile)
    from tissue_filter import TissueFilter
    filt_for_viz = TissueFilter.from_meta(meta)
    bitmap_viz = (filt_for_viz._bitmap.astype(np.uint8) * 255)

    # Margin bitmap
    margin_tiles_k = max(1, -(-margin // (tile_size * scale)))
    k_margin = 2 * margin_tiles_k + 1
    margin_bm = cv2.dilate(filt_for_viz._bitmap.astype(np.uint8),
                           np.ones((k_margin, k_margin), dtype=np.uint8))
    margin_bm_viz = (margin_bm * 255)

    images = {
        'gray': gray,
        'threshold': thresh_vis,
        'mask': pixel_mask,
        'margin_mask': margin_mask,
        'color_grid': color_grid_img,
        'tissue_bitmap': bitmap_viz,
        'margin_bitmap': margin_bm_viz,
    }

    return meta, images


def is_tile_included(meta: dict, level: int, tx: int, ty: int) -> bool:
    """Check if a tile at a given level should be processed.

    Maps the tile coordinates back to the detection level grid and checks
    against the inclusion mask.

    Args:
        meta: The tissue detection metadata dict.
        level: The DZI level number.
        tx, ty: Tile coordinates at that level.

    Returns:
        True if the tile overlaps a tissue region (should be processed).
    """
    detect_level = meta['detect_level']
    max_level = meta['max_level']
    grid_w, grid_h = meta['detect_grid']

    # Build a set for fast lookup (cache this in practice)
    included = set(tuple(t) for t in meta['included_detect_tiles'])

    # Map (level, tx, ty) to detection-level tile range
    level_diff = level - detect_level
    if level_diff >= 0:
        # This level is higher-res than detection level
        # Each detection tile covers 2^level_diff tiles at this level
        det_tx = tx >> level_diff
        det_ty = ty >> level_diff
        return (det_tx, det_ty) in included
    else:
        # This level is lower-res than detection level
        # This tile covers multiple detection tiles
        scale = 2 ** (-level_diff)
        for dy in range(scale):
            for dx in range(scale):
                dtx = tx * scale + dx
                dty = ty * scale + dy
                if (dtx, dty) in included:
                    return True
        return False


def write_svg(meta: dict, svg_path: str):
    """Write an SVG showing tissue contours over the slide dimensions.

    Contours are stored in normalized (0–1) coordinates; the SVG uses a
    1000x1000 viewBox and scales them accordingly.
    """
    w = meta['slide_w_px']
    h = meta['slide_h_px']
    # Use a 1000-wide viewBox, preserving aspect ratio
    vb_w = 1000
    vb_h = int(1000 * h / w)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {vb_w} {vb_h}" width="1200" '
        f'height="{int(1200 * h / w)}">'
    ]

    lines.append(f'  <rect width="{vb_w}" height="{vb_h}" fill="#f8f8f8"/>')

    # Bounding polygons (light blue — outermost, drawn first)
    for contour in meta.get('bound_contours_norm', []):
        pts = ' '.join(f'{x * vb_w:.1f},{y * vb_h:.1f}' for x, y in contour)
        lines.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(60,140,220,0.1)" stroke="#3c8cdc" stroke-width="2" '
            f'stroke-dasharray="6,3"/>')

    # Margin contours (light green fill)
    for contour in meta.get('margin_contours_norm', []):
        pts = ' '.join(f'{x * vb_w:.1f},{y * vb_h:.1f}' for x, y in contour)
        lines.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(0,200,0,0.15)" stroke="#00aa00" stroke-width="1.5"/>')

    # Small contours below min_area (yellow — for debugging)
    for contour in meta.get('small_contours_norm', []):
        pts = ' '.join(f'{x * vb_w:.1f},{y * vb_h:.1f}' for x, y in contour)
        lines.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(220,200,0,0.3)" stroke="#ccaa00" stroke-width="1"/>')

    # Tissue contours (red outline, semi-transparent fill)
    for contour in meta.get('tissue_contours_norm', []):
        pts = ' '.join(f'{x * vb_w:.1f},{y * vb_h:.1f}' for x, y in contour)
        lines.append(
            f'  <polygon points="{pts}" '
            f'fill="rgba(220,40,40,0.2)" stroke="#cc0000" stroke-width="2"/>')

    lines.append('</svg>')

    with open(svg_path, 'w') as f:
        f.write('\n'.join(lines))


def process_slide(dzi_dir: str, slide_name: str, margin: int = 2) -> dict:
    """Process a single slide and write .tissue.json."""
    files_dir = os.path.join(dzi_dir, f'{slide_name}_files')
    if not os.path.isdir(files_dir):
        print(f'  ERROR: {files_dir} not found')
        return None

    levels = sorted(int(d) for d in os.listdir(files_dir) if d.isdigit())
    if not levels:
        print(f'  ERROR: no levels found in {files_dir}')
        return None

    max_level = levels[-1]

    t0 = time.time()
    meta, images = detect_tissue(files_dir, max_level, levels, margin=margin)
    dt = time.time() - t0

    # Write metadata
    out_path = os.path.join(dzi_dir, f'{slide_name}.tissue.json')
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Write SVG
    svg_path = os.path.join(dzi_dir, f'{slide_name}.tissue.svg')
    write_svg(meta, svg_path)

    # Write binary files for fast Rust/Python loading
    from tissue_filter import TissueFilter
    filt = TissueFilter.from_meta(meta)

    # Legacy separate bitmaps (still useful for standalone tools)
    bitmap_path = os.path.join(dzi_dir, f'{slide_name}.tissue.bitmap')
    filt.save_bitmap(bitmap_path)
    margin_bitmap_path = os.path.join(dzi_dir, f'{slide_name}.tissue_margin.bitmap')
    filt.save_margin_bitmap(margin_bitmap_path, margin_l0_px=margin)

    # Combined .tissue.map (bitmaps + color grid — what the tile server reads)
    bcg = meta['blank_color_grid']
    map_path = os.path.join(dzi_dir, f'{slide_name}.tissue.map')
    filt.save_map(map_path, bcg['rgb'],
                  grid_cols=bcg['cols'], grid_rows=bcg['rows'],
                  margin_l0_px=margin)

    # Write debug images
    for name, img in images.items():
        # Use PNG for small pixel-exact images, JPEG for large ones
        if name in ('color_grid', 'tissue_bitmap', 'margin_bitmap'):
            img_path = os.path.join(dzi_dir, f'{slide_name}.tissue_{name}.png')
            Image.fromarray(img).save(img_path, 'PNG')
        else:
            img_path = os.path.join(dzi_dir, f'{slide_name}.tissue_{name}.jpg')
            Image.fromarray(img).save(img_path, 'JPEG', quality=80)

    print(f'  Detected at level {meta["detect_level"]} '
          f'({meta["detect_grid"][0]}x{meta["detect_grid"][1]} grid), '
          f'threshold_std={meta["threshold_std"]:.1f}')
    print(f'  Pixel thresholds: blank>={meta["pixel_threshold_blank"]}, '
          f'center={meta["pixel_threshold_center"]}, '
          f'artifact<={meta["pixel_threshold_artifact"]}'
          f'{" (dark peak)" if meta["has_dark_peak"] else ""}')
    print(f'  Tissue: {meta["tissue_detect_tiles"]}/{meta["total_detect_tiles"]} tiles '
          f'({100*meta["tissue_detect_tiles"]/meta["total_detect_tiles"]:.0f}%), '
          f'with margin: {meta["tissue_with_margin_tiles"]}')
    print(f'  Contours: {len(meta["tissue_contours_norm"])} tissue, '
          f'{len(meta["margin_contours_norm"])} margin')
    print(f'  L0: {meta["l0_included_tiles"]}/{meta["l0_total_tiles"]} tiles to process, '
          f'{meta["l0_skipped_tiles"]} skipped ({meta["l0_skip_pct"]}%)')
    print(f'  Written: {out_path}')
    print(f'  SVG: {svg_path} ({dt:.1f}s)')

    return meta


def main():
    ap = argparse.ArgumentParser(
        description='Detect tissue regions in DZI pyramids')
    ap.add_argument('--dzi-dir', required=True,
                    help='Directory containing .dzi files and _files/ dirs')
    ap.add_argument('--slide', default=None,
                    help='Process only this slide (stem name)')
    ap.add_argument('--margin', type=int, default=512,
                    help='Margin in L0 pixels (default: 512 = 2 L0 tiles)')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing .tissue.json files')
    args = ap.parse_args()

    if not os.path.isdir(args.dzi_dir):
        print(f'Error: {args.dzi_dir} not found')
        sys.exit(1)

    if args.slide:
        slides = [args.slide]
    else:
        slides = sorted(
            f.replace('.dzi', '')
            for f in os.listdir(args.dzi_dir)
            if f.endswith('.dzi')
        )

    print(f'Processing {len(slides)} slide(s) in {args.dzi_dir}')
    print(f'Margin: {args.margin} L0 pixels\n')

    for slide in slides:
        tissue_path = os.path.join(args.dzi_dir, f'{slide}.tissue.json')
        if os.path.exists(tissue_path) and not args.force:
            print(f'{slide[:40]}... — skipped (exists, use --force)')
            continue

        print(f'{slide[:40]}...')
        process_slide(args.dzi_dir, slide, margin=args.margin)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
