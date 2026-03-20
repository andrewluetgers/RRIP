"""
Fast tile inclusion filter based on tissue bounding polygons.

Usage:
    from tissue_filter import TissueFilter

    # Load from pre-built bitmap (fastest — ~0.1ms)
    filt = TissueFilter.load_bitmap("slide.tissue.bitmap")

    # Or load from JSON and rasterize (slower first time — ~5ms)
    filt = TissueFilter.load("slide.tissue.json")

    # O(1) lookup — sub-microsecond per call
    if filt.includes(level=17, tx=150, ty=200):
        serve_tile(...)

    # Batch: get all included L0 tile coordinates
    for tx, ty in filt.included_l0_tiles():
        ...

Bitmap file format (.tissue.bitmap):
    Header (16 bytes):
        magic     : 4s   "TBIT"
        version   : u16  1
        max_level : u16
        l0_w      : u32
        l0_h      : u32
    Body:
        Packed bits, row-major, ceil(l0_w * l0_h / 8) bytes.
        Bit order: LSB first within each byte.
"""

import json
import struct
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class TissueFilter:
    """Pre-rasterized bounding polygon filter for fast tile inclusion checks.

    On init, rasterizes the bounding polygons into a boolean bitmap at L0
    tile resolution. Lookups for any (level, tx, ty) are a single array
    index — O(1).
    """

    _MAGIC = b"TBIT"
    _VERSION = 1
    _HEADER_FMT = "<4sHHII"  # 16 bytes
    _HEADER_SIZE = struct.calcsize("<4sHHII")

    __slots__ = ('_bitmap', '_l0_w', '_l0_h', '_max_level', '_detect_level')

    def __init__(self, bitmap: np.ndarray, l0_w: int, l0_h: int,
                 max_level: int, detect_level: int):
        self._bitmap = bitmap        # shape (l0_h, l0_w), dtype bool
        self._l0_w = l0_w
        self._l0_h = l0_h
        self._max_level = max_level
        self._detect_level = detect_level

    # --- Bitmap file I/O ---

    def save_bitmap(self, path: str):
        """Write the bitmap to a .tissue.bitmap file."""
        self._write_bitmap(self._bitmap, path)

    def _write_bitmap(self, bitmap: np.ndarray, path: str):
        """Write a boolean bitmap to a .tissue.bitmap file."""
        flat = bitmap.ravel().astype(np.uint8)
        packed = np.packbits(flat, bitorder='little')
        with open(path, 'wb') as f:
            f.write(struct.pack(
                self._HEADER_FMT,
                self._MAGIC,
                self._VERSION,
                self._max_level,
                self._l0_w,
                self._l0_h,
            ))
            f.write(packed.tobytes())

    @classmethod
    def load_bitmap(cls, path: str) -> 'TissueFilter':
        """Load from a .tissue.bitmap file. ~0.1ms, no JSON parsing."""
        with open(path, 'rb') as f:
            header = f.read(cls._HEADER_SIZE)
            magic, version, max_level, l0_w, l0_h = struct.unpack(
                cls._HEADER_FMT, header)
            assert magic == cls._MAGIC, f"Bad magic: {magic}"
            assert version == cls._VERSION, f"Bad version: {version}"
            packed = f.read()

        n_bits = l0_w * l0_h
        flat = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8),
            bitorder='little')[:n_bits]
        bitmap = flat.reshape(l0_h, l0_w).astype(bool)
        return cls(bitmap, l0_w, l0_h, max_level, 0)

    def expand_to_l2_families(self) -> np.ndarray:
        """Expand bitmap so that if ANY L0 tile in an L2 family has tissue,
        ALL L0 tiles in that family are included.

        An L2 tile covers a 4x4 block of L0 tiles. This ensures the tile
        server has all children needed to reconstruct L1/L2 at margins.

        Returns the expanded bitmap (does not modify self).
        """
        expanded = self._bitmap.copy()
        # L2 families are 4x4 blocks of L0 tiles
        for by in range(0, self._l0_h, 4):
            for bx in range(0, self._l0_w, 4):
                y1 = min(by + 4, self._l0_h)
                x1 = min(bx + 4, self._l0_w)
                if expanded[by:y1, bx:x1].any():
                    expanded[by:y1, bx:x1] = True
        return expanded

    def save_margin_bitmap(self, path: str, margin_l0_px: int = 512,
                           tile_size: int = 256):
        """Dilate the tissue bitmap by margin and save as a separate file.

        Args:
            path: Output .bitmap path.
            margin_l0_px: Margin in L0 pixels.
            tile_size: L0 tile size in pixels.
        """
        margin_tiles = max(1, -(-margin_l0_px // tile_size))  # ceil div
        k = 2 * margin_tiles + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate(self._bitmap.astype(np.uint8), kernel,
                             iterations=1).astype(bool)
        self._write_bitmap(dilated, path)

    # --- Combined .tissue.map format (TMAP v1) ---
    #
    # Header (20 bytes):
    #   magic       : 4s   "TMAP"
    #   version     : u16  1
    #   max_level   : u16
    #   l0_w        : u32
    #   l0_h        : u32
    #   grid_cols   : u8
    #   grid_rows   : u8
    #   flags       : u16  (bit 0: has margin bitmap)
    #
    # Body (sequential, sizes deterministic from header):
    #   blank_color_grid : grid_cols * grid_rows * 3 bytes (RGB, row-major)
    #   tissue_bitmap    : ceil(l0_w * l0_h / 8) bytes (packed bits, LSB)
    #   margin_bitmap    : ceil(l0_w * l0_h / 8) bytes (if flags & 1)

    _MAP_MAGIC = b"TMAP"
    _MAP_VERSION = 1
    _MAP_HEADER_FMT = "<4sHHIIBBH"  # 20 bytes
    _MAP_HEADER_SIZE = struct.calcsize("<4sHHIIBBH")

    def save_map(self, path: str, blank_color_grid: list,
                 grid_cols: int = 10, grid_rows: int = 10,
                 margin_l0_px: int = 512, tile_size: int = 256):
        """Write combined .tissue.map file with bitmaps + color grid.

        Args:
            path: Output .tissue.map path.
            blank_color_grid: List of [R, G, B] values, row-major.
            grid_cols, grid_rows: Color grid dimensions.
            margin_l0_px: Margin in L0 pixels for generating margin bitmap.
            tile_size: L0 tile size in pixels.
        """
        # Generate margin bitmap
        margin_tiles = max(1, -(-margin_l0_px // tile_size))
        k = 2 * margin_tiles + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        margin_bm = cv2.dilate(self._bitmap.astype(np.uint8), kernel,
                                iterations=1).astype(bool)

        has_margin = True
        flags = 1 if has_margin else 0

        # Pack tissue bitmap
        tissue_packed = np.packbits(
            self._bitmap.ravel().astype(np.uint8), bitorder='little')
        margin_packed = np.packbits(
            margin_bm.ravel().astype(np.uint8), bitorder='little')

        # Pack color grid as flat RGB bytes
        color_bytes = bytearray()
        for rgb in blank_color_grid:
            color_bytes.extend(rgb)

        with open(path, 'wb') as f:
            f.write(struct.pack(
                self._MAP_HEADER_FMT,
                self._MAP_MAGIC,
                self._MAP_VERSION,
                self._max_level,
                self._l0_w,
                self._l0_h,
                grid_cols,
                grid_rows,
                flags,
            ))
            f.write(bytes(color_bytes))
            f.write(tissue_packed.tobytes())
            if has_margin:
                f.write(margin_packed.tobytes())

    @classmethod
    def load_map(cls, path: str) -> tuple['TissueFilter', dict]:
        """Load a .tissue.map file. Returns (filter, info_dict).

        info_dict contains:
            - margin_bitmap: np.ndarray (bool)
            - blank_color_grid: list of [R, G, B]
            - grid_cols, grid_rows: int
        """
        with open(path, 'rb') as f:
            header = f.read(cls._MAP_HEADER_SIZE)
            magic, version, max_level, l0_w, l0_h, grid_cols, grid_rows, flags = \
                struct.unpack(cls._MAP_HEADER_FMT, header)
            assert magic == cls._MAP_MAGIC, f"Bad magic: {magic}"
            assert version == cls._MAP_VERSION, f"Bad version: {version}"

            # Read color grid
            color_data = f.read(grid_cols * grid_rows * 3)
            blank_color_grid = []
            for i in range(grid_cols * grid_rows):
                r, g, b = color_data[i*3], color_data[i*3+1], color_data[i*3+2]
                blank_color_grid.append([r, g, b])

            # Read tissue bitmap
            n_bits = l0_w * l0_h
            n_bytes = (n_bits + 7) // 8
            tissue_packed = f.read(n_bytes)
            tissue_flat = np.unpackbits(
                np.frombuffer(tissue_packed, dtype=np.uint8),
                bitorder='little')[:n_bits]
            tissue_bm = tissue_flat.reshape(l0_h, l0_w).astype(bool)

            # Read margin bitmap if present
            margin_bm = None
            if flags & 1:
                margin_packed = f.read(n_bytes)
                margin_flat = np.unpackbits(
                    np.frombuffer(margin_packed, dtype=np.uint8),
                    bitorder='little')[:n_bits]
                margin_bm = margin_flat.reshape(l0_h, l0_w).astype(bool)

        filt = cls(tissue_bm, l0_w, l0_h, max_level, 0)
        info = {
            'margin_bitmap': margin_bm,
            'blank_color_grid': blank_color_grid,
            'grid_cols': grid_cols,
            'grid_rows': grid_rows,
        }
        return filt, info

    def get_blank_color(self, tx: int, ty: int, blank_color_grid: list,
                        grid_cols: int = 10, grid_rows: int = 10) -> tuple:
        """Get interpolated blank color for a tile position.

        Bilinear interpolation between the 4 nearest grid points.

        Args:
            tx, ty: L0 tile coordinates.
            blank_color_grid: Flat list of [R, G, B], row-major.
            grid_cols, grid_rows: Grid dimensions.

        Returns:
            (R, G, B) tuple.
        """
        # Map tile position to grid coordinates (0..grid_cols-1, 0..grid_rows-1)
        gx = tx / self._l0_w * (grid_cols - 1)
        gy = ty / self._l0_h * (grid_rows - 1)

        # Bilinear interpolation
        gx0 = min(int(gx), grid_cols - 2)
        gy0 = min(int(gy), grid_rows - 2)
        gx1 = gx0 + 1
        gy1 = gy0 + 1
        fx = gx - gx0
        fy = gy - gy0

        c00 = blank_color_grid[gy0 * grid_cols + gx0]
        c10 = blank_color_grid[gy0 * grid_cols + gx1]
        c01 = blank_color_grid[gy1 * grid_cols + gx0]
        c11 = blank_color_grid[gy1 * grid_cols + gx1]

        r = int(c00[0] * (1-fx)*(1-fy) + c10[0] * fx*(1-fy) +
                c01[0] * (1-fx)*fy + c11[0] * fx*fy + 0.5)
        g = int(c00[1] * (1-fx)*(1-fy) + c10[1] * fx*(1-fy) +
                c01[1] * (1-fx)*fy + c11[1] * fx*fy + 0.5)
        b = int(c00[2] * (1-fx)*(1-fy) + c10[2] * fx*(1-fy) +
                c01[2] * (1-fx)*fy + c11[2] * fx*fy + 0.5)

        return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))

    @classmethod
    def load(cls, tissue_json_path: str) -> 'TissueFilter':
        """Load from a .tissue.json file and rasterize bounding polygons."""
        with open(tissue_json_path) as f:
            meta = json.load(f)

        l0_w, l0_h = meta['l0_grid']
        max_level = meta['max_level']
        detect_level = meta['detect_level']
        bound_contours = meta.get('bound_contours_norm', [])

        # Rasterize bounding polygons into L0-tile-resolution bitmap
        bitmap = np.zeros((l0_h, l0_w), dtype=np.uint8)

        if _HAS_CV2 and bound_contours:
            for contour_norm in bound_contours:
                # Convert normalized coords to L0 tile coords
                pts = np.array(contour_norm, dtype=np.float32)
                pts[:, 0] *= l0_w
                pts[:, 1] *= l0_h
                pts = pts.astype(np.int32)
                cv2.fillPoly(bitmap, [pts], 1)
        elif bound_contours:
            # Fallback without cv2: use included_detect_tiles
            scale = 2 ** (max_level - detect_level)
            for tx, ty in meta.get('included_detect_tiles', []):
                for dy in range(scale):
                    for dx in range(scale):
                        lx, ly = tx * scale + dx, ty * scale + dy
                        if lx < l0_w and ly < l0_h:
                            bitmap[ly, lx] = 1

        return cls(bitmap.astype(bool), l0_w, l0_h, max_level, detect_level)

    @classmethod
    def from_meta(cls, meta: dict) -> 'TissueFilter':
        """Create from an already-loaded metadata dict."""
        l0_w, l0_h = meta['l0_grid']
        max_level = meta['max_level']
        detect_level = meta['detect_level']
        bound_contours = meta.get('bound_contours_norm', [])

        bitmap = np.zeros((l0_h, l0_w), dtype=np.uint8)
        if _HAS_CV2 and bound_contours:
            for contour_norm in bound_contours:
                pts = np.array(contour_norm, dtype=np.float32)
                pts[:, 0] *= l0_w
                pts[:, 1] *= l0_h
                pts = pts.astype(np.int32)
                cv2.fillPoly(bitmap, [pts], 1)

        return cls(bitmap.astype(bool), l0_w, l0_h, max_level, detect_level)

    def includes(self, level: int, tx: int, ty: int) -> bool:
        """Check if tile (level, tx, ty) falls within a tissue bounding region.

        Maps the tile to L0 coordinates and checks the pre-rasterized bitmap.
        For levels coarser than L0, returns True if ANY covered L0 tile is
        included.

        Args:
            level: DZI level number (max_level = L0, max_level-1 = L1, etc.)
            tx, ty: Tile coordinates at that level.

        Returns:
            True if the tile should be served/processed.
        """
        level_diff = self._max_level - level  # 0 for L0, 1 for L1, etc.

        if level_diff <= 0:
            # L0 or finer: direct lookup
            if tx < 0 or ty < 0 or tx >= self._l0_w or ty >= self._l0_h:
                return False
            return self._bitmap[ty, tx]

        # Coarser level: this tile covers 2^level_diff L0 tiles per axis
        scale = 1 << level_diff
        l0_x0 = tx * scale
        l0_y0 = ty * scale
        l0_x1 = min(l0_x0 + scale, self._l0_w)
        l0_y1 = min(l0_y0 + scale, self._l0_h)

        if l0_x0 >= self._l0_w or l0_y0 >= self._l0_h:
            return False

        # Check if ANY L0 tile in this block is included
        return self._bitmap[l0_y0:l0_y1, l0_x0:l0_x1].any()

    def included_l0_tiles(self):
        """Yield (tx, ty) for all included L0 tiles."""
        ys, xs = np.where(self._bitmap)
        for x, y in zip(xs, ys):
            yield int(x), int(y)

    @property
    def l0_included_count(self) -> int:
        return int(self._bitmap.sum())

    @property
    def l0_total_count(self) -> int:
        return self._l0_w * self._l0_h

    @property
    def l0_skip_pct(self) -> float:
        total = self._l0_w * self._l0_h
        return 100.0 * (total - self._bitmap.sum()) / total if total > 0 else 0


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python tissue_filter.py <slide.tissue.json> [benchmark]")
        sys.exit(1)

    path = sys.argv[1]
    do_bench = len(sys.argv) > 2 and sys.argv[2] == 'benchmark'

    t0 = time.perf_counter()
    if path.endswith('.bitmap'):
        filt = TissueFilter.load_bitmap(path)
    else:
        filt = TissueFilter.load(path)
    dt_load = (time.perf_counter() - t0) * 1000
    print(f"Loaded in {dt_load:.1f}ms")
    print(f"L0 grid: {filt._l0_w}x{filt._l0_h}")
    print(f"Included: {filt.l0_included_count}/{filt.l0_total_count} "
          f"({100 - filt.l0_skip_pct:.1f}%), skip {filt.l0_skip_pct:.1f}%")

    if do_bench:
        # Benchmark: random tile lookups
        max_level = filt._max_level
        rng = np.random.default_rng(42)
        n = 1_000_000

        # L0 lookups
        txs = rng.integers(0, filt._l0_w, size=n)
        tys = rng.integers(0, filt._l0_h, size=n)
        t0 = time.perf_counter()
        for i in range(n):
            filt.includes(max_level, int(txs[i]), int(tys[i]))
        dt = (time.perf_counter() - t0) * 1e9 / n
        print(f"\nL0 lookup: {dt:.0f}ns/call ({n:,} calls)")

        # L2 lookups
        l2_level = max_level - 2
        l2_w = (filt._l0_w + 3) // 4
        l2_h = (filt._l0_h + 3) // 4
        txs2 = rng.integers(0, max(l2_w, 1), size=n)
        tys2 = rng.integers(0, max(l2_h, 1), size=n)
        t0 = time.perf_counter()
        for i in range(n):
            filt.includes(l2_level, int(txs2[i]), int(tys2[i]))
        dt2 = (time.perf_counter() - t0) * 1e9 / n
        print(f"L2 lookup: {dt2:.0f}ns/call ({n:,} calls)")
