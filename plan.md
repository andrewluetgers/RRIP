# Plan: Set up slide viewers with DICOM originals + JXL variants

## Current State
- **2 slides** have JXL Q40/Q80 variants in `tile_server/`: `3DHISTECH-1` and `1.1401...` (Mayosh DICOM)
- Same 2 slides have DICOM originals in `dzi_trimmed/` (256px JPEG tiles, untouched from scanner)
- Tile server expects subdirectories with `baseline_pyramid.dzi` + `baseline_pyramid_files/` naming
- Source data uses `{name}.dzi` + `{name}_files/` naming
- No slides root directory exists yet in tile-server format

## Steps

### 1. Create a setup script (`scripts/setup_slides_root.sh`)

Creates a slides root directory at `~/dev/data/WSI/slides/` with symlinked entries for all 6 slide variants (2 slides x 3 formats):

```
slides/
  3DHISTECH-1_original/        → DICOM original (256px JPEG, ~388 MB)
    baseline_pyramid.dzi       → symlink to dzi_trimmed/3DHISTECH-1.dzi
    baseline_pyramid_files/    → symlink to dzi_trimmed/3DHISTECH-1_files/
    summary.json               → {"label": "3DHISTECH-1 Original", "mode": "ingest-jpeg-only", ...}

  3DHISTECH-1_jxl80/           → JXL Q80 (1024px JXL tiles)
    baseline_pyramid.dzi       → symlink to tile_server/jxl_q80/3DHISTECH-1/3DHISTECH-1.dzi
    baseline_pyramid_files/    → symlink to tile_server/jxl_q80/3DHISTECH-1/3DHISTECH-1_files/
    summary.json               → {"label": "3DHISTECH-1 JXL Q80", "mode": "jxl-baseline", ...}

  3DHISTECH-1_jxl40/           → JXL Q40
    ...same pattern...

  mayosh_original/             → DICOM original for the Mayosh slide
    ...same pattern...

  mayosh_jxl80/
  mayosh_jxl40/
```

Uses symlinks throughout — no data duplication. Generates `summary.json` for each with label, mode, and disk size.

### 2. Symlink the slides root into the project

```
data/slides -> ~/dev/data/WSI/slides/
```

### 3. Update the homepage (`evals/viewer/public/index.html`)

- Change `DEFAULT_LEFT` from `'jpeg90'` to `'3DHISTECH-1_original'`
- Change `DEFAULT_RIGHT` from `'v2_b90_l0q80'` to `'3DHISTECH-1_jxl40'`
- Update the description text below the viewer to reflect the new defaults (DICOM original vs JXL Q40)

### 4. Start the tile server

Run `origami serve --slides-root data/slides/ --port 3007` to verify all 6 slides are discovered and working.

## Notes
- The tile server already supports both `ingest-jpeg-only` (serves static JPEG tiles) and `jxl-baseline` (decodes JXL to JPEG on demand, slices 1024px→256px)
- The viewer's slide picker is populated dynamically from `/slides.json` — all 6 entries will appear automatically
- The `/compare` endpoint also uses the same slide discovery, so the WSI comparison viewer gets all variants too
