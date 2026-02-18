#!/bin/bash
# ORIGAMI WSI Evaluation - Complete Reproduction Script
# This script reproduces the entire evaluation from encoding to metrics computation

set -e  # Exit on error

# Configuration
SLIDE_PATH="/workspace/data/3DHISTECH-2-256/4_1"
OUTPUT_DIR="evals/runs/wsi_for_family_eval_rerun"
TILE_SIZE=256
BASEQ=80
L1Q=70
L0Q=60
MAX_DELTA=15
BATCH_SIZE=64

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "ORIGAMI WSI Evaluation - Reproduction Script"
echo "================================================================================"
echo ""

# Step 1: GPU Encoding
echo -e "${BLUE}[1/7] GPU Encoding - Generating residual bundle...${NC}"
cd /workspace/RRIP
source $HOME/.cargo/env

time RUST_LOG=info ./gpu-encode/target/release/origami-gpu-encode encode \
  --slide "$SLIDE_PATH" \
  --out "$OUTPUT_DIR" \
  --tile "$TILE_SIZE" \
  --baseq "$BASEQ" \
  --l1q "$L1Q" \
  --l0q "$L0Q" \
  --optl2 \
  --max-delta "$MAX_DELTA" \
  --pack \
  --manifest \
  --batch-size "$BATCH_SIZE"

echo -e "${GREEN}✓ Encoding complete${NC}"
echo ""

# Step 2: Bundle Analysis
echo -e "${BLUE}[2/7] Analyzing bundle for family size distribution...${NC}"
python3 << 'PYEOF'
import struct
from pathlib import Path
import json
import numpy as np

bundle_path = Path("evals/runs/wsi_for_family_eval_rerun/residual_packs/residuals.bundle")
with open(bundle_path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    cols = struct.unpack('<H', f.read(2))[0]
    rows = struct.unpack('<H', f.read(2))[0]
    f.read(20)  # reserved
    
    # Read to end to get index
    f.seek(0, 2)
    file_size = f.tell()
    index_size = (cols * rows) * 12
    index_start = file_size - index_size
    
    f.seek(index_start)
    sizes = []
    for _ in range(cols * rows):
        offset = struct.unpack('<Q', f.read(8))[0]
        length = struct.unpack('<I', f.read(4))[0]
        if length > 0:
            sizes.append(length)

sizes_arr = np.array(sizes)
stats = {
    'min': int(sizes_arr.min()),
    'p50': int(np.percentile(sizes_arr, 50)),
    'p95': int(np.percentile(sizes_arr, 95)),
    'p99': int(np.percentile(sizes_arr, 99)),
    'max': int(sizes_arr.max()),
    'mean': int(sizes_arr.mean()),
}

# Select 10 representative families
percentiles = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
representatives = []
for p in percentiles:
    target_size = np.percentile(sizes_arr, p)
    idx = np.argmin(np.abs(sizes_arr - target_size))
    # Convert flat index to (col, row)
    col = idx % cols
    row = idx // cols
    representatives.append({'percentile': p, 'x2': col, 'y2': row, 'size': sizes_arr[idx]})

analysis = {
    'bundle_path': str(bundle_path),
    'grid': f'{cols}x{rows}',
    'total_families': cols * rows,
    'non_empty_families': len(sizes),
    'empty_families': cols * rows - len(sizes),
    'statistics': stats,
    'representatives': representatives,
}

Path('evals/runs/wsi_for_family_eval_rerun/family_analysis.json').write_text(json.dumps(analysis, indent=2))
print(f"✓ Analyzed {len(sizes):,} non-empty families")
print(f"  P50: {stats['p50']/1024:.1f} KB, P95: {stats['p95']/1024:.1f} KB")
PYEOF

echo -e "${GREEN}✓ Bundle analysis complete${NC}"
echo ""

# Step 3: Extract Pack Files (for testing individual families)
echo -e "${BLUE}[3/7] Extracting sample pack files...${NC}"
python3 << 'PYEOF'
import json
import struct
from pathlib import Path

analysis = json.loads(Path('evals/runs/wsi_for_family_eval_rerun/family_analysis.json').read_text())
bundle_path = Path(analysis['bundle_path'])
extract_dir = bundle_path.parent / 'extracted'
extract_dir.mkdir(exist_ok=True)

# Extract packs for representative families
with open(bundle_path, 'rb') as f:
    f.read(8)  # magic + version
    cols = struct.unpack('<H', f.read(2))[0]
    rows = struct.unpack('<H', f.read(2))[0]
    f.read(20)  # reserved
    
    # Get index
    f.seek(0, 2)
    file_size = f.tell()
    index_size = (cols * rows) * 12
    index_start = file_size - index_size
    
    for rep in analysis['representatives'][:6]:  # First 6 only
        x2, y2 = rep['x2'], rep['y2']
        idx = y2 * cols + x2
        
        f.seek(index_start + idx * 12)
        offset = struct.unpack('<Q', f.read(8))[0]
        length = struct.unpack('<I', f.read(4))[0]
        
        if length > 0:
            f.seek(offset)
            pack_data = f.read(length)
            (extract_dir / f'{x2}_{y2}.pack').write_bytes(pack_data)
            print(f"  Extracted {x2}_{y2}.pack ({length} bytes)")

print(f"✓ Extracted 6 representative pack files")
PYEOF

echo -e "${GREEN}✓ Pack extraction complete${NC}"
echo ""

# Step 4: Decode Tiles
echo -e "${BLUE}[4/7] Decoding tiles from extracted packs...${NC}"
mkdir -p "$OUTPUT_DIR/decoded_tiles_selected/"{L0,L1}

# Get the 6 representative families
FAMILIES=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/family_analysis.json')); print(' '.join([f\"{r['x2']}_{r['y2']}\" for r in d['representatives'][:6]]))")

for fam in $FAMILIES; do
    echo "  Decoding family $fam..."
    ./server/target/release/origami decode \
        --pyramid "$OUTPUT_DIR" \
        --pack-file "$OUTPUT_DIR/residual_packs/extracted/${fam}.pack" \
        --out "$OUTPUT_DIR/decoded_tiles_selected" \
        --tile "$TILE_SIZE" 2>&1 | grep -E "(tiles|complete)" || true
done

echo -e "${GREEN}✓ Tile decoding complete${NC}"
echo ""

# Step 5: Extract DICOM Source Tiles
echo -e "${BLUE}[5/7] Extracting DICOM source tiles for comparison...${NC}"
pip3 install -q pydicom 2>/dev/null || true

python3 << 'PYEOF'
import pydicom
import json
from pathlib import Path

analysis = json.loads(Path('evals/runs/wsi_for_family_eval_rerun/family_analysis.json').read_text())
families = [(r['x2'], r['y2']) for r in analysis['representatives'][:6]]

ds = pydicom.dcmread('/workspace/data/3DHISTECH-2-256/4_1')
tiles_x = int(ds.TotalPixelMatrixColumns) // int(ds.Columns)

out_dir = Path('evals/runs/wsi_for_family_eval_rerun/dicom_source_tiles')
out_dir.joinpath('L0').mkdir(parents=True, exist_ok=True)
out_dir.joinpath('L1').mkdir(parents=True, exist_ok=True)

from pydicom.encaps import generate_pixel_data_frame

total = 0
for x2, y2 in families:
    # L1 tiles
    for dy1 in range(2):
        for dx1 in range(2):
            x1, y1 = x2 * 2 + dx1, y2 * 2 + dy1
            idx = y1 * tiles_x + x1
            if idx < ds.NumberOfFrames:
                frame = next(generate_pixel_data_frame(ds.PixelData, idx))
                out_dir.joinpath(f'L1/{x1}_{y1}.jpg').write_bytes(frame)
                total += 1
    
    # L0 tiles
    for dy0 in range(4):
        for dx0 in range(4):
            x0, y0 = x2 * 4 + dx0, y2 * 4 + dy0
            idx = y0 * tiles_x + x0
            if idx < ds.NumberOfFrames:
                frame = next(generate_pixel_data_frame(ds.PixelData, idx))
                out_dir.joinpath(f'L0/{x0}_{y0}.jpg').write_bytes(frame)
                total += 1

print(f"✓ Extracted {total} DICOM source tiles")
PYEOF

echo -e "${GREEN}✓ DICOM extraction complete${NC}"
echo ""

# Step 6: Compute Quality Metrics
echo -e "${BLUE}[6/7] Computing quality metrics (PSNR/SSIM)...${NC}"
pip3 install -q scikit-image pillow numpy 2>/dev/null || true

python3 << 'PYEOF'
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import json

decoded_dir = Path('evals/runs/wsi_for_family_eval_rerun/decoded_tiles_selected')
source_dir = Path('evals/runs/wsi_for_family_eval_rerun/dicom_source_tiles')

results = {'families': [], 'summary': {}}

for level in ['L0', 'L1']:
    decoded_tiles = sorted(decoded_dir.joinpath(level).glob('*.jpg'))
    
    level_psnr = []
    level_ssim = []
    
    for dec_path in decoded_tiles:
        src_path = source_dir / level / dec_path.name
        
        if not src_path.exists():
            continue
        
        dec_img = np.array(Image.open(dec_path))
        src_img = np.array(Image.open(src_path))
        
        p = psnr(src_img, dec_img)
        
        if len(dec_img.shape) == 3:
            s = np.mean([ssim(src_img[:,:,i], dec_img[:,:,i], data_range=255) 
                        for i in range(dec_img.shape[2])])
        else:
            s = ssim(src_img, dec_img, data_range=255)
        
        level_psnr.append(p)
        level_ssim.append(s)
    
    if level_psnr:
        print(f"{level}: {len(level_psnr)} tiles")
        print(f"  PSNR: {np.mean(level_psnr):.2f} dB (range {np.min(level_psnr):.2f}-{np.max(level_psnr):.2f})")
        print(f"  SSIM: {np.mean(level_ssim):.4f} (range {np.min(level_ssim):.4f}-{np.max(level_ssim):.4f})")
        
        results['summary'][level] = {
            'tile_count': len(level_psnr),
            'psnr_mean': float(np.mean(level_psnr)),
            'psnr_min': float(np.min(level_psnr)),
            'psnr_max': float(np.max(level_psnr)),
            'ssim_mean': float(np.mean(level_ssim)),
            'ssim_min': float(np.min(level_ssim)),
            'ssim_max': float(np.max(level_ssim)),
        }

Path('evals/runs/wsi_for_family_eval_rerun/quality_metrics.json').write_text(json.dumps(results, indent=2))
print(f"\n✓ Metrics saved to quality_metrics.json")
PYEOF

echo -e "${GREEN}✓ Quality metrics computed${NC}"
echo ""

# Step 7: Generate Report
echo -e "${BLUE}[7/7] Generating evaluation report...${NC}"
python3 << 'PYEOF'
import json
from pathlib import Path

fa = json.loads(Path('evals/runs/wsi_for_family_eval_rerun/family_analysis.json').read_text())
qm = json.loads(Path('evals/runs/wsi_for_family_eval_rerun/quality_metrics.json').read_text())

bundle_path = Path('evals/runs/wsi_for_family_eval_rerun/residual_packs/residuals.bundle')
bundle_size_mb = bundle_path.stat().st_size / (1024**2)
dicom_size_mb = 1159.5

print("="*80)
print("ORIGAMI WSI Evaluation Report")
print("="*80)
print()
print(f"Compression: {dicom_size_mb/bundle_size_mb:.2f}x ({dicom_size_mb:.1f} MB → {bundle_size_mb:.1f} MB)")
print(f"Families: {fa['non_empty_families']:,} non-empty (P50: {fa['statistics']['p50']/1024:.1f} KB)")
print()
print("Quality Metrics:")
for level in ['L0', 'L1']:
    if level in qm['summary']:
        m = qm['summary'][level]
        print(f"  {level}: {m['psnr_mean']:.2f} dB PSNR, {m['ssim_mean']:.4f} SSIM ({m['tile_count']} tiles)")
print()
print("="*80)
PYEOF

echo -e "${GREEN}✓ Evaluation complete!${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
