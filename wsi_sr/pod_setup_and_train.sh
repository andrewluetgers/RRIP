#!/bin/bash
set -e

# ============================================================================
# Run this ON the RunPod pod after:
#   1. scp the DICOM file to /workspace/000005.dcm
#   2. git clone / git pull the repo to /workspace/RRIP
#   3. ssh in and run: bash /workspace/RRIP/wsi_sr/pod_setup_and_train.sh
# ============================================================================

echo "=== WSI SR Pod Setup & Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

cd /workspace

# ---- Install dependencies ----
echo "Installing dependencies..."
pip install -q torch torchvision pillow numpy scikit-image lpips onnx onnxruntime pydicom 2>&1 | tail -3

# ---- Extract tiles ----
TILES_DIR=/workspace/tiles
DCM_FILE=/workspace/000005.dcm

if [ ! -d "$TILES_DIR" ] || [ "$(ls $TILES_DIR/*.png 2>/dev/null | wc -l)" -lt 10 ]; then
    echo ""
    echo "Extracting tiles from DICOM..."
    if [ ! -f "$DCM_FILE" ]; then
        echo "ERROR: $DCM_FILE not found. Upload it first:"
        echo "  scp -i ~/.ssh/id_runpod -P <PORT> data/3DHISTECH-1-extract/000005.dcm root@<IP>:/workspace/"
        exit 1
    fi
    cd /workspace/RRIP/wsi_sr
    python prepare_tiles.py --dcm $DCM_FILE --outdir $TILES_DIR --blank-tolerance 10
else
    echo "Tiles already extracted: $(ls $TILES_DIR/*.png | wc -l) tiles"
fi

echo ""
echo "Tile count: $(ls $TILES_DIR/*.png 2>/dev/null | wc -l)"

# ---- Train ----
cd /workspace/RRIP/wsi_sr
CKPT_DIR=/workspace/checkpoints

echo ""
echo "Starting training..."
echo "  Mode: sr (256→1024)"
echo "  Batch: 16"
echo "  Crop: 256 (from 1024x1024 targets)"
echo "  JPEG quality simulation: 95"
echo "  Checkpoints: $CKPT_DIR"
echo ""

python train.py \
    --tiles $TILES_DIR \
    --mode sr \
    --epochs 200 \
    --batch 16 \
    --crop 256 \
    --lr 2e-4 \
    --jpeg-quality 95 \
    --perceptual-weight 0.01 \
    --channels 16 \
    --blocks 5 \
    --outdir $CKPT_DIR \
    --workers 4

echo ""
echo "Training complete!"
echo ""

# ---- Export ----
echo "Exporting model..."
python export.py \
    --checkpoint $CKPT_DIR/best.pt \
    --output /workspace/model_sr.onnx \
    --quantize \
    --benchmark

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Checkpoint: $CKPT_DIR/best.pt"
echo "  ONNX:       /workspace/model_sr.onnx"
echo "  INT8:       /workspace/model_sr_int8.onnx"
echo ""
echo "To download results:"
echo "  scp -i ~/.ssh/id_runpod -P <PORT> root@<IP>:/workspace/model_sr.onnx ./"
echo "  scp -i ~/.ssh/id_runpod -P <PORT> root@<IP>:/workspace/checkpoints/best.pt ./"
