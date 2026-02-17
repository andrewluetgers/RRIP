#!/bin/bash
# Rebuild GPU encoder with pyramid wait fix and test GPU+CPU overlap
set -e

echo "=== Pulling latest code ==="
cd /workspace/RRIP
git pull

echo ""
echo "=== Rebuilding GPU encoder (CUDA sm_90) ==="
cd gpu-encode
CUDA_ARCH=sm_90 cargo build --release

echo ""
echo "=== Testing 5 sequential slides with background pyramids ==="
bash /workspace/RRIP/evals/scripts/run_5_concurrent_wsi.sh
