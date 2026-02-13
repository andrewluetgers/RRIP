#!/bin/bash
# =============================================================================
# generate-evals.sh — Regenerate all evaluation data for the ORIGAMI paper
#
# This reproduces the full set of comparison runs used in the publication:
#   - 9 JPEG baselines (Pillow/libjpeg, 4:2:0)
#   - 60 Rust ORIGAMI runs (4 configs x 9 uniform + 6 split quality points)
#   - 25 Python OptL2 runs (legacy pipeline, 17 uniform + 8 split)
#   - Perceptual metrics (PSNR, SSIM, VIF, Delta E, LPIPS) for all runs
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - Python 3.11+ with UV (https://astral.sh/uv)
#   - Test image: evals/test-images/L0-1024.jpg (not in git, 604 KB)
#   - ~1.2 GB disk space for output
#
# Timing (Apple M-series):
#   - Rust sweep:     ~8 minutes  (encoding + metrics)
#   - Python OptL2:   ~12 minutes (25 runs x ~28s each)
#   - JPEG baselines: ~2 minutes
#   - Total:          ~22 minutes
#
# Output: evals/runs/ (~1.2 GB)
#   94 run directories with manifests, debug images, and metrics
#
# Recommended config: origami encode --subsamp 444 --optl2 --l1q 60 --l0q 40
#   (rs_444_optl2_l1q60_l0q40 in the viewer)
#
# Usage:
#   ./generate-evals.sh          # Run everything
#   ./generate-evals.sh --rust   # Rust runs + baselines only
#   ./generate-evals.sh --python # Python OptL2 runs only
# =============================================================================
set -e
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "ORIGAMI Evaluation Data Generator"
echo "================================="
echo ""

# Check test image
if [ ! -f evals/test-images/L0-1024.jpg ]; then
    echo "ERROR: Test image not found: evals/test-images/L0-1024.jpg"
    echo "This image is not stored in git. Place a 1024x1024 JPEG test image there."
    exit 1
fi

# Check disk space (~1.2 GB needed)
AVAIL_MB=$(df -m . | tail -1 | awk '{print $4}')
if [ "$AVAIL_MB" -lt 2000 ]; then
    echo "WARNING: Only ${AVAIL_MB} MB free. This script generates ~1.2 GB of data."
    echo "Press Ctrl+C to abort, or Enter to continue."
    read -r
fi

RUN_RUST=true
RUN_PYTHON=true

if [ "$1" = "--rust" ]; then
    RUN_PYTHON=false
elif [ "$1" = "--python" ]; then
    RUN_RUST=false
fi

echo "Disk space required: ~1.2 GB"
echo "Estimated time:      ~22 minutes"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Build Rust encoder (if needed)
# ---------------------------------------------------------------------------
BIN=server/target2/release/origami

if [ "$RUN_RUST" = true ]; then
    if [ ! -f "$BIN" ]; then
        echo "=== Building Rust encoder ==="
        (cd server && CMAKE_POLICY_VERSION_MINIMUM=3.5 CARGO_TARGET_DIR=target2 cargo build --release)
        echo ""
    else
        echo "Rust encoder already built: $BIN"
        echo ""
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Rust ORIGAMI runs + JPEG baselines (60 Rust + 9 baselines)
# ---------------------------------------------------------------------------
if [ "$RUN_RUST" = true ]; then
    echo "=== Running Rust full sweep (60 ORIGAMI runs + 9 JPEG baselines + metrics) ==="
    echo "    This takes ~8 minutes..."
    echo ""
    bash evals/scripts/run_full_sweep.sh
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: Python OptL2 runs (25 runs, legacy pipeline for comparison)
# ---------------------------------------------------------------------------
if [ "$RUN_PYTHON" = true ]; then
    echo "=== Running Python OptL2 captures (25 runs) ==="
    echo "    This takes ~12 minutes..."
    echo ""
    bash evals/scripts/run_optl2_captures.sh
    echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Generation Complete ==="
NRUNS=$(find evals/runs -maxdepth 1 -mindepth 1 -type d ! -name '_*' | wc -l | tr -d ' ')
SIZE=$(du -sh evals/runs/ | cut -f1)
echo "  Runs generated: $NRUNS"
echo "  Total size:     $SIZE"
echo ""
echo "To view results:"
echo "  cd evals/viewer && pnpm install && pnpm start"
echo "  Open http://localhost:8084"
