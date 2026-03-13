#!/usr/bin/env bash
# JXL Served vs ORIGAMI v2 sweep — no optL2, 444 subsampling, seed size 256.
#
# Generates:
#   1. JPEGXL baseline runs:  jpegxl_jpeg_baseline_q{Q}  (Q=30..90, step 10)
#   2. ORIGAMI v2 b90 runs:   v2_b90_l0q{Q}_ss256_nooptl2  (Q=30..90, step 5)
#   3. ORIGAMI v2 b95 runs:   v2_b95_l0q{Q}_ss256_nooptl2  (Q=30..90, step 5)
#
# Skips runs that already exist (directory with manifest.json).
# After encoding, runs compute_metrics.py to add full visual metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/../runs"
IMAGE="$SCRIPT_DIR/../test-images/L0-1024.jpg"
BIN="${BIN:-/Users/andrewluetgers/projects/dev/RRIP/server/target2/release/origami}"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Test image not found: $IMAGE"
    exit 1
fi

if [ ! -f "$BIN" ]; then
    echo "ERROR: origami binary not found: $BIN"
    exit 1
fi

echo "=== JXL vs ORIGAMI v2 sweep ==="
echo "  Binary: $BIN"
echo "  Image:  $IMAGE"
echo "  Runs:   $RUNS_DIR"
echo ""

# ─── Phase 1: JPEGXL baseline runs ───
echo "── Phase 1: JPEGXL baselines ──"
for Q in 30 40 50 60 70 80 90; do
    DIR="$RUNS_DIR/jpegxl_jpeg_baseline_q${Q}"
    if [ -f "$DIR/manifest.json" ]; then
        echo "  SKIP jpegxl_jpeg_baseline_q${Q} (exists)"
        continue
    fi
    echo "  RUN  jpegxl_jpeg_baseline_q${Q}"
    uv run python "$SCRIPT_DIR/jpeg_baseline.py" \
        --image "$IMAGE" --quality "$Q" --encoder jpegxl \
        --output "$DIR"
done

# ─── Phase 2: ORIGAMI v2 b90 runs ───
echo ""
echo "── Phase 2: ORIGAMI v2 b90 (no optL2, 444, ss256) ──"
for Q in 30 35 40 45 50 55 60 65 70 75 80 85 90; do
    NAME="v2_b90_l0q${Q}_ss256_nooptl2"
    DIR="$RUNS_DIR/$NAME"
    if [ -f "$DIR/manifest.json" ]; then
        echo "  SKIP $NAME (exists)"
        continue
    fi
    echo "  RUN  $NAME"
    "$BIN" encode \
        --image "$IMAGE" --out "$DIR" \
        --seedq 90 --l0q "$Q" --seed-size 256 \
        --subsamp 444 --upsample-filter lanczos3 \
        --manifest --debug-images --pack
done

# ─── Phase 3: ORIGAMI v2 b95 runs ───
echo ""
echo "── Phase 3: ORIGAMI v2 b95 (no optL2, 444, ss256) ──"
for Q in 30 35 40 45 50 55 60 65 70 75 80 85 90; do
    NAME="v2_b95_l0q${Q}_ss256_nooptl2"
    DIR="$RUNS_DIR/$NAME"
    if [ -f "$DIR/manifest.json" ]; then
        echo "  SKIP $NAME (exists)"
        continue
    fi
    echo "  RUN  $NAME"
    "$BIN" encode \
        --image "$IMAGE" --out "$DIR" \
        --seedq 95 --l0q "$Q" --seed-size 256 \
        --subsamp 444 --upsample-filter lanczos3 \
        --manifest --debug-images --pack
done

# ─── Phase 4: Compute full visual metrics ───
echo ""
echo "── Phase 4: Computing visual metrics ──"

# Compute metrics for all ORIGAMI runs that have compress/decompress dirs
# --serve-quality 95 re-encodes reconstructed tiles through JPEG Q95 before
# measuring, to simulate what the tile server actually delivers.
for BASEQ in 90 95; do
    for Q in 30 35 40 45 50 55 60 65 70 75 80 85 90; do
        NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2"
        DIR="$RUNS_DIR/$NAME"
        if [ -d "$DIR/compress" ] && [ -d "$DIR/decompress" ]; then
            echo "  METRICS $NAME"
            uv run python "$SCRIPT_DIR/compute_metrics.py" --serve-quality 95 "$DIR" 2>&1 | tail -1
        fi
    done
done

echo ""
echo "=== Sweep complete ==="
echo "Now run: uv run python evals/scripts/generate_jxl_vs_origami_charts.py"