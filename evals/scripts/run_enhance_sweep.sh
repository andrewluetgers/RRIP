#!/usr/bin/env bash
# Enhancement sweep — with refine model applied at encode time.
#
# Generates:
#   1. ORIGAMI v2 b90 with enhance:  v2_b90_l0q{Q}_ss256_nooptl2_enh
#   2. ORIGAMI v2 b95 with enhance:  v2_b95_l0q{Q}_ss256_nooptl2_enh
#
# L0Q sweep: 40, 50, 60, 70, 80, 90, 95
#
# Skips runs that already exist (directory with manifest.json).
# After encoding, runs compute_metrics.py to add full visual metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/../runs"
IMAGE="$SCRIPT_DIR/../test-images/L0-1024.jpg"
BIN="${BIN:-/Users/andrewluetgers/projects/dev/RRIP/server/target2/release/origami}"
REFINE_MODEL="${REFINE_MODEL:-/Users/andrewluetgers/projects/dev/RRIP/models/refine_8ch15b.onnx}"

if [ ! -f "$IMAGE" ]; then echo "ERROR: Test image not found: $IMAGE"; exit 1; fi
if [ ! -f "$BIN" ]; then echo "ERROR: origami binary not found: $BIN"; exit 1; fi
if [ ! -f "$REFINE_MODEL" ]; then echo "ERROR: refine model not found: $REFINE_MODEL"; exit 1; fi

echo "=== Enhancement sweep ==="
echo "  Binary: $BIN"
echo "  Image:  $IMAGE"
echo "  Refine: $REFINE_MODEL"
echo ""

QUALITIES="40 50 60 70 80 90 95"

for BASEQ in 90 95; do
    echo "── ORIGAMI v2 b${BASEQ} (enhance) ──"
    for Q in $QUALITIES; do
        NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2_enh"
        DIR="$RUNS_DIR/$NAME"
        if [ -f "$DIR/manifest.json" ]; then
            echo "  SKIP $NAME (exists)"
            continue
        fi
        echo "  RUN  $NAME"
        "$BIN" encode \
            --image "$IMAGE" --out "$DIR" \
            --seedq "$BASEQ" --l0q "$Q" --seed-size 256 \
            --subsamp 444 --upsample-filter lanczos3 \
            --refine-model "$REFINE_MODEL" \
            --manifest --debug-images --pack
    done
    echo ""
done

# ─── Compute full visual metrics ───
echo "── Computing visual metrics ──"
for BASEQ in 90 95; do
    for Q in $QUALITIES; do
        NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2_enh"
        DIR="$RUNS_DIR/$NAME"
        if [ -d "$DIR/compress" ] && [ -d "$DIR/decompress" ]; then
            echo "  METRICS $NAME"
            uv run python "$SCRIPT_DIR/compute_metrics.py" --serve-quality 95 "$DIR" 2>&1 | tail -1
        fi
    done
done

echo ""
echo "=== Enhancement sweep complete ==="
