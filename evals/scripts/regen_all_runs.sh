#!/usr/bin/env bash
# Regenerate ALL existing v2 ss256 runs with the current binary.
# Deletes each run, re-encodes with same params, re-computes metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/../runs"
IMAGE="$SCRIPT_DIR/../test-images/L0-1024.jpg"
BIN="${BIN:-/Users/andrewluetgers/projects/dev/RRIP/server/target2/release/origami}"

if [ ! -f "$IMAGE" ]; then echo "ERROR: Test image not found: $IMAGE"; exit 1; fi
if [ ! -f "$BIN" ]; then echo "ERROR: origami binary not found: $BIN"; exit 1; fi

echo "=== Regenerating all runs with current binary ==="
echo "  Binary: $BIN"
echo "  Image:  $IMAGE"

# Parse each existing run directory and re-encode
for DIR in "$RUNS_DIR"/v2_b*_ss256_*; do
    [ -d "$DIR" ] || continue
    NAME="$(basename "$DIR")"

    # Parse params from manifest
    MANIFEST="$DIR/manifest.json"
    [ -f "$MANIFEST" ] || continue

    SEEDQ=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('seedq', 95))")
    L0Q=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('l0q', 50))")

    DENOISE_FLAG=""
    if [[ "$NAME" == *_denoise ]]; then
        DENOISE_FLAG="--denoise"
    fi

    echo "  REGEN $NAME (seedq=$SEEDQ l0q=$L0Q denoise=${DENOISE_FLAG:-no})"
    rm -rf "$DIR"

    "$BIN" encode \
        --image "$IMAGE" --out "$DIR" \
        --seedq "$SEEDQ" --l0q "$L0Q" --seed-size 256 \
        --subsamp 444 --upsample-filter lanczos3 \
        --manifest --debug-images --pack \
        $DENOISE_FLAG
done

# Also regenerate any other v2 runs (from jxl sweep etc)
for DIR in "$RUNS_DIR"/v2_b*_nooptl2; do
    [ -d "$DIR" ] || continue
    NAME="$(basename "$DIR")"
    [[ "$NAME" == *_ss256_* ]] && continue  # already handled above

    MANIFEST="$DIR/manifest.json"
    [ -f "$MANIFEST" ] || continue

    SEEDQ=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('seedq', 95))")
    L0Q=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('l0q', 50))")

    echo "  REGEN $NAME (seedq=$SEEDQ l0q=$L0Q)"
    rm -rf "$DIR"

    "$BIN" encode \
        --image "$IMAGE" --out "$DIR" \
        --seedq "$SEEDQ" --l0q "$L0Q" --seed-size 256 \
        --subsamp 444 --upsample-filter lanczos3 \
        --manifest --debug-images --pack
done

echo ""
echo "── Computing visual metrics ──"
for DIR in "$RUNS_DIR"/v2_b*; do
    [ -d "$DIR/compress" ] && [ -d "$DIR/decompress" ] || continue
    NAME="$(basename "$DIR")"
    echo "  METRICS $NAME"
    uv run python "$SCRIPT_DIR/compute_metrics.py" --serve-quality 95 "$DIR" 2>&1 | tail -1
done

echo ""
echo "=== Regeneration complete ==="
