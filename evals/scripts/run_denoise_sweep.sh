#!/usr/bin/env bash
# Denoise comparison sweep — with configurable denoise weight and synth strength.
#
# Generates:
#   1. ORIGAMI v2 b{90,95} without denoise:  v2_b{BQ}_l0q{Q}_ss256_nooptl2
#   2. ORIGAMI v2 b{90,95} with denoise:     v2_b{BQ}_l0q{Q}_ss256_nooptl2_dn{DW}_sn{SS}
#
# L0Q sweep: 40, 50, 60, 70, 80, 90, 95
#
# Environment variables:
#   DENOISE_WEIGHT  — denoise weight 0-100 (default: 100 = full denoise)
#   SYNTH_STRENGTH  — synth strength 0-100 (default: 80 = 0.8)
#
# Skips runs that already exist (directory with manifest.json).
# After encoding, runs compute_metrics.py to add full visual metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/../runs"
IMAGE="$SCRIPT_DIR/../test-images/L0-1024.jpg"
BIN="${BIN:-/Users/andrewluetgers/projects/dev/RRIP/server/target2/release/origami}"

DW="${DENOISE_WEIGHT:-100}"
SS="${SYNTH_STRENGTH:-80}"
# Convert to float for CLI
DW_FLOAT=$(python3 -c "print($DW / 100)")
SS_FLOAT=$(python3 -c "print($SS / 100)")

if [ ! -f "$IMAGE" ]; then echo "ERROR: Test image not found: $IMAGE"; exit 1; fi
if [ ! -f "$BIN" ]; then echo "ERROR: origami binary not found: $BIN"; exit 1; fi

echo "=== Denoise comparison sweep ==="
echo "  Binary: $BIN"
echo "  Image:  $IMAGE"
echo "  Denoise weight: $DW_FLOAT ($DW)"
echo "  Synth strength: $SS_FLOAT ($SS)"
echo ""

QUALITIES="40 50 60 70 80 90 95"

for BASEQ in 90 95; do
    # ─── Without denoise ───
    echo "── ORIGAMI v2 b${BASEQ} (no denoise) ──"
    for Q in $QUALITIES; do
        NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2"
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
            --manifest --debug-images --pack
    done

    # ─── With denoise ───
    echo ""
    echo "── ORIGAMI v2 b${BASEQ} (dn${DW} sn${SS}) ──"
    for Q in $QUALITIES; do
        NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2_dn${DW}_sn${SS}"
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
            --denoise --denoise-weight "$DW_FLOAT" --synth-strength "$SS_FLOAT" \
            --manifest --debug-images --pack
    done
    echo ""
done

# ─── Compute full visual metrics ───
echo "── Computing visual metrics ──"
for BASEQ in 90 95; do
    for Q in $QUALITIES; do
        for SUFFIX in "" "_dn${DW}_sn${SS}"; do
            NAME="v2_b${BASEQ}_l0q${Q}_ss256_nooptl2${SUFFIX}"
            DIR="$RUNS_DIR/$NAME"
            if [ -d "$DIR/compress" ] && [ -d "$DIR/decompress" ]; then
                echo "  METRICS $NAME"
                uv run python "$SCRIPT_DIR/compute_metrics.py" --serve-quality 95 "$DIR" 2>&1 | tail -1
            fi
        done
    done
done

echo ""
echo "=== Sweep complete ==="
