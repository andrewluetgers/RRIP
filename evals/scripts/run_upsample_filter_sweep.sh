#!/bin/bash
# Upsample filter sweep — compare bilinear/bicubic/lanczos3 across subsampling modes.
# Tests the impact of the new --upsample-filter and --downsample-filter flags.
# All runs use --optl2 with fixed pipeline order (OptL2 first, sharpen second).
#
# Naming: uf_{filter}_{subsamp}_optl2_l1q{N}_l0q{N}
#   filter: bl (bilinear), bc (bicubic), l3 (lanczos3)
set -e
cd "$(dirname "$0")/../.."

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Run: cd server && CARGO_TARGET_DIR=target2 cargo build --release"
    exit 1
fi
if [ ! -f "$IMG" ]; then
    echo "ERROR: Test image not found: $IMG"
    exit 1
fi

# Filter codes → CLI values (avoid bash 4+ associative arrays for macOS compat)
filter_name() {
    case "$1" in
        bl) echo "bilinear" ;;
        bc) echo "bicubic" ;;
        l3) echo "lanczos3" ;;
    esac
}

FILTERS=(bl bc l3)
SUBSAMPS=(444 420 420opt)
SPLITS=(
    "60:40"
    "80:60"
    "90:70"
)

COUNT=0
TOTAL=$(( ${#FILTERS[@]} * ${#SUBSAMPS[@]} * ${#SPLITS[@]} ))

echo "=== Upsample Filter Sweep ==="
echo "Filters: ${FILTERS[*]}"
echo "Subsampling: ${SUBSAMPS[*]}"
echo "Quality points: ${SPLITS[*]}"
echo "Total runs: $TOTAL"
echo ""

for F in "${FILTERS[@]}"; do
    FVAL=$(filter_name "$F")
    for SS in "${SUBSAMPS[@]}"; do
        for S in "${SPLITS[@]}"; do
            L1Q="${S%%:*}"
            L0Q="${S##*:}"
            DIR="evals/runs/uf_${F}_${SS}_optl2_l1q${L1Q}_l0q${L0Q}"
            COUNT=$((COUNT + 1))

            if [ -d "$DIR" ]; then
                echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
                continue
            fi

            echo "  [$COUNT/$TOTAL] ${FVAL} ${SS} optL2 l1q${L1Q} l0q${L0Q}..."
            $BIN encode --image "$IMG" --out "$DIR" \
                --l1q "$L1Q" --l0q "$L0Q" --subsamp "$SS" --optl2 \
                --upsample-filter "$FVAL" --downsample-filter lanczos3 \
                --manifest --debug-images 2>&1 | tail -1
        done
    done
done

echo ""
echo "=== Sweep complete: $COUNT runs ==="
echo ""
echo "Next steps:"
echo "  1. Compute full metrics:  uv run python evals/scripts/compute_metrics.py evals/runs/uf_*"
echo "  2. Start viewer:          cd evals/viewer && pnpm start"
echo "  3. Open http://localhost:8084 and filter by 'UF' to see results"
