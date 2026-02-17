#!/bin/bash
set -e

# Encode a WSI (DICOM, SVS, etc.) using the CPU ingest pipeline.
#
# Usage:
#   ./evals/scripts/run_wsi_encode_cpu.sh <slide_path> <baseq> <l1q> <l0q> [options...]
#
# Examples:
#   # Encode 3DHISTECH-1 with b80, l1q60, l0q40, 444, optl2, delta 20
#   ./evals/scripts/run_wsi_encode_cpu.sh data/3DHISTECH-1-extract/000005.dcm 80 60 40
#
#   # Same with custom delta
#   ./evals/scripts/run_wsi_encode_cpu.sh data/3DHISTECH-1-extract/000005.dcm 80 60 40 --max-delta 20
#
#   # Without optl2
#   ./evals/scripts/run_wsi_encode_cpu.sh data/3DHISTECH-1-extract/000005.dcm 80 60 40 --no-optl2
#
#   # Custom subsamp
#   ./evals/scripts/run_wsi_encode_cpu.sh data/3DHISTECH-1-extract/000005.dcm 80 60 40 --subsamp 420

SLIDE="$1"
BASEQ="$2"
L1Q="$3"
L0Q="$4"
shift 4

if [ -z "$SLIDE" ] || [ -z "$BASEQ" ] || [ -z "$L1Q" ] || [ -z "$L0Q" ]; then
    echo "Usage: $0 <slide_path> <baseq> <l1q> <l0q> [options...]"
    echo ""
    echo "Required args:"
    echo "  slide_path   Path to WSI file (.dcm, .svs, .tiff, or DICOM directory)"
    echo "  baseq        L2 baseline JPEG quality (e.g. 80, 90, 95)"
    echo "  l1q          L1 residual JPEG quality (e.g. 60)"
    echo "  l0q          L0 residual JPEG quality (e.g. 40)"
    echo ""
    echo "Optional flags (passed after the 4 required args):"
    echo "  --subsamp N      Chroma subsampling: 444, 420, 420opt (default: 444)"
    echo "  --max-delta N    OptL2 max pixel deviation (default: 20)"
    echo "  --no-optl2       Disable OptL2 gradient descent"
    echo "  --max-parents N  Limit families to process (for testing)"
    echo "  --encoder E      Encoder backend: turbojpeg, mozjpeg, jpegli"
    exit 1
fi

# Defaults
SUBSAMP="444"
DELTA="20"
OPTL2="--optl2"
EXTRA_ARGS=()

# Parse optional flags
while [ $# -gt 0 ]; do
    case "$1" in
        --subsamp)
            SUBSAMP="$2"
            shift 2
            ;;
        --max-delta)
            DELTA="$2"
            shift 2
            ;;
        --no-optl2)
            OPTL2=""
            DELTA=""
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Derive slide name from path for output directory
SLIDE_NAME=$(basename "$(dirname "$SLIDE")" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/_$//')
# If the parent dir name is just "." or empty, use the file stem
if [ "$SLIDE_NAME" = "." ] || [ -z "$SLIDE_NAME" ]; then
    SLIDE_NAME=$(basename "$SLIDE" | sed 's/\.[^.]*$//' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
fi

# Build output directory name
OUT_NAME="cpu_ingest_${SLIDE_NAME}_b${BASEQ}_l1q${L1Q}_l0q${L0Q}_${SUBSAMP}"
if [ -n "$OPTL2" ]; then
    OUT_NAME="${OUT_NAME}_optl2_d${DELTA}"
fi
OUT_DIR="evals/runs/${OUT_NAME}"

ORIGAMI="./server/target2/release/origami"

# Check binary exists
if [ ! -x "$ORIGAMI" ]; then
    echo "ERROR: origami binary not found at $ORIGAMI"
    echo "Build with: cd server && TURBOJPEG_SOURCE=pkg-config CMAKE_POLICY_VERSION_MINIMUM=3.5 CARGO_TARGET_DIR=target2 cargo build --release --features openslide"
    exit 1
fi

# Check slide exists
if [ ! -e "$SLIDE" ]; then
    echo "ERROR: Slide not found: $SLIDE"
    exit 1
fi

echo "=== WSI CPU Encode ==="
echo "Slide:    $SLIDE"
echo "Output:   $OUT_DIR"
echo "Settings: baseq=$BASEQ l1q=$L1Q l0q=$L0Q subsamp=$SUBSAMP optl2=$([ -n "$OPTL2" ] && echo "yes d=$DELTA" || echo "no")"
echo ""

mkdir -p "$OUT_DIR"

# Build the command
CMD=("$ORIGAMI" ingest
    --slide "$SLIDE"
    --out "$OUT_DIR"
    --baseq "$BASEQ"
    --l1q "$L1Q"
    --l0q "$L0Q"
    --subsamp "$SUBSAMP"
    --pack)

if [ -n "$OPTL2" ]; then
    CMD+=($OPTL2 --max-delta "$DELTA")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
echo ""

/usr/bin/time -l "${CMD[@]}" 2>&1

echo ""
echo "Done! Output at: $OUT_DIR"
if [ -f "$OUT_DIR/stats.json" ]; then
    echo ""
    echo "Stats:"
    cat "$OUT_DIR/stats.json"
fi
