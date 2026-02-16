#!/bin/bash
set -e

SLIDE_URL="https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip"
DATA_DIR="data/3dhistech1"
ZIP_FILE="data/3DHISTECH-1.zip"

# Download if not present
if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading 3DHISTECH-1 sample slide (345 MB)..."
    mkdir -p data
    curl -L -o "$ZIP_FILE" "$SLIDE_URL"
fi

# Extract
if [ ! -d "data/3DHISTECH-1" ]; then
    echo "Extracting..."
    cd data && unzip -q 3DHISTECH-1.zip && cd ..
fi

# Find the DICOM directory (contains .dcm files)
SLIDE_PATH=$(find data/3DHISTECH-1 -name "*.dcm" -print -quit | xargs dirname)

if [ -z "$SLIDE_PATH" ]; then
    echo "ERROR: No .dcm files found in data/3DHISTECH-1"
    exit 1
fi

echo "Slide path: $SLIDE_PATH"

# Build origami with openslide support
# TURBOJPEG_SOURCE=pkg-config avoids libjpeg ABI version conflict between
# statically-linked libjpeg62 and OpenSlide's dynamically-linked libjpeg80
echo "Building origami with openslide..."
cd server
TURBOJPEG_SOURCE=pkg-config CMAKE_POLICY_VERSION_MINIMUM=3.5 CARGO_TARGET_DIR=target2 \
    cargo build --release --features openslide
cd ..

# Run ingest
echo "Ingesting slide with recommended settings..."
./server/target2/release/origami ingest \
    --slide "$SLIDE_PATH" \
    --out "$DATA_DIR" \
    --subsamp 444 \
    --optl2 \
    --l1q 60 \
    --l0q 40 \
    --baseq 95 \
    --pack

echo ""
echo "Done! Slide ready at: $DATA_DIR"
echo "Start server: origami serve --slides-root data/ --port 3007"
echo "View at: http://localhost:3007/viewer/3dhistech1"
