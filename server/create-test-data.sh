#!/bin/bash
# Create minimal test dataset for CI/CD performance testing
# This creates just enough tiles to test the reconstruction pipeline

set -e

TEST_DATA_DIR="test-data"
SLIDE_NAME="demo_out"
SOURCE_DATA="../data/$SLIDE_NAME"

echo "Creating minimal test dataset in $TEST_DATA_DIR..."

# Clean and create test data directory
rm -rf $TEST_DATA_DIR
mkdir -p $TEST_DATA_DIR/$SLIDE_NAME

# Copy DZI manifest
cp $SOURCE_DATA/baseline_pyramid.dzi $TEST_DATA_DIR/$SLIDE_NAME/

# Copy summary.json if exists
if [ -f "$SOURCE_DATA/summary.json" ]; then
    cp $SOURCE_DATA/summary.json $TEST_DATA_DIR/$SLIDE_NAME/
fi

# Function to copy a tile and its residuals
copy_tile_family() {
    local x2=$1
    local y2=$2

    echo "Copying tile family for L2 tile ($x2, $y2)..."

    # Copy L2 baseline tile (always needed)
    local l2_dir="$TEST_DATA_DIR/$SLIDE_NAME/baseline_pyramid_files/14"
    mkdir -p "$l2_dir"
    if [ -f "$SOURCE_DATA/baseline_pyramid_files/14/${x2}_${y2}.jpg" ]; then
        cp "$SOURCE_DATA/baseline_pyramid_files/14/${x2}_${y2}.jpg" "$l2_dir/"
    fi

    # Copy L1 tiles and residuals (2x2 grid)
    for dy in 0 1; do
        for dx in 0 1; do
            local x1=$((x2 * 2 + dx))
            local y1=$((y2 * 2 + dy))

            # Copy L1 baseline tile
            local l1_dir="$TEST_DATA_DIR/$SLIDE_NAME/baseline_pyramid_files/15"
            mkdir -p "$l1_dir"
            if [ -f "$SOURCE_DATA/baseline_pyramid_files/15/${x1}_${y1}.jpg" ]; then
                cp "$SOURCE_DATA/baseline_pyramid_files/15/${x1}_${y1}.jpg" "$l1_dir/"
            fi

            # Copy L1 residual
            local l1_res_dir="$TEST_DATA_DIR/$SLIDE_NAME/residuals_q32/L1/${x2}_${y2}"
            mkdir -p "$l1_res_dir"
            if [ -f "$SOURCE_DATA/residuals_q32/L1/${x2}_${y2}/${x1}_${y1}.jpg" ]; then
                cp "$SOURCE_DATA/residuals_q32/L1/${x2}_${y2}/${x1}_${y1}.jpg" "$l1_res_dir/"
            fi

            # Copy L0 tiles and residuals (2x2 grid for each L1)
            for dy0 in 0 1; do
                for dx0 in 0 1; do
                    local x0=$((x1 * 2 + dx0))
                    local y0=$((y1 * 2 + dy0))

                    # Copy L0 baseline tile
                    local l0_dir="$TEST_DATA_DIR/$SLIDE_NAME/baseline_pyramid_files/16"
                    mkdir -p "$l0_dir"
                    if [ -f "$SOURCE_DATA/baseline_pyramid_files/16/${x0}_${y0}.jpg" ]; then
                        cp "$SOURCE_DATA/baseline_pyramid_files/16/${x0}_${y0}.jpg" "$l0_dir/"
                    fi

                    # Copy L0 residual
                    local l0_res_dir="$TEST_DATA_DIR/$SLIDE_NAME/residuals_q32/L0/${x2}_${y2}"
                    mkdir -p "$l0_res_dir"
                    if [ -f "$SOURCE_DATA/residuals_q32/L0/${x2}_${y2}/${x0}_${y0}.jpg" ]; then
                        cp "$SOURCE_DATA/residuals_q32/L0/${x2}_${y2}/${x0}_${y0}.jpg" "$l0_res_dir/"
                    fi
                done
            done
        done
    done

    # Copy residual pack if it exists
    local pack_dir="$TEST_DATA_DIR/$SLIDE_NAME/residual_packs"
    mkdir -p "$pack_dir"
    if [ -f "$SOURCE_DATA/residual_packs/${x2}_${y2}.pack" ]; then
        cp "$SOURCE_DATA/residual_packs/${x2}_${y2}.pack" "$pack_dir/"
    fi
}

# Copy lower resolution levels entirely (they're small)
echo "Copying lower resolution levels..."
for level in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    if [ -d "$SOURCE_DATA/baseline_pyramid_files/$level" ]; then
        mkdir -p "$TEST_DATA_DIR/$SLIDE_NAME/baseline_pyramid_files/$level"
        cp -r "$SOURCE_DATA/baseline_pyramid_files/$level"/* \
              "$TEST_DATA_DIR/$SLIDE_NAME/baseline_pyramid_files/$level/" 2>/dev/null || true
    fi
done

# Copy a few tile families from L2 (level 14) for testing
# These coordinates should exist in most datasets
copy_tile_family 100 100  # Center-ish tile
copy_tile_family 50 50    # Different area
copy_tile_family 150 150  # Another area

# Calculate size
echo ""
echo "Test dataset created successfully!"
echo "Size: $(du -sh $TEST_DATA_DIR | cut -f1)"
echo "Files: $(find $TEST_DATA_DIR -type f | wc -l | tr -d ' ')"

# Create a compressed archive for CI
echo ""
echo "Creating compressed archive for CI..."
tar -czf test-data.tar.gz -C $TEST_DATA_DIR .
echo "Archive size: $(du -sh test-data.tar.gz | cut -f1)"

echo ""
echo "To use in CI, upload test-data.tar.gz as a GitHub release artifact"
echo "or store it in a cloud bucket and download during CI runs."