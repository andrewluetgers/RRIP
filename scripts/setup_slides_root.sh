#!/usr/bin/env bash
# Create a slides root directory for the tile server with symlinked entries
# for all slides x variants (original DICOM, JXL Q80, JXL Q40).
#
# Usage: bash scripts/setup_slides_root.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/dev/data/WSI}"
DZI="$DATA_ROOT/dzi"
TRIMMED="$DATA_ROOT/dzi_trimmed"
TILE_SERVER="$DATA_ROOT/tile_server"
SLIDES_ROOT="$DATA_ROOT/slides"

mkdir -p "$SLIDES_ROOT"

# Slides: "slide_id|short_name" pairs
SLIDES=(
    "3DHISTECH-1|3DHISTECH-1"
    "1.14015440989672648675364084886809148499463738260226972174265074|Mayosh-1"
    "1.1696574164448579239218463574781198256317218134321893476353989|Mayosh-2"
    "1.17528887192371641093666263567808351206632016682106023572863174|Mayosh-3"
    "1.18138380607011795689803599019430937604835729012464563085136908|Mayosh-4"
    "1.19937629895563278921851301990326078924375009001496118315957075|Mayosh-5"
    "1.389659144153767661521953025760977742773734078850150366241606|Mayosh-6"
    "1.4799524995723208228415751787896218580176411759051689191694877|Mayosh-7"
    "1.8865208043685245218093034390302935172738148547962593389547055|Mayosh-8"
    "1.9408862073266419912979718694540485277777951032131549953134276|Mayosh-9"
)

compute_disk_size() {
    # Sum sizes of all image files in a directory tree (follows symlinks).
    # Counts .jpeg, .jpg, and .jxl — covers originals, thumbnails, and JXL L0.
    local dir="$1"
    find -L "$dir" \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.jxl" \) -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print int(s)}'
}

link_tissue_assets() {
    # Symlink all tissue detection assets into the slide directory.
    # These are shared across all variants of the same slide.
    local slide_id="$1"
    local dest="$2"

    for ext in tissue.map tissue.json tissue.svg \
               tissue_gray.jpg tissue_threshold.jpg tissue_mask.jpg tissue_margin_mask.jpg \
               tissue_color_grid.png tissue_tissue_bitmap.png tissue_margin_bitmap.png; do
        local src_file="$DZI/${slide_id}.${ext}"
        [ -f "$src_file" ] && ln -sf "$src_file" "$dest/"
    done
}

setup_original() {
    local slide_id="$1"
    local short_name="$2"
    local slug="${short_name}_original"
    local dest="$SLIDES_ROOT/$slug"

    echo "  Setting up $slug"
    mkdir -p "$dest"

    # Symlink DZI and files (idempotent)
    ln -sf "$TRIMMED/${slide_id}.dzi" "$dest/baseline_pyramid.dzi"
    ln -sf "$TRIMMED/${slide_id}_files" "$dest/baseline_pyramid_files"

    # Symlink all tissue detection assets
    link_tissue_assets "$slide_id" "$dest"

    # Compute disk size (all image files: .jpeg, .jpg, .jxl)
    local bytes
    bytes=$(compute_disk_size "$TRIMMED/${slide_id}_files")

    cat > "$dest/summary.json" << EOF
{
    "label": "${short_name} Original",
    "mode": "ingest-jpeg-only",
    "quality": "lossless",
    "total_bytes": $bytes,
    "pipeline_version": 2
}
EOF
}

setup_jxl() {
    local slide_id="$1"
    local short_name="$2"
    local quality="$3"
    local slug="${short_name}_jxl${quality}"
    local src="$TILE_SERVER/jxl_q${quality}/${slide_id}"
    local dest="$SLIDES_ROOT/$slug"

    if [ ! -d "$src" ]; then
        echo "  SKIP $slug (no source at $src)"
        return
    fi

    echo "  Setting up $slug"
    mkdir -p "$dest"

    # Symlink DZI and files (idempotent)
    ln -sf "$src/${slide_id}.dzi" "$dest/baseline_pyramid.dzi"
    ln -sf "$src/${slide_id}_files" "$dest/baseline_pyramid_files"

    # Symlink all tissue detection assets
    link_tissue_assets "$slide_id" "$dest"

    # Compute disk size (JXL L0 tiles + JPEG thumbnail levels)
    local bytes
    bytes=$(compute_disk_size "$src/${slide_id}_files")

    # Extract numeric quality (strip suffixes like _nonoise)
    local qnum="${quality%%_*}"

    cat > "$dest/summary.json" << EOF
{
    "label": "${short_name} JXL Q${quality}",
    "mode": "jxl-baseline",
    "quality": $qnum,
    "total_bytes": $bytes,
    "pipeline_version": 2
}
EOF
}

echo "Setting up slides root: $SLIDES_ROOT"
echo "Source trimmed DZIs: $TRIMMED"
echo "Source JXL variants: $TILE_SERVER"
echo ""

for entry in "${SLIDES[@]}"; do
    slide_id="${entry%%|*}"
    short_name="${entry##*|}"
    echo "=== $short_name ($slide_id) ==="
    setup_original "$slide_id" "$short_name"
    setup_jxl "$slide_id" "$short_name" 80
    setup_jxl "$slide_id" "$short_name" 40
    setup_jxl "$slide_id" "$short_name" "40_nonoise"
    # Static JPEG recompression baselines
    for jq in 80 40; do
        jslug="${short_name}_jpeg${jq}"
        jsrc="$TILE_SERVER/jpeg_q${jq}/${slide_id}"
        jdest="$SLIDES_ROOT/$jslug"
        if [ -d "$jsrc" ]; then
            echo "  Setting up $jslug"
            mkdir -p "$jdest"
            ln -sf "$jsrc/${slide_id}.dzi" "$jdest/baseline_pyramid.dzi"
            ln -sf "$jsrc/${slide_id}_files" "$jdest/baseline_pyramid_files"
            link_tissue_assets "$slide_id" "$jdest"
            # Always write summary with correct short name label
            jbytes=$(compute_disk_size "$jsrc/${slide_id}_files")
            cat > "$jdest/summary.json" << JEOF
{
    "label": "${short_name} JPEG Q${jq}",
    "mode": "ingest-jpeg-only",
    "quality": $jq,
    "total_bytes": $jbytes,
    "pipeline_version": 2
}
JEOF
        fi
    done
    echo ""
done

echo "=== Slides root contents ==="
ls -la "$SLIDES_ROOT"/
echo ""
echo "Start the tile server with:"
echo "  cd server && ./run-server.sh --slides-root $SLIDES_ROOT"
