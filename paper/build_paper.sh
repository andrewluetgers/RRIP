#!/bin/bash

# ORIGAMI Paper Build Script
# Generates the complete PDF with figures, references, and citations

set -e  # Exit on error

echo "================================"
echo "ORIGAMI Paper Build Script"
echo "================================"

# Check if LaTeX is installed
check_latex() {
    # Try to find pdflatex in common locations
    if command -v pdflatex &> /dev/null; then
        PDFLATEX="pdflatex"
    elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
        PDFLATEX="/Library/TeX/texbin/pdflatex"
    elif [ -f "/usr/local/bin/pdflatex" ]; then
        PDFLATEX="/usr/local/bin/pdflatex"
    else
        echo "❌ ERROR: pdflatex not found!"
        echo ""
        echo "Please install LaTeX first:"
        echo "  macOS:    brew install --cask basictex"
        echo "  Linux:    apt-get install texlive-full"
        echo ""
        echo "After installation, you may need to:"
        echo "  1. Restart your terminal"
        echo "  2. Add to PATH: export PATH=\"/Library/TeX/texbin:\$PATH\""
        exit 1
    fi
    echo "✓ LaTeX found: $PDFLATEX"
}

# Check if BibTeX is installed
check_bibtex() {
    # Try to find bibtex in common locations
    if command -v bibtex &> /dev/null; then
        BIBTEX="bibtex"
    elif [ -f "/Library/TeX/texbin/bibtex" ]; then
        BIBTEX="/Library/TeX/texbin/bibtex"
    elif [ -f "/usr/local/bin/bibtex" ]; then
        BIBTEX="/usr/local/bin/bibtex"
    else
        echo "⚠️  WARNING: bibtex not found - references won't be processed"
        return 1
    fi
    echo "✓ BibTeX found: $BIBTEX"
    return 0
}

# Clean previous build artifacts
clean_build() {
    echo ""
    echo "Cleaning previous build artifacts..."
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lot *.lof *.idx *.ind *.ilg
    echo "✓ Cleaned"
}

# Generate placeholder figures if they don't exist
generate_figures() {
    echo ""
    echo "Checking figures..."

    # Create figures directory if it doesn't exist
    mkdir -p figures

    # List of expected figures based on the paper
    declare -a figures=(
        "pyramid_structure.pdf"
        "reconstruction_pipeline.pdf"
        "compression_results.pdf"
        "quality_comparison.pdf"
        "timing_breakdown.pdf"
        "architecture_diagram.pdf"
    )

    # Generate placeholder PDFs if they don't exist
    for fig in "${figures[@]}"; do
        if [ ! -f "figures/$fig" ]; then
            echo "  Creating placeholder: figures/$fig"
            # Create a simple placeholder using PostScript if possible
            if command -v ps2pdf &> /dev/null; then
                cat > "figures/${fig%.pdf}.ps" << EOF
%!PS
/Times-Roman findfont 24 scalefont setfont
100 400 moveto
(Placeholder: $fig) show
100 370 moveto
/Times-Roman findfont 14 scalefont setfont
(Replace with actual figure) show
showpage
EOF
                ps2pdf "figures/${fig%.pdf}.ps" "figures/$fig"
                rm "figures/${fig%.pdf}.ps"
            else
                # Create empty PDF as fallback
                echo "  ⚠️  Cannot create placeholder PDF (ps2pdf not found)"
            fi
        else
            echo "  ✓ Found: figures/$fig"
        fi
    done
}

# Create article class version as fallback
create_article_version() {
    echo "Creating article class version of the paper..."
    # Convert IEEEtran to article class with twocolumn
    sed 's/\\documentclass\[10pt,conference\]{IEEEtran}/\\documentclass[10pt,twocolumn]{article}/' origami_paper.tex > origami_paper_article.tex
    sed -i '' 's/\\IEEEoverridecommandlockouts/%\\IEEEoverridecommandlockouts/' origami_paper_article.tex
    sed -i '' 's/\\begin{IEEEkeywords}/\\textbf{Keywords:}/' origami_paper_article.tex
    sed -i '' 's/\\end{IEEEkeywords}//' origami_paper_article.tex
    sed -i '' 's/\\IEEEauthorblockN/\\textbf/' origami_paper_article.tex
    sed -i '' 's/\\IEEEauthorblockA/\\\\\\small/' origami_paper_article.tex
    echo "✓ Created origami_paper_article.tex"
}

# Create bibliography file if it doesn't exist
create_bibliography() {
    echo ""
    echo "Checking bibliography..."

    if [ ! -f "references.bib" ]; then
        echo "  Creating references.bib..."
        cat > references.bib << 'EOF'
@article{jpeg2000,
    author = {Taubman, David and Marcellin, Michael},
    title = {JPEG2000: Standard for Interactive Imaging},
    journal = {Proceedings of the IEEE},
    year = {2002},
    volume = {90},
    number = {8},
    pages = {1336--1357}
}

@article{deepzoom2008,
    author = {Microsoft Corporation},
    title = {Deep Zoom Technology Overview},
    journal = {Microsoft Developer Network},
    year = {2008}
}

@article{openslide2013,
    author = {Goode, Adam and Gilbert, Benjamin and Harkes, Jan and Jukic, Drazen and Satyanarayanan, Mahadev},
    title = {OpenSlide: A vendor-neutral software foundation for digital pathology},
    journal = {Journal of Pathology Informatics},
    year = {2013},
    volume = {4},
    pages = {27}
}

@inproceedings{hevc2012,
    author = {Sullivan, Gary J. and Ohm, Jens-Rainer and Han, Woo-Jin and Wiegand, Thomas},
    title = {Overview of the High Efficiency Video Coding (HEVC) Standard},
    booktitle = {IEEE Transactions on Circuits and Systems for Video Technology},
    year = {2012},
    volume = {22},
    number = {12},
    pages = {1649--1668}
}

@article{pathology2020,
    author = {Zarella, Mark D. and Bowman, Douglas and Aeffner, Famke and others},
    title = {A Practical Guide to Whole Slide Imaging: A White Paper From the Digital Pathology Association},
    journal = {Archives of Pathology & Laboratory Medicine},
    year = {2019},
    volume = {143},
    number = {2},
    pages = {222--234}
}

@article{residual2016,
    author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
    title = {Deep Residual Learning for Image Recognition},
    journal = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2016},
    pages = {770--778}
}

@inproceedings{turbojpeg2010,
    author = {Independent JPEG Group},
    title = {libjpeg-turbo: A derivative of libjpeg that uses SIMD instructions},
    year = {2010},
    url = {https://libjpeg-turbo.org/}
}

@article{compression2021,
    author = {Chen, Zhengxue and others},
    title = {A Comprehensive Survey on Deep Learning-based Image Compression},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year = {2021}
}

@article{wsi2019,
    author = {Pantanowitz, Liron and others},
    title = {Review of the current state of whole slide imaging in pathology},
    journal = {Journal of Pathology Informatics},
    year = {2019},
    volume = {10},
    pages = {1}
}

@inproceedings{rocksdb2021,
    author = {Facebook},
    title = {RocksDB: A Persistent Key-Value Store for Fast Storage Environments},
    year = {2021},
    url = {https://rocksdb.org/}
}
EOF
        echo "  ✓ Created references.bib"
    else
        echo "  ✓ Found: references.bib"
    fi
}

# Build a single PDF file
build_single_pdf() {
    local TEX_FILE=$1
    local PDF_NAME="${TEX_FILE%.tex}"

    echo ""
    echo "Building ${PDF_NAME}..."
    echo "--------------------------------"

    # First pass - generate aux files
    echo "Pass 1: Generating auxiliary files..."
    $PDFLATEX -interaction=nonstopmode ${TEX_FILE} > ${PDF_NAME}_build.log 2>&1

    # Check if PDF was created (even with warnings)
    if grep -q "Output written on" ${PDF_NAME}_build.log; then
        echo "✓ Pass 1 complete"
    else
        echo "❌ ERROR in first pass for ${TEX_FILE}. Check ${PDF_NAME}_build.log for details"
        return 1
    fi

    # Run BibTeX if available and needed
    if [ -f "${PDF_NAME}.aux" ] && check_bibtex > /dev/null 2>&1; then
        echo "Processing bibliography..."
        $BIBTEX ${PDF_NAME} > ${PDF_NAME}_bibtex.log 2>&1 || {
            echo "⚠️  WARNING: BibTeX had issues. Check ${PDF_NAME}_bibtex.log"
        }
    fi

    # Second pass - incorporate references
    echo "Pass 2: Incorporating references..."
    $PDFLATEX -interaction=nonstopmode ${TEX_FILE} >> ${PDF_NAME}_build.log 2>&1

    # Third pass - finalize cross-references
    echo "Pass 3: Finalizing cross-references..."
    $PDFLATEX -interaction=nonstopmode ${TEX_FILE} >> ${PDF_NAME}_build.log 2>&1

    if [ -f "${PDF_NAME}.pdf" ]; then
        echo "✓ ${PDF_NAME}.pdf generated successfully"
        return 0
    else
        echo "❌ Failed to generate ${PDF_NAME}.pdf"
        return 1
    fi
}

# Build the PDFs (both versions)
build_pdf() {
    echo ""
    echo "Building both paper versions..."
    echo "================================"

    local BUILD_SUCCESS=0

    # Build IEEE version if IEEEtran is available
    if kpsewhich IEEEtran.cls > /dev/null 2>&1; then
        echo "📄 Building IEEE Conference Version..."
        if build_single_pdf "origami_paper.tex"; then
            echo "✓ IEEE version complete"
        else
            echo "⚠️  IEEE version failed (may have font issues)"
            BUILD_SUCCESS=1
        fi
    else
        echo "⚠️  Skipping IEEE version (IEEEtran.cls not found)"

        # Try to download IEEEtran if not present
        if [ ! -f "IEEEtran.cls" ]; then
            echo "Attempting to download IEEEtran.cls..."
            wget -q https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/IEEEtran.cls 2>/dev/null || \
            curl -s -o IEEEtran.cls https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/IEEEtran.cls 2>/dev/null || \
            echo "Could not download IEEEtran.cls"
        fi
    fi

    # Always build article version as fallback
    echo ""
    echo "📄 Building Article Version..."
    if [ -f "origami_paper_article.tex" ]; then
        if build_single_pdf "origami_paper_article.tex"; then
            echo "✓ Article version complete"
        else
            echo "❌ Article version failed"
            BUILD_SUCCESS=1
        fi
    else
        echo "❌ origami_paper_article.tex not found!"
        BUILD_SUCCESS=1
    fi

    return $BUILD_SUCCESS
}

# Check PDF output
check_output() {
    echo ""
    echo "Checking output..."
    echo "================================"

    local FOUND_PDF=0

    # Check IEEE version
    if [ -f "origami_paper.pdf" ]; then
        size=$(ls -lh origami_paper.pdf | awk '{print $5}')
        # Try to get page count if pdfinfo is available
        if command -v pdfinfo &> /dev/null; then
            pages=$(pdfinfo origami_paper.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
        else
            pages="?"
        fi
        echo "✓ IEEE Conference Version:"
        echo "  File: origami_paper.pdf"
        echo "  Size: $size"
        echo "  Pages: $pages"
        FOUND_PDF=1
    else
        echo "⚠️  IEEE version not found (origami_paper.pdf)"
    fi

    # Check Article version
    if [ -f "origami_paper_article.pdf" ]; then
        size=$(ls -lh origami_paper_article.pdf | awk '{print $5}')
        # Try to get page count if pdfinfo is available
        if command -v pdfinfo &> /dev/null; then
            pages=$(pdfinfo origami_paper_article.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
        else
            pages="?"
        fi
        echo "✓ Article Version:"
        echo "  File: origami_paper_article.pdf"
        echo "  Size: $size"
        echo "  Pages: $pages"
        FOUND_PDF=1
    else
        echo "⚠️  Article version not found (origami_paper_article.pdf)"
    fi

    if [ $FOUND_PDF -eq 1 ]; then
        echo ""
        echo "================================"
        echo "✅ Build Complete!"
        echo "================================"
        echo ""
        echo "📚 Both papers contain:"
        echo "  Title: ORIGAMI: Efficient Whole-Slide Image Serving Through"
        echo "         Optimized Residual Image Generation Across Multiscale Interpolation"
        echo "  Author: Andrew Luetgers (andrew.luetgers@gmail.com)"
        echo ""
        echo "View the papers with:"
        echo "  open origami_paper.pdf           # IEEE version (macOS)"
        echo "  open origami_paper_article.pdf   # Article version (macOS)"
        echo ""
        echo "  xdg-open origami_paper.pdf       # Linux"
        echo "  start origami_paper.pdf          # Windows"
    else
        echo ""
        echo "❌ ERROR: No PDF was generated"
        exit 1
    fi
}

# Main execution
main() {
    echo "Starting build process..."

    # Check prerequisites
    check_latex

    # Optional: clean previous build
    if [ "$1" == "--clean" ]; then
        clean_build
    fi

    # Generate missing assets
    generate_figures
    create_bibliography

    # Build the PDF
    build_pdf

    # Check the result
    check_output

    # Optional: clean intermediate files
    if [ "$1" == "--clean" ]; then
        echo ""
        echo "Cleaning intermediate files..."
        rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lot *.lof *.idx *.ind *.ilg
        echo "✓ Cleaned"
    fi
}

# Run the build
main "$@"