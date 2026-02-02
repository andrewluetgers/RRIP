#!/bin/bash

# RRIP Paper Build Script
# Generates the complete PDF with figures, references, and citations

set -e  # Exit on error

echo "================================"
echo "RRIP Paper Build Script"
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
    sed 's/\\documentclass\[10pt,conference\]{IEEEtran}/\\documentclass[10pt,twocolumn]{article}/' rrip_paper.tex > rrip_paper_article.tex
    sed -i '' 's/\\IEEEoverridecommandlockouts/%\\IEEEoverridecommandlockouts/' rrip_paper_article.tex
    sed -i '' 's/\\begin{IEEEkeywords}/\\textbf{Keywords:}/' rrip_paper_article.tex
    sed -i '' 's/\\end{IEEEkeywords}//' rrip_paper_article.tex
    sed -i '' 's/\\IEEEauthorblockN/\\textbf/' rrip_paper_article.tex
    sed -i '' 's/\\IEEEauthorblockA/\\\\\\small/' rrip_paper_article.tex
    echo "✓ Created rrip_paper_article.tex"
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

# Build the PDF
build_pdf() {
    echo ""
    echo "Building PDF..."
    echo "--------------------------------"

    # Check if IEEEtran class is available
    if ! kpsewhich IEEEtran.cls > /dev/null 2>&1; then
        echo "⚠️  WARNING: IEEEtran.cls not found."
        echo "Installing IEEEtran package..."

        # Try to install using tlmgr if available
        if command -v tlmgr &> /dev/null || [ -f "/Library/TeX/texbin/tlmgr" ]; then
            TLMGR=${TLMGR:-$(command -v tlmgr || echo "/Library/TeX/texbin/tlmgr")}
            echo "Using tlmgr to install IEEEtran..."
            sudo $TLMGR update --self 2>/dev/null || true
            sudo $TLMGR install IEEEtran 2>/dev/null || {
                echo "Could not install IEEEtran automatically."
                echo "Creating fallback article version..."
                create_article_version
                PAPER_FILE="rrip_paper_article.tex"
            }
        else
            echo "tlmgr not found. Creating fallback article version..."
            create_article_version
            PAPER_FILE="rrip_paper_article.tex"
        fi
    else
        PAPER_FILE="rrip_paper.tex"
    fi

    # Set base name for all subsequent operations
    BASE_NAME="${PAPER_FILE%.tex}"

    # First pass - generate aux files
    echo "Pass 1: Generating auxiliary files..."
    $PDFLATEX -interaction=nonstopmode ${PAPER_FILE} > build.log 2>&1
    # Check if PDF was actually created (pdflatex can have warnings but still succeed)
    if [ ! -f "${BASE_NAME}.pdf" ] && ! grep -q "Output written on" build.log; then
        echo "❌ ERROR in first pass. Check build.log for details"
        tail -20 build.log
        exit 1
    fi
    echo "✓ Pass 1 complete"

    # Run BibTeX if available and needed
    if [ -f "${BASE_NAME}.aux" ] && check_bibtex > /dev/null 2>&1; then
        echo "Processing bibliography..."
        $BIBTEX ${BASE_NAME} > bibtex.log 2>&1 || {
            echo "⚠️  WARNING: BibTeX had issues. Check bibtex.log"
        }
        echo "✓ Bibliography processed"
    fi

    # Second pass - incorporate references
    echo "Pass 2: Incorporating references..."
    $PDFLATEX -interaction=nonstopmode ${PAPER_FILE} >> build.log 2>&1
    if ! grep -q "Output written on" build.log; then
        echo "❌ ERROR in second pass. Check build.log for details"
        tail -20 build.log
        exit 1
    fi
    echo "✓ Pass 2 complete"

    # Third pass - finalize cross-references
    echo "Pass 3: Finalizing cross-references..."
    $PDFLATEX -interaction=nonstopmode ${PAPER_FILE} >> build.log 2>&1
    if ! grep -q "Output written on" build.log; then
        echo "❌ ERROR in third pass. Check build.log for details"
        tail -20 build.log
        exit 1
    fi
    echo "✓ Pass 3 complete"

    # If we used the article version, rename the output
    if [ "${PAPER_FILE}" = "rrip_paper_article.tex" ]; then
        mv -f rrip_paper_article.pdf rrip_paper.pdf 2>/dev/null || true
    fi
}

# Check PDF output
check_output() {
    echo ""
    echo "Checking output..."
    if [ -f "rrip_paper.pdf" ]; then
        size=$(ls -lh rrip_paper.pdf | awk '{print $5}')
        # Try to get page count if pdfinfo is available
        if command -v pdfinfo &> /dev/null; then
            pages=$(pdfinfo rrip_paper.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
        elif [ -f "/Library/TeX/texbin/pdfinfo" ]; then
            pages=$(/Library/TeX/texbin/pdfinfo rrip_paper.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
        else
            pages="(pdfinfo not available)"
        fi
        echo "✓ PDF generated successfully!"
        echo "  File: rrip_paper.pdf"
        echo "  Size: $size"
        echo "  Pages: $pages"
        echo ""
        echo "================================"
        echo "✅ Build Complete!"
        echo "================================"
        echo ""
        echo "View the paper with:"
        echo "  open rrip_paper.pdf    # macOS"
        echo "  xdg-open rrip_paper.pdf # Linux"
        echo "  start rrip_paper.pdf    # Windows"
    else
        echo "❌ ERROR: PDF was not generated"
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