#!/bin/bash

# Script to install missing LaTeX packages for the ORIGAMI paper

echo "=========================================="
echo "LaTeX Package Installation Script"
echo "=========================================="
echo ""
echo "This script will install the required LaTeX packages."
echo "You may need to enter your password for sudo access."
echo ""

# Check if tlmgr exists
if [ ! -f "/Library/TeX/texbin/tlmgr" ]; then
    echo "❌ ERROR: tlmgr not found at /Library/TeX/texbin/tlmgr"
    echo "Please ensure BasicTeX or MacTeX is installed:"
    echo "  brew install --cask basictex"
    exit 1
fi

TLMGR="/Library/TeX/texbin/tlmgr"

echo "Found tlmgr at: $TLMGR"
echo ""
echo "Updating tlmgr itself..."
sudo $TLMGR update --self

echo ""
echo "Installing required packages..."
echo "--------------------------------"

# List of packages to install
PACKAGES=(
    "IEEEtran"       # IEEE transaction format
    "algorithms"     # Algorithm package (includes algorithmic)
    "algorithmicx"   # Extended algorithmic
    "booktabs"       # Better tables
    "multirow"       # Table multirow support
    "subcaption"     # Subfigures
    "listings"       # Code listings
    "xcolor"         # Extended colors
    "hyperref"       # Hyperlinks
    "graphics"       # Graphics support
    "graphicx"       # Extended graphics
    "float"          # Float positioning
    "caption"        # Caption customization
)

# Install each package
for package in "${PACKAGES[@]}"; do
    echo -n "Installing $package... "
    if sudo $TLMGR install $package 2>/dev/null; then
        echo "✓"
    else
        echo "⚠️  (may already be installed or part of another package)"
    fi
done

echo ""
echo "Checking installed packages..."
echo "--------------------------------"

# Check which packages are now available
for package in IEEEtran algorithmic algorithms booktabs multirow subcaption; do
    if kpsewhich ${package}.sty > /dev/null 2>&1 || kpsewhich ${package}.cls > /dev/null 2>&1; then
        echo "✓ $package is available"
    else
        echo "✗ $package is NOT available"
    fi
done

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "You can now run the build script:"
echo "  ./build_paper.sh"
echo ""
echo "If you still get errors, you may need to install additional packages with:"
echo "  sudo $TLMGR install <package_name>"