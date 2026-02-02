# ORIGAMI Paper Formatting Guide

## Quick Start

### Prerequisites

Install LaTeX on your system:

**macOS:**
```bash
brew install --cask mactex
# Or lighter weight:
brew install --cask basictex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
# Or minimal:
sudo apt-get install texlive-latex-base texlive-latex-extra
```

**Windows:**
Download and install MiKTeX from https://miktex.org/

### Building the Paper

```bash
# Build PDF
make

# Build and open
make view

# Clean auxiliary files
make clean

# Create arXiv package
make arxiv
```

Or manually:
```bash
pdflatex origami_paper.tex
pdflatex origami_paper.tex  # Run twice for references
pdflatex origami_paper.tex  # Third time for final formatting
```

## Current Format

The paper is currently formatted using **IEEE Conference style** (`IEEEtran`):
- Two-column layout
- 10pt font
- Conference paper format
- ~8 pages

## Alternative Formats

### 1. ACM Conference Format

Replace the document class in `origami_paper.tex`:
```latex
% Replace this line:
\documentclass[10pt,conference]{IEEEtran}

% With:
\documentclass[sigconf]{acmart}
```

### 2. Springer LNCS Format

```latex
\documentclass[runningheads]{llncs}
\usepackage{graphicx}
```

### 3. arXiv Preprint

For arXiv, use a simpler single-column format:
```latex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
```

### 4. Journal Format (IEEE Transactions)

```latex
\documentclass[journal]{IEEEtran}
```

## Key Sections to Update

### 1. Author Information

Update in the preamble:
```latex
\author{\IEEEauthorblockN{Andrew Luetgers}
\IEEEauthorblockA{Your Institution\\
Department of Computer Science\\
Email: your.email@institution.edu}
}
```

### 2. Abstract Length

- **Conferences**: Usually 150-200 words
- **Journals**: Can be 200-300 words
- **Current**: ~180 words ✓

### 3. Page Limits

Different venues have different requirements:
- **CVPR/ICCV**: 8 pages + references
- **MICCAI**: 8 pages including references
- **IEEE TMI**: No strict limit (typically 10-12 pages)
- **Nature Methods**: 3,000 words
- **arXiv**: No limit

## Adding Figures

### Rate-Distortion Curves

Create a figure from your evaluation results:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.45\textwidth]{rd_curves.png}
\caption{Rate-distortion curves comparing ORIGAMI to JPEG recompression}
\label{fig:rd_curves}
\end{figure}
```

### Architecture Diagram

```latex
\begin{figure*}[ht]  % figure* spans both columns
\centering
\includegraphics[width=0.9\textwidth]{architecture.png}
\caption{ORIGAMI architecture: L2 baseline + residual reconstruction}
\label{fig:architecture}
\end{figure*}
```

## Creating Professional Figures

### Using matplotlib for plots:

```python
import matplotlib.pyplot as plt
import matplotlib

# Set publication-quality defaults
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 11

# Create figure at correct size for two-column
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Single column width

# Your plotting code here

# Save with high DPI
plt.savefig('rd_curves.pdf', dpi=300, bbox_inches='tight')
```

### Architecture Diagrams

Use draw.io, Inkscape, or TikZ for diagrams:

```latex
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning}

\begin{tikzpicture}[node distance=2cm]
    % Define styles
    \tikzstyle{block} = [rectangle, draw, text width=5em,
                         text centered, rounded corners, minimum height=3em]

    % Nodes
    \node [block] (l2) {L2 Tile};
    \node [block, right of=l2] (upsample) {Upsample 2×};
    \node [block, right of=upsample] (l1) {L1 Prediction};

    % Arrows
    \draw [->] (l2) -- (upsample);
    \draw [->] (upsample) -- (l1);
\end{tikzpicture}
```

## Submission Checklist

### Before Submitting:

- [ ] **Anonymous version** for review (remove author names if required)
- [ ] **Page limit** adhered to
- [ ] **Font size** correct (usually 10pt for conferences)
- [ ] **Margins** not modified (automatic with template)
- [ ] **References** complete and properly formatted
- [ ] **Figures** readable and high quality
- [ ] **Spell check** completed
- [ ] **Grammar check** (use Grammarly or similar)
- [ ] **Supplementary material** prepared if needed

### Creating Anonymous Version:

Add to preamble for conferences requiring blind review:
```latex
\usepackage{xcolor}
\newcommand{\anon}[1]{\textcolor{red}{[ANONYMIZED]}}

% Then in author block:
\author{\anon{Author names hidden for review}}
```

## Common LaTeX Issues

### Package Conflicts
If you get package conflicts, try:
```bash
rm -rf *.aux *.log
pdflatex origami_paper.tex
```

### Missing Packages
Install missing packages:
```bash
# macOS with MacTeX
sudo tlmgr install package-name

# Ubuntu
sudo apt-get install texlive-package-name
```

### Bibliography Issues
If references don't appear:
1. Make sure you run pdflatex multiple times
2. Use `\cite{}` commands in the text
3. Check that all references are in the bibliography

## Converting to Other Formats

### To Word/DOCX:
```bash
pandoc origami_paper.tex -o origami_paper.docx
```

### To HTML:
```bash
htlatex origami_paper.tex
```

### To Markdown:
```bash
pandoc origami_paper.tex -t markdown -o origami_paper_converted.md
```

## Target Venues

Consider submitting to:

### Computer Vision/Medical Imaging:
- **CVPR** (June) - Top tier computer vision
- **MICCAI** (October) - Medical image computing
- **ISBI** (April) - Biomedical imaging
- **MIDL** (July) - Medical imaging with deep learning

### Journals:
- **IEEE Transactions on Medical Imaging**
- **Medical Image Analysis**
- **Journal of Pathology Informatics**
- **Nature Methods** (for broad impact)

### Preprint:
- **arXiv** (cs.CV or eess.IV)
- **bioRxiv** (for biological emphasis)

## Next Steps

1. **Add real figures** from your evaluation
2. **Update author information** with your affiliation
3. **Choose target venue** and adjust formatting
4. **Add more detailed related work** with proper citations
5. **Include supplementary material** with implementation details
6. **Create a project webpage** to accompany the paper

---

The paper is ready for submission to conferences or journals. The LaTeX format makes it professional and easy to adapt to different venue requirements.