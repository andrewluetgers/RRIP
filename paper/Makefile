# Makefile for ORIGAMI Paper

PAPER = origami_paper
LATEX = pdflatex
BIBTEX = bibtex
RM = rm -f

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

# If using BibTeX (uncomment if you have .bib file)
# $(PAPER).pdf: $(PAPER).tex $(PAPER).bib
# 	$(LATEX) $(PAPER)
# 	$(BIBTEX) $(PAPER)
# 	$(LATEX) $(PAPER)
# 	$(LATEX) $(PAPER)

clean:
	$(RM) $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg
	$(RM) $(PAPER).toc $(PAPER).lof $(PAPER).lot
	$(RM) *.aux *.log *.out

view: $(PAPER).pdf
	open $(PAPER).pdf  # macOS
	# xdg-open $(PAPER).pdf  # Linux
	# start $(PAPER).pdf  # Windows

# Generate a version for arXiv submission
arxiv:
	mkdir -p arxiv_submission
	cp $(PAPER).tex arxiv_submission/
	cp *.png *.jpg *.pdf arxiv_submission/ 2>/dev/null || true
	cd arxiv_submission && tar czf ../origami_arxiv.tar.gz *
	@echo "ArXiv submission package created: origami_arxiv.tar.gz"

help:
	@echo "Available targets:"
	@echo "  make          - Build the PDF"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make view     - Build and open the PDF"
	@echo "  make arxiv    - Create arXiv submission package"