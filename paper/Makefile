.PHONY: figures main.pdf clean help

PYTHON      ?= uv run python
FIGURE_DIR  := figures
OUT_DIR     := build
LATEX       ?= pdflatex
BIBTEX      ?= bibtex

# Auto-discovered figure scripts: every figures/figure_*.py produces a PNG of
# the same stem in `build/`.
FIGURE_SOURCES := $(wildcard $(FIGURE_DIR)/figure_*.py)
FIGURE_OUTPUTS := $(patsubst $(FIGURE_DIR)/%.py,$(OUT_DIR)/%.png,$(FIGURE_SOURCES))

help:
	@echo "Targets:"
	@echo "  figures   — regenerate every figure from its script (paper/figures/figure_*.py → build/*.png)"
	@echo "  main.pdf  — compile the paper (requires pdflatex + bibtex on PATH)"
	@echo "  clean     — remove build/ artefacts"

figures: $(FIGURE_OUTPUTS)

$(OUT_DIR)/%.png: $(FIGURE_DIR)/%.py | $(OUT_DIR)
	$(PYTHON) $< --output $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

main.pdf: figures main.tex refs.bib
	$(LATEX) -output-directory $(OUT_DIR) main.tex
	cd $(OUT_DIR) && $(BIBTEX) main || true
	$(LATEX) -output-directory $(OUT_DIR) main.tex
	$(LATEX) -output-directory $(OUT_DIR) main.tex
	cp $(OUT_DIR)/main.pdf .

clean:
	rm -rf $(OUT_DIR)
