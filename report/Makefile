PDFLATEX = $(HOME)/.TinyTeX/bin/x86_64-linux/pdflatex
BIBTEX   = $(HOME)/.TinyTeX/bin/x86_64-linux/bibtex
PYTHON   = $(HOME)/miniconda3/envs/rtdetr/bin/python

MAIN     = main
TEX_SRC  = $(MAIN).tex $(wildcard sections/*.tex) bibliography.bib
FIG_SRC  = scripts/generate_figures.py
FIGS     = figures/learning_curves.pdf figures/gate_evolution.pdf \
           figures/ablation_bars.pdf figures/gate_reparam.pdf \
           figures/full_metrics_v2.pdf

.PHONY: all figures clean view

all: $(MAIN).pdf

figures: $(FIGS)

$(FIGS): $(FIG_SRC)
	$(PYTHON) $(FIG_SRC)

$(MAIN).pdf: $(TEX_SRC) $(FIGS)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot \
	      *.fls *.fdb_latexmk sections/*.aux $(MAIN).pdf
