PDFLATEX = $(HOME)/.TinyTeX/bin/x86_64-linux/pdflatex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex
	$(PDFLATEX) -interaction=nonstopmode main.tex
	$(PDFLATEX) -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb main.pdf
