SOURCE = $(wildcard *.tex)

tutorial.pdf: tutorial.ps
	ps2pdf -dAutoRotatePages=/None  tutorial.ps tutorial.pdf

tutorial.ps: tutorial.dvi
	dvips -z tutorial.dvi

tutorial.dvi: $(SOURCE)	tutorial.aux
	latex tutorial.tex 

tutorial.aux: $(SOURCE)
	latex tutorial.tex

all: tutorial.pdf tutorial

# Note. Text in tables may not work properly - have to 
# insert that by hand
tutorial: $(SOURCE)
	mkdir -p tutorial
	latex2html -local_icons -dir tutorial tutorial 

clean:
	rm -f *.aux *.dvi *.log *.out *.pdf *.ps *.tmp *.toc *~
	rm -rf tutorial

.PHONY: all clean
