TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -synctex=1 \
			 -output-directory=build

filename = VXX

all: build/$(filename).pdf

# Python:
build/plot.pdf: plot.py matplotlibrc header-matplotlib.tex | build 
	TEXINPUTS=.: python plot.py

# hier weitere Abhängigkeiten für build/$(filename).pdf deklarieren:
build/$(filename).pdf: build/plot.pdf

build/$(filename).pdf: FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) $(filename).tex
	cp build/$(filename).pdf $(filename).pdf

only_latex: FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) $(filename).tex
	cp build/$(filename).pdf $(filename).pdf

only_latex_fast: FORCE | build
	lualatex --output-directory=build --interaction=nonstopmode --synctex=1 $(filename).tex

build:
	mkdir -p build

clean:
	rm -rf build

redo: clean all

FORCE:

.PHONY: all clean re