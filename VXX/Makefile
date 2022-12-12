TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -synctex=1 \
			 -output-directory=build

versuchsbezeichung = VXX

all: build/$(versuchsbezeichung).pdf

# Python:
build/plot.pdf: plot.py matplotlibrc header-matplotlib.tex | build 
	TEXINPUTS=.: python plot.py

# hier weitere Abhängigkeiten für build/$(versuchsbezeichung).pdf deklarieren:
build/$(versuchsbezeichung).pdf: build/plot.pdf

build/$(versuchsbezeichung).pdf: FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) $(versuchsbezeichung).tex
	cp build/$(versuchsbezeichung).pdf $(versuchsbezeichung).pdf

only_latex: FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) $(versuchsbezeichung).tex
	cp build/$(versuchsbezeichung).pdf $(versuchsbezeichung).pdf

only_latex_fast: FORCE | build
	lualatex --output-directory=build --interaction=nonstopmode --synctex=1 $(versuchsbezeichung).tex

build:
	mkdir -p build

clean:
	rm -rf build

redo: clean all

FORCE:

.PHONY: all clean re
