# stelle sicher das pwd (PrintWorkingDirectory) auch das aktuelle Verzeichnis beinhaltet
ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -synctex=1 \
			 -output-directory=build

all: build/main.pdf

only_latex: clean FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) main.tex
	cp build/main.pdf VXX.pdf

only_latex_fast: clean FORCE | build
	lualatex --output-directory=build --interaction=nonstopmode --synctex=1 main.tex


# hier Python-Skripte:
build/plot.pdf: plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=.: python plot.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/plot.pdf

build/main.pdf: FORCE | build
	max_print_line=1048576 \
	latexmk $(TeXOptions) main.tex
	cp build/main.pdf VXX.pdf


build:
	mkdir -p build

clean:
	rm -rf build

re: clean all

FORCE:

.PHONY: all clean re

	# TEXINPUTS="$(call translate,$(pwd):)" python plot.py
	# TEXINPUTS=.: \
	# BIBINPUTS=./latex: \