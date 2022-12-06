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

all: build/V44.pdf
	cp build/V44.pdf V44_Sedlaczek_Schoenfeld.pdf
#  | build/plot.pdf 

fast: FORCE | build
	# rm -f thesis.pdf
	lualatex --output-directory=build --interaction=nonstopmode --synctex=1 V44.tex

# hier Python-Skripte:
build/plot.pdf: plot.py latex/fix/header-matplotlib.tex | build
	TEXINPUTS="$(call translate,$(pwd):)" python plot.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/V44.pdf: build/plot.pdf

build/V44.pdf: build/plot.pdf FORCE | build
	  TEXINPUTS="$(call translate,$(pwd):)" \
	  BIBINPUTS="$(call translate,$(pwd):)" \
	  max_print_line=1048576 \
	  latexmk $(TeXOptions) V44.tex



build:
	mkdir -p build

clean:
	rm -rf build
	
re: clean all

FORCE:

.PHONY: all clean re

