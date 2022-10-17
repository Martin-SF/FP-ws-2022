versuch = V*.tex
bcf = build/V*.bcf
log = build/V*.log

all: build/$(versuch) 
	
build/$(versuch): ../*.pdf $(bcf)
	biber $(bcf)
	lualatex -output-directory=build $(versuch)
	
../*.pdf: ../plot_V*.py
	cd ..
	python plot_V*.py
	cd Latex
	
$(bcf):
	lualatex -output-directory=build $(versuch)
	
build:
	mkdir -p build
	
.PHONY: all