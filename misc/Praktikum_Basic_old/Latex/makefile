versuch = V*.tex
bcf = build/V*.bcf

all: build/$(versuch) 
	
build/$(versuch): $(bcf)
	biber $(bcf)
	lualatex -output-directory=build $(versuch)

$(bcf):
	lualatex -output-directory=build $(versuch)
	
build:
	mkdir -p build
	
.PHONY: all