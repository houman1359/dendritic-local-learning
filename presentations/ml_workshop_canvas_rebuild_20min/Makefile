SHELL := /bin/bash

PDF := dendritic_credit_canvas_rebuild.pdf
PNGS := $(shell seq -f 'png/slide_%02g.png' 1 20)

.PHONY: all pdf png contact guide clean

all: pdf png contact guide

pdf: $(PDF)

$(PDF): deck.tex
	pdflatex -interaction=nonstopmode -halt-on-error deck.tex
	pdflatex -interaction=nonstopmode -halt-on-error deck.tex
	mv deck.pdf $(PDF)

png: $(PDF)
	mkdir -p png
	pdftoppm -png -r 192 $(PDF) png/render >/dev/null 2>&1
	@i=1; for f in png/render-*.png; do \
	  printf -v out 'png/slide_%02d.png' $$i; \
	  convert "$$f" -resize 2560x1440\! "$$out"; \
	  i=$$((i+1)); \
	done
	rm -f png/render-*.png

contact: png
	montage png/slide_*.png -thumbnail 512x288 -tile 4x5 -geometry +12+16 \
	  -background white contact_sheet.png

guide: png
	python3 build_speaker_guide.py

clean:
	rm -f deck.aux deck.log deck.nav deck.out deck.snm deck.toc deck.vrb
	rm -f $(PDF) contact_sheet.png png/slide_*.png png/render-*.png
