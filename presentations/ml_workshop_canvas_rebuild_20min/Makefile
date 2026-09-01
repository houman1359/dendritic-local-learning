SHELL := /bin/bash

PDF := dendritic_credit_canvas_rebuild.pdf
PNGS := $(shell seq -f 'png/slide_%02g.png' 1 25)

.PHONY: all pdf png contact guide qa clean

all: pdf png contact guide qa

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
	montage png/slide_*.png -thumbnail 512x288 -tile 5x5 -geometry +12+16 \
	  -background white contact_sheet.png

guide: png
	python3 build_speaker_guide.py

qa: guide png
	mkdir -p guide_qa qa
	rm -f guide_qa/page-*.png
	pdftoppm -png -r 96 dendritic_credit_canvas_rebuild_speaker_guide.pdf \
	  guide_qa/page >/dev/null 2>&1
	montage guide_qa/page-*.png -thumbnail 384x216 -tile 5x5 -geometry +8+10 \
	  -background white guide_contact_sheet.png
	convert guide_qa/page-01.png -resize 1600x1236 guide_qa/page_01_large.png
	convert guide_qa/page-20.png -resize 1600x1236 guide_qa/page_20_large.png
	@for n in 2 3 4 5 10 14 16 17 19 20; do \
	  printf -v src 'png/slide_%02d.png' $$n; \
	  cp "$$src" "qa/slide_$$n.png"; \
	done

clean:
	rm -f deck.aux deck.log deck.nav deck.out deck.snm deck.toc deck.vrb
	rm -f $(PDF) contact_sheet.png png/slide_*.png png/render-*.png
