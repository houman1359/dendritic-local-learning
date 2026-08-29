# Workshop presentation bundle

## Canonical 15–20 minute workshop deck

The audience-facing workshop presentation is:

- `ml_workshop_15_20min/dendritic_credit_ml_workshop.pdf` — 20-slide core;
- `ml_workshop_15_20min/png/` — 2560 × 1440 projection PNGs;
- `ml_workshop_15_20min/workshop_deck.html` — editable HTML/SVG master;
- `ml_workshop_15_20min/speaker_notes.md` — 17:20 scripted narration;
- `ml_workshop_15_20min/slide_map.md` — 15-, 17-, and 20-minute cuts;
- `workshop_slide_production_plan.md` — exact scientific and visual
  specification for every slide.

Rebuild it from this directory with:

```bash
python ml_workshop_15_20min/build_slides.py
```

The builder regenerates presentation-scale result crops from the current
canonical manuscript assets, writes the HTML/SVG master, exports all PNGs,
builds the PDF and contact sheet, and validates dimensions, page count, and
16:9 geometry.

## Longer technical source

`dendritic_credit_workshop_25.pdf` is the longer 30-minute technical source
and backup deck. Its TeX sources, notes, and build scripts remain available:

- `dendritic_credit_workshop_25.tex`, `_core.tex`, `_appendix.tex`;
- `speaker_notes_workshop_25.md`;
- `workshop_slide_map_25.md`;
- `build_presentation_25.py`.

It is not the canonical 15–20 minute workshop deck.

## Shared resources

- `credit_tree_lib.tex` — reusable parameterized dendritic-tree schematics;
- `credit_tree_lib_test.tex` / `.pdf` — visual regression sheet;
- `eq_style.tex` — TeX equation styling;
- `pdf_assets/` — vector crops used by the longer TeX decks;
- `canvas_assets/` — composition references, never numerical sources;
- `legacy_decks/` — recoverable earlier talk variants.

All scientific claims are downstream of `journal/main.tex`. When manuscript
figures or numerical results change, rebuild the ML deck and inspect
`ml_workshop_15_20min/contact_sheet.png` before presenting.
