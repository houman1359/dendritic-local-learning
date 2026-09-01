# ML workshop deck — 20 minutes

This is the projection-ready PNG deck for the workshop talk **“When dendritic
structure helps local credit assignment.”** It is written for a technical ML
audience that does not already know dendritic biophysics.

## Deliverables

- `png/slide_01.png` through `png/slide_20.png` — 2560 × 1440 sRGB slides.
- `dendritic_credit_ml_workshop.pdf` — the same 20 slides in one 16:9 PDF.
- `workshop_deck.html` — editable HTML/SVG master; open with
  `?slide=1`, `?slide=2`, and so forth.
- `contact_sheet.png` — one-page visual inventory.
- `speaker_notes.md` — timed narration, transitions, and claim qualifications.
- `dendritic_credit_ml_workshop_speaker_guide.pdf` — one landscape page per
  slide, with the slide thumbnail, exact script, visual reading, likely
  question and answer, claim guardrail, and transition.
- `speaker_guide.html` — editable master for the speaker-guide PDF.
- `slide_map.md` — 15-, 17-, and 20-minute cuts.
- `../workshop_slide_production_plan.md` — exact visible content, equations,
  interpretation, and transition for every slide.
- `build_slides.py` — deterministic asset and export pipeline.
- `build_speaker_guide.py` — deterministic notes and speaker-guide pipeline.

## Build

From the repository root:

```bash
python presentations/ml_workshop_15_20min/build_slides.py
python presentations/ml_workshop_15_20min/build_speaker_guide.py
```

The builder regenerates all result panels from the canonical vector assets,
creates the editable HTML/SVG master, exports the PNGs, builds the PDF and
contact sheet, and validates dimensions, page count, renderability, and 16:9
geometry.

## Production rules

- Native SVG supplies every conceptual schematic.
- Numerical panels come from frozen manuscript assets; Canvas images are
  visual references only and are never scientific sources.
- Semantic colors remain fixed: blue = neuron coordinate/additive, purple =
  subtree address, teal/green = anatomy or aligned route, orange = scalar or
  shared feedback, red = conflict/shunt, gray = matched controls.
- The bottom ribbon carries one interpretation; secondary statistics belong
  in the notes.
- The core deck uses `K` for within-neuron feedback bandwidth, `χ` for branch
  conflict, `\widetilde α_n` for directed-tree path gain, `a` for imposed
  task–route alignment, and `A_u c_u` for the available compartment field.
  Forward physical depth is omitted from the core and specified as an optional
  backup module because it asks a separate computation question.
- “Exact compartment field” is the scientific condition name. “Exact path”
  appears only where it is the frozen implementation label on a source plot.
- The focal-shunting control matches baseline first-order focal current and
  restores somatic voltage; it does not match local dendritic voltage.

The deck is scientifically downstream of `journal/main.tex`. When a result or
panel changes, rebuild this folder and inspect `contact_sheet.png` before use.
