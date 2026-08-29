# Workshop talk bundle — Kempner Learning Dynamics (Sep 2–4, 2026)

Self-contained copy of everything used for the talk
"What dendritic structure adds to learning".

## Canonical workshop deck

The visually reviewed compact deck is the canonical workshop presentation.
The earlier expanded deck remains beside it as a recoverable technical source:

- `dendritic_credit_workshop_25.pdf` — 26 core slides with no act dividers,
  one Backup divider, and seven backup slides (34 pages total).
- `dendritic_credit_workshop_25.tex`, `_core.tex`, `_appendix.tex` — native
  16:9 source; Canvas images are design references only and are not embedded.
- `speaker_notes_workshop_25.md` — 29–30 minute narration and backup jump
  table.
- `workshop_slide_map_25.md` — 15/20/full-deck cuts and asset-provenance rules.
- `build_presentation_25.py` — compiles twice; verifies page count, 16:9
  geometry, renderability, and absence of Canvas dependencies.

Rebuild the canonical deck with `python3 build_presentation_25.py`.

The credit-conflict task on core slides 16–17 is rebuilt from the same source
data and schematic logic as Supplementary Fig. S29, but at projection-scale
type sizes and with \(\chi\) used consistently for conflict probability:

    python3 build_path_necessity_talk_assets.py
    python3 build_presentation_25.py

This writes `pdf_assets/path_necessity_task.pdf` and
`pdf_assets/path_necessity_results.pdf`. Both are vector PDFs with no embedded
raster images. The result establishes a benefit of branch-selective credit in
a controlled conflict regime; the matched gated-point implementation makes
clear that the required resource is an address, not dendritic material alone.

## Contents
- `dendritic_credit_workshop.pdf` — the earlier expanded deck (42 pages: 35 core slides,
  Backup divider, 6 backup slides for Q&A).
- `speaker_notes_workshop.md` — 30-minute narration; secondary statistics
  removed from the slides are woven in here in bold (say-only).
- `workshop_slide_map.md` — 15 / 30 / 45-minute cut maps.
- `dendritic_credit_workshop.tex` + `_core.tex` + `_appendix.tex` — deck source.
- `credit_tree_lib.tex` — the parameterized credit-tree schematic library
  (all modes: forward / eligibility / transport / scalar / coordinate /
  address K=2,4,8 / gain / shunt / deranged).
- `eq_style.tex` — hero-equation typography (`\heroeq`, `\term`).
- `credit_tree_lib_test.tex` / `.pdf` — visual regression sheet for the library.
- `pdf_assets/` — the 24 vector figure crops, data panels, and shared schematics the deck
  embeds (cut or copied from the
  journal manuscript's figures).

## Rebuild (self-contained)
    python3 build_pdf_assets.py          # if journal figures changed: regenerates pdf_assets/
    python3 build_presentation.py        # compiles twice + validates 42 pages, 16:9, clean log

`build_pdf_assets.py` holds this deck's 21 crop definitions (600-dpi pixel
boxes against the journal's canonical figure sheets); if a journal sheet is
regenerated with a new layout, re-measure the affected boxes before running it.
Plain `pdflatex dendritic_credit_workshop.tex` (twice) also works.

## Legacy decks
`legacy_decks/` holds the earlier iterations of this talk (expanded 37-slide,
story 22-slide, and math 22-slide variants, each with a with-appendix build,
plus the pptx exports) and their own build scripts (`build_pdf_presentation.py`,
`build_pdf_assets.py`). Moved here 2026-08-20 from
`drafts/presentation/credit_assignment/`, which no longer exists.

## Canonical home
This folder is the canonical home of the workshop talk as of 2026-08-19 (the
sources were MOVED here from `drafts/presentation/credit_assignment/`, whose
`build_pdf_presentation.py` no longer builds this deck). It is fully
self-contained: the rebuild command above needs nothing outside this folder.
`pdf_assets/` is regenerated locally by this folder's own
`build_pdf_assets.py`; nothing is needed from the legacy package.

A stale duplicate may exist on netscratch
(`/n/netscratch/kempner_dev/hsafaai/dendritic-local-learning-presentation/`,
auto-purged; created during a temporary quota block).
