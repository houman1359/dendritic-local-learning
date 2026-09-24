# input_coverage_depth (panel A of N9) — SI text pass ledger (2026-09-23)

- Source key: N9, panel A only (`('N9','A')`, printed as panel D of the SI sheet)
- Builder: `scripts/build_supplementary_figure_input_coverage_mnist_native.py` (no arguments; it also writes `figure_input_coverage_mnist_native.png` and `source_data/curated_publication/si_mnist_coverage_plotted.csv`)
- Output: `figures/supplementary/figure_input_coverage_mnist_native.pdf`
- Reproducibility gate: PASS with the right interpreter.
  - `python3` (matplotlib 3.10.6) differed by one byte, the PDF Producer string "v3.10.6" against "v3.10.9". The original PDF and PNG were restored.
  - `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python` (matplotlib 3.10.9) reproduced the PDF (sha256 de123d55…a82a3ad), the PNG and the CSV byte for byte.
- Edited: **no**. The builder and all three outputs are byte-identical to their pre-pass state.

## Removed

Nothing. Panel A carries no title, footnote, statistics note, reference-line label or embedded count.

## Caption additions

None.

## Kept

- y label `Spatial − random accuracy (pp)`: already in sentence case.
- Tick labels `BP`, `MW scalar`, `Neuron`, `Exact path`: categories; the caption defines each abbreviation.

## Layout changes

None.

## Pre-existing issues (unresolved, reported)

1. Strict audit, 3 violations, all caused by the one-panel strip canvas the SI crops from:
   - `[canvas-aspect] aspect 3.60 outside 1.05-1.55`
   - `[fill-height] content fills 76.7%`
   - `[panel-aspect] panel 'A' is a letterbox strip … 450 x 88 pt, aspect 5.12`

   Clearing them would require a canvas at least 334 pt tall, with the panel about 280 pt tall. That is a redesign of the SI panel, not a text pass.
2. `letter_ink_audit.py`: "A: letter top only -2.6 pt above panel ink (topmost: text 'Spatial − random accuracy (pp)')". The y title is longer than the 88 pt axes. The consolidated sheet `figures/supplementary/curated/input_coverage_depth.pdf` already passes the letter audit through `letter_relocation.py`.
   - Tested in a temp copy only (`si_pass/tmp/n9_twoline_builder.py`, output `si_pass/tmp/n9_twoline.pdf`): a two-line y label `Spatial − random` / `accuracy (pp)` clears the letter audit, but the three strict violations remain.
   - I did not apply it, because an edited PDF must pass the strict audit.
