# physical_architecture — SI clarity pass 2026-09-23

- Source key: N22 (whole sheet)
- Builder: `scripts/build_supplementary_figure_physical_architecture_native.py`
- Output: `figures/supplementary/figure_physical_architecture_native.pdf`
- Reproducibility gate: PASS with the repository interpreter
  `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python`
  (matplotlib 3.10.9, the producer recorded in the frozen PDF): the unchanged
  builder rewrote the output byte-identically (sha256 cd4e1f0183cebd004218cd6e12591845ceebf424a8ecbc12c3dc1c1c90239844).
  - The system `python3` (matplotlib 3.10.6) gave different bytes (98a21914…).
    The only difference was the PDF Producer string. Text, words, 714 paths and the
    150-dpi raster were identical. That output was restored at once.
  - The final output was built with the `.venv` interpreter.
  - Saved original: `si_pass/renders/physical_architecture_orig.pdf`. Before render:
    `si_pass/renders/physical_architecture_before.png`.
- New output sha256: 911b7bff06b3a537639dc4137d7da72d74406852ff7ff62f4e64f12058c9e326
  (518.4 x 475.0 pt, was 518.4 x 484.0 pt).
  - The registry entry N22 must be re-registered.
  - The HEAD `original_assets.json` already recorded sha e7b8f866… and eight letters
    A–H for N22. That matches neither the pre-pass sheet nor this one. Both have
    seven panels, A–G.
- Checks:
  - Strict canvas audit: 0 violations.
  - `letter_ink_audit.py` on the native PDF: 0 problems. Before the pass it reported
    "E: letter top only -9.8 pt above panel ink (topmost: text 'p')": the subscript
    of B's depth label sat nearer E's axes than B's.
  - Offline replay of the whole-sheet paste (`si_pass/sim_reflow.py`): 0 relocations,
    0 problems. Before, the replay needed B and G moved -15.6 pt, because the foot
    key's ink sat in G's strip at x = 239.4 pt.
- Marks: 714 → 708 paths. The 6 paths removed are all foot-key samples: the seed dot,
  the filled mean with 2 caps and a bar, and the open mean. Every data path is kept.

## Removed

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `Architectures compared` (title) | caption already states it: 'Serial D3 tree ($[2,1,2]$, eight modules in three stages), resource-identical grouped point model (one stage), and flexible point multilayer perceptron (MLP).' |
| B | `Four-tier task` (title) | caption already states it: 'Four-tier task under aligned sensors or reversed tier placement.' |
| C | `Exact path at D3` (title) | caption already states it: 'Serial-minus-grouped-point accuracy at D3 under exact-path LocalCA' |
| D | `Serial vs star, BP` (title) | caption already states it: 'Serial-minus-star accuracy at D1--D3 under BP.' |
| E | `Flexible point controls` (title) | caption already states it: 'Flexible point networks matched to active or total parameter count (grey), compared with serial D3 exact BP (black).' |
| F | `Alignment dose, serial BP` (title) | caption already states it: 'Serial-BP accuracy against task--sensor alignment $\alpha$' |
| G | `Depth benefit versus alignment` (title) | caption already states it: 'Paired D3-minus-D1 advantage at each alignment level' |
| foot key | `one seed (10 per condition)` (with seed dot) | caption already states it: 'Small dots show ten seeds per condition' |
| foot key | `mean and 95 % seed-bootstrap interval` (with mean-and-whisker sample) | caption already states it: 'large symbols and bars are means and 95\% seed-bootstrap intervals, paired for differences' |
| foot key | `derived difference of two rows (D)` (with open marker) | caption already states it: 'the open symbol shows the aligned-minus-reversed D3 difference' |

## Caption additions

None. Every removed string is already in the caption.

## Kept

- `same 8 modules and contacts` (A): the schematic's bracket label. It says which two models share
  resources. Five words, no verb.
- `serial` / `tree` / `3 stages`, `grouped` / `point` / `1 stage`, `point` / `MLP` / `matched` / `params` (A):
  schematic cell labels.
- `excitatory contact ×8`, `module` (A): compact glyph key.
- `aligned sensors`, `reversed tier placement` (B): block labels. They separate the two otherwise
  identical halves of the grid.
- `not run` (B): marks the empty raw-additive reversed cells, so the blank row does not look like missing
  data. The caption repeats it ('The raw-additive reversed arm was not run.').
- Row labels `serial BP`, `exact path`, `shared soma`, `grouped point`, `raw additive` (B), and the series key
  `nested factors`, `flat factors`, `local ratios` (C): direct series labels and a compact key.
- `D3, aligned − reversed` (D row label), `active match`, `total match`, `serial D3` (E tick labels),
  `D1`/`D2`/`D3` (F direct labels), `α = 0` … `α = 1` (G row labels): tick and direct labels.
- Axis labels, now in sentence case: `Serial physical depth D`+subscript `p` (B), `Test accuracy (%)`
  (B colour rail, E, F), `Serial − grouped point (pp)` and `Task–sensor alignment α` (C, F),
  `Serial − all-active star (pp)` (D), `D3 − D1 accuracy (pp)` (G, unchanged).

## Layout changes

- All seven panel titles are removed, and so is the foot key. Letters now sit just above the axes.
  Rows A/B, C/D/E and F/G each share a baseline. A/C/F share a column, as do B/G.
- B's depth label now drops 11.5 pt below the grid, not 14 pt (`depth_axis_label(ax, drop_pt=11.5)`).
  So the label and its subscript stay nearer B than E's axes top. This fixes the pre-existing native
  letter-audit failure of E.
- Canvas 484 → 475 pt tall (aspect 1.07 → 1.09). Bottom margin 44 → 35 pt; below 35 pt a bottom lock
  would shorten F/G and trip the 1.35x panel-emphasis rule. Row weights, gutters and every axes box are
  unchanged: A/B 178.2 x 135, C–E 96.8 x 98, F/G 178.2 x 95.7 pt. The schematic in A is redrawn at
  the same frame.
