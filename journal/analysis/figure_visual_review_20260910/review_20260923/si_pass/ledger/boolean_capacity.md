# boolean_capacity (N14) — SI text pass, group B

- Source key: N14
- Builder: `scripts/build_supplementary_figure_boolean_capacity_native.py` (no argparse; run as `python3 scripts/build_supplementary_figure_boolean_capacity_native.py`)
- Output: `figures/supplementary/figure_boolean_capacity_native.pdf`
- Reproducibility gate: PASS. The unchanged builder rebuilt the output byte-identically (sha256 `613407dd274ba81bc77342e06f419c66bca3311577b2bf372a464d1d5805048e` for both). Original saved as `si_pass/renders/boolean_capacity_orig.pdf`; renders `boolean_capacity_before.png` / `boolean_capacity_after.png`.
- New PDF: strict audit 0 violations; `letter_ink_audit.py` total problems 0; render inspected. New sha256 prefix `bc3054d28860a99d`.
- Data unchanged: the builder's data printouts ([A]–[F]) are identical to the gate run, and the vector-path census is identical to the original (400 paths, no differing kinds).

## Removed

| # | Panel | String (verbatim) | Status |
|---|---|---|---|
| 1 | A | `Seven exact Boolean truth tables` (title) | caption already states it: 'Truth tables of seven four-input templates on the sixteen equally weighted patterns' |
| 2 | B | `All trees: regression obstruction` (title) | caption already states it: 'regression lower bounds for all fifteen labeled binary trees' |
| 3 | C | `Equal resources; different minimum depth` (title) | caption already states it: 'OR-of-AND and nested each use two AND gates and one OR gate' … 'but need exact depths two and three' |
| 4 | D | `Associative gates are structure controls` (title) | ADD TO CAPTION. The counts are in the caption ('15/15 and 3/3 for AND, OR and parity'), but the interpretation is not. |
| 5 | E | `Post hoc: target information in a pair` (title) | caption already states it: 'Post hoc target-projection energy' … 'on the six two-input subsets' and '\textbf{E} is local target information' |
| 6 | E | `dot area = energy` (key line above E) | caption already states it: 'dot area is proportional to the energy' |
| 7 | F | `Canonical conditional credit` (title) | caption already states it: 'Canonical derivatives \(\partial F/\partial u\)'; caption title: 'Boolean interactions distinguish structure, depth and canonical credit.' |

## Caption additions

1. **D sentence**: after `1/15 and 0/3 for nested.`, append:
   ` The associative-gate targets (AND, OR, parity) are structure controls.`

With this addition the caption grows from 368 to about 378 words. It was already the longest of the four.

## Kept

- `1` / `0` swatch key above A: a compact symbol key (the caption also gives 'white, 0; slate, 1').
- Family row labels `AND`, `OR`, `parity`, `OR(AND)`, `XOR(AND)`, `AND(XOR)`, `nested` (A, B, D, E): truth-table/family labels.
- `balanced` / `comb` (B) and `aligned` / `crossed` (E): one-word column-group labels.
- `max` with the row maxima `0`, `0.294`, `0.533`, `0.667`, `0.199` (B): per-row values the caption names ('row maxima are printed at right').
- `exact` with its ring (B, beside the colour bar): compact marker key.
- `NMSE lower bound`: colour-bar label.
- C: tree names `OR(AND)` / `nested`, gate labels `AND` / `OR`, leaves `a`–`d`: schematic content.
- C: `minimum depth 2` / `minimum depth 3`: the one result annotation per tree of the schematic. C has no depth axis, so without them the reader must count levels. Short, no verb; the caption says the same ('but need exact depths two and three').
- D: headers `exact` / `balanced` and the counts `15/15` … `0/3`: compact per-row counts with one-word headers.
- E: the values printed beside the dots: data labels ('values printed beside the dots').
- F: `AND`, `OR`, `XOR`: direct series labels.
- Axis labels (now sentence case) and `∂F/∂u`.

## Layout changes

- Removed all six panel titles, together with the title pads `TITLE_PAD` (16 pt), `TITLE_PAD_KEYED` (25 pt) and `BAND_KEY_PT`. The band above each row now holds only the group labels and keys (about 12.7 pt).
- Gave the freed title bands to the rows: `VGUTTER_PT` 64.5 → 54.0, `MARGINS.top` 32 → 23, `MARGINS.bottom` 32 → 30. The canvas is unchanged (518.4 × 493 pt). Every row's axes box grows from 100 to 110.7 pt tall; widths, dot sizes (pt) and E's column layout are unchanged. No row gets a top or bottom lock reserve, so the three rows keep one axes height.
- Letters: row 0 at 4.8 pt (unchanged), row 1 at 171.4 pt (was 169.3 pt), row 2 at 336.1 pt (was 324.8 pt). Each letter row is 16–18 pt above its axes and its band furniture, as before.
- Axis labels changed to sentence case: `Input pattern abcd`, `Labeled binary tree, T index`, `Minimum exact depth`, `Projection energy`, `Input pair`, `Other branch output v`.
- Updated the module docstring and removed the unused `PT_TITLE` import.
