# local_gate_controls — SI clarity pass 2026-09-23

- Source key: N21 (whole sheet)
- Builder: `scripts/build_supplementary_figure_local_gate_controls_native.py`
- Output: `figures/supplementary/figure_local_gate_controls_native.pdf`
- Reproducibility gate: PASS, with the repository interpreter
  `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python`
  (matplotlib 3.10.9, the producer recorded in the frozen PDF). The unchanged builder
  rewrote the output byte-identically (sha256 31d1d07812deef3fd15bb75fd676a1dc152dfa44cb38bf019d69d91657eee21e).
  - The system `python3` (matplotlib 3.10.6) gave different bytes (d084b073…).
    The only difference was the PDF Producer string. Text, words, 2,389 paths and
    the 150-dpi raster were identical. That output was restored at once.
  - The final output was built with the `.venv` interpreter.
  - Saved original: `si_pass/renders/local_gate_controls_orig.pdf`; before render
    `si_pass/renders/local_gate_controls_before.png`.
- New output sha256: 5d5d1ed12908ee68157a4fd78a022cf97d6b589c97af3dba5aa4605af2f1ebf9
  (518.4 x 478.0 pt, was 518.4 x 492.0 pt).
  - The registry entry N21 must be re-registered.
  - Note: `original_assets.json` in HEAD already recorded sha 72d2e71a… for N21. That
    matched neither the pre-pass output nor this one.
- Checks:
  - Strict canvas audit: 0 violations.
  - `letter_ink_audit.py` on the native PDF: 0 problems. Before: "C: letter right edge
    only -0.5 pt left of panel ink (path 'fs')", from the old key's first marker under
    letter C.
  - Offline replay of the whole-sheet paste (`si_pass/sim_reflow.py`): 0 relocations,
    0 problems. Before, the replay reproduced the shipped sheet's "C: letter right edge
    only 2.1 pt left of panel ink", with relocations A/C -2.1 pt and B/D -10.85 pt.
- Marks: all 2,385 data and furniture paths kept. The 4 paths removed are key marks
  only: the seed-dot sample and the whisker sample (bar plus 2 caps).
  - Axes widths are unchanged: A/C 215.4 pt, B/D 191.5 pt, 196 pt tall.
  - So the hidden-interval rule, which depends on points per decade, still hides the
    same 13 intervals: A 2, B 0, C 8, D 3.

## Removed

| Where | Removed text (verbatim) | Caption status |
|---|---|---|
| key, line 1 | `mean at Adam rate 0.01 (top strip of a rule)` (with ▲) | reduced to `rate 0.01`. Caption already states it: 'Each rule has three strips, from top to bottom: Adam rates 0.01 (upward triangle), 0.03 (diamond; primary) and 0.1 (downward triangle)' and 'larger markers … are arithmetic means' |
| key, line 1 | `mean at 0.03, the primary rate (middle strip)` (with ◆) | reduced to `0.03`. Caption already states it: '0.03 (diamond; primary)' |
| key, line 1 | `mean at 0.1 (bottom strip)` (with ▼) | reduced to `0.1`. Caption already states it: '0.1 (downward triangle)' |
| key, line 1 | `one seed (20 per strip)` (with the seed dot) | caption already states it: 'Small dots are test NMSE at each seed's validation-selected checkpoint' … '($n=20$ fresh paired seeds)' |
| key, line 2 | `95 % CI of the mean (20,000 whole-seed bootstrap draws);` (with the whisker sample) | caption already states it: 'larger markers and whiskers are arithmetic means and 95\% percentile intervals from 20,000 whole-seed bootstrap draws' |
| key, line 2 | `13 of 96 CIs lie entirely under their marker and are not drawn` | ADD TO CAPTION (correction). The caption says 'Sixteen intervals narrower than 0.18 decade lie within their markers and are not drawn.', which disagrees with the artwork. 16 intervals are narrower than 0.18 decade, but 3 of them are drawn because one arm reaches just past the 3.6 pt marker: A exact path at 0.03 (0.175 decade), and C hard and continuous distal gates at 0.03 (0.177). The sheet hides 13 of 96, before and after this pass. |

## Caption additions

Replace this sentence of the `local_gate_controls` caption (specification.py, FIGURES, 5th element):

`Sixteen intervals narrower than 0.18 decade lie within their markers and are not drawn.`

with:

`Thirteen of the 96 intervals lie within their markers and are not drawn.`

No other addition is needed.

## Kept

- `Aligned task, 4,096 updates`, `Aligned task, 16,384 updates`, `Opposed task, 4,096 updates`,
  `Opposed task, 16,384 updates` (A–D titles): short condition labels. They separate four
  panels that are otherwise identical.
- Row labels `Exact path`, `Three-pattern oracle`, `Two-profile oracle`, `Hard distal gate`,
  `Continuous distal gate`, `Gate also proximal`, `Wrong branch`: direct series labels.
- `Unit broadcast (≈ calibrated)`: a direct series label. The parenthesis names the second rule
  this one row stands for. Without it the calibrated rule would vanish from the artwork. The
  caption gives the detail ('The unit-broadcast row also represents the calibrated rule's mean
  to two significant figures …').
- `Test NMSE at the validation-selected checkpoint` (C, D): the axis label, now in sentence case,
  defines the plotted quantity. It matches `test NMSE, last checkpoint before contact` on the
  conductance_optimization sheet.
- Key `▲ rate 0.01  ◆ 0.03  ▼ 0.1`: the compact strip-position key.

## Layout changes

- The two-line key (five entries) is now one line of three marker entries, 7 pt type, under the
  C/D axis labels.
  - The key is centred on the page, then shifted left so that it ends 6 pt short of letter D
    (193.1–295.9 pt).
  - In the whole-sheet paste no key mark enters the strip beside letter D, and none sits under
    letter C. That was the old C-letter failure.
- Canvas 492 → 478 pt tall (aspect 1.05 → 1.08). Bottom margin 52 → 38 pt. Rows, gutters and
  axes boxes are unchanged.
- Letters: A/C share a column, as do B/D. A/B share a row baseline, as do C/D. Relocations
  needed after the paste: none.
