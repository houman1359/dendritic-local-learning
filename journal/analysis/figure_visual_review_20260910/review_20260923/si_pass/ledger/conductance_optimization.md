# conductance_optimization — SI clarity pass 2026-09-23

- Source key: N20 (whole sheet)
- Builder: `scripts/build_supplementary_figure_conductance_optimization_native.py`
- Output: `figures/supplementary/figure_conductance_optimization_native.pdf`
- Reproducibility gate: PASS. The unchanged builder, run with `python3`
  (matplotlib 3.10.6, the producer recorded in the frozen PDF), rewrote the output
  byte-identically (sha256 7472557cd667e2991d970a2824b96ebe42a24773525330bf2679c0e4353d6173). Saved original:
  `si_pass/renders/conductance_optimization_orig.pdf`; before render
  `si_pass/renders/conductance_optimization_before.png`.
- New output sha256: eb686c12d10324726febb11dd5fd2617a887782d53654a831de7c11a64712df3
  (518.4 x 486.0 pt, was 518.4 x 492.0 pt). The registry entry N20 must be re-registered.
- Checks: strict canvas audit 0 violations; `letter_ink_audit.py` 0 problems on the
  native PDF. Offline replay of the whole-sheet paste (`si_pass/sim_reflow.py`:
  the `make_bounds` letter partition plus `letter_relocation.relocate_letters`)
  gives 0 relocations and 0 letter problems. Before this pass the same replay
  reproduced the shipped sheet's "E: letter right edge only -1.4 pt left of panel ink".
- Marks: all 688 vector paths kept (same count per colour, fill and width class).
  Only the key handles changed, from 2-segment to 1-segment paths.

## Removed

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `Wider conductance bounds retain the gap` (title) | caption already states it: 'Parameter-range, development and rate controls retain the opposed-task broadcast deficit.' plus 'each under original $[-7,7]$ and wider $[-20,20]$ log-conductance bounds … All 160 broadcast seed values are positive' |
| A | `no interaction` (label of the zero line) | caption already states it: 'the dashed rule is zero' |
| A | `n = 20 paired seeds per row; mean [95 % CI]` | caption already states it: 'Small dots are the twenty paired seeds of each row, the large symbol their mean and the bar its 95\% paired bootstrap interval' |
| A | `means and 95 % CIs all within` / `±0.000005 of zero (not drawable)` | caption already states it: 'the oracle means and intervals lie within $\pm5\times10^{-6}$ of zero (too narrow to draw)' |
| B | `The gap precedes bound contact` (title) | caption already states it: 'The calibrated-broadcast gap at each seed's last saved checkpoint before its fit first reaches an original bound … all twenty gaps are positive' |
| B | `20/20 gaps > 0 (n = 20 seeds)` | caption already states it: 'one dot per seed, $n=20$, no interval): all twenty gaps are positive' |
| B | `no gap` (label of the zero line) | caption already states it: 'the dashed rule is zero' |
| C | `n = 3 seeds; symbol = mean` | caption already states it: 'small dots are the three development seeds, symbols their means' |
| C | `(best of three rates)` (x-label is now `Gating regime`) | caption already states it: 'each rule at its best of three rates (0.03 for all)' |
| D, E | `n = 3 seeds; symbol = mean` (x2) | caption already states it: 'Small dots are the three development seeds behind every point and lines join the three-seed means' |
| D | `unit and calibrated` / `broadcast coincide` / `(within 0.03 %)` | caption already states it: 'Unit and calibrated broadcast coincide within 0.03\% under Adam and 7.4\% under SGD' |

## Caption additions

None. Every removed string is already in the caption. The existing caption stays accurate.
It already says 'One key serves \textbf{A} and \textbf{C--E}' and 'the dotted vertical line marks the originally selected rate'.

## Kept

- `calibrated broadcast − exact path`, `unit broadcast − exact path`,
  `three-profile oracle − exact path` (A): direct group labels, each with its rule marker.
- `Adam ±7`, `Adam ±20`, `SGD ±7`, `SGD ±20` (A): row tick labels (optimizer and bound).
- `20/20 > 0`, `10/20 > 0`, `17/20 > 0` (A, one per row): compact per-row sign counts.
- `Adam, 4,096 updates`, `Adam, opposed, 16,384 updates`, `SGD, opposed, 16,384 updates` (C, D, E):
  short condition labels that separate three otherwise identical panels. No verbs and no findings.
- `selected rate` (D, E): two-word direct label of the dotted rule. Its meaning does not
  follow from the axis.
- `strong` / `moderate` / `ungated`, `aligned` / `opposed` (C): two-level category tick labels.
- Key: `exact path`, `three-profile oracle`, `calibrated broadcast`, `unit broadcast`,
  a compact symbol key.
- Axis labels, now in sentence case: `Task-by-credit interaction: opposed − aligned` /
  `difference in (rule − exact path) test NMSE`; `Update of first bound contact (Adam)`;
  `Calibrated broadcast − exact path` / `test NMSE, last checkpoint before contact`;
  `Development validation NMSE`; `Learning rate`; `Gating regime`.

## Layout changes

- A and B lost their titles. The C–E title pad went from 11 pt to 4 pt because the n tag is gone.
- Canvas 492 → 486 pt tall (aspect 1.05 → 1.07). Row weights 210:166 → 216:160.
  Vertical gutter 44 → 56 pt, so A's two-line x label stays nearer A than the row-1 titles
  (the native letter audit assigns ink to the nearest axes box). Margins: top 18 → 10 and
  bottom 58 → 44. Axes: A and B 213.8/152.4 x 201.0 → 205.0 pt; C–E 116.7 x 153.4 → 158.0 pt
  (aspect 0.76 → 0.74).
- Known E-letter problem fixed at source. In the consolidated sheet the old centred key ran
  under all three row-1 panels, and its `calibrated broadcast` label crossed the strip beside
  letter E, which the whole-sheet paste assigns to E. Letter relocation could move E only
  5.35 pt left before it met D's title, so E stayed 1.4 pt short. The key is now one row
  beside C's axis label, drawn as two pairs. `exact path`, `three-profile oracle` end 6 pt
  before letter E. `calibrated broadcast`, `unit broadcast` start at E's y label, which is
  E's own leftmost ink. Handles are 14 pt (2.0 em), text pad 3.5 pt and entry spacing 8.4 pt.
  The builder asserts both pairs stay clear of the D and E letter strips.
- Letters: A/B share a row baseline, as do C/D/E. A/C share a column. Relocations needed after the paste: none.
