# finite_horizon — clarity-pass ledger

- **Source key:** N35 (whole sheet)
- **Builder:** `scripts/build_supplementary_figure_finite_horizon_native.py` (gitignored; pre-edit copy at `si_pass/builder_backup/build_supplementary_figure_finite_horizon_native.py.orig`)
- **Output:** `figures/supplementary/figure_finite_horizon_native.pdf`
- **Reproducibility gate:** PASS. The unchanged builder under `python3` (Mambaforge, matplotlib 3.10.6) wrote a bitwise-identical PDF (sha256 `a43d67fd9bd3ea2bfe6c4fba1555c35c3d24efe114669ae545f3f1b0f0651f98`, saved as `si_pass/renders/finite_horizon_orig.pdf`; before render `finite_horizon_before.png`).
- **After edit:** sha256 `1d7f741102a46f5bdc44f0c2062548854f0a3139bd84eb39d8b8af9d337a4906`; canvas 518.4 × 453.3 pt (was 476.0). Strict canvas audit: 0 violations. `letter_ink_audit.py`: total problems 0.
- **Unchanged:** vector-mark census (1054 paths with the same type, colour, width and item count). Every axes box is identical to the original, so E/F keep the same equal-scale limits (x −0.087..0.927, y −0.030..0.830), hexagon grid and colour range (1..1,824).

## Removed

| # | Panel | Verbatim string | Caption status |
|---|---|---|---|
| 1 | A | `Feedback only: final selection` (title) → condition label `Feedback only` | caption already states it: '\textbf{A,B}, Final test loss + cost regret ($\times 1{,}000$) pooled over generating ranks in the feedback-only and joint-transfer fixed-cache learners' |
| 2 | B | `Joint transfer: final selection` (title) → condition label `Joint transfer` | caption already states it (same sentence as #1) |
| 3 | C | `Strong baselines in the joint arm` (title) | caption already states it: '\textbf{C}, The four strong selectors at each generating rank in the joint arm' |
| 4 | D | `Measured single-CPU cost` (title) | caption already states it: '\textbf{D}, Median elapsed time on one central processing unit per twenty-candidate decision' |
| 5 | E | `Feedback only: original scalar forecast` (title) → condition label `Original scalar` | caption already states it: '\textbf{E}, Original scalar forecast versus final test half-mean-squared-error in the feedback-only arm.' |
| 6 | F | `Feedback only: Gaussian SGD forecast` (title) → condition label `Gaussian SGD` | caption already states it: '\textbf{F}, The calibration-derived Gaussian SGD forecast on the same cases and arm.' |
| 7 | A, B | `n = 20 seed blocks; mean [95 % CI]` (tag, ×2) | caption already states it: 'Markers and whiskers in \textbf{A--C} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws over twenty seed blocks' |
| 8 | A, B | `zero regret` (reference-rule label, ×2) | caption already states it: 'with the dashed rule at zero regret' |
| 9 | A | `= Gaussian full-batch` / `(320/320 tasks)` (note on the context-count row) | caption already states it: 'in \textbf{A} the Gaussian full-batch and context-count rows coincide because they chose the same candidate in all 320 tasks' |
| 10 | D | `median [IQR] of 320–1,280 timing draws per row` (tag) | caption already states it: '\textbf{D} shows medians and interquartile ranges of 320 (context count), 640 (forecasts and original scalar) or 1,280 (pilot and full training) timing draws' |
| 11 | D | `(reference)` from the rule label `all 256-update fits (reference)` | caption already states it: 'the dashed rule with its grey band is the median and interquartile range of `all 256-update fits', the cost of training every candidate' |
| 12 | E, F | `n = 6,400 candidate fits` (tag, ×2) | caption already states it: 'bin the 6,400 fits per panel' |
| 13 | E, F | `equality` (identity-line label, ×2) | caption already states it: 'the dashed grey line marks equality' |
| 14 | F | `(E, F)` from the colour-bar label `fits per hexagon (E, F)` | caption already states it: 'on the logarithmic colour bar beside \textbf{F}, shared by \textbf{E,F}' |

Axis labels changed to sentence case (text only): `test loss + cost regret (×1,000)` → `Test loss + …` (A, B x; C y, two lines); `generating rank` → `Generating rank`; `seconds per 20-candidate decision` → `Seconds per …`; `predicted final half-MSE` → `Predicted …`; `observed test half-MSE` → `Observed …`; colour bar `fits per hexagon` → `Fits per hexagon`.

## Caption additions

None. Every removed string is already in the current caption.

## Kept

- `Feedback only`, `Joint transfer` (A, B) and `Original scalar`, `Gaussian SGD` (E, F): short condition labels, each telling apart two otherwise-identical panels.
- `RMSE 0.256`, `RMSE 0.015` (E, F): the effect sizes that carry each panel's result. The caption already says 'each panel prints its root-mean-square forecast error'.
- `all 256-update fits` (D): direct label naming the reference rule and band, which is not a trivial reference. The caption quotes the same name.
- C key (`Gaussian SGD`, `Gaussian full-batch`, `context count`, `population oracle`): compact symbol key.
- Row labels of A, B and D: category labels.

## Layout changes

- Titles: A, B, E and F carry the condition label at a 4 pt pad (`CONDITION_PAD`). The pads were 22 pt and 11 pt before, to clear the removed tags. C and D have no title. `TITLE_PAD` and `TITLE_PAD_KEYED` were removed, and so was the `tag()` helper, which has no callers left.
- `ROW_PT` changed from [138, 112, 110] to [123.24, 105.04, 109.04]. Each slot is the original axes-box height plus the row's new top reserve:
  - row 0: the measured letter band, 7.0 pt;
  - row 1: a declared `ROW1_TOP_PT` of 14 pt for C's two-row key and the letter;
  - row 2: a declared `ROW2_TOP_PT` of 9 pt for the condition label and the letter.
- All axes boxes are identical to the original: A, B 154.46 × 116.24; C, D 154.46 × 91.04; E, F 118.093 × 100.04 pt.
- `CANVAS_H_PT` is now computed: 453.32 pt (was 476.0).
- Added `assert ns == [320, 640, 1280]` in `panel_cost`, so the timing-draw counts the legend now carries alone stay checked against the data.
