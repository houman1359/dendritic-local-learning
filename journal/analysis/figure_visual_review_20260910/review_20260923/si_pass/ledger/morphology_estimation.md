# morphology_estimation — clarity-pass ledger

- **Source key:** N36 (whole sheet)
- **Builder:** `scripts/build_supplementary_figure_morphology_estimation_native.py` (gitignored; pre-edit copy at `si_pass/builder_backup/build_supplementary_figure_morphology_estimation_native.py.orig`)
- **Output:** `figures/supplementary/figure_morphology_estimation_native.pdf`
- **Reproducibility gate:** PASS. The unchanged builder under `python3` (Mambaforge, matplotlib 3.10.6) wrote a bitwise-identical PDF (sha256 `06f874cbb45f43a3c6b7902038c3b3c25f88e2bf2b0a4a6319270bbf7cc780f3`, saved as `si_pass/renders/morphology_estimation_orig.pdf`; before render `morphology_estimation_before.png`).
- **After edit:** sha256 `48c4a42f8936aab206bd035f27eb46545fbc43117ae9dabcb110c3b195c37a01`; canvas 518.4 × 493.0 pt (unchanged). Strict canvas audit: 0 violations.
- **`letter_ink_audit.py`:** total problems 0. The original had 3: A and B sat under the sheet-wide colour key, and H sat below its own over-long y label.
- **Vector marks:** 1746 → 1752 paths. The 6 added paths are the glyphs of the two new compact keys: C's two line handles and two markers, and G's filled and open markers. Every data mark is unchanged (same type, colour, width and item count).
- **Printed statistics:** the builder's `[B]`–`[H]` lines are identical, except B's derived "margin … pt on the axis" (0.70 → 0.74 pt, from the taller axes).

## Known letter problem: fixed at source

The sheet-wide colour key ran across the top of the page, above the A/B letter row. It is now one compact row centred under the bottom panels (`sheet_key`, `loc="lower center"`, 3 pt above the canvas bottom), below H's three-line category labels. Its labels were compacted too (see Removed, rows 1–3).

## Removed

| # | Panel | Verbatim string | Caption status |
|---|---|---|---|
| 1 | key | `estimated interactions / tree` → `estimated` | caption already states it: 'green, estimated interactions' (D); 'estimated (green) … trees' (G); 'Secondary adaptive trees (green; ALS fits)' (E) |
| 2 | key | `fixed / estimated-rank tree (menu of twelve)` → `fixed menu tree` | caption already states each grey role: 'grey, fixed/estimated-rank baseline' (D), 'the best trained candidate (grey)' (E), 'development-fixed (grey)' (G). The menu size of twelve comes back through caption addition A ('12 candidate cuts') |
| 3 | key | `(oracle)` from `target-informed tree (oracle)` → `target-informed tree` | ADD TO CAPTION (additions E and G) |
| 4 | A | `Sealed calibration protocol` (title) | caption already states it: '\textbf{A}, Sealed calibration protocol' |
| 5 | A | `256 noisy labels (noise SD 0.5)` → `256 noisy labels` | ADD TO CAPTION (addition A: noise SD 0.5) |
| 6 | A | `192 labels: fit the estimator or the pilots` → `192: fit` | ADD TO CAPTION (addition A) |
| 7 | A | `estimate interactions` / `(Lasso, 255 Walsh terms)` → `estimate interactions` | ADD TO CAPTION (addition A: Lasso over 255 Walsh terms) |
| 8 | A | `score the 12 candidate cuts` → `score 12 cuts` | ADD TO CAPTION (addition A: '12 candidate cuts') |
| 9 | A | `fit the 12 pilots` / `(2 sweeps each)` → `fit 12 pilots` | caption already states it: 'orange, two-sweep pilot' |
| 10 | A | `gate on the 64 held-out` → `gate on 64` | ADD TO CAPTION (addition A: '64 held-out labels') |
| 11 | A | `seal choices; reset; train every candidate afresh` → `seal; reset; train afresh` | partly stated ('both reset before training'); ADD TO CAPTION (addition A) |
| 12 | B | `Two prespecified pooled comparisons` (title) | caption already states it: '\textbf{B}, Pooled fixed-baseline-minus-estimated (left, grey) and pilot-minus-estimated (right, orange) NMSE regret' |
| 13 | B | `(mean; Bonferroni 97.5% CI)` (second line of the y label) | caption already states it: 'short rule, pooled mean; whisker, Bonferroni-adjusted 97.5\% paired interval' |
| 14 | B | `dots: 80 tasks per column` | caption already states it: 'small dots, 80 per-task paired differences' |
| 15 | B | `open: family means (n = 4)` | caption already states it: 'open squares, family means' |
| 16 | B | `lower bounds` / `> 0.01 NMSE` | caption already states it: 'the 0.01-NMSE margin lies below both lower confidence bounds (0.083, 0.027)' |
| 17 | C | `Regret by calibration labels and label noise` (title) | caption already states it: '\textbf{C}, Pooled menu regret (…) against calibration labels' |
| 18 | C | `solid: noise SD 0.5; dashed: noiseless` → glyph key `noise SD 0.5` (solid, filled) / `noiseless` (dashed, open) | caption already states it: 'solid curves, filled markers, noise SD 0.5; dashed, open markers, no noise' |
| 19 | C | `bands: 95% seed bootstrap (n = 20 seeds)` | caption already states it: 'bands, 95\% intervals' and 'Intervals in \textbf{C--E} and \textbf{G}: pointwise 95\%, whole-seed bootstrap, $n=20$.' |
| 20 | C | `grey fixed baseline: one curve, both noise levels` | caption already states it: 'the fixed / estimated-rank baseline is one grey curve for both noise levels' |
| 21 | D | `Primary condition: regret by family` (title) | caption already states it: '\textbf{D}, Family-specific primary regret' |
| 22 | D | `dots: 20 tasks per mark; whisker 95%` | caption already states it: 'small dots, 20 per-task regrets; short rule, mean; whisker, 95\% interval' |
| 23 | D | `green quartic CI 0.0014–0.0046` / `is narrower than the rule` | caption already states it: 'the quartic estimated-cut interval (0.0027 [0.0014, 0.0046]) is narrower than the rule' |
| 24 | E | `Secondary adaptive construction (ALS fits)` (title) | caption already states it: '\textbf{E}, Secondary adaptive trees (green; ALS fits)' |
| 25 | E | `dots: 20 tasks per mark; whisker: 95% seed bootstrap` | caption already states it: 'marks as in \textbf{D}' and 'Intervals in \textbf{C--E} and \textbf{G}: pointwise 95\%, whole-seed bootstrap, $n=20$.' |
| 26 | E | `grey: best trained of twelve (oracle)` | 'the best trained candidate (grey)' already stated; '(oracle)': ADD TO CAPTION (addition E) |
| 27 | E | `purple: target-informed tree (oracle)` | 'the true-target tree (purple)' already stated; '(oracle)': ADD TO CAPTION (addition E) |
| 28 | F | `Selection cost before final training` (title) | caption already states it: '\textbf{F}, One-CPU selection time for eighty primary tasks' |
| 29 | F | `dots: 80 tasks per arm` | caption already states it: 'eighty primary tasks: dots, task times' |
| 30 | F | `rule: median; whisker: IQR` | caption already states it: 'rules, medians (…); whiskers, interquartile ranges' |
| 31 | G | `Fresh Adam cohort: all six conditions` (title) | caption already states it: '\textbf{G}, Fresh Adam cohort: estimated (green), development-fixed (grey) and target-informed (purple) trees under exact credit (filled) and root broadcast (open)' |
| 32 | G | `(20 seeds per mark; 95% CI)` (second line of the y label) | caption already states it: 'small dots, 20 per-seed endpoints; whisker, 95\% interval' |
| 33 | G | `filled: exact credit (0.01)` → glyph key `exact` (filled) | 'exact credit (filled)' already stated; the rate 0.01: ADD TO CAPTION (addition G) |
| 34 | G | `open: root broadcast (0.003)` → glyph key `broadcast` (open) | 'root broadcast (open)' already stated; the rate 0.003: ADD TO CAPTION (addition G) |
| 35 | G | `dashed: noise floor 0.0225` | caption already states it: 'dashed rule, label-noise floor 0.0225' |
| 36 | H | `Fresh Adam cohort: pooled paired contrasts` (title) | caption already states it: '\textbf{H}, Pooled paired noisy-test differences' |
| 37 | H | `difference (mean; CI as labelled)` (second line of the y label; now `Paired noisy-test` / `NMSE difference`) | caption already states it: 'short rule, pooled mean; whisker, Bonferroni-adjusted 97.5\% interval for the two primary contrasts and pointwise 95\% for estimated-minus-target-informed' |
| 38 | H | `97.5% CI` (×2) and `95% CI` (from `(exact) 95% CI`) in the category labels | caption already states it (same sentence as #37) |
| 39 | H | `dots: 20 seed means (4 families averaged)` | caption already states it: 'Pooled paired noisy-test differences …: small dots, 20 seed differences' |
| 40 | H | `open: family means (n = 4)` | caption already states it: 'open squares, family means (matching, quartic, nested, random, left to right)' |

Axis labels changed to sentence case, with long labels split so each fits within its 82 pt axes (text only):
- B: `Baseline −` / `estimated regret`
- C: `Pooled menu` / `regret (NMSE)`
- D: `Menu regret (NMSE)`
- E: `Clean-test NMSE`
- F: `Elapsed time on` / `one CPU (s)`
- G: `Noisy-test NMSE`
- H: `Paired noisy-test` / `NMSE difference`

## Caption additions

1. **A sentence:** replace
   '\textbf{A}, Sealed calibration protocol (Section~\ref{note:finite_calibration_morphology}): green, estimator; orange, two-sweep pilot; both reset before training.'
   with
   `\textbf{A}, Sealed calibration protocol (Section~\ref{note:finite_calibration_morphology}) at the primary condition (256 labels, noise SD 0.5): 192 labels fit the estimator (green; Lasso over 255 Walsh terms), which scores the 12 candidate cuts, or the 12 two-sweep pilots (orange), which are gated on the 64 held-out labels; choices are sealed and both are reset before every candidate is trained afresh.`
2. **E sentence:** replace 'against the best trained candidate (grey) and the true-target tree (purple),' with
   `against two retrospective oracles, the best trained candidate (grey) and the true-target tree (purple),`
3. **G sentence:** replace 'target-informed (purple) trees under exact credit (filled) and root broadcast (open)' with
   `target-informed (purple; oracle) trees under exact credit (filled; Adam learning rate 0.01) and root broadcast (open; 0.003)`
4. **Optional, for the moved key** (the caption never says where the sheet key is): append `Colours as in the key below \textbf{G,H}.` to the end of the caption.

## Kept

- A box labels `256 noisy labels`, `192: fit`, `64: gate`, `estimate interactions`, `score 12 cuts`, `fit 12 pilots`, `gate on 64`, `seal; reset; train afresh`: the protocol schematic's boxes, shortened to terse labels.
- Sheet key `estimated`, `fixed menu tree`, `two-sweep pilot`, `target-informed tree`: compact colour key shared by all panels (1–3 words each). `fixed menu tree` is the umbrella for the per-panel grey roles defined in the caption.
- C key `noise SD 0.5` / `noiseless` and G key `exact` / `broadcast`: compact glyph keys, converted from the removed text notes. G's key is one row placed above the 1.5 grid rule and clear of the 1.76 outlier. `exact` and `broadcast` match H's category wording.
- B ticks `fixed / estimated-rank`, `two-sweep pilot`; C ticks `64 labels`, `256 labels`, `1024 labels`; F ticks `estimate + score twelve`, `twelve two-sweep pilots`; family names; H categories `fixed − estimated` / `(exact credit)`, `broadcast − exact` / `(estimated tree)`, `estimated −` / `target-informed` / `(exact)`: category and tick labels. The parentheses in H are condition labels (credit rule, tree).

## Layout changes

- All eight panel titles and all eight in-panel notes (two of them in B) were removed. The `note()` and `note_lines()` helpers were replaced by `glyph_key()`. E's two note-band assertions went too, since they only justified the note's placement. C's band assertion now guards its key, and G's outlier assertion guards its key.
- The sheet key moved from above row 0 to the bottom, as described above.
- A: `box1_h` changed from 17 to 11 pt (one-line labels). The slack is redistributed into the gaps, so the drawing still fills its cell, and `require_delta0` is unchanged.
- `MARGINS` top changed from 34 to 21 pt (the letter band only) and bottom from 38 to 42 pt (adds the key band under H's three-line labels). `VGUTTER_PT` changed from 36 to 34 pt.
- The canvas stays 493 pt (the 1.05 aspect floor). The freed space lengthens each row from 78.25 to 82.0 pt, so every axes box grows from 180.2 × 78.2 to 180.2 × 82.0 pt. All data limits and ticks are unchanged, all eight boxes keep one size, and the panel aspect goes from 2.30 to 2.20.
- Letters now sit ≥ 2 pt above and ≥ 3 pt left of all their panel's ink, and each grid row and column stays aligned.
