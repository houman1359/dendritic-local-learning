# measured_transfer_geometry — clarity-pass ledger

- **Source key:** N31 (whole sheet)
- **Builder:** `scripts/build_supplementary_figure_measured_transfer_geometry_native.py` (gitignored; pre-edit copy at `si_pass/builder_backup/build_supplementary_figure_measured_transfer_geometry_native.py.orig`)
- **Output:** `figures/supplementary/figure_measured_transfer_geometry_native.pdf`
- **Reproducibility gate:** PASS, but only with the right interpreter.
  - The saved original (sha256 `13c9a4f277ee888ca55f9cdab3a87a33c5d40545cb46901360788905cab14c85`, `si_pass/renders/measured_transfer_geometry_orig.pdf`) was written by matplotlib **3.10.9**.
  - Under the default `python3` (Mambaforge, matplotlib 3.10.6) the unchanged builder gave a different hash (`820befea…`) with identical text spans. That output was kept as `si_pass/renders/measured_transfer_geometry_regen_unchanged_builder.pdf`, and the original was copied back over the output at once.
  - Under `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python` (matplotlib 3.10.9) the unchanged builder reproduced the original **bitwise**.
  - The edited sheet was therefore built with that `.venv` interpreter. **Rebuild with the `.venv` python**, not Mambaforge `python3`.
- **After edit:** sha256 `2f81dadccc66e2fe1ded42e1fd7e441ae6f2469ed1e7905348abd7088be78cb6`; canvas 518.4 × 386.6 pt (was 403.0). Strict canvas audit: 0 violations. `letter_ink_audit.py`: total problems 0. Axes boxes identical to the original: A, B 159.9 × 131.2 pt; C, D 159.9 × 162.2 pt.
- **Vector marks:** 495 → 494 paths. The one removed path is the dashed legend handle of the dropped key entry `simulated mean = calibration target`. Every data mark is unchanged.

## Removed

| # | Panel | Verbatim string | Caption status |
|---|---|---|---|
| 1 | A | `Measured cohort` (panel title) | caption already states it: '\textbf{A}, Seven targets from one mouse, one selected scan per target: mapped functional partners (filled circles), manually curated subset (open circles), and partner split-half repeat reliability' |
| 2 | B | `Inputs reached per route` (panel title) | caption already states it: '\textbf{B}, Input coordinates per selected route across all thirteen scans, ordered by mapped-input count.' |
| 3 | B | `(1 in 6 of 13 scans)` (second line of the x label) | caption already states it: 'Six scans have one coordinate per route; the mean is 1.31.' |
| 4 | B | `, 13 scans` from the y label `mapped inputs, 13 scans` | caption already states it: 'across all thirteen scans, ordered by mapped-input count' |
| 5 | C | `Fixed-profile fidelity: update reconstruction` (panel title) | caption already states it: '\textbf{C}, Common-checkpoint update reconstruction, $\mathcal M_D$ (…) for the unrestricted fixed transfer profile (blue), ancestry-restricted fixed profile (green circles) …' |
| 6 | C | `n = 7 targets; mean [95 % CI]` (stats tag) | caption already states it: 'Small dots are seven target means; larger symbols and bars show their mean and 95\% target-bootstrap interval (20,000 draws).' |
| 7 | C | `CI 0.964–0.986` (callout on the unrestricted-profile row) | ADD TO CAPTION (see below) |
| 8 | D | `Reliability calibration: simulated − target` (panel title) | caption already states it: '\textbf{D}, Reliability calibration: mean simulated split-half Spearman reliability minus the nonnegative target $r_+=\max(r_{\rm measured},0)$' |
| 9 | D | `5 of 125 beyond ±1.96 SE; largest |difference| 0.0069` (tag) | caption already states it: 'Five of 125 simulated means differ from the calibration target by more than 1.96 standard errors; the largest discrepancy is 0.0069.' |
| 10 | D | key entry `record: mean and ±1.96 SE whisker from` / `1,000 simulated datasets (n = 123 of 125)` → `partner–scan record` | caption already states it: 'from 1,000 independent calibration datasets. Dots represent 125 partner--scan records; whiskers show $\pm1.96$ Monte Carlo standard errors.' (123 = 125 − the two open squares) |
| 11 | D | key entry `negative measured r, set to 0 (n = 2 of 125;` / `raw −0.098 and −0.188)` → `measured r < 0` | partly stated: 'Open squares mark two negative measured reliabilities assigned zero reliable variance.' The raw values are not in the caption: ADD TO CAPTION (see below) |
| 12 | D | key entry `simulated mean = calibration target` (dashed rule, with its legend handle) | caption already states it: 'The dashed line marks zero discrepancy.' |

Axis labels changed to sentence case (text only): `partners` → `Partners`; `split-half r` → `Split-half r`; `inputs per route` → `Inputs per route`; `mapped inputs` → `Mapped inputs`; `update reconstruction (1 = exact)` → `Update reconstruction (1 = exact)`; `calibration target (split-half r)` → `Split-half r (calibration target)` (reordered rather than capitalised: `tests/test_supplement_review_20260914.py::test_calibration_residual_and_labels_use_clipped_target` asserts the lower-case phrase "calibration target" in D's x label; that test and `test_si_prose_does_not_publish_production_notes` pass); `simulated − calibration target` / `(split-half r)` → `Simulated − …`.

## Caption additions

1. **C sentence:** replace 'The unrestricted fixed profile nearly reconstructs the update;' with
   `The unrestricted fixed profile nearly reconstructs the update (95\% interval 0.964--0.986, narrower than its symbol);`
2. **D sentence:** replace 'Open squares mark two negative measured reliabilities assigned zero reliable variance.' with
   `Open squares mark two negative measured reliabilities ($-0.098$ and $-0.188$) assigned zero reliable variance.`

## Kept

- `mapped partners`, `manual subset`, `partner record`, `target median` (A keys), `scan`, `main Fig. 10E scan` (B key), `fixed transfer profile`, `oracle trialwise amplitudes` (C key), `partner–scan record`, `measured r < 0` (D key): compact symbol keys, each a marker or line glyph with at most 4 words.
- `(1 = exact)` in C's x label: it anchors the scale like a unit, and the caption gives the same anchor ('1 = exact').
- `target 1` … `target 7`, B's right-hand mapped-input counts, C's row labels (`unrestricted` / `fixed profile`, …): tick and category labels. B's counts are the ordering variable of the rows.

## Layout changes

- The four panel titles and their `TITLE_PAD` were removed, along with the now-unused `PT_EMPH` import.
- `CANVAS_H_PT` is now computed from the margins, rows and gutter (it was a literal, 403.0).
- `ROW_PT` changed from [140, 165] to [136.2, 165.4]: the untitled rows' lock reserves (5.0 and 3.2 pt) plus the original axes heights, so every axes box keeps its size.
- `VGUTTER_PT` changed from 52 to 39 pt: no title sits in the gutter any more, and B's x label is one line.
- D: the compact two-entry key lowers the measured key headroom (`calibration_headroom`) from about 45 pt to 20.6 pt. The data band keeps its symmetric ±lim range and ticks, and only the empty space above the data for the key shrinks.
- The C row callout was removed with no geometry change.
