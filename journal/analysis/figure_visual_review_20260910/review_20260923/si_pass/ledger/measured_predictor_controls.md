# measured_predictor_controls — SI clarity pass 2026-09-23

**Status: NOT EDITED.** The builder failed the reproducibility gate on a side output, so the
builder, the figures/ and source_data/ outputs and the registry are untouched. A verified
clarity-pass **proposal** (not applied) is in `si_pass/proposed/`; see "To apply" at the end.

- Source key: S34 (whole sheet, `('S34','*')`)
- Builder: `scripts/build_review_response_baselines_figure.py` (no arguments)
- Outputs: `figures/supplementary/figure_S34_panels_A-D.pdf`,
  `source_data/review_response_baselines/figure_S34_preview.png`,
  `source_data/review_response_baselines/figure_S34_provenance.json`
- Reproducibility gate: **FAIL (visual, stale side output).**
  - S34 PDF: byte-identical on rerun (sha256 6d04a5f2…797593 = registry).
  - `figure_S34_provenance.json`: byte-identical (afd01c9f…a163a; this hash is locked in
    `source_data/provenance_manifest.tsv` and the credit_first provenance tables).
  - `figure_S34_preview.png`: **differs visually** (14% of pixels, max channel difference 225).
    The on-disk preview (7 Sep, 88178bb3…) is a stale render of an older builder version, with a
    "Train mean" row, "Exact nonlinear" rows, a footer sentence and a green panel D. The rerun
    (ccd75e8a…) matches the registered PDF. No registry or provenance file references this PNG.
  - All three originals were copied back and their sha256 re-verified. Saved copies:
    `si_pass/renders/measured_predictor_controls_orig.pdf`, `…_preview_orig.png`, `…_provenance_orig.json`.
    Rerun copies: `…_rerun_unchanged.pdf`, `…_preview_rerun_unchanged.png`. Before render:
    `measured_predictor_controls_before.png`. Builder backup:
    `si_pass/builder_backup/build_review_response_baselines_figure.py.orig`.

## Removed (in the proposal)

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `Prediction baselines` (title) | caption already states it: 'Ordinary least-squares (OLS) and nested ridge-regression baselines against the archived exact compartment-error fit' |
| B | `Observed-input sensitivity` (title) | caption already states it: 'Ridge prediction when retaining nested random fractions of observed presynaptic inputs' |
| C | `Measurement-noise sensitivity` (title) | caption already states it: 'Added Gaussian predictor noise in both training and test data' |
| D | `Training-only partner filtering` (title) | caption already states it: 'Reliability filtering based only on training repeats' |

## Caption additions

None. Every removed string is already in the caption.

## Kept

- D tick labels `All` / `(9.0)`, `≥0` / `(8.7)`, `≥0.1` / `(7.5)`, `≥0.2` / `(5.6)` and the x-label line
  `(mean retained partner count)`. The numbers describe each filtering condition (a second x
  scale), not sample sizes, and the caption points to them: 'parentheses give mean retained partner counts'.
- Row labels `Linear (OLS)`, `Nested ridge`, `Exact compartment` / `error`, `Ridge, raw inputs`,
  `Topology routes`, `Random routes`, `Site-shuffled routes` (A): category tick labels.
- `25%`, `50%`, `75%`, `All`, `Manual` / `only` (B): category tick labels. Manual-only is a separate condition.
- Axis labels, already sentence case: `Held-out normalized MSE`,
  `Difference from nested ridge (normalized MSE)`, `Retained observed presynaptic inputs`,
  `Added predictor noise (training feature SD)`, `Training repeat reliability` /
  `(mean retained partner count)`.

## Layout changes (in the proposal)

- Sheet height 6.6 → 6.3 in (475.2 → 453.6 pt; aspect 1.09 → 1.14). The gridspec row gap is
  10.8 pt smaller. Both axes rows keep their 153.96 pt height; widths and the bottom margin are
  unchanged. The blank bands (top 20.0, between rows 30.0, bottom 14.5 pt) equal the original's.
- The unused `panel_title` import was dropped. Letters are still placed by `finish_panel_letters`:
  tops 7.0 pt above the panel ink, right edges 5.5–42.8 pt left of it; rows and columns aligned.

## Checks (proposal PDF `si_pass/proposed/out/figure_S34_panels_A-D.pdf`)

- `figure_canvas.py --audit --strict`: only `[manifest-missing]` (panel-letter-layout/1 sheet, as in the original).
- `letter_ink_audit.py`: "no native-canvas manifest". `si_pass/proposed/letter_check.py`: 0 problems.
- Supplement replay (`si_pass/sim_reflow.py` + `letter_ink_audit.py`): 0 relocations, 0 problems.
- `si_pass/proposed/compare_data.py s34`: all data marks and limits identical, every axes box the
  same size, and only the four titles differ.
- The proposal writes the provenance JSON byte-identically (afd01c9f…).
- Visual: `si_pass/renders/measured_predictor_controls_proposed.png`. Clean.

## To apply (only if the coordinator accepts replacing the stale preview PNG)

`cp si_pass/proposed/build_review_response_baselines_figure.py scripts/` (diff alongside), then
`python3 scripts/build_review_response_baselines_figure.py`. Expected: S34 sha256 eb090e4f…7b55f5c31
(518.4 × 453.6 pt). Provenance JSON unchanged (afd01c9f…). The preview PNG becomes 01c62f15…837861.
Re-register S34.
