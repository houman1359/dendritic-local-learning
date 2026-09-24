# anatomy_preprocessing — SI clarity pass 2026-09-23

**Status: NOT EDITED.** The builder failed the reproducibility gate, so the builder,
the figures/ outputs and the registry are untouched. A verified clarity-pass
**proposal** (not applied) is in `si_pass/proposed/`; see "To apply" at the end.

- Source key: S33 (whole sheet, `('S33','*')`)
- Builder: `scripts/build_review_completion_figures.py` (default args; `morphology()` writes S33,
  `encoder()` writes S32, which is cropped into ancestry_coefficients)
- Outputs: `figures/supplementary/figure_S33_panels_A-F.pdf`, `figures/supplementary/figure_S32_panels_A-F.pdf`
- Reproducibility gate: **FAIL (byte-only).**
  - S33 PDF: byte-identical on rerun (sha256 3a6ad356…99392a1 = registry).
  - S32 PDF: differs in exactly 5 bytes, all inside `/CreationDate` (registered
    `D:20260913015201-04'00'`, rerun stamps the current time). `encoder()` saves without
    `metadata={'CreationDate': None}`. File size is the same (41,283 bytes) and the 150 dpi
    renders are pixel-identical (max abs difference 0).
  - Both originals were copied back and their sha256 re-verified. Saved copies:
    `si_pass/renders/anatomy_preprocessing_orig.pdf`, `si_pass/renders/ancestry_coefficients_S32_orig.pdf`.
    Rerun copies: `*_rerun_unchanged.pdf`. Before renders: `anatomy_preprocessing_before.png`,
    `ancestry_coefficients_S32_before.png`. Builder source backup:
    `si_pass/builder_backup/build_review_completion_figures.py.orig`.
- S32 C/D: not cleaned, because the gate failed on S32 itself. Candidates if it is cleared later:
  the titles `Calibration data and computation` and `Cue / eligibility timing mismatch`, and D's italic
  note `soft, hard and mismatched` / `coincide at delays 1 and 4` / `(within 1 pp)`. The
  ancestry_coefficients caption already states the note. Outside C/D, the shared-key footnote
  `* maximum-probability route selection by the same frozen encoder (exploratory paired sensitivity)`
  is also already in that caption. The proposal leaves `encoder()` byte-for-byte unchanged.

## Removed (in the proposal)

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `Direct / proxy label disagreement` (title) | caption already states it: 'Direct-presynaptic versus target-proxy excitatory/inhibitory (E/I) labels for 2,012 jointly labeled mapped contacts' |
| B | `Direct-label coverage is uneven` (title) | ADD TO CAPTION (see below) |
| B | `$n$ = 342`, `$n$ = 5,568`, `$n$ = 25,580` (second line of the tick labels; also the sheet's only mathtext) | caption already states it: '342 pooled; … 5,568 pooled; … 25,580 pooled' |
| B | `(pooled contacts)` (x label is now `Compartment`) | caption already states it: '342 pooled; 157--1,541 internal, 5,568 pooled; …'. The suffix only explained the removed counts. |
| C | `Mapping and label choices alter routes` (title) | caption already states it: 'With hybrid labels, 5 of 8 cells change at least one route at $2\,\mu$m (mean 0.72) and 3 of 8 at $10\,\mu$m (mean 0.82); with direct-only labels all eight differ at every threshold' |
| C | `reference dictionary` / `(Jaccard = 1 by definition)` and its leader line | caption already states it: 'the open symbol at hybrid, $5\,\mu$m is the reference compared with itself (Jaccard $=1$)' |
| D | `Radius sensitivity on a fixed probe` (title) | caption already states it: 'Capture of the fixed nominal field under heterogeneous log-radius perturbations' |
| E | `Axial resistance after compression` (title) | caption already states it: 'Distribution of the mean-radius compressed axial resistance divided by the exact series sum over raw edges' |
| E | `72 of 608 segments (8 cells)` / `below 0.89 (0.05 log units);` / `worst case 0.003` | caption already states it: '72 of 608 segments (8 cells) fall below 0.89 (0.05 log units), the worst to 0.003 (2.5 log units)' |
| F | `Series-resistance sensitivity` (title) | caption already states it: 'Series-resistance minus mean-radius capture of the nominal field' |
| F | `+0.049`, `+0.017` (callouts at the two nonzero points) | caption already states it: 'two differ by $+0.049$ and $+0.017$' |
| F | `6 of 8 cells identical`, shortened to `6/8 identical` | caption already states it: 'Six of eight cells are identical' |

## Caption additions (only if the proposal is applied)

- Append to the \textbf{B} sentences, after '… weight cells equally.':
  `Direct-label coverage falls from the soma (cell mean 20.8\%) to internal (12.4\%) and terminal (7.0\%) compartments.`
  (Cell means of the eight cells, recomputed from `source_data/review_morphology_uncertainty/label_missingness.csv`.)

## Kept

- `489` / `(85.3%)` and the other three heatmap cells (A): the plotted quantity. The caption says 'cells print counts and row percentages'.
- `Row (%)` (A): colour-bar label.
- `hybrid`, `direct only` (C, D): compact symbol keys.
- `6/8 identical` (F): compact sign count (house rule 2).
- Tick labels `E`, `I`, `Soma`, `Internal`, `Terminal`, and every axis label, all already sentence case:
  `Target-proxy label`, `Direct label`, `Directly typed contacts (%)`, `Compartment`,
  `Maximum mapping distance (µm)`, `Selected-route Jaccard`, `Log-radius perturbation SD`,
  `Nominal-field capture`, `Compressed / series axial resistance`, `Cable segments`,
  `Mean of the two captures`, `Series-resistance minus` / `mean-radius capture`.

## Layout changes (in the proposal)

- Sheet height 7.0 → 6.33 in (504.0 → 455.8 pt; aspect 1.03 → 1.14). The axes grid keeps its
  7.0 in absolute height, so every axes box keeps its size (heatmap 167.68 × 99.99 pt, B–F
  188.41 × 99.99 pt, colour bar 5.0 × 99.99 pt). `centre_grid` then gives blank gutters of
  28.2 / 28.2 pt between rows (original 28.2 / 28.5 pt).
- A: the colour bar moves 4.9 pt toward the heatmap (gap 9.4 → 4.5 pt; heatmap and bar sizes
  unchanged). Without titles the row's letters sit level with the bar's `100` tick label. The
  supplement's whole-sheet partition (`make_bounds`) cuts at the B/D/F letter column, and without
  the move it assigns `Row (%)` to panel B. The replayed `relocate_letters` could then move
  B/D/F only 7.0 pt and left B 5.1 pt inside "its" ink. The original sheet passes only because
  B/D/F are pushed 15.7 pt left, into A's colour-bar gutter. With the move there are 0
  relocations and 0 problems.
- The titles and provisional letters drawn by `panel()` are gone from `morphology()`.
  `finish_panel_letters` places the letters as before. Measured: tops 7.0–9.6 pt above the
  panel ink, right edges 6.3–24.5 pt left of it; each row shares a top and each column a left edge.
- E: the unused `tail` count was removed; the no-segment-exceeds guard stays (its message now
  says "caption").

## Checks (proposal PDF `si_pass/proposed/out/figure_S33_panels_A-F.pdf`)

- `figure_canvas.py --audit --strict`: no native-canvas manifest (panel-letter-layout/1), so
  `[manifest-missing]`, plus `[edge-clearance] 1.9 pt` at the top. That flag is identical in the
  original and comes from `finish_panel_letters`' 2 pt top strip. The original's `canvas-aspect`
  flag is gone.
- `letter_ink_audit.py`: "no native-canvas manifest". Letters were measured instead with
  `si_pass/proposed/letter_check.py` (panel-letter-layout boxes + raster ink): 0 problems.
- Supplement replay (`si_pass/sim_reflow.py` + `letter_ink_audit.py`): 0 relocations, 0 problems.
- `si_pass/proposed/compare_data.py s33` (original and proposed builders run in memory):
  all data marks and limits identical, every axes box the same size, S32 identical, and the
  text diff equals the table above.
- Visual: `si_pass/renders/anatomy_preprocessing_proposed.png`. No overlaps or clipping, even gutters.

## To apply (only if the coordinator accepts the byte-only S32 difference)

`cp si_pass/proposed/build_review_completion_figures.py scripts/` (diff:
`si_pass/proposed/build_review_completion_figures.diff`), then `python3 scripts/build_review_completion_figures.py`.
Expected: S33 sha256 01751e05…f3f9c64 (518.4 × 455.76 pt), equal to the sandbox output. S32 is
pixel-identical but gets a new `/CreationDate`; restore `ancestry_coefficients_S32_orig.pdf` if its
registered sha must hold. Re-register S33.
