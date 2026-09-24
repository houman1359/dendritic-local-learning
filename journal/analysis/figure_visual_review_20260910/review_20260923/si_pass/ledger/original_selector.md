# original_selector — SI clarity pass 2026-09-23

**Status: NOT EDITED.** The builder failed the reproducibility gate on a side output, so the
builder, the figures/ outputs and the registry are untouched. A verified clarity-pass
**proposal** (not applied) is in `si_pass/proposed/`; see "To apply" at the end.

- Source key: S35 (whole sheet, `('S35','*')`)
- Builder: `scripts/build_supplementary_figure_s35_native.py` (no arguments; `NativeCanvas.save`
  writes the PDF and a 600 dpi PNG)
- Outputs: `figures/supplementary/figure_S35_panels_A-F.pdf`, `figures/supplementary/figure_S35_panels_A-F.png`
- Reproducibility gate: **FAIL (visual, stale side output).**
  - S35 PDF: byte-identical on rerun (sha256 f97defa9…28d720c = registry).
  - S35 PNG: **differs visually** (13% of pixels, max 255). The on-disk PNG (7 Sep, fd5cdaaa…) is a
    stale render of an older six-panel version, with separate feedback-only and joint-transfer
    panels C/D and a two-tree schematic. The rerun (e8797011…) matches the registered PDF. No
    registry or provenance file references this PNG.
  - Both originals were copied back and their sha256 re-verified. Saved copies:
    `si_pass/renders/original_selector_orig.pdf`, `…_orig.png`. Rerun copies: `…_rerun_unchanged.pdf/.png`.
    Downscaled comparison: `…_orig_png_small.png`, `…_rerun_png_small.png`. Before render:
    `original_selector_before.png`. Builder backup:
    `si_pass/builder_backup/build_supplementary_figure_s35_native.py.orig`.

## Removed (in the proposal)

| Panel | Removed text (verbatim) | Caption status |
|---|---|---|
| A | `Explicit morphology / routing candidates` (title) | caption already states it: 'Four balanced leaf assignments (balanced 1--4; …) and one comb, each at route budgets $K=1,2,4,8$: twenty candidates with prescribed cost.' |
| A | `filled nodes: the K` / `feedback channels` / `(one subtree each);` / `5 trees × 4 budgets` | caption already states it: 'Filled nodes mark the $K$ feedback channels, one subtree each.' and 'twenty candidates' |
| B | `Selection is sealed before training` (title) | caption already states it: 'Development, independent calibration, sealed choices and confirmation training.' |
| B | Box texts `Development tasks` / `fit score-to-loss scale`; `Independent calibration samples` / `seal all candidate scores and choices`; `Held-out tasks / 20 new seeds` / `train every candidate, then assess regret`. Shortened to `Development tasks` / `score-to-loss scale`; `Calibration samples` / `sealed scores and choices`; `Held-out tasks` / `training and regret` | ADD TO CAPTION (the verbs, 'independent', '20 new seeds'); see B below |
| C | `Strong baselines beat the moment selector in both arms` (title) | caption already states it: 'The fixed / max-budget baseline (…) and the privileged rank-only baseline outperform the moment selector' (both arms are drawn and keyed) |
| D | `Selected budgets differ from the best` (title) | ADD TO CAPTION; see D below |
| D | `retrospectively best: K = rank` (label on the dashed line) | caption already states it: 'the gray dashed line the retrospectively best candidate's budget, which equals the rank in every task of both arms' |
| E | `Local agreement is stronger` (title) | ADD TO CAPTION; see E below |

## Caption additions (only if the proposal is applied)

- Replace the \textbf{B} sentence `\textbf{B}, Development, independent calibration, sealed choices and confirmation training.` with:
  `\textbf{B}, Protocol: development tasks fit the score-to-loss scale; independent calibration samples seal all candidate scores and choices; held-out tasks on 20 new seeds then train every candidate and assess regret.`
- Append after the \textbf{D} sentence (after '… which equals the rank in every task of both arms.'):
  `Mean selected budgets differ from it at every rank in both arms.`
  (Checked: feedback-only means 5.4, 5.0, 5.8, 3.8 and joint 6.2, 7.4, 5.9, 2.8 at ranks 1, 2, 4, 8.)
- Append after the \textbf{E} sentence (after '… ($n=20$ seed blocks).'):
  `Agreement is stronger for the first step (means 0.98 and 0.97) than for the final ranking (0.62 in both arms).`
  (Checked: first step 0.984 and 0.974, final 0.620 and 0.624. First step exceeds final in 20/20 seed blocks per arm.)

## Kept

- `balanced 1`–`balanced 4`, `comb`, `K = 1`, `K = 2`, `K = 4`, `K = 8`, and leaf indices `0`–`7` (A):
  schematic condition labels. The caption uses the same names.
- The three shortened protocol boxes and their arrows (B): terse noun-phrase labels, no verbs.
- `moment selector`, `rank-only`, `fixed / max-budget`, `random expectation` (C rows) and the key
  `feedback only`, `joint transfer` (C): category labels and a compact symbol key.
- `first-step test-loss` / `decrease`, `final test-loss` / `ranking` (E): category tick labels.
- Axis labels, now sentence case (first letter only): `Test loss + cost regret (excess over best trained candidate)`,
  `Task rank`, `Route budget K of selected candidate`, `Mean within-task Spearman ρ`.

## Layout changes (in the proposal)

- `canvas.panel(...)` is called without `title=`. The canvas geometry is unchanged (490 pt, row weights
  118/118/128, gutters 34/32 pt, margins). Every axes box keeps its size (A 220.79 × 119.27,
  B 173.25 × 119.27, C 428.04 × 120.27, D 220.79 × 120.26, E 173.25 × 120.26 pt).
  NativeCanvas re-seats the letters against the remaining ink.
- A: with the side note gone, the fifth bottom slot is empty. The K-cut trees stay under balanced 1–4.

## Checks (proposal PDF `si_pass/proposed/out/figure_S35_panels_A-F.pdf`, native-canvas manifest)

- `figure_canvas.py --audit --strict`: **0 violations**.
- `letter_ink_audit.py`: **total problems 0**.
- Supplement replay (`si_pass/sim_reflow.py` + `letter_ink_audit.py`): B and E moved −11.6 pt, the
  same as for the original sheet; 0 problems.
- `si_pass/proposed/compare_data.py s35`: all data marks and limits identical, every axes box the
  same size, and the text diff equals the table above.
- Build-time console `TEXT-ON-DATA` notes for the three protocol boxes (text on its own box)
  appear for the original too. The strict PDF audit does not flag them.
- Visual: `si_pass/renders/original_selector_proposed.png`. Clean.

## To apply (only if the coordinator accepts replacing the stale PNG)

`cp si_pass/proposed/build_supplementary_figure_s35_native.py scripts/` (diff alongside), then
`python3 scripts/build_supplementary_figure_s35_native.py`. Expected: PDF sha256 4c43767a…57093028
(518.4 × 490 pt). The PNG becomes 2e38ca64…4418d7. Re-register S35.
