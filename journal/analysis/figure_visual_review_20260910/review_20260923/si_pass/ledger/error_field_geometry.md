# error_field_geometry — SI text pass ledger (2026-09-23)

- Source key: N8 (whole sheet, `('N8','*')`)
- Builder: `scripts/build_supplementary_figure_error_field_geometry_native.py` (run as `python3 <builder>`, no arguments)
- Output: `figures/supplementary/figure_error_field_geometry_native.pdf`
- Reproducibility gate: PASS. The unchanged builder run with `python3` (Mambaforge, matplotlib 3.10.6) gave a byte-identical PDF (sha256 19ecb97c…ffa6e6). Original at `si_pass/renders/error_field_geometry_orig.pdf`, renders at `si_pass/renders/error_field_geometry_{before,after}.png`, builder backup in `si_pass/builder_backup/`.
- Edited: yes. New sha256 10321a6b…4637, 518.4 × 364 pt (size unchanged).
- Verification: `figure_canvas.py --audit --strict` 0 violations; `letter_ink_audit.py` 0 problems (smallest gaps: top 6.5 pt, left 4.3 pt).
- Data check: the builder's printed plotted values (all means, intervals and whisker lengths) match the original builder's output line for line. Vector paths in A, B and C are identical. Only the key's handles changed: the whisker and dashed-rule handles were removed and the one-seed handle moved up.

## Removed

| # | Where | String removed (verbatim) | Caption status |
|---|---|---|---|
| 1 | A | `Direction: gradient cosine` (title) | Already in the caption: "\textbf{A}, Branch-gradient cosine with the exact gradient at common fixed checkpoints trained with per-neuron feedback" |
| 2 | B | `Amplitude: transport by depth` (title) | Already in the caption: "\textbf{B}, Exact voltage-error magnitude by depth at exact-path checkpoints" |
| 3 | C | `Variation: within-depth residual` (title) | Already in the caption: "\textbf{C}, Within-depth path-specific residual error energy at the exact-path checkpoints of \textbf{B}" |
| 4 | A | `0 = no alignment` (label on the dashed rule; the rule stays) | Already in the caption: "The dashed grey rule in \textbf{A} marks cosine zero" |
| 5 | A | ` at common checkpoints` (x label `feedback field at common checkpoints` → `Feedback field`) | Already in the caption: "at common fixed checkpoints trained with per-neuron feedback" |
| 6 | B | `95% seed-bootstrap intervals` / `≤ 0.08 ratio units: within` / `the mean symbols, not drawn` (in-panel note) | Already in the caption: "in \textbf{B} every interval is at most 0.08 ratio units, within the symbol, and none is drawn" |
| 7 | B | `soma = 1 by normalization` (label on the dashed rule; the rule stays) | Already in the caption: "with somatic value one by normalization, marked by the dashed grey rule at one from which every per-seed profile starts" |
| 8 | B | `, soma = 1` (y label `batch-RMS ratio, soma = 1`, reworded to `Batch-RMS ratio to soma`) | Already in the caption: "the batch root-mean-square ratio $\operatorname{RMS}(\delta^V_{n,u})/\operatorname{RMS}(\delta^V_{0,u})$ … with somatic value one by normalization" |
| 9 | key | `Key` (heading) | A heading with no content; see caption edit 2. |
| 10 | key | `whisker: 95% seed-bootstrap interval` / `of the mean (10,000 whole-seed draws)` (entry and its whisker glyph) | Already in the caption: "whiskers in the series colour are 95\% seed-bootstrap intervals of the mean (10,000 whole-seed draws)" |
| 11 | key | `thin line, small mark: ` and `(15 paired seeds per architecture)` (entry `thin line, small mark: one seed` / `(15 paired seeds per architecture)` → `one seed`) | Already in the caption: "the thin lines with small marks are the fifteen paired seeds per architecture" |
| 12 | key | `dashed rule: reference constant` (entry and its dashed glyph) | Already in the caption: "The dashed grey rule in \textbf{A} marks cosine zero" and, for \textbf{B}, "marked by the dashed grey rule at one" |
| 13 | key cell | `Cohort: 128 directed [3,3] trees per network, 15 paired seeds` / `per architecture, flattened MNIST; A at checkpoints trained with` / `per-neuron feedback, B and C at exact-path checkpoints.` / `Exact-path cosine in A is 1 by construction and is not drawn.` | Mostly already in the caption: "Independent fifteen-seed flattened-MNIST cohort of 128 directed $[3,3]$ trees", "fifteen paired seeds per architecture", "\textbf{A} … trained with per-neuron feedback", "\textbf{B} … at exact-path checkpoints", "\textbf{C} … at the exact-path checkpoints of \textbf{B}", "the exact-path cosine is one by construction and is not drawn". Only "per network" is missing. **ADD TO CAPTION** (edit 1) |

## Caption additions

Edits to the `error_field_geometry` caption (5th element of its `FIGURES` tuple):

1. In the opening sentence, replace `cohort of 128 directed $[3,3]$ trees;` with `cohort of 128 directed $[3,3]$ trees per network;`
2. The key now names only the seed-mean and single-seed glyphs; whiskers and dashed rules are described in the caption alone. Replace `; the key on the sheet names each glyph.` with `; the key names the seed-mean and single-seed glyphs.`

## Text re-cased (sentence case)

`branch-gradient cosine` / `to exact gradient` → `Branch-gradient cosine` / `to exact gradient`; `depth` → `Depth` (B, C); `path-specific error energy (%)` → `Path-specific error energy (%)`. The rewordings of the A x label and B y label are listed under Removed (5 and 8).

## Kept

- Key entries `shunting: seed mean`, `raw additive: seed mean` and `one seed`: compact symbol keys. The caption relies on the sheet's key for the glyphs.
- Tick labels `matched-width` / `fallback`, `neuron-` / `specific`, `soma`, `mid`, `distal`: categories.
- The unlabelled dashed reference rules in A (cosine 0) and B (ratio 1): data-frame references the caption names.

## Layout changes

- Canvas, margins, gutters and every axes box are unchanged (518.4 × 364 pt; A/B/C 195.2 × 125.5 pt).
- I did not keep a narrower gutter. At 50 pt, A's x label "Feedback field" sat as close to C's axes as to A's, and the independent letter audit assigned it to C (C: -19.8 pt).
- Letters A/B moved from baseline 13.3 to 18.5 pt from the top (their titles are gone). C moved from 192.8 to 198.0 pt. x positions are unchanged (15.1, 272.2).
- The key legend has three entries and sits at the top-left of the row-1 right-hand cell, as before. The heading and cohort paragraph below it are gone.
- Consolidation: the N8 letter positions changed, so the N8 entry in `original_assets.json` must be refreshed by the coordinator. I did not run `refresh_native_assets.py`.
