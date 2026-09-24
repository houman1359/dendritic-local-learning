# credit_optimizer_controls: SI clarity pass, 2026-09-23

- Source key: N17 (whole sheet)
- Builder: `scripts/build_supplementary_figure_credit_optimizer_controls_native.py` (no argparse; `build()` writes the default output)
- Output: `figures/supplementary/figure_credit_optimizer_controls_native.pdf`
- Reproducibility gate: PASS. The unchanged builder reproduced the saved original bitwise (sha256 `193cf1d65c9fc6bb42cd40c51b658b32277cf94c02d8c149985c1d17c2a5e8f7`). Original: `si_pass/renders/credit_optimizer_controls_orig.pdf`. Pre-edit builder: `si_pass/builders_before/credit_optimizer_controls.py` (from `0843de1^`, matches the edited file's base).
- After the edit: sha256 `80d1f4725c4e0ff24aa9549a66e2295627190439077241bdfd4594b9aa983b15`, and a second run gives the same bytes. Strict canvas audit: 0 violations. `letter_ink_audit.py`: 0 problems. Renders: `si_pass/renders/credit_optimizer_controls_{before,after}.png`.
- Data check (`si_pass/compare_drawings.py`): 374 of 374 data, axis and reference paths are unchanged. The only differences are the 6 shared-key handles, which moved when the centred key was re-laid out.
- Registry note: the `figS17` rows of `source_data/provenance_manifest.tsv` and `original_assets.json` still carry the old sha256.

## Removed

- "Nested targets, Adam, selected rates" (A title): shortened to "Nested, Adam, selected rates" (see Kept). Caption already states "targets": "Nested-target control from the main text, under Adam at selected rates".
- "Pairwise targets, SGD, common rate 0.03" (B title): shortened to "Pairwise, SGD, common rate 0.03", because the original was 6 words. Caption already states it: "Pairwise targets under SGD at common rate 0.03".
- "Quartic targets, SGD, selected rates" (C title): shortened to "Quartic, SGD, selected rates". Caption already states it: "Quartic targets under SGD at selected rates".
- "Quartic targets, SGD, common rate 0.03" (D title): shortened to "Quartic, SGD, common rate 0.03".
- "3 broadcast controls within 6 %" / "after 1,024 updates" (A): ADD TO CAPTION (item 1). The caption says only "the three broadcast curves nearly coincide in \textbf{A}". The builder still asserts a 5–6 % spread; the measured value is 5.6 %.
- "noise-only NMSE 0.0225" (A and B floor labels) and "noise-only NMSE 0.045" (C floor label): caption already states it: "grey horizontal lines mark noise-only NMSE (0.0225 for nested/pairwise, 0.045 for quartic)".
- "unit broadcast:" / "2/20 seeds end above 1" (B): the count is already stated: "two unit-broadcast seeds diverge while eighteen finish near the noise floor". The threshold is not: ADD TO CAPTION (item 3). The final values are 2.28 and 4.16.
- "initial profile:" / "5/20 seeds end above 0.05" (B): the count is already stated: "five initial-profile seeds also retain larger errors". The threshold is not: ADD TO CAPTION (item 4).
- "3 broadcast controls" / "0.94–1.09, none diverges" (C): the caption says only "remain near one in \textbf{C}". ADD TO CAPTION (item 2). The builder still asserts the 0.94–1.09 mean range. It now also asserts that every seed of every C rule stays below NMSE 2 (maximum 1.930, unit broadcast).
- "unit broadcast:" / "18/20 seeds end above 1" and "initial sign:" / "15/20 seeds end above 1" (D): caption already states it: "In \textbf{D}, eighteen unit-broadcast and fifteen initial-sign seeds exceed NMSE one". The "end" (the 16,384-update checkpoint) is not stated: ADD TO CAPTION (item 5).
- "exact path and initial profile:" / "rate 0.03 in C and D" / "(same runs), drawn in C" (D): caption already states it: "Exact and initial-profile runs are identical across these views and appear only in \textbf{C}", together with "(\textbf{C}: 0.03 for exact and initial-profile credit; ...) and common rate 0.03 (\textbf{D})".
- Key "initial profile (calibrated broadcast)": shortened to "initial profile". Caption already states it: "blue solid, initial-profile (calibrated) broadcast".
- Key "noise-only NMSE (value printed)": shortened to "noise-only NMSE", since the values are no longer printed. The caption gives both values (see above).

## Caption additions

The last four items go in one sentence of the current caption: "By contrast, the three broadcast curves nearly coincide in \textbf{A} and remain near one in \textbf{C}; ..."

1. Insert this after "the three broadcast curves nearly coincide in \textbf{A}":
   ` (means within 6\% of one another after 1,024 updates)`
2. Insert this after "and remain near one in \textbf{C}":
   ` (means 0.94--1.09; no seed exceeds NMSE 2)`
3. Insert this after "two unit-broadcast seeds diverge":
   ` (final NMSE above 1)`
4. Insert this after "five initial-profile seeds also retain larger errors":
   ` (final NMSE above 0.05)`
5. Insert this after "seeds exceed NMSE one", in the \textbf{D} sentence:
   ` at 16,384 updates`

## Kept

- "Nested, Adam, selected rates", "Pairwise, SGD, common rate 0.03", "Quartic, SGD, selected rates", "Quartic, SGD, common rate 0.03" (titles): short condition labels (5 words or fewer, no verbs). The four panels share one axis design, and these labels are what distinguish them.
- Key "exact path", "unit broadcast", "initial profile", "initial-sign broadcast", "noise-only NMSE", "original 1,024-update cap": compact symbol keys (colour and dash).
- Axis labels, now sentence case: "Training updates", "Test NMSE".

## Layout changes

- None to the geometry: canvas 518.4 × 470 pt, rows, gutters, margins, panel boxes and letters are identical to the original.
- The axis limits are deliberately unchanged. A–C keep the strip under the noise rule that used to hold the printed floor value, and C keeps the head-room that held its control note. Changing them would move data marks.
- The shared key is re-centred because two labels are shorter.
