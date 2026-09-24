# fixed_profile_budget: SI clarity pass, 2026-09-23

- Source key: N16 (whole sheet)
- Builder: `scripts/build_supplementary_figure_fixed_profile_budget_native.py` (no argparse; `build()` writes the default output)
- Output: `figures/supplementary/figure_fixed_profile_budget_native.pdf`
- Reproducibility gate: PASS. The unchanged builder reproduced the saved original bitwise (sha256 `703c18620bb20b716fa400e53fe89232799dc781ca20c00c39d118e250c9713f`). Original: `si_pass/renders/fixed_profile_budget_orig.pdf`. Pre-edit builder: `si_pass/builders_before/fixed_profile_budget.py`, which is `git show 0843de1^:journal/scripts/...` and matches the file that was edited.
- After the edit: sha256 `b8c7ac4be994162251528567b933ce109460a374ad90f71afae0e69c9dd68a12`, and a second run gives the same bytes. `figure_canvas.py --audit --strict` reports 0 violations. `letter_ink_audit.py` reports 0 problems. Renders: `si_pass/renders/fixed_profile_budget_{before,after}.png`.
- Data check (`si_pass/compare_drawings.py`): 1,567 of 1,567 data, axis and reference paths are unchanged. The two differences are expected: the leader line under "unchanged error" is gone, and the 12 shared-key handles moved 22 pt right because the key is centred and two labels are now shorter.
- Registry note: the `figS16` rows of `source_data/provenance_manifest.tsv` and `scripts/supplement_consolidation/original_assets.json` still carry the old sha256.

## Removed

Current caption: the 5th element of the `fixed_profile_budget` tuple in `specification.py`.

- "Pairwise: rates selected per rule" (A title): reworded to "Pairwise, selected rates" (see Kept). Caption already states it: "Pairwise-target learning under Adam at development-selected and common rates".
- "Pairwise: common rate 0.003" (B title): reworded to "Pairwise, common rate 0.003" (colon to comma).
- "Quartic: selected rates and common rate" (C title): reworded to "Quartic, selected and common rates".
- "Pairwise at 16,384 updates: every seed" (D title): caption already states it: "\textbf{D}, Pairwise NMSE at 16,384 updates: dots, twenty seeds".
- "n = 20 seeds per condition" (D): caption already states it: "dots, twenty seeds".
- "Credit deficit over budget" (E title): caption already states it: "\textbf{E}, Quartic-minus-pairwise difference in initial-profile-minus-exact NMSE across four budgets".
- "positive in 20/20 seeds" / "at every budget" (E): caption already states it: "every difference is positive in all twenty seeds". The builder still asserts 20/20 positive for all 16 means.
- "zero deficit" (E, label on the grey zero line): ADD TO CAPTION (item 1). The line lies on the axis floor at the 0.0 tick, so item 1 is optional.
- "Quartic seeds, two budgets" (F title): caption already states it: "\textbf{F}, Quartic errors at 1,024 versus 16,384 updates for all twenty exact-path and twenty initial-profile seeds".
- "n = 20 seeds per rule" (F): caption already states it: "for all twenty exact-path and twenty initial-profile seeds".
- "3/20 exact seeds" / "stalled at 1,024" (F): ADD TO CAPTION (item 2). The builder still asserts exactly three exact seeds above 0.5 NMSE at 1,024 updates.
- "unchanged error", together with its grey leader line (F): caption already states it: "The diagonal marks unchanged error".
- "Exact seeds at the floor, 20×" (title of the right view of F): ADD TO CAPTION (item 3). The caption already has "The right view resolves fits near that floor" but not the magnification or that only exact seeds are drawn.
- "16/20 exact seeds" / "within 0.043–0.050" (right view of F): ADD TO CAPTION (item 3). The builder still asserts 16 of 20 exact seeds inside the window.
- "noise-only 0.045" (right view of F, floor label): caption already states it: "grey guides mark the noise floor" and "(0.0225 pairwise, 0.045 quartic)".
- Shared key "thick (C): rates selected per rule": shortened to "selected rates (C)". Caption already states it: "\textbf{C}, Quartic learning at selected (thick) and common (thin) rates".
- Shared key "thin (C): common rate 0.003": shortened to "common rate (C)". Caption already states it: "the common rate is 0.003", plus the C sentence above.

## Caption additions

1. In the \textbf{E} block, after "every difference is positive in all twenty seeds." add this sentence (optional, see Removed):
   `The grey line at zero marks no deficit.`
2. In the \textbf{F} block, insert this after "Exact seeds with high early error":
   ` (three of twenty, above 0.5 NMSE at 1,024 updates)`
   The sentence then reads "Exact seeds with high early error (three of twenty, above 0.5 NMSE at 1,024 updates) reach the floor by the longer budget."
3. In the \textbf{F} block, insert this after "The right view":
   ` (linear axes, magnified about $23\times$; open circles, the sixteen exact seeds within 0.043--0.050)`
   The artwork said "20×". Measured against the log axes of the left view, the right view magnifies 23.1× at the 0.045 floor, 22.2× at 0.0432 and 25.9× at 0.0504. Write "about $20\times$" if the old wording is preferred.

## Kept

- "Pairwise, selected rates", "Pairwise, common rate 0.003", "Quartic, selected and common rates" (A–C titles): short condition labels. A–C share one axis design and differ only in task or rate view. Each is 5 words or fewer, has no verb and states no finding.
- D category ticks "exact path", "unit broadcast", "profile / sign rate 0.01", "profile / sign rate 0.003": tick labels. The rate is the only thing that separates the two profile/sign columns.
- E legend "selected rates, terminal", "selected rates, validation", "common rate, terminal", "common rate, validation": compact symbol key for colour and marker/dash.
- Shared key "exact path", "unit broadcast", "initial profile", "initial sign", "1,024-update cap", "noise-only NMSE", "selected rates (C)", "common rate (C)": compact symbol keys. "(C)" limits the thick/thin samples to the one panel that uses them.
- Axis labels, now sentence case: "Training update", "Test NMSE", "Test NMSE at 16,384 updates", "Maximum training updates", "Quartic − pairwise credit deficit", "Test NMSE at 1,024 updates".

## Layout changes

- No change to NativeCanvas geometry: canvas 518.4 × 493 pt, rows, gutters, margins and reserves are the same. Every panel axes box is identical to the original (checked in the manifest).
- Removing the D, E, F and zoom titles needed no geometry change. The row-2 top reserve is set by the letter band, not the title. The row-2 letters E and F were re-placed automatically and sit 5.2 pt lower. The strict audit found no blank band, aspect or fill violation.
- The shared key is re-centred (handles +22 pt) because two labels are shorter.
