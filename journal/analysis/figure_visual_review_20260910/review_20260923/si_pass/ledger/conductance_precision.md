# conductance_precision: SI clarity pass, 2026-09-23

- Source key: N19 (whole sheet)
- Builder: `scripts/build_supplementary_figure_conductance_precision_native.py` (no argparse; `build()` writes the default output with `png=False`)
- Output: `figures/supplementary/figure_conductance_precision_native.pdf`
- Reproducibility gate: PASS. The unchanged builder reproduced the saved original bitwise (sha256 `5b54316d76e494380b1c7797c696f56084d444250f6457201708041d6a4535d2`). Original: `si_pass/renders/conductance_precision_orig.pdf`. Pre-edit builder: `si_pass/builders_before/conductance_precision.py` (from `0843de1^`, matches the edited file's base).
- After the edit: sha256 `3c38c61bcaf8deede94d2089d38d2c873427e0fde9cbb6e59b99cc6f3abca9ae`, and a second run gives the same bytes. Strict canvas audit: 0 violations. `letter_ink_audit.py`: 0 problems. Renders: `si_pass/renders/conductance_precision_{before,after}.png`.
- Data check (`si_pass/compare_drawings.py`): 698 of 698 data, axis and reference paths are unchanged. The only differences are in A's in-panel key: its "single seeds" hairline sample is gone, and its three rule handles moved one row down after the key title and that entry were removed.
- Registry note: the `figS19` rows of `source_data/provenance_manifest.tsv` and `original_assets.json` still carry the old sha256.

## Removed

- "4,096-update budget" (A and B, label on the dotted vertical line): caption already states it: "The dotted vertical line marks the original 4,096-update budget."
- "means, 95 % CI whiskers" (title of the A key): caption already states it: "Means and 95\% paired-seed bootstrap intervals summarize twenty fresh seeds".
- "single seeds (20 per rule)" (entry of the A key, together with its hairline sample): caption already states it: "thin lines show every trajectory", together with "twenty fresh seeds".
- "SGD, own y scale" (title of C's SGD facet): shortened to "SGD" (see Kept). Caption already states the removed part: "Adam and separately tuned SGD use different vertical scales".
- "no gap" (both C facets, label on the zero line): caption already states it: "The dashed line marks zero."
- "Path-field capture at exact-rule states" (D title): caption already states it: "\textbf{D}, At exact-rule validation-selected states, best rank-one capture of the six-compartment path field (dark red) and eligibility-weighted capture by the student's fixed initial profile (blue)."
- "= 1 by construction" (D, over the ungated rank-one marks): caption already states it: "Ungated path capture is one by construction."
- "95 % CI within the marker" / "for 3 of 4 means" (D): stated approximately: "symbols and bars show means and 95\% bootstrap intervals, mostly within the symbols". The exact count is not stated: ADD TO CAPTION (item 1). The builder still measures the count on the final axes box and asserts 3 of 4, so a later layout change cannot silently falsify the caption.
- "20 seeds per mean" (title of the D key): caption already states it: "Dots are twenty seeds".

## Caption additions

1. In the \textbf{D} block, insert this after "mostly within the symbols":
   ` (three of the four)`

## Kept

- "Ungated, independent inputs" and "Gated, conflicting inputs" (A, B titles): short condition labels. A and B share one axis design.
- A key "exact path", "unit broadcast", "initial profile": compact symbol keys (colour, dash and marker).
- C facet titles "Adam", "SGD": short condition labels for the two facets.
- C x tick "20/20 seeds > 0" in each facet: compact per-facet sign count, a form the house rules keep.
- C y label "Initial profile − exact, / gated-task test NMSE × 1000", now sentence case. It carries the ×1000 unit scaling.
- D key "best rank-one path capture" and "eligibility-weighted initial-profile capture": compact symbol keys. D ticks "ungated" and "gated conflict".
- Axis labels, now sentence case: "Training update", "Captured squared energy" and "Log10 test NMSE" (was "log10 test NMSE"). The subscript is not typeset for three reasons: mathtext is banned, the journal face (Nimbus Sans) has no subscript-digit glyphs, and no rotated chained-span helper exists. The caption writes $\log_{10}$.

## Layout changes

- None to the geometry: canvas 518.4 × 370 pt, row weights, gutters, margins, the uniform declared reserves, the panel boxes and the letters are all identical to the original.
- D's title removal needed no geometry change, because the C facet titles still set row 1's letter height.
- The A key is shorter (title and one entry removed) and still anchored lower left.
