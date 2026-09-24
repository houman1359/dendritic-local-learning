# Main-figure quality pass — 23 September 2026

Baseline: paper commit e264c1f; builders snapshotted in `before/`.

## House rules applied to every main figure

1. **Letters.** 9 pt bold, top-left of each panel. Letter top at least ~5 pt above the
   panel's topmost ink and right edge at least ~3 pt left of its leftmost ink, measured on
   the PDF (`letter_ink_audit.py`, not the builder's recorded bounds). Letters in one grid
   row share a baseline; letters in one grid column share an x. Where layouts allow, panels
   in successive rows start at the same grid column so letter columns line up. Schematic
   panels fill their slots so letters do not float far from their content.
2. **No prose in the artwork.** Removed: headline titles over schematics, explanatory
   sentences and footers, sample counts and identifiers, provenance notes, legends that
   restate the caption, and reference-line labels such as "no difference" or "no effect"
   whose meaning follows from the axis. Kept: axis labels, tick labels, direct series
   labels, compact symbol keys, short condition labels that distinguish panels, and the
   few effect-size annotations that carry the result. Anything removed that the legend did
   not already state moves into the legend.
3. **Typography.** Axis labels in sentence case; Greek, subscripts and superscripts drawn
   as 7 pt token spans on a common baseline (mathtext is banned by the canvas audit).
4. **Gates.** Strict canvas audit, `letter_ink_audit.py`, row separation, panel gaps,
   letter geometry, legend word limit (350), caption-constant tests, full test suite.

## Supplementary pass (same rules, 37 sheets)

- **Native sheets.** 23 builder scripts edited (titles, footers, counts and
  reference-line names removed; keys compacted; letter clashes fixed at source).
  N20 (conductance_optimization) and N36 (morphology_estimation) were redrawn so no
  sheet-wide title or key sits above a letter row. The 25 ledgers in `si_pass/ledger/`
  list every removed string and where the legend states it.
- **Frozen sources.** Registered renders that cannot be rebuilt keep their bytes; the
  consolidation build removes or rewords their panel text through
  `PANEL_TEXT_EDITS` in `scripts/supplement_consolidation/specification.py` (21 sheets).
- **Letters.** `letter_relocation.py` runs on every consolidated sheet; whole-source
  sheets use the native manifest's letter boxes (`native_regions`). After this pass it
  moves nothing: `letter_ink_audit.py` reports 0 problems on all compiled sheets.
- **Legends.** About fifty legend edits carry the removed facts. Three legends then
  overflowed their pages (S14 by 10.9 pt, S18 by 1.9 pt, S37 by 35.0 pt); they were
  shortened by removing phrases that repeat axis titles, tick labels or keys, keeping
  every fact the ledgers required. `audit_latex_layout.py` passes.
