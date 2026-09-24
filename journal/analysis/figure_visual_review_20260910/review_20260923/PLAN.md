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
