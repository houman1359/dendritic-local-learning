# boolean_learning (N15) — SI text pass, group B

- Source key: N15
- Builder: `scripts/build_supplementary_figure_boolean_learning_native.py` (no argparse; run as `python3 scripts/build_supplementary_figure_boolean_learning_native.py`)
- Output: `figures/supplementary/figure_boolean_learning_native.pdf`
- Reproducibility gate: PASS. The unchanged builder rebuilt the output byte-identically (sha256 `c908da0b1b0dbbbced89a564f602f8bfa9430be4fa5618986d8cb95d5a2cca9f` for both). Original saved as `si_pass/renders/boolean_learning_orig.pdf`; renders `boolean_learning_before.png` / `boolean_learning_after.png`.
- Baseline note: the unchanged sheet already FAILED the letter audit: `D: letter top only -8.3 pt above panel ink (topmost: text '0.01')`. The colour rail's middle tick label sat in the row gap, nearest D's box. Fixed by this pass (see Layout changes).
- New PDF: strict audit 0 violations; `letter_ink_audit.py` total problems 0; render inspected. New sha256 prefix `6a2d0cff8b184772`.
- Data unchanged: the builder's data printouts ([E]–[H]) are identical to the gate run, and the vector-path census is identical to the original (811 paths, no differing kinds).

## Removed

| # | Panel | String (verbatim) | Status |
|---|---|---|---|
| 1 | E | `own y scale` (note in the credit subpanel) | caption already states it: 'the credit subpanel has its own y range' |
| 2 | E | `passes` (x tick label of the grouping subpanel, a verdict) | caption already states it: 'Grouping passes both criteria, credit does not' |
| 3 | E | `below margin` (x tick label of the credit subpanel, a verdict) | 'credit does not [pass]': caption already states it: 'Grouping passes both criteria, credit does not'. That its mean lies below the margin: ADD TO CAPTION |
| 4 | F | `Classification versus regression` (title) | caption already states it: 'Accuracy (gray circles) and balanced accuracy (blue crosses) versus NMSE for the 112 conditions' |
| 5 | F | `60/112 at 1.00` / `on both metrics` | caption already states it: 'sixty coincide at 1.00 on both' |
| 6 | G | `symlog y axis` (note in the SGD subpanel) | caption already states it: 'The SGD symmetric-log axis is linear within $\pm 0.002$' |
| 7 | H | `Broadcast gradients at own trained states` (title) | caption already states it: 'Broadcast gradient cosine on the aligned balanced tree' … 'at each learner's own state' |
| 8 | H | `cosine 1` (label on the amber dotted rule at y = 1) | caption already states it: 'amber dotted, cosine one' |

## Caption additions

1. **E sentence**: after `Grouping passes both criteria, credit does not`, insert:
   `, its mean lying below the margin`
   The sentence then reads: "Grouping passes both criteria, credit does not, its mean lying below the margin; both rules classify …"

## Kept

- A–D titles `Adam, exact credit, rate 0.003`, `Adam, broadcast credit, rate 0.01`, `SGD, exact credit, rate 0.03`, `SGD, broadcast credit, rate 0.01`: condition labels of four otherwise identical maps (the pass rules name this case as allowed).
- `Grouping contrast` / `Credit contrast` (E) and `Adam` / `SGD` (G): condition labels that tell the two subpanels apart.
- `0.01 margin` (×2, E): direct label of the amber dotted margin. The margin is a prespecified design value; it does not follow from the axis.
- F: `accuracy` and `balanced accuracy` (direct series labels); the star with `AND/OR majority reference` (compact symbol key, 3 words).
- H: `XOR(AND)` and `parity` (direct series labels); the `Adam` / `SGD` line-style key.
- `Clean NMSE`: colour-bar label (first word now capitalised).
- Tree labels (`ab|cd`, `ac|bd`, `ad|bc`, `a|(b|cd)`) and family row labels: categorical tick labels.

## Layout changes

- Removed the F and H titles. E and G keep theirs, so rows 2–3 keep their title band. All axes boxes, gutters, margins and the canvas (518.4 × 493 pt) are unchanged, and all letters sit where they did.
- E: both subpanels now have no x ticks (`set_xticks([])`). No geometry change, because row 2's bottom reserve is set by F's x label.
- Colour rail (fixes the baseline letter-audit failure): the rail and its label now span the 2 × 2 block together. The label `Clean` / `NMSE` has its foot on D's axes bottom, and the rail runs from 5 pt above the label to B's axes top. Before, the rail spanned the full block and the label hung below D's bottom. The rail's middle tick label `0.01` now stands beside the top row (nearest B), so letter D is clear of all its ink. The 5 pt gap also clears a pre-existing bbox overlap between the `0.0001` tick label and `Clean`. Rail ticks, norm, colour map and width are unchanged; the rail is 17 pt shorter.
- Axis labels changed to sentence case: `Clean population NMSE`, `Threshold performance`, `Common rate` (×2), `Broadcast − exact NMSE`, `Training update`, `Population-gradient cosine`.
- Updated the module docstring. (`MARKER_MS` was already an unused import before this pass; left untouched.)
