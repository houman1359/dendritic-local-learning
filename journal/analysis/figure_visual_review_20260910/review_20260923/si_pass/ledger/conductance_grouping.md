# conductance_grouping: SI clarity pass, 2026-09-23

- Source key: N18 (whole sheet)
- Builder: `scripts/build_supplementary_figure_conductance_grouping_native.py` (no argparse; `build()` writes the default output)
- Output: `figures/supplementary/figure_conductance_grouping_native.pdf`
- Reproducibility gate: PASS. The unchanged builder reproduced the saved original bitwise (sha256 `67062a920cfd48c0eea9fe59691741a8ba9dc2e9b51b700adbb85fc33a60b651`). Original: `si_pass/renders/conductance_grouping_orig.pdf`. Pre-edit builder: `si_pass/builders_before/conductance_grouping.py` (from `0843de1^`, matches the edited file's base).
- After the edit: sha256 `42d001ea534d656919821b649874b0986bb05c2697037c0207d57e1e556d3067`, and a second run gives the same bytes. Strict canvas audit: 0 violations. `letter_ink_audit.py`: 0 problems. Renders: `si_pass/renders/conductance_grouping_{before,after}.png`.
- Data check (`si_pass/compare_drawings.py`): 985 of 987 paths are unchanged, including every data mark, the schematic trees and the reference lines. The other two are the inhibitory-contact glyph and the coupling sample of A's glyph key, which moved left because the key labels are shorter.
- Registry note: the `figS18` rows of `source_data/provenance_manifest.tsv` and `original_assets.json` still carry the old sha256.

## Removed

- "Fixed physical shape; three input groupings" (A title): caption already states it: "\textbf{A}, Seven-compartment directed shunting model (one physical shape; ...) under the groupings $01|23$, $02|13$ and $03|12$".
- "excitatory contact ×4", "inhibitory contact ×6", "coupling ×6" (A glyph key): the counts were dropped and the labels kept (see Kept). Caption already states it: "sixteen positive conductances: four excitatory and six inhibitory contacts, six couplings".
- "Interaction bound" (B title): caption already states it: "\textbf{B}, Analytic population lower bound from interactions crossing the student's two proximal input blocks".
- "0 exactly," / "all 20 seeds" (B, compatible column): caption already states it: "compatible groupings are exactly zero in all twenty seeds (superimposed)".
- "Adam: test error by credit rule" and "SGD: test error by credit rule" (C, D titles): shortened to "Adam" and "SGD" (see Kept). Caption already states the removed part: "Held-out sample NMSE at checkpoints ... under Adam and stochastic gradient descent (SGD) for exact path (dark red circles), ...".
- "(all four rules)" (second line of the "incompatible" band label in C and D): caption already states it: "grey band, the union of the four rules' 95\% intervals for the incompatible groupings".
- "Compatible trees: paired credit effect" (E title): caption already states it: "\textbf{E}, Paired compatible-tree NMSE differences from exact path at 1,000 updates for the three other rules under Adam and SGD".
- "Adam: common-state gradient alignment" (F title): "Adam" and "gradient alignment" are already stated: "\textbf{F}, Aggregate gradient cosine on 256 calibration examples, compatible Adam models, same checkpoints". "Common-state" is not: ADD TO CAPTION (item 1). The source README says "Credit geometry is measured at common exact-trained checkpoint states", and every diagnostics row has `state == common_exact_state`.
- "exact path = 1 by construction" (F, label on the dashed reference): caption already states it: "exact path is 1 by construction, drawn as the dashed reference".

## Caption additions

1. In the \textbf{F} block, insert this after "compatible Adam models, same checkpoints":
   ` (common exact-trained states)`
   The sentence then reads "... compatible Adam models, same checkpoints (common exact-trained states): dots, the twenty seed means; ..."

## Kept

- "01 | 23", "02 | 13", "03 | 12" (A): grouping labels that distinguish the three otherwise identical trees. Leaf indices and the δ0 error label (a chained-span subscript, not mathtext) are kept too.
- A glyph key "excitatory contact", "inhibitory contact", "coupling": compact symbol key.
- B y label "Population NMSE / lower bound (quadrature)", now sentence case: an axis label. "(quadrature)" says what the plotted quantity is; the 2026-09-12 review moved that caveat onto the axis. For the coordinator: the caption also states it ("converged quadrature, not a certified bound"), so the label could shrink to "Population NMSE lower bound" if preferred.
- B x label "Student grouping vs task", with ticks "incompatible" and "compatible".
- C, D titles "Adam", "SGD": short condition labels. C and D share one axis design.
- C, D direct labels "incompatible" (grey band) and "compatible" (curves): direct series labels.
- E y label "Credit rule − exact path / test NMSE", with ticks "Adam" and "SGD". F y label "Calibration-gradient / cosine". C, D, F x label "Training update". C, D y label "Test NMSE". All are now sentence case.
- Shared key "exact path", "fixed calibrated broadcast", "one oracle profile", "two subtree profiles (oracle)": compact symbol keys of 4 words or fewer.

## Layout changes

- None to the geometry: canvas 518.4 × 484 pt, row weights, gutters, margins and every panel axes box are identical to the original. A schematic panel still passes cell-fill.
- The builder's hand-set lift for the row-2 titles is removed, because no row-2 title remains (it was `set_title(..., pad=11.0)` for E and F).
- Panel letters were re-placed automatically against the remaining ink. Row 0 (A, B) moved 5.2 pt down and row 2 (E, F) moved 13.2 pt down. Rows and columns stay aligned. The row-2 letters now sit about 15 pt below the C/D x labels. The strict audit found no blank band.
- A's glyph key entries sit closer together because the count suffixes are gone.
