# Final main-figure and new-control visual audit

Scope: canonical main Figures 1–9, Supplementary Figures S48–S51, their current captions, and actual manuscript placement for the main figures. All panels were rendered and inspected, including the final shunt C/D swap. No scientific outcomes were modified.

## Findings and actions

- Main Figures 2–9 are readable at the current full-width placement. Normal native text is at least 6.8 pt (Figure 4 begins at 7.2 pt); smaller glyphs are mathematical subscripts. With the current 0.78-inch manuscript margins, the 518.4-pt native canvases are placed at approximately 0.964 scale, giving a minimum ordinary printed size of 6.55 pt. The smallest annotations remain readable; panel letters, ordinary labels and line styles are consistent across the sequence.
- Panel alignment, inset labels, axis titles, legends and caption placement were checked visually. No clipped labels, overlapping text, or caption/figure collisions were found. The full-width ancestry contrast and electrotonic boundary retain the prominence requested in the review.
- Main Figure 6A had one remaining near-invisible condition: the broadcast and shared-soma LocalCA accuracy curves differ by at most 0.0629997 percentage points across D1–D3. Both curves remain plotted, and the legend now names their near overlap with a paired symbol, matching the existing treatment in Figure 6B. The data, axes, intervals and other panels are unchanged. Updated `scripts/physical_depth_budget/build_main_figure5.py`, its native outputs and canonical `figure_06.pdf`. The native audit reports no text outside the canvas and a minimum ordinary size of 6.8 pt.
- Main Figure 8 now has the intended narrative order: A ancestry-partition mechanism; B tree-relation selectivity; C adjoint factor freezing; D signed physical calibration; E electrotonic boundary. The figure was not overwritten during this audit. Its caption matches the swapped C/D panels and distinguishes attenuation amplitude from spatial selectivity.
- Main Figure 5 correctly distinguishes fixed-checkpoint learning curves (C,D) from validation-selected paired contrasts (E), and compares gradient estimates at identical weights and examples in F. Its oracle three-profile coefficients, binary context and joint unit/calibrated legend are explicit. No necessity claim is made about the number of independent external errors.
- Supplementary S50 and S51 were compiled together using the actual SI preamble. Each fits its own page with its caption, with no overfull or underfull boxes. Their normal native labels begin at 7.2 pt, corresponding to approximately 6.86 pt at the current SI full width. S48 retains its exact 3.6-inch native placement, preserving its minimum 7.2-pt labels.
- Two S51 label changes were applied by the owning conductance agent and verified in the canonical PDF: replace “SGD also needs context-resolved credit” with “SGD also favors routed credit”; add `(NMSE)` to the panel B interaction axis. The captions already state the finite budget, fixed rate, wider-bound restarts, pre-bound comparison, and retained development regimes accurately.

## Native geometry and ordinary type

| Asset | Native width × height (pt) | Minimum ordinary type (pt) |
|---|---:|---:|
| Main 1 | 518.4 × 490 | 6.8 |
| Main 2 | 518.4 × 450 | 6.8 |
| Main 3 | 518.4 × 556 | 6.8 |
| Main 4 | 518.4 × 490 | 7.2 |
| Main 5 | 518.4 × 490 | 6.8 |
| Main 6 | 518.4 × 529.2 | 6.8 |
| Main 7 | 518.4 × 526 | 6.8 |
| Main 8 | 518.4 × 490 | 6.8 |
| Main 9 | 518.4 × 472 | 6.8 |
| S48 | 259.2 × 194.4 | 7.2 |
| S49 | 518.4 × 504 | 6.8 |
| S50 | 518.4 × 425 | 7.2 |
| S51 | 518.4 × 430 | 7.2 |

The local review renders and two-page SI preflight are in `/tmp/credit_final_visual_audit/`; these are review artifacts rather than release contents. The S51 label changes have been verified in the canonical PDF. The six-arm Figure 1 check is completed below.

## Completed S49 placement check

The final six-arm S49 was inspected at its actual placement on SI page 120. Figure and complete caption fit cleanly on one page, and the revised two-line arm labels remain readable. The minimum ordinary native label is 6.8 pt; smaller spans are logarithmic superscripts. Its caption distinguishes selected/common learning rates, activation-space capture, and equal common-mode means from unequal field norms. No caption collision or clipped label was found.

## Completed fresh Figure 1

The final Figure 1 uses all 120 selected-rate fresh outcomes and the final round-trip-parsed summary tables. Its source means and four paired-resolution contrast means are recomputed as builder assertions. Panels C/D retain the fresh ten-seed cohort; panel E contains only separately labeled legacy DFA/CIFAR controls. K=1/3/12 count nonsomatic spatial profiles, with an unchanged separate exact somatic error in the projected/exact conditions. The decoder-only reference is visually separated.

The native audit reports zero findings after aligning the two lower axes and expanding the eligibility schematic within its original cell; ordinary fonts were not reduced. The final caption was synchronized from root-owned main.tex, and an isolated compile under the actual main preamble fits the entire figure and caption on one page without float-size or box warnings. The rendered page was inspected for legend readability, label collisions and caption contact; none were found. Final canonical PDF SHA-256: `e45d1005056ab239dd68bd7156aa51823352d6d74d3ecab6e8cc5b70073a121f`.
