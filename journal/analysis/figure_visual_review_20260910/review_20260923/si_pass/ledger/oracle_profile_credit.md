# oracle_profile_credit (N13) — SI text pass, group B

- Source key: N13
- Builder: `scripts/build_supplementary_figure_oracle_profile_credit_native.py` (no argparse; run as `python3 scripts/build_supplementary_figure_oracle_profile_credit_native.py`)
- Output: `figures/supplementary/figure_oracle_profile_credit_native.pdf`
- Reproducibility gate: PASS. The unchanged builder rebuilt the output byte-identically (sha256 `992dd2dec2079a955dbceb3a2fec752c759f9f36c6e768862172f91182adaeff` for both). Original saved as `si_pass/renders/oracle_profile_credit_orig.pdf`; renders `oracle_profile_credit_before.png` / `oracle_profile_credit_after.png`.
- New PDF: strict audit 0 violations; `letter_ink_audit.py` total problems 0; render inspected. New sha256 prefix `cc7289737ba862d7`.
- Data unchanged: the builder's data printouts (bootstrap replays, [A/B]–[E/F] lines) are identical to the gate run, and the vector-path census is identical to the original (2,290 paths, no differing kinds).

## Removed

| # | Panel | String (verbatim) | Status |
|---|---|---|---|
| 1 | A | `Adam, compatible trees: final test error`. Title shortened to the condition label `Adam`; removed part: `compatible trees: final test error`. | caption already states it: 'Final test normalized mean squared error (NMSE) on oracle-compatible trees under Adam and stochastic gradient descent (SGD), respectively.' |
| 2 | B | `SGD, compatible trees: final test error`, shortened to `SGD` | caption already states it (same quote as #1) |
| 3 | C | `Exact credit: input assignment matters` (title) | caption already states it: 'Paired exact-credit NMSE increase after reassigning the leaf inputs' |
| 4 | C | `[0.507, 0.517]` (interval printed under the Adam quartic point) | caption already states it: 'the Adam quartic interval (0.507--0.517)'. The caption's clause 'is printed beside the point' becomes false, so a caption EDIT is needed (see below). |
| 5 | D | `Adam, compatible trees: gradient alignment` (title) | 'Adam' and 'gradient alignment': caption already states it: 'Adam gradient cosine between the exact and delivered aggregate clean-population updates'. 'compatible trees': ADD TO CAPTION (the D sentence does not name the arm). |
| 6 | D | `exact path = 1 by definition` (label on the dashed reference) | caption already states it: 'the exact-path cosine is 1 by definition and is the dashed red-brown reference, not a series' |
| 7 | E | `Adam: all three learning rates`. Title shortened to the condition label `Adam`; removed part: `all three learning rates`. | caption already states it: 'Final NMSE at the three tested learning rates 0.003, 0.01 and 0.03' |
| 8 | F | `SGD: all three learning rates`, shortened to `SGD` | caption already states it (same quote as #7) |

## Caption additions

1. **D sentence**: after `\textbf{D}, Adam gradient cosine`, insert:
   ` on oracle-compatible trees`
2. **C sentence, edit (required, not only an addition)**: replace
   `and the Adam quartic interval (0.507--0.517), narrower than its marker, is printed beside the point.`
   with
   `and the Adam quartic interval (0.507--0.517) is narrower than its marker.`

## Kept

- Titles `Adam` / `SGD` on A/B and on E/F: condition labels that tell the otherwise identical panel pairs apart. The E,F caption sentence does not say which of E and F is Adam, so these labels carry that mapping.
- `20/20 > 0` (×5) and `19/20 > 0`: compact per-strip sign counts (the caption refers to them: 'the number of positive differences out of twenty is printed above each strip').
- Shared key entries `exact path`, `root broadcast`, `one oracle profile`, `two subtree profiles`, `two shuffled profiles` (series keys), `label-noise floor (A,B)`, `development-selected rate (A–F)` and `other tested rate (E,F)`: compact symbol keys (marker/line + ≤3 words + panel scope).
- Axis titles: `Input-reassigned − compatible` / `test NMSE (exact credit)` (C), `Population-gradient` / `cosine` (D). These two-line y labels name the plotted quantity.
- Category tick labels (`matching`, `quartic`, `nested`, `Adam`, `SGD`, rates and update counts).

## Layout changes

- Titles: C and D now have none. A, B, E and F carry only the optimizer. Removed D's title-pad override (`pad=11`, which made room for the reference label).
- All axes boxes, the gutters, the margins and the canvas (518.4 × 490 pt) are unchanged. Only the row-1 letters moved: C and D from 157.8 to 170.9 pt (top), now just above their own panels.
- Axis labels changed to sentence case: `Test NMSE` (×2), `Input-reassigned − compatible` (first line of C's label), `Training update`, `Population-gradient` (first line of D's label), `Learning rate` (×2), `Mean test NMSE` (×2).
- Kept the assertions on the Adam quartic interval (width < 0.012; bounds 0.507/0.517) in the builder. Only the printed callout is gone.
- Updated the module docstring to match.
