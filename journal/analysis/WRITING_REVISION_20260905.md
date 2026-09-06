# Complete writing and caption revision — 5 September 2026

The manuscript and Supplementary Information have received a complete editorial revision for clear, connected scientific writing. The pass covers the title, abstract, Introduction, every Results and Methods subsection, Discussion, declarations, all six included supplementary methods fragments, and every figure and table caption. It follows the scientific-writing skill and preserves the distinction between model capacity, estimation, learning and biological evidence.

This is a revision of the paper's language and presentation. No model was retrained and no numerical outcome was changed. The existing disclosure of computational and language assistance remains; the revision makes no claim of human-only authorship.

## Coverage and organization

All eight TeX inputs were read and edited: `main.tex`, `supplementary/supplementary.tex`, and the morphology follow-up, calibration, end-to-end, credit, conductance and Boolean methods fragments. The nine-word title remains accurate. The abstract is now 189 words and follows the main argument without equating task rank with an optimal biological morphology.

The opening defines a route dictionary, its profiles and example-dependent coefficients before introducing the detailed operators. Results proceed from the exact conductance gradient to feedback resolution, update moments, branch selection, ancestry, forward computation, constructive morphology, anatomy and measured-response tests. Each experiment is introduced by the question left open by the preceding analysis. Dense output-to-update operator bookkeeping and detailed retrospective score comparisons now sit in Methods. The Discussion separates what can be represented, estimated and learned, then develops the biological implications and limitations.

The main narrative is 8,748 mechanically counted prose words, compared with 8,183 before this pass. The added local definitions and explanations leave it about 9% above the author's approximate 8,000-word working target. This count excludes Methods, captions, mathematics and references; it is not a submission-portal count. The approximately 5,000-word journal guidance remains advisory in the existing format audit. No claim is made that an editor has approved the current length.

## Definitions and mathematical exposition

Symbols are introduced near their first use, with indices and coordinate systems stated locally. The revision explains neuronal and compartment errors, eligibility, directed path gains, reciprocal adjoints, dictionary dimensions, coefficients, projection metrics, signal/noise moments, and distinct meanings of rank and depth. The constructive sections define binary and Rademacher inputs, multi-affine coefficients, target interactions, centered cut matrices, cut singular values, normalization and selection regret. Methods identify calibration budgets, noise levels, learning rules, optimizer choices and inference units before interpreting outcomes.

Several small corrections were needed to state the existing mathematics accurately:

- The implicit adjoint assumes a nonsingular voltage Jacobian.
- Positive alignment of the complete update with the exact gradient can guarantee local descent; a positive parameter-weighted average of branch cosines alone cannot.
- The focal-shunt amplitude derivative is nonnegative, becoming strictly positive when the baseline adjoint at that site is nonzero. The selectivity equivalence requires positive dose, nonzero amplitude and a positive comparison median.
- The inverse-operator perturbation has rank at most one, with rank one for a strictly positive shunt.
- Zero-energy reliability blocks contribute zero; any gain is optimal for such a block. The optimized smoothness-bound step retains its stated smoothness assumptions.
- The address dimension is infinite when no tested dictionary attains the requested capture tolerance. Minimum sufficient depth is an exact-representation statement when an exact tree exists in the stated model class.
- A stage-average index typo was corrected, the gradient-floor notation was made consistent, and seven unsupported blackboard-bold numeral glyphs were replaced with a readable bold indicator.

These are scope and typesetting corrections, not altered models or numerical experiments. Original citation keys, their multiplicities, equation/figure/table labels and recursive input calls are retained. The bibliography is byte-identical to the preceding revision. Full before/after sources, editorial proposals and preservation checks are stored under `analysis/writing_revision_20260905/`.

## Figures and tables

All 89 captions were reviewed: nine main figures, one main table, 44 supplementary figures and 35 supplementary tables. The check compared captions with actual artwork, panel letters, source builders, units, normalization, line/marker encodings and statistical aggregation. Captions now explain the actual panel sets, including files whose historical filename suffixes no longer match their panel ranges. All main captions are below 350 words; the longest is 328.

Main Figure 3C now has a direct narrative call. The focal Results describe the intervention and passive relation/dose/decomposition panels before physical calibration, which motivates the active-model settings. The panel-order inventory was updated to reflect that deliberate sequence. Full-width supplementary figures and the established caption font size were retained; two tall figures use reduced space above their captions. All captions fit within their compiled pages.

Four supplementary assets required narrowly scoped repairs:

1. **S1:** the retained additive aggregates do not establish their normalization settings through executed run identifiers. The label is now “additive,” with that provenance limit stated.
2. **S2:** the legacy feedback cohort includes resolved raw-additive executions, while its original five seeds lack resolved configurations. The artwork now says “additive (legacy)” rather than uniformly “normalized-additive”; the caption identifies the limited execution coverage.
3. **S18:** panel F is an aligned-minus-reversed interaction in the D2-minus-D1 depth benefit, calculated separately for four methods. Its title and caption now state that estimand. Panels A and E also name their serial-versus-point/star interactions explicitly.
4. **S22:** all eight panels are drawn natively from the same source tables. This removes clipped ordinates and stray fragments left by the previous cropped composition. Eleven plotted-data equality checks passed, including the deterministic uncertainty calculations.

The other 49 publication assets are byte-identical to their previous versions. Source builders and original repaired assets were preserved before editing.

## Source-backed clarifications

The review found one table row that repeated results already excluded elsewhere. The “Noise-task feedback” row in Supplementary Table S23 reproduced the unresolved S3D shunting aggregates exactly. It has been removed from the current results table, with an adjoining exclusion note; the original aggregates and lineage remain archived. The two preceding MNIST stress rows use different sources and remain. This correction does not create a new exclusion or change a numerical result.

Pinky's inherited settings are explicitly those of the separate minnie65 v661 reference cohort. In physical sensitivity experiments the compensation procedure stayed fixed, while the numerical current was re-solved for each condition and intervention to restore its baseline somatic voltage. The active-model Results now identify the actual low-resistance/background-conductance regime. The external animal endpoint is identified as error-reduction minus error-increase soma–dendrite residual in the retained source's z-score units. Its original workbook was not available at the checked local paths, so this pass did not independently reverify the publication's normalization procedure.

Calibration timing now matches the retained protocol: choices were fixed before confirmatory candidates completed final training or exposed endpoints. Wording no longer makes an unsupported stronger assertion that all choices preceded the start of fitting.

## Scientific boundaries retained

The failed original 12,800-fit prospective selector remains visible, together with its improved finite-horizon analysis and strong observed-context baseline. The Boolean grouping contrast remains positive, its small primary credit effect still fails the practical margin, and the larger parity effect remains descriptive. The measured-response cohort still provides no preferential subtree-alignment result. Standard passive calibration, noisy coefficient limitations, optimizer dependence, source gaps, exclusions and the distinction between capacity and learned coefficients remain explicit.

## Verification and deliverables

- Fifteen focused manuscript, Source Data inventory, PDF-link and software-packaging tests passed. The panel-order test initially detected the intentional Figure 8 sequence change; its documented inventory was updated and the checks rerun successfully.
- All 1,473 distinct registered numerical Source Data files match their pre-edit hashes; 1,495 manifest entries were checked.
- Main and supplementary provenance audits each report 1,573 entries, zero errors and zero warnings. All 53 figure assets resolve.
- Citation and text-reuse audits pass. The style audit passes 45 generators and all 53 bounded-width figure placements.
- The main and supplementary PDFs compile without overfull boxes, oversized floats, unresolved citations/references or multiply defined labels. The main PDF has 41 pages and the supplement 128.
- Visual checks cover the repaired assets and the densest revised caption pages. The indicator-glyph repair was inspected again in the compiled main and supplementary PDFs.
- The 169-page combined reading copy preserves all page text, 365 internal links, 20 external links and 82 bookmarks, with no unresolved named links.

The current reading files are `main.pdf`, `supplementary/supplementary.pdf` and `main_with_supplementary.pdf`. Source Data, software, Overleaf and submission archives are rebuilt from the finalized sources. Archive hashes and byte-level release checks are recorded separately in `analysis/writing_revision_20260905/VALIDATION.md` and `FINAL_VALIDATION.json`, avoiding a self-referential archive checksum.

The prior manuscript, PDFs and release archives remain preserved under `analysis/writing_revision_20260905/before/`. The active repositories are not committed by the release build, and no external submission is performed. Existing author-specific metadata, disclosure/access fields and final coauthor/form approvals remain the actions already documented in `submission/AUTHOR_ACTIONS.md`.
