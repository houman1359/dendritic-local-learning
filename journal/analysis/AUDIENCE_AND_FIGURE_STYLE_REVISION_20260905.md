# Audience accessibility and figure presentation revision

This revision addresses readability for computational neuroscientists and broader neuroscience readers, together with the presentation of all nine main and 44 supplementary figures. It builds on the complete writing revision in `WRITING_REVISION_20260905.md`. This is an editorial and visual assessment, not a test with independent human readers.

## Writing for both audiences

The main obstacle was the order of explanation: several passages introduced mathematical machinery before saying what it meant for a synapse or a branch. The revision gives the interpretation first and then defines the quantities needed to assess the claim.

- Credit means how a small synaptic change would affect task error; eligibility is the synapse's local sensitivity. A voltage error is explicitly a loss derivative.
- A route dictionary is introduced as reusable spatial patterns, with coefficients setting the signal delivered on each trial. Profile count, independent patterns and coefficient information remain distinct resources.
- Capture is defined as a fraction of squared signal magnitude. It is distinguished from classification accuracy, update reconstruction and progress during learning.
- Rank, sensitivity spectra, affine and multi-affine operations, Fourier interactions, a cut through a subtree, selection regret and privileged oracle access are explained at their first substantive use.
- The two NMSE normalizations are separately defined: known target variance in artificial tasks and prediction by the training mean for measured responses.
- The Introduction now sets out three questions about feedback resolution, task-compatible structure and biological relevance. The SI overview follows the same progression and introduces the cohort and reproducibility tables.

The central gradient factorization, path transport, route representation, update bound and structural cut criterion remain in Results. Auxiliary adjoint equations, the detailed scalar fallback, the fixed routing operator, synthetic generators and route formulas, and the focal-kernel ratio are now in the relevant Methods. Detailed retrospective score comparisons and numerical diagnostics accompany their protocols there. All 30 main mathematical displays are retained, together with the citations, labels and included methods files.

The paper remains a quantitative modeling study. Its proofs and complete protocols require technical detail, but the main narrative now supplies the biological question and practical interpretation before that detail. The revised narrative has 8,191 words, the abstract 196, and the Methods 8,079 under the existing prose counter. Definitions and relocations shorten the narrative while increasing Methods; they do not reduce the total article by the same amount.

The revision preserves the original failed prospective selector, the strong simple baseline for its repaired forecast, the failed primary Boolean credit margin, optimizer dependence, the negative measured-response result and all model-class restrictions. It does not turn a fixed-state optimal projection into a claim of universally optimal biological morphology.

## Figure design

All nine main figures use a common 0.92-textwidth placement and the shared native canvas width. Individual canvas heights and gutters accommodate content and captions without changing the type scale from one main figure to another. Supplementary figures remain full width. A production audit now checks the common main placement explicitly and covers 69 figure builders, including native canvases.

The shared palette and line/type tokens are retained. Marker-edge width is explicitly reset by the journal style so standalone and full-sequence renders agree. Same-hue darker direct labels improve contrast on light backgrounds without changing plotted data colors or white heatmap annotations. Small algebraic expressions that were difficult to read inside plots are replaced by concise verbal labels; exact definitions remain in the captions or Methods.

Concrete repairs include the main Figure 5 contrast-label column, longer-title gutters, row alignment, wrapped annotations, and main Figure 2's tiny normalized-error formulas. Supplementary repairs address cramped category labels, inconsistent legacy headers and colors, clipped axis text, colliding schematic annotations and legends covering markers in S41G and S42C. S17–S20 are rebuilt as native panels from their frozen tables, with S18 arranged in three rows of two panels.

Numerical preservation is checked independently of appearance. Source hashes and plotted line coordinates, scatter offsets, image matrices, interval paths and bar rectangles are compared with the pre-revision versions. Font and bounding-box inventories at manuscript placement scale supplement visual review. Their flags are interpreted visually: mathematical subscripts and intentionally wide forest plots cannot be judged by a generic prose-size or panel-aspect threshold alone.

## Validation and deliverables

Final pre-release checks pass: 15 focused tests, both provenance audits (1,573 entries; zero errors or warnings), and compilation of 42 main plus 128 supplementary pages. All 1,473 distinct registered numerical Source Data files and the bibliography remain unchanged. The PDF census finds no text outside any of the 53 figure canvases; its only small prose-like flag is a mathematical subscript. Exact numerical-artist and vector-path comparisons support preservation of the plotted results.

The final validation record is stored separately in `analysis/audience_style_revision_20260905/VALIDATION.md`, with machine-readable results and the complete before/after source diff in the same directory. Keeping archive hashes outside the packaged input report avoids a self-referential archive hash.

The current deliverables are `main.pdf`, `supplementary/supplementary.pdf`, `main_with_supplementary.pdf`, and the regenerated Source Data, software, Overleaf and submission archives under `submission/`. The numerical experiments are unchanged; this revision uses saved outcomes and render-only code. The existing author-specific submission actions and disclosure remain in place.
