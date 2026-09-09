# Main-figure restoration and supplement consolidation

The revision keeps the credit-first narrative and nine-figure sequence. Selected explanatory and quantitative material has returned to the main figures, while the Supplementary Information has been reorganized by scientific question. This is an editorial consolidation of completed studies; no new training experiments were run for this change.

| Publication component | Before this consolidation | Current |
|---|---:|---:|
| Main figures | 9 | 9 |
| Supplementary figures | 56 | 35 |
| Supplementary tables | 37 | 12 |
| Supplement PDF | 153 pages | 110 pages |
| Main PDF | 32 pages | 33 pages |
| Combined reading copy | 185 pages | 143 pages |

The current main narrative is approximately 7,200 words, below the author's 8,000-word working target. Figure legends are 171–246 words. This working target is distinct from the journal's shorter general guidance.

## What was restored to the main figures

- **Figure 1:** the path from a task error to a local update, actual one-/three-/twelve-profile dictionaries, and an illustrative noise–resolution tradeoff appear beside the six-rule image experiment. The small three-profile learning effect and decoder-only reference remain visible. The noise panel states a conditional one-step bound; it does not revive the failed initialization selector.
- **Figure 4:** task drawings now explain why pairwise and quartic targets are representable on the same tree despite their different interactions. Existing learning trajectories extend to 16,384 updates. The credit-capture panel includes existing initial, 1,024-update and 16,384-update diagnostics, placing the learning deficit beside the multidirectional trained field. No new fits or rate choices were introduced. The original and continued diagnostic rows agree to floating-point precision.
- **Figure 6:** task-family drawings, the task-hierarchy/physical-depth grid, and serial-versus-grouped computation return beside the longer learning trajectories. The accuracy and cross-entropy gaps are both shown. This restores the forward-morphology result while retaining its optimizer and budget qualifications, including the H4 result and D1/D3 stopping differences.
- **Figure 7:** an example arbor is connected explicitly to a dictionary containing a common profile. Capture across complete 47-cell budgets, the surrogate comparison, and the capture–wiring-density comparison show what the anatomical dictionary provides. The passive-field dependence on ancestry is stated. The K=16 subset remains in Source Data; the main budget curve uses K=1,2,4,8, where all 47 cells are available.
- **Figure 9:** the measured alignment estimates are now accompanied by their conditional detection sensitivity and observed repeat reliabilities. Realized route supports, ridge comparisons and transfer/update reconstruction are grouped together in the supplement, so the empirical result and its sensitivity lead the main display.

Figures 2, 3, 5 and 8 retain their current native displays: branch conflict, ancestry at matched bandwidth, the local inhibitory gate, and shunting as an ancestry-partition gain. The newer interaction and local-gate evidence remains central.

## How the supplement was reduced

The supplement now has ten scientific sections: exact and restricted credit; image learning; branch conflict and ancestry; interaction and Boolean tasks; conductance learning; physical depth; anatomical dictionaries; shunting; measured responses; and structure selection/statistics. A contents table and guide map all nine main figures to their supporting sections and displays.

The 35 figures select and reflow useful vector panels instead of appending entire earlier sheets. Repeated endpoints, duplicated task maps, separate versions of the same control and large endpoint grids were removed from the printed sequence. Tables were consolidated into essential definitions, distinct scientific comparisons, reproducibility evidence and a source index. The source index points to complete numerical tables rather than printing every row again.

The following remain explicit: the failed prospective selector; soft-encoder leakage and the exploratory hard-readout rescue; same-span and reliability limitations; the depth accuracy crossover and moving loss gap; the H4 mismatch with a universal depth rule; passive anatomical fields' structural dependence; weak-channel linearization scope; the measured-response null, limited sensitivity and ridge comparison; and large same-seed replay discrepancies. Removing a redundant plot does not remove an unfavorable result.

**Logical tasks remain in Supplementary Figures S13–S14.** The main interaction section now introduces AND, OR, parity and mixed-gate examples. The detailed truth-table/capacity and learning comparisons remain separate because forward representability and credit-dependent learning answer different questions; compressing both into a small main panel would obscure that distinction.

## Source preservation and publication hygiene

The manuscript uses semantic figure labels and an explicit old-panel-to-current-panel map. Omitted panels point to the appropriate scientific section, table or Source Data rather than inheriting an incorrect new panel letter. All 72 exact table destinations are retained as complete source files with authenticated hashes. Existing filtered ownership subsets retain their original 80/4/120/6/160 row counts, and complete source tables are available separately. All 65 anatomical evaluation files and 149,331 underlying rows remain included.

The original numerical-source set is preserved. Historical vector inputs remain available for reproducible rendering, but unused manuscript fragments and displays are excluded from the Overleaf and submission projects. Internal review material and the review contact sheet do not enter the publication packages. The software package retains the authenticated vector and code inputs needed to rebuild the curated figures.

## Validation

The main and supplementary documents compile without undefined references, duplicate labels, overflowing boxes or oversized floats. All nine main displays and all 35 supplementary displays received visual/caption review, including uncertain crops at native size. Every supplementary PDF reproduced byte-identically on rebuild. Independent checks verified all cited supplementary sections, tables and panel destinations, and numerical-source closure for every current supplementary figure.

The full article test suite passed 191 tests, with one optional test skipped. The build also checks the main-figure numerical summaries, embedded fonts, source hashes, figure allowlists and nested TeX inputs. Source Data, software, Overleaf and submission packages are generated locally from the finalized sources. Packaging and local commits do not submit or upload the paper.
