# Nature Communications reporting and package checklist

Article: *Dendritic morphology as a dictionary for local credit assignment*.
Status: `[x]` implemented in the revised source; `[ ]` final verification or
author-specific action still required. This audit does not replace official forms.

## Structure and editorial scope

- [x] Revised title and a main narrative organized as Introduction, Results, Discussion and Methods.
- [x] Ten main figures and 36 supplementary figures, grouped by scientific question. Single-neuron inhibitory selection is Figure 5; population selection and rescue are Figure 6; physical depth is Figure 7. The supplement has a contents table, a main-figure guide and nineteen evidence tables.
- [x] Cover letter, editorial summary and overlap statement reflect the dictionary formulation and conditional results.
- [x] The initial costed selector in a finite linear tree class retains its negative long-horizon result in Supplementary Fig. S34. Main Figure 4 links matched pairwise/quartic learning to learned credit geometry; forward construction and noisy-query tree selection remain supporting analyses in the same paper.
- [ ] Verify current abstract/title/reference/display guidance before upload. The author has chosen approximately 8,000 narrative words for this revision, while the local audit retains approximately 5,000 as advisory journal guidance.
- [ ] Confirm current related-manuscript status, authorship and author-approved declarations.

## Design and biological scope

- [x] Artificial-network comparisons use paired training seeds as the independent unit where applicable.
- [x] Cells, contacts, sites, scans and stimulus splits are distinguished from animal replication.
- [x] Structural analyses include two MICrONS mice; original minnie65, the disjoint 47-cell cohort, functional and focal analyses remain within one mouse. Pinky contributes structural capacity comparisons, including the common-mode follow-up, with eight cells eligible at K=8.
- [x] Functional analyses use seven postsynaptic targets and 13 eligible scans; scan and stimulus-split observations are nested within targets.
- [x] The external six-animal result concerns signed response coordinates, not measured dendritic credit gradients.
- [x] No new animal or human experiments are claimed; public-source reuse and model assumptions are described.
- [x] Dendritic depth, spatial credit routing, weak-channel linearization and actual learning measurements are distinguished.
- [ ] Confirm inclusion/exclusion tables, outcome-free selection statements, source-data IDs and proxy-table access conditions against the final release.

## Reanalyses and new model experiments

- [x] `analyze_review_evidence.py` reconstructs the global logistic curvature bound and deterministic update-moment score from the original full-batch factorial; hypothetical minibatch noise remains separately labeled.
- [x] Direct checks reproduce 270 stored gradient moments and verify actual one-step decreases exceed the global bound; all 2,700 denominators are finite and positive.
- [x] Pooled score correlations use 5,000 whole-seed bootstrap draws; within-family prediction, degeneracy, regret, endpoint baselines and nine distinct nonconstant mean curves are exported.
- [x] K=4 ancestry effects include individual paired controls, the declared four-control best comparator, bootstrap intervals and separately identified multiplicity-corrected tests. Exact accuracy-count ties are restored before Wilcoxon ranks.
- [x] Main Figure 10 shows target-level response correlations and model-conditional sensitivity; realized input-route supports and the separate learning/geometry comparisons are retained in Supplementary Figs. S31–S32.
- [x] `analyze_fulltree_within_span_oracle.py` replays 130 exact-learning fits and compares frozen and trial-dependent coefficients within fixed dictionaries at common checkpoints. Three replays drift beyond 1e-5 NMSE; maximum drift is 0.000574, retained without exclusions. No oracle-trained trajectories are inferred.
- [x] Both analysis folders include source/script hashes, numerical outputs and methodological notes.
- [x] Completed a prospectively sealed held-out candidate-tree experiment: 20 independent seeds, 12,800 candidate fits, fixed reference spectra under rotation and genuine minibatch updates. The moment selector loses to rank-only and fixed baselines; no general morphology law is claimed.
- [x] Added exhaustive isospectral interaction diagnostics, exact tree/depth construction, 25,600 fresh finite-horizon fits and a strong observed-context baseline.
- [x] The new bridges retain 7,760 fresh fits: 2,240 finite-calibration candidate fits, 3,600 credit fits, 1,440 conductance fits and 480 separate end-to-end fits. Each cohort uses twenty independent paired seed blocks; tasks, candidates and restarts are nested observations. Frozen protocols disclose privileged true-target/reference inputs, optimizer sensitivity, failed pre-fit attempts and implementation amendments. Each of the finite-calibration and end-to-end cohorts has two predeclared contrasts with Bonferroni-adjusted 97.5% intervals and a 0.01 NMSE mean-improvement margin; credit and conductance intervals are pointwise. Hard random-interaction outcomes and task-dependent credit effects are retained. Exact capacity does not imply a universal biological morphology law.

- [x] Supplementary Figs. S14–S15 show seven fixed Boolean templates, all sixteen uniform input patterns, exact centering/variance normalization, and four matched trees. The separate Boolean cohort retains 6,720 fresh fits, 1,680 development fits and a 336-fit excluded smoke; these counts are not pooled with the preceding 7,760 bridge fits. Twenty fresh seeds are independent; templates, candidates, rates and checkpoints are paired conditions.
- [x] The primary Adam XOR-of-AND grouping improvement is 0.563174 NMSE, adjusted 97.5% interval [0.549144, 0.582749]. The compatible broadcast-minus-exact improvement is 0.002507 [0.001780, 0.003318], positive but below the prespecified 0.01 mean-effect margin. Both rules classify every compatible pattern correctly. Larger parity credit effects are descriptive.
- [x] Boolean code/protocol and development-selected rates were sealed before fresh training. All three rates, both optimizers, clipping records, final weights and streams are retained. Independent audits reproduce all 8,400 development/fresh endpoints and rate/primary statistics, and all 67,200 checkpoint losses satisfy the certified structural bounds. The posthoc subset-projection diagnostic is explicitly separated from the confirmatory training comparisons. See `source_data/boolean_morphology/`, `source_data/boolean_theory/` and the matching script folders.

- [x] Figure 4 noise controls use twenty additional paired seeds and 720 trajectories at inherited rates; clean, fixed-absolute and relative-noise conditions retain the interaction deficit.
- [x] Figure 6 reports ordinary-test, selected-rate stress and common-rate stress effects together. The original nonlinear resistance gate is worse on average than broadcast; training bound contacts are distinguished from selected-state occupancy.
- [x] Parent sensitivity rescues Adam learning in twenty new paired seeds; within-parent/context shuffling, common-rate, wider-bound and same-seed separable-target controls delimit the result. Expanded exact-SGD development fails the frozen low-error criterion; no fresh SGD cohort was launched.
- [x] A full released-software replay of one predetermined seed and four rules reproduces all 132 validation-history points and all ordinary/stress outcomes within the frozen tolerances, with identical selected steps. This verifies existing results, not new independent replication.

## Statistical and visual reporting

- [x] Central contrasts identify the seed, cell, target or animal unit as appropriate.
- [x] Main ancestry comparisons report individual seeds, effect sizes and paired intervals; exact sign-flip and Wilcoxon results are not conflated.
- [x] Supplementary Fig. S8 describes the plotted owner standard deviations; Supplementary Section S9 and Table S7 distinguish descriptive neuron-level SEM from separate animal-level inference.
- [x] Optimizer-dependent physical-depth contrasts are displayed without calling the unmatched comparison a fully controlled factorial interaction.
- [x] The conceptual evidence map and final-panel checklist are removed; the local utility and noise screens are retained in Supplementary Fig. S3 beside their derivations.
- [ ] Verify every final interval/test label against its generating table and inspect panel readability in the compiled PDFs.
- [x] Completed the bounded run-lineage audit: historical regular-tree shunting execution remains unresolved (Supplementary Section S2 and Table S3). The input-validity records supporting Supplementary Fig. S9 join all 320 runs to the retained audit, with 160 additive runs retained and 160 shunting runs excluded. Raw resolved configs are absent; no validity is inferred from config names.

## Release and official forms

- [ ] Review the final provenance and submission audit records for the exact version approved by all authors.
- [ ] Review the completed Source Data, reviewer-software, Overleaf and submission build records, manifests, hashes and isolated source snapshot metadata. Fingerprints and final verification records remain external to the archives.
- [ ] Confirm immutable public/reviewer data and software records, license, environment instructions, raw-cache access conditions and absence of secrets.
- [ ] Approve the populated technical machine-learning draft, transfer the prepared software and reporting-summary answers into their official forms, complete author-controlled fields and verify saved forms in Adobe Acrobat Reader.
- [ ] Confirm all-author approval, exclusive consideration, funding, contributions, conflicts and any required disclosure on the submission date.
