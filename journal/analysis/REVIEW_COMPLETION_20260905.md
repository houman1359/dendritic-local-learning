# Completion of the Nature Communications revision — 5 September 2026

This report supersedes the **status statements** in `REVIEW_IMPLEMENTATION_20260905.md`; the original review and first implementation report remain as historical records. The revised Article is **Dendritic morphology as a dictionary for local credit assignment**. It is written as the complete standalone paper. The author expects the earlier NeurIPS submission not to be accepted, but its official decision is still pending. No rejection, withdrawal, exclusive consideration or all-author approval has been inferred.

The working narrative follows the author's approximately 8,000-word target. Methods has been consolidated from 12,892 to 4,412 prose words, including the new experiments. The abstract is 190 words. There are nine main figures, one Methods cohort table and 34 supplementary figures. The exact final counts, PDF checks, test outcomes and archive fingerprints are recorded separately in `completion_20260905/VALIDATION.md`.

## Scientific outcome of the additional work

The revision supports a conditional dictionary framework, not a demonstrated universal law connecting task complexity to optimal dendritic morphology. New experiments were retained whether they supported or challenged the proposed mechanism.

Two numerical statements in the original review were corrected during the first reanalysis. The 25 representation-by-family mean cells contain **nine** distinct nonconstant curves. The corrected original-four-control stochastic contrast selector matches **8/20** seed optima, and the full-batch selector matches **13/20**; the older 6/20 used a different comparator. These corrections remain in the manuscript and source records rather than being concealed by the review's preserved historical wording.

1. **Prospective morphology selection is complete.** Five explicit fifteen-node trees and four dictionary budgets were evaluated in separate feedback-only and joint forward/feedback arms. Development tasks set the selector and baselines; 20 fresh task seeds and two held-out rotation angles generated 320 confirmatory tasks. All 3,200 policy choices were sealed before any of the 12,800 candidate fits. The reference spectrum is invariant under rotation at fixed rank and noise, and minibatch noise is present during training. All candidates and failures of the selector are retained.

   The initialization-moment selector has mean costed-endpoint regret 0.0806 and 0.1040 in the two arms, compared with 0.0083 and 0.0082 for the development-trained baseline supplied with the generating rank. It also loses to the fixed maximum-budget baseline in every seed average. Nevertheless, its mean within-task rank correlations with first-step progress are 0.984 and 0.974, and all 12,800 calibration-step bounds pass. This is a failure of long-horizon extrapolation despite a useful local bound. The supplied context decoder, fixed candidate grid, linear forward learner and declared cable/channel proxy limit the physical interpretation. Main Figure 6 presents the test and its negative primary result; no outcome-dependent tuning was used to replace it.

2. **Learned route coefficients are now tested.** Twenty fresh seeds compare oracle context coefficients, a separately trained local-cue estimator, a frozen profile and a mismatched estimator over calibration size, cue noise and delay. At 256 calibration examples and noise SD 0.5, soft delivery reaches 25.83% accuracy versus 80.29% for the oracle and 18.67% for the frozen profile. Its 7.16-point advantage over the frozen profile coexists with a 54.47-point oracle gap. Reliable cues nearly close that gap. An explicitly exploratory follow-up uses the same fitted estimator with hard route selection and reaches 64.03%; it is not labeled a confirmatory rescue. Delay across independent trials removes the benefit. Larger calibration sets also imply more optimization steps at fixed epochs, so this is not a pure sample-complexity experiment. Figure S32 reports all these distinctions.

3. **Branch-credit alignment is measured throughout training.** The original 20-seed experiment was re-executed for all branch counts, conflict doses and rules, producing 28,800 gradient-diagnostic rows at six checkpoints. Both common exact-rule states and each rule's own states are retained. Historical endpoint concordance is within archived CSV rounding. At four branches and conflict dose 0.5, shared-versus-exact cosine at the common exact trajectory changes from 0.987 to −0.519. The centered initial threshold is therefore not a fixed descent boundary throughout training. Figure S29 now includes these trajectories. These are same-seed implementation reruns, not additional independent statistical replicates.

4. **Anatomical uncertainty is quantified as sensitivity.** The audit reports the direct/proxy confusion matrix (2,012 mapped contacts, 73.01% agreement), label availability by compartment and path-depth stratum, and 1,056 mapping/radius/label/electrical conditions in eight cells. Direct-only classification changes selected routes most strongly; both mapping and radius changes matter. The same nominal modeled fields and weights are used for all capture comparisons. Chain compression preserves length and the radius–length product, but not heterogeneous axial resistance. Replacing the approximate coupling by exact raw-edge series resistance changes nominal-field capture by 0.0082 on average. Figure S33 distinguishes this controlled sensitivity from a calibrated reconstruction posterior or endogenous task test.

5. **Measured-response baselines and observed-input sensitivity are complete.** There are 3,380 additional linear-model evaluations on the 130 reconstructed outer splits, spanning all 13 scans and seven targets. All preprocessing, reliability filtering and ridge tuning use training data only. Nested ridge achieves normalized MSE 0.8028 versus 0.7877 for the archived exact nonlinear model; the paired nonlinear improvement is 0.0152, with 95% target-bootstrap interval 0.0042–0.0271. Partner removal and presynaptic predictor noise reduce predictability. A held-out-data corruption challenge leaves all 338 checked tuning/filtering decisions unchanged; historical split counts and stimulus disjointness pass for all 130 splits. Figure S34 reports the baselines and sensitivities. This does not estimate power under unknown missing partners or provide new animals.

## Final figure and manuscript structure

| Main display | Role |
|---|---|
| Figure 1, A–E | Combined conceptual entry and concrete dictionary matrices/capture; former Figures 1 and 2 are merged. |
| Figure 2, A–G | Neuronal versus within-arbor feedback resolution and paired controls. |
| Figure 3, A–H | Actual-update moment theory, deterministic validation and explicitly conceptual evidence map. |
| Figure 4, A–F | Controlled branch-selector requirement, zero inactive exact derivatives and learned endpoints. |
| Figure 5, A–I | Hierarchical route factorial and enlarged paired four-route ancestry effect. |
| Figure 6, A–F | New prospective morphology-selection test, baselines, selected budgets and first-step/final horizon comparison. |
| Figure 7, A–G | Independent modeled fields, route-generated capacity control and second-mouse structural check. |
| Figure 8, A–G | Focal transport, calibration and the physiological boundary. |
| Figure 9, A–F | Measured-response boundary with native-unit contrasts and imposed-alignment control. |
| Methods Table 1 | Compact cohort, resource, comparison-unit and source index. |

Physical depth remains in Figure S31, with the optimizer comparison visible. The new prospective test occupies the main display freed by the conceptual merge. The supplement follows the same scientific progression, includes linked contents and retains the external animal re-expression as supplementary context. Repeated panels have been removed from S18, S19 and S22. Historical unresolved shunting curves in S1–S3 are removed from the current artwork while the underlying source records and their limitations remain available. Figure filenames sometimes retain legacy panel ranges; current captions and the figure inventory specify the actual panels.

The final package check also found and fixed a navigation defect in the old PDF merger. The combined document now retains 326 internal links, 14 external links and 73 supplementary bookmarks with correct destination pages and coordinates. All 138 merged pages render identically to their standalone originals. The Makefile uses the tested link-preserving merger.

## All 76 review recommendations

“Addressed” means the scientific or editorial problem was corrected. “Scoped” means the claim was narrowed or the affected evidence removed, with the remaining limitation stated explicitly; it does not claim the suggested measurement was performed. “Author action” identifies declarations or external publication steps that cannot be supplied from computational work.

| No. | Topic | Resolution in the final revision |
|---:|---|---|
| 1 | Central contribution | Addressed: morphology-constrained spatial credit dictionaries organize the title, abstract, Results and Discussion. |
| 2 | Feedback versus physical morphology | Addressed: distinguished in definitions, resource accounting and the two prospective arms. |
| 3 | Meaning of optimality | Addressed: explicit optimization over candidate tree, budget and dictionary, with calibration loss, moment utility and cost; all candidates tested prospectively. |
| 4 | One dictionary per task | Addressed: removed the unsupported conclusion; new test exposes selection regret and failure against baselines. |
| 5 | Rank eight and maximum budget | Addressed: construction-induced rank is disclosed; old near-equality is not evidence for a morphology law. |
| 6 | Task complexity | Addressed: credit rank is field-, coordinate- and metric-dependent; generating rank supplied to the new baseline is explicit. |
| 7 | Error dimension and spatial capacity | Addressed: fixed profiles, span rank, external errors, local gates and state-dependent transport are separate resources. |
| 8 | Universal gradient-operator assertion | Addressed: actual routed-update moments define the general bound; a fixed transform is only a special case. |
| 9 | Training noise versus diagnostic noise | Addressed: original full-batch deterministic analysis added; new prospective experiment has genuine minibatch noise. |
| 10 | Curvature guarantee | Addressed: global logistic curvature used where valid; actual one-step inequalities checked directly. |
| 11 | Mean capture versus energy ratio | Addressed: per-field capture and ensemble energy capture distinguished; uncentered second moments named correctly. |
| 12 | Branch conflict mechanism | Addressed: missing selector/gate contaminates updates where inactive exact derivatives are zero. |
| 13 | Conflict threshold through training | Addressed: new six-checkpoint, common-state/own-state gradient sweep; analytic boundary remains a centered initial approximation. |
| 14 | Physical-depth optimizer interaction | Addressed: 10.96-point LocalCA contrast shown alongside 0.64-point standard-optimizer contrast and optimizer-change cost. |
| 15 | Title | Addressed: “Dendritic morphology as a dictionary for local credit assignment.” |
| 16 | Abstract | Addressed: 190 words; includes random between-neuron feedback, prospective negative result and biological boundary. |
| 17 | Introduction | Addressed: concise problem, prior mechanisms, dictionary distinction and corrected experimental roadmap. |
| 18 | Results argument | Addressed: each comparison identifies manipulation, resource control, measured outcome and claim boundary. |
| 19 | Narrative length | Addressed to author target: approximately 8,000 words; journal advisory guidance is reported separately. |
| 20 | Methods consolidation | Addressed: 12,892 to 4,412 prose words, nine thematic subsections, one cohort table and SI protocol detail. |
| 21 | Dictionary object table | Addressed: SI Table S14 maps coordinates, signs, ranks, metrics, coefficient sources and forward assumptions. |
| 22 | Distinction from prior work | Addressed: established adjoints and chain-rule identities are attributed; complete formulation is retained in this standalone Article. |
| 23 | Biological precedent | Addressed: motivates the Introduction; external signed-coordinate re-expression is supplementary. |
| 24 | Main figure sequence | Addressed: conceptual merge frees Figure 6 for prospective selection; physical depth retains the optimizer comparison in S31. |
| 25 | Conceptual Figure 1 | Addressed: native five-panel entry point with worked matrix multiplication and shared visual language. |
| 26 | Atlas merge | Addressed: concrete matrices and trained field capture become Figure 1D,E. |
| 27 | Feedback comparisons | Addressed/scoped: paired endpoint contrasts retained and strict-scalar versus matched-width diagnostics labeled accurately. No new strict-scalar checkpoint diagnostic is represented as having been run. |
| 28 | Theory synthesis figure | Addressed: deterministic/global-curvature utility and a conceptual map without fitted phase boundaries or optimum ring. |
| 29 | Branch gradients | Addressed: main selector schematic plus new S29 trajectory diagnostics. |
| 30 | Ancestry-specific evidence | Addressed: enlarged paired K=4 contrast, all four controls and distinct multiplicity families. |
| 31 | Anatomy targets and costs | Addressed: independent cable fields separated from route-generated controls; coefficient counts remain a proxy. |
| 32 | Focal physiology | Addressed/scoped: passive calibration is prominent; active results are fixed-baseline-Jacobian sensitivities, not measured plasticity. |
| 33 | Measured-response ending | Addressed: native-unit effects, imposed-alignment control, common-state oracle, and new linear baselines/sensitivities. |
| 34 | Figure readability | Addressed: native redraws, reduced duplicated sheets, corrected labels, panel spacing and final-size inspection. |
| 35 | Main captions | Addressed: every main legend remains below 350 words with units, independent n and interval definitions. |
| 36 | SI organization | Addressed: reordered thematic sections, linked contents, updated main/SI pointers and 34-figure sequence. |
| 37 | Historical input validity | Scoped: row-level lineage audit retained; unresolved shunting series removed from current S1–S3 artwork. Configuration intent is not execution evidence. |
| 38 | Error bars | Addressed with archival limits: verified plotting/summary definitions corrected, especially S7, S11, S12, S17, S21, S24 and S27; absent original raw outcomes are not reconstructed by assertion. |
| 39 | Individual SI revisions | Addressed: caption/artwork pass across S1–S31, duplication removal, S29 expansion and new S32–S34. Detailed handoffs preserve panel-specific decisions. |
| 40 | K=4 control effect | Addressed: 1.27 points, paired interval, 15/20 positive seeds, exact paired-mean sign-flip and restored-count Wilcoxon analyses. |
| 41 | Oracle terminology | Addressed: fitted capacity, prescribed PCA, exact transport and within-span update diagnostics are distinguished. |
| 42 | Predictive validation | Addressed: held-out tasks, development-only policy fitting, fixed/rank-only/random baselines, regret and all negative primary comparisons. |
| 43 | Isospectral alignment | Addressed: new orthogonal rotations preserve the reference spectrum at each rank–noise pair; joint transfer may change parameter-gradient spectrum. |
| 44 | Laminar necessity | Addressed: sufficient leading-eigenspace condition, projector domain and no unsupported “exactly when” claim. |
| 45 | Interior optimum | Addressed: finite-grid conditions stated; generator-designed optima and retrospective selection remain distinct. |
| 46 | Gain versus address | Addressed: nonzero static column gains preserve span; physical shunts can reshape transfer profiles. |
| 47 | Resource matching | Addressed: contact-, parameter-, dictionary- and cost-matched comparisons identified locally; flexible MLP is only approximately parameter matched. |
| 48 | Capture per coefficient | Addressed: proxy for routing sparsity, not measured wiring or energy optimality. |
| 49 | Biological replication | Addressed: same-mouse cells, second structural mouse, seven functional targets and six external animals kept separate. |
| 50 | Coarse geometry | Addressed: coarse branching/depth explains much capacity; limited fine-topology effects are not promoted. |
| 51 | Scope of response null | Addressed: specific fixed-profile rule, observed responses and seven-target cohort; within-span oracle geometry remains unfavorable. |
| 52 | Sensitivity of biological test | Addressed/scoped: native-unit effects, nested linear baselines and observed-input degradation added. Unknown missing-partner power is not claimed. |
| 53 | Prediction for plasticity | Scoped: discriminating shunt-minus-current experiment specified; no new biological plasticity measurement was possible or claimed. |
| 54 | Physiological boundary | Addressed: conductance-state dependence and a current control whose effect need not vanish. |
| 55 | Generator equations | Addressed: executable noise–floor–gain order and flat-factor support; rectified noise explicitly described. |
| 56 | Experiment index | Addressed: compact Methods cohort table plus complete SI configurations, hashes, units and source pointers. |
| 57 | Statistics | Addressed: paired effects, resampling hierarchy, uncertainty units and prespecified multiplicity families stated. |
| 58 | Equivalence | Addressed: formal equality/equivalence limited to algebraic identities or explicit predefined tests. |
| 59 | Prespecification | Addressed: private protocol/source seals, held-out prospective tests, same-seed reruns and exploratory hard readout distinguished. No public preregistration claimed. |
| 60 | Submission package | Addressed by final build: synchronized PDFs, sources, Source Data, software and Overleaf; exact verification recorded in final validation report. |
| 61 | Documentation | Addressed: standalone framing, nine main/34 SI figures, correct biological cohorts and revised scientific outcomes. |
| 62 | Public data/code release | Local package addressed; author/external action remains for permanent DOI/version and actual reviewer/public access. No identifier fabricated. |
| 63 | Cover letter | Addressed: one complete Article; bounded primary claims and prospective failure stated; no false independent-cohort claim. |
| 64 | Related submission | Updated from author: expected nonacceptance, official decision pending, Nature Communications intended as sole publication. Official status and submission declarations remain author actions. |
| 65 | Discussion | Addressed: mechanisms, coefficient availability, evolving credit, resource constraints and biological use synthesized. |
| 66 | Sentence-level edit | Addressed: scientific-writing pass across main, Methods, captions and relevant SI; final factual cross-check corrected sensitivity and rank-baseline wording. |
| 67 | Major prospective experiment | Addressed: 12,800 retained fits; tested and rejected the initialization surrogate's stronger long-horizon claim. |
| 68 | Forward/feedback distinction in new test | Addressed: separate identity-forward and explicit passive-tree-transfer arms; imposed decoder and linear-learner boundary stated. |
| 69 | Coefficient acquisition | Addressed: learned local cue, oracle/frozen/mismatch comparisons and calibration/noise/delay grid; hard follow-up labeled exploratory. |
| 70 | Existing-data analyses | Addressed: deterministic utility, native biological contrasts, within-family regret, ancestry controls and within-span geometry. |
| 71 | Negative results and controls | Addressed: no selective removal of failed morphology selector, noisy encoder gap, weak fine topology, passive boundary or response null. |
| 72 | Dependency order | Addressed: settled scientific claims, ran tests, reorganized text/figures, then froze and rebuilt reviewer materials. |
| 73 | Tables and units | Addressed: 35 numbered SI tables, corrected definitions/units/contrasts, local sample scope and missing first-use references. |
| 74 | Bibliography | Addressed: established mathematical tools attributed; accent/metadata fixes; no undefined citation keys. |
| 75 | Labels and preprocessing | Addressed/scoped: confusion/coverage, mapping/radius sensitivity and exact-series comparison; donor/completeness limits remain explicit. |
| 76 | Reproducibility and forms | Current frozen-data, CPU-example and full-training entry points supplied; historical checkpoint/environment gaps and replay drift retained. Official templates and verified technical draft answers prepared; author declarations and final Reader/XFA checks remain external actions. |

## What remains outside this computational revision

The local scientific revision is complete at the stated scope. It does not supply a new wet-lab experiment, recover historical artifacts that were never retained, establish population power from one functional mouse, or turn the failed prospective selector into a positive result. Those limits are part of the paper's interpretation.

Before submission the authors must confirm the official conference decision or withdrawal and actual exclusivity, all-author approval and contributions, funding/conflicts/ethics declarations, and permanent data/code access identifiers. The official forms have technical draft answers; author-controlled fields and Adobe Reader verification remain pending. The draft stays unsubmitted. See `submission/AUTHOR_ACTIONS.md` and `submission/official_forms/COMPLETION_ANSWERS.md`.

## Audit and reproduction entry points

- `completion_20260905/VALIDATION.md`: final build/test/package evidence and fingerprints.
- `METHODS_SI_COMPLETION_HANDOFF_20260905.md`: protocol consolidation and caption uncertainty audit.
- `source_data/prospective_morphology_selection/`: source/selection seals, all candidate results and numerical validation.
- `source_data/review_coefficient_encoder/` and `source_data/review_coefficient_hard_readout/`: primary and exploratory coefficient results.
- `source_data/review_branch_trajectories/`: common-state and own-state gradients plus historical endpoint replay.
- `source_data/review_morphology_uncertainty/`: label contingency, missingness, mapping, radius and compression audit.
- `source_data/review_response_baselines/`: nested baselines, degradation conditions and split/leakage challenges.

Source-data paths above are relative to the journal root. Public-source access requirements and historical execution limits are retained in the released READMEs; derived Source Data is separate from raw upstream caches.
