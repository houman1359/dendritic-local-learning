# Implementation of the manuscript review — 5 September 2026

This is a revised working manuscript, not a claim that all 76 recommendations or the prospective morphology experiments are complete. The author requested approximately 8,000 main-text words; the journal's approximately 5,000-word guidance was not changed.

The main narrative was reduced from about 13,800 to about 7,970 prose words. The abstract is 172 words. Nine main figures and 31 supplementary figures are retained. Methods remains about 12,900 words and needs a separate consolidation pass.

## Scientific changes supported by new computations

- The general bound uses the actual routed update's mean and second moment; a fixed transform of the exact gradient is only a special case.
- Full-batch/global-curvature utility correlates with archived one-step progress at 0.951 and final accuracy at 0.924. All 270 directly checked actual one-step losses satisfy the bound, and regenerated moments agree within 4.95e-10.
- The four-route ancestry advantage is 1.267 percentage points, paired 95% interval 0.593–1.946, with 15/20 positive seeds. Exact mean-sign-flip P=0.002594, Holm-across-four-budgets P=0.005188. The separately reported Wilcoxon/Holm P=0.01014 accounts for tied integer test counts.
- The deterministic score recovers the mean ancestry-contrast peak at K=4 retrospectively, with 13/20 seed-level optimum matches and 1.94-point mean contrast regret. Ordinary within-family best-bandwidth selection gives 95/100 matches, exactly the maximum-budget baseline.
- A trialwise eligibility-weighted within-span oracle does not rescue subtree update alignment in the measured-response model. These are diagnostics at 130 re-executed exact-learning states, not trained oracle accuracies. Three replayed endpoints exceed 1e-5 drift; maximum normalized-MSE discrepancy is 5.74e-4. No runs were removed.

Two earlier review conclusions are superseded by this reanalysis: the 25 representation-by-family mean cells contain **nine**, not five, distinct nonconstant numerical curves; the corrected original-four-control stochastic contrast selector matches **8/20**, while the full-batch selector matches **13/20**. The older 6/20 result used a different comparator. The original review is preserved as a historical document.

## Files and audit trail

- `main.tex`, `main.pdf`: revised title, abstract, narrative, captions and targeted Methods corrections.
- `supplementary/supplementary.tex` and `.pdf`: corrected theory, tables, captions, object map, linked contents and oracle diagnostic.
- `source_data/review_evidence_reanalysis/`: full-batch scores, paired controls, native effects and direct validation.
- `source_data/fulltree_within_span_oracle/`: re-executed state diagnostics, route identities and replay limits.
- `source_data/review_curve_lineage/`: 78 curve/condition records and the 320-run S8 join. Historical S1/S3 shunting execution is unresolved.
- `analysis/revision_20260905/baseline/`: pre-revision TeX sources. Unified source diffs and agent handoffs are saved in the same revision directory. The manuscript belongs to the nested `drafts/dendritic-local-learning` Git repository, although the outer modeling repository ignores `drafts/`. An isolated clean revision snapshot identifies the software archive without committing the edits on the active branch.

## Recommendation-by-recommendation status

“Implemented” refers to the stated correction or analysis, not evidence for an unperformed experiment. “Partial” identifies remaining work rather than hiding it in a completed checklist.

| No. | Recommendation | Status | Concrete change or remaining work |
|---:|---|---|---|
| 1 | State the paper’s central contribution in one precise sentence. | Implemented | Title, abstract, Introduction and Results now center morphology-constrained credit dictionaries. |
| 2 | Separate optimal feedback allocation from optimal physical morphology. | Implemented | Feedback-only and joint forward/feedback morphology selection are explicitly separate. |
| 3 | Define “optimal” with an objective, feasible set and resource constraint. | Implemented | New morphology-selection equation defines candidate trees, admissible dictionaries, resource cost and a calibration utility; it is labeled a proposal. |
| 4 | Remove the strong “one dictionary per task” conclusion from the current evidence. | Implemented | Removed the one-dictionary-per-task conclusion. Added endpoint baselines, unique-curve counts, regret and retrospective scope. |
| 5 | Do not infer a morphology law from K=8 matching effective rank 7.99. | Implemented | Removed the near-equality of K=8 and effective rank as evidence for a morphology law. |
| 6 | Define task complexity more narrowly than dataset difficulty. | Implemented | Defined credit rank as field-, coordinate- and metric-dependent; distinguished it from dataset difficulty. |
| 7 | Clarify external error dimension versus fixed spatial delivery capacity. | Implemented | Separated fixed profiles, algebraic rank, external error dimension, local gates and state-dependent transport. |
| 8 | Repair the universal parameter-gradient operator claim. | Implemented | General routed-update moment bound precedes the fixed-M special case in main and SI. |
| 9 | Separate hypothetical minibatch noise from the noise used during training. | Implemented | Added full-batch deterministic scores and explicitly retained minibatch64 noise only as a hypothetical diagnostic. |
| 10 | Do not call initial local curvature a general smoothness guarantee. | Implemented | Recomputed scores with the global block-logistic curvature bound and validated actual one-step inequalities. |
| 11 | Correct the expectation-of-capture statement. | Implemented | Separated mean per-field capture from aggregate energy capture; corrected uncentered second-moment terminology. |
| 12 | Reframe branch conflict as a branch-selection requirement in the tested model. | Implemented | Branch experiment now tests access to a selector; inactive exact derivatives are explicitly zero. |
| 13 | State the assumptions behind the conflict threshold and distinguish it from chance crossing. | Partial | Qualified the centered mean-field threshold and distinguished trained chance crossings. No new full-trajectory gradient-alignment sweep. |
| 14 | Report the optimizer interaction beside the physical-depth credit benefit. | Implemented | Main and S31H report 10.96 pp under LocalCA versus 0.64 pp under standard BP optimization, plus the optimizer-change cost. No complete interaction test is claimed. |
| 15 | Choose a more specific title while keeping its claims defensible. | Implemented | Title: Dendritic morphology as a dictionary for local credit assignment. |
| 16 | Rewrite the abstract around the question, construction, main learning result and biological boundary. | Implemented | 172-word abstract retains the fixed random between-neuron feedback result and removes optimum-selection overclaim. |
| 17 | Shorten the Introduction to approximately 600–700 words. | Implemented | Introduction now opens with distributed synapses and the unresolved morphology question; textbook setup is compressed. |
| 18 | Use a four-step argument throughout the Results. | Implemented | Rewrote Results around question, manipulation, controls, outcome and scope; retained complete protocols outside the main narrative. |
| 19 | Cut the main narrative to approximately 5,000–6,000 words through consolidation. | Implemented with author target | Approximately 7,970 narrative words against the requested 8,000 working target. Journal 5,000-word guidance remains separately reported. |
| 20 | Consolidate Methods rather than moving all repetition to the supplement. | Deferred | Methods remains long (about 12,900 prose words). Full consolidation with the SI is a remaining editorial pass; this revision preserves protocol detail. |
| 21 | Explain the unification through a concrete object table. | Implemented | Replaced SI Table S14 with the five-construction map: coordinates, profiles, rank, signs, metric, coefficient source, state dependence and forward model. |
| 22 | Make the distinction from earlier work explicit and fair. | Implemented | Fairer treatment of cable-learning approximations and related dendritic learning; the chain rule is not presented as the new contribution. |
| 23 | Bring the relevant biological precedent into the Introduction and reduce its role as a new result. | Implemented | Francioni motivates neuronal specificity in the Introduction; its re-expression now appears only in SI. |
| 24 | Reorder the figures around the basis question and restore physical morphology to the main sequence. | Revised decision | Kept the nine-figure sequence. Physical depth stays in S31 with the optimizer comparison; merging Figures 1/2 and adding prospective morphology selection remain coupled decisions. |
| 25 | Figure 1: simplify the conceptual entry point. | Partial | Shortened Figure 1 caption and clarified state-dependent transport. Full artwork consolidation remains. |
| 26 | Figure 2: retain its concrete matrices, but merge the roadmap material. | Partial | Preserved atlas matrices and corrected roadmap claims. Did not merge the atlas into Figure 1. |
| 27 | Figure 3: foreground paired feedback effects and keep feedback definitions consistent. | Partial | Preserved paired effects and DFA, clarified strict scalar versus matched-width diagnostic. No new strict-scalar checkpoint diagnostic. |
| 28 | Figure 4: replace the crowded synthesis with testable predictions. | Implemented | Figure 4G uses deterministic/global-curvature utility; H is a conceptual map without shaded phase boundaries or optimum ring. |
| 29 | Figure 5: keep the clean causal manipulation, correct the interpretation and show gradients. | Partial | Figure 5 shows zero inactive exact gradients and corrected gating language. Existing crossing display remains; no extra training-time gradient sweep. |
| 30 | Figure 6: make the ancestry-specific result visually central. | Implemented | Figure 6H enlarges the paired K=4 contrast; I exposes all four controls with declared multiplicity families. |
| 31 | Figure 7: keep independent fields and route-generated controls visibly separate. | Implemented | Figure 7E/F explicitly identify route-generated fields; main distinguishes coarse geometry, dense-relative capture and coefficient costs. |
| 32 | Figure 8: prioritize physical calibration and avoid implying a plasticity experiment. | Implemented | Main foregrounds passive calibration and labels active calculations as fixed-baseline-Jacobian sensitivities, including gate derivatives. |
| 33 | Figure 9: focus the ending on the measured-response test. | Implemented with follow-up nuance | Figure 9C uses native units; D/E alignment control stays; external animal panel moves to SI; within-span geometry diagnostic added. |
| 34 | Improve figure readability at the size used in the manuscript. | Partial | Targeted clipping, labels, figure density and caption sizing improved and compiled pages checked. Broad schematic consolidation remains. |
| 35 | Keep the main captions self-contained while shortening their explanatory burden. | Implemented | Eight main captions rewritten; all nine below 350 words. Paired units, controls and interval definitions retained. |
| 36 | Reorganize the supplement by the revised main-figure order. | Partial | Added linked SI contents, paginated notation, updated object table and callouts. Full reordering by experiment family remains. |
| 37 | Resolve historical input-validity status at the level of plotted rows. | Partial; evidence limit explicit | Audited 78 curve/condition records and 320 S8 runs. S1/S3 historical shunting execution remains unresolved; those series are excluded from conductance-valid inference. |
| 38 | Audit supplementary error bars against the plotting code, not only against captions. | Partial | Corrected verified S7, S17 and S24 uncertainty descriptions. Unverified legacy SEM-versus-bootstrap issues are not asserted resolved. |
| 39 | Use the following per-figure supplement revision list. | Partial | Fixed targeted S6/S7/S8/S10/S16/S21/S23/S26/S29/S30/S31 display issues; detailed generator handoff is saved alongside this report. |
| 40 | Report the ancestry-specific control comparison without hiding its small size. | Implemented | Original four-control comparator, individual seeds, confidence interval, exact mean-sign-flip and restored-count-grid Wilcoxon diagnostics are supplied. |
| 41 | Explain the role of each “oracle.” | Implemented | Separated fitted capacity oracles, prescribed trained PCA reference, exact transport, and re-executed within-span geometry oracles. |
| 42 | Strengthen prediction validation with comparisons that can fail. | Partial | Added within-family prediction, degenerate-score abstention, maximum-budget baseline, regret and deterministic comparison. No held-out-task prospective selection. |
| 43 | Use isospectral rotations to isolate alignment from rank. | Deferred experiment | Corrected the mixture interpretation; an isospectral rotation sweep remains to be run. |
| 44 | Correct the necessity language around laminar optimality. | Implemented | Laminar spectral result is sufficient, with projector-domain and leading-eigenspace qualifications. |
| 45 | State the finite-grid conditions for an interior optimum. | Implemented | Finite-grid crossover conditions and generator-designed resolution optimum are explicit. |
| 46 | Distinguish static column gains from physical conductance changes. | Implemented | Nonzero diagonal gains preserve span; physical shunts can reshape spatial transfer. |
| 47 | Use resource matching appropriate to the claim. | Implemented | Resource matching is scoped by comparison; MLP is only approximately parameter matched and S31 bracket excludes it. |
| 48 | Treat capture per coefficient as a proxy, not physical wiring optimality. | Implemented | Nonzero routing coefficients are identified as a proxy, not a physical wiring or energy optimum. |
| 49 | Keep anatomical inference at the scale the samples support. | Implemented | Same-mouse disjoint cohorts, second-mouse structural check and external six-animal context are separated throughout. |
| 50 | Give coarse geometry the credit supported by the controls. | Implemented | Coarse depth and branching geometry receive the supported attribution; fine-topology limits are stated. |
| 51 | Narrow the measured-response null to the actual test. | Implemented | Null is scoped to measured visual responses, fixed fields and seven targets; trialwise within-span oracle geometry also remains unfavorable. |
| 52 | Quantify the biological test’s sensitivity before interpreting absence of alignment. | Partial | Native-unit target contrasts and within-span oracle diagnostics address sensitivity. No complete missing-data-pattern/power sweep. |
| 53 | Specify what the shunting result predicts about learning. | Partial | Revised branch-plasticity prediction and specified a shunt-minus-current contrast. No new biological plasticity experiment. |
| 54 | Make the physiological boundary visible and the proposed experiment discriminating. | Implemented | Passive boundary is prominent and current-control effect is not required to vanish. |
| 55 | Correct the task-generator equations against the executable generator. | Implemented | Main/Methods/SI use noise-floor-gain order; rectification and the shared flat-factor partition are stated. |
| 56 | Give every experiment a compact reproducibility row. | Partial | Expanded dictionary map and release metadata; full compact experiment/cohort table remains part of Methods consolidation. |
| 57 | Simplify statistical reporting while retaining its strengths. | Implemented | Units, paired samples, resampling hierarchy and distinct multiplicity families clarified; no new biological replication inferred. |
| 58 | Use formal equivalence language only where an equivalence criterion was tested. | Implemented | Limited equality language to algebraic identity or explicitly tested practical equivalence. |
| 59 | Distinguish prespecification from public preregistration and independent replication. | Implemented | Retrospective analyses, prespecified configurations, same-seed reruns and independent biological samples remain distinguished. |
| 60 | Regenerate and validate the actual submission package. | Implemented | Rebuilt main, SI, combined PDF, Source Data, software, Overleaf and submission archives; see validation record below. |
| 61 | Update stale documentation rather than relying on checked boxes. | Implemented | Updated nine-main/31-SI maps, title, animals, source files and release documentation. |
| 62 | Complete accessible data/code release details. | Partial | Local reviewer archives rebuilt with new code/data and provenance. Public archival DOI and final availability declarations remain author actions. |
| 63 | Shorten and correct the cover letter around the final scientific claim. | Implemented | Cover letter and editorial summary rewritten around dictionary constraints, retrospective evidence and conditional physical effects. |
| 64 | Resolve related-manuscript status with the authors at submission. | Author action | Current conference-review status and final submission declarations are not inferable from this workspace; documents retain explicit author checks. |
| 65 | Make the Discussion synthesize mechanisms instead of reciting every experiment. | Implemented | Discussion synthesizes information requirements, representation, optimization and biological limitations. |
| 66 | Apply a targeted sentence-level edit after settling the claims. | Implemented | Main narrative rewritten and targeted SI/Methods prose corrected; citations and equations validated. |
| 67 | If adding one major experiment, make it prospective morphology selection. | Deferred experiment | Defined the prospective costed morphology-selection problem and held-out test, but did not run a new joint morphology sweep. |
| 68 | If the new experiment is restricted to backward routing, name its result accordingly. | Implemented | Feedback-only selection is explicitly distinct from forward-plus-feedback morphology. |
| 69 | If adding a second experiment, test how coefficients are obtained. | Deferred learning experiment | Within-span oracle geometry was evaluated, but no new coefficient-learning/biological encoder trajectories were trained. |
| 70 | Prioritize a small set of existing-data analyses before launching more training. | Implemented | Completed deterministic utility, within-family/regret, K=4 controls, optimizer panel, native-unit contrasts and within-span replay analyses. |
| 71 | Preserve the strongest controls and negative results. | Implemented | Preserved exact/DFA controls, point equivalences, low-budget ancestry losses, standard-passive boundary and measured-response null. |
| 72 | Revise in dependency order. | Implemented | Claims and mathematics were settled before revised displays and final archive construction. |
| 73 | Correct table-level definitions and units, and add missing table callouts. | Implemented | Corrected SI table metrics, units, sample scope and missing callouts; all 35 numbered tables retain references. |
| 74 | Finish the bibliography and small textual corrections after restructuring. | Implemented | Added three mathematical references, corrected Peña accent, and checked all cited keys. |
| 75 | Give label uncertainty and morphology preprocessing a clearer quantitative account. | Partial | Direct-label coverage, proxy use and coarse-geometry limits are explicit. No new label-imputation or mapping-sensitivity experiment. |
| 76 | Make historical reproducibility gaps explicit without understating the current release. | Implemented | Historical config/checkpoint gaps, replay drift and current hash-identified release are separately described. |

## Validation and remaining decisions

Final validation results are recorded in `revision_20260905/VALIDATION.md`. The builds retain author-specific declarations as pending: conference-review status, all-author approval, official forms and public permanent release identifiers. No external submission or publication was performed.

The next scientific decision is whether to pursue a prospective physical-morphology selection experiment. Its design should use fixed-spectrum rotations, genuine minibatch updates when scoring noise, a held-out task/seed split, explicit costs and separate feedback-only versus joint-design arms. The current manuscript presents that objective as a proposal. The main remaining editorial tasks are Methods consolidation and deciding whether to merge Figures 1 and 2 to make room for that experiment.
