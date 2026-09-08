# Response to the panel and completed scientific revision

The panel's criticism was justified: the response overstated several disagreements while leaving coefficient generation and optimization scope insufficiently resolved. The revision preserves **Dendritic morphology as a dictionary for local credit assignment** and its task-to-credit progression. It adds a local implementation where the earlier conductance evidence used oracle coefficients, resolves the slow quartic fits, and states the limits of the anatomical and measured-response evidence more precisely.

This document supersedes the earlier response's recommendations and disputed rebuttals. The historical comparison remains between Wednesday 2 September (`1a3bab3`) and the reviewed release (`2a298a9`); the new experiments below are subsequent work. The panel's scratch runs informed the local-gate protocol but are excluded from every reported outcome. Protocol commits preceded new execution. They are repository freezes, not public preregistrations.

## 1. Local coefficients: the new experiment succeeds

The implementable rule is now explicit. Each terminal compartment receives somatic error multiplied by an indicator that its parent is uninhibited. Proximal compartments and the soma retain ungated error. All rules use the same local eligibilities. The local rule evaluates neither exact paths nor projections of the exact field during learning.

Twenty new seeds were run for both aligned and opposed targets, with nine rules at three common Adam rates. All **1,080 trajectories** continued through 16,384 updates. The primary rate 0.03 and 4,096-update window were fixed before execution. Test outcomes use the state with minimum validation error within the declared window; fixed endpoints are also retained.

| Rule | Aligned NMSE | Opposed NMSE |
|---|---:|---:|
| Exact path | 0.00000753 | 0.00002179 |
| Local distal gate, unit proximal credit | 0.00001055 | 0.00002328 |
| Two leaf profiles, oracle amplitudes, unit proximal credit | 0.00000907 | 0.00002280 |
| Unit broadcast | 0.00055337 | 0.4873967 |
| Swapped distal gate | 0.0053807 | 0.9492567 |
| Gate applied at proximal compartments too | 0.00001196 | 0.1232042 |

The change from the previous 0.664 contrast to the new approximately 0.487 contrast reflects a new twenty-seed cohort and a broadcast-minus-local-gate estimand, rather than altered historical outcomes. The previous cohort remains intact and its historical replay reproduces.

The primary opposed-minus-aligned difference in broadcast-minus-local-gate error is **0.4868 [0.3603, 0.6148]**, positive in all twenty seeds. At the longer window it is **0.4681 [0.3321, 0.6039]**, again positive in every seed. The hard-gate-minus-exact difference on opposed targets is 0.00000149 [−0.00000160, 0.00000486]. This meets the protocol's descriptive practical-precision reference; it is not an asymptotic equivalence test.

The wrong proximal gate freezes both inhibitory gains because their eligibility is nonzero only on inhibited trials, when the gate closes. The correct rule changes all 24 parameters. It succeeds at all three tested rates, and the continuous shunt-proportional rule also succeeds. Exact and proper local-gate fits contact no conductance bounds; most opposed broadcast fits do, so finite-budget and parameter-bound scope remains explicit.

At fixed weights the binary-context path field has rank at most two. The useful distinction is the relative gain between distal subtrees. The third profile in the previous oracle dictionary supplied proximal credit; it was not evidence that another context distinction was required. The new two-leaf-profile control makes that clear experimentally. Local-gate success is expected from coarsening the strong context-dependent gain ratio, and is presented that way. The task still supplies inhibitory context and somatic error; it does not establish their endogenous origin.

**Paper change:** new main Fig. 5, explicit rule and controls in Results and Methods, complete rate/budget sheet in S53, and a coefficient-generation paragraph in Discussion. The old conductance cohorts, wider-bound controls and their source records remain separate.

## 2. The encoder criticism was accurate

The **54.47 percentage-point soft-encoder deficit is the prespecified primary result**. Calling the exploratory hard-readout rescue a correction to that statement was wrong. The Discussion now reports the primary result and treats the rescue as a sensitivity analysis.

The noise-free, sixteen-calibration-trial condition is particularly informative: context identification is perfect, yet soft routing reaches only 19.392% task accuracy. Maximum-probability readout of the same encoder reaches the 80.291% oracle accuracy. The failure is leakage into incompatible routes, not failure to identify context. In the noisy primary condition, exploratory hard readout narrows the oracle gap to 16.26 points but does not eliminate it.

Exploratory hard-readout tables no longer carry the primary-family Holm columns. Their means, intervals and unadjusted tests are unchanged and labeled descriptive. The primary soft-encoder multiplicity family is preserved.

## 3. Longer quartic training resolves the bimodality

All **720 unique algebraic trajectories** were retained across the same twenty seeds, three tasks, four rules, two optimizers and original selected/common-rate union. Every historical checkpoint through 1,024 updates reproduced before continuation to 16,384. No rates were retuned, and no stalled seed was removed. This is an extension of an observed cohort, not fresh confirmation.

At the original budget, exact pairwise credit beats the calibrated profile in **15/20 seeds**; a slow fit reverses their mean ordering. The revised text states this instead of implying that the mean describes the majority. The pairwise mean lag disappears by 4,096 updates.

All twenty selected-rate Adam exact quartic fits reach NMSE **0.0443–0.0484 by 4,096 updates**, close to the 0.045 noise floor. At 16,384, exact quartic mean/median NMSE is 0.0464/0.0461 versus 0.9994/1.0769 for the initial profile. The task-by-credit interaction remains **0.9516 [0.8829, 1.0123]** at terminal selected-rate states, and **0.7903 [0.6915, 0.8797]** with validation-based state selection. Common-rate results remain positive too; all four contrasts are positive in all twenty seeds.

Every selected-rate Adam initial-profile arm contacts coefficient bounds by the longest budget; corresponding exact arms do not. The extension strengthens the bounded, finite-budget contrast and resolves slow exact optimization. It does not prove that fixed profiles must fail under every optimizer or unbounded parameterization. Original Fig. 4 capture checkpoints remain unchanged; S54–S55 and Table S37 display the extension; Source Data retains every control.

The original pairwise term **increases** the interaction by 0.00815; the panel has accepted that arithmetic correction. Fixed positive scaling under Adam explains calibrated/sign-only agreement, not the entire transient exact-credit trajectory. The manuscript does not claim that it does.

## 4. Depth is budget-indexed on both sides

Existing trajectories answer the referee's offered crossover alternative, not the stronger request for convergence. No converged winner is claimed.

- LocalCA exact-minus-shared accuracy is +10.86 points at 180 epochs, reaches −2.94 at epoch 486, and is −1.52 [−1.83, −1.23] at 600.
- The mean remains negative from epoch 315; its return towards zero at the cap is visible.
- Exact LocalCA retains lower mean cross-entropy throughout epochs 180–600. Its loss gap is non-monotone and is plotted beside the accuracy gap.
- Eight of ten D1 references early-stop. All fifty D3 fits and two D1 fits reach 600; D3 validation loss is still falling and selected states are near the cap.
- The exact-BP D3-minus-D1 depth gain grows from 30.82 points at 180 to 38.17 at 600. This compares different optimization maturity under the specified budgets.

Main Fig. 6 restores a compact task/tree drawing and shows accuracy, validation loss, accuracy-gap and cross-entropy-gap trajectories. The 180-epoch figure remains supplementary. “Validation-selected” now describes the selection rule, not a guarantee of convergence.

## 5. Anatomy measures compression of an ancestry-favoring passive family

The field was generated without inspecting the selected dictionary. It is nevertheless structurally ancestry-favoring under passive transport and a somatic loss. Another passive perturbation family would not remove this dependence. A discriminating test beyond these assumptions requires measured fields, non-somatic error sources or active dynamics.

The actual-versus-degree–depth-surrogate comparison now leads: mean capture **0.619 versus 0.578**, difference **0.041 [0.020, 0.062]**, positive in 38/47 cells. The stronger ancestry-versus-depth-bin contrast remains a secondary comparison. Heterogeneity is explicit: in **11/47 cells**, more than half of the 200 surrogates match or exceed the real tree.

The complete focal field has **97.11–99.93%** partition capture, but this does not make the driving-force term small. Its weighted squared energy is **0.99–3.50 times** full-field energy and its correlation with the adjoint term is **−0.979 to −0.913**. The terms largely cancel. The revised exposition reports their energies, correlation and full-field capture together and distinguishes the complete focal partition from the restricted seven-route dictionary. The shunt identity still unifies morphology and conductance gain, with the required basis qualification.

## 6. Single-layer and measured-response scope

DFA changes weight transport; it leaves a single dendritic layer and a fixed random image of the same output error. It does not answer the rank/single-layer objection. The abstract now explicitly restricts the image statement to classifiers with one dendritic neuronal layer, and the Results explain the exact readout errors and what DFA changes.

The measured association is described as **providing no evidence of alignment**, with a conditional sensitivity estimate. A frozen Gaussian response model preserves 13 scans, seven targets, 102 unique presynaptic roots, shared recordings, stimulus identities and repeats. Noise is calibrated to measured split-half reliabilities. Pair-level inputs, contact geometry and required response arrays are now released.

Across 4,000 simulated datasets, 80% detection corresponds to a mean simulated target-level partial rank correlation of approximately **0.249**, under this specified model. The tested ancestry-variance bracket is **0.55–0.60**; power at 0.60 is 80.85% [79.60%, 82.04%]. The lower Monte Carlo interval first exceeds 80% at 0.65. Null two-sided rejection is 4.95% [4.32%, 5.67%]. Reliability calibration discrepancies average 0.0020 without retuning.

Seven units give a discrete minimum two-sided signed-rank P of 0.015625, not a universal ceiling on power. Maximum detection in this particular signal family is 96.23% with measured reliability and 99.78% with perfect reliability. These are model-conditional sensitivities, not bounds excluding smaller effects, other covariance structures or missing partners.

The offline learner remains a separate transfer-geometry diagnostic. Restricted rules have worse mean NMSE than ridge; ancestry-minus-ridge is 0.029 [−0.013, 0.074], so that individual contrast is uncertain. Fig. 9D now includes ridge as a reference line.

## 7. Writing, figures and historical accounting

The Eq. 2 punctuation defect is fixed. Profiles, route supports, matrix columns and trial coefficients are distinguished. Nominal teacher tuning and the students' shared nominal scaffold with independent 0.4 log-space perturbations are explicit. Legenstein–Maass 2011, Poirazi–Mel 2001, Jones–Kording 2021 and Beniaguev and colleagues 2021 are added with a concrete statement of what this Article adds to the predecessor preprint.

The supplementary contents and cohort index already existed; the text now points to them and includes a main-figure guide. The reviewed supplement had **36 tables**. The new extension adds Table S37 and four figures, S53–S56. S35 is a new prospective experiment after Wednesday, not a relocation of Wednesday's utility evidence. Only the physical-depth figure lost the task drawing that is restored here; the prior generalization about removing all task drawings was too broad.

The broad supplementary-PDF change count is removed from the response. The user's corrected accounting is 26 of the 29 referenced items, with S1–S3 unchanged; our same-path canonical-asset blob audit against `1a3bab3` and `2a298a9` still differs for those three, while inherited copies remain identical. Those are different path-level comparisons. The exact-path audit is retained without using a blanket byte-change count as scientific evidence or a rebuttal.

Compared with Wednesday, the framework, branch-conflict and ancestry results preserve the paper's original question. The failed initialization selector moves to supporting evidence because it did not predict the best endpoint; the learned-credit comparison strengthens the task-to-credit connection; the local-gate experiment now supplies a concrete conductance implementation. Depth and anatomy are retained with the limitations the data require. These changes make the narrative more defensible without replacing it with a structure-estimation paper.

All new scientific records, failed controls and previous cohorts are retained. Internal response documents remain outside submission bundles. No manuscript has been submitted or sent to others during this revision.
