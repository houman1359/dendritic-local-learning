# Response to the 6 September panel review

This is an assessment and implementation plan, not a claim that the proposed revisions or training experiments have been completed. It addresses the panel ledger, the current manuscript, the last committed manuscript at `560d5f2`, the final 5 September release, and independent checks of the central scientific and release criticisms. The companion audits are in `panel_response_20260906/`. No manuscript, publication figure, source result, or release archive was changed during this assessment. The anatomical broadcast comparison described below is a new, explicitly post hoc calculation from existing data; no new model fits were run.

The central criticism is justified. The paper has gained useful science but lost a clear hierarchy of claims. The previous review concentrated on definitions, preservation of results and visual consistency. Those checks did not establish that the experiments justified the narrative, and they missed concrete semantic errors, including an incorrectly registered axis. The previous statement that no visual findings remained was too broad.

## The answer the paper should give

**Within the tested model classes, task interactions determine which dendritic groupings can represent a target; estimation and credit delivery determine how much of that capacity learning realizes.**

This is an organizing claim with three different kinds of evidence:

1. **Represent:** Which interactions can pass through a scalar branch output? The centered-cut condition answers this exactly in the multi-affine model. Separate conductance experiments show that compatible grouping can also matter under different branch equations.
2. **Estimate:** Can a useful grouping be inferred without knowing the target function? The noisy-example estimator improves on the fixed tree and the tested short fitting pilot. Its advantage over a thoroughly fitted pilot is not established.
3. **Learn:** Can the learner exploit the chosen structure using its available feedback? Exact credit helps on some tasks and recipes, while broadcast works well on others. Anatomy and conductance state constrain the available spatial profiles, but those constraints alone do not establish that neurons use them for learning.

The route dictionary remains useful as a description of feedback delivery. It is not the mathematical object in the forward representability theorem or the Fourier interaction estimator. The paper should distinguish a forward representation from a feedback route dictionary explicitly. It must also distinguish the target's **input-sensitivity spectrum** from the learner's **parameter-credit spectrum**: the new isospectral construction equates the former, not the latter.

A working title is **“Task interactions and credit delivery constrain learning in dendritic trees.”** The final title should follow the evidence after the essential controls, rather than promise an optimal biological morphology.

A five-sentence abstract can be built around the following draft; this is proposed language, not a replacement already made in the manuscript:

> Dendritic branching changes both how synaptic inputs are combined and how local changes influence a neuron's output, but these roles impose different requirements on learning. Here we distinguish the capacity to represent a task, the ability to estimate a useful tree from examples, and the credit signals needed to learn its parameters. In a scalar tree model, interactions across branch boundaries characterize representability, and estimating those interactions from noisy observations improves structure selection and subsequent prediction relative to a fixed tree. Compatible input grouping also benefits conductance-based models, whereas the additional benefit of spatially resolved credit depends on the task and training conditions. Reconstructed arbors provide candidate feedback routes, while cable and measured-response analyses delimit the roles of conductance state and transfer geometry, linking structural capacity to explicit conditions for learning.

## What has changed relative to the earlier paper

| Earlier interpretation | What the present evidence supports | Consequence |
|---|---|---|
| A moment-based utility can guide useful or optimal morphology. | The one-step bound is sound. Its deterministic retrospective score is a scaled clipped cosine. Its fixed-step relative failed prospective final-structure selection against a fixed tree in all 20 seeds in each arm. | Keep the bound as a local diagnostic; state the prospective failure with its comparator. Do not use successful forward selection as validation of this score. |
| Task rank/complexity can specify dendritic structure through a basis description. | Equal input-sensitivity spectra can conceal different interactions and compatible groupings. The centered-cut criterion provides a constructive condition in a restricted scalar model. | A stronger and more precise forward-structure result has been added, but it addresses a different object from the credit dictionary. |
| Estimated interaction structure provides a superior selector. | It beats the fixed and two-sweep pilot baselines under the tested protocol. The ledger's converged-pilot reanalysis substantially narrows the difference. | Benchmark selection error against labels and actual computation before claiming superiority over fitting-based selection. |
| Finer dendritic credit generally explains the learning benefit. | Neuron-specific feedback accounts for most image-task gains; exact paths are slightly worse on CIFAR-10. Credit advantages in algebraic and physical-depth tasks depend on feedback definition and optimization. | Make conditional learning the claim and test the missing matched controls. |
| Coarse anatomical routes outperform alternatives and compress independent fields. | The field and positive control are different endpoints. The leading anatomical field mode is well approximated by a broadcast; dictionaries were not equally equipped to capture it. | Recompute all families with the same broadcast allowance and account for wiring cost. |
| Shunting demonstrates a new spatial mechanism beyond column gains and robustness to active channels. | Passive adjoint gains are piecewise constant on ancestry-defined regions. The channel ensemble is a weak perturbation of a passive operating point. | Use the exact gain structure to connect cable transport to routes; narrow the channel claim. |
| The measured-response learning comparison tests endogenous subtree alignment. | Much of its comparison concerns retention of a nearly fixed transfer profile by sparse, incompletely covering dictionaries. A separate response-similarity analysis remains informative but limited. | Separate the geometry diagnostic from the empirical structure–function test. |

The revisions therefore did not simply make the same claim stronger. They weakened a general credit-to-optimal-morphology inference while adding a more constructive account of forward grouping. This change should be stated directly and used to organize the manuscript.

## Scientific corrections and qualified disagreements

### Shunting: correct the false dichotomy, specify the basis

For passive conductance matrix `G`, let `Z = G^{-1}` and let `q = Z e_soma` be the soma-loss adjoint apart from its scalar error. Adding conductance `kappa` at site `k` gives

`q' = q - [kappa q_k / (1 + kappa Z_kk)] Z_:k`.

For every descendant `i` of `k`, tree factorization gives exactly

`q'_i / q_i = 1 / (1 + kappa Z_kk)`.

Other sites share a gain when they belong to the same lowest-common-ancestor class relative to the shunt. This makes the adjoint response a useful bridge to ancestry-based spatial profiles. However, diagonal gains are exact in a suitably defined, baseline-transfer-weighted partition basis; they are not automatically diagonal column gains on any chosen overlapping subtree dictionary. Full synaptic gradients additionally contain the changing driving force, so their within-subtree ratios need not be constant. The manuscript and the report both need this qualification.

At more compact passive calibrations, attenuation of descendants can persist while off-route sparing diminishes. The main physical result should concern **spatial selectivity**, with signed on-route and off-route changes and absolute doses shown. An unsigned localization difference alone can obscure whether the control attenuates or amplifies gradients.

### The channel ensemble is a linearization check

The recorded model includes voltage-dependent channels, so it is not literally devoid of active terms. But their contribution to the operator at the chosen operating point is small and the localization closely tracks the passive calculation. Present this as a weak-channel, fixed-Jacobian sensitivity check; report the gating equations, density distributions and active/passive operator fractions. A broad active-dendrite claim would require a separate physiological investigation, not a few density increases chosen because they retain the desired ordering. That expansion is not needed for the revised central thesis.

### Physical depth: fixed budget, not established convergence

The cap issue is real. The headline depth and credit contrasts must be labeled as outcomes at 180 epochs. The report summary overstates the D1/D3 stopping distinction: many D1 fits also reached the cap, but their curves were comparatively flat. The critical fact is continued improvement in D3, especially under different learning-rate recipes. Existing controls locate an optimizer dependence; they do not identify learning rate, parameter splitting or asymptotic performance as its cause. The four-factor depth optimum also depends on the learning rule; the BP statement cannot be generalized to shared-soma LocalCA.

### Anatomical fields: a promising correction, not a rank-one collapse

Independent reanalysis of all eight arbors gives mean rank-one oracle capture of **0.326 of total field energy** and uniform-broadcast capture of **0.314**. The latter is **0.961 of the rank-one oracle on average**, not 96% of the total field or exactly 96% in every cell. The independent anatomical field is therefore not nearly rank one. The measured-response field discussed below is a different case.

The existing comparison nevertheless favors depth bins by giving them a constant component that the selected subtree dictionary lacks. Recomputed dictionaries containing a constant and `K-1` selected routes yield:

| Total profiles K | Broadcast plus selected routes | Depth bins | Cells favoring augmented routes |
|---:|---:|---:|---:|
| 2 | 0.402 | 0.330 | 7/8 |
| 4 | 0.474 | 0.374 | 8/8 |
| 8 | 0.585 | 0.445 | 8/8 |
| 16 | 0.750 | 0.572 | 8/8 |

These are new post hoc calculations from frozen anatomy, not prespecified replication results. At K=8, the augmented dictionary has 17.25% nonzero density, versus 6.35% for the original subtree dictionary and 12.5% for depth bins. The previous efficiency headline cannot be transferred to this new comparison. Every family needs the same common component, and marginal capture must be reported with its cost. A weighted residual analysis after removing the broadcast is a complementary comparison, with its residual-energy denominator stated explicitly.

The cable-generated target is independent of the selected dictionary's fitted coefficients, but it shares the tree's structural constraints. Describe it as generated by cable physics rather than sampled from the tested dictionary. The 47-cell and second-mouse results currently replicate the favorable route-generated field, not this cable-field comparison. Repeat the corrected cable analysis on the disjoint cohort before making a replicated anatomical claim.

### Measured responses: distinguish two tests

The complete-tree learning field is overwhelmingly described by output error times a nearly fixed transfer vector. The four selected routes cover only about **48% of input sites on average**, and most are single-site distal supports. This undermines the interpretation of the learning comparison as a broad test of subtree alignment and requires a literal drawing of the implemented routes.

The separate ancestry–response similarity test does use measured response relationships and should not be dismissed as null by construction. Report that result with its target-level uncertainty. Treat the learning comparison as a diagnostic of transfer-profile coverage unless a revised pipeline first demonstrates sensitivity to spatially varying credit. Include the unprojected frozen transfer profile, ridge baseline, support/coverage statistics and exact small-sample tests.

An imposed field whose projection is correct by construction does not establish sensitivity of the measured-response pipeline. Nor does simply replacing target labels necessarily create trial-varying credit in a near-linear soma-readout model. A valid positive control must demonstrate the intended spatial variation before applying the same missing-data, training and evaluation procedure. Any sensitivity calculation should be simulation-based or expressed through interval-supported effect bounds, not retrospective power computed from the observed effect.

### Utility and attribution

The deterministic score is proportional to squared positive update cosine within a seed; its correlation with first-step progress does not test noise rejection. However, the report's bandwidth-only interpretation is too strong: within-budget correlations with accuracy remain approximately 0.76, 0.86 and 0.90 at K=1, 2 and 4. The failure is in using this association to claim selection of the best final model, not in the existence of an alignment association.

The original prospective selector used the fixed-step descent bound, not exactly the optimized-step utility. It produced regret approximately 0.0806/0.1040 against 0.0449/0.0427 for the fixed maximum-budget tree, losing in every seed in both arms. State that comparison explicitly. The improved finite-horizon forecast remains distinct from the interaction estimator and its simple strong baseline must remain visible.

The K=4 ancestry result is a real, small learning difference, but its location follows the deliberately chosen task coefficients and the full-rank tie. Present it as a controlled demonstration with those coefficients disclosed. Likewise, Boolean grouping experiments test whether training approaches an analytically constrained solution, not whether random variation can overturn a proven representational obstruction.

The tree-tensor/hierarchical-Tucker origin of the rank argument and the standard smoothness origin of the moment bound need direct attribution. The relevant primary foundations include [Hackbusch and Kühn](https://files-www.mis.mpg.de/mpi-typo3/preprints/2009/preprint2009_2.pdf) and [Grasedyck](https://files-www.mis.mpg.de/mpi-typo3/preprints/2009/preprint2009_27.pdf); the constant-preserving tensor identification is derived in the companion theory audit. The defensible claim is a centered, constant-preserving specialization and application; whether even that specialization is mathematically novel requires careful comparison, not a new novelty assertion. The predecessor preprint must be cited at the inherited derivation and in the main account of what this paper adds, regardless of the NeurIPS decision.

## Follow-up work that can change the claims

1. **Depth and optimization:** Extend the existing paired aligned-D3 arms and matched D1 controls with a predeclared longer budget, record best-validation epoch and late loss slope, and retain complete trajectories. If the curves still decline, report fixed-budget behavior. Separate credit rule, global learning rate and parameter-group recipe before attributing a cause. Report learning curves and computation as well as endpoints.
2. **Selection baseline:** Compare the estimator with converged fitting pilots using the same labels, validation split and candidate pool, several prespecified restart budgets, and measured computation. The ledger already reports that a stronger pilot narrows the advantage; independently reproduce that screen, then freeze comparisons before fresh confirmation. The supported result may be comparable selection with less computation rather than lower error; either outcome is useful if measured.
3. **Credit-rule bridge:** Evaluate unit broadcast, a frozen calibrated transfer profile and exact transport in both multi-affine and positive-conductance models with comparable development tuning. Sign-changing derivatives are a mechanism to test, not an explanation already isolated by Fig. 6. The successful broadcast XOR case must remain.
4. **Anatomical common mode:** Complete all-family broadcast-augmented and residual-field comparisons, preserving total channel and wiring accounting, then apply the frozen procedure to the disjoint anatomical cohort. Report the original eight-cell result as development/post hoc.
5. **Measured-response sensitivity:** First determine whether the scientific question is identifiable in the available pipeline. Add matched coverage/common-profile controls and a sensitivity benchmark if it is. If it is not, narrow the section to the transfer-geometry diagnostic and the separate observed structure–function analysis.

The first four directly address claims intended for the main paper. The fifth determines the appropriate interpretation and prominence of the measured-response material. No experiment should be repeated simply until a favorable conclusion appears. Preserve failed comparisons, distinguish exploratory corrections from new confirmation, and change the claim if the stronger baseline succeeds.

## Proposed manuscript and figure organization

The current ten Results sections give unequal evidence equal prominence. Organize the argument around representation, estimation and learning, followed by anatomical realization and biological reach. The following is a seven-figure working layout, to be finalized after the decisive controls:

1. **Define spatial credit and establish the image-task reference.** Merge selected panels from current Figs. 1 and 2: one eligibility/route drawing, one visible population-feedback comparison, and the principal accuracy and field-decomposition results. Move redundant dictionary schematics and secondary controls to SI; do not squeeze both complete figures into one.
2. **Represent, estimate, then learn.** Simplify current Fig. 6 to the interaction/cut example, noisy-example selection with a strong fitting comparator, and learning on chosen trees. Draw the actual AND/OR/XOR and higher-order tasks. Keep the full census, optimizer variants and conductance replication detail in SI with clear main-text pointers.
3. **When branch-specific credit helps.** Combine selected current Fig. 4 and 5 panels. Show the task generator, the conflict-dependent result, and the ancestry contrast on an adequately sized axis. Remove the large derangement effect from the axis intended to resolve small matched-control differences. Retain the design-implied low-budget signs and full-rank tie.
4. **Physical depth and learning dynamics.** Promote a compact S31, including the task/architecture distinction, sensor-placement control and optimizer comparison, updated with budget/convergence evidence. This gives the physical morphology argument a main figure.
5. **Anatomical profiles beyond a broadcast.** Put the corrected cable-field comparison and marginal capture/cost first; label route-generated results as positive controls and distinguish replication cohorts. Avoid an efficiency headline based solely on the favorable generated field.
6. **Shunting and spatial selectivity.** Show the ancestry gain identity and signed descendant/off-route effects. Give the calibrated electrotonic comparison a wide axis and state doses in physical units. Move the weak-channel check to SI unless it adds a distinct result.
7. **What measured responses can test.** Draw the actual selected support, show the empirical structure–function result and the relevant transfer/ridge controls with their uncertainty. Remove current 9F and move the by-construction alignment panels to SI. Its main prominence should depend on what the identifiability audit supports.

Remove current Fig. 3H. Most analytic phase screens belong beside their derivations in SI; retain the one-step bound and its limited empirical role in concise prose. The prospective failure stays explicitly in the main text near the structure-selection claim, even if its full figure is supplementary.

One shared palette is insufficient: the same sign and semantic category must retain the same visual meaning. Correct the inverted good/bad heatmap convention, define family encodings, standardize condition names and confidence-interval labels, and use automatic panel/figure references. Review each key contrast at final print size. S8F needs an explicit data-coordinate-to-tick check; preserving the numeric artist arrays did not detect categorical misregistration.

Reorganize SI by scientific topic, give morphology theory its own section, place figures close to their explanations and number them through generated references in order of citation. Every caption must describe marks actually drawn. Keep full task definitions and reproducibility details, but remove repeated Results and revision chronology from Methods.

## Length and Discussion

The existing approximate 8,000-word preference should not be silently replaced by the panel's 5,500-word target. Nevertheless, there is substantial removable repetition. First produce a coherent shorter draft, provisionally around 6,000–6,500 narrative words, with roughly 3,500–4,000 words of main Methods and complete protocols in SI. These are editorial working targets, not acceptance requirements or already achieved counts. Preserve necessary explanation rather than filling or cutting to a quota.

Nature Communications describes 5,000 main-text words as an ideal and Methods as typically under 3,000, while explicitly allowing reasonable flexibility at initial submission: [official article guidance](https://www.nature.com/ncomms/submit/article). The compelling reason to shorten this paper is its fragmented argument, not a claim of automatic rejection over length.

Use three interpretive Discussion paragraphs for representation, estimation and learning, followed by biological predictions and scope. Each paragraph should state what the reader should believe and why. Consolidate duplicated limitations, but retain the scope needed to understand a standalone caption and the central negative results. Replace repeated negations with bounded positive statements where possible; do not mechanically delete qualifiers according to a hedge count.

## Mechanical corrections and release plan

The S8F depth shift, S2 `extbfE` corruption, erroneous Greedy DOI, inconsistent ancestry P-value families, CIFAR ordering statement, missing task definition, mislabeled residual-norm quantities and stale figure counts require direct correction. The noise-resilience task is low-rank-noise MNIST, and its generator is absent from the reviewer software archive. Defining it in prose is insufficient: the released code must run it.

Choose the ancestry primary analysis based on the recorded protocol, not the smallest P value; identify later sensitivity tests separately. Explain differing estimands and normalizations once in a central table and use them consistently. Preserve the corrected biological scope when checking all captions, abstracts, cover materials and SI summaries.

The release criticism is substantially right but the commit wording needs correction. Commit `5483817839c16a9d48c343062dee4f6340ed84cd` is real in the retained isolated snapshot; it is not available in the canonical paper repository. The final PDFs and archives are internally consistent. The required repair is durable, reachable provenance, not reconstruction of nonexistent scientific data.

Before implementing changes, preserve the current state and establish a named revision branch/checkpoint containing the paper-specific work and source lineage. Keep concurrently edited project work separately attributed. Build the final release from a clean reachable commit, include the required history or repository bundle, and verify its resolution in a clean clone.

Replace broad software exports with a dependency-aware allowlist. Exclude the 187 MB PowerPoint and unrelated transformer work from the release without deleting their working copies. Include all task generators and required dependencies. Exclude internal review/revision logs from journal-facing bundles while retaining scientific protocols, computational provenance and the required assistance disclosure. Use one canonical producer for each figure; test that retired builders cannot overwrite the final native panels. Build and exercise the minimal released software in a clean environment, including the previously absent dataset generator, before rebuilding the Overleaf, Source Data, software and submission packages.

Completion means that the narrative's central claims survive the fair controls, each plotted contrast has the correct scientific meaning and source registration, all manuscript materials agree, and the minimal committed release reproduces its intended entry points. A successful PDF build and matching file hashes remain necessary checks, but are not sufficient evidence of that outcome.

Companion audits: [theory and learning](panel_response_20260906/theory_learning.md), [scientific framing](panel_response_20260906/scientific_framing.md), and [release and figures](panel_response_20260906/release_figures.md). The physical-depth trajectories and converged-pilot extensions are explicitly attributed to the ledger's verifier: their recorded scratch files were not available at those paths for an independent rerun during this assessment. Their reported outcomes motivate the follow-ups above; they are not newly confirmed experiments by this audit.
