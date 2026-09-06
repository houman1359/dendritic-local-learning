# Reassessment of the prospective morphology test

This is a post-hoc diagnosis of the frozen experiment, prompted by the author's concern about its implications. No protocol, outcome, manuscript or release archive was changed during this assessment. It does not report an improved selector or a new confirmatory experiment.

## What failed

The tested endpoint predictor extrapolates a single initialization update-moment bound to 256 SGD updates using one development-fitted scalar per arm (1.88264 for feedback-only and 2.75318 for joint transfer). Its failure does not refute the local inequality, which passes numerical checks, or show that every dictionary-based morphology predictor must fail. It does establish that this particular extrapolation is inadequate.

Failure has a clear rank dependence. Mean selector regret in feedback-only tasks is 0.04897, 0.03303, 0.01945 and 0.22075 at ranks 1, 2, 4 and 8. The corresponding joint-arm rank-eight regret is 0.27785. Low-rank tasks pay unnecessary routing cost; rank-eight tasks lose predictive accuracy through insufficient selected capacity. The known-rank baseline receives generating information, so its success must not be described as an independently estimated task-complexity prediction.

## What the construction can establish

Write the allowed weights as W=Phi Z, with orthonormal route columns Phi, forward transfer H, context subspace Q_r and orthogonal teacher T. The task is a reduced linear regression with design G=Q_r^T H Phi. If G has rank r, unrestricted Z can realize every target even when the route span does not contain the original context-credit field. Geometry affects conditioning and finite-time learning; it need not change the attainable endpoint.

The independent theory check found rank(G)=min(r,K) in all 6,400 unique confirmatory designs after removing duplicate noise conditions. Since Q_r^T T has orthonormal rows, the unregularized population minimum half-MSE is

    noise_variance/2 + (r - min(r,K))/(2r).

This formula concerns the population optimum, not the actual finite-sample/noisy-SGD endpoint. Under the imposed cost, each effective mode below rank r reduces ideal loss by 1/(2r), which exceeds the 0.01 per-channel penalty for every tested rank. Beyond r, only cost increases. The contiguous balanced tree has the smallest declared cable cost. Consequently, the cheapest tree with K=r is the ideal asymptotic choice throughout this particular confirmatory family. That relation is largely a property of the construction, rather than evidence for task-specific detailed morphology selection.

There is a further geometric preference. The rotation generator obeys A^2=-I, so Q(theta)=(cos(theta) I+sin(theta) A)Q_Haar. For the original contiguous rank-r Haar span, the restricted overlap has a minimum singular value at least |cos(theta)|. At the held-out angles it is well conditioned. The independent check found median condition number about 1.004 for contiguous K=r at ranks 2/4, versus about 4.81–9.46 for other trees. Rotations really change lower-rank task subspaces, but remain anchored to the favored hierarchy. At rank eight the task second-moment projector is the identity and rotation cannot change it. In the feedback-only arm, all trees also share exactly the same K=1 and K=8 projectors.

## Recommended next test

1. Derive a finite-training-horizon predictor using the restricted learning operator: distinguish unattainable target components, convergence rates, estimator/SGD noise and resource costs. A single initial progress value cannot generally summarize all four. Develop and audit it on existing development/diagnostic data; do not recast those data as fresh confirmation.
2. Establish an experiment where optimal morphology genuinely changes across task families. Use independently varied task interaction structures/hierarchies, including comparisons at the same rank and resource budget. Local input masks, admissible synaptic signs or nonlinear forward compartments should prevent arbitrary compensation by an unrestricted dense weight matrix. Verify these manipulations have the intended consequences before treating the family as a confirmatory test.
3. Compare against estimated-rank, fixed-tree, maximum-budget and matched-computation short-pilot-training baselines. Report feedback-only and joint physical designs separately. Generating rank is an oracle diagnostic, not the default deployable baseline.
4. Freeze the revised method, baselines, costs and success criteria, then test genuinely new seeds and task families. Retain all outcomes.

If the new method succeeds, the initial-score failure can serve as a supplementary motivation for including learning dynamics. If it does not, the Article should center conditional credit-routing benefits and morphology-constrained dictionaries, and present prediction of optimal physical morphology as unresolved. Reducing the present experiment's prominence is justified by its limited scope; retaining a broad endpoint-prediction claim while omitting directly relevant counterevidence would not resolve that claim's weakness.

The finite-time spectral direction has methodological precedent, not validation for this particular model: Lampinen and Ganguli, *An analytic theory of generalization dynamics and transfer learning in deep linear networks*, https://arxiv.org/abs/1809.10374.

## Source record

- `scripts/run_prospective_morphology_selection.py`: candidate trees, rotations, imposed decoder, moment score and development-only scalar fit.
- `source_data/prospective_morphology_selection/protocol.json`, `development_fit.json`, `policy_outcomes.csv`, `candidate_outcomes.csv` and the sealed outcome manifests.
- No new training was performed for this reassessment.

Follow-up record: the subsequently authorized investigation has now completed new finite-horizon experiments and nonlinear structural constructions. See [the completed investigation](morphology_investigation_20260905/INVESTIGATION_REPORT.md). The present note is retained as the pre-experiment diagnosis; its no-new-training statement describes that earlier assessment only.
