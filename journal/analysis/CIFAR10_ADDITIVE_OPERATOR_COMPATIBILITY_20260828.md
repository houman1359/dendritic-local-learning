# CIFAR-10 additive-operator compatibility audit

Frozen 28 August 2026 before execution and before inspecting any outcomes.

## Question

The archived 27 April CIFAR-10 cohort reported 48.3% mean BP accuracy for a
model named `dendritic_additive`.  Source audit shows that the alias then
implied per-sample normalized additive integration and fixed reactivation
gates (`m=0.1,b=0`).  Since commit `09eb657` (10 June), the same bare alias
means raw additive integration, and normalized additive integration has the
explicit alias `dendritic_normalized_additive`.  The provisional current-code
raw-additive cohort reached 38.98%.  Those values cannot be compared as if
they were repeated measurements of one forward model.

This bounded audit asks whether the discrepancy is explained by the forward
operator and/or by the change in adaptive initialization.

## Frozen design

All conditions use the current clean source at commit
`e516c7fec3169253ff8c14bc5f4ab1325469e4f5`, the archived compact
`[3,3,3,3]` architecture and BP recipe (200 epochs, patience 40, one optimizer
group, learning rate 0.001 and weight decay 0.01).  Reactivation is fixed at
`m=0.1,b=0`.  The factorial crosses:

- forward operator: raw additive or legacy normalized additive;
- adaptive network initialization: disabled or center-preserving enabled.

The original five seeds 42--46 are paired across all four cells, for 20 runs.
W&B is disabled.  Jobs run on `kempner_h100_priority`; all generated configs,
logs, checkpoints and results are written to `kempner_project_b`.

Execution was generated from the clean checkout and frozen at
`/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/sweep_runs/cifar10_additive_operator_compatibility/journal_cifar10_additive_operator_compatibility_20260828102842`.
The Slurm array job is `42547612`.

## Analysis and interpretation

The independent seed is the paired unit.  Report every seed and each cell's
mean and sample standard deviation.  Primary descriptive contrasts are
normalized minus raw additive at each adaptive setting and adaptive minus
non-adaptive initialization within each operator.  The historical reference
of 48.3% is contextual, not pooled with the new runs, because its source hash
was not embedded and uncommitted historical changes cannot be excluded.

This audit diagnoses configuration compatibility; it is not a tuned learning
comparison.  A normalized-additive recovery would explain why the historical
BP ceiling was higher, but would not validate the provisional raw-additive
feedback ladder.  Any journal CIFAR-10 ladder must use an explicitly named
forward operator and a fresh, identically configured cohort for every feedback
condition.

## Outcome

All 20 runs completed from the clean frozen source and passed the provenance,
configuration and result-file audit.  With adaptive initialization disabled,
raw additive BP reached 46.744% mean test accuracy across five seeds and
normalized additive BP reached 47.330%.  The paired normalized-minus-raw
difference was +0.586 percentage points (bootstrap 95% CI,
-0.212 to +1.762; positive in 3/5 seeds).  Thus normalization alone did
not explain the approximately nine-point gap between the provisional 38.98%
cohort and the archived result.

Adaptive initialization reduced accuracy in this compact CIFAR-10 regime.  It
changed raw additive BP by -3.216 percentage points (95% CI,
-4.698 to -1.586; 0/5 positive seeds) and normalized additive BP by
-0.358 percentage points (95% CI, -0.716 to -0.052; 1/5 positive
seeds).  The explicit normalized-additive, non-adaptive condition therefore
passed the frozen 45% compatibility gate, but the more important result is
that the archived optimizer recipe and fixed initialization recovered most of
the missing BP performance even for the raw operator.

Because percentile bootstrap intervals are discrete with only five seeds, a
post-audit statistical supplement reports paired Student-t intervals and exact
sign-flip sensitivities.  The no-adaptive normalized-minus-raw interval widens
to -1.044 to +2.216 percentage points.  The operator-by-adaptive-initialization
interaction is +2.858 points (paired t 95% CI, +0.544 to +5.172; 4/5
positive seeds; exact two-sided sign-flip P=0.125).  This coupled interaction,
rather than a stable marginal normalization benefit, is the clearest
diagnostic.  These small-n comparisons remain descriptive.

The interpretation remains diagnostic rather than comparative: the
provisional 38.98% additive BP value was configuration-limited and must not
serve as the additive ceiling in the paper or talk.  A feedback-ladder
confirmation, if launched after the companion recipe screen and pinned-source
reproduction close, must apply one frozen operator and recipe identically to
all feedback conditions.

Machine-readable outcomes and all seed-level values are stored under
`kempner_project_b` in
`analysis/cifar10_additive_operator_compatibility_20260828/`; the augmented
small-sample analysis is stored in the parallel `_v2` directory.
