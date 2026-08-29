# Historical CIFAR-10 BP reproduction

Frozen 28 August 2026 before execution and before inspecting any new outcomes.

## Purpose

The archived 27 April source table reports mean test accuracy of 48.30% for
the then-normalized additive model and 49.516% for the shunting model.  The
archived output did not embed a source hash or retain raw checkpoints.  This
bounded reproduction tests the strongest timestamp-supported source identity
and the exact archived configurations rather than inferring compatibility from
the current implementation.

## Frozen execution

- source: detached clean root worktree at
  `9809341760ca64738f97a5e02a38b6377e2264fc`, the latest committed source
  before the archived figure timestamp;
- configurations: the unchanged archived
  `cifar10_additive_compactei_depth4_standard.yaml` and
  `cifar10_shunting_compactei_depth4_standard.yaml` files;
- seeds: the original paired seeds 42--46;
- training: the archived 200-epoch, patience-40, weight-decay-0.01 BP recipe;
- compute: `kempner_h100_priority`, W&B disabled;
- storage: generated configs, logs, checkpoints and results under
  `kempner_project_b` only.

The frozen sweep root is
`/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/sweep_runs/cifar10_historical_bp_reproduction/journal_cifar10_historical_bp_reproduction_20260828103427`;
its source/config hashes are recorded in `frozen_historical_manifest.json`.
The Slurm array job is `42548857`.

The historical additive alias resolves to per-sample normalized additive
integration with fixed `m=0.1,b=0` gates and adaptive initialization disabled.
The shunting config uses the contemporaneous shunting alias under the same
training recipe.

## Reporting rule

Every seed is retained.  Report each architecture's mean, sample standard
deviation and descriptive bootstrap interval, plus the paired shunting-minus-
additive contrast.  Compare the new means descriptively to 48.30% and 49.516%;
do not pool them with the archived outputs.  Reproduction is considered
compatible when each new mean lies within two percentage points of its
archived mean.  Failure triggers a source/runtime discrepancy report, not
parameter tuning.

## Environment-only relaunch

The first array (`42548857`) stopped before model construction because the
frozen source imports W&B unconditionally and the user-site W&B installation
was incompatible.  W&B was already disabled in every frozen configuration,
and no outcome was produced.  The array was relaunched as `42553253` with a
recorded no-op import shim placed ahead of the source tree on `PYTHONPATH`.
Neither tracked historical source nor any model, data, optimizer or training
setting changed.  The shim and launcher hashes, together with the reason for
the compatibility intervention, are recorded in
`frozen_historical_manifest.json`.

## Outcome

All ten relaunched runs completed and passed the frozen source, configuration,
checkpoint and result audit.  The normalized-additive model reproduced the
archived mean exactly to the reported precision: 48.300% across five seeds
(sample s.d. 0.989 percentage points), compared with the archived 48.300%.
The shunting model reached 49.542% (s.d. 0.739 points), 0.026 points above the
archived 49.516%.  Both arms therefore pass the predeclared two-percentage-point
compatibility rule by a wide margin.

The paired shunting-minus-normalized-additive difference was 1.242 percentage
points and positive in 5/5 seeds.  This is a descriptive forward-architecture
contrast under the historical operator definitions, not evidence that
shunting improves local credit transport.  The principal conclusion of this
audit is provenance: the inherited CIFAR-10 values are reproducible when the
historical normalized-additive operator and fixed initialization are restored.

Machine-readable summaries and seed-level provenance are stored under
kempner_project_b in
analysis/cifar10_historical_bp_reproduction_20260828/.
