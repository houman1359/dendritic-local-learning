# Calibration bridge between algebraic and positive conductance trees

This experiment compares exact path credit with three frozen one-signal rules:
unit broadcast, an initial label-free mean-path profile, and its sign alone.
It preserves the two existing forward models without changing their code or
earlier results. It is a fixed-budget learning test, not a convergence test.

The protocol is `../../configs/credit_rule_bridge/protocol.json`. The source
hashes and first freeze time are retained in `../../source_data/credit_rule_bridge`.
Development and fresh seed sets are disjoint. Each rule receives three rate
choices under each optimizer. A rule's selected rate minimizes development
validation error over all tasks; a common-rate control is also frozen before
fresh runs. The benchmark is separate and excluded from rate selection.

Matching and quartet targets share exactly the same compatible balanced tree,
leaf assignment, coefficient initialization, input matrices, noise realization
and minibatch sequence within a seed. Their input-gradient covariance matrices
both equal one quarter of the identity. Their target variances differ, and each
loss is normalized by its own variance. Nested targets use the prior prefix
interaction construction and their own compatible tree: they are a separate
replication, not an isospectral or same-tree comparison. The conductance model
uses the existing positive teacher with all three compatible input permutations;
it does not implement the algebraic task families.

The initial calibration profile uses the student's output-to-unit derivatives
on 256 unlabeled examples, and is frozen without normalization. This control
requires initial model path knowledge. It is not an asserted biological profile
estimator. Every rule keeps the somatic derivative exactly one. The sign rule
equals unit broadcast identically in the positive conductance model and is kept
as an explicit implementation check.

The diagnostic tables distinguish actual delivered-field error from oracle
projection capture. A one-profile oracle may choose its amplitude from the exact
field separately for each example. The best rank-one oracle additionally fits
its spatial direction from the current field. Neither is a learning arm. High
rank-one capture does not imply that a uniform profile is adequate, and poor
calibrated learning does not prove that full compartment rank is necessary.

Each NPZ file stores `theta[checkpoint, condition, ...]`, checkpoint steps,
initial profiles, unlabeled calibration inputs and diagnostic inputs/targets.
Its JSON companion contains the exact condition order, task and tree metadata.
Algebraic weights have shape 7 by 4; conductance weights are 16 log-conductances.
CSV files retain all learning curves, all rule-specific and common rates, field
spectra summaries, local-gradient cosines and clipping/bound events.

Reproduction from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s drafts/dendritic-local-learning/journal/scripts/credit_rule_bridge -p test_bridge.py -v
OPENBLAS_NUM_THREADS=1 python drafts/dendritic-local-learning/journal/scripts/credit_rule_bridge/run.py freeze
```

Use `worker.sh development` as a six-task Slurm array (0–5), then run
`run.py select` after every development task finishes. Use `worker.sh fresh`
as a forty-task array (0–39), then `run.py analyze`. The worker expects submission
from the repository root. Completed seed outcomes are immutable; reproduce into
a new output location if results already exist.

One initial launcher attempt used Slurm's copied-script directory incorrectly
and exited before starting any Python fit. Its log is preserved. The corrected
launcher resolves the code under `SLURM_SUBMIT_DIR`; no scientific input or
frozen setting changed.
