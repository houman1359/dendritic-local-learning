# Calibrated credit bridge: completed results

The new same-tree comparison supports a task-to-credit narrative. An initially
calibrated fixed spatial profile learns the pairwise target under Adam, but it
does not rescue quartic or nested learning. The result survives a common
learning rate. The compatible positive conductance teacher, however, is learned
accurately by every tested rule under Adam. These findings support conditional
credit demands; they do not establish a universal interaction-order law or the
necessity of six independent channels.

## Design and validation

The protocol and code were frozen before development. Three development seeds
received the same three-rate tuning budget for every rule and optimizer. Rate
selection used final validation NMSE averaged over all three tasks, separately
for each model, optimizer and rule. One common rate per model and optimizer was
also selected. Both selections were frozen before 20 independent fresh paired
seed blocks. All selected/common-rate outcomes were retained.

The completed experiment contains 432 development fits and 1,200 fresh fits,
with seven saved checkpoints per fit at steps 0, 1, 16, 64, 256, 512 and 1,024.
The 48 short benchmark fits were excluded from selection and inference. No old
source data or model implementation changed. A failed Slurm launcher attempt
exited before any Python fit; its log is retained.

Matching and quartet targets have exactly the same compatible balanced tree,
leaf assignment, coefficient initialization, calibration inputs, input
examples, additive label noise and minibatch order within each seed. Both input-gradient covariance
matrices equal I/4. Nested targets use their own compatible tree and do not have
the same input spectrum. The conductance replication uses the prior positive
teacher across all three compatible input permutations, not the algebraic
interaction-order tasks. All random signs and conditions were retained.

Five independent tests pass, including finite-difference agreement for every
parameter in both models and controlled projection formulas. The independent
checkpoint audit verifies 138 task files across all 23 seeds and both models;
the largest discrepancy on recomputing held-out NMSE is 2.22e-16. All profile
formulas, matched initial states, spectrum identities, full-rank baselines and
oracle-capture bounds pass. Conductance unit and sign trajectories are exactly
identical, as positivity requires.

## Fixed-budget learning result

Mean held-out NMSE under the frozen rule-specific Adam rates is:

| Algebraic target | Exact path | Unit broadcast | Calibrated profile | Sign profile |
|---|---:|---:|---:|---:|
| Matching | 0.03173 | 0.02516 | 0.02358 | 0.02358 |
| Quartet | 0.15164 | 0.96824 | 0.98130 | 0.98130 |
| Nested | 0.02295 | 0.75222 | 0.76896 | 0.76896 |

The calibrated-minus-exact gap is -0.00815 for matching (95% paired seed
bootstrap interval -0.02489 to 0.00069), 0.82967 for quartet (0.70743 to 0.92880),
and 0.74601 for nested (0.72682 to 0.76130). The predefined primary interaction,
quartet's gap minus matching's gap, is **0.83782 (0.71904 to 0.93075)** and is
positive in all 20 seeds. At the common Adam rate of 0.003, it is **0.84348
(0.72719 to 0.93472)**, again positive in all 20 seeds. These are descriptive
95% intervals from 10,000 whole-seed bootstrap draws, with no multiplicity claim.

Initial calibration does not improve substantially over unit broadcast for the
higher-order tasks. It rules out the particular explanation that their large
deficits arise merely because unit broadcast ignores the student's initial
mean path gains. It does not rule out a profile learned during training, an
oracle time-dependent amplitude, an intermediate-dimensional dictionary, or
different optimization protocols.

Under SGD, calibrated broadcasting is also slower on matching: mean NMSE is
0.29813 compared with 0.06479 for exact, 0.02362 for sign and 0.02804 for unit.
The fixed initial gain magnitudes therefore matter to SGD learning speed. Full
SGD results and common-rate sensitivity are retained in the tables; the
matching-success statement must specify Adam rather than generalize across
optimizers.

## Spatial credit geometry

For exact-trained Adam checkpoints, mean capture of the six nonsomatic path
derivatives at step 1,024 is:

| Target | Uniform one-profile oracle | Initial calibrated one-profile oracle | Best current rank-one oracle | Effective rank |
|---|---:|---:|---:|---:|
| Matching | 0.12240 | 0.56547 | 0.99726 | 1.00565 |
| Quartet | 0.16864 | 0.30021 | 0.46100 | 3.23825 |
| Nested | 0.17569 | 0.30534 | 0.44825 | 3.41670 |

The corresponding best-rank-one capture of the loss-error-weighted credit field
is 0.99730, 0.50631 and 0.55275. Every full-rank capture is one. The saved inputs,
weights and targets permit complete spectra and state-matched counterfactuals
to be recomputed independently.

This distinction is central: **uniform broadcast capture does not separate the
task families in the predicted direction; best rank-one capture does.** The
pairwise field is almost one-dimensional but generally has a signed spatial
profile. Quartet and nested credit occupy more spatial dimensions. Matching and
quartet provide the clean evidence that identical input spectra can accompany
different learned credit spectra on the same tree. These are learned-state
geometry measurements, not an initialization-based structure selector.

## Positive conductance comparison

Averaging the three paired compatible input permutations within each seed,
Adam's mean final NMSE is 9.39e-6 for exact credit and 1.35e-6 for each broadcast
control. The calibrated-minus-exact difference is -8.04e-6 (95% interval
-2.19e-5 to -2.74e-7), which is very small in task-error units and favors the
broadcast control. The exact path's best-rank-one capture is about 0.991,
whereas uniform capture is about 0.726 and initial-profile capture about 0.904.
High predictive accuracy here does not require high Euclidean match to the
exact path field.

Under SGD the seed-averaged exact, unit and calibrated errors are 0.02093,
0.00728 and 0.02521. The calibrated-minus-exact gap is 0.00429 (0.00211 to
0.00717), whereas unit-minus-exact is -0.01365 (-0.01744 to -0.01013). These
optimizer-dependent differences preclude a general claim that exact transport
is superior for the conductance teacher.

Positive rescaling of a fixed profile largely cancels in Adam's coordinatewise
normalization. Thus calibrated and sign-only Adam trajectories should be nearly
identical; their close match is a mechanistic check, not independent evidence
from two distinct successful or failed controls. Conductance sign and unit are
exactly the same rule. Algebraic calibrated/sign endpoint discrepancies are at
most 7.75e-6; conductance discrepancies are at most 3.03e-11.

## Scope for manuscript use

The defensible statement is: on one shared representationally compatible tree,
input-spectrum-matched pairwise and quartic tasks produce different learned
spatial credit spectra, and the tested fixed profiles suffice for the former
but lose substantial accuracy on the latter under matched Adam conditions.
The nested target replicates that learning deficit on its compatible tree.

All outcomes are fixed at 1,024 steps. Some exact quartic fits retain appreciable
error; there is no convergence or global-optimality claim. Parameter bounds are
the prior algebraic [-2,2] box; some restricted-rule fits contact those bounds,
which are recorded. The matching and quartet target variances are respectively
1 and 0.5 with the same additive noise SD 0.15; each training/reporting loss is
normalized by its own variance. Their common input spectrum therefore does not
mean equal output variance or signal-to-noise ratio. The clean full-domain
population errors are retained as secondary outcomes.

Specifically, clean target variance is 1 for matching and nested and 0.5 for
quartet. Additive label-noise variance is 0.0225 for every algebraic family, so
the expected noise-only NMSE is 0.0225, 0.0225 and 0.045, respectively. The
training gradient divides the residual by that family's clean target variance,
as required by the normalized half-MSE objective. This is verified by finite
differences, not inferred from the output table. The variance/noise table is
`summaries/task_variance_and_noise_floor.csv`. Path spectra are additionally
state- and coordinate-dependent; the independent gauge analysis should be
consulted before treating their magnitudes as a morphology-invariant demand.

The conductance result narrows biological transfer: it confirms that the
algebraic gap is not explained solely by inconsistent definitions of broadcast,
but it does not reproduce that gap in a positive conductance teacher. Avoid
claiming that interaction order alone universally determines biological credit
resolution or that poor fixed-profile learning proves full rank is necessary.

## Files

- `protocol_freeze.json`, `freeze_timestamp.json`, and `selection_freeze.json`
  record the prospective choices and source hashes.
- `runs/{development,fresh}/{algebraic,conductance}/` contains immutable curves,
  diagnostics, checkpoint NPZ files, metadata and audits.
- `summaries/paired_contrasts.csv` and `paired_seed_contrasts.csv` contain all
  selected/common-rate, rule, task and optimizer contrasts.
- `summaries/exact_adam_credit_geometry.csv` contains the bridge's field metrics.
- `summaries/independent_audit.json` records independent numerical verification.
- The reproducible scripts and tests are in `scripts/credit_rule_bridge/`.
