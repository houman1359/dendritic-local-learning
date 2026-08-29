# CIFAR-10 BP recipe and initialization audit

Frozen 28 August 2026 before execution of calibration seeds 10700--10702.

Execution was launched as Slurm array job `42545270` from clean commit
`e516c7fec3169253ff8c14bc5f4ab1325469e4f5`.  The generated sweep root is
`/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/sweep_runs/cifar10_bp_recipe_init_screen/journal_cifar10_bp_recipe_init_screen_20260828101020`.
The analysis code and decision thresholds were frozen while the first array
tasks were still training and before any final outcomes were inspected.

## Source-history amendment before final-outcome inspection

A subsequent line-by-line source audit identified a third, more fundamental
configuration difference that the current-code screen does not reproduce.
At the time of the archived 27 April cohort, the `dendritic_additive` alias
meant a per-sample normalized additive forward operator and forced fixed gates
with `m=0.1,b=0` at every stage; adaptive network scaling was disabled.  Since
commit `09eb657` (10 June), the same alias names the raw additive operator,
while the normalized operator has the explicit alias
`dendritic_normalized_additive`.  The current source also enables
center-preserving adaptive scaling by default.  Thus the archived 48.3% and
provisional 38.98% BP values do not estimate the same forward model.  Because
the archived output did not embed a source hash, commit `980934176` is the
strongest timestamp-supported source identity, but uncommitted source changes
cannot be excluded.

Consequently, this screen answers which explicit initialization and training
recipe works best for the current raw-additive implementation; it is not
labeled an exact historical reproduction or compared numerically to the old
normalized-additive ceiling.  Additive `occupancy_quantile` and `analytical`
arms are treated as one effective gate whenever the occupancy fit reverts to
the analytical values.  A bounded compatibility check will evaluate the
explicit current `dendritic_normalized_additive` control under its historical
fixed gate and archived BP recipe.  This distinguishes a source regression
from a forward-operator change; it does not authorize open-ended tuning or
manuscript inclusion of the provisional CIFAR-10 ladder.

## Why this audit is required

The clean five-seed CIFAR-10 feedback ladder produced mean additive
backpropagation accuracy of 38.98%, whereas the archived compact-E/I cohort
reported approximately 48.3% with the same nominal forward architecture.  The
new value is therefore provisional and will not enter the manuscript until the
discrepancy is resolved.

Two known configuration changes could explain the discrepancy.  First, the
new matched ladder used split optimizer groups with zero sparse-maintenance
rate, whereas the archived BP recipe used one optimizer group with a
model-specific sparse active-weight maintenance rate of 0.01. Adam's built-in
weight decay was zero in both recipes.
Second, current typed configuration loading assigns an omitted additive
reactivation policy to `occupancy_quantile`.  In the completed ladder, the
occupancy fit exceeded the allowed slope at every additive layer and reverted
to an analytical gate with `m=0.1` and `b=0`.  The separate empirical-
initialization project cannot settle this comparison because it uses a
different population width, branching depth, input normalization, inhibition
and decoder.

## Frozen calibration screen

The screen retains the journal ladder's flattened, unnormalized CIFAR-10 data,
compact `[3,3,3,3]` architecture, train/validation split, decoder and input
maps.  It crosses:

- architecture: additive or shunting;
- explicit reactivation policy: `fixed` (`m=1.5`, `b=0.5`), `analytical`,
  `empirical`, or `occupancy_quantile`;
- training recipe:
  - archived BP: 200 epochs, patience 40, sparse active-weight maintenance
    rate 0.01, one optimizer group;
  - local-matched BP: 400 epochs, patience 50, zero sparse-maintenance rate, split
    optimizer groups.

Every cell uses three independent calibration seeds, 10700--10702, for 48 BP
runs.  W&B is disabled.  Outputs, logs and resolved configurations are written
only to `kempner_project_b`, and all jobs use the same clean source checkout on
`kempner_h100_priority`.

## Selection and decision rules

The primary selection endpoint is mean validation accuracy, not test accuracy.
Within each architecture, the eligible policy--recipe cell with the highest
mean validation accuracy is selected.  If two cells differ by less than 0.5
percentage points, the cell with lower validation standard deviation is
selected; a remaining tie favors the archived BP recipe and then the explicit
fixed policy.  Numerical failures, missing checkpoints and reverted empirical
calibrations are ineligible.  An `occupancy_quantile` cell remains eligible
after a documented safety reversion, but is labeled by the gate actually
applied rather than by the requested policy.

A fresh feedback-ladder confirmation is warranted only if the selected
additive cell reaches at least 45% mean test accuracy and exceeds the
provisional additive BP mean by at least three percentage points.  The
confirmation will use fresh seeds and will compare strict scalar,
neuron-specific, exact-path and matched BP feedback within each architecture
under that architecture's selected recipe.  Architecture-to-architecture
accuracy will not be an inferential contrast when the selected recipes differ.

If no additive cell passes this gate, the new CIFAR ladder is excluded from the
paper rather than optimized further.  Calibration-screen test outcomes remain
exploratory and are not pooled with the confirmation cohort.

## Outcome

All 48 runs completed from the frozen clean source with the expected result and
checkpoint artifacts.  The frozen validation-only selection rule chose the
raw-additive cell that couples center-preserving adaptive network
initialization with data-driven empirical reactivation calibration and the
archived optimizer recipe.  Adaptive network initialization was enabled in
the resolved configurations; the empirical procedure then calibrated the
reactivation gates from three training batches before optimization.  Across
the three calibration seeds, the applied leaf slopes were 21.716--21.879 with
centers near zero, whereas the four downstream levels used slopes
1.192--1.267 and centers 1.498--1.503.  The selected cell reached 49.960% mean
validation accuracy and 49.723% mean test accuracy.  This exceeds the
provisional raw-additive BP value by 10.747 percentage points and passes both
prewritten thresholds for launching a fresh feedback ladder.

The selected shunting cell likewise coupled adaptive network initialization
with empirical reactivation calibration, but used the local-matched optimizer
recipe (48.700% validation; 49.093% test).  Because the architectures selected
different recipes, these screen values are not a shunting-versus-additive
contrast.  They show only that both implementations can reach approximately
49% BP accuracy under their validation-selected configurations.

The screen therefore identifies a viable raw-additive configuration and shows
that the provisional low ceiling was configuration dependent.  It does not
attribute the recovery to empirical reactivation calibration alone, because
all screen cells also used adaptive network initialization and the screen did
not factor those two procedures.  The provisional cohort remains excluded
from the paper and talk.  A confirmation ladder must use fresh paired seeds,
hold the full selected additive operator, coupled initialization procedure and
optimizer recipe fixed across feedback conditions, and be reported regardless
of outcome.

### Post-outcome analyzer correction

The first analyzer incorrectly used `init_gate_stats.json`, which records the
post-build state before data-driven calibration, to label empirical and
occupancy-based conditions.  It therefore described the selected empirical
cell as `m=0.1,b=0` and could group it with a reverted occupancy cell.  The
corrected analyzer instead requires a converged
`reactivation_calibration.json`, verifies its applied values against
`post_calibration_gate_stats.json`, checks documented reversions, and audits
the resolved adaptive-initialization and optimizer settings.  Non-data-driven
analytical cells are now correctly audited from their post-build gate state.

This correction does not alter the frozen selection algorithm: validation
accuracy remains the selection endpoint, the 0.5-percentage-point near-tie
rule and tie breakers are unchanged, and the prewritten test gate is unchanged.
The unmodified first analysis is retained under kempner_project_b in
`analysis/cifar10_bp_recipe_init_screen_20260828/`; corrected version-2 outputs
are stored separately in
`analysis/cifar10_bp_recipe_init_screen_20260828_v2_postcalibration/`.  The two
analyses select the same additive and shunting cells and reach the same launch
decision.
