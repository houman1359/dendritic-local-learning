# Boolean target families and paired learning protocol

This study tests parameter learning in seven fixed logical templates, not
generalization to unseen functional families. Inputs are four independent
uniform Rademacher variables. Semantic Boolean variables are
`(x[:, permutation] + 1)/2`, where one independently seeded permutation is
shared by all task definitions and candidate trees within a seed. The seven
targets are AND4, OR4, four-way parity, `(a AND b) OR (c AND d)`,
`(a AND b) XOR (c AND d)`, `(a XOR b) AND (c XOR d)`, and
`a AND [b OR (c AND d)]`. Their positive truth-table counts are respectively
1, 15, 8, 7, 6, 4 and 5 out of 16.

Every target is centered by its exact uniform truth-table mean and divided by
its exact standard deviation. The learner therefore predicts a zero-mean,
unit-variance continuous target; nonzero Boolean means are not discarded.
The main endpoint is mean squared error over all 16 clean patterns, divided by
the known normalized target variance of one. A zero prediction has NMSE one.
Classification thresholds are `(0.5 - raw_mean)/raw_sd`; accuracy and balanced
accuracy are secondary. This distinction matters for AND4 and OR4, whose
constant majority-class predictions achieve 15/16 accuracy but only 0.5
balanced accuracy. All target-coordinate constants are exported in
`target_normalization.csv`.

The four candidates are the balanced pairings `ab|cd`, `ac|bd`, `ad|bc`, and
the comb `a|(b|(c,d))`. Each is a directed binary tree with four terminal
inputs, three trainable internal units, twelve scalar coefficients, six edges
and a root-only output. Every local computation is
`u = theta0 + theta1*l + theta2*r + theta3*l*r`. The balanced trees have depth
two and the comb depth three, counting edges from an input to the root.
No dense compensating readout, shared input access, auxiliary output or fixed
teacher gate is added. The independent Boolean-theory study provides exact
construction and capacity certificates; those target-informed weights are
never supplied to the learner.

For each task within a seed, initial node coefficients are Gaussian with
standard deviation 0.5, biases are zero and initial coefficients are clipped
to `[-2,2]`. Initialization is independent of target labels and is shared
across all four trees, two rules, two optimizers and three learning rates.
Paired conditions also share 256 independently sampled training patterns,
normalized-label Gaussian noise of standard deviation 0.15, and all minibatch
indices. Training uses 2,048 updates of batch size 32 sampled with replacement
from this fixed training cache. The independent noisy test stream contains
1,024 further patterns with fresh noise of standard deviation 0.15. The input
domain is finite, so patterns can recur across independently drawn streams;
this is not a claim of disjoint input support. No validation or test labels
are used for within-run stopping or parameter fitting.

The update differentiates half the mean squared normalized-label error.
Exact credit uses the full chain-rule root sensitivity for every node.
Unit broadcast sets both nonroot sensitivities to one, retaining the same
local features. Root sensitivity is exactly one in both rules. All twelve
coefficients remain trainable under both rules. Gradients are clipped at
Euclidean norm ten before SGD or Adam updates, then every parameter is
projected into `[-2,2]`. Adam uses betas 0.9/0.999 and epsilon `1e-8`.
Gradient-clipped steps, box-projected steps and projected coefficient counts
are retained for every fit. These are bounded optimization recipes, not an
exhaustive comparison of optimizer tuning.

The fixed rate grid is 0.003, 0.01 and 0.03 for both rules and optimizers.
Five development seeds, 2026110–2026114, provide 1,680 fits. For each
optimizer/rule, the chosen rate minimizes mean final clean population NMSE
over all development seeds, seven tasks and four trees. Ties within absolute
`1e-12` choose the smaller rate. Choices are Adam 0.003/0.01 and SGD
0.03/0.01 for exact/broadcast, respectively. The code and protocol were
frozen before development. All development files and rate choices were
hashed in a second global seal before any fresh fit began. Twenty fresh
seeds, 2026210–2026229, provide `20*7*4*2*2*3 = 6,720` confirmatory fits.
All rates, checkpoints and final weights are retained; the chosen rates
determine the primary endpoint tables. Checkpoints are 0, 1, 16, 64, 256,
512, 1,024 and 2,048. No fit failed and no fresh outcome was excluded.

There are two predeclared primary contrasts, both for Adam on XOR-of-AND:
the mean of the two crossed balanced pairings minus the compatible `ab|cd`
tree under exact credit, and broadcast minus exact credit within `ab|cd`.
Each seed supplies one paired difference. We use 10,000 bootstrap samples of
twenty whole seed blocks, with Bonferroni-adjusted two-sided 97.5% intervals
for these two comparisons. A comparison meets the prespecified practical
criterion only if its adjusted lower bound is positive and its mean
improvement is at least 0.01 NMSE. The latter is a mean-effect criterion,
not a claim that the lower interval bound exceeds 0.01. Other task, rate,
optimizer, depth and gradient comparisons have descriptive 95% intervals.
Same-rate contrasts, constant-mean prediction and uniform random-candidate
expectations are retained as controls. Candidate/task rows and repeated
checkpoints are not counted as additional independent seeds.

The complete excluded implementation smoke used seed 2026000 and all 336
conditions through 2,048 updates, evaluation and serialization. It preceded
development and is not part of the confirmatory sample. Three pre-freeze
tests check truth-table normalization, all 48 parameter finite differences
across the four candidate trees, exact/broadcast root equality, and common
leaf-permutation equivalence. Post-run validation reconstructs all 8,400
development/fresh endpoints and random streams from saved weights and seeds,
and replays all 336 fits of one fresh seed exactly. Protocol/code hashes,
selection timing, numerical tolerance and replay results are supplied in the
JSON audit records. The original core and protocol required no amendments.

The cluster jobs used one CPU each on `serial_requeue`, account
`kempner_dev`; fresh concurrency was capped at eight. Scheduler accounting
records 342 allocated CPU-seconds across 26 smoke/development/fresh tasks.
This is allocated elapsed time, not a measured CPU utilization or carbon
estimate. Numerical validation used Python 3.10.13, NumPy 2.2.6 and pandas
2.3.3 in the current environment. It does not certify a fresh dependency
installation.
