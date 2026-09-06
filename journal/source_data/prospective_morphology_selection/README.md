# Prospective costed selection in tree-constrained linear learning

This experiment tests whether a calibrated initialization update-moment score
selects useful route capacity and tree geometry for subsequent minibatch
learning. It uses an explicitly named differentiable linear tree abstraction.
The imposed context decoder is not a natural somatic readout, and this finite
candidate experiment does not establish a biophysical morphology law.

## Protocol and temporal separation

`protocol.json` is the frozen protocol; `protocol_freeze.json` records its SHA-256
and the runner hash before development or confirmatory training. Development
uses five task seeds and rotation angles 0, pi/4 and pi/2. Confirmation uses
20 different task seeds and held-out angles pi/8 and 3pi/8. Each seed contains
ranks 1, 2, 4 and 8 and observation-noise SDs 0 and 0.75. The seed is the
independent unit; the 16 conditions within each confirmatory seed are paired.

All calibration samples are independent of training and test samples. The
calibration and training samplers use the same declared distribution, but draw
separate examples. Angle/noise comparisons share underlying random draws
within each seed/rank. `split_manifest.json` records these identities.

The development outcomes fit one nonnegative utility-to-loss scale per arm
and choose the fixed and rank-only baselines. All 3,200 confirmatory policy
choices were saved in `sealed_confirmatory_selections.csv` and hashed at
16:11:16 UTC on 5 September 2026, before any confirmatory candidate training.
Every confirmatory shard checks the source, protocol, calibration and selection
hashes. `selection_freeze.json` records the complete seal. Nothing was tuned
or reselected after confirmatory outcomes.

## Model and reference spectrum

Eight leaves receive currents or linear drives `W x`, with 64 trainable
weights. A context vector `a` defines the imposed scalar readout `a^T H W x`.
The feedback-only arm sets `H=I` for every candidate. The joint arm sets `H`
to the leaf block of `(I+kappa L_T)^(-1)` for the full 15-node tree Laplacian,
with kappa=1, unit leak at every node and inverse-length edge conductance.
Internal nodes are explicitly included. The tests verify the Schur reduction
and equivalence at kappa=0.

A random orthogonal teacher maps isotropic Gaussian inputs to labels. Contexts
sample the first `r` orthonormal directions of a rotating basis uniformly.
At zero initialization, the exact decoder-space credit second moment is
`(1+noise_sd^2) Q diag(1/r,...,1/r,0,...) Q^T`. The rotation is orthogonal, so
its spectrum is exactly fixed as alignment changes. This analytic reference
spectrum is distinguished from finite-sample estimates. In the joint arm,
transport through `H` can change the transported and parameter-gradient
spectra; no invariance of those spectra is claimed.

Five explicit labeled trees (four balanced hierarchies and one comb) provide
subtree-cut dictionaries with K=1,2,4,8. Each cut is orthonormal, with projector
P. The actual synaptic update is `P (H^T a) delta x^T`. Coefficients depend on
the known decoder and transfer; this is a prescribed information source, not
a biologically learned encoder. All candidates have 64 trainable weights and
64 fixed task-decoder coefficients. Additional counts include K channels,
8*K possible context-to-route encoder coefficients, eight nonzero route-delivery
entries, 15 electrical nodes and 14 edges. The candidate manifest supplies
matrices, topology, embedding, counts and costs.

The declared cost is `C=K/8+0.1*length/max_candidate_length`. Length is a
geometric proxy from a fixed two-dimensional embedding, not measured biological
wiring or energy. The final objective is held-out half MSE plus `0.08*C`.
Some candidates have identical feedback projectors but different cable costs;
`candidate_equivalence_classes.csv` records this redundancy. They are not
counted as independent replicates.

## Calibration, learning and endpoints

Calibration uses 1,024 examples to compute the actual routed update mean and
its exact empirical covariance trace for iid minibatches of size 32. Thus
`E||d_batch||^2 = ||E d||^2 + tr(Cov(d_single))/32`. The score is the descent
bound at the actual learning rate 0.35:
`U=eta*gradL dot E[d] - eta^2*L*E||d_batch||^2/2`.
Here L is global in the weights for the finite calibration quadratic loss.
It does not provide a population or independent-test bound.

Each candidate is trained for 256 genuine iid minibatch updates on 2,048
training examples and evaluated on 4,096 independent test examples. All
initial weights are zero, and paired candidates receive the same minibatch
stream. Checkpoints 16,64,256 are retained; only 256 defines the primary
endpoint. Confirmation contains 12,800 candidate fits, with 38,400 checkpoint
rows. Development contains 4,800 candidate fits. No outcomes were excluded.

Policies are the moment selector, development-best fixed candidate, the
maximum-budget candidate with development-selected tree, a development-fitted
rank-only candidate, a sampled random candidate, and the exact expectation of
a uniform random choice. The retrospectively observed best candidate is a
reference oracle used only to compute regret. It was never available to the
prospective selector. Both loss and costed loss are retained. Intervals use
10,000 bootstrap resamples of the 20 whole-seed averages.

## Result and scope

The frozen moment selector beats random selection but loses to both the fixed
maximum-budget and rank-only baselines in both arms. Its mean regret is 0.08055
for feedback-only routing and 0.10404 for joint forward/feedback design. The
rank-only baseline has regret 0.00831 and 0.00821. These negative comparisons
are primary results, not excluded conditions.

The retrospectively best budget equals the imposed task rank in every
confirmatory task. Detailed tree identity is dominated by the contiguous
balanced candidate (308/320 feedback-only and 303/320 joint tasks). This
supports a restricted task-rank/capacity relation in the constructed linear
class, with little evidence for reliable selection of detailed morphology.
It does not establish a general mapping from task complexity to natural trees.

`audit_prospective_morphology_selection.py` adds a clearly posthoc diagnostic:
for every condition it computes exact expected calibration one-step decrease
using the full Hessian and actual minibatch covariance, checks the local bound,
and reconstructs the actual first training update to evaluate independent-test
loss decrease. It does not alter any primary selection or training outcome.
These results distinguish a valid local bound from a failed long-horizon
selection rule. `posthoc_validation_report.json` and the first-step tables
record the numerical checks and empirical first-step correlations.

## Reproduction

From the repository root, use the runner under `journal/scripts/` and the
three Slurm recipes under `journal/configs/prospective_morphology_selection/`.
The stage order is `invariants`, `freeze`, development/calibration jobs,
`select`, confirmatory jobs, `analyze`, then optional posthoc first-step jobs
and audit `--summarize`. The primary runner refuses to reseal selections after
confirmatory outcomes exist. To reproduce from scratch, use a fresh isolated
copy of the experiment output directory; preserve the archived frozen record.

The completed run used CPU-only Slurm arrays 44624492, 44624676 and 44624938
on the shared partition with account kempner_dev, one CPU per task and at most
12 simultaneous tasks. No GPU jobs or unrelated jobs were modified. Scheduler
logs remain in the analysis directory; the release contains numerical data,
protocols, checksums and accounting summaries without private raw caches.
