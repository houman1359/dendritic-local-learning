# Spatial credit resolution: post hoc checkpoint analysis

This audit replays 20 existing seed blocks, three task families and ten selected compatible-tree conditions per family. Original parameter checkpoints were not saved; every deterministic archived metric is checked before the recovered states are used.

The routing field has six entries: one for each nonroot internal unit. The soma has exact sensitivity one under every rule and is excluded from field capture. Leaves are inputs, not additional routed internal units.

At fixed weights, tree and input, the path field q does not depend on the target. Multiplication by the scalar output residual changes total/error-weighted field energy but cancels from normalized capture for each example. Thus a credit-spectrum difference at trained states can describe different learned computations; it cannot by itself establish an initialization-time task-only law.

A uniform broadcast is one particular spatial profile. A best fixed spatial rank-one profile is an oracle chosen over the evaluation distribution. High rank-one capture with low uniform capture points to gain/sign calibration. Both quantities are distinct from eligibility-weighted update matching and from eventual learning success.

Matching and quartet targets share input-gradient covariance 0.25 I8. Their original fits have the same balanced topology up to input assignment, but family-specific random initialization and input assignment. Nested targets use a deeper compatible tree and are reported separately. A fully paired same-tree learning bridge is a separate new experiment.

Raw singular-energy fractions and effective rank depend on the units used to express each internal state. An invertible static rescaling preserves algebraic matrix rank but can change energy capture. The companion gauge audit divides each spatial coordinate by its RMS path sensitivity and verifies invariance of that normalized spectrum to static diagonal rescaling. This is a diagnostic coordinate change, not evidence that learning is invariant to reparameterization.

## Exact-trained checkpoints, step1024

| Optimizer | Task | Dictionary | Field energy capture/fidelity | Error-weighted energy | Population NMSE |
|---|---|---|---:|---:|---:|
| adam | matching | ancestry_K2 | 0.1956 | 0.1965 | 0.0024 |
| adam | matching | ancestry_K4 | 0.6288 | 0.6283 | 0.0024 |
| adam | matching | best_fixed_rank1_q | 0.9988 | 0.9987 | 0.0024 |
| adam | matching | initial_mean_fixed | 0.3357 | 0.3375 | 0.0024 |
| adam | matching | uniform_projection | 0.1195 | 0.1213 | 0.0024 |
| adam | nested | ancestry_K2 | 0.4938 | 0.5275 | 0.0019 |
| adam | nested | ancestry_K4 | 0.7754 | 0.8038 | 0.0019 |
| adam | nested | best_fixed_rank1_q | 0.4201 | 0.4546 | 0.0019 |
| adam | nested | initial_mean_fixed | -0.0299 | -0.0369 | 0.0019 |
| adam | nested | uniform_projection | 0.1659 | 0.1710 | 0.0019 |
| adam | quartet | ancestry_K2 | 0.3339 | 0.3337 | 0.0043 |
| adam | quartet | ancestry_K4 | 0.6797 | 0.6722 | 0.0043 |
| adam | quartet | best_fixed_rank1_q | 0.4873 | 0.4874 | 0.0043 |
| adam | quartet | initial_mean_fixed | 0.1928 | 0.1746 | 0.0043 |
| adam | quartet | uniform_projection | 0.1594 | 0.1454 | 0.0043 |
| sgd | matching | ancestry_K2 | 0.2415 | 0.2412 | 0.0341 |
| sgd | matching | ancestry_K4 | 0.6106 | 0.6140 | 0.0341 |
| sgd | matching | best_fixed_rank1_q | 0.9957 | 0.9941 | 0.0341 |
| sgd | matching | initial_mean_fixed | 0.2983 | 0.3000 | 0.0341 |
| sgd | matching | uniform_projection | 0.1171 | 0.1177 | 0.0341 |
| sgd | nested | ancestry_K2 | 0.4537 | 0.4829 | 0.1414 |
| sgd | nested | ancestry_K4 | 0.7668 | 0.8016 | 0.1414 |
| sgd | nested | best_fixed_rank1_q | 0.4674 | 0.4756 | 0.1414 |
| sgd | nested | initial_mean_fixed | -0.0808 | -0.0639 | 0.1414 |
| sgd | nested | uniform_projection | 0.1635 | 0.1736 | 0.1414 |
| sgd | quartet | ancestry_K2 | 0.3293 | 0.3351 | 0.3463 |
| sgd | quartet | ancestry_K4 | 0.6619 | 0.6691 | 0.3463 |
| sgd | quartet | best_fixed_rank1_q | 0.5689 | 0.5705 | 0.3463 |
| sgd | quartet | initial_mean_fixed | 0.1209 | 0.1098 | 0.3463 |
| sgd | quartet | uniform_projection | 0.1356 | 0.1382 | 0.3463 |

The initial-mean fixed profile is a delivered field with a fixed coefficient; its column is fidelity and may be negative. Other listed rows are orthogonal projection capture with per-example coefficients computed from the exact field. They are oracle capacity diagnostics, not local learning rules.

## Source tables

- `algebraic_capture.csv`: every checkpoint, reference learning rule and candidate spatial dictionary; all metrics use the same checkpoint within a row group.
- `algebraic_spectra.csv` and `algebraic_eigenvalues.csv`: path, loss-credit, parameter Jacobian and parameter-gradient spectra; no centering.
- `algebraic_metric_validation.csv`: reconstructed versus archived metrics and tolerances.
- `algebraic_paired_contrasts.csv`: descriptive whole-seed bootstrap contrasts; no multiplicity adjustment.
- `conductance_saved_state_capture.csv`: corresponding actual saved conductance endpoint diagnostic, including both optimizers.
- `replay/`: newly recovered checkpoint arrays, tree/condition metadata and file hashes.

## Interpretation boundaries

Error weighting can make a low-error trained model appear to need different directions by concentrating the remaining residual on a few patterns. We therefore show the unweighted path field, the mean normalized per-example measure and the error-weighted field separately. The best residual-weighted spatial profile is also fitted separately, rather than treating an unweighted PCA bound as the error-weighted optimum.

A spectrum with more than one direction demonstrates that one fixed profile cannot reproduce the exact field over the sampled domain. It does not prove that learning requires exact reproduction. Population-gradient cosines can differ substantially from per-example update capture because contributions cancel across examples, especially near a solution. The independent matched learning controls determine whether these geometric differences matter for the tested optimization.

All analyses are post hoc. Numerical precision of the exact finite-domain measurement does not remove model dependence or uncertainty across training seeds. No biological use of the computed field is established here.
