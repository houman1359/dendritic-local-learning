# Finite-horizon prediction investigation: results and limits

The finite-horizon forecast repairs the original scalar selector's failure on new seeds. This is a successful diagnosis of the learning-dynamics approximation, but it is **not strong new evidence that detailed dendritic morphology can be predicted from task complexity**. In this task family, a cheap observed-context-count baseline already obtains almost the same outcome.

All original experiment files, manuscripts and release artifacts remain unchanged. The original negative experiment is retained. The new code is under `scripts/morphology_dynamics/`; all new results are under this directory.

## Design and validation

The primary method was frozen before fresh calibration or training: estimate a Gaussian conditional linear model from 1,024 calibration examples, then propagate its exact fresh-example SGD first and second moments for the original 256 updates. Learning rate, batch size, cost and candidate set were unchanged. There were no new fitted endpoint-scaling parameters or outcome-driven choices of horizon or cost.

Five original development seeds were diagnostic. Their original fixed-cache training trajectories were reproduced to maximum half-MSE discrepancy **9.91e-17**. Twenty new seeds, 9300–9319, then supplied 320 task conditions per arm and training regime. Both the original finite-cache learner and a fresh-example learner were run: **25,600 fresh candidate fits and 76,800 checkpoint outcomes**, all retained. The code and protocol were hashed before fresh runs, and each seed's non-pilot predictions were sealed before its outcome training.

The observed-context-count and rank-restricted conditioning baselines were added after the primary protocol and development inspection. Their definitions were separately frozen before fresh outcomes. They are explicitly secondary additions. They infer the count of distinct observed context vectors; they do not read the generating rank to make their choice. In this constructed task, that observed count equaled generating rank in every condition. This is not a general estimator of credit rank.

Four focused tests pass: empirical spectral versus direct gradient descent with a singular Hessian; population spectral versus full-batch recurrence; stochastic Gram moments versus exact Gaussian quadrature over all two-example minibatches from a nonzero random weight state; and a guard that forbids endpoint predictors from requesting training or test observations. The only privileged inputs enter methods explicitly named `gaussian_oracle_*`. The pilot uses 16 training updates and calibration evaluation. The derivation and exactness assumptions are in `DERIVATION.md`.

## Fresh fixed-cache results

Entries below are mean final costed-loss regret relative to the best evaluated candidate on that task. Conditions are averaged within each of 20 independent seeds. Complete 95% seed-bootstrap intervals and rank-specific results are in `summaries/fresh/endpoint_summary_with_secondary_baselines.csv`.

| Selector | Feedback only | Joint forward and feedback |
|---|---:|---:|
| Original initialization score × fitted scalar | 0.081429 | 0.100691 |
| Original development-fixed maximum-budget tree | 0.044693 | 0.042695 |
| Original generating-rank policy | 0.008242 | 0.008197 |
| Sixteen-step pilot | 0.009248 | 0.019465 |
| Empirical full-batch spectral forecast | 0.000153 | 0.000363 |
| Calibration Gaussian full-batch spectral forecast | 0.000134 | 0.000034 |
| **Calibration Gaussian SGD moment forecast (primary)** | **0.000064** | **0.000042** |
| Observed context count + cheapest tree | 0.000134 | 0.000197 |
| Population-oracle SGD forecast | 0.000079 | 0.000009 |

The primary forecast reduces regret against the original scalar, original rank policy and pilot in every seed-average comparison in both arms. Its 95% regret intervals are approximately [0.000030, 0.000104] and [0.000008, 0.000089]. This is an independently tested repair of the scalar extrapolation within the original task family.

The more important comparison is the cheap context-count baseline. The primary method's regret reduction against that baseline is:

- Feedback only: **0.000069**, 95% CI **[-0.000013, 0.000149]**; uncertain.
- Joint: **0.000155**, 95% CI **[0.000040, 0.000300]**; positive but small in absolute objective units.

All finite-horizon methods choose `K=r` on all 320 tasks per arm. Restricting the primary method to that observed-count budget gives exactly the same choices. The primary method selects the contiguous tree in 296/320 feedback tasks and 310/320 joint tasks. Consequently, most of the apparent recovery is the task's simple capacity/cost relation. The joint improvement over the cheapest tree mainly occurs at rank four. The full-batch plug-in forecast is marginally better than the SGD plug-in forecast in the joint arm, so the stochastic correction is not necessary to obtain the selection improvement.

| Rank | Feedback original | Feedback primary | Feedback count-only | Joint original | Joint primary | Joint count-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.048412 | 0.000000 | 0.000000 | 0.058228 | 0.000000 | 0.000000 |
| 2 | 0.032364 | 0.000202 | 0.000490 | 0.051882 | 0.000014 | 0.000008 |
| 4 | 0.019641 | 0.000055 | 0.000044 | 0.024378 | 0.000126 | 0.000754 |
| 8 | 0.225298 | 0.000000 | 0.000000 | 0.268276 | 0.000026 | 0.000026 |

## Loss prediction is substantially more accurate

For the fixed-cache learner, final half-MSE prediction RMSE falls from **0.25645 to 0.01549** in feedback-only tasks and from **0.26215 to 0.01537** in joint tasks. Mean within-task Spearman correlations between predicted and actual costed objectives rise from **0.1519 to 0.9896** and from **-0.0302 to 0.9871**. These are endpoint relationships, not the original one-step correlations.

The primary and original endpoint correlations are defined on all 320 tasks per arm. Some secondary full-batch forecasts become constant at numerical precision in effectively solved noiseless cases. Those correlations are retained as missing, with explicit defined-task counts, rather than silently deleting outcomes. Across all methods, horizons and regimes there are 242 undefined loss correlations. See `prediction_summary_by_rank_with_defined_counts.csv`.

## The finite-cache assumption is a real limitation

The SGD moment equation is exact for independent fresh population examples. It is not exact when minibatches repeatedly reuse a finite training set. The new experiment directly quantifies this distinction using the population-oracle predictor and the exact population loss of the realized trained weights.

At the final horizon, mean oracle prediction minus realized population loss is:

| Regime | Feedback only, mean [95% CI] | Joint, mean [95% CI] |
|---|---:|---:|
| Fresh iid | +0.000211 [-0.000234, +0.000617] | +0.000114 [-0.000170, +0.000345] |
| Reused finite cache | -0.001509 [-0.001924, -0.001096] | -0.001645 [-0.001871, -0.001394] |

The aggregate fresh-example errors are consistent with zero. Reusing a cache produces an underprediction of final risk, particularly at high rank with noise. The paired cache-versus-fresh shifts are -0.001719 and -0.001759, with intervals excluding zero. The calibration model introduces additional estimation error; it is never described as an exact forecast of the original cached-data process.

## Computational cost

Median single-CPU implementation time per 20-candidate decision was approximately:

| Method | Seconds |
|---|---:|
| Observed-context count decision | 0.00163 |
| Gaussian plug-in full-batch spectral forecast | 0.00363 |
| Empirical split-calibration spectral forecast | 0.01029 |
| Sixteen-update pilot plus calibration evaluation | 0.01454 |
| Original scalar score computation | 0.02897 |
| Gaussian plug-in SGD moment forecast | 0.04188 |
| All candidates, 256 training updates | 0.15293 |

The plug-in times include calibration-model preparation. The count-only time excludes shared calibration-data generation. These are measured implementation times, not hardware-independent operation counts or an optimized speed contest. The SGD predictor is about three times slower than the short pilot, although still cheaper than full training. The simpler spectral forecast is the more economical option here.

## Implication for the paper and dendritic_replacement

The original failure can now be attributed much more specifically: one initial progress bound, multiplied by one scalar, does not encode the mode-specific finite-time dynamics and saturation required for endpoint selection. A proper dynamics model can predict those endpoints well in fresh data from this linear family.

This does not resolve the separate morphology question. Unrestricted output weights can compensate for generic route rotations; the task exposes its context rank, and the cheapest contiguous tree is already close to optimal. Claims about predicting detailed morphology still need the separate study where task structure and local input constraints cause the preferred morphology to change at fixed rank and cost.

For dendritic_replacement, the exact Gaussian recurrence should not be transferred to nonlinear dendrites, arbitrary input covariance, context-dependent adaptive optimizers or non-somatic decoding without re-derivation. The transferable lesson is to model the restricted learning operator and its finite horizon, and to benchmark any resulting forecast against an inexpensive pilot and a strong capacity/conditioning baseline. A tangent-kernel approximation there would need its own empirical validity check as training changes the features.

## Artifacts

- `DERIVATION.md`: equations, proofs and scope.
- `protocol.json`, `supplemental_baseline_protocol.json`: frozen definitions and hashes.
- `runs/development/`, `runs/fresh/`: all predictions, outcomes, pilots, timings, seals and audits.
- `summaries/fresh/`: endpoint and rank-specific regret, paired comparisons, selection counts, prediction errors, computation cost and cache sensitivity.
- `figures/finite_horizon_selection.pdf`: selection, strong baselines and computation cost.
- `figures/finite_horizon_prediction_diagnostics.pdf`: endpoint forecasts and cache mismatch.
- `fresh_validation.json`, `operator_test_results.txt`, `slurm_accounting.tsv`: completeness and execution checks.

Both PDFs and PNG previews were visually inspected. The smallest nominally nonnegative prediction was -2.51e-34, a retained floating-point round-off value; there were no material negative risks or nonfinite outcomes. All primary source hashes and prediction seals passed. No new result was inserted into the manuscript or canonical figures.
