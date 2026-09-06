# Directed conductance input-grouping and credit experiment

This is a planted-composition positive control in a seven-compartment directed shunting model. It tests input grouping at fixed physical shape and resource counts. It does not vary depth, establish a general optimal morphology, or identify a biological error encoder.

Twenty fresh independent seed blocks (15400–15419) each contain three paired teacher-task input permutations, three student groupings, four feedback rules and two optimizers: **1,440 fits and 7,200 checkpoint outcomes**. All outcomes are retained. NMSE is MSE divided by the evaluation split's target variance and is dimensionless. The independent statistical unit is the seed; task permutations and candidate fits are paired conditions within a seed.

## Main numerical findings

At 1,000 updates with the development-selected Adam rate 0.01:

| Condition or paired contrast | Mean NMSE or NMSE difference | 95% seed-bootstrap interval |
|---|---:|---:|
| Exact path, compatible grouping | 0.000150 | [0.000068, 0.000244] |
| Exact path, mean incompatible grouping | 0.028921 | [0.027066, 0.030867] |
| Incompatible minus compatible, exact path | 0.028771 | [0.026919, 0.030725] |
| Calibrated broadcast minus exact path, compatible | 0.000059 | [-0.000008, 0.000154] |

The grouping contrast was positive in all 20 seed blocks. The Adam calibrated-broadcast credit contrast remains uncertain. At the development-selected SGD rate 0.3, the compatible calibrated-broadcast-minus-exact contrast was 0.000392 [0.000230, 0.000584]. All four credit rules and both optimizers are retained in the tables; no positive general credit claim follows from the Adam result.

An independently specified analytic interaction bound gives a mean incompatible **population NMSE lower bound** of 0.0132221 [0.0122808, 0.0142325] across teacher seed blocks. This is not an observed test error. Its numerical moments were evaluated by converged Gauss–Hermite quadrature, not certified integration-error arithmetic. The bound applies to the stipulated independent inputs and students additive across their two proximal input blocks, with identity soma and no direct somatic inputs. It does not automatically extend to arbitrary conductance models.

## Protocol and provenance

- `development_protocol.json` records the original model, input distribution, sample counts, five development seeds, rates, contrast definitions and code/production-forward hashes before development.
- `development_sgd_bracket_protocol.json` records an additional development-only SGD bracket after the original optimum fell at the upper rate boundary. The original outcomes remain intact. The unstable high-rate outcomes are also retained.
- `development_learning_rate_selection.csv` contains mean final validation NMSE over all development task/candidate/rule conditions. The selected single rate per optimizer is Adam 0.01 or SGD 0.3.
- `fresh_protocol.json` freezes the rates, seed list, primary contrasts, development source hashes and protocol hashes before fresh calibration or training.
- `interaction_bound_protocol.json` separately freezes the mechanism diagnostic before fresh training outcomes; the diagnostic did not alter training or rate selection.
- Slurm array jobs were 44644574 (initial development), 44644950 (additional SGD development) and 44645438 (fresh). Their standard output logs are retained here. Each task requested one CPU and 4 GB on `serial_requeue` with account `kempner_dev`; no GPU was used.

Each task uses 1,024 training, 1,024 validation, 4,096 test and 256 label-free calibration examples. Inputs are strictly positive independent lognormal variables, and targets are noiseless outputs of an independently perturbed conductance teacher. Training uses 1,000 updates with minibatches of 128, half-MSE divided by training target variance, and checkpoints at 0, 10, 100, 300 and 1,000. Student initial parameters, data and minibatch indices are shared across paired conditions. All models have seven compartments, six couplings, four excitatory contacts, six inhibitory contacts and sixteen positive trainable conductances in the same admissible range. Total learned conductance is recorded but is not constrained to equality.

The fixed calibrated broadcast is the mean initial student path field on label-free calibration inputs. Its profile stays frozen. The one-profile and two-subtree projection rules fit coefficients to the current exact student field on every trial; these are explicitly **oracle coefficient diagnostics**, not implementable encoder demonstrations. Credit geometry is measured at common exact-trained checkpoint states. It uses nonsomatic path fields, whereas the soma error stays exact in every condition.

## Tables and aggregation

- `runs/{development,development_sgd_bracket,fresh}/seed_*_curves.csv`: one raw fit/checkpoint per row, including validation/test/train NMSE and physical checks.
- `runs/.../seed_*_credit_diagnostics.csv`: rule-specific gradient and path geometry at the common exact-trained state.
- `runs/.../seed_*_final_states.npz`: final log conductances and initial calibration profiles.
- `runs/.../seed_*_audit.json`: count, timing, protocol hashes and paired task-spectrum checks.
- `summaries/fresh/all_learning_curves.csv`: concatenated raw nested outcomes, not seed-aggregated values.
- `summaries/fresh/learning_summary.csv`: mean and 95% interval after averaging compatible conditions over three tasks within each seed, or incompatible conditions over three tasks and two mismatched candidates within each seed. The intervals use 10,000 whole-seed bootstrap draws. `n_fits` is a nested fit count, not the independent sample size.
- `summaries/fresh/paired_seed_contrasts.csv`: the 20 independent paired seed differences for each named contrast, suitable for individual points in figures.
- `summaries/fresh/paired_contrasts.csv`: the corresponding means, pointwise 95% seed-bootstrap intervals and positive-seed counts.
- `summaries/fresh/credit_geometry_summary.csv` and `physical_parameter_summary.csv`: descriptive geometry and learned-conductance summaries. The geometry table supplies means without confidence intervals; bootstrap intervals should be derived from its raw source if plotted.
- `interaction_bound/population_bounds.csv`: the 128-node population bound for all 180 seed/task/candidate combinations, including compatible zeros.
- `interaction_bound/quadrature_bounds.csv`, `omitted_components.csv`, `report.json`, `seed_summary.json`: convergence, individual orthogonal components and seed-level bound summaries.

The two primary contrasts were the Adam exact-gradient grouping effect and the Adam calibrated-broadcast-minus-exact effect within compatible groupings. The remaining contrasts are descriptive. Intervals are pointwise, not simultaneous familywise intervals.

## Verification and reproduction

The full protocol equations and interaction-bound derivation are supplied in `analysis/morphology_credit_revision_20260905/METHODS_CONDUCTANCE_BRIDGE.tex` relative to the journal root. Code is in `scripts/morphology_conductance/`.

From the journal root, focused verification is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s scripts/morphology_conductance -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/morphology_conductance/audit.py
```

`test_results.txt` records all five passing tests. They compare the conductance forward pass and all sixteen parameter/ten input derivatives with the repository production implementation and autograd; check paired spectrum invariance, projection normal equations and label-free calibration; and independently check the missing mixed interaction and moment-factorization bound. `validation.json` records the independent source/seal/count/contrast/state audit. The maximum paired input-gradient spectrum difference was 2.43e-17, the maximum independently recomputed contrast/interval difference was 9.72e-17, no fresh parameter bound was reached, and all retained outcomes were finite.

The training commands in the retained Slurm scripts consume the frozen protocols and refuse to overwrite completed seed outcomes. To rerun the complete development/freeze/fresh pipeline, use an isolated copy with a separate output directory; do not replace the retained source data or call the original three-rate selection routine over the amended development study. The amended selection is implemented in `sgd_bracket.py select_fresh` and includes all seven SGD rates.
