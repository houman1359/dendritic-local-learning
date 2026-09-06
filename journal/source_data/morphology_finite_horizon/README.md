# Finite-horizon morphology dynamics investigation

Start with [REPORT.md](REPORT.md) and [DERIVATION.md](DERIVATION.md). This directory contains a new, separate investigation. Original experiment and manuscript sources were not edited.

The primary protocol was frozen before fresh calibration or training. Development seeds 6100–6104 were diagnostic; fresh seeds 9300–9319 were the independent within-family evaluation. The original 20 confirmation seeds were not rerun. All outcomes are retained, including the original selector's failure and the finite-cache mismatch of the population recurrence.

From the journal directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s scripts/morphology_dynamics -p 'test_*.py' -v
python scripts/morphology_dynamics/investigate.py analyze --cohort fresh
python scripts/morphology_dynamics/supplemental_diagnostics.py analyze --cohort fresh
python scripts/morphology_dynamics/render_and_audit.py
```

The array runner and scheduler scripts record the original execution recipe. Completed seed outcomes are deliberately not overwritten by `run`; reproductions should use a copied investigation directory with original outputs preserved. The frozen code checks the original runner, protocol and development-fit hashes. Prediction scripts require NumPy, SciPy and pandas; rendering also requires matplotlib.

Interpretation safeguards:

- `gaussian_plugin_*` methods estimate a conditional Gaussian model from calibration data only.
- `gaussian_oracle_*` methods use known population parameters and are privileged diagnostics.
- `empirical_split_fullbatch` fits the first 512 calibration examples and evaluates on the other 512.
- `pilot16` trains each candidate for 16 steps, then evaluates on calibration data; its training cost is reported.
- The original scalar and original development baselines are unchanged.
- The inexpensive observed-context baseline and its conditioning variant are secondary additions, sealed after the primary freeze but before fresh outcomes.
- Seed is the statistical unit. Tables preserve undefined correlations explicitly and retain every corresponding outcome.

Scheduler jobs: development array 44636894; fresh array 44637438; summaries and figures 44638030. All completed successfully on single-CPU tasks. Execution accounting is retained in `slurm_accounting.tsv`.
