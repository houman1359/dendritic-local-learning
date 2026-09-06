# Finite-calibration morphology evidence

This directory contains two separately frozen experiments in the eight-input, seven-node multilinear tree model. It contains no replacement-project production changes or biological data.

The primary finite-calibration experiment uses 20 fresh seed blocks, four function families and six calibration conditions. A generic Lasso estimates interactions from noisy observations; selection among twelve fixed candidates is primary, and adaptive construction at 256 queries/noise SD 0.5 is secondary. Independent noisy samples train the selected architecture's coefficients by alternating least squares. `METHODS.md`, `RESULTS.md` and `SI_NOTE.md` provide the manuscript-ready account.

The `end_to_end/` follow-up uses another 20 fresh seed blocks. The frozen estimated tree feeds a gradient learner initialized independently of target labels. Estimated, fixed and privileged oracle-informed trees are crossed with exact and broadcast credit under fixed Adam recipes. Its protocol, selection seal, per-seed trajectories and final parameter arrays are retained separately.

| Primary source | Contents |
|---|---|
| `protocol.json`, `protocol_freeze.json` | Scientific settings, development-selected scale/baselines, original source hashes and freeze time |
| `confirmatory_selection_seal.json` | Hashes of all 100 calibration observation/choice files, sealed before final fits |
| `development/` | All five development seeds, candidate fits, score records and calibration observations |
| `confirmatory/` | Original observations, selections, all 2,240 restart outcomes and source-stream metadata |
| `all_fit_outcomes.csv`, `candidate_outcomes.csv` | Convenience aggregates; the latter retains validation-selected restarts |
| `policy_outcomes.csv`, `policy_summary.csv` | Per-task paired policies and descriptive 20-seed intervals |
| `primary_contrasts.csv`, `paired_contrast_*.csv` | Primary and secondary paired contrasts with declared multiplicity handling |
| `calibration_timing.csv` | Measured estimator, candidate-score and pilot time |
| `candidate_bound_diagnostics.csv` | True population-error lower-bound checks made only after sealing |
| `validation_report.json` | Independent reconstruction of all choices, source hashes, restart decisions and representative full fits |

Original per-seed CSVs preserve roundtrip floating-point values. Read those with `pandas.read_csv(..., float_precision="round_trip")` when checking last-bit restart decisions: convenience aggregate parse/export can collapse numerical ties. This does not affect the reported effect sizes.

Implementation failures are retained. The first primary training attempt failed before fitting because sorted JSON node keys were decoded in lexical order. A subsequent launcher resolved a historical module with the same name. `serialization_amendment.json` and `serialization_dispatch_amendment.json` document the narrowly scoped, pre-outcome corrections. The frozen scientific files and all selection records remained unchanged. The functioning primary training entry point is `scripts/morphology_calibration/run_serialization_fix_v2.py`; the original failing entry points are retained for provenance. The end-to-end follow-up separately records its helper-interface correction, failed logs and complete excluded-development smoke.

From the journal directory, summary and independent validation can be reproduced in the current numerical environment with:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python scripts/morphology_calibration/run.py summarize
python scripts/morphology_calibration/validate_results.py
python scripts/morphology_calibration/end_to_end.py summarize
python scripts/morphology_calibration/validate_end_to_end.py
```

These commands regenerate summary/validation files. Training is separate: the primary workers use `run_confirm_fixed_v2.sbatch`, and the end-to-end workers use `run_end_to_end_fixed.sbatch`. To reproduce a campaign from scratch, use a separate copy of the workspace and its output directory, preserving these frozen source records. The source dependency is the retained `scripts/morphology_structure/` model and constructor, plus the explicitly imported `scripts/morphology_credit/experiment.py` for the gradient follow-up.

The biological scope is limited: known Rademacher coordinates, an exponential-size sparse-polynomial dictionary, signed multilinear units and an edge/parameter budget. Neither experiment proves optimal conductance morphology or transfer to unknown task families. Full-target and hindsight references are privileged and are labeled separately from observation-based selectors.
