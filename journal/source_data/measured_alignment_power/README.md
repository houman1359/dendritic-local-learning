# Measured ancestry-response sensitivity

This directory contains a frozen, explicitly model-based sensitivity analysis of the original seven-target partial-rank test. It also releases the small pair-level and response inputs previously retained only in the credit-routing results directory. The observed correlations and original manuscript outcomes are unchanged.

Start with `PROTOCOL.md`, `protocol_freeze.json`, and `input_audit.json`. The protocol was committed before scientific simulation. `submission_record.json` and any `scheduling_amendments.json` describe execution; scheduling changes do not alter the registered model or outcomes. After completion, `RESULTS.md`, `RESULTS.json`, `power_summary.csv`, and `figures/measured_alignment_sensitivity.pdf` give the results. `runs/chunk_*.npz` retains every scan and target effect and exact P value, so summary calculations do not require rerunning simulations.

## Portable inputs

`input_manifest.json` lists original paths and SHA-256 digests, released hashes, transformations, and definitions. Each scan folder contains byte-identical original contact pairs, mapped contacts and the original summary, plus a small NPZ containing only the mapped presynaptic responses, exact stimulus identities and repeat means. `recording_identity.csv` permits exact identification of observations duplicated across extractions. `ancestral_segment_closure.csv` supplies only the tree parent/length records needed to reproduce the shared-path kernel. No unrelated task features, target fluorescence, private revision logs or external data are required.

The 13 scans contain 125 mapped-partner observations of 102 distinct presynaptic roots and 124 distinct optical recordings. Twenty target/root combinations appear in multiple scans; two roots are shared between targets. Each scan has 136 repeated stimulus conditions, whose original condition hashes identify 316 distinct repeated stimuli across the complete dataset. The input audit independently reconstructs the original per-scan partial-rank correlations and split-half reliabilities to floating-point precision.

## Reproduce

From the journal directory, with NumPy, SciPy, pandas, Matplotlib and pytest available:

```
python -m pytest -q scripts/measured_alignment_power/test_model.py
python scripts/measured_alignment_power/portable_replay.py --verify-only
python scripts/measured_alignment_power/portable_replay.py --chunk 0 --output /path/to/new/replay_folder
python scripts/measured_alignment_power/report.py
python scripts/measured_alignment_power/audit_results.py
```

Complete simulation requires chunks 0 through 19. Existing scientific chunk outputs are deliberately not overwritten. `portable_replay.py` verifies every registered scientific-source and archived-input identity, then calls the unchanged numerical runner in a separate empty directory and compares the arrays with the corresponding released chunk. If release packaging translates absolute provenance paths or the scheduler working directory, the original expected hashes remain unchanged and the common release verifier requires an authenticated original-to-released hash chain. A translated file without that chain, or any unregistered scientific edit, is rejected. `test_portable_replay.py` checks this behavior. The report and arithmetic audit leave scientific chunks and observed inputs unchanged.

The original `run.py --chunk N` and `run.py --calibration-audit` commands remain available for an unmodified, initially empty working copy. The latter reproduces the frozen repeat-reliability calibration. It is not needed to rebuild the supplied summaries or figures.

## Meaning of the sensitivity number

The primary signal axis is the fraction of reliable tuning variance supplied by the specified positive-semidefinite Brownian ancestry kernel. Its corresponding mean observed partial-rank effect is reported as a model-conditional interpretation, not a universal detectable correlation. Both the 80% crossing and maximal-signal power depend on the observed sampling, reliability model and injected covariance family. The seven-unit exact signed-rank test is discrete, but sample size seven alone does not impose a universal power ceiling below one. See the full protocol for these distinctions.
