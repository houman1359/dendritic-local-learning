# Four-input Boolean morphology learning

This folder retains the complete frozen seven-template study. Its primary
result is a large compatible-grouping advantage for XOR-of-AND. Its second
primary result is a small exact-credit advantage that does **not** meet the
prespecified practical margin: unit broadcast also learns that compatible
tree. Larger parity credit effects are descriptive. Read `METHODS.md` and
`RESULTS.md` before reusing the tables.

The original core is `scripts/boolean_morphology/{model,experiment}.py` and
`__init__.py`. `protocol_freeze.json` binds those files before any fitting;
`selection_freeze.json` binds all development outputs and selected rates
before fresh fitting. No runtime amendments were necessary. All original
data, final weights and minibatches are retained in per-seed NPZ files.
`runs/excluded_smoke/` is an implementation check, not fresh confirmation.

Run from the journal directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q scripts/boolean_morphology/test_model.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python scripts/boolean_morphology/validate.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python scripts/boolean_morphology/analyze.py
```

Training entry points are `experiment.py freeze`, `run --split development
--seed ID`, `select`, and `run --split fresh --seed ID`, in that order. The
runner refuses to overwrite completed seed files. To regenerate training,
use a separate copy of the package and a fresh output directory, retain the
declared seeds/protocol, run the complete excluded smoke first, then all five
development seeds before selecting rates and running all twenty fresh seeds.
The scheduler wrapper is `run_cpu.sbatch`; `sbatch --array=0-19%8 ...
run_cpu.sbatch fresh 2026210` reproduces the fresh array. Array task IDs add
to the seed base. These scheduling files describe the local execution; the
Python CLI is the portable single-process entry point.

Primary display files are `condition_summary.csv`, `primary_contrasts.csv`,
`paired_primary_contrasts.csv`, `trajectory_summary.csv` and
`all_rate_condition_summary.csv`. Raw and selected-rate endpoint/trajectory
tables, same-rate contrasts and per-seed reference controls provide the full
context. `SCHEMA.md` describes coordinates and dimensions.

`validation_report.json` records complete endpoint/stream/hash checks and a
full one-seed training replay. Independent mathematical and saved-output
checks live separately under `source_data/boolean_theory/`. The statistical
unit is the seed; the functional scope remains seven fixed templates.
