# Portable replay of the frozen conductance studies

`portable_run.py` runs the original, verified fit functions in a restored reviewer package and a new output directory. The original scientific files remain unchanged. Their original environment guard requires NumPy 2.2.6; the portable entry point verifies the scientific source and protocol independently, then records the actual Python, NumPy and pandas versions separately. It does not claim that another environment matches the original one.

Run these commands from the restored journal directory, with the reviewer environment activated. Restore the corresponding Source Data under `source_data/conductance_credit_demand`, or supply its separate location through `--study-root`. The journal must contain `code/release_noise/release_hashes.py`, the released scientific scripts, and their release hash sidecars when transformations were applied.

```bash
python scripts/conductance_credit_demand/portable_run.py \
  --study-root source_data/conductance_credit_demand \
  --family opponent --phase fresh --seed 2101 --verify-only

python scripts/conductance_credit_demand/portable_run.py \
  --study-root source_data/conductance_credit_demand \
  --family opponent --phase fresh --seed 2101 \
  --output-root /tmp/opponent-reviewer-canary \
  --excluded-smoke-steps 4

python scripts/conductance_credit_demand/portable_run.py \
  --study-root source_data/conductance_credit_demand \
  --family opponent --phase fresh --seed 2101 \
  --output-root /tmp/opponent-reviewer-primary

python scripts/conductance_credit_demand/portable_run.py \
  --study-root source_data/conductance_credit_demand \
  --family opponent --phase extension --seed 2101 \
  --output-root /tmp/opponent-reviewer-extended
```

The primary and extended commands execute all 16 frozen conditions for that seed at 4,096 and 16,384 updates, respectively. Extension mode replays the original initialization and minibatch stream through the predeclared extended checkpoints; it does not import historical optimizer state. Use `--family first --seed 1101` for the first study (12 conditions), or `--phase development --seed 101` for first-study development and `--phase development --seed 201` for opponent development. Only cohort seeds declared in the frozen protocol are accepted. The launcher refuses an existing output directory or an output directory inside the restored study. `--journal-root` can point to a separately restored journal.

The optional smoke budget must be explicitly requested, lies between 1 and 64 updates, and is marked `excluded_from_scientific_results: true`. It preserves data sizes, rule identities, rates and all other scientific settings. Full replays retain their original budgets. Audits identify the verified inputs, original and actual environments, launcher hash, declared changes, output hashes and any failure. Floating-point trajectories may change across library versions; successful verification is not a promise of bitwise equality across environments.

The immutable `science_handoff_inventory_20260906.tsv` anchors the original protocol and scientific source digests. It is a historical inventory; unrelated presentation duplicates listed there need not be present in Source Data. Relevant original files must either match their original hashes or have an independently verified original-to-released chain through `release_noise`. A numeric change to the scientific protocol is rejected even if a release transformation is declared. The separate final inventory records later portable and posthoc files without rewriting the experimental freezes.

Validation uses `test_portable.py` and `portable_validation/run_clean_checks.py`. To repeat the clean-environment checks, pass the environment's Python executable without resolving its virtual-environment symlink:

```bash
python scripts/conductance_credit_demand/portable_validation/run_clean_checks.py \
  --python /path/to/reviewer-environment/bin/python
```

These installation checks are excluded from paper outcomes. They run two four-update canaries and one complete opponent seed at each full budget, recording the actual environment in `portable_validation/clean_environment_summary.json`. For historical Slurm wrappers, submit from the journal directory or set `CONDUCTANCE_JOURNAL_ROOT` explicitly; the portable launcher is the recommended reviewer entry point.
