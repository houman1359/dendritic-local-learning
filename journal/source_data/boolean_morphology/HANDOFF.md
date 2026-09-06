# Boolean study handoff

The study is complete, with no scientific-source amendments, numerical
failures or excluded fresh outcomes. The original morphology experiments,
manuscript, figure builders and release builders were not edited by this task.
New material is restricted to `scripts/boolean_morphology/` and
`source_data/boolean_morphology/`; the independent mathematical audit is owned
separately under `boolean_theory/`.

## Findings to preserve in the paper

1. The primary Adam XOR-of-AND compatible-grouping advantage is 0.563174
   clean NMSE, adjusted 97.5% interval [0.549144, 0.582749]. It passes the
   prespecified interval and 0.01 mean-effect criteria.
2. The second primary effect, compatible broadcast minus exact, is 0.002507
   [0.001780, 0.003318]. Its interval is positive, but its mean is below the
   prespecified practical margin. Both rules reach 100% exhaustive
   classification on this compatible task in every fresh seed. Preserve
   this limited primary credit result.
3. Parity has a large descriptive credit effect across every tree; it was
   not the primary task. AND4/OR4 work across shapes; mixed tasks distinguish
   input groupings, and the nested template favors the comb. These are seven
   fixed templates, not newly sampled Boolean function families.
4. Canonical gate sensitivities illustrate one internal representation.
   Freely trained coefficients can change that representation, so canonical
   signed fields do not establish that broadcast must fail to learn the same
   root function. The actual small XOR-of-AND credit effect demonstrates this
   distinction. Classification and continuous regression capacity also
   remain distinct: the analytic lower bounds concern clean squared error.
5. Both optimizers retain three rates, all conditions and all final weights.
   Rate-grid and parameter-box comparisons are bounded recipes. The primary
   exact/broadcast comparison uses separately development-selected rates;
   same-rate descriptive checks are exported.

## Protocol and results

- Five development seeds: 2026110–2026114, 1,680 fits.
- Twenty fresh seeds: 2026210–2026229, 6,720 fits, 53,760 checkpoint rows.
- Excluded complete smoke: seed 2026000, 336 fits; never pooled with fresh
  confirmation. A separate full replay validates one fresh seed's 336 fits
  and is not an additional statistical replicate.
- Adam selected rates: exact 0.003, broadcast 0.01. SGD: exact 0.03,
  broadcast 0.01. Selection pools all seven tasks and all four trees in
  development; no fresh outcome selects a rate.
- Protocol SHA-256:
  `4373c421dd8ad60de8275d8384f441fc7c19d000232081bbb5be2eb7df0597df`.
- Selection seal SHA-256:
  `965160e7d9126f974da817735dfb0ec1df35b6f09cc2056fabdab5a9f3c5a8a1`.
- `METHODS.md` gives coordinates, streams, update rules, safeguards, seed
  accounting and primary/secondary scope. `RESULTS.md` gives the primary
  table and readable family comparisons. `SCHEMA.md` gives all source shapes.

For S44, use `condition_summary.csv` (four optimizer/rule matrices),
`primary_contrasts.csv` and `paired_primary_contrasts.csv`,
`all_rate_condition_summary.csv`, `trajectory_summary.csv`, and
`selected_endpoints.csv`. Metrics include continuous clean NMSE, threshold
accuracy, balanced accuracy and gradient alignment. `seed_reference_controls.csv`
adds constant-mean and uniform-random-candidate expectations. All rate-specific
raw outcomes are in `all_rate_endpoints.csv` and `all_trajectories.csv`.

## Validation and compute

Three pre-freeze tests passed, including all 48 exact-gradient central
differences across four trees and truth-table centering/variance checks.
Own post-run validation reconstructed all 8,400 development/fresh clean and
noisy endpoints to at most 2.22e-16 error, reproduced every saved random
stream and verified all source/seal/output hashes and timing. One fresh
seed's entire training replay was bit-identical, apart from elapsed time.

The separately written Boolean-theory audit imported no learning core. It
matched all 8,400 endpoint and initialization records, all metrics, rate
selection and both paired/bootstrap summaries. All 67,200 development/fresh
checkpoint losses obey the independent structural lower bounds; all 3,600
incompatible endpoints exceed their bounds, with smallest margin 0.000321828.
See `source_data/boolean_theory/learning_validation.json` for exact independent
checks and `validation_report.json` for the replay and stream audit.

Slurm jobs 44660707 (smoke), 44660840 (development) and 44661180 (fresh)
completed successfully. Fresh concurrency was capped at eight one-CPU jobs.
`slurm_accounting.csv` records 342 allocated CPU-seconds over 26 tasks;
scheduler CPU-use and RSS fields are missing/zero and are not interpreted
as actual utilization. Numerical validation used the current environment;
no fresh dependency installation is claimed.

## Integration and release inputs

Register the entire scientific folder in numerical Source Data, including
development, original fresh data, excluded-smoke records with their explicit
status, seals, audits and all candidate/rate controls. Python execution
requires NumPy and pandas; tests use pytest. No PyTorch or GPU is used.
The seven code/job files and hashes are in `software_file_inventory.tsv`.
The portable execution entry point is `experiment.py`; `run_cpu.sbatch`
records local scheduling. No external datasets, submissions or unrelated
cluster jobs were accessed or modified.

The figure/manuscript/release owners can now integrate these frozen results.
No additional training or hyperparameter exploration is required for the
prespecified study. Author interpretation should retain the failed practical
credit criterion and distinguish seven-template learning from a general
morphology-selection or biological law.
