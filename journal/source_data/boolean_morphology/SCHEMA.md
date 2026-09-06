# Numerical source schema

- `protocol.json`, `protocol_freeze.json`: scientific choices and pre-fit core
  source hashes. `selected_rates.json`, `selection_freeze.json`: development
  choices and every development output hash before fresh runs.
- `development_rate_scores.csv`: twelve pooled development scores, one per
  optimizer/rule/rate. Pooling includes five seeds, seven tasks and four trees.
- `runs/{development,fresh}/seed_ID.csv`: 2,688 rows per seed, indexed by
  `condition_id` (0–335) and `step`. Conditions contain `family`, `tree`,
  `optimizer`, `rule`, `rate`. Metrics are `population_nmse` (clean primary),
  `test_noisy_nmse`, full-domain `accuracy`, `balanced_accuracy`, and
  `gradient_cosine` between delivered and exact full-domain gradients.
  Clipping counts are cumulative. `failed` and `failure_step` retain failure
  status; `elapsed_seconds` is cumulative elapsed training/evaluation time.
- Corresponding `seed_ID.npz`: final weights `(336,3,4)`, postorder child
  indices `(336,3,2)`, condition-to-family index `(336,)`, permutation `(4,)`,
  initial weights `(7,3,4)`, training inputs/labels `(7,256,4)/(7,256)`, test
  inputs/labels `(7,1024,4)/(7,1024)`, batch indices `(7,2048,32)`, population
  inputs `(16,4)`, and normalized/raw labels `(7,16)`. Leaves have indices
  0–3; internal nodes are 4–6; coefficient order is constant,left,right,product.
  The semantic variable order is `physical_x[:, permutation]`. Family order
  is declared in `protocol.json`. Final weights correspond to CSV condition IDs.
- Corresponding `seed_ID.json`: source and output hashes, timestamps,
  scheduler identifier, fit counts, failure counts and selection-seal hash.
- `all_trajectories.csv`: 53,760 fresh rows. `all_rate_endpoints.csv`: 6,720
  fresh endpoints. `selected_trajectories.csv`: 17,920 rows;
  `selected_endpoints.csv`: 2,240 endpoints at the frozen selected rates.
- `condition_summary.csv`: 112 family/tree/optimizer/rule rows at selected
  rates. `all_rate_condition_summary.csv`: 336 rows including rate.
  `trajectory_summary.csv`: 896 rows including step. Each reports twenty
  seed means and pointwise 95% bootstrap intervals. Metric column pattern is
  `mean_METRIC`, `METRIC_ci95_low`, `METRIC_ci95_high`. Nested fit count is
  not an independent sample count.
- `paired_primary_contrasts.csv`: forty seed differences for the two
  prespecified Adam XOR-of-AND comparisons. `primary_contrasts.csv`: two rows,
  with 95% descriptive and 97.5% adjusted intervals, mean difference, 0.01
  margin and combined criterion result. Positive differences favor compatible
  exact-credit learning. Per-rule selected rates may differ.
- `paired_same_rate_contrasts.csv`, `same_rate_contrast_summary.csv`: both
  contrasts under every optimizer and common rate; descriptive controls.
  The primary flag is false, and no new confirmatory hypothesis is introduced.
- `seed_reference_controls.csv`: per-seed constant-mean prediction and the
  expected endpoint under uniform selection of the four candidates. The
  constant baseline uses no fitted parameters. Random candidate selection
  averages retained endpoints; it does not add training fits.
- `target_normalization.csv`: full truth-table class proportions, variances,
  thresholds and majority/constant baselines. All NMSE values use the exact
  normalized clean target variance, not the noisy sample variance.
- `independent_validation_rows.csv`, `validation_report.json`: endpoint,
  stream, timing and hash checks; the report also describes one complete
  training replay. `analysis_record.json` binds the summary script and seals.
- `slurm_accounting.csv`: scheduler resource records. `ElapsedRaw*AllocCPUS`
  is allocated CPU-seconds, not measured utilization; missing RSS is preserved.

All fresh conditions and rates are retained, including ties and near-zero
loss. Files in `runs/excluded_smoke/` belong to the explicitly excluded
complete implementation smoke and must not be pooled with confirmation.
