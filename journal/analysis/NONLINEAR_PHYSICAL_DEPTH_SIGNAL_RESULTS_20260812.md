# Final exploratory signal-accessibility result

Completed: 12 August 2026.  Status: the frozen ladder selector failed narrowly;
one fixed boundary-resolution point was frozen before further outcomes.

| Signal delta | D1 `[8]` | D2 `[2,3]` | D3 `[2,1,2]` | D3 minus D1 |
|---:|---:|---:|---:|---:|
| 0.24 | 0.5270 | 0.5524 | 0.6273 | +0.1003 |
| 0.36 | 0.5439 | 0.5814 | 0.7254 | +0.1815 |
| 0.48 | 0.5633 | 0.6064 | 0.7976 | +0.2343 |
| 0.72 | 0.5969 | 0.6578 | 0.8967 | +0.2998 |

Values are paired two-seed mean test accuracies.  The D3-minus-D1 contrast is
positive in every signal cell and in both seeds, reaching +0.2914 and +0.3082
at signal 0.72.  Nevertheless, the prespecified selector requires every
morphology mean to be at least 0.60.  D1 reaches only 0.59695, so no ladder
cell passes and none is promoted.

The miss is 0.00305, and both D1 seeds exceed the individual 0.57 floor.  A
single signal 0.80 boundary check with two new seeds was therefore frozen.
There will be no further exploratory signal search.  Any confirmatory study
must use later, disjoint seeds and report all failed pilots.

## Provenance

- Run:
  `nonlinear_physical_depth_runs/journal_exploratory_physical_depth_signal_accessibility_bp_20260812002559`
- Frozen manifest SHA256:
  `15f1587014f4f59ca854eae472eee3fea0acfd1ec52f31e1d008c0e273db1750`
- Row-level source: `source_data/nonlinear_physical_depth_signal/bp_seed_rows.csv`
