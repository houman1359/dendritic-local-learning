# Physical depth under a longer training budget

All 60 planned restarts completed from the pinned clean implementation. Each run reused its original task, seed, model and optimizer recipe, with the epoch cap raised from 180 to 600 and the original patience 30 retained. Endpoints are selected by validation loss; the test set is used only for reporting.

The 180 and 600 comparisons below come from the same new trajectories. Original outcomes remain in their existing files, and full first-window validation discrepancies from the historical runs are reported separately. A larger maximum budget is not a claim that every run converged.

| Contrast | Budget | Mean difference (pp) | 95% paired-seed interval |
|---|---:|---:|---:|
| depth_gain_exact_bp | 180 | 30.82 | 30.47 to 31.25 |
| localca_path_minus_shared | 180 | 10.86 | 10.15 to 11.62 |
| bp_exact_minus_broadcast | 180 | 0.60 | 0.21 to 1.03 |
| broadcast_bp_minus_localca_recipe | 180 | 13.45 | 12.87 to 14.04 |
| depth_gain_exact_bp | 600 | 38.17 | 37.98 to 38.39 |
| localca_path_minus_shared | 600 | -1.52 | -1.83 to -1.23 |
| bp_exact_minus_broadcast | 600 | 0.48 | 0.40 to 0.55 |
| broadcast_bp_minus_localca_recipe | 600 | 1.54 | 1.39 to 1.70 |

52 of 60 runs reached the 600-epoch cap. Their late validation slopes and best epochs are retained in extension_endpoints.csv. The finite-budget qualification remains wherever those curves are still declining.

Intervals are descriptive 95% whole-seed bootstrap intervals with 10 paired seeds. There is no new confirmatory multiplicity claim. All null, small, reversed and positive results are retained.
