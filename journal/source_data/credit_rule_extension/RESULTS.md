# Balanced longer-budget credit-rule follow-up

Balanced longer-budget follow-up after inspecting all released 1024-step outcomes and the September7 referee review. Same seeds are reused: not a fresh confirmatory sample.

All720 original algebraic trajectories were replayed and extended to16,384 updates. No rules, seeds, tasks or rate-union members were removed. The original1024 checkpoint arrays, numerical curves, diagnostics and primary summaries reproduce. Both terminal and validation-selected outcomes remain available.

| Budget | Endpoint | Rate view | Quartic-minus-pairwise calibrated-minus-exact NMSE | 95% paired interval | Positive seeds |
|---:|---|---|---:|---|---:|
| 1024 | terminal | selected_rate | 0.837817 | [0.719044, 0.930753] | 20/20 |
| 1024 | validation_selected | selected_rate | 0.766091 | [0.645759, 0.870430] | 20/20 |
| 4096 | terminal | selected_rate | 0.915464 | [0.842422, 0.975367] | 20/20 |
| 4096 | validation_selected | selected_rate | 0.793674 | [0.694588, 0.883094] | 20/20 |
| 8192 | terminal | selected_rate | 0.933634 | [0.863975, 0.990722] | 20/20 |
| 8192 | validation_selected | selected_rate | 0.793720 | [0.694604, 0.883324] | 20/20 |
| 16384 | terminal | selected_rate | 0.951626 | [0.882898, 1.012319] | 20/20 |
| 16384 | validation_selected | selected_rate | 0.790339 | [0.691480, 0.879739] | 20/20 |
| 1024 | terminal | common_rate | 0.843484 | [0.727191, 0.934723] | 20/20 |
| 1024 | validation_selected | common_rate | 0.820016 | [0.706296, 0.908348] | 20/20 |
| 4096 | terminal | common_rate | 0.902575 | [0.827461, 0.963917] | 20/20 |
| 4096 | validation_selected | common_rate | 0.839922 | [0.758313, 0.912083] | 20/20 |
| 8192 | terminal | common_rate | 0.908792 | [0.841456, 0.962103] | 20/20 |
| 8192 | validation_selected | common_rate | 0.817633 | [0.733951, 0.891251] | 20/20 |
| 16384 | terminal | common_rate | 0.902761 | [0.837125, 0.957907] | 20/20 |
| 16384 | validation_selected | common_rate | 0.795220 | [0.701764, 0.877528] | 20/20 |

These are budget-indexed contrasts at historical rate choices and bounded coefficients, not convergence or minimum-feedback-rank results. Validation-selected states use only the declared checkpoints and their validation NMSE. The same original twenty seed blocks had already been examined; intervals are descriptive and are not fresh confirmatory evidence.

The condition/paired tables retain mean, median, intervals, all paired win counts and every seed. The quartic distribution table retains near-floor and stalled outcomes with explicit descriptive thresholds. Parameter projection and clipping are retained for all terminal trajectories, independently of checkpoint selection.
