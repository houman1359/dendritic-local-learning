# Finite-calibration results

The frozen interaction selector improved candidate selection from noisy observations. With 256 calibration queries and label-noise SD 0.5, its mean regret was 0.02736 normalized MSE, compared with 0.15610 for the development-best fixed tree and 0.08723 for the two-sweep fitting pilot. The paired improvements were 0.12874 (Bonferroni-adjusted 97.5% interval 0.08259–0.17714) and 0.05987 (0.02677–0.09386), respectively. Both comparisons passed the preregistered positive-interval and 0.01-effect-margin criteria. These are absolute differences in target-variance-normalized MSE, not percentages of baseline error.

All 80 confirmatory functions were generated from 20 fresh seed blocks, with one function from each of four registered families per block. Their signs, coefficient magnitudes and input assignments differed from development. The estimator received a generic 255-monomial dictionary and 192 noisy fit observations; 64 additional calibration observations formed the pilot's gate. It received no target support or family label. This is confirmation on new functions within a known task mixture, rather than transfer to an unregistered task family.

| Policy at 256 queries/noise SD 0.5 | Mean independent-test NMSE | Mean regret against best of twelve trained candidates | 95% seed-bootstrap interval for regret |
|---|---:|---:|---:|
| Estimated interaction cuts | 0.36415 | 0.02736 | 0.01278–0.04553 |
| Development-best fixed / estimated-rank-only | 0.49289 | 0.15610 | 0.11961–0.19263 |
| Two-sweep fitting pilot | 0.42402 | 0.08723 | 0.06586–0.11002 |
| Uniform random choice, exact expectation | 0.52203 | 0.18523 | 0.16482–0.20613 |
| Full-target cut selector, privileged reference | 0.35836 | 0.02156 | 0.00986–0.03521 |
| Best trained candidate in hindsight | 0.33680 | 0 | — |

The rank-only mapping and development-best fixed policy chose the same balanced candidate. They are therefore redundant comparisons, rather than independent confirmations. The fixed-menu selector's improvement over the pilot was not uniform across task families: their matching-task means were nearly identical. The overall comparison averages the four families within each seed.

The secondary adaptive constructor selected a tree outside the twelve-candidate pool using the estimated coefficients. Its mean test NMSE was 0.07086, compared with 0.06172 for a privileged true-coefficient DP reference under the same finite training. Its negative regret against the twelve-tree pool was retained: it reflects a larger searched tree space with the same 28 coefficients and 14 edges, rather than an invalid negative loss.

| Family | Estimated adaptive tree NMSE | True-coefficient adaptive reference NMSE | Estimated fixed-menu selector NMSE | Two-sweep pilot NMSE |
|---|---:|---:|---:|---:|
| Matching | 0.001322 | 0.001279 | 0.299511 | 0.299700 |
| Quartet | 0.001436 | 0.001249 | 0.292481 | 0.365078 |
| Nested prefix | 0.001332 | 0.001366 | 0.399722 | 0.527512 |
| Random interactions | 0.279369 | 0.242983 | 0.464895 | 0.503800 |

The structured families were learned accurately after estimated construction, but the random controls retained substantial error, even under the target-informed reference. A small difference favoring an estimated tree over an oracle-informed tree in one family is possible because the reference optimizes a structural bound, not the finite-training endpoint. It is not a mathematically guaranteed endpoint optimum.

At the primary calibration condition, median measured CPU time was 0.00265 s for coefficient estimation and 0.04364 s for scoring the twelve candidates, versus 0.06615 s for the two-sweep pilot. These are small-model timings with a known exponential-size feature dictionary; they do not establish a scaling advantage. Calibration-size and noise sensitivities are retained in `policy_summary.csv`, along with all family-specific outcomes.

All 2,240 final fits completed without recorded numerical failure, and the Lasso fits reported no convergence failures. Independent validation reconstructed all 480 calibration selections, checked 100 sealed file hashes and all 1,120 validation-selected restart decisions, and verified 1,120 population-error lower-bound inequalities. Eight complete 32-sweep fit replays reproduced their stored metrics exactly. The maximum final parameter norm was 2.6615; no final fit exceeded a norm of 10,000. See `validation_report.json` for the checks and `candidate_bound_diagnostics.csv` for lower-bound gaps.

This experiment establishes finite-observation selection and construction within a sparse eight-variable multilinear setting. Alternating least squares supplied the final learner; biologically local learning and general conductance dynamics were not tested here. The separate `end_to_end/` experiment tests whether the frozen estimated construction also helps a gradient learner started from weights independent of target labels.
