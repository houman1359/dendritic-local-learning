# Interior-optimum reanalysis

Deterministic secondary analysis of the frozen depth-phase and
subtree-factorial outputs; see script docstring for definitions.

## Depth phase (aligned tree)
- seed x task-depth pairs: 200
- predicted D* (argmax utility) equals observed D* (argmin loss): 135/200 (0.675)
- observed D* = H fraction: 0.805
- predicted D* = H fraction: 0.545
- group-level argmax match per H: {1: {'observed_best_depth': 1, 'predicted_best_depth': 1}, 2: {'observed_best_depth': 2, 'predicted_best_depth': 2}, 3: {'observed_best_depth': 3, 'predicted_best_depth': 3}, 4: {'observed_best_depth': 4, 'predicted_best_depth': 2}}

## Factorial ancestry-minus-best-control contrast
- pipeline check vs published K=4 contrast: {'published': {'mean': 0.0127, 'ci_low': 0.0059, 'ci_high': 0.0197, 'wins': 15}, 'reproduced': {'mean': 0.012670898445, 'ci_low': 0.005810546874875, 'ci_high': 0.019506835950000002, 'wins': 15, 'n': 20}, 'passes': True}
- group-level peak: {'observed_peak_k': 4, 'predicted_peak_k': 8}
- per-seed peak agreement: 6/20

Per-K means:
 budget_k  mean_accuracy_contrast  accuracy_ci_low  accuracy_ci_high  accuracy_positive_seeds  mean_utility_contrast  utility_ci_low  utility_ci_high  utility_positive_seeds  mean_utility_contrast_non_oracle  utility_non_oracle_ci_low  utility_non_oracle_ci_high  utility_non_oracle_positive_seeds  n_seeds
        1               -0.453564        -0.502809         -0.405811                        0              -0.175880       -0.220917        -0.141851                       0                         -0.142853                  -0.200137                   -0.093788                                  0       20
        2               -0.243481        -0.283423         -0.204173                        0              -0.273091       -0.306969        -0.241811                       0                         -0.208688                  -0.267467                   -0.148872                                  0       20
        4                0.012671         0.005811          0.019580                       15              -0.141447       -0.194246        -0.092541                       1                         -0.100659                  -0.175099                   -0.023010                                  5       20
        8                0.000000         0.000000          0.000000                        0               0.000000        0.000000         0.000000                       0                          0.000000                   0.000000                    0.000000                                  0       20
