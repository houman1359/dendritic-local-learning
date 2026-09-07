# Post hoc expanded learning-rate screen

All 288 fits completed. Larger learning rates did not improve opposed-task broadcast learning over the best original-grid rate at the matched 16,384-update budget. The original-rate optimum and the optimum across all six rates coincide for both broadcasts and both optimizers. This supports the reported conductance-task effect within the tested finite protocol.

The original three development seeds each received both primary tasks, all four credit rules, both optimizers and all six original-plus-added rates. Every outcome is retained in `runs`, with file hashes in each seed's audit. `protocol_freeze.json` pins the scientific adapter before outcomes; `validation_trigger_addendum.json` adds the comparison with the actual originally selected rates before any screen outcome was inspected. `expanded_rates.py` is immutable under that freeze. `selection_review.json` evaluates the original trigger, and `complete_trigger_review.json` evaluates both frozen triggers. Neither is satisfied, so no follow-up fits on the existing fresh blocks were run.

| Optimizer and broadcast | Originally selected rate | Best rate at 16,384 updates | Original-rate validation NMSE | Best validation NMSE | Relative improvement |
|---|---:|---:|---:|---:|---:|
| Adam unit | 0.03 | 0.003 | 0.7262471 | 0.7197761 | 0.8910% |
| Adam calibrated | 0.03 | 0.003 | 0.7262469 | 0.7197761 | 0.8910% |
| SGD unit | 0.3 | 0.1 | 0.7507204 | 0.7172427 | 4.4594% |
| SGD calibrated | 0.3 | 0.3 | 0.7699247 | 0.7699247 | 0% |

The threshold required at least 0.01 NMSE and at least 10% improvement for an opposed-task broadcast. The SGD-unit improvement meets the absolute threshold alone. Every rule received equal tuning, including exact and oracle controls; their full optima are in `selected_rates.csv`. No test outcomes entered the selection or follow-up decision. Checkpoint minima span the stated finite update budget; this is not evidence of convergence for every optimizer or learning rate.

The initial Slurm submission failed in the wrapper before any Python fits started because Slurm relocated the shell script to its spool directory. The retained execution record and failed logs identify that attempt. The corrected wrapper uses the submission directory or explicit `CONDUCTANCE_JOURNAL_ROOT`; its successful three seed jobs took 285, 230 and 229 scheduled seconds. No failed scientific candidate was removed.

Supplementary Fig. S52 is `figure_expanded_rates.pdf`; the complete plotted table is `figure_expanded_rates_source.csv`. The builder uses the common native figure canvas, font sizes, colors and line widths. Captions and Methods are supplied as separate TeX fragments for integration into the journal supplement.
