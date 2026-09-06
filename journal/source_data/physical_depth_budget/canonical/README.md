# Physical-depth budget sensitivity: complete compact Source Data

The 60 fits restart the original configurations and seeds with the maximum budget increased from 180 to 600 epochs. The original stopping rule and validation selection are retained. These are budget controls on existing seeds, not a fresh replication or a convergence guarantee.

`observed_training_histories.csv` retains every observed training and validation loss, best epoch, patience counter and elapsed time. `observed_epoch_metrics.csv` retains every metric logged at every epoch, including test metrics that were never used for selection. The separate plotting table carries stopped runs forward and marks carried rows.

Each run has its exact original and extension configurations, endpoint metrics, resource counts, seed metadata and audit. `raw_source_inventory.csv` preserves the complete SHA-256 inventory of the underlying raw JSON files and checkpoints and the original source files. Raw checkpoints and per-epoch JSON trees are preserved locally; their data are consolidated here. Absolute paths in frozen protocols and configurations document the source environment.

Rebuild with `python scripts/physical_depth_budget/analyze_extension.py` and `python scripts/physical_depth_budget/export_compact.py` after the frozen extension jobs complete. No original result file is replaced.
