# Clean Figure 2 feedback rerun

The current Figure 2 accuracy comparison was frozen before execution and ran
on 31 July 2026.

- Shunting array: Slurm `36616195`, 30 runs.
- Additive array: Slurm `36616196`, 30 runs.
- Design: seeds 42--56 crossed with matched-width/scalar fallback and
  neuron-indexed ancestry sharing in each architecture.
- Hardware: one NVIDIA H200 per task.
- Summed Slurm elapsed time: 2.108 GPU-hours for shunting and 1.898 GPU-hours
  for additive, 4.006 GPU-hours in total.
- Python: 3.10.13.
- Git identity recorded at generation: `4ba38bde22c2a3b8a564bfb6db398610b6a8269f`;
  the tracked worktree was recorded as dirty, so critical source-file hashes
  are retained in both frozen sweep manifests.

`scripts/collect_feedback_rerun.py` required exactly 60 unique
architecture-by-feedback-by-seed results, checked each scientific
configuration after removing only the intended design axes, required every
final checkpoint, and recorded the configuration, metric, and checkpoint
SHA-256 hash for every row. The validation result is
`analysis/feedback_rerun_validation/validation_summary.json`.

The clean cohort prospectively replaced the former mixed archive regardless
of whether the measured effect grew or shrank. The prior table is retained at
`analysis/archive/figure2_pre_clean_20260731/feedback_accuracy_runs_mixed.csv`.
The clean publication table has SHA-256
`15498b41b3f1c51f202aeebdece28c1eaadedc1c66415bac9156d8a3654a8926`.

Mean test accuracy (sample standard deviation, 15 paired seeds) was:

| Architecture | Scalar fallback | Neuron-indexed ancestry | Mean paired gain |
|---|---:|---:|---:|
| Shunting | 0.90538 (0.00539) | 0.97159 (0.00111) | 0.06621 |
| Additive | 0.91763 (0.00783) | 0.97140 (0.00107) | 0.05377 |

All 15 within-architecture pairs improved in both architectures. The
architecture-specific gate-initialization policies were fixed explicitly;
these results support the feedback intervention within architecture and are
not used as a causal ranking of shunting versus additive integration.

## Fixed-checkpoint gradient diagnostic

Slurm array `36625216` evaluated both feedback fields at every checkpoint in
the clean cohort on NVIDIA A100-SXM4-40GB GPUs. Its two tasks completed in
268 and 238 seconds. The collector required 120 unique rows: seeds 42--56,
two trained-feedback modes, two diagnostic fields, and two architectures. It
also joined every row to the corresponding clean checkpoint hash.

For Figure 2c, both diagnostic fields were evaluated at the same
neuron-indexed-trained checkpoint. Mean dendritic-gradient cosine (sample
standard deviation, 15 seeds) was:

| Architecture | Scalar fallback | Neuron indexed | Mean paired gain |
|---|---:|---:|---:|
| Shunting | 0.01246 (0.05859) | 0.70820 (0.02695) | 0.69574 |
| Additive | -0.00066 (0.04670) | 0.62517 (0.03475) | 0.62583 |

All 15 paired checkpoints improved in both architectures. The clean gradient
table has SHA-256
`7a8a10d47e23ad13535dd4bb9246e4137483db01d50c2046d955b8683618767d`.
The superseded mixed table is retained under
`analysis/archive/figure2_pre_clean_20260731/`.
