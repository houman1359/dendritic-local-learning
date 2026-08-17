# Validated regular-tree feedback rerun

These specifications define the current-code replacement for the former
mixed-provenance 15-seed result. They ran on 31 July 2026 as Slurm jobs
36616195 (shunting) and 36616196 (additive). All 60 outputs passed the frozen
checks below and now supply Figure 2.

## Frozen design

Each YAML produces 30 configurations: seeds 42-56 crossed with
matched-width/scalar-fallback and neuron-indexed ancestry-shared feedback.
Shunting and additive are separate sweeps so their gate policies are explicit
and cannot change through future alias defaults:

- shunting: `init_policy: analytical`;
- raw additive: `init_policy: occupancy_quantile`.

All other architecture, sparsity, optimizer, schedule, decoder, and local-rule
settings match the fully archived July extension. Each array is limited to six
concurrent tasks, so running both allows at most 12 GPUs.

## Validated generation

On 31 July 2026, both files passed generate-only validation with the current
sweep manager. Each generated exactly 30 resolved configurations and two job
scripts, and each frozen manifest recorded the expected count. The validated
YAML hashes were:

- shunting: `0a1f9a4db29778835f212256e2a107d3d984c1ed604570e01bcfb4e14e1f4d7f`;
- additive: `33c48c002f9b3eaa10a1c7bf9b9c0995451bf03d2823d2524501b2f8570ddf61`.

Validation used Python 3.10.13 at Git commit
`4ba38bde22c2a3b8a564bfb6db398610b6a8269f`; the tracked worktree was dirty,
and the generator froze source hashes in each generated manifest.

Generate manifests and array scripts without submitting:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
  --config drafts/dendritic-local-learning/journal/configs/reruns/feedback_definition_shunting_15seed.yaml \
  --generate-only
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
  --config drafts/dendritic-local-learning/journal/configs/reruns/feedback_definition_additive_15seed.yaml \
  --generate-only
```

The submitted commands, after inspection of the generated manifests and
scheduler fields, were:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
  --config drafts/dendritic-local-learning/journal/configs/reruns/feedback_definition_shunting_15seed.yaml \
  --run
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
  --config drafts/dendritic-local-learning/journal/configs/reruns/feedback_definition_additive_15seed.yaml \
  --run
```

While both arrays were pending, their scheduler time limits were reduced
from four hours to one hour to improve backfill eligibility. Archived tasks in
the matched protocol averaged 219--447 seconds, so this did not change any
scientific configuration or truncate a completed run. The change is a
scheduler-only provenance field and must be retained with final Slurm
accounting. The pending arrays were then moved from `kempner_requeue` to the
permitted H200 `kempner_eng` partition to obtain allocation. This likewise
changed only scheduler placement, not a scientific field.

Override `--slurm-account` or `--slurm-partition` for another cluster.

## Completed acceptance checks

1. All 60 tasks produced exactly one final result and checkpoint per seed,
   architecture, and feedback condition.
2. Every resolved configuration matched the frozen design on all scientific
   fields; only the recorded scheduler fields changed.
3. Logs, checkpoints, manifests, Slurm accounting, software identity,
   hardware, and source hashes were retained.
4. The clean and former cohorts were compared before replacement, under the
   prespecified rule that the clean cohort would be used regardless of effect.
5. The validated clean table replaced the Figure 2 accuracy source; the old
   table was moved to `analysis/archive/figure2_pre_clean_20260731/`.

The validation output is
`analysis/feedback_rerun_validation/validation_summary.json`.

The matching fixed-checkpoint gradient diagnostic ran as Slurm array
`36625216` on two A100-SXM4-40GB GPUs. It evaluated the scalar-fallback and
neuron-indexed fields at all 60 clean checkpoints. The non-destructive
collector `scripts/collect_feedback_gradient_rerun.py` required the complete
120-row diagnostic design, joined every row to its checkpoint hash, and wrote
`analysis/feedback_rerun_validation/gradient_validation_summary.json` before
the table was promoted to Figure 2c.

## Measured compute

All runs used one NVIDIA H200. Summed task time was 2.108 GPU-hours for the
shunting array and 1.898 GPU-hours for the additive array, or 4.006 GPU-hours
for all 60 tasks. These are sums of Slurm elapsed time, not elapsed wall time
under parallel execution.
