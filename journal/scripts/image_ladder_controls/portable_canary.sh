#!/bin/bash
#SBATCH --job-name=mnist_portable
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTHONPATH=/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:${PYTHONPATH:-}
python "${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/image_ladder_controls/portable_run.py" \
 --study-root "${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/source_data/image_ladder_controls" \
 --runtime-root /n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906/portable_runtime_allowlisted \
 --dataset-root "${SLURM_SUBMIT_DIR}/data" \
 --output-root "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906/portable_allowlist_canary_${SLURM_ARRAY_TASK_ID}" \
 --stage canary --index "$SLURM_ARRAY_TASK_ID"
