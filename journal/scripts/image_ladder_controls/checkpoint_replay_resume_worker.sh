#!/bin/bash
#SBATCH --job-name=mnist_checkpoint_replay_fixed
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export PYTHONPATH=/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:${PYTHONPATH:-}
python '/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal/scripts/image_ladder_controls/portable_capture.py' \
 --checkpoint-root '/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906/exports/checkpoints_replay' \
 --study-root '/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal/source_data/image_ladder_controls' \
 --runtime-root '/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906/portable_runtime_allowlisted' \
 --dataset-root '/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/data' \
 --output-root '/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906/exports/capture_replay' --device cpu
