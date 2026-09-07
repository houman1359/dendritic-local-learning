#!/bin/bash
#SBATCH --job-name=mnist_capture
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:${PYTHONPATH:-}
python "${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/image_ladder_controls/capture.py" --mode "${1:-fresh}" --index "$SLURM_ARRAY_TASK_ID"
