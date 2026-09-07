#!/bin/bash
#SBATCH --job-name=mnist_projectedK1
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTHONPATH=/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:${PYTHONPATH:-}
python "${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/image_ladder_controls/projected_k1.py" run --phase "${1:-development}" --index "$SLURM_ARRAY_TASK_ID"
