#!/bin/bash
#SBATCH --job-name=selection-dendrinet
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue,shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SELECTION_ROOT:?Use persistent project_b storage}"
: "${SELECTION_PHASE:?development or fresh}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SELECTION_ROOT/base:$SELECTION_ROOT/runtime/src"
python -B "$SELECTION_ROOT/study/experiment.py" --root "$SELECTION_ROOT" --phase "$SELECTION_PHASE" --index "$SLURM_ARRAY_TASK_ID"
