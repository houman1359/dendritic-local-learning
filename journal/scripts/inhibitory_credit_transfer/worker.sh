#!/bin/bash
#SBATCH --job-name=inhib-credit-dev
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=06:00:00
set -euo pipefail
: "${INHIBITION_STUDY_ROOT:?Point to the isolated project_b snapshot}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$INHIBITION_STUDY_ROOT/runtime/src"
python -B "$INHIBITION_STUDY_ROOT/study/run.py" --root "$INHIBITION_STUDY_ROOT" --index "$SLURM_ARRAY_TASK_ID"
