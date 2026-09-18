#!/bin/bash
#SBATCH --job-name=inhib-credit-anatomy
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -euo pipefail
: "${INHIBITION_STUDY_ROOT:?Point to the isolated project_b snapshot}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$INHIBITION_STUDY_ROOT/runtime/src"
if [ "$SLURM_ARRAY_TASK_ID" = 3 ]; then
    python -B "$INHIBITION_STUDY_ROOT/study/functional.py" --root "$INHIBITION_STUDY_ROOT"
else
    cohorts=(microns_pilot microns_replication pinky)
    python -B "$INHIBITION_STUDY_ROOT/study/anatomy.py" --root "$INHIBITION_STUDY_ROOT" --cohort "${cohorts[$SLURM_ARRAY_TASK_ID]}"
fi
