#!/bin/bash
#SBATCH --job-name=inhib-credit-report
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
set -euo pipefail
: "${INHIBITION_STUDY_ROOT:?Point to the isolated project_b snapshot}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$INHIBITION_STUDY_ROOT/study:$INHIBITION_STUDY_ROOT/runtime/src"
python -B "$INHIBITION_STUDY_ROOT/reporting/report.py" --root "$INHIBITION_STUDY_ROOT" --output "$INHIBITION_STUDY_ROOT/development_report"
