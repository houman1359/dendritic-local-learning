#!/bin/bash
#SBATCH --job-name=selection-rates
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SELECTION_ROOT:?Provide the selection study}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SELECTION_ROOT/study:$SELECTION_ROOT/base:$SELECTION_ROOT/runtime/src"
python -B "$SELECTION_ROOT/reporting/rate_sensitivity.py" prepare --root "$SELECTION_ROOT"
python -B "$SELECTION_ROOT/reporting/rate_sensitivity.py" batch --root "$SELECTION_ROOT"
