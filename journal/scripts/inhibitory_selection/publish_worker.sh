#!/bin/bash
#SBATCH --job-name=selection-verify
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SELECTION_ROOT:?Provide the frozen selection study}"
: "${SELECTION_JOURNAL:?Provide the paper journal folder}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SELECTION_ROOT/study:$SELECTION_ROOT/base:$SELECTION_ROOT/runtime/src"
python -B "$SELECTION_ROOT/reporting_revision_02/publish.py" --root "$SELECTION_ROOT" --journal "$SELECTION_JOURNAL"
