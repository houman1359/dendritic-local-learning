#!/bin/bash
#SBATCH --job-name=selection-dendrinet
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SELECTION_ROOT:?Provide the frozen selection study}"
: "${SELECTION_PHASE:?Provide development or fresh}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SELECTION_ROOT/base:$SELECTION_ROOT/runtime/src"
python -B "$SELECTION_ROOT/scheduling/batch_worker.py"
