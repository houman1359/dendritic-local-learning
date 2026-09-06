#!/bin/bash
#SBATCH --job-name=depth_budget_extension
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BUDGET_SCRIPT="${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/physical_depth_budget/extend.py"
if [[ "${1:-}" == benchmark ]]; then
    python "$BUDGET_SCRIPT" benchmark --index 40
else
    python "$BUDGET_SCRIPT" run --index "${SLURM_ARRAY_TASK_ID}"
fi
