#!/bin/bash
#SBATCH --job-name=conductance-grid
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CONDUCTANCE_JOURNAL_ROOT="${CONDUCTANCE_JOURNAL_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$CONDUCTANCE_JOURNAL_ROOT" ]]; then
    CONDUCTANCE_JOURNAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd -- "$CONDUCTANCE_JOURNAL_ROOT"
python scripts/conductance_credit_demand/expanded_rates.py run --seed "$((201 + SLURM_ARRAY_TASK_ID))"
