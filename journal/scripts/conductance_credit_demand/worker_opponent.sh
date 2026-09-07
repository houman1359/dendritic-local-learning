#!/bin/bash
#SBATCH --job-name=conductance-opponent
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CONDUCTANCE_JOURNAL_ROOT="${CONDUCTANCE_JOURNAL_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$CONDUCTANCE_JOURNAL_ROOT" ]]; then
    CONDUCTANCE_JOURNAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd -- "$CONDUCTANCE_JOURNAL_ROOT"
python scripts/conductance_credit_demand/run_opponent.py run --phase "$1" --seed "$(( $2 + SLURM_ARRAY_TASK_ID ))"
