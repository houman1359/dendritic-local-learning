#!/bin/bash
#SBATCH --job-name=credit-local-gate
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd -- "${LOCAL_GATE_JOURNAL_ROOT:?Provide journal root}"
if [[ "${1:-fresh}" == canary ]]; then
    python -B -m pytest -q scripts/conductance_local_gate/test_model.py
    python -B scripts/conductance_local_gate/run.py --seed 2026090799 --canary
else
    python -B scripts/conductance_local_gate/run.py --seed "$((2026090801 + SLURM_ARRAY_TASK_ID))"
fi
