#!/bin/bash
#SBATCH --job-name=gate-report
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:10:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd -- "${LOCAL_GATE_JOURNAL_ROOT:?}"
python -B scripts/conductance_local_gate/report.py
