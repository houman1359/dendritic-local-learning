#!/bin/bash
#SBATCH --job-name=conductance-sgd-bracket
#SBATCH --partition=serial_requeue
#SBATCH --account=kempner_dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=source_data/morphology_conductance/bracket-%A_%a.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
seed=$((15300 + SLURM_ARRAY_TASK_ID))
python scripts/morphology_conductance/sgd_bracket.py run --seed "$seed"
