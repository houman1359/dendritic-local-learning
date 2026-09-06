#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --job-name=boolean_morphology
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
study_split="${1:?split required}"
study_seed_base="${2:?seed base required}"
study_seed=$((study_seed_base + ${SLURM_ARRAY_TASK_ID:-0}))
python scripts/boolean_morphology/experiment.py run --split "$study_split" --seed "$study_seed"
