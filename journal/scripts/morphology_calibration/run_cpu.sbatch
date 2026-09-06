#!/bin/bash
#SBATCH --job-name=morph_calibration
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:45:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal
#SBATCH --output=source_data/morphology_calibration/logs/%x_%A_%a.out
#SBATCH --error=source_data/morphology_calibration/logs/%x_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python scripts/morphology_calibration/run.py "$1" --index "$SLURM_ARRAY_TASK_ID"
