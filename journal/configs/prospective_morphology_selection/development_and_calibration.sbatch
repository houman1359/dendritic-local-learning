#!/bin/bash
#SBATCH --job-name=tree_select_calib
#SBATCH --partition=shared
#SBATCH --account=kempner_dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-24%12
#SBATCH --output=drafts/dendritic-local-learning/journal/analysis/prospective_morphology_selection/logs/calibration_%A_%a.out
#SBATCH --error=drafts/dendritic-local-learning/journal/analysis/prospective_morphology_selection/logs/calibration_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
runner=drafts/dendritic-local-learning/journal/scripts/run_prospective_morphology_selection.py
if (( SLURM_ARRAY_TASK_ID < 5 )); then
  task_seed=$((6100 + SLURM_ARRAY_TASK_ID))
  python "$runner" calibrate --split development --seed "$task_seed"
  python "$runner" train --split development --seed "$task_seed"
else
  task_seed=$((7200 + SLURM_ARRAY_TASK_ID - 5))
  python "$runner" calibrate --split confirmatory --seed "$task_seed"
fi
