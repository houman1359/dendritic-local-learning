#!/bin/bash
#SBATCH --job-name=tree_select_step1
#SBATCH --partition=shared
#SBATCH --account=kempner_dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%12
#SBATCH --output=drafts/dendritic-local-learning/journal/analysis/prospective_morphology_selection/logs/step1_%A_%a.out
#SBATCH --error=drafts/dendritic-local-learning/journal/analysis/prospective_morphology_selection/logs/step1_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_seed=$((7200 + SLURM_ARRAY_TASK_ID))
python drafts/dendritic-local-learning/journal/scripts/audit_prospective_morphology_selection.py --seed "$task_seed"
