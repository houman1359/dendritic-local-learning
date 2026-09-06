#!/bin/bash
#SBATCH --job-name=morph_credit
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal
#SBATCH --output=source_data/morphology_credit/slurm_%A_%a.out
#SBATCH --error=source_data/morphology_credit/slurm_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
study_split=${1:-development}
if [[ "$study_split" == development ]]; then seed=$((126100+SLURM_ARRAY_TASK_ID)); else seed=$((127200+SLURM_ARRAY_TASK_ID)); fi
python scripts/morphology_credit/experiment.py run --split "$study_split" --seed "$seed"
