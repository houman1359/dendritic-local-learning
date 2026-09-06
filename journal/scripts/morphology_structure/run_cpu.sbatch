#!/bin/bash
#SBATCH --job-name=morph_structure
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-13%7
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal
#SBATCH --output=analysis/morphology_investigation_20260905/structure/slurm_%A_%a.out
#SBATCH --error=analysis/morphology_investigation_20260905/structure/slurm_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/morphology_structure/run.py run --shard "$SLURM_ARRAY_TASK_ID" --nshards 14
