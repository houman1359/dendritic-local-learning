#!/bin/bash
#SBATCH --job-name=morph_construct
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal
#SBATCH --output=analysis/morphology_investigation_20260905/structure/constructive/slurm_%j.out
#SBATCH --error=analysis/morphology_investigation_20260905/structure/constructive/slurm_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/morphology_structure/constructive.py run
