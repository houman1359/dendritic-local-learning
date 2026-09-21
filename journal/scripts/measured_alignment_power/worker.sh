#!/bin/bash
#SBATCH --job-name=alignment-power
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal
if [[ "${SLURM_ARRAY_TASK_ID}" == "20" ]]; then
 python scripts/measured_alignment_power/run.py --calibration-audit
else
 python scripts/measured_alignment_power/run.py --chunk "${SLURM_ARRAY_TASK_ID}"
fi
