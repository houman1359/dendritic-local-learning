#!/bin/bash
#SBATCH --job-name=journal_extra_controls
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --qos=normal
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/additional_figure_controls_20260922/logs/%A_%a.log
set -euo pipefail
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd /n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/additional_figure_controls_20260922
if [[ "${1:-}" == preflight ]]; then
    /n/sw/Mambaforge-23.11.0-0/bin/python run_controls.py --preflight
else
    /n/sw/Mambaforge-23.11.0-0/bin/python run_controls.py --index "$SLURM_ARRAY_TASK_ID"
fi
