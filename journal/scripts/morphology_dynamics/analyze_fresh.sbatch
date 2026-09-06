#!/bin/bash
#SBATCH --job-name=morph-dynamics-summary
#SBATCH --partition=serial_requeue
#SBATCH --account=kempner_dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/morphology_investigation_20260905/dynamics/analysis-%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python scripts/morphology_dynamics/investigate.py analyze --cohort fresh
python scripts/morphology_dynamics/supplemental_diagnostics.py analyze --cohort fresh
python scripts/morphology_dynamics/render_and_audit.py
