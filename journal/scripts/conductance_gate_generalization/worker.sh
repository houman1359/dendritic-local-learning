#!/bin/bash
#SBATCH --job-name=credit-gate-pilot
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true
: "${GATE_PILOT_ROOT:?Set an isolated output directory on kempner_project_b}"
python -B "$GATE_PILOT_ROOT/scripts/conductance_gate_generalization/experiment.py" \
    --seed "$((2026091700 + SLURM_ARRAY_TASK_ID))" --steps 4096 \
    --output "$GATE_PILOT_ROOT/results"
