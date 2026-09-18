#!/bin/bash
#SBATCH --job-name=credit-gate-fresh
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true
: "${GATE_PILOT_ROOT:?Set an isolated output directory on kempner_project_b}"
seed_offsets=(0 4 8 12 16)
if [[ -z "${SLURM_ARRAY_TASK_ID+x}" ]]; then
    seed_offsets=({0..19})
fi
for offset in "${seed_offsets[@]}"; do
    python -B "$GATE_PILOT_ROOT/scripts/conductance_gate_generalization/followup.py" run \
        --root "$GATE_PILOT_ROOT" --seed "$((2026091800 + ${SLURM_ARRAY_TASK_ID:-0} + offset))"
done
