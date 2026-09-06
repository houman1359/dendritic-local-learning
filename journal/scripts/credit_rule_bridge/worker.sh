#!/bin/bash
#SBATCH --job-name=credit_rule_bridge
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BRIDGE_SCRIPT_DIR="${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/credit_rule_bridge"
if [[ "${1:-}" == benchmark ]]; then
    python "$BRIDGE_SCRIPT_DIR/run.py" benchmark --model algebraic --seed 210100
    python "$BRIDGE_SCRIPT_DIR/run.py" benchmark --model conductance --seed 210100
else
    BRIDGE_PHASE="$1"
    BRIDGE_INDEX="${SLURM_ARRAY_TASK_ID}"
    if [[ "$BRIDGE_PHASE" == development ]]; then
        BRIDGE_SEED=$((210100 + BRIDGE_INDEX / 2))
    else
        BRIDGE_SEED=$((211200 + BRIDGE_INDEX / 2))
    fi
    if (( BRIDGE_INDEX % 2 == 0 )); then BRIDGE_MODEL=algebraic; else BRIDGE_MODEL=conductance; fi
    python "$BRIDGE_SCRIPT_DIR/run.py" run --phase "$BRIDGE_PHASE" --model "$BRIDGE_MODEL" --seed "$BRIDGE_SEED"
fi
