#!/bin/bash
#SBATCH --job-name=credit_rule_extension
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
EXTENSION_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
# Slurm copies the script; use an explicit frozen journal path passed at submission.
EXTENSION_JOURNAL="$1"
EXTENSION_PROTOCOL_HASH="$2"
EXTENSION_SEED=$((211200 + SLURM_ARRAY_TASK_ID))
python -B "$EXTENSION_JOURNAL/scripts/credit_rule_extension/run.py" run --seed "$EXTENSION_SEED" --protocol-sha256 "$EXTENSION_PROTOCOL_HASH"
