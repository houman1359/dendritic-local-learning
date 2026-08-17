#!/bin/bash
#SBATCH --job-name=local_ca_sweep
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-0%1
#SBATCH --output=results/local_ca_sweep/jobs/%A_%a/out.log
#SBATCH --error=results/local_ca_sweep/jobs/%A_%a/err.log

set -euo pipefail

REPO_ROOT="$(pwd)"
CONFIG_DIR="${REPO_ROOT}/drafts/dendritic-local-learning/configs/generated"
OUTPUT_DIR="${REPO_ROOT}/results/local_ca_sweep"

mapfile -t CONFIG_FILES < <(ls "${CONFIG_DIR}"/unified_config_*.yaml | sort)
CONFIG="${CONFIG_FILES[${SLURM_ARRAY_TASK_ID}]}"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/results/config_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${RUN_OUTPUT_DIR}"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit \
  python -u "${REPO_ROOT}/src/dendritic_modeling/scripts/training/train_experiments.py" \
  "${CONFIG}" --output_dir "${RUN_OUTPUT_DIR}"
