#!/bin/bash
#SBATCH --job-name=journal_credit_mechanism
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_requeue
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --array=0-159%8
#SBATCH --output=drafts/dendritic-local-learning/journal/logs/prospective_mechanism_%A_%a.out
#SBATCH --error=drafts/dendritic-local-learning/journal/logs/prospective_mechanism_%A_%a.err

set -euo pipefail

: "${MNIST_ADDITIVE_BP_ROOT:?missing MNIST_ADDITIVE_BP_ROOT}"
: "${MNIST_SHUNTING_BP_ROOT:?missing MNIST_SHUNTING_BP_ROOT}"
: "${NOISE_ADDITIVE_BP_ROOT:?missing NOISE_ADDITIVE_BP_ROOT}"
: "${NOISE_SHUNTING_BP_ROOT:?missing NOISE_SHUNTING_BP_ROOT}"

module purge
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

roots=(
  "$MNIST_ADDITIVE_BP_ROOT"
  "$MNIST_SHUNTING_BP_ROOT"
  "$NOISE_ADDITIVE_BP_ROOT"
  "$NOISE_SHUNTING_BP_ROOT"
)

group=$((SLURM_ARRAY_TASK_ID / 40))
config_index=$((SLURM_ARRAY_TASK_ID % 40))
run_dir="${roots[$group]}/results/config_${config_index}"
output_dir="drafts/dendritic-local-learning/journal/prospective_runs/prospective_mechanism_20260802"
output_prefix="${output_dir}/item_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$output_dir" drafts/dendritic-local-learning/journal/logs

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit \
  python -u drafts/dendritic-local-learning/neurips/scripts/measure_feedback_learning_relevance.py \
  --run-dir "$run_dir" \
  --split test \
  --batch-size 128 \
  --relative-step 1e-6 1e-5 1e-4 1e-3 \
  --device cuda \
  --output-prefix "$output_prefix"
