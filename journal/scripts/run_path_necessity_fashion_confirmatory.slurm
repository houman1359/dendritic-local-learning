#!/bin/bash
#SBATCH --job-name=path_need_fashion
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%8
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning
#SBATCH --output=/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/path_necessity_fashion/logs/confirmatory_%A_%a.out
#SBATCH --error=/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/path_necessity_fashion/logs/confirmatory_%A_%a.err

set -euo pipefail

PATH_TASK_SEED=$((11000 + SLURM_ARRAY_TASK_ID))
/n/sw/Mambaforge-23.11.0-0/bin/python \
  journal/scripts/run_path_necessity_fashion.py \
  --phase confirmatory \
  --seed "${PATH_TASK_SEED}" \
  --device cuda
