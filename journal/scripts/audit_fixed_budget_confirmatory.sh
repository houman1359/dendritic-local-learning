#!/bin/bash
#SBATCH --job-name=journal_fixed_depth_audit
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=drafts/dendritic-local-learning/journal/logs/fixed_depth_audit_%j.out
#SBATCH --error=drafts/dendritic-local-learning/journal/logs/fixed_depth_audit_%j.err

set -euo pipefail

module purge
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p drafts/dendritic-local-learning/journal/logs

python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study fixed_budget --phase confirmatory
