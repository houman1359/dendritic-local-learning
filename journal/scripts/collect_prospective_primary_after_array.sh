#!/bin/bash
#SBATCH --job-name=journal_primary_collect
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=drafts/dendritic-local-learning/journal/logs/primary_collect_%j.out
#SBATCH --error=drafts/dendritic-local-learning/journal/logs/primary_collect_%j.err

set -euo pipefail

module purge
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p drafts/dendritic-local-learning/journal/logs

python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study primary --phase confirmatory --require-current-source
python drafts/dendritic-local-learning/journal/scripts/analyze_prospective_learning_results.py
