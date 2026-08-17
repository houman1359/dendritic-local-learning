#!/bin/bash
#SBATCH --job-name=journal_mechanism_collect
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=drafts/dendritic-local-learning/journal/logs/mechanism_collect_%j.out
#SBATCH --error=drafts/dendritic-local-learning/journal/logs/mechanism_collect_%j.err

set -euo pipefail

module purge
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p drafts/dendritic-local-learning/journal/logs

python drafts/dendritic-local-learning/journal/scripts/collect_prospective_mechanism_diagnostics.py
