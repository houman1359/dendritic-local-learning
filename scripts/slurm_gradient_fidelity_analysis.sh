#!/bin/bash
#SBATCH --job-name=analysis_gradfid
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_gradfid_%j.out
#SBATCH --error=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_gradfid_%j.err

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
DRAFT_DIR="${REPO_ROOT}/drafts/dendritic-local-learning"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== Gradient Fidelity Analysis ==="
echo "Running: ${DRAFT_DIR}/scripts/run_gradient_fidelity_analysis.sh"
echo ""

bash "${DRAFT_DIR}/scripts/run_gradient_fidelity_analysis.sh"
