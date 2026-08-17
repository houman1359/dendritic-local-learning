#!/bin/bash
#SBATCH --job-name=analysis_lcurves
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_lcurves_%j.out
#SBATCH --error=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_lcurves_%j.err

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
DRAFT_DIR="${REPO_ROOT}/drafts/dendritic-local-learning"
SWEEP_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
OUTPUT_DIR="${DRAFT_DIR}/analysis/learning_curves"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "=== Extract Learning Curves ==="
echo "Sweep root: ${SWEEP_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Process each of the neurips sweeps
for sweep_prefix in "sweep_neurips_fashion_mnist" "sweep_neurips_fa_dfa_baselines" "sweep_neurips_mlp_local_learning"; do
    sweep_dir=$(ls -td "${SWEEP_ROOT}"/${sweep_prefix}_* 2>/dev/null | head -1)
    if [ -z "$sweep_dir" ] || [ ! -d "$sweep_dir" ]; then
        echo "WARNING: No sweep directory found for ${sweep_prefix}, skipping"
        continue
    fi
    echo "Processing: $(basename "$sweep_dir")"
    python "${DRAFT_DIR}/scripts/extract_learning_curves.py" \
        --sweep-root "${SWEEP_ROOT}" \
        --sweep-name "$(basename "$sweep_dir")" \
        --output-dir "${OUTPUT_DIR}" \
        --group-col "core_type" \
        --hue-col "rule_variant" || {
            echo "  WARNING: Failed for ${sweep_prefix}"
        }
done

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
