#!/bin/bash
#SBATCH --job-name=analysis_stats
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_stats_%j.out
#SBATCH --error=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/logs/analysis_stats_%j.err

set -euo pipefail

REPO_ROOT="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
DRAFT_DIR="${REPO_ROOT}/drafts/dendritic-local-learning"
SWEEP_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
OUTPUT_DIR="${DRAFT_DIR}/analysis/stats"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "=== Statistical Tests ==="
echo "Sweep root: ${SWEEP_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Run statistical tests on all neurips sweeps together
python "${DRAFT_DIR}/scripts/statistical_tests.py" \
    --sweep-root "${SWEEP_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --condition-cols "strategy" "core_type" "rule_variant" \
    --metric "test_accuracy" \
    --filter-sweep "neurips"

echo ""

# Also run per-sweep for detailed breakdowns
for sweep_prefix in "fashion_mnist" "fa_dfa" "mlp_local"; do
    sub_dir="${OUTPUT_DIR}/${sweep_prefix}"
    mkdir -p "${sub_dir}"
    echo "--- Per-sweep stats: ${sweep_prefix} ---"
    python "${DRAFT_DIR}/scripts/statistical_tests.py" \
        --sweep-root "${SWEEP_ROOT}" \
        --output-dir "${sub_dir}" \
        --condition-cols "strategy" "core_type" "rule_variant" \
        --metric "test_accuracy" \
        --filter-sweep "${sweep_prefix}" || {
            echo "  WARNING: Failed for ${sweep_prefix}"
        }
done

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
