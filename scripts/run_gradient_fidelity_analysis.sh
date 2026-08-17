#!/bin/bash
# Post-hoc gradient fidelity analysis on checkpoint sweep results
# Run this AFTER sweep_neurips_gradient_fidelity_checkpoints completes
#
# Usage: bash scripts/run_gradient_fidelity_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$DRAFT_DIR")")"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

SWEEP_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
OUTPUT_DIR="${DRAFT_DIR}/analysis/gradient_fidelity"
mkdir -p "${OUTPUT_DIR}"

# Find the latest gradient fidelity checkpoint sweep
SWEEP_DIR=$(ls -td "${SWEEP_ROOT}"/sweep_neurips_gradient_fidelity_checkpoints_* 2>/dev/null | head -1)

if [ -z "$SWEEP_DIR" ] || [ ! -d "$SWEEP_DIR" ]; then
    echo "ERROR: No gradient fidelity checkpoint sweep found in ${SWEEP_ROOT}"
    exit 1
fi

echo "Using sweep: ${SWEEP_DIR}"
RESULTS_DIR="${SWEEP_DIR}/results"

# Process each config that has checkpoints
for config_dir in "${RESULTS_DIR}"/config_*; do
    [ -d "$config_dir" ] || continue

    ckpt_dir="${config_dir}/main_network/standard_checkpoints"
    config_yaml="${config_dir}/config.json"

    if [ ! -d "$ckpt_dir" ]; then
        echo "  No checkpoints for $(basename "$config_dir"), skipping"
        continue
    fi

    n_ckpts=$(ls "$ckpt_dir"/epoch_*.pt 2>/dev/null | wc -l)
    if [ "$n_ckpts" -eq 0 ]; then
        echo "  No epoch_*.pt files in $(basename "$config_dir"), skipping"
        continue
    fi

    config_id=$(basename "$config_dir")
    out_subdir="${OUTPUT_DIR}/${config_id}"
    mkdir -p "$out_subdir"

    echo "Processing ${config_id} (${n_ckpts} checkpoints)..."

    # Use the config.json saved with each sweep run
    config_json="${config_dir}/config.json"

    python "${SCRIPT_DIR}/gradient_fidelity_over_training.py" \
        --config-json "${config_json}" \
        --checkpoint-dir "${ckpt_dir}" \
        --output-dir "${out_subdir}" \
        --rule-variants "3f,4f,5f" \
        --broadcast-mode "per_soma" \
        --max-checkpoints 15 \
        --seed 42 || {
            echo "  WARNING: Failed for ${config_id}"
            continue
        }

    echo "  -> Saved to ${out_subdir}"
done

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
