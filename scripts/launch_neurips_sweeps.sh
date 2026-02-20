#!/bin/bash
# Launch all NeurIPS experiment sweeps
# Usage: bash scripts/launch_neurips_sweeps.sh [--dry-run] [--generate-only]
#
# Sweeps:
#   1. Fashion-MNIST: local rules × shunting/additive × broadcast mode
#   2. FA/DFA baselines: FA, DFA, backprop × DendriNet, MLP
#   3. MLP + local learning: tests if dendritic structure is necessary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$DRAFT_DIR")")"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"
CONFIG_DIR="${DRAFT_DIR}/configs/sweeps"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EXTRA_FLAGS="${@}"

echo "============================================"
echo " NeurIPS Local Credit Assignment Sweeps"
echo "============================================"
echo "Repository: ${REPO_ROOT}"
echo "Sweep manager: ${SWEEP_MANAGER}"
echo "Extra flags: ${EXTRA_FLAGS}"
echo ""

SWEEPS=(
    "sweep_neurips_fashion_mnist.yaml"
    "sweep_neurips_fa_dfa_baselines.yaml"
    "sweep_neurips_mlp_local_learning.yaml"
)

for sweep_yaml in "${SWEEPS[@]}"; do
    config_path="${CONFIG_DIR}/${sweep_yaml}"
    if [ ! -f "$config_path" ]; then
        echo "WARNING: Missing config: ${config_path}, skipping."
        continue
    fi

    echo "--------------------------------------------"
    echo "Launching: ${sweep_yaml}"
    echo "--------------------------------------------"
    python "${SWEEP_MANAGER}" --config "${config_path}" ${EXTRA_FLAGS}
    echo ""
done

echo "============================================"
echo " All sweeps submitted. Check SLURM queue."
echo "============================================"
