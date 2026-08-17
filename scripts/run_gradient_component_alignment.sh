#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/gradient_component_alignment.py"

CONFIG_PATH="${1:-${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml}"
RUN_LABEL="${2:-alignment_smoke_$(date +%Y%m%d_%H%M%S)}"
CORE_TYPES="${3:-dendritic_shunting,dendritic_additive}"
RULE_VARIANTS="${4:-3f,4f,5f}"
BROADCAST_MODES="${5:-scalar}"

OUT_DIR="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/diagnostics/${RUN_LABEL}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${REPO_ROOT}/src" python "${PYTHON_SCRIPT}" \
  "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}" \
  --core-types "${CORE_TYPES}" \
  --rule-variants "${RULE_VARIANTS}" \
  --error-broadcast-modes "${BROADCAST_MODES}" \
  --batch-size 256 \
  --seed 42

echo "Gradient alignment outputs: ${OUT_DIR}"

