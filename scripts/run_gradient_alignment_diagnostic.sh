#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DIAG_SCRIPT="${SCRIPT_DIR}/gradient_alignment_diagnostic.py"
OUTPUT_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"

DEFAULT_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml"
CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
RUN_LABEL="${2:-$(basename "${CONFIG_PATH}" .yaml)}"

OUT_DIR="${OUTPUT_ROOT}/diagnostics/${RUN_LABEL}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${REPO_ROOT}/src" python "${DIAG_SCRIPT}" \
  "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}"
