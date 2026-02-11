#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"

DEFAULT_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml"
CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
RUN_LABEL="${2:-$(basename "${CONFIG_PATH}" .yaml)}"
bash "${SAFE_RUNNER}" "${CONFIG_PATH}" "${RUN_LABEL}" "smoke_runs"
