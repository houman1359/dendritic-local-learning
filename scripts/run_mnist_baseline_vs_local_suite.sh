#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"

STANDARD_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/standard_mnist_smoke_linear_decoder.yaml"
LOCAL_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml"

bash "${SAFE_RUNNER}" "${STANDARD_CFG}" "standard_mnist_linear_smoke" "smoke_runs"
bash "${SAFE_RUNNER}" "${LOCAL_CFG}" "localca_mnist_linear_smoke_repeat" "smoke_runs"
