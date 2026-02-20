#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_localca_smoke.sh"
OUTPUT_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"

LINEAR_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml"
NONLINEAR_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_smoke_nonlinear_decoder.yaml"

mkdir -p "${OUTPUT_ROOT}/smoke_runs"

bash "${RUN_SCRIPT}" "${LINEAR_CFG}" "localca_linear_decoder_smoke"
bash "${RUN_SCRIPT}" "${NONLINEAR_CFG}" "localca_nonlinear_decoder_smoke"
