#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SEED_SWEEP_RUNNER="${SCRIPT_DIR}/run_seed_sweep.sh"

SEEDS="${1:-42,43,44}"
OUTPUT_SUBDIR="${2:-pilot_seed_sweeps}"

STANDARD_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/standard_mnist_pilot_linear_decoder.yaml"
LOCAL_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_pilot_linear_decoder.yaml"

bash "${SEED_SWEEP_RUNNER}" "${STANDARD_CFG}" "standard_mnist_pilot_linear" "${SEEDS}" "${OUTPUT_SUBDIR}"
bash "${SEED_SWEEP_RUNNER}" "${LOCAL_CFG}" "localca_mnist_pilot_linear" "${SEEDS}" "${OUTPUT_SUBDIR}"
