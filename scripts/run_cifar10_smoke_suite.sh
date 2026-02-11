#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"
DIAG_RUNNER="${SCRIPT_DIR}/run_gradient_alignment_diagnostic.sh"

LINEAR_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_cifar10_smoke_linear_decoder.yaml"
NONLINEAR_CFG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_cifar10_smoke_nonlinear_decoder.yaml"

# Always capture diagnostics first.
bash "${DIAG_RUNNER}" "${LINEAR_CFG}" "cifar10_linear_decoder_smoke"
bash "${DIAG_RUNNER}" "${NONLINEAR_CFG}" "cifar10_nonlinear_decoder_smoke"

# Then run training smoke jobs.
bash "${SAFE_RUNNER}" "${LINEAR_CFG}" "localca_cifar10_linear_smoke" "smoke_runs"
bash "${SAFE_RUNNER}" "${NONLINEAR_CFG}" "localca_cifar10_nonlinear_smoke" "smoke_runs"
