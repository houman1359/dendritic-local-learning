#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/EI_sweeps_example.yaml"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"

if [[ "$*" == *"--config"* ]]; then
  python "${SWEEP_MANAGER}" "$@"
else
  python "${SWEEP_MANAGER}" --config "${DEFAULT_CONFIG}" "$@"
fi
