#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_unified_sweep_local.sh"
SWEEP_DIR="${REPO_ROOT}/drafts/dendritic-local-learning/configs/sweeps"

SUITE="${1:-all}"
MAX_CONFIGS="${2:-0}"
WORKER_CPU_COUNT="${3:-4}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}"
  exit 1
fi

declare -A CONFIGS
CONFIGS[claim1]="${SWEEP_DIR}/sweep_neurips_claim1_mechanism_mnist.yaml"
CONFIGS[claim2]="${SWEEP_DIR}/sweep_neurips_claim2_decoder_locality_multidataset.yaml"
CONFIGS[claim3]="${SWEEP_DIR}/sweep_neurips_claim3_shunting_regime.yaml"
CONFIGS[claim4]="${SWEEP_DIR}/sweep_neurips_claim4_source_analysis.yaml"

run_one() {
  local key="$1"
  local config_path="${CONFIGS[${key}]}"
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${key}: ${config_path}"
    exit 1
  fi

  echo "============================================================"
  echo "Running ${key}"
  echo "Config: ${config_path}"
  echo "============================================================"
  bash "${RUNNER}" "${config_path}" "${MAX_CONFIGS}" "${WORKER_CPU_COUNT}"
}

case "${SUITE}" in
  all)
    ORDER=(claim1 claim2 claim3 claim4)
    ;;
  claim1|claim2|claim3|claim4)
    ORDER=("${SUITE}")
    ;;
  *)
    echo "Usage: $0 [all|claim1|claim2|claim3|claim4] [max_configs] [worker_cpu_count]"
    echo "Example: $0 all 0 4"
    echo "Example: $0 claim1 0 4"
    exit 1
    ;;
esac

for key in "${ORDER[@]}"; do
  run_one "${key}"
done

echo "Finished suite: ${SUITE}"
