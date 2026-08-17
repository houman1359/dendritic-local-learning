#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"
SWEEP_DIR="${REPO_ROOT}/drafts/dendritic-local-learning/configs/sweeps"

SUITE="${1:-all}"
SPLIT_JOBS="${2:-0}"
SLURM_ACCOUNT="${3:-kempner_dev}"
SLURM_PARTITION="${4:-kempner_eng}"
MODE="${5:-submit}" # submit | generate-only | dry-run

declare -A CONFIGS
CONFIGS[claim2]="${SWEEP_DIR}/sweep_neurips_claim2_decoder_locality_multidataset_robust.yaml"
CONFIGS[claim3]="${SWEEP_DIR}/sweep_neurips_claim3_shunting_regime_robust.yaml"
CONFIGS[claim4]="${SWEEP_DIR}/sweep_neurips_claim4_source_analysis_robust.yaml"

run_one() {
  local key="$1"
  local config_path="${CONFIGS[${key}]}"
  local extra_args=()

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${key}: ${config_path}"
    exit 1
  fi

  if [[ "${SPLIT_JOBS}" -gt 0 ]]; then
    extra_args+=(--split-jobs "${SPLIT_JOBS}")
  fi

  case "${MODE}" in
    submit)
      ;;
    generate-only)
      extra_args+=(--generate-only)
      ;;
    dry-run)
      extra_args+=(--dry-run)
      ;;
    *)
      echo "Invalid mode: ${MODE}"
      exit 1
      ;;
  esac

  echo "============================================================"
  echo "Submitting robust ${key} via sweep_manager.py"
  echo "Config: ${config_path}"
  echo "Account: ${SLURM_ACCOUNT}"
  echo "Partition: ${SLURM_PARTITION}"
  echo "Mode: ${MODE}"
  echo "============================================================"

  python "${SWEEP_MANAGER}" \
    --config "${config_path}" \
    --slurm-account "${SLURM_ACCOUNT}" \
    --slurm-partition "${SLURM_PARTITION}" \
    "${extra_args[@]}"
}

case "${SUITE}" in
  all)
    ORDER=(claim2 claim3 claim4)
    ;;
  claim2|claim3|claim4)
    ORDER=("${SUITE}")
    ;;
  *)
    echo "Usage: $0 [all|claim2|claim3|claim4] [split_jobs] [account] [partition] [submit|generate-only|dry-run]"
    exit 1
    ;;
esac

for key in "${ORDER[@]}"; do
  run_one "${key}"
done

echo "Completed robust suite: ${SUITE}"
