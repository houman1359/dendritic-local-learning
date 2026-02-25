#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"
SWEEP_DIR="${REPO_ROOT}/drafts/dendritic-local-learning/configs/sweeps"

SUITE="${1:-all}"
SPLIT_JOBS="${2:-0}"
SLURM_ACCOUNT="${3:-kempner_dev}"
SLURM_PARTITION="${4:-kempner_h100_priority3}"
MODE="${5:-submit}" # submit | generate-only | dry-run

CONFIG_EI_GRID="${SWEEP_DIR}/sweep_neurips_claimA_ei_grid_pilot.yaml"
CONFIG_EI_GRID_INFO="${SWEEP_DIR}/sweep_neurips_claimA_ei_grid_info_shunting_pilot.yaml"
CONFIG_MORPH_EI="${SWEEP_DIR}/sweep_neurips_claimB_morphology_ei_pilot.yaml"

resolve_config() {
  local key="$1"
  case "${key}" in
    claimA_ei_grid_pilot) echo "${CONFIG_EI_GRID}" ;;
    claimA_ei_grid_info_shunting_pilot) echo "${CONFIG_EI_GRID_INFO}" ;;
    claimB_morphology_ei_pilot) echo "${CONFIG_MORPH_EI}" ;;
    *)
      echo "Unknown suite key: ${key}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local key="$1"
  local config_path
  config_path="$(resolve_config "${key}")"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${key}: ${config_path}"
    exit 1
  fi

  local extra_args=()
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
  echo "Submitting ${key} via sweep_manager.py"
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
    ORDER=(claimA_ei_grid_pilot claimA_ei_grid_info_shunting_pilot claimB_morphology_ei_pilot)
    ;;
  claimA_ei_grid_pilot|claimA_ei_grid_info_shunting_pilot|claimB_morphology_ei_pilot)
    ORDER=("${SUITE}")
    ;;
  *)
    echo "Usage: $0 [all|claimA_ei_grid_pilot|claimA_ei_grid_info_shunting_pilot|claimB_morphology_ei_pilot] [split_jobs] [account] [partition] [submit|generate-only|dry-run]"
    exit 1
    ;;
esac

for key in "${ORDER[@]}"; do
  run_one "${key}"
done

echo "Completed suite: ${SUITE}"
