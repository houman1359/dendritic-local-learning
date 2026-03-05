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

CONFIG_RULE_BROADCAST_CORE="${SWEEP_DIR}/sweep_neurips_scope_rule_broadcast_core_h100.yaml"
CONFIG_SHUNT_STRAT_INTERACT="${SWEEP_DIR}/sweep_neurips_scope_shunting_strategy_interaction_h100.yaml"
CONFIG_MORPH_INTERACT="${SWEEP_DIR}/sweep_neurips_scope_morphology_interaction_h100.yaml"

resolve_config() {
  local key="$1"
  case "${key}" in
    rule_broadcast_core) echo "${CONFIG_RULE_BROADCAST_CORE}" ;;
    shunting_strategy_interaction) echo "${CONFIG_SHUNT_STRAT_INTERACT}" ;;
    morphology_interaction) echo "${CONFIG_MORPH_INTERACT}" ;;
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
    ORDER=(rule_broadcast_core shunting_strategy_interaction morphology_interaction)
    ;;
  rule_broadcast_core|shunting_strategy_interaction|morphology_interaction)
    ORDER=("${SUITE}")
    ;;
  *)
    echo "Usage: $0 [all|rule_broadcast_core|shunting_strategy_interaction|morphology_interaction] [split_jobs] [account] [partition] [submit|generate-only|dry-run]"
    exit 1
    ;;
esac

for key in "${ORDER[@]}"; do
  run_one "${key}"
done

echo "Completed suite: ${SUITE}"

