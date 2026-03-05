#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"
CONFIG_PATH="${REPO_ROOT}/drafts/dendritic-local-learning/configs/sweeps/sweep_neurips_additive_dynamics_fairness_audit.yaml"

SPLIT_JOBS="${1:-0}"
SLURM_ACCOUNT="${2:-kempner_dev}"
SLURM_PARTITION="${3:-kempner_h100_priority3}"
MODE="${4:-submit}" # submit | generate-only | dry-run

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${SPLIT_JOBS}" -gt 0 ]]; then
  EXTRA_ARGS+=(--split-jobs "${SPLIT_JOBS}")
fi

case "${MODE}" in
  submit)
    ;;
  generate-only)
    EXTRA_ARGS+=(--generate-only)
    ;;
  dry-run)
    EXTRA_ARGS+=(--dry-run)
    ;;
  *)
    echo "Invalid mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "Submitting additive fairness audit via sweep_manager.py"
echo "Config: ${CONFIG_PATH}"
echo "Account: ${SLURM_ACCOUNT}"
echo "Partition: ${SLURM_PARTITION}"
echo "Mode: ${MODE}"
echo "============================================================"

python "${SWEEP_MANAGER}" \
  --config "${CONFIG_PATH}" \
  --slurm-account "${SLURM_ACCOUNT}" \
  --slurm-partition "${SLURM_PARTITION}" \
  "${EXTRA_ARGS[@]}"
