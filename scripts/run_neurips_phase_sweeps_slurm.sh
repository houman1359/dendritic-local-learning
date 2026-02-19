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

# Explicit config map
CONFIG_PHASE1_CAPACITY="${SWEEP_DIR}/sweep_neurips_phase1_capacity_calibration.yaml"
CONFIG_PHASE1_CIFAR="${SWEEP_DIR}/sweep_neurips_phase1_cifar_sanity.yaml"
CONFIG_PHASE2_SIGNAL="${SWEEP_DIR}/sweep_neurips_phase2_local_competence_signal.yaml"
CONFIG_PHASE2_MORPH="${SWEEP_DIR}/sweep_neurips_phase2_local_competence_morphology.yaml"
CONFIG_PHASE2_3F="${SWEEP_DIR}/sweep_neurips_phase2_local_competence_three_factor.yaml"
CONFIG_PHASE2_HSIC="${SWEEP_DIR}/sweep_neurips_phase2_local_competence_hsic.yaml"
CONFIG_PHASE2B_GAP_PILOT="${SWEEP_DIR}/sweep_neurips_phase2b_gap_closing_pilot.yaml"
CONFIG_CLAIM_A="${SWEEP_DIR}/sweep_neurips_phase3_claimA_shunting_regime_strong.yaml"
CONFIG_CLAIM_B="${SWEEP_DIR}/sweep_neurips_phase3_claimB_morphology_scaling.yaml"
CONFIG_CLAIM_C="${SWEEP_DIR}/sweep_neurips_phase3_claimC_error_shaping.yaml"
CONFIG_INFO_PANEL="${SWEEP_DIR}/sweep_neurips_phase3_information_panel.yaml"

resolve_config() {
  local key="$1"
  case "${key}" in
    phase1_capacity) echo "${CONFIG_PHASE1_CAPACITY}" ;;
    phase1_cifar) echo "${CONFIG_PHASE1_CIFAR}" ;;
    phase2_signal) echo "${CONFIG_PHASE2_SIGNAL}" ;;
    phase2_morphology) echo "${CONFIG_PHASE2_MORPH}" ;;
    phase2_three_factor) echo "${CONFIG_PHASE2_3F}" ;;
    phase2_hsic) echo "${CONFIG_PHASE2_HSIC}" ;;
    phase2b_pilot) echo "${CONFIG_PHASE2B_GAP_PILOT}" ;;
    claimA) echo "${CONFIG_CLAIM_A}" ;;
    claimB) echo "${CONFIG_CLAIM_B}" ;;
    claimC) echo "${CONFIG_CLAIM_C}" ;;
    info_panel) echo "${CONFIG_INFO_PANEL}" ;;
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
    ORDER=(
      phase1_capacity
      phase1_cifar
      phase2_signal
      phase2_morphology
      phase2_three_factor
      phase2_hsic
      claimA
      claimB
      claimC
      info_panel
    )
    ;;
  phase1)
    ORDER=(phase1_capacity phase1_cifar)
    ;;
  phase2)
    ORDER=(phase2_signal phase2_morphology phase2_three_factor phase2_hsic)
    ;;
  phase2b)
    ORDER=(phase2b_pilot)
    ;;
  phase3)
    ORDER=(claimA claimB claimC info_panel)
    ;;
  phase1_capacity|phase1_cifar|phase2_signal|phase2_morphology|phase2_three_factor|phase2_hsic|phase2b_pilot|claimA|claimB|claimC|info_panel)
    ORDER=("${SUITE}")
    ;;
  *)
    echo "Usage: $0 [all|phase1|phase2|phase2b|phase3|phase1_capacity|phase1_cifar|phase2_signal|phase2_morphology|phase2_three_factor|phase2_hsic|phase2b_pilot|claimA|claimB|claimC|info_panel] [split_jobs] [account] [partition] [submit|generate-only|dry-run]"
    exit 1
    ;;
esac

for key in "${ORDER[@]}"; do
  run_one "${key}"
done

echo "Completed suite: ${SUITE}"
