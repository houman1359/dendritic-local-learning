#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ROOT_DEFAULT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
SWEEP_ROOT="${SWEEP_ROOT:-${SWEEP_ROOT_DEFAULT}}"
WORKER_CPU_COUNT="${1:-1}"
CLAIM3_DIR_INPUT="${2:-}"

RUN_SUITE="${SCRIPT_DIR}/run_neurips_claim_sweeps_local.sh"
RUN_RESUME="${SCRIPT_DIR}/run_unified_sweep_resume_local.sh"
ANALYZE_SWEEP="${SCRIPT_DIR}/analyze_unified_sweep_local.sh"
SUMMARIZE_CLAIMS="${SCRIPT_DIR}/summarize_neurips_claim_sweeps.py"

log() {
  echo "[$(date '+%F %T')] $*"
}

cmd_running() {
  local needle="$1"
  ps -eo cmd | grep -F -- "${needle}" >/dev/null 2>&1
}

latest_sweep_dir() {
  local pattern="$1"
  ls -1dt "${SWEEP_ROOT}/${pattern}" 2>/dev/null | head -n 1 || true
}

num_configs() {
  local sweep_dir="$1"
  find "${sweep_dir}/configs" -maxdepth 1 -type f -name 'unified_config_*.yaml' | wc -l
}

num_completed() {
  local sweep_dir="$1"
  find "${sweep_dir}/results" -path '*/performance/final.json' | wc -l
}

wait_until_complete() {
  local sweep_dir="$1"
  local label="$2"
  local target
  target="$(num_configs "${sweep_dir}")"
  if [[ "${target}" -eq 0 ]]; then
    log "No configs found for ${label}: ${sweep_dir}"
    return 1
  fi

  while true; do
    local done
    done="$(num_completed "${sweep_dir}")"
    log "${label}: ${done}/${target} completed"
    if [[ "${done}" -ge "${target}" ]]; then
      return 0
    fi
    sleep 90
  done
}

ensure_complete() {
  local sweep_dir="$1"
  local label="$2"
  local target
  local done
  target="$(num_configs "${sweep_dir}")"
  done="$(num_completed "${sweep_dir}")"

  if [[ "${done}" -ge "${target}" ]]; then
    log "${label} already complete (${done}/${target})"
    return 0
  fi

  if cmd_running "run_neurips_claim_sweeps_local.sh ${label}"; then
    log "${label} runner already active; waiting for completion"
    wait_until_complete "${sweep_dir}" "${label}"
    return 0
  fi

  if cmd_running "run_unified_sweep_resume_local.sh ${sweep_dir}"; then
    log "${label} resume runner already active; waiting for completion"
    wait_until_complete "${sweep_dir}" "${label}"
    return 0
  fi

  log "${label} incomplete and no active runner; resuming locally"
  bash "${RUN_RESUME}" "${sweep_dir}" "${WORKER_CPU_COUNT}" || true
  done="$(num_completed "${sweep_dir}")"
  if [[ "${done}" -lt "${target}" ]]; then
    log "WARNING: ${label} still incomplete after resume (${done}/${target})"
    return 1
  fi
  return 0
}

analyze_sweep_safe() {
  local sweep_dir="$1"
  log "Analyzing sweep: ${sweep_dir}"
  bash "${ANALYZE_SWEEP}" "${sweep_dir}" local_learning || true
}

main() {
  local claim1_dir
  local claim2_dir
  local claim3_dir
  local claim4_dir

  claim1_dir="$(latest_sweep_dir 'sweep_neurips_claim1_mechanism_mnist_*')"
  claim2_dir="$(latest_sweep_dir 'sweep_neurips_claim2_decoder_locality_multidataset_*')"
  claim3_dir="${CLAIM3_DIR_INPUT:-$(latest_sweep_dir 'sweep_neurips_claim3_shunting_regime_*')}"

  if [[ -z "${claim3_dir}" ]]; then
    log "No claim3 sweep found; launching claim3"
    bash "${RUN_SUITE}" claim3 0 "${WORKER_CPU_COUNT}"
    claim3_dir="$(latest_sweep_dir 'sweep_neurips_claim3_shunting_regime_*')"
  fi

  if [[ -z "${claim3_dir}" ]]; then
    log "ERROR: claim3 sweep directory is unavailable"
    exit 1
  fi

  ensure_complete "${claim3_dir}" "claim3"
  analyze_sweep_safe "${claim3_dir}"

  claim4_dir="$(latest_sweep_dir 'sweep_neurips_claim4_source_analysis_*')"
  if [[ -z "${claim4_dir}" ]]; then
    log "Launching claim4"
    bash "${RUN_SUITE}" claim4 0 "${WORKER_CPU_COUNT}"
    claim4_dir="$(latest_sweep_dir 'sweep_neurips_claim4_source_analysis_*')"
  fi

  if [[ -n "${claim4_dir}" ]]; then
    ensure_complete "${claim4_dir}" "claim4" || true
    analyze_sweep_safe "${claim4_dir}"
  else
    log "WARNING: claim4 sweep directory was not found after launch"
  fi

  if [[ -n "${claim1_dir}" && -n "${claim2_dir}" && -n "${claim3_dir}" && -n "${claim4_dir}" ]]; then
    log "Writing consolidated claim summary"
    python "${SUMMARIZE_CLAIMS}" \
      --sweep-dir "${claim1_dir}" \
      --sweep-dir "${claim2_dir}" \
      --sweep-dir "${claim3_dir}" \
      --sweep-dir "${claim4_dir}"
  else
    log "WARNING: missing one or more claim sweep directories; skipping full summary"
  fi

  log "Pipeline complete"
}

main "$@"
