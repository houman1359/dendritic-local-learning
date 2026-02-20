#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ROOT="${1:-/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs}"
POLL_SECONDS="${2:-120}"

ANALYZE_ONE="${SCRIPT_DIR}/analyze_unified_sweep_local.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_neurips_claim_sweeps.py"

latest_dir() {
  local pattern="$1"
  ls -1dt ${SWEEP_ROOT}/${pattern} 2>/dev/null | head -n 1 || true
}

num_configs() {
  local sweep_dir="$1"
  find "${sweep_dir}/configs" -maxdepth 1 -type f -name 'unified_config_*.yaml' | wc -l
}

num_completed() {
  local sweep_dir="$1"
  find "${sweep_dir}/results" -path '*/performance/final.json' | wc -l
}

wait_sweep() {
  local label="$1"
  local sweep_dir="$2"
  local target
  target="$(num_configs "${sweep_dir}")"

  if [[ "${target}" -eq 0 ]]; then
    echo "[${label}] No configs found in ${sweep_dir}"
    return 1
  fi

  while true; do
    local done_count
    done_count="$(num_completed "${sweep_dir}")"
    echo "[${label}] ${done_count}/${target} completed"
    if [[ "${done_count}" -ge "${target}" ]]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

CLAIM1_DIR="$(latest_dir "sweep_neurips_claim1_mechanism_mnist_*")"
CLAIM2_DIR="$(latest_dir "sweep_neurips_claim2_decoder_locality_multidataset_robust_*")"
CLAIM3_DIR="$(latest_dir "sweep_neurips_claim3_shunting_regime_robust_*")"
CLAIM4_DIR="$(latest_dir "sweep_neurips_claim4_source_analysis_robust_*")"

if [[ -z "${CLAIM2_DIR}" || -z "${CLAIM3_DIR}" || -z "${CLAIM4_DIR}" ]]; then
  echo "Missing one or more robust sweep dirs under ${SWEEP_ROOT}"
  echo "claim2: ${CLAIM2_DIR}"
  echo "claim3: ${CLAIM3_DIR}"
  echo "claim4: ${CLAIM4_DIR}"
  exit 1
fi

wait_sweep "claim2-robust" "${CLAIM2_DIR}"
wait_sweep "claim3-robust" "${CLAIM3_DIR}"
wait_sweep "claim4-robust" "${CLAIM4_DIR}"

for dir in "${CLAIM2_DIR}" "${CLAIM3_DIR}" "${CLAIM4_DIR}"; do
  echo "Analyzing ${dir}"
  bash "${ANALYZE_ONE}" "${dir}" local_learning || true
done

SUMMARY_ARGS=(--sweep-root "${SWEEP_ROOT}")
if [[ -n "${CLAIM1_DIR}" ]]; then SUMMARY_ARGS+=(--sweep-dir "${CLAIM1_DIR}"); fi
SUMMARY_ARGS+=(--sweep-dir "${CLAIM2_DIR}")
SUMMARY_ARGS+=(--sweep-dir "${CLAIM3_DIR}")
SUMMARY_ARGS+=(--sweep-dir "${CLAIM4_DIR}")

python "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}"

echo "Robust NeurIPS claim summary completed."
