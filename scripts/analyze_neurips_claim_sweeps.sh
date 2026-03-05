#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ANALYZE_ONE="${SCRIPT_DIR}/analyze_unified_sweep_local.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_neurips_claim_sweeps.py"

SWEEP_ROOT="${1:-/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs}"

latest_dir() {
  local pattern="$1"
  ls -1dt "${SWEEP_ROOT}/${pattern}" 2>/dev/null | head -n 1 || true
}

CLAIM1_DIR="$(latest_dir "sweep_neurips_claim1_mechanism_mnist_*")"
CLAIM2_DIR="$(latest_dir "sweep_neurips_claim2_decoder_locality_multidataset_*")"
CLAIM3_DIR="$(latest_dir "sweep_neurips_claim3_shunting_regime_*")"
CLAIM4_DIR="$(latest_dir "sweep_neurips_claim4_source_analysis_*")"

for dir in "${CLAIM1_DIR}" "${CLAIM2_DIR}" "${CLAIM3_DIR}" "${CLAIM4_DIR}"; do
  if [[ -n "${dir}" && -d "${dir}" ]]; then
    echo "Analyzing ${dir}"
    bash "${ANALYZE_ONE}" "${dir}" local_learning || true
  fi
done

SUMMARY_ARGS=(--sweep-root "${SWEEP_ROOT}")
if [[ -n "${CLAIM1_DIR}" ]]; then SUMMARY_ARGS+=(--sweep-dir "${CLAIM1_DIR}"); fi
if [[ -n "${CLAIM2_DIR}" ]]; then SUMMARY_ARGS+=(--sweep-dir "${CLAIM2_DIR}"); fi
if [[ -n "${CLAIM3_DIR}" ]]; then SUMMARY_ARGS+=(--sweep-dir "${CLAIM3_DIR}"); fi
if [[ -n "${CLAIM4_DIR}" ]]; then SUMMARY_ARGS+=(--sweep-dir "${CLAIM4_DIR}"); fi

python "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}"

echo "NeurIPS claim summary completed."
