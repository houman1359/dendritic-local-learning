#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_ROOT="${1:-/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs}"
OUTPUT_TAG="${2:-fair_$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_DIR="${3:-/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/publication_bundle_${OUTPUT_TAG}}"
FIGURE_DIR="${4:-${REPO_ROOT}/drafts/dendritic-local-learning/figures}"
MANIFEST_PATH="${5:-${REPO_ROOT}/drafts/dendritic-local-learning/configs/publication_sweeps_${OUTPUT_TAG}.txt}"

PREFIXES=(
  "sweep_neurips_phase1_capacity_calibration"
  "sweep_neurips_phase1_cifar_sanity"
  "sweep_neurips_phase2b_gap_closing_pilot"
  "sweep_neurips_claimA_ei_grid_pilot"
  "sweep_neurips_claimA_ei_grid_info_shunting_pilot"
  "sweep_neurips_claimB_morphology_ei_pilot"
  "sweep_neurips_phase3_claimC_error_shaping"
  "sweep_neurips_phase3_information_panel"
  "sweep_neurips_depth_scaling"
  "sweep_neurips_noise_robustness"
  "sweep_neurips_additive_dynamics_fairness_audit"
  "sweep_neurips_localca_core_fair_tuning"
)

mkdir -p "$(dirname "${MANIFEST_PATH}")"
mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${FIGURE_DIR}"

{
  for prefix in "${PREFIXES[@]}"; do
    latest="$(ls -td "${SWEEP_ROOT}/${prefix}"_* 2>/dev/null | head -n1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "${latest}"
    fi
  done
} > "${MANIFEST_PATH}"

if [[ ! -s "${MANIFEST_PATH}" ]]; then
  echo "No sweep directories found in ${SWEEP_ROOT}" >&2
  exit 1
fi

echo "Using manifest: ${MANIFEST_PATH}"
cat "${MANIFEST_PATH}"
echo ""

"${REPO_ROOT}/drafts/dendritic-local-learning/scripts/regenerate_publication_bundle.sh" \
  "${MANIFEST_PATH}" \
  "${OUTPUT_TAG}" \
  "${ANALYSIS_DIR}" \
  "${FIGURE_DIR}"

echo ""
echo "Fair publication bundle complete."
echo "Analysis: ${ANALYSIS_DIR}"
echo "Figures:  ${FIGURE_DIR}"
