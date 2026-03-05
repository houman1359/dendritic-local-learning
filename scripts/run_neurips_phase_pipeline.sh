#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ROOT="${1:-/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs}"
POLL_SECONDS="${2:-120}"
SUITE="${3:-all}"

ANALYZER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/result_analyzer.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_neurips_phase_sweeps.py"
FIGURE_SCRIPT="${SCRIPT_DIR}/generate_neurips_phase_figures.py"

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

resolve_patterns() {
  case "${SUITE}" in
    all)
      PATTERNS=(
        "sweep_neurips_phase1_capacity_calibration_*"
        "sweep_neurips_phase1_cifar_sanity_*"
        "sweep_neurips_phase2_local_competence_signal_*"
        "sweep_neurips_phase2_local_competence_morphology_*"
        "sweep_neurips_phase2_local_competence_three_factor_*"
        "sweep_neurips_phase2_local_competence_hsic_*"
        "sweep_neurips_phase3_claimA_shunting_regime_strong_*"
        "sweep_neurips_phase3_claimB_morphology_scaling_*"
        "sweep_neurips_phase3_claimC_error_shaping_*"
        "sweep_neurips_phase3_information_panel_*"
      )
      ;;
    phase1)
      PATTERNS=(
        "sweep_neurips_phase1_capacity_calibration_*"
        "sweep_neurips_phase1_cifar_sanity_*"
      )
      ;;
    phase2)
      PATTERNS=(
        "sweep_neurips_phase2_local_competence_signal_*"
        "sweep_neurips_phase2_local_competence_morphology_*"
        "sweep_neurips_phase2_local_competence_three_factor_*"
        "sweep_neurips_phase2_local_competence_hsic_*"
      )
      ;;
    phase3)
      PATTERNS=(
        "sweep_neurips_phase3_claimA_shunting_regime_strong_*"
        "sweep_neurips_phase3_claimB_morphology_scaling_*"
        "sweep_neurips_phase3_claimC_error_shaping_*"
        "sweep_neurips_phase3_information_panel_*"
      )
      ;;
    *)
      PATTERNS=("${SUITE}")
      ;;
  esac
}

resolve_patterns

FOUND_DIRS=()
for pattern in "${PATTERNS[@]}"; do
  sweep_dir="$(latest_dir "${pattern}")"
  if [[ -z "${sweep_dir}" || ! -d "${sweep_dir}" ]]; then
    echo "No sweep directory found for pattern: ${pattern}"
    continue
  fi
  FOUND_DIRS+=("${sweep_dir}")
done

if [[ "${#FOUND_DIRS[@]}" -eq 0 ]]; then
  echo "No matching sweep directories found under ${SWEEP_ROOT}"
  exit 1
fi

for dir in "${FOUND_DIRS[@]}"; do
  label="$(basename "${dir}")"
  wait_sweep "${label}" "${dir}"
  echo "Analyzing ${dir}"
  python "${ANALYZER}" "${dir}" || true

done

SUMMARY_ARGS=(--sweep-root "${SWEEP_ROOT}")
for dir in "${FOUND_DIRS[@]}"; do
  SUMMARY_ARGS+=(--sweep-dir "${dir}")
done

python "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}"
python "${FIGURE_SCRIPT}"

echo "Phase pipeline complete for suite: ${SUITE}"
