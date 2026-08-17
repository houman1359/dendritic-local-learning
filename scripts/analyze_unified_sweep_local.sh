#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ANALYZER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/result_analyzer.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_results_dir> [sweep_type]"
  echo "Example: $0 /n/holylfs06/.../sweep_localca_rule_variant_pilot_20260211001000 local_learning"
  exit 1
fi

RESULTS_DIR="$1"
SWEEP_TYPE="${2:-local_learning}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Sweep results directory not found: ${RESULTS_DIR}"
  exit 1
fi

python "${ANALYZER}" "${RESULTS_DIR}" --type "${SWEEP_TYPE}"

echo "Analyzer output directory: ${RESULTS_DIR}/plots"
