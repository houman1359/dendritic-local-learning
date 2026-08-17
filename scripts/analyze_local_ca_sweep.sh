#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <results_dir> [extra result_analyzer args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ANALYZER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/result_analyzer.py"

RESULTS_DIR="$1"
shift

python "${ANALYZER}" "${RESULTS_DIR}" --type local_learning "$@"
