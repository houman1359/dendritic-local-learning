#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_LIST="${1:-${REPO_ROOT}/drafts/dendritic-local-learning/configs/publication_sweeps.txt}"
OUTPUT_TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_DIR="${3:-${REPO_ROOT}/drafts/dendritic-local-learning/analysis/publication_bundle_${OUTPUT_TAG}}"
FIGURE_DIR="${4:-${REPO_ROOT}/drafts/dendritic-local-learning/figures}"

if [[ ! -f "${SWEEP_LIST}" ]]; then
  echo "Sweep list not found: ${SWEEP_LIST}" >&2
  exit 1
fi

mapfile -t SWEEP_DIRS < <(grep -v '^[[:space:]]*$' "${SWEEP_LIST}" | grep -v '^[[:space:]]*#')
if [[ "${#SWEEP_DIRS[@]}" -eq 0 ]]; then
  echo "No sweep directories listed in: ${SWEEP_LIST}" >&2
  exit 1
fi

for sweep_dir in "${SWEEP_DIRS[@]}"; do
  if [[ ! -d "${sweep_dir}" ]]; then
    echo "Missing sweep directory: ${sweep_dir}" >&2
    exit 1
  fi
done

SUMMARY_SCRIPT="${REPO_ROOT}/drafts/dendritic-local-learning/scripts/summarize_neurips_phase_sweeps.py"
FIGURE_SCRIPT="${REPO_ROOT}/drafts/dendritic-local-learning/scripts/generate_neurips_phase_figures.py"

mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${FIGURE_DIR}"

echo "Running phase summary into: ${ANALYSIS_DIR}"
SUMMARY_ARGS=()
for sweep_dir in "${SWEEP_DIRS[@]}"; do
  SUMMARY_ARGS+=(--sweep-dir "${sweep_dir}")
done

PYTHONPATH="${REPO_ROOT}/src" python "${SUMMARY_SCRIPT}" \
  --output-dir "${ANALYSIS_DIR}" \
  "${SUMMARY_ARGS[@]}"

echo "Generating figures into: ${FIGURE_DIR}"
PYTHONPATH="${REPO_ROOT}/src" python "${FIGURE_SCRIPT}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --output-dir "${FIGURE_DIR}"

MANIFEST_JSON="${ANALYSIS_DIR}/publication_bundle_manifest.json"
python - <<PY
import json
from pathlib import Path
sweep_list_path = Path("${SWEEP_LIST}")
sweeps = [
    line.strip()
    for line in sweep_list_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]
manifest = {
    "sweep_list": str(sweep_list_path),
    "analysis_dir": str(Path("${ANALYSIS_DIR}")),
    "figure_dir": str(Path("${FIGURE_DIR}")),
    "sweeps": sweeps,
}
Path("${MANIFEST_JSON}").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote manifest: {Path('${MANIFEST_JSON}')}")
PY

echo "Done."
