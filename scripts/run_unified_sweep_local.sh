#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_MANAGER="${REPO_ROOT}/src/dendritic_modeling/scripts/sweeps/sweep_manager.py"
TRAIN_SCRIPT="${REPO_ROOT}/src/dendritic_modeling/scripts/training/train_experiments.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_config_path> [max_configs] [worker_cpu_count]"
  echo "Example: $0 drafts/dendritic-local-learning/configs/sweeps/sweep_localca_rule_variant_pilot.yaml 0 4"
  exit 1
fi

SWEEP_CONFIG="$1"
MAX_CONFIGS="${2:-0}"
WORKER_CPU_COUNT="${3:-4}"

if [[ ! -f "${SWEEP_CONFIG}" ]]; then
  echo "Sweep config not found: ${SWEEP_CONFIG}"
  exit 1
fi

SWEEP_BASE="$(python - "${SWEEP_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
print(cfg.get("output_dir", "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"))
PY
)"

RUN_PREFIX="$(python - "${SWEEP_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
base_cfg = cfg.get("base_config", {})
outputs = base_cfg.get("outputs", {}) if base_cfg else {}
print(outputs.get("run_name", "unified_sweep"))
PY
)"

mkdir -p "${SWEEP_BASE}"

echo "Generating sweep configs"
echo "  Sweep config: ${SWEEP_CONFIG}"
echo "  Sweep base dir: ${SWEEP_BASE}"
python "${SWEEP_MANAGER}" --config "${SWEEP_CONFIG}" --generate-only --output-dir "${SWEEP_BASE}"

SWEEP_DIR="$(ls -dt "${SWEEP_BASE}/${RUN_PREFIX}"_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${SWEEP_DIR}" ]]; then
  echo "Unable to locate generated sweep directory under ${SWEEP_BASE} with prefix ${RUN_PREFIX}_"
  exit 1
fi

echo "Running generated configs locally"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Worker CPU count cap: ${WORKER_CPU_COUNT}"

count=0
for cfg_path in "${SWEEP_DIR}"/configs/*.yaml; do
  cfg_name="$(basename "${cfg_path}")"
  if [[ "${cfg_name}" == "sweep_metadata.yaml" ]]; then
    continue
  fi

  if [[ "${MAX_CONFIGS}" -gt 0 && "${count}" -ge "${MAX_CONFIGS}" ]]; then
    break
  fi

  echo "[$((count + 1))] Running ${cfg_name}"
  PYTHONPATH="${REPO_ROOT}/src" python - "${TRAIN_SCRIPT}" "${cfg_path}" "${WORKER_CPU_COUNT}" <<'PY'
import multiprocessing
import runpy
import sys

train_script = sys.argv[1]
config_path = sys.argv[2]
worker_cpu_count = int(sys.argv[3])

# Avoid excessive dataloader workers on shared CPU nodes.
multiprocessing.cpu_count = lambda: worker_cpu_count

sys.argv = [train_script, config_path]
runpy.run_path(train_script, run_name="__main__")
PY

  count=$((count + 1))
done

echo "Completed ${count} configs"
echo "Sweep results directory: ${SWEEP_DIR}"
