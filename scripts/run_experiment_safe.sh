#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/src/dendritic_modeling/scripts/training/train_experiments.py"
OUTPUT_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
WORKER_CPU_COUNT="${WORKER_CPU_COUNT:-4}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_path> <run_label> [output_subdir]"
  exit 1
fi

CONFIG_PATH="$1"
RUN_LABEL="$2"
OUTPUT_SUBDIR="${3:-smoke_runs}"
TARGET_OUTPUT="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"

mkdir -p "${TARGET_OUTPUT}"

echo "Running experiment safely"
echo "  Config: ${CONFIG_PATH}"
echo "  Output root: ${TARGET_OUTPUT}"
echo "  Run label: ${RUN_LABEL}"
echo "  Worker CPU count cap: ${WORKER_CPU_COUNT}"

PYTHONPATH="${REPO_ROOT}/src" python - "${TRAIN_SCRIPT}" "${CONFIG_PATH}" "${TARGET_OUTPUT}" "${RUN_LABEL}" "${WORKER_CPU_COUNT}" <<'PY'
import multiprocessing
import runpy
import sys

train_script = sys.argv[1]
config_path = sys.argv[2]
output_dir = sys.argv[3]
run_name = sys.argv[4]
worker_cpu_count = int(sys.argv[5])

# Avoid excessive dataloader workers on shared CPU nodes.
multiprocessing.cpu_count = lambda: worker_cpu_count

sys.argv = [
    train_script,
    config_path,
    "--output_dir",
    output_dir,
    "--run_name",
    run_name,
]
runpy.run_path(train_script, run_name="__main__")
PY
