#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/src/dendritic_modeling/scripts/training/train_experiments.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <existing_sweep_dir> [worker_cpu_count]"
  echo "Example: $0 /n/holylfs06/.../sweep_neurips_claim2_decoder_locality_multidataset_20260211014332 4"
  exit 1
fi

SWEEP_DIR="$1"
WORKER_CPU_COUNT="${2:-4}"

CONFIG_DIR="${SWEEP_DIR}/configs"
RESULTS_DIR="${SWEEP_DIR}/results"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Missing configs dir: ${CONFIG_DIR}"
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Resuming sweep locally"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Worker CPU count cap: ${WORKER_CPU_COUNT}"

count_run=0
count_skip=0
count_fail=0
failed_ids=()

for cfg_path in "${CONFIG_DIR}"/*.yaml; do
  cfg_name="$(basename "${cfg_path}")"
  if [[ "${cfg_name}" == "sweep_metadata.yaml" ]]; then
    continue
  fi

  cfg_id="$(echo "${cfg_name}" | sed -n 's/[^0-9]*\([0-9]\+\)\.yaml/\1/p')"
  if [[ -z "${cfg_id}" ]]; then
    echo "Skipping unrecognized config naming: ${cfg_name}"
    count_skip=$((count_skip + 1))
    continue
  fi

  final_json="${RESULTS_DIR}/config_${cfg_id}/performance/final.json"
  if [[ -f "${final_json}" ]]; then
    echo "[skip] config_${cfg_id} already complete"
    count_skip=$((count_skip + 1))
    continue
  fi

  echo "[run ] config_${cfg_id} (${cfg_name})"
  set +e
  PYTHONPATH="${REPO_ROOT}/src" python - "${TRAIN_SCRIPT}" "${cfg_path}" "${WORKER_CPU_COUNT}" <<'PY'
import multiprocessing
import runpy
import sys

train_script = sys.argv[1]
config_path = sys.argv[2]
worker_cpu_count = int(sys.argv[3])

multiprocessing.cpu_count = lambda: worker_cpu_count
sys.argv = [train_script, config_path]
runpy.run_path(train_script, run_name="__main__")
PY
  status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    count_run=$((count_run + 1))
  else
    echo "[fail] config_${cfg_id} exited with status ${status}"
    count_fail=$((count_fail + 1))
    failed_ids+=("config_${cfg_id}")
  fi
done

echo "Resume complete"
echo "  Ran: ${count_run}"
echo "  Skipped: ${count_skip}"
echo "  Failed: ${count_fail}"
if [[ ${count_fail} -gt 0 ]]; then
  echo "  Failed IDs: ${failed_ids[*]}"
fi
echo "  Sweep dir: ${SWEEP_DIR}"

if [[ ${count_fail} -gt 0 ]]; then
  exit 1
fi
