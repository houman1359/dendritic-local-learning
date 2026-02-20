#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"
OUTPUT_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <base_config> <run_prefix> <seed_csv> [output_subdir]"
  echo "Example: $0 drafts/dendritic-local-learning/configs/localca_mnist_smoke_linear_decoder.yaml localca_mnist_linear \"42,43,44\" seed_sweeps"
  exit 1
fi

BASE_CONFIG="$1"
RUN_PREFIX="$2"
SEED_CSV="$3"
OUTPUT_SUBDIR="${4:-seed_sweeps}"

IFS=',' read -r -a SEEDS <<< "${SEED_CSV}"
mkdir -p "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"

for seed in "${SEEDS[@]}"; do
  seed_trimmed="$(echo "${seed}" | xargs)"
  if [[ -z "${seed_trimmed}" ]]; then
    continue
  fi

  TMP_CONFIG="$(mktemp /tmp/localca_seed_${seed_trimmed}_XXXX.yaml)"
  python - "${BASE_CONFIG}" "${TMP_CONFIG}" "${seed_trimmed}" <<'PY'
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]
seed = int(sys.argv[3])

cfg = OmegaConf.load(src)
cfg.experiment.seed = seed
OmegaConf.save(cfg, dst)
print(dst)
PY

  RUN_LABEL="${RUN_PREFIX}_seed${seed_trimmed}"
  bash "${SAFE_RUNNER}" "${TMP_CONFIG}" "${RUN_LABEL}" "${OUTPUT_SUBDIR}"

  rm -f "${TMP_CONFIG}"
done
