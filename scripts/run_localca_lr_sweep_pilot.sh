#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"
BASE_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_pilot_linear_decoder.yaml"

SEED="${1:-42}"
LR_CSV="${2:-0.0005,0.001,0.002}"
OUTPUT_SUBDIR="${3:-pilot_lr_sweeps}"

IFS=',' read -r -a LRS <<< "${LR_CSV}"

for lr in "${LRS[@]}"; do
  lr_trimmed="$(echo "${lr}" | xargs)"
  if [[ -z "${lr_trimmed}" ]]; then
    continue
  fi

  tmp_config="$(mktemp /tmp/localca_pilot_lr_${SEED}_XXXX.yaml)"
  python - "${BASE_CONFIG}" "${tmp_config}" "${SEED}" "${lr_trimmed}" <<'PY'
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]
seed = int(sys.argv[3])
lr = float(sys.argv[4])

cfg = OmegaConf.load(src)
cfg.experiment.seed = seed
cfg.training.main.common.param_groups.lr = lr
cfg.training.main.common.param_groups.topk_lr = lr
cfg.training.main.common.param_groups.decoder_lr = lr
OmegaConf.save(cfg, dst)
print(dst)
PY

  lr_tag="${lr_trimmed//./p}"
  run_label="localca_mnist_pilot_linear_lr${lr_tag}_seed${SEED}"
  bash "${SAFE_RUNNER}" "${tmp_config}" "${run_label}" "${OUTPUT_SUBDIR}"
  rm -f "${tmp_config}"
done
