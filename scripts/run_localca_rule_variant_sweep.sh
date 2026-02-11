#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"
BASE_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_pilot_linear_decoder_lr0p002_compLR.yaml"

SEEDS_CSV="${1:-42}"
VARIANTS_CSV="${2:-3f,4f,5f}"
OUTPUT_SUBDIR="${3:-pilot_rule_variant_sweeps_after_fix}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a VARIANTS <<< "${VARIANTS_CSV}"

for seed in "${SEEDS[@]}"; do
  seed_trimmed="$(echo "${seed}" | xargs)"
  [[ -z "${seed_trimmed}" ]] && continue

  for variant in "${VARIANTS[@]}"; do
    variant_trimmed="$(echo "${variant}" | xargs)"
    [[ -z "${variant_trimmed}" ]] && continue

    tmp_config="$(mktemp /tmp/localca_rv_${variant_trimmed}_${seed_trimmed}_XXXX.yaml)"
    python - "${BASE_CONFIG}" "${tmp_config}" "${seed_trimmed}" "${variant_trimmed}" <<'PY'
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]
seed = int(sys.argv[3])
variant = sys.argv[4]

cfg = OmegaConf.load(src)
cfg.experiment.seed = seed
cfg.training.main.strategy = "local_ca"
cfg.training.main.learning_strategy_config.rule_variant = variant
OmegaConf.save(cfg, dst)
print(dst)
PY

    run_label="localca_mnist_pilot_rv${variant_trimmed}_compLR_seed${seed_trimmed}"
    bash "${SAFE_RUNNER}" "${tmp_config}" "${run_label}" "${OUTPUT_SUBDIR}"
    rm -f "${tmp_config}"
  done
done
