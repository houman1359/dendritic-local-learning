#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAFE_RUNNER="${SCRIPT_DIR}/run_experiment_safe.sh"
BASE_CONFIG="${REPO_ROOT}/drafts/dendritic-local-learning/configs/localca_mnist_pilot_linear_decoder_lr0p002_compLR.yaml"

SEEDS_CSV="${1:-42}"
FEATURES_CSV="${2:-none,path,depth,norm,branch,full}"
OUTPUT_SUBDIR="${3:-pilot_morphology_feature_sweeps_after_fix}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a FEATURES <<< "${FEATURES_CSV}"

for seed in "${SEEDS[@]}"; do
  seed_trimmed="$(echo "${seed}" | xargs)"
  [[ -z "${seed_trimmed}" ]] && continue

  for feature in "${FEATURES[@]}"; do
    feature_trimmed="$(echo "${feature}" | xargs)"
    [[ -z "${feature_trimmed}" ]] && continue

    tmp_config="$(mktemp /tmp/localca_morph_${feature_trimmed}_${seed_trimmed}_XXXX.yaml)"
    python - "${BASE_CONFIG}" "${tmp_config}" "${seed_trimmed}" "${feature_trimmed}" <<'PY'
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]
seed = int(sys.argv[3])
feature = sys.argv[4].lower()

cfg = OmegaConf.load(src)
cfg.experiment.seed = seed
cfg.training.main.strategy = "local_ca"
cfg.training.main.learning_strategy_config.rule_variant = "5f"

morph = cfg.training.main.learning_strategy_config.morphology_aware
morph.use_path_propagation = False
morph.morphology_modulator_mode = "none"
morph.morphology_depth_offset = 1.0
morph.morphology_centrality_metric = "betweenness"
morph.use_dendritic_normalization = False
morph.use_branch_type_rules = False
morph.apical_branch_scale = 1.0
morph.basal_branch_scale = 1.0
morph.use_branch_length_modulation = False

if feature == "path":
    morph.use_path_propagation = True
elif feature == "depth":
    morph.morphology_modulator_mode = "depth"
elif feature == "norm":
    morph.use_dendritic_normalization = True
elif feature == "branch":
    morph.use_branch_type_rules = True
    morph.apical_branch_scale = 1.2
    morph.basal_branch_scale = 0.8
    morph.use_branch_length_modulation = True
elif feature == "full":
    morph.use_path_propagation = True
    morph.morphology_modulator_mode = "depth"
    morph.use_dendritic_normalization = True
    morph.use_branch_type_rules = True
    morph.apical_branch_scale = 1.2
    morph.basal_branch_scale = 0.8
    morph.use_branch_length_modulation = True
elif feature != "none":
    raise ValueError(f"Unknown morphology feature option: {feature}")

OmegaConf.save(cfg, dst)
print(dst)
PY

    run_label="localca_mnist_pilot_morph${feature_trimmed}_rv5f_compLR_seed${seed_trimmed}"
    bash "${SAFE_RUNNER}" "${tmp_config}" "${run_label}" "${OUTPUT_SUBDIR}"
    rm -f "${tmp_config}"
  done
done
