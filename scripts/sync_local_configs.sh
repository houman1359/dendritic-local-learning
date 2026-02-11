#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DRAFT_CONFIG_DIR="${REPO_ROOT}/drafts/dendritic-local-learning/configs"

mkdir -p "${DRAFT_CONFIG_DIR}"
cp "${REPO_ROOT}/configs/config_exp.yaml" "${DRAFT_CONFIG_DIR}/config_exp.yaml"
cp "${REPO_ROOT}/configs/EI_sweeps_example.yaml" "${DRAFT_CONFIG_DIR}/EI_sweeps_example.yaml"
cp "${REPO_ROOT}/configs/config_exp_fsdp.yaml" "${DRAFT_CONFIG_DIR}/config_exp_fsdp.yaml"

echo "Synced local-learning draft config snapshots to ${DRAFT_CONFIG_DIR}"
