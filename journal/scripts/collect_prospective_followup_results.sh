#!/bin/bash
# Run after both frozen follow-up confirmatory cohorts have finished.

set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study followup --phase confirmatory --require-current-source
python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study fixed_budget --phase confirmatory --require-current-source
python drafts/dendritic-local-learning/journal/scripts/analyze_prospective_followup_results.py
