#!/bin/bash
#SBATCH --job-name=journal_followup_gate
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=drafts/dendritic-local-learning/journal/logs/followup_gate_%j.out
#SBATCH --error=drafts/dendritic-local-learning/journal/logs/followup_gate_%j.err

set -euo pipefail

module purge
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p drafts/dendritic-local-learning/journal/logs

python drafts/dendritic-local-learning/journal/scripts/generate_shunting_topology_sweeps.py --check
python drafts/dendritic-local-learning/journal/scripts/generate_fixed_budget_depth_sweeps.py --check
python drafts/dendritic-local-learning/journal/scripts/audit_prospective_source_equivalence.py
python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study followup --phase canary --require-current-source \
  --verified-source-equivalence \
  drafts/dendritic-local-learning/journal/analysis/prospective_source_equivalence.json
python drafts/dendritic-local-learning/journal/scripts/audit_prospective_learning_runs.py \
  --study fixed_budget --phase canary --require-current-source \
  --verified-source-equivalence \
  drafts/dendritic-local-learning/journal/analysis/prospective_source_equivalence.json

first_wave=(
  drafts/dendritic-local-learning/journal/configs/prospective_shunting_topology/confirmatory_routing_*.yaml
  drafts/dendritic-local-learning/journal/configs/prospective_shunting_topology/confirmatory_inhibition_*.yaml
  drafts/dendritic-local-learning/journal/configs/prospective_shunting_topology/confirmatory_spatial_*_local3f.yaml
)

first_ids=()
for config in "${first_wave[@]}"; do
  submission=$(python -m dendritic_modeling.scripts.sweeps.sweep_manager \
    --config "$config" --run 2>&1)
  printf '%s\n' "$submission"
  job_id=$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | tail -1)
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Could not recover Slurm job id for %s\n' "$config" >&2
    exit 1
  fi
  first_ids+=("$job_id")
done

first_dependency=$(IFS=:; printf 'afterany:%s' "${first_ids[*]}")
second_ids=()
for config in drafts/dendritic-local-learning/journal/configs/prospective_shunting_topology/confirmatory_spatial_*_backprop.yaml; do
  generated=$(python -m dendritic_modeling.scripts.sweeps.sweep_manager \
    --config "$config" --generate-only)
  printf '%s\n' "$generated"
  sweep_dir=$(printf '%s\n' "$generated" | sed -n 's/^Created output directory: //p' | tail -1)
  submission=$(sbatch --dependency="$first_dependency" "$sweep_dir/jobs/run_array_sweep.sh")
  printf '%s\n' "$submission"
  job_id=$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | tail -1)
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Could not recover Slurm job id for %s\n' "$config" >&2
    exit 1
  fi
  second_ids+=("$job_id")
done

second_dependency=$(IFS=:; printf 'afterany:%s' "${second_ids[*]}")

fixed_first_wave=(
  drafts/dendritic-local-learning/journal/configs/prospective_fixed_budget_depth/confirmatory_*_local3f.yaml
  drafts/dendritic-local-learning/journal/configs/prospective_fixed_budget_depth/confirmatory_d1_16_*_backprop.yaml
  drafts/dendritic-local-learning/journal/configs/prospective_fixed_budget_depth/confirmatory_d4_2x2x2x2_*_backprop.yaml
)

fixed_first_ids=()
for config in "${fixed_first_wave[@]}"; do
  generated=$(python -m dendritic_modeling.scripts.sweeps.sweep_manager \
    --config "$config" --generate-only)
  printf '%s\n' "$generated"
  sweep_dir=$(printf '%s\n' "$generated" | sed -n 's/^Created output directory: //p' | tail -1)
  submission=$(sbatch --dependency="$second_dependency" "$sweep_dir/jobs/run_array_sweep.sh")
  printf '%s\n' "$submission"
  job_id=$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | tail -1)
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Could not recover Slurm job id for %s\n' "$config" >&2
    exit 1
  fi
  fixed_first_ids+=("$job_id")
done

fixed_first_dependency=$(IFS=:; printf 'afterany:%s' "${fixed_first_ids[*]}")
fixed_second_ids=()
for config in \
  drafts/dendritic-local-learning/journal/configs/prospective_fixed_budget_depth/confirmatory_d2_4x4_*_backprop.yaml \
  drafts/dendritic-local-learning/journal/configs/prospective_fixed_budget_depth/confirmatory_d3_2x2x4_*_backprop.yaml
do
  generated=$(python -m dendritic_modeling.scripts.sweeps.sweep_manager \
    --config "$config" --generate-only)
  printf '%s\n' "$generated"
  sweep_dir=$(printf '%s\n' "$generated" | sed -n 's/^Created output directory: //p' | tail -1)
  submission=$(sbatch --dependency="$fixed_first_dependency" "$sweep_dir/jobs/run_array_sweep.sh")
  printf '%s\n' "$submission"
  job_id=$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | tail -1)
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Could not recover Slurm job id for %s\n' "$config" >&2
    exit 1
  fi
  fixed_second_ids+=("$job_id")
done

fixed_second_dependency=$(IFS=:; printf 'afterany:%s' "${fixed_second_ids[*]}")
sbatch --dependency="$fixed_second_dependency" \
  drafts/dendritic-local-learning/journal/scripts/audit_followup_confirmatory.sh
sbatch --dependency="$fixed_second_dependency" \
  drafts/dendritic-local-learning/journal/scripts/audit_fixed_budget_confirmatory.sh
