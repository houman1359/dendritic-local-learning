#!/bin/bash
#SBATCH --job-name=journal_extra_audit
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/additional_figure_controls_20260922/logs/analysis_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/additional_figure_controls_20260922
sha256sum --check frozen.sha256 > logs/final_hash_check.log
/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/release-cleanroom-20260906/bin/python analyze_controls.py
