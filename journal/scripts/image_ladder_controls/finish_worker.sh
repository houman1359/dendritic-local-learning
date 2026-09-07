#!/bin/bash
#SBATCH --job-name=mnist_complete_analysis
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:${PYTHONPATH:-}
MNIST_SCRIPTS="${SLURM_SUBMIT_DIR}/drafts/dendritic-local-learning/journal/scripts/image_ladder_controls"
python "$MNIST_SCRIPTS/analyze_addendum.py"
python "$MNIST_SCRIPTS/capture.py" --mode summarize
python "$MNIST_SCRIPTS/figure.py"
python "$MNIST_SCRIPTS/export_checkpoints.py"
python "$MNIST_SCRIPTS/prepare_portable.py"
