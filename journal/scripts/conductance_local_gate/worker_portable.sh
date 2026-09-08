#!/bin/bash
#SBATCH --job-name=gate-portable
#SBATCH --account=kempner_dev
#SBATCH --partition=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd -- "${LOCAL_GATE_JOURNAL_ROOT:?}"
python -B scripts/conductance_local_gate/portable_replay.py --seed 2026090801 --task opposed_strong --rule hard_distal_unit_proximal --rate .03 --steps 4096 --output scripts/conductance_local_gate/portable_validation/hard_distal
python -B scripts/conductance_local_gate/portable_replay.py --seed 2026090801 --task opposed_strong --rule ancestry_two_leaf_oracle_unit_proximal --rate .03 --steps 4096 --output scripts/conductance_local_gate/portable_validation/two_leaf_oracle
