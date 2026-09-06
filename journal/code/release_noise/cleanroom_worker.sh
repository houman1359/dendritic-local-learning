#!/bin/bash
#SBATCH --job-name=credit_release_smoke
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
RELEASE_STAGE="$1"
RELEASE_ENV="$2"
RELEASE_REPORTS="$3"
# Setuptools may write build/ and egg-info into its source directory. Build from
# a private copy so validation never changes the checksummed release stage.
RELEASE_BUILD_COPY=$(mktemp -d "$RELEASE_REPORTS/package_build.XXXXXX")
cp -R "$RELEASE_STAGE/dendritic_modeling/." "$RELEASE_BUILD_COPY/"
if [[ "${4:-install}" == install ]]; then
    python -m venv "$RELEASE_ENV"
    "$RELEASE_ENV/bin/python" -m pip install --upgrade pip
    "$RELEASE_ENV/bin/python" -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
    "$RELEASE_ENV/bin/python" -m pip install -c "$RELEASE_STAGE/article_analysis/code/release_noise/constraints.txt" "$RELEASE_BUILD_COPY[test]" PyMuPDF
elif [[ "${4:-}" == reinstall ]]; then
    "$RELEASE_ENV/bin/python" -m pip install --no-deps --force-reinstall "$RELEASE_BUILD_COPY"
fi
"$RELEASE_ENV/bin/python" -m pip check > "$RELEASE_REPORTS/pip_check.txt"
"$RELEASE_ENV/bin/python" -m pip freeze > "$RELEASE_REPORTS/pip_freeze.txt"
cd "$RELEASE_REPORTS"
"$RELEASE_ENV/bin/python" -I -B "$RELEASE_STAGE/article_analysis/code/release_noise/cleanroom_smoke.py" --release-root "$RELEASE_STAGE" > cleanroom_smoke.json
"$RELEASE_ENV/bin/python" -B -m pytest -p no:cacheprovider -q "$RELEASE_STAGE/article_analysis/tests/test_release_noise.py" "$RELEASE_STAGE/article_analysis/tests/test_software_release_repositories.py" > cleanroom_tests.txt
