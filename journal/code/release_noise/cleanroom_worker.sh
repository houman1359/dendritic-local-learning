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
# Resolve all caller-relative paths before changing directories, including paths
# for a new environment or report directory that does not exist yet.
resolve_release_path() {
    python -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}
RELEASE_STAGE=$(resolve_release_path "${1:?Usage: cleanroom_worker.sh SOFTWARE ENV REPORTS [install|reinstall|reuse]}")
RELEASE_ENV=$(resolve_release_path "${2:?Missing virtual-environment path}")
RELEASE_REPORTS=$(resolve_release_path "${3:?Missing report-directory path}")
mkdir -p "$RELEASE_REPORTS"
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
cd "$RELEASE_STAGE/article_analysis"
"$RELEASE_ENV/bin/python" -I -B "$RELEASE_STAGE/article_analysis/code/release_noise/cleanroom_smoke.py" --release-root "$RELEASE_STAGE" > "$RELEASE_REPORTS/cleanroom_smoke.json"
# Keep pytest from discovering configuration/conftest files in the caller's
# implementation checkout or inaccessible shared-filesystem ancestors.
"$RELEASE_ENV/bin/python" -B -m pytest -c /dev/null --rootdir . --confcutdir . -p no:cacheprovider -q tests/test_release_noise.py tests/test_software_release_repositories.py > "$RELEASE_REPORTS/cleanroom_tests.txt"
