#!/usr/bin/env python3
"""Build and validate the reviewer software archive.

The release has two deliberately distinct source trees:

* ``dendritic_modeling/`` is exported from the repository's committed HEAD
  through the same lightweight release filter used for article code.
  Uncommitted working-tree changes never enter this snapshot.
* ``article_analysis/`` contains an explicit allow-list from this journal
  package.  It includes analysis code and frozen configurations, but no raw
  data, model checkpoints, scheduler scripts, logs, or caches.

The build is deterministic for a fixed Git HEAD and fixed journal inputs.
Machine-specific paths in historical, committed auxiliary configurations are
replaced with documented placeholders in the release copy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = JOURNAL_ROOT.parent
SUBMISSION_ROOT = JOURNAL_ROOT / "submission"
STAGE_NAME = "software_release"
ARCHIVE_NAME = "Dendritic_credit_assignment_software.zip"

# The full committed repository is retained.  These replacements affect only
# historical machine-local defaults; each changed file is listed in
# PORTABILITY_PATCHES.tsv.  Tokens are intentionally conspicuous so that a
# reviewer cannot mistake them for working paths.
PORTABILITY_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling",
        "${DENDRITIC_MODELING_ROOT}",
        "replace archived figure-generator repository root",
    ),
    (
        "/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/"
        "vdc_paper_v2/bin/python",
        "python",
        "replace historical environment interpreter with PATH lookup",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling",
        "${DENDRITIC_MODELING_ROOT}",
        "replace historical repository root",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_revision_2026",
        "${DENDRITIC_EXTERNAL_ARTIFACTS}",
        "replace external-artifact root not distributed with the article",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_rnn_runs_protected",
        "${DENDRITIC_RUNS_ROOT}",
        "replace historical run root not distributed with the article",
    ),
)

JOURNAL_DIRECTORIES = ("code", "configs", "tests", "reproducibility")
JOURNAL_ANALYSIS_RECORDS = (
    "ANIMAL_DATA_CONTRACT.md",
    "CREDIT_PHASE_THEORY_EXPERIMENT_CONTRACT_20260811.md",
    "EXPERIMENT_CONTRACT.md",
    "EVIDENCE_LEDGER.md",
    "MICRONS_FUNCTIONAL_INHIBITORY_CONTRACT.md",
    "NEURIPS_FIGURE_LINEAGE_AUDIT_20260804.md",
    "NONLINEAR_PHYSICAL_DEPTH_CODE_THEORY_AUDIT_20260812.md",
    "NONLINEAR_PHYSICAL_DEPTH_CONFIRMATORY_CONTRACT_20260812.md",
    "POSITIVE_CONDUCTANCE_STEP_CONSISTENT_CONTRACT_20260811.md",
    "REVIEW_IMPLEMENTATION_MATRIX_20260811.md",
    "SAME_SPAN_COEFFICIENT_LEARNING_CONTRACT_20260811.md",
    "TRAINED_SUBTREE_ADDRESS_EXPERIMENT_CONTRACT.md",
)
JOURNAL_SCRIPTS = (
    "aggregate_all_scan_functional_topology.py",
    "analyze_bandwidth_matched_routing.py",
    "analyze_branch_credit_interference.py",
    "analyze_credit_phase_existing.py",
    "analyze_credit_phase_spectral_bound.py",
    "analyze_focal_gradient_decomposition.py",
    "analyze_focal_gradient_shapley.py",
    "analyze_francioni_signed_credit.py",
    "analyze_microns_inhibitory_routes.py",
    "analyze_nonlinear_physical_depth_confirmatory.py",
    "analyze_remaining_physical_experiments.py",
    "analyze_physical_cable_sensitivity.py",
    "analyze_prospective_followup_results.py",
    "analyze_prospective_learning_results.py",
    "analyze_positive_conductance_reliability.py",
    "analyze_reciprocal_routing_controls.py",
    "analyze_spatial_topology_audit.py",
    "analyze_same_span_coefficient_learning.py",
    "audit_prospective_learning_runs.py",
    "audit_figure_style_lineage.py",
    "audit_nature_communications_format.py",
    "audit_submission.py",
    "build_journal_figures.py",
    "build_new_confirmatory_figures.py",
    "build_prospective_input_validity_audit.py",
    "build_alignment_animal_figure.py",
    "build_credit_phase_figure.py",
    "build_microns_v661_replication_figure.py",
    "build_nature_source_data.py",
    "build_positive_conductance_reliability_figure.py",
    "build_same_span_coefficient_figure.py",
    "build_focal_selectivity_figure.py",
    "build_regular_tree_regime_figure.py",
    "build_trained_subtree_address_figure.py",
    "build_static_microns_replication.py",
    "collect_feedback_gradient_rerun.py",
    "collect_feedback_rerun.py",
    "collect_clean_exact_bp_rerun.py",
    "export_figure2d_seed_data.py",
    "export_regular_tree_source_data.py",
    "fetch_expanded_microns_cohort.py",
    "figure1_neurips_components.py",
    "diagnose_nonlinear_physical_depth.py",
    "generate_nonlinear_physical_depth_confirmatory.py",
    "generate_nonlinear_physical_depth_sweeps.py",
    "generate_remaining_physical_experiments.py",
    "journal_style.py",
    "neurips_style.py",
    "run_alignment_controlled_learning.py",
    "render_nonlinear_physical_depth_calibration.py",
    "run_credit_phase_theory_experiment.py",
    "run_focal_selectivity_active_ensemble.py",
    "run_focal_selectivity_phase1.py",
    "run_positive_conductance_reliability.py",
    "run_reconstructed_tree_task_learning.py",
    "run_same_span_coefficient_learning.py",
    "run_trained_subtree_address_full_factorial.py",
    "run_trained_subtree_address_phase1.py",
    "summarize_static_microns_replication.py",
    "update_provenance_hashes.py",
)
ARCHIVED_ANALYSIS_SCRIPTS = (
    (
        Path("neurips/scripts/summarize_init_policy_factorial.py"),
        "Figure 2 architecture-by-initialization-policy factorial summarizer",
    ),
    (
        Path("neurips/scripts/measure_layer_soma_factorial.py"),
        "Figure 2 layer/soma feedback and backward-only gradient diagnostic",
    ),
    (
        Path("neurips/scripts/measure_theory_diagnostics.py"),
        "Figure 2 exact-gradient and compartment-error diagnostic dependency",
    ),
)
# The reconstructed-tree task-learning bridge remains an exploratory local
# analysis because its prespecified stability gate was not met. It is excluded
# from the reviewer release and from publication-facing source data.
OPTIONAL_JOURNAL_ARCHIVED_SCRIPTS = ()

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "data",
    "external_data",
    "logs",
    "reproduced_results",
    "reproduction_runs",
    "results",
}
EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".err",
    ".log",
    ".out",
    ".pem",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
}
EXCLUDED_FILE_NAMES = {
    ".env",
    "credentials.json",
    "id_rsa",
    "secrets.json",
    # Its corresponding exploratory analysis is intentionally not released.
    "test_reconstructed_tree_task_learning.py",
}

PRIVATE_PATH_PATTERNS = (
    re.compile(rb"/n/(?:home[^/]*|holylabs)/"),
    re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
)
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("OpenAI-style token", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
)


def run_git(*args: str, text: bool = True) -> str | bytes:
    """Run Git against the source repository and return stdout."""

    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def run_journal_git(*args: str) -> str:
    """Run Git against the nested journal repository."""

    completed = subprocess.run(
        ["git", "-C", str(JOURNAL_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_submission_target(path: Path) -> None:
    """Reject output paths outside the journal submission directory."""

    submission = SUBMISSION_ROOT.resolve()
    resolved = path.resolve()
    if resolved == submission or submission not in resolved.parents:
        raise RuntimeError(f"Refusing unsafe output target: {resolved}")


def remove_generated_tree(path: Path) -> None:
    """Remove one validated generated tree, tolerating brief NFS latency."""

    ensure_submission_target(path)
    if not path.exists():
        return
    last_error: OSError | None = None
    for _ in range(8):
        try:
            shutil.rmtree(path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def excluded(relative: Path) -> bool:
    parts = set(relative.parts)
    if parts & EXCLUDED_DIRECTORY_NAMES:
        return True
    if relative.name in EXCLUDED_FILE_NAMES:
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if relative.name.startswith("slurm_"):
        return True
    return False


def extract_git_head(destination: Path, commit: str) -> None:
    """Safely extract the release-eligible files from a committed Git tree."""

    payload = run_git("archive", "--format=tar", commit, text=False)
    assert isinstance(payload, bytes)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe Git archive member: {member.name}")
            release_relative = Path(*relative.parts)
            if excluded(release_relative):
                continue
            target = destination / release_relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Unsupported non-regular Git member: {member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot extract Git member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(0o755 if member.mode & stat.S_IXUSR else 0o644)


def copy_tree_allowlisted(source: Path, destination: Path) -> int:
    """Copy regular files while applying the release exclusion policy."""

    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if excluded(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o755 if os.access(path, os.X_OK) else 0o644)
        copied += 1
    return copied


def copy_journal_material(destination: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in JOURNAL_DIRECTORIES:
        source = JOURNAL_ROOT / name
        if not source.is_dir():
            raise RuntimeError(f"Missing journal release input: {source}")
        counts[name] = copy_tree_allowlisted(source, destination / name)

    scripts_destination = destination / "scripts"
    scripts_destination.mkdir(parents=True, exist_ok=True)
    copied_scripts = 0
    for name in JOURNAL_SCRIPTS:
        source = JOURNAL_ROOT / "scripts" / name
        if not source.is_file():
            raise RuntimeError(f"Missing journal script: {source}")
        target = scripts_destination / name
        shutil.copyfile(source, target)
        target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
        copied_scripts += 1
    counts["scripts"] = copied_scripts
    inherited_source = JOURNAL_ROOT / "scripts" / "inherited_neurips"
    if not inherited_source.is_dir():
        raise RuntimeError(f"Missing inherited figure generators: {inherited_source}")
    counts["inherited_figure_generators"] = copy_tree_allowlisted(
        inherited_source, scripts_destination / "inherited_neurips"
    )

    # Source Data are distributed as a separate archive.  These two compact
    # files retain the interpretation and panel-level provenance contract.
    source_data_destination = destination / "source_data_metadata"
    source_data_destination.mkdir(parents=True, exist_ok=True)
    for relative in (Path("source_data/README.md"), Path("source_data/provenance_manifest.tsv")):
        source = JOURNAL_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"Missing source-data metadata: {source}")
        target = source_data_destination / source.name
        shutil.copyfile(source, target)
        target.chmod(0o644)
    counts["source_data_metadata"] = 2

    analysis_destination = destination / "analysis_records"
    analysis_destination.mkdir(parents=True, exist_ok=True)
    for name in JOURNAL_ANALYSIS_RECORDS:
        source = JOURNAL_ROOT / "analysis" / name
        if not source.is_file():
            raise RuntimeError(f"Missing journal analysis record: {source}")
        target = analysis_destination / name
        shutil.copyfile(source, target)
        target.chmod(0o644)
    counts["analysis_records"] = len(JOURNAL_ANALYSIS_RECORDS)
    return counts


def git_tracked(relative: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "--error-unmatch", str(relative)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def copy_archived_analysis_scripts(destination: Path) -> dict[str, object]:
    """Copy exact diagnostic sources that live outside the committed tree."""

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    requested = list(ARCHIVED_ANALYSIS_SCRIPTS)
    for relative, role in OPTIONAL_JOURNAL_ARCHIVED_SCRIPTS:
        source = JOURNAL_ROOT / relative
        if source.is_file():
            requested.append((source.relative_to(REPOSITORY_ROOT), role))

    for relative, role in requested:
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"Missing explicitly archived analysis script: {source}")
        target = destination / source.name
        if target.exists():
            raise RuntimeError(f"Duplicate archived script name: {target.name}")
        shutil.copyfile(source, target)
        target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
        source_digest = sha256(source)
        copy_digest = sha256(target)
        if source_digest != copy_digest:
            raise RuntimeError(f"Archived script copy differs: {relative}")
        records.append(
            {
                "filename": target.name,
                "origin": relative.as_posix(),
                "origin_sha256": source_digest,
                "copy_sha256": copy_digest,
                "bytes": target.stat().st_size,
                "git_tracked_at_release_head": git_tracked(relative),
                "copy_status": "byte-identical",
                "role": role,
            }
        )

    origins = destination / "ORIGINS.tsv"
    with origins.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "filename",
                "origin",
                "origin_sha256",
                "copy_sha256",
                "bytes",
                "git_tracked_at_release_head",
                "copy_status",
                "role",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    origins.chmod(0o644)

    script_lines = "\n".join(
        f"- `{record['filename']}`: {record['role']}; SHA-256 "
        f"`{record['copy_sha256']}`."
        for record in records
    )
    (destination / "README.md").write_text(
        "# Archived analysis scripts\n\n"
        "These are byte-identical copies of the diagnostic scripts used for "
        "the journal analyses but not present in the committed repository "
        "snapshot. Their repository-relative origins, source and copy hashes, "
        "sizes, and Git-tracking state are recorded in `ORIGINS.tsv`.\n\n"
        f"{script_lines}\n\n"
        "The two Figure 2 diagnostics operate on trained checkpoint directories, "
        "which are not redistributed. They can be inspected directly and run "
        "against reviewer-generated checkpoints after installing the included "
        "`dendritic_modeling` package. Their original relative repository-root "
        "discovery remains unchanged; an installed package on `PYTHONPATH` is "
        "sufficient when they are run from this archive.\n",
        encoding="utf-8",
    )
    (destination / "README.md").chmod(0o644)
    return {
        "script_count": len(records),
        "records": records,
        "optional_reconstructed_tree_script_included": any(
            record["filename"] == "run_reconstructed_tree_task_learning.py"
            for record in records
        ),
    }


def sanitize_git_snapshot(root: Path) -> list[dict[str, str | int]]:
    """Replace machine-local paths in the release copy and record each edit."""

    changes: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        payload = path.read_bytes()
        if b"\x00" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        original = text
        file_reasons: list[str] = []
        replacement_count = 0
        for old, new, reason in PORTABILITY_REPLACEMENTS:
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                replacement_count += count
                file_reasons.append(f"{reason} ({count})")
        if text != original:
            path.write_text(text, encoding="utf-8")
            changes.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "replacement_count": replacement_count,
                    "reason": "; ".join(file_reasons),
                }
            )
    return changes


def write_portability_manifest(
    path: Path, changes: Iterable[dict[str, str | int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "replacement_count", "reason"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(changes)


def release_readme(commit: str, journal_commit: str) -> str:
    return f"""# Dendritic credit-assignment software

This reviewer archive accompanies *Dendritic topology and conductance organize
local credit assignment*. It contains the complete committed Dendritic
Modeling implementation at Git commit `{commit}` and the article-specific
analysis, configuration, validation, and provenance code from journal-package
commit `{journal_commit}`.

## Layout

- `dendritic_modeling/`: a clean export of the repository Git HEAD, including
  the installable package, training and local-learning implementations,
  configurations, tests, and documentation. Uncommitted files are excluded.
- `article_analysis/code/`: standalone regular-tree checks, reconstructed-tree
  analyses, and the portable CAVE/DANDI measured-response pipeline.
- `article_analysis/configs/`: frozen and portable experiment specifications.
- `article_analysis/scripts/`: figure, source-data, cohort, perturbation,
  rerun-validation, and controlled-learning scripts.
- `article_analysis/archived_analysis_scripts/`: byte-identical diagnostic
  scripts used for the clean Figure 2 gradient analysis but not tracked at the
  release commit, with origin and SHA-256 records.
- `article_analysis/tests/`: article-level validation tests.
- `article_analysis/reproducibility/`: source hashes, cohort manifests,
  archived hardware accounting, and archive boundaries.
- `article_analysis/analysis_records/`: the frozen experiment contract and
  claim-to-evidence ledger.
- `PORTABILITY_PATCHES.tsv`: machine-local defaults changed in the release
  copy of the Git export. Scientific parameters are not modified.
- `SHA256SUMS.tsv`: SHA-256 digest and size of every other released file.

Numerical panel data are distributed separately in `Source_Data.zip`. Raw
MICRONS/CAVE and DANDI/NWB assets are not redistributed. Public identifiers,
asset paths, access requirements, and derived-data provenance are documented
under `article_analysis/reproducibility/` and in the Source Data archive.

## Environment

Python 3.10 or 3.11 is recommended. From the extracted release root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./dendritic_modeling[test]"
```

The core package dependencies are declared in
`dendritic_modeling/pyproject.toml`. CAVE and NWB retrieval additionally
require service-specific packages and, for restricted queries, user-provided
credentials. No credentials are included here.

## Fast validation

```bash
python article_analysis/code/regular_tree/test_additive_reference.py
python article_analysis/code/reconstructed_tree/verify_credit_capture_bound.py \\
  --help
pytest -q article_analysis/tests/test_alignment_controlled_learning.py
```

The other article tests document full-package checks that require archived run
directories or the complete working-paper source-data tree and are therefore
not standalone in this software-only archive. The complete repository test
suite is available under `dendritic_modeling/tests/`; some integration tests
require optional data, GPU, or distributed-runtime dependencies.

## Reproducing analyses

Read the README in each analysis directory before execution. The regular-tree
training sweeps use the installed `dendritic_modeling` package and the frozen
YAML files under `article_analysis/configs/`. For example, a portable sweep can
be materialized without scheduler submission with:

```bash
python dendritic_modeling/src/dendritic_modeling/scripts/sweeps/sweep_manager.py \\
  --config article_analysis/configs/reruns/feedback_definition_shunting_15seed.yaml \\
  --generate-only
```

Cluster account, partition, data-root, and output-root fields must be replaced
for the reviewer's environment. Historical auxiliary configurations that
referenced undistributed local artifacts contain explicit placeholders such
as `${{DENDRITIC_EXTERNAL_ARTIFACTS}}`; they are listed in
`PORTABILITY_PATCHES.tsv` and are not needed to execute the article's primary
portable analyses.

## Scope

This archive contains source code and lightweight configuration/provenance
records, not trained checkpoints, raw datasets, scheduler logs, or data
caches. The Git snapshot is taken from committed HEAD so unrelated dirty
working-tree changes cannot enter the release. Article-specific files are
current journal-package copies and are individually checksummed.

The software is released under the MIT license included at
`dendritic_modeling/LICENSE`.
"""


def scan_release(root: Path) -> list[str]:
    """Return high-confidence private-path, credential, and policy findings."""

    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if excluded(relative):
            findings.append(f"excluded artifact present: {relative.as_posix()}")
            continue
        payload = path.read_bytes()
        # These two packaging utilities contain the literal private-path
        # regular expressions used to detect and sanitize release content.
        # Their own detector patterns are not filesystem references.
        if relative.name not in {
            "build_nature_source_data.py",
            "build_software_release.py",
        }:
            for pattern in PRIVATE_PATH_PATTERNS:
                if pattern.search(payload):
                    findings.append(f"private absolute path: {relative.as_posix()}")
                    break
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                findings.append(f"possible {label}: {relative.as_posix()}")
    return findings


def list_files(root: Path, *, omit: set[str] | None = None) -> list[Path]:
    omitted = omit or set()
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in omitted
    ]


def write_checksums(root: Path) -> int:
    manifest = root / "SHA256SUMS.tsv"
    files = list_files(root, omit={manifest.name})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sha256", "bytes", "path"))
        for path in files:
            writer.writerow(
                (
                    sha256(path),
                    path.stat().st_size,
                    path.relative_to(root).as_posix(),
                )
            )
    manifest.chmod(0o644)
    return len(files)


def validate_checksums(root: Path) -> int:
    manifest = root / "SHA256SUMS.tsv"
    expected_paths: set[str] = set()
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["sha256", "bytes", "path"]:
            raise RuntimeError("Malformed SHA256SUMS.tsv header")
        for row in reader:
            relative = row["path"]
            if relative in expected_paths:
                raise RuntimeError(f"Duplicate checksum path: {relative}")
            expected_paths.add(relative)
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"Missing checksummed file: {relative}")
            if path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"Size mismatch: {relative}")
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"Digest mismatch: {relative}")
    observed = {
        path.relative_to(root).as_posix()
        for path in list_files(root, omit={manifest.name})
    }
    if observed != expected_paths:
        missing = sorted(observed - expected_paths)
        extra = sorted(expected_paths - observed)
        raise RuntimeError(
            f"Checksum coverage mismatch; unlisted={missing}, absent={extra}"
        )
    return len(expected_paths)


def zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    dt = datetime.fromtimestamp(max(epoch, 315532800), tz=timezone.utc)
    # ZIP stores timestamps at two-second resolution.
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second // 2 * 2)


def write_deterministic_zip(root: Path, archive_path: Path, epoch: int) -> None:
    timestamp = zip_datetime(epoch)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path in list_files(root):
            relative = Path(STAGE_NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=timestamp)
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compresslevel=9)


def validate_zip(archive_path: Path, release_root: Path) -> int:
    disk_paths = {
        (Path(STAGE_NAME) / path.relative_to(release_root)).as_posix()
        for path in list_files(release_root)
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP contains duplicate member names")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe ZIP member: {name}")
        if set(names) != disk_paths:
            raise RuntimeError("ZIP membership differs from staged release")
    return len(disk_paths)


def build(force: bool) -> dict[str, object]:
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    stage = SUBMISSION_ROOT / STAGE_NAME
    archive = SUBMISSION_ROOT / ARCHIVE_NAME
    digest_file = SUBMISSION_ROOT / f"{ARCHIVE_NAME}.sha256"
    for target in (stage, archive, digest_file):
        ensure_submission_target(target)
        if target.exists() and not force:
            raise RuntimeError(
                f"Output already exists: {target}. Re-run with --force to replace "
                "only these generated release targets."
            )

    commit = str(run_git("rev-parse", "HEAD")).strip()
    commit_epoch = int(str(run_git("show", "-s", "--format=%ct", commit)).strip())
    commit_utc = datetime.fromtimestamp(commit_epoch, tz=timezone.utc).isoformat()
    subject = str(run_git("show", "-s", "--format=%s", commit)).strip()
    status_lines = [
        line
        for line in str(
            run_git("status", "--porcelain=v1", "--untracked-files=all")
        ).splitlines()
        if line.strip()
    ]
    journal_commit = run_journal_git("rev-parse", "HEAD").strip()
    journal_status_lines = [
        line
        for line in run_journal_git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line.strip()
    ]
    if journal_status_lines:
        raise RuntimeError(
            "The journal repository must be clean before building the software "
            "release so every article-specific file is tied to a commit."
        )

    temporary_parent = SUBMISSION_ROOT / ".software_release_build"
    ensure_submission_target(temporary_parent)
    if temporary_parent.exists():
        if not force:
            raise RuntimeError(f"Stale temporary directory: {temporary_parent}")
        remove_generated_tree(temporary_parent)
    temporary_parent.mkdir(parents=True)

    try:
        temporary_stage = temporary_parent / STAGE_NAME
        repository_destination = temporary_stage / "dendritic_modeling"
        article_destination = temporary_stage / "article_analysis"
        repository_destination.mkdir(parents=True)
        article_destination.mkdir(parents=True)

        extract_git_head(repository_destination, commit)
        git_file_count = len(list_files(repository_destination))
        portability_changes = sanitize_git_snapshot(repository_destination)
        journal_counts = copy_journal_material(article_destination)
        archived_scripts = copy_archived_analysis_scripts(
            article_destination / "archived_analysis_scripts"
        )
        article_portability_changes = sanitize_git_snapshot(article_destination)
        for change in article_portability_changes:
            change["path"] = f"article_analysis/{change['path']}"
        portability_changes.extend(article_portability_changes)

        (temporary_stage / "README.md").write_text(
            release_readme(commit, journal_commit), encoding="utf-8"
        )
        (temporary_stage / "README.md").chmod(0o644)
        shutil.copyfile(REPOSITORY_ROOT / "LICENSE", temporary_stage / "LICENSE")
        (temporary_stage / "LICENSE").chmod(0o644)
        write_portability_manifest(
            temporary_stage / "PORTABILITY_PATCHES.tsv", portability_changes
        )

        metadata = {
            "release": "Dendritic credit-assignment software",
            "release_schema": 1,
            "repository_commit": commit,
            "repository_commit_utc": commit_utc,
            "repository_commit_subject": subject,
            "repository_snapshot_method": "git archive HEAD",
            "journal_repository_commit": journal_commit,
            "journal_repository_clean_at_build": True,
            "working_tree_dirty_at_build": bool(status_lines),
            "working_tree_change_count": len(status_lines),
            "working_tree_material_included": False,
            "journal_material_source": "current allow-listed journal package files",
            "git_snapshot_file_count": git_file_count,
            "journal_file_counts": journal_counts,
            "archived_analysis_scripts": archived_scripts,
            "portability_patch_file_count": len(portability_changes),
            "portability_replacement_count": sum(
                int(change["replacement_count"]) for change in portability_changes
            ),
            "excluded_classes": [
                "raw data and data caches",
                "model checkpoints",
                "scheduler scripts and logs",
                "temporary results",
                "Python and test caches",
                "credential files",
            ],
            "raw_microns_or_dandi_data_included": False,
            "trained_checkpoints_included": False,
            "credentials_included": False,
            "source_data_distribution": "separate Source_Data.zip",
            "deterministic_timestamp_epoch": commit_epoch,
        }
        (temporary_stage / "METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_stage / "METADATA.json").chmod(0o644)

        findings = scan_release(temporary_stage)
        if findings:
            raise RuntimeError(
                "Release policy scan failed:\n- " + "\n- ".join(findings)
            )

        checksummed = write_checksums(temporary_stage)
        validated = validate_checksums(temporary_stage)
        if validated != checksummed:
            raise RuntimeError("Internal checksum-count mismatch")

        temporary_archive = temporary_parent / ARCHIVE_NAME
        write_deterministic_zip(temporary_stage, temporary_archive, commit_epoch)
        zip_members = validate_zip(temporary_archive, temporary_stage)

        if stage.exists():
            remove_generated_tree(stage)
        for target in (archive, digest_file):
            if target.exists():
                target.unlink()
        temporary_stage.replace(stage)
        temporary_archive.replace(archive)
        digest_file.write_text(
            f"{sha256(archive)}  {archive.name}\n", encoding="utf-8"
        )
        digest_file.chmod(0o644)

        # Revalidate after the atomic moves.
        validate_checksums(stage)
        validate_zip(archive, stage)
        post_findings = scan_release(stage)
        if post_findings:
            raise RuntimeError(
                "Post-move release scan failed:\n- " + "\n- ".join(post_findings)
            )
        expected_digest = digest_file.read_text(encoding="utf-8").split()[0]
        if sha256(archive) != expected_digest:
            raise RuntimeError("Published ZIP checksum mismatch")

        return {
            "status": "ok",
            "stage": str(stage.relative_to(JOURNAL_ROOT)),
            "archive": str(archive.relative_to(JOURNAL_ROOT)),
            "archive_sha256": expected_digest,
            "archive_bytes": archive.stat().st_size,
            "zip_members": zip_members,
            "checksummed_files": validated,
            "git_commit": commit,
            "journal_git_commit": journal_commit,
            "git_snapshot_files": git_file_count,
            "journal_file_counts": journal_counts,
            "portability_patched_files": len(portability_changes),
            "portability_replacements": sum(
                int(change["replacement_count"]) for change in portability_changes
            ),
            "policy_scan_findings": 0,
        }
    finally:
        if temporary_parent.exists():
            remove_generated_tree(temporary_parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the generated software-release stage and archive",
    )
    args = parser.parse_args()
    print(json.dumps(build(force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
