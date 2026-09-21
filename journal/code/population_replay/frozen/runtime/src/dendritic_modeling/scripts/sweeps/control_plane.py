"""Reproducibility and scheduler policy for generated sweeps.

The functions in this module deliberately have no dependency on Slurm itself.
They validate the small set of compute profiles approved for this project and
freeze the inputs that determine a generated sweep before any job is submitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEDULER_PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True)
class SchedulerProfile:
    """One resolved account/partition pairing."""

    profile_id: str
    account: str
    partition: str
    validated: bool = True
    placeholder: bool = False

    def as_dict(self) -> dict[str, str | bool]:
        """Return the stable JSON/YAML representation used in manifests."""
        return asdict(self)


# This is the executable policy. Project-specific YAML files may mirror these
# names for documentation, but cannot expand the accepted pairings.
ALLOWED_SCHEDULER_PROFILES: tuple[SchedulerProfile, ...] = (
    SchedulerProfile(
        profile_id="kempner_dev_priority",
        account="kempner_dev",
        partition="kempner_priority",
    ),
    SchedulerProfile(
        profile_id="kempner_dev_eng",
        account="kempner_dev",
        partition="kempner_eng",
    ),
    SchedulerProfile(
        profile_id="kempner_dev_requeue",
        account="kempner_dev",
        partition="kempner_requeue",
    ),
    SchedulerProfile(
        profile_id="kempner_dev_rtx",
        account="kempner_dev",
        partition="kempner_rtx",
    ),
    SchedulerProfile(
        profile_id="kempner_bsabatini_lab_h100",
        account="kempner_bsabatini_lab",
        partition="kempner_h100",
    ),
    SchedulerProfile(
        profile_id="kempner_bsabatini_lab_h100_priority",
        account="kempner_bsabatini_lab",
        partition="kempner_h100_priority",
    ),
)

_PROFILE_BY_PAIR = {
    (profile.account, profile.partition): profile
    for profile in ALLOWED_SCHEDULER_PROFILES
}


def _required_scheduler_value(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Scheduler {field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Scheduler {field_name} is required")
    return normalized


def validate_scheduler_profile(
    account: Any,
    partition: Any,
    *,
    allow_placeholder: bool = False,
) -> SchedulerProfile:
    """Validate and name an approved scheduler account/partition pair.

    A placeholder is accepted only for non-submitting generation paths. An
    explicit, non-placeholder pair is always validated, including during a dry
    run, so a dry run remains a faithful submission preflight.
    """
    normalized_account = _required_scheduler_value(account, "account")
    normalized_partition = _required_scheduler_value(partition, "partition")

    has_placeholder = SCHEDULER_PLACEHOLDER in {
        normalized_account,
        normalized_partition,
    }
    if has_placeholder:
        if not allow_placeholder:
            raise ValueError(
                "Scheduler placeholders are permitted only for non-submitting "
                "generate-only or dry-run workflows"
            )
        return SchedulerProfile(
            profile_id="non_submitting_placeholder",
            account=normalized_account,
            partition=normalized_partition,
            validated=False,
            placeholder=True,
        )

    profile = _PROFILE_BY_PAIR.get((normalized_account, normalized_partition))
    if profile is not None:
        return profile

    allowed = ", ".join(
        f"({item.account}, {item.partition})" for item in ALLOWED_SCHEDULER_PROFILES
    )
    raise ValueError(
        "Unapproved scheduler account/partition pair "
        f"({normalized_account}, {normalized_partition}). Allowed pairs: {allowed}"
    )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _manifest_path(path: Path, *, repo_root: Path, output_dir: Path) -> str:
    resolved = path.resolve()
    for base, prefix in ((output_dir.resolve(), ""), (repo_root.resolve(), "repo:")):
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        return f"{prefix}{relative.as_posix()}"
    return str(resolved)


def discover_source_identity(
    repo_root: str | Path,
    *,
    source_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Collect source and Git identity without requiring a Git checkout."""
    root = Path(repo_root).resolve()
    commit = _git_value(root, "rev-parse", "HEAD")
    branch = _git_value(root, "symbolic-ref", "--short", "-q", "HEAD")
    tracked_status = _git_value(root, "status", "--porcelain", "--untracked-files=no")
    tracked_diff = _git_bytes(root, "diff", "--binary", "HEAD")

    files = []
    for source_file in source_files:
        path = Path(source_file).resolve()
        if not path.is_file():
            continue
        try:
            display_path = f"repo:{path.relative_to(root).as_posix()}"
        except ValueError:
            display_path = str(path)
        files.append({"path": display_path, "sha256": sha256_file(path)})

    git_identity: dict[str, Any] = {"available": commit is not None}
    if commit is not None:
        git_identity["commit"] = commit
        git_identity["branch"] = branch or None
        git_identity["tracked_worktree_dirty"] = (
            None if tracked_status is None else bool(tracked_status)
        )
        git_identity["tracked_diff_sha256"] = (
            None if tracked_diff is None else hashlib.sha256(tracked_diff).hexdigest()
        )

    return {
        "repository_name": root.name,
        "git": git_identity,
        "python_version": platform.python_version(),
        "files": sorted(files, key=lambda item: item["path"]),
    }


def write_frozen_sweep_manifest(
    *,
    output_dir: str | Path,
    input_config_path: str | Path,
    resolved_config_path: str | Path,
    generated_config_paths: Iterable[str | Path],
    expected_config_count: int,
    scheduler_profile: SchedulerProfile,
    repo_root: str | Path,
    source_files: Iterable[str | Path] = (),
    generator_identity: str | None = None,
    generation_mode: str,
) -> Path:
    """Write an immutable-by-convention manifest for one generated sweep."""
    output_root = Path(output_dir).resolve()
    repository_root = Path(repo_root).resolve()
    generated_paths = [Path(path).resolve() for path in generated_config_paths]
    expected = int(expected_config_count)
    if expected < 0:
        raise ValueError("expected_config_count must be non-negative")
    if len(generated_paths) != expected:
        raise ValueError(
            "Generated config count does not match the frozen expectation: "
            f"expected {expected}, observed {len(generated_paths)}"
        )

    input_path = Path(input_config_path).resolve()
    resolved_path = Path(resolved_config_path).resolve()
    generated_records = [
        {
            "index": index,
            "path": _manifest_path(
                path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(path),
        }
        for index, path in enumerate(generated_paths)
    ]

    payload = {
        "schema_version": 1,
        "manifest_type": "dendritic_modeling.frozen_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_mode": str(generation_mode),
        "expected_config_count": expected,
        "original_yaml": {
            "path": _manifest_path(
                input_path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(input_path),
        },
        "resolved_original_yaml": {
            "path": _manifest_path(
                resolved_path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(resolved_path),
        },
        "generated_configs": generated_records,
        "scheduler_profile": scheduler_profile.as_dict(),
        "generator": generator_identity,
        "source_identity": discover_source_identity(
            repository_root,
            source_files=source_files,
        ),
    }

    manifest_path = output_root / "frozen_sweep_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Frozen sweep manifest already exists: {manifest_path}")
    temporary_path = output_root / f".{manifest_path.name}.tmp-{os.getpid()}"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


__all__ = [
    "ALLOWED_SCHEDULER_PROFILES",
    "SCHEDULER_PLACEHOLDER",
    "SchedulerProfile",
    "discover_source_identity",
    "sha256_file",
    "validate_scheduler_profile",
    "write_frozen_sweep_manifest",
]
