"""Fail-closed provenance helpers for NVIDIA Model Optimizer baselines.

Model Optimizer is an optional dependency.  Importing this module never imports
``modelopt.torch`` (which is expensive and registers framework plugins); an
experiment calls :func:`require_pinned_modelopt` only after its model, data, and
output identities have passed their own checks.

The helpers deliberately do not call a ModelOpt technique themselves.  They
bind the external software and the matched-comparison contract around a run so
that PTQ, pruning, NAS, or distillation cannot be compared against a dendritic
arm with different data, recovery budget, storage accounting, or hardware.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

MODELOPT_DISTRIBUTION = "nvidia-modelopt"
PINNED_MODELOPT_VERSION = "0.46.0"
MODELOPT_REPOSITORY = "https://github.com/NVIDIA/Model-Optimizer"
MODELOPT_LICENSE = "Apache-2.0"
MODELOPT_BASELINE_SCHEMA = "dendritic_modelopt_baseline_preflight/v1"

# A comparison may report additional diagnostics, but none of these may be
# omitted from a claim that one method is a better compression/runtime point.
REQUIRED_MATCHED_AXES = (
    "teacher_artifact_identity",
    "calibration_window_identity",
    "evaluation_window_identity",
    "recovery_data_and_optimizer_budget",
    "quality_metrics",
    "stored_bytes_including_indices_and_scales",
    "active_parameters_and_realized_operations",
    "peak_device_memory",
    "prefill_and_decode_runtime",
    "hardware_and_deployment_backend",
)


@dataclass(frozen=True)
class ModelOptEnvironment:
    """The installed external package and local execution environment."""

    installed: bool
    distribution: str
    version: str | None
    expected_version: str
    version_matches: bool
    module_path: str | None
    source_git_commit: str | None
    source_git_dirty: bool | None
    torch_version: str
    cuda_runtime: str | None
    cuda_available: bool
    cuda_devices: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_identity(start: Path | None) -> tuple[str | None, bool | None]:
    if start is None:
        return None, None
    candidate = start if start.is_dir() else start.parent
    root: Path | None = None
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            root = parent
            break
    if root is None:
        return None, None
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or status.returncode != 0:
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def inspect_modelopt_environment(
    *, expected_version: str = PINNED_MODELOPT_VERSION
) -> ModelOptEnvironment:
    """Inspect ModelOpt without importing its heavyweight torch plugins."""

    spec = importlib.util.find_spec("modelopt")
    module_path = None if spec is None or spec.origin is None else str(spec.origin)
    try:
        version = importlib.metadata.version(MODELOPT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    # A wheel under site-packages may itself live below an unrelated parent Git
    # repository (as Harvard home/workspace layouts sometimes do).  Only an
    # external module path is evidence of an editable source checkout.
    module_source = None if module_path is None else Path(module_path)
    if module_source is not None and "site-packages" in module_source.parts:
        commit, dirty = None, None
    else:
        commit, dirty = _git_identity(module_source)
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    installed = spec is not None and version is not None
    return ModelOptEnvironment(
        installed=installed,
        distribution=MODELOPT_DISTRIBUTION,
        version=version,
        expected_version=str(expected_version),
        version_matches=installed and version == str(expected_version),
        module_path=module_path,
        source_git_commit=commit,
        source_git_dirty=dirty,
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cuda_devices=tuple(devices),
    )


def require_pinned_modelopt(
    *, expected_version: str = PINNED_MODELOPT_VERSION
) -> ModelOptEnvironment:
    """Return the environment or fail before an evidence-producing run."""

    environment = inspect_modelopt_environment(expected_version=expected_version)
    if not environment.installed:
        raise RuntimeError(
            f"{MODELOPT_DISTRIBUTION} is not installed; use the optional "
            f"dendritic_modeling[modelopt] environment pinned to {expected_version}"
        )
    if not environment.version_matches:
        raise RuntimeError(
            "ModelOpt version drift: installed "
            f"{environment.version}, required {expected_version}"
        )
    if environment.source_git_dirty:
        raise RuntimeError("editable ModelOpt source checkout is dirty")
    return environment


def file_identity(path: Path) -> dict[str, Any]:
    """Hash one required regular-file input after rejecting symlinks."""

    unresolved = Path(os.path.abspath(path))
    if unresolved.is_symlink():
        raise ValueError(f"baseline input must not be a symbolic link: {path}")
    source = unresolved.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"baseline input is not a regular file: {source}")
    initial = (int(source.stat().st_size), _file_sha256(source))
    final = (int(source.stat().st_size), _file_sha256(source))
    if initial != final:
        raise RuntimeError(f"baseline input changed while hashing: {source}")
    return {"path": str(source), "size_bytes": initial[0], "sha256": initial[1]}


def build_baseline_preflight(
    *,
    technique: str,
    model_family: str,
    model_artifact_manifest: Path,
    frozen_window_slice_set_manifest: Path,
    recipe_file: Path,
    recovery_budget: Mapping[str, Any],
    expected_modelopt_version: str = PINNED_MODELOPT_VERSION,
    require_installed: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed external-baseline preflight record.

    This is intentionally technique-neutral.  It can bind an NVIDIA PTQ
    recipe today and a Minitron/Puzzletron or QAD recipe later without changing
    the matched axes or inventing a second evidence schema.
    """

    normalized_technique = str(technique).strip().lower()
    if normalized_technique not in {
        "ptq",
        "qat",
        "qad",
        "distillation",
        "minitron",
        "puzzletron",
        "sparsity",
        "speculative_decoding",
    }:
        raise ValueError(f"unsupported ModelOpt comparison technique: {technique!r}")
    if not str(model_family).strip():
        raise ValueError("model_family must be non-empty")
    if not isinstance(recovery_budget, Mapping) or not recovery_budget:
        raise ValueError("recovery_budget must be a non-empty mapping")

    environment = inspect_modelopt_environment(
        expected_version=expected_modelopt_version
    )
    if require_installed:
        environment = require_pinned_modelopt(
            expected_version=expected_modelopt_version
        )
    inputs = {
        "model_artifact_manifest": file_identity(model_artifact_manifest),
        "frozen_window_slice_set_manifest": file_identity(
            frozen_window_slice_set_manifest
        ),
        "recipe_file": file_identity(recipe_file),
    }
    record: dict[str, Any] = {
        "schema": MODELOPT_BASELINE_SCHEMA,
        "status": "preflight_only",
        "technique": normalized_technique,
        "model_family": str(model_family),
        "external_software": {
            "repository": MODELOPT_REPOSITORY,
            "license": MODELOPT_LICENSE,
            **environment.as_dict(),
        },
        "inputs": inputs,
        "recovery_budget": dict(recovery_budget),
        "required_matched_axes": list(REQUIRED_MATCHED_AXES),
        "claim_boundary": (
            "This record proves only that the external software, recipe, model "
            "identity, frozen data partition, and declared recovery budget were "
            "bound before execution. It is not a quality, compression, runtime, "
            "or superiority result."
        ),
        "extra": {} if extra is None else dict(extra),
    }
    record["content_sha256"] = _canonical_sha256(record)
    return record


def verify_baseline_preflight(record: Mapping[str, Any]) -> None:
    """Re-hash a preflight record and all bound regular files."""

    if record.get("schema") != MODELOPT_BASELINE_SCHEMA:
        raise ValueError("unsupported ModelOpt baseline preflight schema")
    expected = record.get("content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("preflight content_sha256 is invalid")
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    if _canonical_sha256(body) != expected:
        raise ValueError("preflight content identity changed")
    inputs = record.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("preflight inputs are absent")
    for label, raw in inputs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"preflight input {label} is invalid")
        current = file_identity(Path(str(raw.get("path", ""))))
        if current != dict(raw):
            raise ValueError(f"preflight input changed: {label}")


def write_preflight_atomic(record: Mapping[str, Any], output: Path) -> None:
    """Write a validated preflight without exposing a partial JSON file."""

    verify_baseline_preflight(record)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def require_complete_matched_metrics(
    metrics: Mapping[str, Any], *, required_axes: Sequence[str] = REQUIRED_MATCHED_AXES
) -> None:
    """Reject a superiority/Pareto claim whose matched ledger is incomplete."""

    missing = [axis for axis in required_axes if axis not in metrics]
    if missing:
        raise ValueError(f"matched comparison metrics are incomplete: {missing}")


__all__ = [
    "MODELOPT_BASELINE_SCHEMA",
    "MODELOPT_DISTRIBUTION",
    "MODELOPT_LICENSE",
    "MODELOPT_REPOSITORY",
    "PINNED_MODELOPT_VERSION",
    "REQUIRED_MATCHED_AXES",
    "ModelOptEnvironment",
    "build_baseline_preflight",
    "file_identity",
    "inspect_modelopt_environment",
    "require_complete_matched_metrics",
    "require_pinned_modelopt",
    "verify_baseline_preflight",
    "write_preflight_atomic",
]
