"""Fail-closed site-exposure audit for a prospective FMI-v3 experiment.

The audit answers a narrow question: under declared artifact roots and required
historical manifests, has any proposed model/target site already appeared in a
student screen, result, recovery, or earlier prospective declaration?  It does
not claim knowledge of files outside those roots.  Teacher-only profiles and
deletion maps are inventoried separately because FMI-v3 is allowed to consume
teacher statistics without seeing student-arm outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_REGISTRATION_SCHEMA = "dendritic_fmi_prospective_registration/v3"
EXPOSURE_INDEX_SCHEMA = "dendritic_fmi_site_exposure_index/v3"
FROZEN_PROTOCOL_SCHEMA = "dendritic_fmi_prospective_protocol/v3"

EXIT_PRIOR_EXPOSURE = 74
EXIT_ARTIFACT_MISMATCH = 75
EXIT_EXISTING_OUTPUT = 76

_TEXT_SUFFIXES = frozenset(
    {".json", ".yaml", ".yml", ".tsv", ".txt", ".out", ".sha256", ".sbatch"}
)
_MAX_TEXT_BYTES = 32 * 1024 * 1024
_TARGET_PATTERNS = (
    re.compile(r"model[._/]layers[._/](?P<layer>\d+)[._/]mlp", re.IGNORECASE),
    re.compile(r"model_layers_(?P<layer>\d+)_mlp", re.IGNORECASE),
)
_PATH_LAYER_PATTERN = re.compile(
    r"(?:^|[/_.-])L(?:ayer)?[_-]?(?P<layer>\d+)(?:$|[/_.-])", re.IGNORECASE
)
_STUDENT_MARKERS = frozenset(
    {
        "screen",
        "score",
        "result",
        "recovery",
        "checkpoint",
        "cell_pilot",
        "rung",
        "predictions",
        "prospective",
        "student",
        "cells/",
    }
)
_TEACHER_ONLY_MARKERS = frozenset(
    {"deletion_map", "teacher_profile", "profile_base", "teacher-side"}
)


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a full lowercase SHA-256 digest")
    return normalized


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _model_relevant(value: str, aliases: Sequence[str]) -> bool:
    normalized = _normalize_alias(value)
    return any(_normalize_alias(alias) in normalized for alias in aliases)


def _layers_from_target_strings(value: str) -> set[int]:
    layers: set[int] = set()
    for pattern in _TARGET_PATTERNS:
        layers.update(int(match.group("layer")) for match in pattern.finditer(value))
    return layers


def _layers_from_json(value: Any, *, target_module: str) -> set[int]:
    """Extract only explicit site fields and target-module paths."""

    layers: set[int] = set()
    if isinstance(value, str):
        return _layers_from_target_strings(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            layers.update(_layers_from_json(item, target_module=target_module))
        return layers
    if not isinstance(value, Mapping):
        return layers
    declared_target = str(value.get("target_module", target_module)).strip().lower()
    target_matches = declared_target == target_module.lower()
    for raw_key, item in value.items():
        key = str(raw_key).strip().lower()
        if key in {"held_out_layers", "new_site_layers", "proposed_layers"}:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                layers.update(
                    int(layer)
                    for layer in item
                    if isinstance(layer, int) and not isinstance(layer, bool)
                )
        elif key == "layers" and target_matches:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                layers.update(
                    int(layer)
                    for layer in item
                    if isinstance(layer, int) and not isinstance(layer, bool)
                )
            elif isinstance(item, Mapping):
                layers.update(int(layer) for layer in item if str(layer).isdigit())
        elif key in {"layer", "layer_index"} and target_matches:
            if isinstance(item, int) and not isinstance(item, bool):
                layers.add(int(item))
        layers.update(_layers_from_json(item, target_module=target_module))
    return layers


def _classify_evidence(path_text: str) -> str:
    lowered = path_text.lower()
    teacher_only = any(marker in lowered for marker in _TEACHER_ONLY_MARKERS)
    student = any(marker in lowered for marker in _STUDENT_MARKERS)
    if teacher_only and not student:
        return "teacher_only"
    if path_text.endswith(".sbatch"):
        return "planned_student_exposure"
    return "student_or_declared_exposure"


def _evidence_record(
    *,
    root: Path,
    path: Path,
    layers: set[int],
    evidence_class: str,
    content_sha256: str | None,
) -> dict[str, Any]:
    stat = path.lstat()
    return {
        "root": str(root),
        "relative_path": str(path.relative_to(root)),
        "file_type": "symlink" if path.is_symlink() else "file",
        "size_bytes": int(stat.st_size),
        "content_sha256": content_sha256,
        "layers": sorted(layers),
        "evidence_class": evidence_class,
    }


def scan_site_exposures(
    *,
    scan_roots: Sequence[Path | str],
    model_aliases: Sequence[str],
    target_module: str,
    layer_count: int,
) -> dict[str, Any]:
    """Build a canonical inventory of prior target-site exposure.

    Relevant text artifacts are content-hashed. Large binary checkpoints are
    identified by their relative path and size because their role here is only
    to prove that a student site existed, not to validate checkpoint tensors.
    """

    aliases = tuple(str(alias).strip() for alias in model_aliases if str(alias).strip())
    if not aliases:
        raise ValueError("model_aliases must be non-empty")
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    roots: list[Path] = []
    for raw_root in scan_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"site-exposure scan root is not a directory: {root}"
            )
        if root in roots:
            raise ValueError(f"duplicate resolved scan root: {root}")
        roots.append(root)
    if not roots:
        raise ValueError("at least one scan root is required")

    records: list[dict[str, Any]] = []
    scanned_file_count = 0
    relevant_file_count = 0
    for root in sorted(roots):
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in sorted(directory_names):
                directory_path = base / name
                if directory_path.is_symlink() and _model_relevant(
                    str(directory_path), aliases
                ):
                    raise ValueError(
                        "model-relevant symlinked directories make the exposure "
                        f"scan incomplete: {directory_path}"
                    )
            for name in sorted(file_names):
                path = base / name
                scanned_file_count += 1
                path_text = str(path)
                path_relevant = _model_relevant(path_text, aliases)
                path_layers = _layers_from_target_strings(path_text)
                if target_module.lower() in path_text.lower():
                    path_layers.update(
                        int(match.group("layer"))
                        for match in _PATH_LAYER_PATTERN.finditer(path_text)
                    )
                content_relevant = False
                content_layers: set[int] = set()
                content_sha256: str | None = None
                if path.suffix.lower() in _TEXT_SUFFIXES:
                    size = path.lstat().st_size
                    if (path_relevant or path_layers) and size > _MAX_TEXT_BYTES:
                        raise ValueError(
                            "model-relevant text artifact exceeds the fail-closed "
                            f"scan limit: {path} ({size} bytes)"
                        )
                    if size <= _MAX_TEXT_BYTES:
                        try:
                            payload = path.read_bytes()
                        except OSError as exc:
                            if path_relevant or path_layers:
                                raise ValueError(
                                    f"cannot read model-relevant artifact {path}: {exc}"
                                ) from exc
                            continue
                        decoded = payload.decode("utf-8", errors="replace")
                        content_relevant = _model_relevant(decoded, aliases)
                        if path_relevant or content_relevant:
                            content_sha256 = hashlib.sha256(payload).hexdigest()
                            content_layers.update(_layers_from_target_strings(decoded))
                            if path.suffix.lower() == ".json":
                                try:
                                    parsed = json.loads(decoded)
                                except json.JSONDecodeError as exc:
                                    raise ValueError(
                                        f"invalid model-relevant JSON artifact {path}: {exc}"
                                    ) from exc
                                content_layers.update(
                                    _layers_from_json(
                                        parsed, target_module=target_module
                                    )
                                )
                if not (path_relevant or content_relevant):
                    continue
                layers = {
                    layer
                    for layer in path_layers | content_layers
                    if 0 <= layer < layer_count
                }
                if not layers:
                    continue
                relevant_file_count += 1
                records.append(
                    _evidence_record(
                        root=root,
                        path=path,
                        layers=layers,
                        evidence_class=_classify_evidence(path_text),
                        content_sha256=content_sha256,
                    )
                )

    exposure: dict[int, list[dict[str, Any]]] = {}
    teacher_only: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        destination = (
            teacher_only if record["evidence_class"] == "teacher_only" else exposure
        )
        for layer in record["layers"]:
            destination.setdefault(int(layer), []).append(record)
    payload = {
        "schema": EXPOSURE_INDEX_SCHEMA,
        "scan_contract": {
            "roots": [str(root) for root in sorted(roots)],
            "model_aliases": list(aliases),
            "target_module": str(target_module),
            "layer_count": int(layer_count),
            "follow_directory_symlinks": False,
            "maximum_text_artifact_bytes": _MAX_TEXT_BYTES,
            "text_identity": "relative_path_size_and_sha256",
            "binary_identity": "relative_path_and_size_only",
            "scope_limitation": (
                "The index proves absence only under the declared roots at scan "
                "time; files elsewhere are outside its claim boundary."
            ),
        },
        "inventory": {
            "scanned_file_count": scanned_file_count,
            "relevant_file_count": relevant_file_count,
            "records": records,
        },
        "student_or_declared_exposed_layers": {
            str(layer): rows for layer, rows in sorted(exposure.items())
        },
        "teacher_only_layers": {
            str(layer): rows for layer, rows in sorted(teacher_only.items())
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _load_registration(path: Path) -> tuple[dict[str, Any], str]:
    payload_bytes = path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid registration JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("registration root must be a JSON object")
    if payload.get("schema") != PROTOCOL_REGISTRATION_SCHEMA:
        raise ValueError("unsupported FMI-v3 prospective registration schema")
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def _validate_layers(
    values: object, *, label: str, layer_count: int
) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    layers = tuple(int(value) for value in values)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError(f"{label} must be non-empty and unique")
    if any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError(f"{label} contains a layer outside the model")
    return layers


def _verify_required_artifacts(
    registration: Mapping[str, Any],
    supplied_paths: Mapping[str, Path | str],
    *,
    target_module: str,
) -> tuple[dict[str, Any], set[int]]:
    required = registration.get("required_development_artifacts")
    if not isinstance(required, Mapping) or not required:
        raise ValueError("registration requires development artifacts")
    if set(supplied_paths) != set(required):
        raise ValueError(
            "supplied development artifacts must match registration labels exactly"
        )
    audit: dict[str, Any] = {}
    development_layers: set[int] = set()
    for label in sorted(required):
        spec = required[label]
        if not isinstance(spec, Mapping):
            raise TypeError(f"development artifact spec {label!r} must be a mapping")
        path = Path(supplied_paths[label]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required development artifact is missing: {path}")
        observed_sha = _sha256_file(path)
        expected_sha = _full_sha256(spec.get("sha256"), label=f"{label}.sha256")
        if observed_sha != expected_sha:
            raise ValueError(
                f"development artifact hash mismatch for {label!r}: "
                f"{observed_sha} != {expected_sha}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed_layers = _layers_from_json(payload, target_module=target_module)
        expected_layers = {int(layer) for layer in spec.get("layers", [])}
        if not expected_layers or not expected_layers.issubset(observed_layers):
            raise ValueError(
                f"development artifact {label!r} does not prove its registered sites; "
                f"expected={sorted(expected_layers)}, observed={sorted(observed_layers)}"
            )
        development_layers.update(expected_layers)
        audit[label] = {
            "path": str(path),
            "sha256": observed_sha,
            "role": str(spec.get("role", "development_only")),
            "registered_layers": sorted(expected_layers),
            "observed_layers": sorted(observed_layers),
        }
    return audit, development_layers


def _git_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[4]

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    return {
        "repository_root": str(repository_root),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(status.stdout.strip()),
        "tracked_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
    }


def freeze_prospective_protocol_v3(
    *,
    registration_path: Path | str,
    scan_roots: Sequence[Path | str],
    development_artifact_paths: Mapping[str, Path | str],
) -> dict[str, Any]:
    """Verify untouched sites and return a frozen, outcome-blind protocol."""

    registration_file = Path(registration_path).expanduser().resolve()
    registration, registration_sha = _load_registration(registration_file)
    model = registration.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("registration.model must be a mapping")
    layer_count = int(model.get("layer_count", 0))
    aliases = model.get("aliases")
    target_module = str(model.get("target_module", "mlp"))
    proposed = _validate_layers(
        registration.get("new_site_layers"),
        label="new_site_layers",
        layer_count=layer_count,
    )
    required_audit, development_layers = _verify_required_artifacts(
        registration,
        development_artifact_paths,
        target_module=target_module,
    )
    registered_development = registration.get("development_layer_sets")
    if not isinstance(registered_development, Mapping):
        raise ValueError("development_layer_sets must be a mapping")
    for label, layers in registered_development.items():
        development_layers.update(
            _validate_layers(
                layers,
                label=f"development_layer_sets.{label}",
                layer_count=layer_count,
            )
        )
    direct_overlap = sorted(set(proposed).intersection(development_layers))
    if direct_overlap:
        raise RuntimeError(
            f"proposed FMI-v3 sites overlap development sites: {direct_overlap}"
        )
    exposure = scan_site_exposures(
        scan_roots=scan_roots,
        model_aliases=aliases,
        target_module=target_module,
        layer_count=layer_count,
    )
    exposed_layers = {
        int(layer) for layer in exposure["student_or_declared_exposed_layers"]
    }
    overlap = sorted(set(proposed).intersection(exposed_layers))
    if overlap:
        evidence = {
            str(layer): exposure["student_or_declared_exposed_layers"][str(layer)]
            for layer in overlap
        }
        raise RuntimeError(
            "FAIL-CLOSED: proposed FMI-v3 sites have prior student/declared "
            f"exposure: layers={overlap}, evidence={evidence}"
        )
    payload = {
        "schema": FROZEN_PROTOCOL_SCHEMA,
        "protocol_id": str(registration.get("protocol_id")),
        "status": "frozen_before_new_site_teacher_profiles_or_student_training",
        "outcome_blind": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": _git_provenance(),
        "registration": registration,
        "registration_provenance": {
            "path": str(registration_file),
            "sha256": registration_sha,
        },
        "required_development_artifacts": required_audit,
        "development_layers": sorted(development_layers),
        "new_site_layers": list(proposed),
        "new_site_verdict": "no_prior_student_or_declared_exposure_under_scan_contract",
        "exposure_index": exposure,
        "claim_boundary": (
            "The sites are untouched with respect to the registered development "
            "artifacts and declared scan roots. The eventual final-regret split "
            "must remain unread until local fitting and promotion are frozen."
        ),
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def write_frozen_protocol(
    payload: Mapping[str, Any], *, output_path: Path | str
) -> tuple[Path, Path]:
    """Create, never overwrite, a protocol plus a verifying SHA-256 sidecar."""

    output = Path(output_path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen protocol or sidecar: {output}, {sidecar}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    output.chmod(0o444)
    sidecar.chmod(0o444)
    return output, sidecar


__all__ = [
    "EXIT_ARTIFACT_MISMATCH",
    "EXIT_EXISTING_OUTPUT",
    "EXIT_PRIOR_EXPOSURE",
    "EXPOSURE_INDEX_SCHEMA",
    "FROZEN_PROTOCOL_SCHEMA",
    "PROTOCOL_REGISTRATION_SCHEMA",
    "freeze_prospective_protocol_v3",
    "scan_site_exposures",
    "write_frozen_protocol",
]
