"""Hash-bind a local Hugging Face model and tokenizer artifact set.

Capability gates must identify the model that was actually loaded, rather
than trusting a user-supplied opaque digest.  This module discovers the files
that determine a local Transformers checkpoint, hashes their contents, and
later revalidates every file before an adaptive experiment starts.

The identity includes the model configuration, the exact weight index and
all referenced weight shards, tokenizer assets, and local Python modeling
code.  It deliberately excludes training checkpoints, optimizer state, and
unrelated files that ``from_pretrained`` does not consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ARTIFACT_IDENTITY_SCHEMA = "dendritic_hf_model_artifact_identity/v1"
COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA = "dendritic_hf_model_artifact_identity/v2"
MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA_V1 = (
    "dendritic_hf_model_artifact_preverification_receipt/v1"
)
MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA = (
    "dendritic_hf_model_artifact_preverification_receipt/v2"
)
MODEL_ARTIFACT_PREVERIFICATION_CLAIM_BOUNDARY_V1 = (
    "Full SHA-256 verification of every v2-manifest artifact at CPU receipt "
    "creation and exact SHA-256 verification of the immutable receipt at the "
    "consumer are the cryptographic boundaries. Complete path, symlink, device, "
    "inode, mode, size, mtime, and ctime guards bridge those boundaries and are "
    "rechecked immediately before publication; the stat guards detect mutation "
    "but do not independently hash artifact content."
)
MODEL_ARTIFACT_PREVERIFICATION_CLAIM_BOUNDARY = (
    "Full SHA-256 verification of every v2-manifest artifact at CPU receipt "
    "creation and exact SHA-256 verification of the immutable receipt at the "
    "consumer are the cryptographic boundaries. Complete path, symlink, inode, "
    "mode, size, mtime, and ctime guards bridge those boundaries and are "
    "rechecked immediately before publication. Device identifiers are admitted "
    "only through one consistent bijective source-to-consumer mount-namespace "
    "remap across the model root, manifest, and every artifact; the stat guards "
    "detect mutation but do not independently hash artifact content."
)
_PORTABLE_DEVICE_NAMESPACE_POLICY = (
    "consistent_bijective_remap_across_complete_snapshot_set"
)
_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_SINGLE_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)
_OPTIONAL_CONFIG_NAMES = (
    "generation_config.json",
    "preprocessor_config.json",
)
_TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_CHAT_TEMPLATE_DIRECTORY = "additional_chat_templates"
_RECEIPT_LINK_CONVERGENCE_TIMEOUT_SECONDS = 30.0
_RECEIPT_LINK_CONVERGENCE_POLL_SECONDS = 0.05


def _chat_template_files(root: Path) -> set[str]:
    """Return every standalone Jinja template consumed by Transformers.

    Transformers gives a root-level ``chat_template.jinja`` priority over a
    template embedded in ``tokenizer_config.json``.  It also discovers named
    templates from ``additional_chat_templates/*.jinja`` for local tokenizers.
    Those files therefore belong to the tokenizer identity even though they
    are not vocabulary assets.
    """

    relative: set[str] = set()
    default = root / _CHAT_TEMPLATE_FILE
    if default.is_file():
        relative.add(_CHAT_TEMPLATE_FILE)
    additional = root / _CHAT_TEMPLATE_DIRECTORY
    if additional.is_dir():
        relative.update(
            str(path.relative_to(root))
            for path in additional.glob("*.jinja")
            if path.is_file()
        )
    return relative


def _strict_object_pairs(
    pairs: list[tuple[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} in {label}")
        result[key] = value
    return result


def _reject_json_constant(value: str, *, label: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} in {label}")


def _strict_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate keys."""

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _strict_object_pairs(pairs, label=label),
            parse_constant=lambda value: _reject_json_constant(value, label=label),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed


def _sha256_file(path: Path) -> str:
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


def _safe_relative_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay inside the model directory")
    path = root / candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
        # Hugging Face cache snapshots commonly use symlinks whose targets sit
        # in a sibling blob store.  A user-managed local model, however, must
        # not silently escape its declared root.
        if not path.is_symlink() or "snapshots" not in resolved_root.parts:
            raise ValueError(f"{label} resolves outside the model directory")
    return path


def _weight_files(root: Path) -> set[str]:
    indices = [name for name in _INDEX_NAMES if (root / name).is_file()]
    singles = [name for name in _SINGLE_WEIGHT_NAMES if (root / name).is_file()]
    if len(indices) + len(singles) != 1:
        raise ValueError(
            "model directory must expose exactly one supported weight artifact "
            "or index (safetensors or pytorch_model)"
        )
    if singles:
        return {singles[0]}

    index_name = indices[0]
    index = _strict_json_mapping(root / index_name, label="weight index")
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("weight index must contain a non-empty weight_map")
    shards = set()
    for tensor_name, relative in weight_map.items():
        _safe_relative_file(
            root,
            relative,
            label=f"weight_map[{tensor_name!r}]",
        )
        shards.add(str(relative))
    return {index_name, *shards}


def discover_model_artifacts(
    model_root: Path,
    *,
    complete_consumed_tokenizer_identity: bool = False,
) -> list[str]:
    """Return the deterministic artifact set for one identity generation.

    Version-one identities intentionally retain their historical boundary.
    Version two additionally binds standalone chat templates consumed by modern
    Transformers.  Keeping the switch explicit lets old evidentiary receipts
    remain verifiable while capability evaluation can require the complete
    version-two boundary.
    """

    root = Path(model_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not (root / "config.json").is_file():
        raise FileNotFoundError(root / "config.json")

    relative = {"config.json", *_weight_files(root)}
    relative.update(name for name in _OPTIONAL_CONFIG_NAMES if (root / name).is_file())
    relative.update(name for name in _TOKENIZER_NAMES if (root / name).is_file())
    if complete_consumed_tokenizer_identity:
        relative.update(_chat_template_files(root))
    relative.update(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    )
    for name in relative:
        _safe_relative_file(root, name, label="discovered artifact")
    return sorted(relative)


def build_model_artifact_manifest(
    model_root: Path,
    *,
    complete_consumed_tokenizer_identity: bool = False,
) -> dict[str, Any]:
    """Hash all discovered artifacts and return a portable identity manifest."""

    root = Path(model_root).resolve()
    rows = []
    for relative in discover_model_artifacts(
        root,
        complete_consumed_tokenizer_identity=(complete_consumed_tokenizer_identity),
    ):
        path = _safe_relative_file(root, relative, label="artifact")
        stat = path.stat()
        rows.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path),
            }
        )
    schema = (
        COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
        if complete_consumed_tokenizer_identity
        else MODEL_ARTIFACT_IDENTITY_SCHEMA
    )
    identity = {
        "schema": schema,
        "artifacts": rows,
    }
    return {
        **identity,
        "model_root": str(root),
        "artifact_set_sha256": _canonical_sha256(identity),
        "total_bytes": int(sum(row["size_bytes"] for row in rows)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "identity_boundary": (
            (
                "config, exact model weights/index, tokenizer vocabulary and chat-"
                "template assets, and local Python modeling code consumed from "
                "this directory"
            )
            if complete_consumed_tokenizer_identity
            else (
                "historical v1 boundary: config, exact model weights/index, "
                "enumerated tokenizer vocabulary assets, and local Python modeling "
                "code; standalone chat templates are not covered"
            )
        ),
    }


@dataclass(frozen=True)
class VerifiedModelArtifactIdentity:
    """A manifest that was rehashed successfully against its declared root."""

    path: Path
    file_sha256: str
    artifact_set_sha256: str
    model_root: Path
    total_bytes: int
    record: dict[str, Any]


@dataclass(frozen=True)
class _FileMutationSnapshot:
    """Path and inode identity used only between SHA-verified boundaries."""

    requested_path: str
    resolved_path: str
    symlink_target: str | None
    lstat_identity: tuple[int, int, int, int, int, int]
    stat_identity: tuple[int, int, int, int, int, int]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _path_mutation_snapshot(
    path: Path,
    *,
    require_regular_file: bool,
) -> _FileMutationSnapshot:
    """Capture replacement, symlink-retargeting, and in-place-mutation guards."""

    requested = Path(path)
    link_stat = requested.lstat()
    target_stat = requested.stat()
    if require_regular_file and not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(f"model artifact is not a regular file: {requested}")
    if not require_regular_file and not stat.S_ISDIR(target_stat.st_mode):
        raise ValueError(f"model artifact root is not a directory: {requested}")
    return _FileMutationSnapshot(
        requested_path=str(requested.absolute()),
        resolved_path=str(requested.resolve(strict=True)),
        symlink_target=(os.readlink(requested) if requested.is_symlink() else None),
        lstat_identity=_stat_identity(link_stat),
        stat_identity=_stat_identity(target_stat),
    )


def _file_mutation_snapshot(path: Path) -> _FileMutationSnapshot:
    return _path_mutation_snapshot(path, require_regular_file=True)


def _directory_mutation_snapshot(path: Path) -> _FileMutationSnapshot:
    return _path_mutation_snapshot(path, require_regular_file=False)


_STAT_IDENTITY_FIELDS = (
    "device",
    "inode",
    "mode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
)


def _stat_identity_record(
    identity: tuple[int, int, int, int, int, int],
) -> dict[str, int]:
    return dict(zip(_STAT_IDENTITY_FIELDS, identity, strict=True))


def _snapshot_record(snapshot: _FileMutationSnapshot) -> dict[str, Any]:
    return {
        "requested_path": snapshot.requested_path,
        "resolved_path": snapshot.resolved_path,
        "symlink_target": snapshot.symlink_target,
        "lstat": _stat_identity_record(snapshot.lstat_identity),
        "stat": _stat_identity_record(snapshot.stat_identity),
    }


def _parse_stat_identity(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, dict) or set(value) != set(_STAT_IDENTITY_FIELDS):
        raise ValueError(f"{label} has an invalid stat schema")
    normalized = []
    for field in _STAT_IDENTITY_FIELDS:
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{label}.{field} must be a nonnegative integer")
        normalized.append(int(item))
    return tuple(normalized)


def _snapshot_from_record(value: object, *, label: str) -> _FileMutationSnapshot:
    required = {
        "requested_path",
        "resolved_path",
        "symlink_target",
        "lstat",
        "stat",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} has an invalid mutation-snapshot schema")
    requested = value["requested_path"]
    resolved = value["resolved_path"]
    if (
        not isinstance(requested, str)
        or not requested
        or not Path(requested).is_absolute()
        or not isinstance(resolved, str)
        or not resolved
        or not Path(resolved).is_absolute()
    ):
        raise ValueError(f"{label} paths must be non-empty absolute paths")
    symlink_target = value["symlink_target"]
    if symlink_target is not None and not isinstance(symlink_target, str):
        raise ValueError(f"{label}.symlink_target must be a string or null")
    return _FileMutationSnapshot(
        requested_path=requested,
        resolved_path=resolved,
        symlink_target=symlink_target,
        lstat_identity=_parse_stat_identity(value["lstat"], label=f"{label}.lstat"),
        stat_identity=_parse_stat_identity(value["stat"], label=f"{label}.stat"),
    )


class _MutationSnapshotVerifier:
    """Compare exact snapshots across host-local filesystem device namespaces."""

    def __init__(self, *, allow_device_namespace_remap: bool) -> None:
        self.allow_device_namespace_remap = bool(allow_device_namespace_remap)
        self._source_to_observed_device: dict[int, int] = {}
        self._observed_to_source_device: dict[int, int] = {}

    def _identity_matches(
        self,
        expected: tuple[int, int, int, int, int, int],
        observed: tuple[int, int, int, int, int, int],
    ) -> bool:
        if expected[1:] != observed[1:]:
            return False
        source_device = expected[0]
        observed_device = observed[0]
        if not self.allow_device_namespace_remap:
            return source_device == observed_device
        prior_observed = self._source_to_observed_device.get(source_device)
        prior_source = self._observed_to_source_device.get(observed_device)
        if prior_observed is not None and prior_observed != observed_device:
            return False
        if prior_source is not None and prior_source != source_device:
            return False
        self._source_to_observed_device[source_device] = observed_device
        self._observed_to_source_device[observed_device] = source_device
        return True

    def matches(
        self,
        expected: _FileMutationSnapshot,
        observed: _FileMutationSnapshot,
    ) -> bool:
        if (
            expected.requested_path != observed.requested_path
            or expected.resolved_path != observed.resolved_path
            or expected.symlink_target != observed.symlink_target
        ):
            return False
        return self._identity_matches(
            expected.lstat_identity, observed.lstat_identity
        ) and self._identity_matches(expected.stat_identity, observed.stat_identity)


@dataclass(frozen=True)
class _CachedModelArtifactVerification:
    verified: VerifiedModelArtifactIdentity
    manifest_snapshot: _FileMutationSnapshot
    artifact_snapshots: tuple[tuple[str, _FileMutationSnapshot], ...]


class ModelArtifactVerificationCache:
    """Reuse SHA verification between mandatory full boundaries in one operation.

    A hit requires the exact manifest bytes, exact discovered artifact set, and
    unchanged resolved path plus lstat/stat device, inode, mode, size, mtime,
    and ctime for every artifact.  These mutation guards cheaply reject common
    drift; they do not replace a cryptographic boundary.  Evidentiary callers
    must use ``force_rehash`` after model loading and before final publication.
    """

    def __init__(self) -> None:
        self._entries: dict[Path, _CachedModelArtifactVerification] = {}
        self.full_rehashes = 0
        self.cache_hits = 0
        self.receipt_primes = 0

    def _lookup(
        self,
        source: Path,
        *,
        expected_model_root: Path | None,
        require_tokenizer_assets: bool,
        require_complete_consumed_tokenizer_identity: bool,
    ) -> VerifiedModelArtifactIdentity | None:
        entry = self._entries.get(source)
        if entry is None:
            return None
        current_record, current_file_sha256, current_manifest_snapshot = (
            _strict_json_mapping_snapshot(source, label="model artifact manifest")
        )
        verified = entry.verified
        if (
            current_record != verified.record
            or current_file_sha256 != verified.file_sha256
            or current_manifest_snapshot != entry.manifest_snapshot
        ):
            raise ValueError(
                "model artifact manifest changed during the cached operation"
            )
        root = Path(str(current_record.get("model_root", ""))).resolve()
        if root != verified.model_root:
            raise ValueError("model artifact root changed during cached verification")
        if (
            expected_model_root is not None
            and root != Path(expected_model_root).resolve()
        ):
            raise ValueError("model artifact manifest pins a different model root")
        schema = current_record.get("schema")
        if schema not in {
            MODEL_ARTIFACT_IDENTITY_SCHEMA,
            COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA,
        }:
            raise ValueError("unsupported model artifact identity schema")
        if require_complete_consumed_tokenizer_identity and schema != (
            COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
        ):
            raise ValueError(
                "capability evaluation requires a complete v2 consumed-tokenizer "
                "artifact identity"
            )
        recorded_paths = [str(row["path"]) for row in current_record["artifacts"]]
        if (
            discover_model_artifacts(
                root,
                complete_consumed_tokenizer_identity=(
                    schema == COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
                ),
            )
            != recorded_paths
        ):
            raise ValueError(
                "model artifact discovery set changed during the cached operation"
            )
        if require_tokenizer_assets and not any(
            relative in _TOKENIZER_NAMES for relative in recorded_paths
        ):
            raise ValueError(
                "model artifact identity must include tokenizer assets for "
                "capability scoring"
            )
        expected_snapshots = dict(entry.artifact_snapshots)
        for relative in recorded_paths:
            path = _safe_relative_file(root, relative, label="cached artifact")
            if _file_mutation_snapshot(path) != expected_snapshots[relative]:
                raise ValueError(
                    f"model artifact changed during the cached operation: {relative}"
                )
        self.cache_hits += 1
        return verified

    def _store(
        self,
        verified: VerifiedModelArtifactIdentity,
        *,
        expected_manifest_snapshot: _FileMutationSnapshot,
        expected_artifact_snapshots: Mapping[str, _FileMutationSnapshot],
    ) -> None:
        root = verified.model_root
        artifact_snapshots = tuple(
            (
                str(row["path"]),
                _file_mutation_snapshot(
                    _safe_relative_file(root, row["path"], label="artifact")
                ),
            )
            for row in verified.record["artifacts"]
        )
        manifest_snapshot = _file_mutation_snapshot(verified.path)
        if manifest_snapshot != expected_manifest_snapshot or dict(
            artifact_snapshots
        ) != dict(expected_artifact_snapshots):
            raise ValueError(
                "model artifacts changed while the verification receipt was "
                "being cached"
            )
        self._entries[verified.path] = _CachedModelArtifactVerification(
            verified=verified,
            manifest_snapshot=manifest_snapshot,
            artifact_snapshots=artifact_snapshots,
        )
        self.full_rehashes += 1

    def _prime_from_preverification_receipt(
        self,
        verified: VerifiedModelArtifactIdentity,
        *,
        manifest_snapshot: _FileMutationSnapshot,
        artifact_snapshots: Mapping[str, _FileMutationSnapshot],
    ) -> None:
        """Seed one operation cache from a fully validated persisted receipt."""

        expected_paths = [str(row["path"]) for row in verified.record["artifacts"]]
        if sorted(artifact_snapshots) != sorted(expected_paths):
            raise ValueError("preverification receipt artifact inventory is incomplete")
        current_manifest = _file_mutation_snapshot(verified.path)
        current_artifacts = tuple(
            (
                relative,
                _file_mutation_snapshot(
                    _safe_relative_file(
                        verified.model_root,
                        relative,
                        label="preverified artifact",
                    )
                ),
            )
            for relative in expected_paths
        )
        if current_manifest != manifest_snapshot or dict(current_artifacts) != dict(
            artifact_snapshots
        ):
            raise ValueError(
                "model artifacts changed while the preverification receipt was "
                "being admitted"
            )
        prior = self._entries.get(verified.path)
        if prior is not None and (
            prior.verified != verified
            or prior.manifest_snapshot != manifest_snapshot
            or dict(prior.artifact_snapshots) != dict(artifact_snapshots)
        ):
            raise ValueError(
                "preverification receipt differs from the operation's prior model "
                "identity"
            )
        self._entries[verified.path] = _CachedModelArtifactVerification(
            verified=verified,
            manifest_snapshot=manifest_snapshot,
            artifact_snapshots=current_artifacts,
        )
        self.receipt_primes += 1


def _strict_json_mapping_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str, _FileMutationSnapshot]:
    """Read one strict JSON object from a mutation-stable byte snapshot."""

    before = _file_mutation_snapshot(path)
    try:
        payload = path.read_bytes()
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _strict_object_pairs(pairs, label=label),
            parse_constant=lambda value: _reject_json_constant(value, label=label),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} {path}") from exc
    after = _file_mutation_snapshot(path)
    if before != after:
        raise ValueError(f"{label} changed while it was being read")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed, hashlib.sha256(payload).hexdigest(), after


def load_and_verify_model_artifact_manifest(
    manifest_path: Path,
    *,
    expected_model_root: Path | None = None,
    require_tokenizer_assets: bool = False,
    require_complete_consumed_tokenizer_identity: bool = False,
    verification_cache: ModelArtifactVerificationCache | None = None,
    force_rehash: bool = False,
) -> VerifiedModelArtifactIdentity:
    """Rehash every pinned artifact and fail closed on any drift.

    ``verification_cache`` is deliberately explicit and operation-scoped.  A
    cached hit still revalidates the complete path/inode mutation snapshot and
    discovery set.  ``force_rehash`` performs the exact SHA boundary and also
    refuses any identity change from an earlier entry in the same operation.
    ``require_complete_consumed_tokenizer_identity`` is the capability boundary:
    it rejects historical v1 manifests even though they remain valid for their
    explicitly narrower token-level provenance purpose.
    """

    source = Path(manifest_path).resolve()
    if not isinstance(force_rehash, bool):
        raise TypeError("force_rehash must be boolean")
    if not isinstance(require_complete_consumed_tokenizer_identity, bool):
        raise TypeError("require_complete_consumed_tokenizer_identity must be boolean")
    if verification_cache is not None and not isinstance(
        verification_cache, ModelArtifactVerificationCache
    ):
        raise TypeError("verification_cache must be a ModelArtifactVerificationCache")
    previous = (
        None if verification_cache is None else verification_cache._entries.get(source)
    )
    if verification_cache is not None and not force_rehash:
        cached = verification_cache._lookup(
            source,
            expected_model_root=expected_model_root,
            require_tokenizer_assets=require_tokenizer_assets,
            require_complete_consumed_tokenizer_identity=(
                require_complete_consumed_tokenizer_identity
            ),
        )
        if cached is not None:
            return cached
    record, manifest_file_sha256, manifest_snapshot = _strict_json_mapping_snapshot(
        source, label="model artifact manifest"
    )
    schema = record.get("schema") if isinstance(record, Mapping) else None
    if schema not in {
        MODEL_ARTIFACT_IDENTITY_SCHEMA,
        COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA,
    }:
        raise ValueError("unsupported model artifact identity schema")
    if require_complete_consumed_tokenizer_identity and schema != (
        COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
    ):
        raise ValueError(
            "capability evaluation requires a complete v2 consumed-tokenizer "
            "artifact identity"
        )
    root = Path(str(record.get("model_root", ""))).resolve()
    if expected_model_root is not None and root != Path(expected_model_root).resolve():
        raise ValueError("model artifact manifest pins a different model root")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("model artifact manifest has no artifacts")

    normalized = []
    hashed_snapshots: dict[str, _FileMutationSnapshot] = {}
    seen: set[str] = set()
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"artifact row {index} has an invalid schema")
        relative = str(raw["path"])
        if relative in seen:
            raise ValueError(f"duplicate model artifact {relative!r}")
        seen.add(relative)
        path = _safe_relative_file(root, relative, label=f"artifact row {index}")
        before_hash = _file_mutation_snapshot(path)
        actual_size = int(path.stat().st_size)
        if actual_size != int(raw["size_bytes"]):
            raise ValueError(f"model artifact size changed: {relative}")
        actual_sha256 = _sha256_file(path)
        after_hash = _file_mutation_snapshot(path)
        if after_hash != before_hash:
            raise ValueError(
                f"model artifact changed while it was being hashed: {relative}"
            )
        if actual_sha256 != str(raw["sha256"]):
            raise ValueError(f"model artifact sha256 changed: {relative}")
        hashed_snapshots[relative] = after_hash
        normalized.append(
            {
                "path": relative,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
            }
        )
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise ValueError("model artifacts must be sorted by relative path")
    discovered = discover_model_artifacts(
        root,
        complete_consumed_tokenizer_identity=(
            schema == COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
        ),
    )
    recorded_paths = [row["path"] for row in normalized]
    if recorded_paths != discovered:
        missing = sorted(set(discovered) - set(recorded_paths))
        extra = sorted(set(recorded_paths) - set(discovered))
        raise ValueError(
            "model artifact manifest is not the exact currently discovered set; "
            f"missing={missing}, extra={extra}"
        )
    if require_tokenizer_assets and not any(
        row["path"] in _TOKENIZER_NAMES for row in normalized
    ):
        raise ValueError(
            "model artifact identity must include tokenizer assets for capability "
            "scoring"
        )

    identity = {"schema": schema, "artifacts": normalized}
    actual_identity = _canonical_sha256(identity)
    if actual_identity != str(record.get("artifact_set_sha256", "")):
        raise ValueError("model artifact-set identity differs from the manifest")
    total_bytes = int(sum(row["size_bytes"] for row in normalized))
    if total_bytes != int(record.get("total_bytes", -1)):
        raise ValueError("model artifact total_bytes differs from the manifest")
    if _file_mutation_snapshot(source) != manifest_snapshot:
        raise ValueError("model artifact manifest changed during verification")
    for relative, expected_snapshot in hashed_snapshots.items():
        path = _safe_relative_file(root, relative, label="verified artifact")
        if _file_mutation_snapshot(path) != expected_snapshot:
            raise ValueError(
                f"model artifact changed before verification completed: {relative}"
            )
    verified = VerifiedModelArtifactIdentity(
        path=source,
        file_sha256=manifest_file_sha256,
        artifact_set_sha256=actual_identity,
        model_root=root,
        total_bytes=total_bytes,
        record=dict(record),
    )
    if previous is not None:
        prior = previous.verified
        if (
            verified.file_sha256 != prior.file_sha256
            or verified.artifact_set_sha256 != prior.artifact_set_sha256
            or verified.model_root != prior.model_root
            or verified.record != prior.record
        ):
            raise ValueError(
                "model artifact identity changed during the cached operation"
            )
    if verification_cache is not None:
        verification_cache._store(
            verified,
            expected_manifest_snapshot=manifest_snapshot,
            expected_artifact_snapshots=hashed_snapshots,
        )
    return verified


@dataclass(frozen=True)
class VerifiedModelArtifactPreverificationReceipt:
    """A persisted full-hash boundary admitted through unchanged stat guards."""

    path: Path
    file_sha256: str
    content_sha256: str
    model_identity: VerifiedModelArtifactIdentity
    manifest_snapshot: _FileMutationSnapshot
    model_root_snapshot: _FileMutationSnapshot
    artifact_snapshots: tuple[tuple[str, _FileMutationSnapshot], ...]
    record: dict[str, Any]


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: object,
    required: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        missing = sorted(required - set(value) if isinstance(value, dict) else required)
        extra = sorted(set(value) - required if isinstance(value, dict) else [])
        raise ValueError(f"{label} fields differ: missing={missing}, extra={extra}")
    return value


def _immutable_content_addressed_write(
    output_directory: Path,
    payload: bytes,
) -> tuple[Path, str]:
    directory_requested = Path(output_directory)
    if directory_requested.exists() and directory_requested.is_symlink():
        raise ValueError("preverification receipt directory may not be a symlink")
    directory_requested.mkdir(parents=True, exist_ok=True)
    directory = directory_requested.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    destination = directory / f"{file_sha256}.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to replace immutable preverification receipt {destination}"
        )
    temporary = directory / f".{file_sha256}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        directory_descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _wait_for_single_receipt_link(destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination, file_sha256


def _wait_for_single_receipt_link(
    path: Path,
    *,
    timeout_seconds: float = _RECEIPT_LINK_CONVERGENCE_TIMEOUT_SECONDS,
    poll_seconds: float = _RECEIPT_LINK_CONVERGENCE_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    link_count: Callable[[Path], int] = lambda candidate: candidate.lstat().st_nlink,
) -> None:
    """Wait for a just-unlinked publication link to converge on shared storage.

    Some distributed filesystems briefly report the pre-unlink hard-link count
    after the directory fsync.  Receipt admission still requires exactly one
    link; this bounded wait only lets that metadata converge before admission.
    """

    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("receipt link convergence timing must be positive")
    deadline = monotonic() + timeout_seconds
    while True:
        observed = link_count(path)
        if observed == 1:
            return
        now = monotonic()
        if now >= deadline:
            raise RuntimeError(
                "preverification receipt link metadata did not converge to one "
                f"link within {timeout_seconds:g} seconds; observed {observed}"
            )
        sleep(min(poll_seconds, deadline - now))


def write_model_artifact_preverification_receipt(
    manifest_path: Path,
    output_directory: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_artifact_set_sha256: str | None = None,
    expected_model_root: Path | None = None,
) -> VerifiedModelArtifactPreverificationReceipt:
    """Fully rehash one v2 model on CPU and persist its mutation snapshot.

    The returned receipt is content addressed by its exact file SHA-256 and is
    mode 0444.  It can remove repeated full-model hashing from an expensive GPU
    allocation only while every recorded mutation guard remains unchanged.
    """

    cache = ModelArtifactVerificationCache()
    verified = load_and_verify_model_artifact_manifest(
        manifest_path,
        expected_model_root=expected_model_root,
        require_tokenizer_assets=True,
        require_complete_consumed_tokenizer_identity=True,
        verification_cache=cache,
        force_rehash=True,
    )
    if expected_manifest_file_sha256 is not None and verified.file_sha256 != (
        _require_sha256(
            expected_manifest_file_sha256,
            label="expected model manifest SHA-256",
        )
    ):
        raise ValueError("model artifact manifest SHA-256 differs from expectation")
    if expected_artifact_set_sha256 is not None and verified.artifact_set_sha256 != (
        _require_sha256(
            expected_artifact_set_sha256,
            label="expected model artifact-set SHA-256",
        )
    ):
        raise ValueError("model artifact-set SHA-256 differs from expectation")
    cached = cache._entries.get(verified.path)
    if cached is None:
        raise AssertionError("full model verification did not retain its snapshots")
    root_snapshot = _directory_mutation_snapshot(verified.model_root)
    artifact_snapshot_map = dict(cached.artifact_snapshots)
    artifact_rows = []
    for row in verified.record["artifacts"]:
        relative = str(row["path"])
        artifact_rows.append(
            {
                "path": relative,
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
                "snapshot": _snapshot_record(artifact_snapshot_map[relative]),
            }
        )
    discovered = discover_model_artifacts(
        verified.model_root,
        complete_consumed_tokenizer_identity=True,
    )
    semantic = {
        "schema": MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": {
            "hash_algorithm": "sha256",
            "method": "full_sha256_of_every_v2_manifest_artifact",
            "receipt_storage": "immutable_mode_0444_content_addressed_by_file_sha256",
        },
        "device_namespace_policy": _PORTABLE_DEVICE_NAMESPACE_POLICY,
        "model": {
            "artifact_manifest": str(verified.path),
            "artifact_manifest_file_sha256": verified.file_sha256,
            "artifact_set_sha256": verified.artifact_set_sha256,
            "manifest_schema": COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA,
            "model_root": str(verified.model_root),
            "total_bytes": verified.total_bytes,
        },
        "discovery": {
            "paths": discovered,
            "paths_sha256": _canonical_sha256(discovered),
        },
        "snapshots": {
            "manifest": _snapshot_record(cached.manifest_snapshot),
            "model_root": _snapshot_record(root_snapshot),
            "artifacts": artifact_rows,
        },
        "claim_boundary": MODEL_ARTIFACT_PREVERIFICATION_CLAIM_BOUNDARY,
    }
    receipt_record = {
        **semantic,
        "receipt_content_sha256": _canonical_sha256(semantic),
    }

    # Close the small interval between the full content hash and publication of
    # the receipt.  Any model-root, manifest, discovery, symlink, or inode drift
    # makes the expensive verification unusable instead of silently stale.
    if _directory_mutation_snapshot(verified.model_root) != root_snapshot:
        raise ValueError("model root changed before receipt publication")
    if (
        discover_model_artifacts(
            verified.model_root,
            complete_consumed_tokenizer_identity=True,
        )
        != discovered
    ):
        raise ValueError(
            "model artifact discovery set changed before receipt publication"
        )
    if _file_mutation_snapshot(verified.path) != cached.manifest_snapshot:
        raise ValueError("model artifact manifest changed before receipt publication")
    for relative, expected_snapshot in cached.artifact_snapshots:
        observed = _file_mutation_snapshot(
            _safe_relative_file(verified.model_root, relative, label="artifact")
        )
        if observed != expected_snapshot:
            raise ValueError(
                f"model artifact changed before receipt publication: {relative}"
            )

    payload = (
        json.dumps(receipt_record, indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    destination, file_sha256 = _immutable_content_addressed_write(
        output_directory, payload
    )
    return load_and_verify_model_artifact_preverification_receipt(
        destination,
        expected_receipt_file_sha256=file_sha256,
        expected_manifest_path=verified.path,
        expected_manifest_file_sha256=verified.file_sha256,
        expected_artifact_set_sha256=verified.artifact_set_sha256,
        expected_model_root=verified.model_root,
    )


def load_and_verify_model_artifact_preverification_receipt(
    receipt_path: Path,
    *,
    expected_receipt_file_sha256: str,
    expected_manifest_path: Path | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_artifact_set_sha256: str | None = None,
    expected_model_root: Path | None = None,
    verification_cache: ModelArtifactVerificationCache | None = None,
) -> VerifiedModelArtifactPreverificationReceipt:
    """Admit an exact receipt only while its complete mutation state is current.

    This function intentionally does not rehash model weight contents.  It
    validates the exact SHA-256 of the receipt and manifest, the complete v2
    discovery set, and every persisted stat/symlink guard.  Call it once before
    model loading and again immediately before publication.
    """

    expected_receipt_sha = _require_sha256(
        expected_receipt_file_sha256,
        label="expected preverification receipt SHA-256",
    )
    requested = Path(receipt_path)
    if requested.is_symlink():
        raise ValueError("preverification receipt may not be a symbolic link")
    source = requested.resolve(strict=True)
    receipt_stat = source.stat()
    if (
        not stat.S_ISREG(receipt_stat.st_mode)
        or receipt_stat.st_nlink != 1
        or receipt_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError(
            "preverification receipt must be a single-link immutable regular file"
        )
    value, file_sha256, receipt_snapshot = _strict_json_mapping_snapshot(
        source, label="model artifact preverification receipt"
    )
    if file_sha256 != expected_receipt_sha:
        raise ValueError("preverification receipt SHA-256 differs from expectation")
    if source.name != f"{file_sha256}.json":
        raise ValueError("preverification receipt path is not content addressed")
    raw_schema = value.get("schema")
    if raw_schema not in {
        MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA_V1,
        MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA,
    }:
        raise ValueError("unsupported model artifact preverification receipt schema")
    required_receipt_keys = {
        "schema",
        "created_at_utc",
        "verification",
        "model",
        "discovery",
        "snapshots",
        "claim_boundary",
        "receipt_content_sha256",
    }
    if raw_schema == MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA:
        required_receipt_keys.add("device_namespace_policy")
    receipt = _require_exact_keys(
        value,
        required_receipt_keys,
        label="preverification receipt",
    )
    if not isinstance(receipt["created_at_utc"], str) or not receipt["created_at_utc"]:
        raise ValueError("preverification receipt creation time is invalid")
    portable_device_namespace = (
        raw_schema == MODEL_ARTIFACT_PREVERIFICATION_RECEIPT_SCHEMA
    )
    expected_claim_boundary = (
        MODEL_ARTIFACT_PREVERIFICATION_CLAIM_BOUNDARY
        if portable_device_namespace
        else MODEL_ARTIFACT_PREVERIFICATION_CLAIM_BOUNDARY_V1
    )
    if receipt["claim_boundary"] != expected_claim_boundary:
        raise ValueError("preverification receipt claim boundary changed")
    if portable_device_namespace and receipt["device_namespace_policy"] != (
        _PORTABLE_DEVICE_NAMESPACE_POLICY
    ):
        raise ValueError("preverification receipt device namespace policy changed")
    snapshot_verifier = _MutationSnapshotVerifier(
        allow_device_namespace_remap=portable_device_namespace
    )
    expected_content_sha = _require_sha256(
        receipt["receipt_content_sha256"],
        label="preverification receipt content SHA-256",
    )
    content = dict(receipt)
    content.pop("receipt_content_sha256")
    if _canonical_sha256(content) != expected_content_sha:
        raise ValueError("preverification receipt semantic content changed")
    verification = _require_exact_keys(
        receipt["verification"],
        {"hash_algorithm", "method", "receipt_storage"},
        label="preverification receipt verification",
    )
    if verification != {
        "hash_algorithm": "sha256",
        "method": "full_sha256_of_every_v2_manifest_artifact",
        "receipt_storage": "immutable_mode_0444_content_addressed_by_file_sha256",
    }:
        raise ValueError("preverification receipt verification contract changed")
    model = _require_exact_keys(
        receipt["model"],
        {
            "artifact_manifest",
            "artifact_manifest_file_sha256",
            "artifact_set_sha256",
            "manifest_schema",
            "model_root",
            "total_bytes",
        },
        label="preverification receipt model",
    )
    manifest_file_sha = _require_sha256(
        model["artifact_manifest_file_sha256"],
        label="receipt model manifest SHA-256",
    )
    artifact_set_sha = _require_sha256(
        model["artifact_set_sha256"],
        label="receipt model artifact-set SHA-256",
    )
    if model["manifest_schema"] != COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA:
        raise ValueError("preverification receipt must bind a complete v2 manifest")
    if (
        not isinstance(model["total_bytes"], int)
        or isinstance(model["total_bytes"], bool)
        or model["total_bytes"] < 0
    ):
        raise ValueError("preverification receipt total_bytes is invalid")
    manifest_path = Path(str(model["artifact_manifest"]))
    model_root = Path(str(model["model_root"]))
    if not manifest_path.is_absolute() or not model_root.is_absolute():
        raise ValueError("preverification receipt model paths must be absolute")
    manifest_path = manifest_path.resolve(strict=True)
    model_root = model_root.resolve(strict=True)
    if expected_manifest_path is not None and manifest_path != Path(
        expected_manifest_path
    ).resolve(strict=True):
        raise ValueError("preverification receipt binds a different manifest")
    if expected_model_root is not None and model_root != Path(
        expected_model_root
    ).resolve(strict=True):
        raise ValueError("preverification receipt binds a different model root")
    if expected_manifest_file_sha256 is not None and manifest_file_sha != (
        _require_sha256(
            expected_manifest_file_sha256,
            label="expected model manifest SHA-256",
        )
    ):
        raise ValueError("preverification receipt binds a different manifest SHA-256")
    if expected_artifact_set_sha256 is not None and artifact_set_sha != (
        _require_sha256(
            expected_artifact_set_sha256,
            label="expected model artifact-set SHA-256",
        )
    ):
        raise ValueError(
            "preverification receipt binds a different artifact-set SHA-256"
        )

    snapshots = _require_exact_keys(
        receipt["snapshots"],
        {"manifest", "model_root", "artifacts"},
        label="preverification receipt snapshots",
    )
    manifest_snapshot = _snapshot_from_record(
        snapshots["manifest"], label="receipt manifest snapshot"
    )
    root_snapshot = _snapshot_from_record(
        snapshots["model_root"], label="receipt model-root snapshot"
    )
    observed_root_snapshot = _directory_mutation_snapshot(model_root)
    if not snapshot_verifier.matches(root_snapshot, observed_root_snapshot):
        raise ValueError("model root changed since CPU preverification")
    manifest_record, observed_manifest_sha, observed_manifest_snapshot = (
        _strict_json_mapping_snapshot(manifest_path, label="model artifact manifest")
    )
    if observed_manifest_sha != manifest_file_sha:
        raise ValueError("model artifact manifest changed since CPU preverification")
    if not snapshot_verifier.matches(manifest_snapshot, observed_manifest_snapshot):
        raise ValueError(
            "model artifact manifest path or inode changed since CPU preverification"
        )
    if manifest_record.get("schema") != COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA:
        raise ValueError("preverified model artifact manifest is not v2")
    manifest_artifacts = manifest_record.get("artifacts")
    if not isinstance(manifest_artifacts, list) or not manifest_artifacts:
        raise ValueError("preverified model artifact manifest has no artifacts")
    normalized_manifest_rows = []
    for index, raw in enumerate(manifest_artifacts):
        row = _require_exact_keys(
            raw,
            {"path", "size_bytes", "sha256"},
            label=f"model manifest artifact {index}",
        )
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError("model manifest artifact path is invalid")
        if (
            not isinstance(row["size_bytes"], int)
            or isinstance(row["size_bytes"], bool)
            or row["size_bytes"] < 0
        ):
            raise ValueError("model manifest artifact size is invalid")
        normalized_manifest_rows.append(
            {
                "path": row["path"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": _require_sha256(
                    row["sha256"], label=f"model manifest artifact {index} SHA-256"
                ),
            }
        )
    if normalized_manifest_rows != sorted(
        normalized_manifest_rows, key=lambda row: row["path"]
    ) or len({row["path"] for row in normalized_manifest_rows}) != len(
        normalized_manifest_rows
    ):
        raise ValueError("model manifest artifacts are not sorted and unique")
    manifest_identity = {
        "schema": COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA,
        "artifacts": normalized_manifest_rows,
    }
    if _canonical_sha256(manifest_identity) != artifact_set_sha or (
        manifest_record.get("artifact_set_sha256") != artifact_set_sha
    ):
        raise ValueError("preverified model artifact-set identity changed")
    total_bytes = sum(row["size_bytes"] for row in normalized_manifest_rows)
    if total_bytes != int(model["total_bytes"]) or total_bytes != manifest_record.get(
        "total_bytes"
    ):
        raise ValueError("preverified model total_bytes changed")

    discovery = _require_exact_keys(
        receipt["discovery"],
        {"paths", "paths_sha256"},
        label="preverification receipt discovery",
    )
    if (
        not isinstance(discovery["paths"], list)
        or any(not isinstance(path, str) or not path for path in discovery["paths"])
        or discovery["paths"] != sorted(set(discovery["paths"]))
    ):
        raise ValueError("preverification receipt discovery paths are invalid")
    discovery_sha = _require_sha256(
        discovery["paths_sha256"], label="receipt discovery SHA-256"
    )
    if _canonical_sha256(discovery["paths"]) != discovery_sha:
        raise ValueError("preverification receipt discovery identity changed")
    current_discovery = discover_model_artifacts(
        model_root, complete_consumed_tokenizer_identity=True
    )
    manifest_paths = [row["path"] for row in normalized_manifest_rows]
    if discovery["paths"] != manifest_paths or current_discovery != manifest_paths:
        raise ValueError(
            "model artifact discovery set changed since CPU preverification"
        )

    raw_artifact_snapshots = snapshots["artifacts"]
    if not isinstance(raw_artifact_snapshots, list):
        raise ValueError("preverification receipt artifact snapshots must be a list")
    manifest_by_path = {row["path"]: row for row in normalized_manifest_rows}
    artifact_snapshots: list[tuple[str, _FileMutationSnapshot]] = []
    observed_artifact_snapshots: list[tuple[str, _FileMutationSnapshot]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_artifact_snapshots):
        row = _require_exact_keys(
            raw,
            {"path", "size_bytes", "sha256", "snapshot"},
            label=f"receipt artifact snapshot {index}",
        )
        relative = row["path"]
        if not isinstance(relative, str) or relative not in manifest_by_path:
            raise ValueError("receipt artifact snapshot has an unknown path")
        if relative in seen:
            raise ValueError("receipt artifact snapshot path is duplicated")
        seen.add(relative)
        manifest_row = manifest_by_path[relative]
        if (
            row["size_bytes"] != manifest_row["size_bytes"]
            or row["sha256"] != manifest_row["sha256"]
        ):
            raise ValueError("receipt artifact metadata differs from the manifest")
        expected_snapshot = _snapshot_from_record(
            row["snapshot"], label=f"receipt artifact {relative!r} snapshot"
        )
        artifact_path = _safe_relative_file(
            model_root, relative, label="preverified artifact"
        )
        observed_snapshot = _file_mutation_snapshot(artifact_path)
        if not snapshot_verifier.matches(expected_snapshot, observed_snapshot):
            raise ValueError(
                f"model artifact changed since CPU preverification: {relative}"
            )
        artifact_snapshots.append((relative, expected_snapshot))
        observed_artifact_snapshots.append((relative, observed_snapshot))
    if [relative for relative, _snapshot in artifact_snapshots] != manifest_paths:
        raise ValueError("preverification receipt artifact snapshots are incomplete")
    # Close the scan interval: an early artifact must not be able to move while
    # later snapshots are being checked.  This mirrors the final mutation sweep
    # at the end of a full SHA verification without rereading weight contents.
    if _directory_mutation_snapshot(model_root) != observed_root_snapshot:
        raise ValueError("model root changed during receipt verification")
    if _file_mutation_snapshot(manifest_path) != observed_manifest_snapshot:
        raise ValueError("model artifact manifest changed during receipt verification")
    if (
        discover_model_artifacts(model_root, complete_consumed_tokenizer_identity=True)
        != manifest_paths
    ):
        raise ValueError("model discovery set changed during receipt verification")
    for relative, expected_snapshot in observed_artifact_snapshots:
        artifact_path = _safe_relative_file(
            model_root, relative, label="preverified artifact"
        )
        if _file_mutation_snapshot(artifact_path) != expected_snapshot:
            raise ValueError(
                f"model artifact changed during receipt verification: {relative}"
            )
    if _file_mutation_snapshot(source) != receipt_snapshot:
        raise ValueError("preverification receipt changed while it was verified")

    verified_model = VerifiedModelArtifactIdentity(
        path=manifest_path,
        file_sha256=manifest_file_sha,
        artifact_set_sha256=artifact_set_sha,
        model_root=model_root,
        total_bytes=total_bytes,
        record=dict(manifest_record),
    )
    if verification_cache is not None:
        if not isinstance(verification_cache, ModelArtifactVerificationCache):
            raise TypeError(
                "verification_cache must be a ModelArtifactVerificationCache"
            )
        verification_cache._prime_from_preverification_receipt(
            verified_model,
            manifest_snapshot=observed_manifest_snapshot,
            artifact_snapshots=dict(observed_artifact_snapshots),
        )
    return VerifiedModelArtifactPreverificationReceipt(
        path=source,
        file_sha256=file_sha256,
        content_sha256=expected_content_sha,
        model_identity=verified_model,
        manifest_snapshot=observed_manifest_snapshot,
        model_root_snapshot=observed_root_snapshot,
        artifact_snapshots=tuple(observed_artifact_snapshots),
        record=dict(receipt),
    )


def write_model_artifact_manifest(
    model_root: Path,
    output: Path,
    *,
    complete_consumed_tokenizer_identity: bool = False,
) -> dict[str, Any]:
    """Create one manifest atomically."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = build_model_artifact_manifest(
        model_root,
        complete_consumed_tokenizer_identity=(complete_consumed_tokenizer_identity),
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("model_root", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument(
        "--complete-consumed-tokenizer-identity",
        action="store_true",
        help=(
            "write schema v2 and bind standalone chat templates consumed by "
            "Transformers; required for capability evaluation"
        ),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--model-root", type=Path, default=None)
    verify.add_argument(
        "--require-complete-consumed-tokenizer-identity",
        action="store_true",
    )
    receipt_create = subparsers.add_parser(
        "create-preverification-receipt",
        help="fully hash one v2 manifest and write a content-addressed receipt",
    )
    receipt_create.add_argument("manifest", type=Path)
    receipt_create.add_argument("output_directory", type=Path)
    receipt_create.add_argument("--model-root", type=Path, default=None)
    receipt_create.add_argument("--expected-manifest-sha256", default=None)
    receipt_create.add_argument("--expected-artifact-set-sha256", default=None)
    receipt_verify = subparsers.add_parser(
        "verify-preverification-receipt",
        help="verify a receipt and every persisted mutation guard without rehashing",
    )
    receipt_verify.add_argument("receipt", type=Path)
    receipt_verify.add_argument("--expected-receipt-sha256", required=True)
    receipt_verify.add_argument("--manifest", type=Path, default=None)
    receipt_verify.add_argument("--expected-manifest-sha256", default=None)
    receipt_verify.add_argument("--expected-artifact-set-sha256", default=None)
    receipt_verify.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "create":
        record = write_model_artifact_manifest(
            args.model_root,
            args.output,
            complete_consumed_tokenizer_identity=bool(
                args.complete_consumed_tokenizer_identity
            ),
        )
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "verify":
        verified = load_and_verify_model_artifact_manifest(
            args.manifest,
            expected_model_root=args.model_root,
            require_tokenizer_assets=bool(
                args.require_complete_consumed_tokenizer_identity
            ),
            require_complete_consumed_tokenizer_identity=bool(
                args.require_complete_consumed_tokenizer_identity
            ),
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "manifest": str(verified.path),
                    "manifest_file_sha256": verified.file_sha256,
                    "artifact_set_sha256": verified.artifact_set_sha256,
                    "model_root": str(verified.model_root),
                    "total_bytes": verified.total_bytes,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "create-preverification-receipt":
        receipt = write_model_artifact_preverification_receipt(
            args.manifest,
            args.output_directory,
            expected_manifest_file_sha256=args.expected_manifest_sha256,
            expected_artifact_set_sha256=args.expected_artifact_set_sha256,
            expected_model_root=args.model_root,
        )
        print(
            json.dumps(
                {
                    "status": "verified_and_persisted",
                    "receipt": str(receipt.path),
                    "receipt_file_sha256": receipt.file_sha256,
                    "receipt_content_sha256": receipt.content_sha256,
                    "model_manifest_file_sha256": (receipt.model_identity.file_sha256),
                    "model_artifact_set_sha256": (
                        receipt.model_identity.artifact_set_sha256
                    ),
                    "model_root": str(receipt.model_identity.model_root),
                    "total_bytes": receipt.model_identity.total_bytes,
                    "claim_boundary": receipt.record["claim_boundary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        receipt = load_and_verify_model_artifact_preverification_receipt(
            args.receipt,
            expected_receipt_file_sha256=args.expected_receipt_sha256,
            expected_manifest_path=args.manifest,
            expected_manifest_file_sha256=args.expected_manifest_sha256,
            expected_artifact_set_sha256=args.expected_artifact_set_sha256,
            expected_model_root=args.model_root,
        )
        print(
            json.dumps(
                {
                    "status": "preverification_receipt_current",
                    "receipt": str(receipt.path),
                    "receipt_file_sha256": receipt.file_sha256,
                    "model_manifest_file_sha256": (receipt.model_identity.file_sha256),
                    "model_artifact_set_sha256": (
                        receipt.model_identity.artifact_set_sha256
                    ),
                    "stat_guard_boundary": receipt.record["claim_boundary"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
