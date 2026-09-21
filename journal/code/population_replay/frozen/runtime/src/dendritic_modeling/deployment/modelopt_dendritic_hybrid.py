"""Standalone physical bundles for ModelOpt OLMo plus dendritic cells.

The ModelOpt Puzzletron checkpoint saver expects every decoder MLP to retain
three dense SwiGLU projections.  A composed ModelOpt--dendritic model no
longer has those modules, so using that saver either fails or silently ceases
to describe the physical deployment.  This module defines a source-independent
bundle instead:

* the non-replaced backbone is written to deterministic safetensors shards;
* each replacement remains a self-describing compact dendritic artifact;
* the replaced ``gate_proj``, ``up_proj``, and ``down_proj`` tensors are proven
  absent rather than counted as if they were removed;
* tied state aliases are omitted once on disk and reconstructed explicitly;
* configuration, tokenizer, and chat-template assets are copied as ordinary
  files; and
* every payload file is content addressed before an atomic directory rename.

The loader constructs the heterogeneous OLMo skeleton from the bundled
``config.json`` under ModelOpt's patcher.  It never calls ``from_pretrained``
and never opens the source checkpoint used during export.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dendritic_modeling.deployment.ledger import module_storage_ledger
from dendritic_modeling.integrations.modelopt_puzzletron_olmo3 import (
    get_registered_olmo3_puzzletron_classes,
    register_olmo3_puzzletron_adapter,
    validate_olmo3_attention_contract,
)
from dendritic_modeling.integrations.modelopt_puzzletron_recovery import (
    _load_recovery_api,
)
from dendritic_modeling.networks.architectures.replacement import (
    compiled_replacement_plan_from_mapping,
)
from dendritic_modeling.networks.architectures.transformer.patching import (
    build_population_replacement_from_compiled_plan,
)
from dendritic_modeling.scripts.text.model_artifact_identity import (
    ModelArtifactVerificationCache,
    VerifiedModelArtifactIdentity,
    load_and_verify_model_artifact_manifest,
)
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _load_replacement_state_dict,
)
from dendritic_modeling.training._transformer_replacement.model_sources import (
    normalize_transformer_model_source,
)

BUNDLE_SCHEMA = "dendritic_modelopt_olmo3_hybrid_bundle/v1"
COMPLETION_SCHEMA = "dendritic_modelopt_olmo3_hybrid_completion/v1"
LOAD_SCHEMA = "dendritic_modelopt_olmo3_hybrid_load/v1"
PARITY_SCHEMA = "dendritic_modelopt_olmo3_hybrid_bfloat16_parity/v1"
MANIFEST_NAME = "hybrid_manifest.json"
COMPLETION_NAME = "COMPLETED.json"

_SOURCE_BINDING_SCHEMA = "dendritic_modelopt_dendritic_cascade_source/v1"
_SOURCE_BINDING_STATUS = "selected_by_registered_upstream_rule_outcome_blind"
_TOKENIZER_SUPPLEMENT_SCHEMA = (
    "dendritic_modelopt_puzzletron_capability_tokenizer_identity/v2"
)
_SOURCE_HASH_BRACKET_SCHEMA = "dendritic_modelopt_hybrid_source_export_hash_bracket/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "tokenizer.model",
    "spiece.model",
)
_REQUIRED_ASSETS = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}
_CAPABILITY_TOKENIZER_FILES = {
    name
    for name in _ASSET_FILES
    if name not in {"config.json", "generation_config.json"}
}
_REQUIRED_CAPABILITY_TOKENIZER_FILES = {
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
}
_SOURCE_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class LoadedModelOptDendriticHybrid:
    """A source-free reconstructed hybrid and its verification receipt."""

    bundle: Path
    model: nn.Module = field(repr=False, compare=False)
    manifest: dict[str, Any]
    receipt: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _regular_file(path: Path, *, label: str) -> Path:
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = requested.resolve(strict=True)
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be one regular file: {resolved}")
    return resolved


def _real_directory(path: Path, *, label: str) -> Path:
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _safe_relative(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay within the bundle")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{label} escapes the bundle")
    return resolved


def _plain_json(value: Any, *, label: str) -> Any:
    """Round-trip a construction declaration through strict JSON."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be strict JSON data") from exc
    return json.loads(encoded)


def _lower_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Read one exact, unsymlinked JSON object under a caller-pinned digest."""

    expected = _lower_sha256(expected_sha256, label=f"{label} digest")
    source = _regular_file(path, label=label)
    before = source.stat()

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {label}")

    try:
        payload = source.read_bytes()
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    after = source.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"{label} changed while it was being read")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{label} SHA-256 differs from its exact binding")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return source, parsed


@dataclass
class _BoundExportSource:
    declaration: dict[str, Any]
    source: Path
    binding_path: Path
    binding_file_sha256: str
    binding: dict[str, Any]
    tokenizer_supplement_path: Path
    tokenizer_supplement_file_sha256: str
    tokenizer_supplement: dict[str, Any]
    verified_before: VerifiedModelArtifactIdentity
    verification_cache: ModelArtifactVerificationCache = field(repr=False)


def _verify_tokenizer_supplement(
    value: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_keys = {
        "schema",
        "selected_checkpoint",
        "original_teacher_checkpoint",
        "narrow_v1_model_artifact_manifest",
        "narrow_v1_model_artifact_manifest_file_sha256",
        "narrow_v1_artifact_set_sha256",
        "artifacts",
        "tokenizer_artifact_set_sha256",
        "total_bytes",
        "chat_template_included",
        "selected_matches_original_teacher_exactly",
        "capability_evaluation_admissible",
        "claim_boundary",
    }
    if set(value) != expected_keys or value.get("schema") != (
        _TOKENIZER_SUPPLEMENT_SCHEMA
    ):
        raise ValueError("capability tokenizer v2 supplement schema differs")
    source = _real_directory(
        Path(str(value["selected_checkpoint"])),
        label="tokenizer supplement selected checkpoint",
    )
    manifest = _regular_file(
        Path(str(value["narrow_v1_model_artifact_manifest"])),
        label="tokenizer supplement narrow v1 manifest",
    )
    if source != Path(str(declaration["checkpoint"])) or manifest != Path(
        str(declaration["artifact_manifest"])
    ):
        raise ValueError("tokenizer supplement points to a different model source")
    if (
        _lower_sha256(
            value["narrow_v1_model_artifact_manifest_file_sha256"],
            label="tokenizer supplement narrow-manifest digest",
        )
        != declaration["artifact_manifest_file_sha256"]
        or _lower_sha256(
            value["narrow_v1_artifact_set_sha256"],
            label="tokenizer supplement narrow artifact-set digest",
        )
        != declaration["artifact_set_sha256"]
    ):
        raise ValueError("tokenizer supplement narrow v1 identity differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("capability tokenizer v2 supplement has no asset rows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(artifacts):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"capability tokenizer asset row {index} is invalid")
        relative = row["path"]
        size = row["size_bytes"]
        if (
            not isinstance(relative, str)
            or relative not in _CAPABILITY_TOKENIZER_FILES
            or relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"capability tokenizer asset row {index} differs")
        seen.add(relative)
        normalized.append(
            {
                "path": relative,
                "size_bytes": int(size),
                "sha256": _lower_sha256(
                    row["sha256"],
                    label=f"capability tokenizer asset {relative} digest",
                ),
            }
        )
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise ValueError("capability tokenizer v2 assets are not sorted")
    missing = sorted(_REQUIRED_CAPABILITY_TOKENIZER_FILES - seen)
    if missing:
        raise ValueError(f"capability tokenizer v2 assets are incomplete: {missing}")
    identity = {
        "schema": _TOKENIZER_SUPPLEMENT_SCHEMA,
        "selected_checkpoint": value["selected_checkpoint"],
        "original_teacher_checkpoint": value["original_teacher_checkpoint"],
        "narrow_v1_model_artifact_manifest": value["narrow_v1_model_artifact_manifest"],
        "narrow_v1_model_artifact_manifest_file_sha256": value[
            "narrow_v1_model_artifact_manifest_file_sha256"
        ],
        "narrow_v1_artifact_set_sha256": value["narrow_v1_artifact_set_sha256"],
        "artifacts": normalized,
    }
    if _lower_sha256(
        value["tokenizer_artifact_set_sha256"],
        label="capability tokenizer artifact-set digest",
    ) != _canonical_sha256(identity):
        raise ValueError("capability tokenizer v2 artifact-set identity differs")
    if (
        isinstance(value["total_bytes"], bool)
        or not isinstance(value["total_bytes"], int)
        or int(value["total_bytes"])
        != sum(int(row["size_bytes"]) for row in normalized)
        or value["chat_template_included"] is not True
        or value["selected_matches_original_teacher_exactly"] is not True
        or value["capability_evaluation_admissible"] is not True
        or not isinstance(value["claim_boundary"], str)
        or not value["claim_boundary"]
    ):
        raise ValueError("capability tokenizer v2 contract differs")
    return normalized


def _prepare_bound_export_source(
    *,
    source_declaration: Mapping[str, Any],
    source_binding_path: Path,
    source_binding_file_sha256: str,
    tokenizer_supplement_path: Path,
    tokenizer_supplement_file_sha256: str,
) -> _BoundExportSource:
    declaration = normalize_transformer_model_source(source_declaration)
    binding_sha = _lower_sha256(
        source_binding_file_sha256,
        label="source-binding digest",
    )
    supplement_sha = _lower_sha256(
        tokenizer_supplement_file_sha256,
        label="tokenizer-supplement digest",
    )
    binding_path, binding = _strict_bound_json(
        source_binding_path,
        expected_sha256=binding_sha,
        label="ModelOpt cascade source binding",
    )
    if (
        binding.get("schema") != _SOURCE_BINDING_SCHEMA
        or binding.get("status") != _SOURCE_BINDING_STATUS
        or not isinstance(binding.get("model_source"), Mapping)
        or normalize_transformer_model_source(binding["model_source"]) != declaration
    ):
        raise ValueError("ModelOpt cascade source declaration differs from its binding")
    supplement_path, supplement = _strict_bound_json(
        tokenizer_supplement_path,
        expected_sha256=supplement_sha,
        label="ModelOpt capability tokenizer v2 supplement",
    )
    supplement_rows = _verify_tokenizer_supplement(
        supplement,
        declaration=declaration,
    )
    binding_supplement = binding.get("capability_tokenizer_supplement")
    if not isinstance(binding_supplement, Mapping):
        raise ValueError("source binding has no capability tokenizer supplement")
    try:
        bound_supplement_path = _regular_file(
            Path(str(binding_supplement.get("path", ""))),
            label="source-bound tokenizer supplement",
        )
    except (FileNotFoundError, OSError):
        raise ValueError("source-bound tokenizer supplement path differs") from None
    if (
        binding_supplement.get("required") is not True
        or binding_supplement.get("schema") != _TOKENIZER_SUPPLEMENT_SCHEMA
        or binding_supplement.get("chat_template_included") is not True
        or bound_supplement_path != supplement_path
        or binding_supplement.get("file_sha256") != supplement_sha
        or binding_supplement.get("tokenizer_artifact_set_sha256")
        != supplement["tokenizer_artifact_set_sha256"]
    ):
        raise ValueError("source binding capability-tokenizer identity differs")
    if not supplement_rows:
        raise AssertionError("validated tokenizer supplement unexpectedly has no rows")

    source = _real_directory(
        Path(declaration["checkpoint"]),
        label="source ModelOpt checkpoint",
    )
    cache = ModelArtifactVerificationCache()
    verified = load_and_verify_model_artifact_manifest(
        Path(declaration["artifact_manifest"]),
        expected_model_root=source,
        require_tokenizer_assets=False,
        verification_cache=cache,
        force_rehash=True,
    )
    if (
        verified.file_sha256 != declaration["artifact_manifest_file_sha256"]
        or verified.artifact_set_sha256 != declaration["artifact_set_sha256"]
        or verified.path != Path(declaration["artifact_manifest"])
        or verified.model_root != source
        or cache.full_rehashes != 1
    ):
        raise ValueError("source artifact identity differs from its declaration")
    return _BoundExportSource(
        declaration=declaration,
        source=source,
        binding_path=binding_path,
        binding_file_sha256=binding_sha,
        binding=binding,
        tokenizer_supplement_path=supplement_path,
        tokenizer_supplement_file_sha256=supplement_sha,
        tokenizer_supplement=supplement,
        verified_before=verified,
        verification_cache=cache,
    )


def _finalize_bound_export_source(
    source: _BoundExportSource,
) -> VerifiedModelArtifactIdentity:
    """Close the second full-hash boundary before publishing the bundle."""

    verified = load_and_verify_model_artifact_manifest(
        Path(source.declaration["artifact_manifest"]),
        expected_model_root=source.source,
        require_tokenizer_assets=False,
        verification_cache=source.verification_cache,
        force_rehash=True,
    )
    if (
        verified != source.verified_before
        or source.verification_cache.full_rehashes != 2
    ):
        raise ValueError("source artifact identity changed across physical export")
    binding_path, binding = _strict_bound_json(
        source.binding_path,
        expected_sha256=source.binding_file_sha256,
        label="ModelOpt cascade source binding",
    )
    supplement_path, supplement = _strict_bound_json(
        source.tokenizer_supplement_path,
        expected_sha256=source.tokenizer_supplement_file_sha256,
        label="ModelOpt capability tokenizer v2 supplement",
    )
    if (
        binding_path != source.binding_path
        or binding != source.binding
        or supplement_path != source.tokenizer_supplement_path
        or supplement != source.tokenizer_supplement
    ):
        raise ValueError("source provenance records changed across physical export")
    return verified


def _verify_copied_tokenizer_assets(
    asset_rows: Sequence[Mapping[str, Any]],
    supplement: Mapping[str, Any],
) -> None:
    expected = {
        str(row["path"]): {
            "path": str(row["path"]),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in supplement["artifacts"]
    }
    observed = {
        str(row["path"]): {
            "path": str(row["path"]),
            "size_bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in asset_rows
        if str(row["path"]) in _CAPABILITY_TOKENIZER_FILES
    }
    if observed != expected or "chat_template.jinja" not in observed:
        differing = sorted(
            name
            for name in set(observed) | set(expected)
            if observed.get(name) != expected.get(name)
        )
        raise ValueError(
            "copied tokenizer/chat assets differ from the source-bound v2 rows: "
            + ", ".join(differing)
        )


def _resolve_attr(value: object, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            raise ValueError("module paths may not contain empty components")
        current = getattr(current, part)
    return current


def _set_attr(value: object, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent = value
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        cpu = tensor.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {"dtype": str(cpu.dtype), "name": name, "shape": list(cpu.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _key_set_sha256(keys: Sequence[str] | set[str]) -> str:
    return _canonical_sha256(sorted(keys))


def _storage_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.device.type == "meta":
        raise ValueError("hybrid export cannot serialize meta tensors")
    return (
        tensor.device.type,
        tensor.device.index,
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _state_aliases(
    state: Mapping[str, torch.Tensor],
    *,
    prefer_embedding_over_lm_head: bool,
) -> tuple[dict[str, str], list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name, tensor in state.items():
        groups.setdefault(_storage_signature(tensor), []).append(name)
    aliases: dict[str, str] = {}
    canonicals: list[str] = []
    for names in groups.values():
        ordered = sorted(names)
        if (
            prefer_embedding_over_lm_head
            and "lm_head.weight" in ordered
            and "model.embed_tokens.weight" in ordered
        ):
            canonical = "model.embed_tokens.weight"
        else:
            canonical = ordered[0]
        canonicals.append(canonical)
        for name in ordered:
            if name != canonical:
                aliases[name] = canonical
    return aliases, sorted(canonicals)


def _source_safetensor_layout(
    root: Path,
    *,
    inspect_keys: set[str],
) -> tuple[
    set[str],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Read source key/shape metadata without materializing source weights."""

    try:
        from safetensors import safe_open
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("safetensors is required for hybrid deployment") from exc

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_symlink() or single_path.is_symlink():
        raise ValueError("source safetensors metadata may not be a symbolic link")
    if index_path.is_file() == single_path.is_file():
        raise ValueError("source must have exactly one safetensors layout")
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("source safetensors index is invalid") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("source safetensors index has no weight map")
        if any(not isinstance(key, str) for key in weight_map):
            raise ValueError("source safetensors index has a non-string state key")
        shard_by_key = {str(key): str(value) for key, value in weight_map.items()}
    else:
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            shard_by_key = dict.fromkeys(handle.keys(), single_path.name)

    shards: dict[str, Path] = {}
    for relative in sorted(set(shard_by_key.values())):
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source safetensors shard path is unsafe")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError("source safetensors shard may not be a symbolic link")
        shard = unresolved.resolve(strict=True)
        if root != shard and root not in shard.parents:
            raise ValueError("source safetensors shard escapes its checkpoint")
        if not shard.is_file() or shard.suffix != ".safetensors":
            raise ValueError("source safetensors shard is not a regular tensor file")
        shards[relative] = shard

    missing = sorted(inspect_keys - set(shard_by_key))
    if missing:
        raise ValueError(f"source checkpoint lacks replaced projections: {missing}")
    inspected: dict[str, dict[str, Any]] = {}
    all_metadata: dict[str, dict[str, Any]] = {}
    by_shard: dict[str, list[str]] = {}
    for key, relative in sorted(shard_by_key.items()):
        by_shard.setdefault(relative, []).append(key)
    payload_bytes_by_shard: dict[str, int] = {}
    for relative, keys in by_shard.items():
        with safe_open(shards[relative], framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            if available != set(keys):
                raise ValueError(
                    f"source safetensors index/shard coverage differs for {relative}"
                )
            for key in keys:
                tensor_slice = handle.get_slice(key)
                shape = [int(value) for value in tensor_slice.get_shape()]
                dtype = str(tensor_slice.get_dtype())
                if dtype not in _SAFETENSOR_DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype!r}")
                values = 1
                for dimension in shape:
                    values *= dimension
                record = {
                    "shape": shape,
                    "dtype": dtype,
                    "values": int(values),
                    "bytes": int(values * _SAFETENSOR_DTYPE_BYTES[dtype]),
                }
                all_metadata[key] = record
                if key in inspect_keys:
                    inspected[key] = record
        payload_bytes_by_shard[relative] = sum(
            all_metadata[key]["bytes"] for key in keys
        )
    shard_rows = [
        {
            "relative_path": relative,
            "bytes": int(path.stat().st_size),
            "tensor_payload_bytes": int(payload_bytes_by_shard[relative]),
            "state_key_count": len(by_shard[relative]),
        }
        for relative, path in sorted(shards.items())
    ]
    per_dtype_bytes: dict[str, int] = {}
    for record in all_metadata.values():
        dtype = str(record["dtype"])
        per_dtype_bytes[dtype] = per_dtype_bytes.get(dtype, 0) + int(record["bytes"])
    summary = {
        "tensor_values": int(sum(record["values"] for record in all_metadata.values())),
        "tensor_payload_bytes": int(
            sum(record["bytes"] for record in all_metadata.values())
        ),
        "per_dtype_bytes": dict(sorted(per_dtype_bytes.items())),
        "tensor_metadata": all_metadata,
    }
    return set(shard_by_key), inspected, shard_rows, summary


def _verify_declared_source_structure(
    raw_config: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any],
    physical_tensor_values: int,
) -> None:
    expected = declaration["expected_structure"]
    blocks = raw_config.get("block_configs")
    if not isinstance(blocks, list):
        raise ValueError("source block_configs are missing")
    hidden_size = raw_config.get("hidden_size")
    teacher_width = raw_config.get("intermediate_size")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 1
        or isinstance(teacher_width, bool)
        or not isinstance(teacher_width, int)
        or teacher_width < 1
    ):
        raise ValueError("source hidden/intermediate geometry is invalid")
    widths: list[int | None] = []
    no_op_layers: list[int] = []
    for layer, block in enumerate(blocks):
        ffn = block.get("ffn") if isinstance(block, Mapping) else None
        if not isinstance(ffn, Mapping):
            raise ValueError(f"source layer {layer} has no FFN declaration")
        if ffn.get("no_op") is True:
            widths.append(None)
            no_op_layers.append(layer)
            continue
        width = ffn.get("intermediate_size")
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError(f"source layer {layer} has no positive FFN width")
        widths.append(int(width))
    changed = [
        layer for layer, width in enumerate(widths) if width != int(teacher_width)
    ]
    unchanged = [layer for layer in range(len(widths)) if layer not in set(changed)]
    observed = {
        "model_type": raw_config.get("model_type"),
        "hidden_size": int(hidden_size),
        "num_layers": len(blocks),
        "teacher_intermediate_size": int(teacher_width),
        "widths_by_layer": widths,
        "no_op_layers": no_op_layers,
        "changed_ffn_layers": changed,
        "unchanged_ffn_layers": unchanged,
        "ffn_parameter_values_by_layer": [
            0 if width is None else 3 * int(hidden_size) * int(width)
            for width in widths
        ],
        "total_parameter_values": int(physical_tensor_values),
    }
    if observed != expected:
        differing = sorted(
            name for name in observed if observed[name] != expected.get(name)
        )
        raise ValueError(
            "source checkpoint structure differs from its normalized declaration: "
            + ", ".join(differing)
        )


def _copy_assets(source: Path, destination: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    present: set[str] = set()
    for name in _ASSET_FILES:
        candidate = source / name
        if candidate.is_symlink():
            raise ValueError(f"source asset must not be a symbolic link: {name}")
        if not candidate.exists():
            continue
        resolved = _regular_file(candidate, label=f"source asset {name}")
        target = destination / name
        shutil.copy2(resolved, target)
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise RuntimeError(f"asset copy did not create a regular file: {name}")
        rows.append(
            {
                "path": name,
                "bytes": int(target.stat().st_size),
                "sha256": _sha256_file(target),
            }
        )
        present.add(name)
    missing = sorted(_REQUIRED_ASSETS - present)
    if missing:
        raise ValueError(f"source lacks required deployment assets: {missing}")
    return rows


def _construction_options(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    selection = source.get("selection", {})
    if not isinstance(selection, Mapping):
        raise TypeError("transformer replacement selection must be a mapping")
    result: dict[str, Any] = {}
    for name in (
        "input_transform",
        "output_bias",
        "output_init_std",
        "output_scale",
        "pre_norm",
        "replacement_kwargs",
        "runtime_overrides",
    ):
        if name in source:
            result[name] = source[name]
    if "runtime_overrides" in selection:
        result["selection"] = {"runtime_overrides": selection["runtime_overrides"]}
    return _plain_json(result, label="transformer replacement construction options")


def _write_backbone_shards(
    root: Path,
    state: Mapping[str, torch.Tensor],
    keys: Sequence[str],
    *,
    max_shard_size_bytes: int,
) -> dict[str, Any]:
    try:
        from safetensors.torch import save_file
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("safetensors is required for hybrid deployment") from exc
    if isinstance(max_shard_size_bytes, bool) or max_shard_size_bytes < 1:
        raise ValueError("max_shard_size_bytes must be a positive integer")
    ordered = sorted(keys)
    if not ordered:
        raise ValueError("hybrid bundle has no retained backbone tensors")
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in ordered:
        size = _tensor_bytes(state[key])
        if current and current_bytes + size > int(max_shard_size_bytes):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    groups.append(current)

    backbone = root / "backbone"
    backbone.mkdir(mode=0o750)
    count = len(groups)
    weight_map: dict[str, str] = {}
    shard_rows: list[dict[str, Any]] = []
    tensor_payload_bytes = 0
    for index, group in enumerate(groups, start=1):
        name = f"model-{index:05d}-of-{count:05d}.safetensors"
        target = backbone / name
        payload = {
            key: state[key].detach().to(device="cpu").contiguous().clone()
            for key in group
        }
        payload_bytes = sum(_tensor_bytes(tensor) for tensor in payload.values())
        save_file(payload, target, metadata={"format": "pt"})
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise RuntimeError("safetensors writer produced a non-regular shard")
        for key in group:
            weight_map[key] = name
        tensor_payload_bytes += payload_bytes
        shard_rows.append(
            {
                "path": f"backbone/{name}",
                "bytes": int(target.stat().st_size),
                "sha256": _sha256_file(target),
                "tensor_payload_bytes": int(payload_bytes),
                "state_key_count": len(group),
                "state_key_set_sha256": _key_set_sha256(group),
            }
        )
        del payload
    index_payload = {
        "metadata": {"format": "pt", "total_size": int(tensor_payload_bytes)},
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    index_path = backbone / "model.safetensors.index.json"
    _write_json(index_path, index_payload)
    return {
        "index": "backbone/model.safetensors.index.json",
        "index_file_sha256": _sha256_file(index_path),
        "state_key_count": len(ordered),
        "state_key_set_sha256": _key_set_sha256(ordered),
        "tensor_payload_bytes": int(tensor_payload_bytes),
        "safetensors_file_bytes": int(sum(row["bytes"] for row in shard_rows)),
        "shards": shard_rows,
    }


def _compare_tensor_states(
    expected: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(expected) != set(observed):
        raise RuntimeError(f"{label} state keys differ")
    for name in expected:
        left = expected[name].detach().to(device="cpu")
        right = observed[name].detach().to(device="cpu")
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise RuntimeError(f"{label} tensor differs: {name}")


def _copy_and_verify_cells(
    root: Path,
    records: Sequence[Any],
    compact_cell_checkpoints: Mapping[int, Path],
) -> list[dict[str, Any]]:
    cells = root / "dendritic_cells"
    cells.mkdir(mode=0o750)
    rows: list[dict[str, Any]] = []
    expected_layers = {int(record.layer_index) for record in records}
    provided_layers = {int(layer) for layer in compact_cell_checkpoints}
    if provided_layers != expected_layers:
        raise ValueError(
            "compact-cell coverage differs: "
            f"missing={sorted(expected_layers - provided_layers)}, "
            f"extra={sorted(provided_layers - expected_layers)}"
        )
    for record in sorted(records, key=lambda item: int(item.layer_index)):
        layer = int(record.layer_index)
        replacement = record.replacement
        source = _regular_file(
            Path(compact_cell_checkpoints[layer]),
            label=f"compact cell for layer {layer}",
        )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"compact cell {layer} is not a mapping")
        if "layer_index" in payload and int(payload["layer_index"]) != layer:
            raise ValueError(f"compact cell {layer} embeds another layer")
        serialized_state = payload.get("state_dict")
        topology = payload.get("sparse_topology_manifest", [])
        if (
            not isinstance(serialized_state, Mapping)
            or not serialized_state
            or any(
                not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                for name, tensor in serialized_state.items()
            )
            or not isinstance(topology, list)
        ):
            raise ValueError(f"compact cell {layer} has invalid state or topology")
        if any(
            tensor.is_floating_point() and tensor.dtype != torch.bfloat16
            for tensor in serialized_state.values()
        ):
            raise ValueError(f"compact cell {layer} is not a BF16 deployment cell")
        compiled_plan = getattr(replacement, "compiled_replacement_plan", None)
        if not isinstance(compiled_plan, Mapping) or not compiled_plan:
            raise ValueError(f"replacement {layer} has no compiled plan")
        embedded_plan = payload.get("compiled_replacement_plan")
        if embedded_plan is not None and _plain_json(
            embedded_plan, label=f"compact cell {layer} compiled plan"
        ) != _plain_json(compiled_plan, label=f"replacement {layer} compiled plan"):
            raise ValueError(f"compact cell {layer} compiled plan differs")

        candidate = copy.deepcopy(replacement).to(device="cpu")
        _load_replacement_state_dict(
            candidate,
            serialized_state,
            topology,
            verify_semantic_identity=True,
        )
        candidate.eval()
        replacement_state = {
            name: tensor.detach().to(device="cpu")
            for name, tensor in replacement.state_dict().items()
        }
        candidate_state = dict(candidate.state_dict())
        _compare_tensor_states(
            replacement_state,
            candidate_state,
            label=f"compact cell {layer} versus installed replacement",
        )
        destination = cells / f"layer_{layer:05d}_replacement.pt"
        shutil.copy2(source, destination)
        if destination.is_symlink() or not stat.S_ISREG(destination.lstat().st_mode):
            raise RuntimeError("compact cell copy is not a regular file")
        rows.append(
            {
                "layer_index": layer,
                "path": f"dendritic_cells/{destination.name}",
                "bytes": int(destination.stat().st_size),
                "sha256": _sha256_file(destination),
                "serialized_tensor_payload_bytes": int(
                    sum(_tensor_bytes(tensor) for tensor in serialized_state.values())
                ),
                "serialized_state_sha256": _tensor_state_sha256(serialized_state),
                "runtime_state_tensor_bytes": int(
                    sum(_tensor_bytes(tensor) for tensor in candidate_state.values())
                ),
                "runtime_state_sha256": _tensor_state_sha256(candidate_state),
                "runtime_state_key_set_sha256": _key_set_sha256(set(candidate_state)),
                "topology_manifest_sha256": _canonical_sha256(topology),
                "compiled_plan": _plain_json(
                    compiled_plan, label=f"replacement {layer} compiled plan"
                ),
                "compiled_plan_sha256": _canonical_sha256(compiled_plan),
                "selection_manifest": _plain_json(
                    getattr(replacement, "selection_manifest", {}) or {},
                    label=f"replacement {layer} selection manifest",
                ),
            }
        )
        del candidate, candidate_state, payload
    return rows


def _tree_rows(root: Path, *, exclude_metadata: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"hybrid bundle contains a symbolic link: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"hybrid bundle contains a special file: {relative}")
        if exclude_metadata and relative in {MANIFEST_NAME, COMPLETION_NAME}:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": int(metadata.st_size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _runtime_storage(model: nn.Module) -> dict[str, Any]:
    state = dict(model.state_dict())
    logical_bytes = sum(_tensor_bytes(tensor) for tensor in state.values())
    aliases, canonicals = _state_aliases(
        state,
        prefer_embedding_over_lm_head=bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        ),
    )
    unique_bytes = sum(_tensor_bytes(state[name]) for name in canonicals)
    ledger = module_storage_ledger(model)
    return {
        "logical_state_tensor_bytes_including_aliases": int(logical_bytes),
        "unique_persistent_state_tensor_bytes": int(unique_bytes),
        "parameter_bytes": int(ledger["parameter_bytes"]),
        "buffer_bytes": int(ledger["buffer_bytes"]),
        "index_bytes": int(ledger["index_bytes"]),
        "module_runtime_total_bytes": int(ledger["total_bytes"]),
        "state_alias_count": len(aliases),
        "state_key_count": len(state),
        "state_key_set_sha256": _key_set_sha256(set(state)),
    }


def _replacement_runtime_storage(records: Sequence[Any]) -> dict[str, int]:
    seen: set[int] = set()
    parameter_values = 0
    parameter_bytes = 0
    buffer_bytes = 0
    index_bytes = 0
    for record in records:
        replacement = record.replacement
        if id(replacement) in seen:
            raise ValueError(
                "the v1 physical hybrid contract requires independent replacement cells"
            )
        seen.add(id(replacement))
        ledger = module_storage_ledger(replacement)
        parameter_values += sum(
            parameter.numel() for parameter in replacement.parameters()
        )
        parameter_bytes += int(ledger["parameter_bytes"])
        buffer_bytes += int(ledger["buffer_bytes"])
        index_bytes += int(ledger["index_bytes"])
    return {
        "cell_count": len(seen),
        "parameter_values": int(parameter_values),
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "index_bytes": int(index_bytes),
        "runtime_stored_bytes": int(parameter_bytes + buffer_bytes),
    }


def _source_relative_runtime_reconciliation(
    *,
    model: nn.Module,
    records: Sequence[Any],
    source_keys: set[str],
    source_tensor_metadata: Mapping[str, Mapping[str, Any]],
    omitted_keys: set[str],
    omitted_bytes: int,
    replacement_prefixes: Sequence[str],
    expected: Mapping[str, int] | None,
) -> dict[str, Any]:
    """Prove ``source - dense targets + compact cells == hybrid runtime``."""

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    backbone_parameter_names = {
        name
        for name in named_parameters
        if not any(name.startswith(prefix + ".") for prefix in replacement_prefixes)
    }
    source_parameter_keys = {
        key
        for key in source_keys
        if key in backbone_parameter_names or key in omitted_keys
    }
    unclassified = source_keys - source_parameter_keys
    source_parameter_bytes = sum(
        int(source_tensor_metadata[key]["bytes"]) for key in source_parameter_keys
    )
    source_buffer_bytes = sum(
        int(source_tensor_metadata[key]["bytes"]) for key in unclassified
    )
    source_runtime_bytes = source_parameter_bytes + source_buffer_bytes
    replacement = _replacement_runtime_storage(records)
    runtime = module_storage_ledger(model)
    retained_source_parameter_bytes = source_parameter_bytes - int(omitted_bytes)
    derived = {
        "source_parameter_bytes": int(source_parameter_bytes),
        "source_buffer_bytes": int(source_buffer_bytes),
        "source_runtime_stored_bytes": int(source_runtime_bytes),
        "removed_source_ffn_parameter_bytes": int(omitted_bytes),
        "retained_source_parameter_bytes": int(retained_source_parameter_bytes),
        "replacement_parameter_values": int(replacement["parameter_values"]),
        "replacement_parameter_bytes": int(replacement["parameter_bytes"]),
        "replacement_buffer_bytes": int(replacement["buffer_bytes"]),
        "replacement_index_bytes": int(replacement["index_bytes"]),
        "replacement_runtime_stored_bytes": int(replacement["runtime_stored_bytes"]),
        "hybrid_parameter_bytes": int(runtime["parameter_bytes"]),
        "hybrid_buffer_bytes": int(runtime["buffer_bytes"]),
        "hybrid_index_bytes": int(runtime["index_bytes"]),
        "hybrid_runtime_stored_bytes": int(runtime["total_bytes"]),
    }
    expected_parameter_bytes = (
        retained_source_parameter_bytes + replacement["parameter_bytes"]
    )
    expected_buffer_bytes = source_buffer_bytes + replacement["buffer_bytes"]
    if (
        int(runtime["parameter_bytes"]) != int(expected_parameter_bytes)
        or int(runtime["buffer_bytes"]) != int(expected_buffer_bytes)
        or int(runtime["total_bytes"])
        != int(
            source_runtime_bytes - omitted_bytes + replacement["runtime_stored_bytes"]
        )
    ):
        raise RuntimeError(
            "actual hybrid runtime does not reconcile as source minus replaced "
            "dense FFNs plus compact dendritic cells"
        )
    expected_values = dict(expected or {})
    unknown = sorted(set(expected_values) - set(derived))
    if unknown:
        raise ValueError(f"unknown expected reconciliation fields: {unknown}")
    differing = sorted(
        name
        for name, value in expected_values.items()
        if isinstance(value, bool)
        or not isinstance(value, int)
        or int(derived[name]) != int(value)
    )
    if differing:
        raise RuntimeError(
            "actual hybrid runtime differs from frozen expected accounting: "
            + ", ".join(differing)
        )
    return {
        "schema": "dendritic_modelopt_source_relative_runtime_reconciliation/v1",
        "status": "verified_actual_export",
        **derived,
        "values_only_parameter_bytes_are_separate_from_topology_buffers": True,
        "compact_topology_is_included_in_runtime_stored_bytes": True,
        "source_minus_removed_plus_replacements_equals_hybrid": True,
        "expected_accounting_checked": expected_values,
        "hybrid_stored_bytes_reduction_fraction_vs_source": float(
            1.0 - int(runtime["total_bytes"]) / int(source_runtime_bytes)
        ),
        "source_to_hybrid_stored_bytes_ratio": float(
            int(source_runtime_bytes) / int(runtime["total_bytes"])
        ),
    }


def _finalize_metadata(root: Path, manifest: dict[str, Any]) -> None:
    """Solve the small JSON-size fixed point for exact physical bundle bytes."""

    manifest_path = root / MANIFEST_NAME
    completion_path = root / COMPLETION_NAME
    storage = manifest["storage"]
    storage["physical_bundle_file_bytes"] = 0
    storage["metadata_file_bytes"] = 0
    for _ in range(12):
        _write_json(manifest_path, manifest)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "status": "complete",
            "manifest": MANIFEST_NAME,
            "manifest_file_sha256": _sha256_file(manifest_path),
            "artifact_set_sha256": manifest["artifact_set_sha256"],
        }
        _write_json(completion_path, completion)
        manifest_bytes = int(manifest_path.stat().st_size)
        completion_bytes = int(completion_path.stat().st_size)
        bundle_bytes = sum(
            row["bytes"] for row in _tree_rows(root, exclude_metadata=False)
        )
        observed = (bundle_bytes, manifest_bytes + completion_bytes)
        expected = (
            int(storage["physical_bundle_file_bytes"]),
            int(storage["metadata_file_bytes"]),
        )
        if observed == expected:
            return
        storage["physical_bundle_file_bytes"] = int(bundle_bytes)
        storage["metadata_file_bytes"] = int(manifest_bytes + completion_bytes)
    raise RuntimeError("hybrid metadata byte ledger did not converge")


def export_modelopt_dendritic_hybrid_bundle(
    *,
    model: nn.Module,
    replacement_records: Sequence[Any],
    compact_cell_checkpoints: Mapping[int, Path],
    source_declaration: Mapping[str, Any],
    source_binding_path: Path,
    source_binding_file_sha256: str,
    tokenizer_supplement_path: Path,
    tokenizer_supplement_file_sha256: str,
    output_dir: Path,
    transformer_replacement_options: Mapping[str, Any] | None = None,
    layers_path: str = "model.layers",
    max_shard_size_bytes: int = 2_000_000_000,
    expected_runtime_reconciliation: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Atomically export a source-independent ModelOpt--dendritic hybrid.

    ``model`` must already contain the replacements represented by
    ``replacement_records`` and by the supplied compact cell files.  The
    source checkpoint is used only to copy public configuration/tokenizer
    assets and to prove which dense FFN tensors are physically omitted.  Its
    normalized declaration, outcome-blind source binding, narrow v1 artifact
    identity, and chat-complete v2 tokenizer supplement are mandatory.  Two
    complete artifact rehashes bracket every read used by the physical export.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    records = list(replacement_records)
    if not records:
        raise ValueError("hybrid export requires at least one replacement")
    layers = _resolve_attr(model, layers_path)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("layers_path does not resolve to a decoder-layer sequence")
    layer_indices = [int(record.layer_index) for record in records]
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("replacement records contain duplicate layers")
    if layer_indices != sorted(layer_indices):
        records.sort(key=lambda record: int(record.layer_index))
        layer_indices = sorted(layer_indices)

    bound_source = _prepare_bound_export_source(
        source_declaration=source_declaration,
        source_binding_path=source_binding_path,
        source_binding_file_sha256=source_binding_file_sha256,
        tokenizer_supplement_path=tokenizer_supplement_path,
        tokenizer_supplement_file_sha256=tokenizer_supplement_file_sha256,
    )
    source = bound_source.source
    config_path = _regular_file(source / "config.json", label="source config")
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source config.json is invalid") from exc
    if raw_config.get("model_type") != "olmo3":
        raise ValueError("hybrid export currently requires model_type='olmo3'")
    blocks = raw_config.get("block_configs")
    if not isinstance(blocks, list) or len(blocks) != len(layers):
        raise ValueError("source block_configs do not cover every decoder layer")
    validate_olmo3_attention_contract(raw_config)

    state = dict(model.state_dict())
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model state_dict contains a non-tensor value")
    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.bfloat16}:
        raise ValueError(
            f"hybrid deployment parameters must be uniformly BF16: {parameter_dtypes}"
        )

    omitted_keys: set[str] = set()
    replacement_prefixes: list[str] = []
    architecture_rows: list[dict[str, Any]] = []
    for record in records:
        layer = int(record.layer_index)
        if layer < 0 or layer >= len(layers):
            raise IndexError(f"replacement layer {layer} is out of range")
        mlp_attr = str(getattr(record, "mlp_attr", "mlp"))
        installed = _resolve_attr(layers[layer], mlp_attr)
        if installed is not record.replacement:
            raise RuntimeError(f"replacement record {layer} is not installed in model")
        prefix = f"{layers_path}.{layer}.{mlp_attr}"
        replacement_prefixes.append(prefix)
        source_projection_keys = [
            f"{prefix}.{projection}.weight" for projection in _SOURCE_PROJECTIONS
        ]
        omitted_keys.update(source_projection_keys)
        ffn = blocks[layer].get("ffn") if isinstance(blocks[layer], Mapping) else None
        reference_width = (
            ffn.get("intermediate_size") if isinstance(ffn, Mapping) else None
        )
        if isinstance(reference_width, bool) or not isinstance(reference_width, int):
            raise ValueError(f"source layer {layer} has no positive FFN width")
        architecture_rows.append(
            {
                "layer_index": layer,
                "mlp_attr": mlp_attr,
                "state_prefix": prefix,
                "reference_intermediate_size": int(reference_width),
                "omitted_source_projection_keys": source_projection_keys,
            }
        )
    if any(key in state for key in omitted_keys):
        raise RuntimeError("composed model still contains an omitted dense FFN tensor")
    for prefix in replacement_prefixes:
        if not any(key.startswith(prefix + ".") for key in state):
            raise RuntimeError(f"replacement state is empty at {prefix}")

    (
        source_keys,
        omitted_metadata,
        source_shards,
        source_tensor_summary,
    ) = _source_safetensor_layout(source, inspect_keys=omitted_keys)
    _verify_declared_source_structure(
        raw_config,
        declaration=bound_source.declaration,
        physical_tensor_values=int(source_tensor_summary["tensor_values"]),
    )
    for row in architecture_rows:
        prefix = str(row["state_prefix"])
        source_under_prefix = {
            key for key in source_keys if key.startswith(prefix + ".")
        }
        expected_under_prefix = set(row["omitted_source_projection_keys"])
        if source_under_prefix != expected_under_prefix:
            raise RuntimeError(
                f"source FFN at layer {row['layer_index']} is not exactly gate/up/down"
            )
        row["omitted_source_projection_tensors"] = [
            {"key": key, **omitted_metadata[key]}
            for key in row["omitted_source_projection_keys"]
        ]

    replacement_state_keys = {
        key
        for key in state
        if any(key.startswith(prefix + ".") for prefix in replacement_prefixes)
    }
    backbone_logical_keys = set(state) - replacement_state_keys
    backbone_state = {key: state[key] for key in sorted(backbone_logical_keys)}
    tie_embeddings = bool(raw_config.get("tie_word_embeddings", False))
    aliases, canonical_keys = _state_aliases(
        backbone_state,
        prefer_embedding_over_lm_head=tie_embeddings,
    )
    expected_source_logical_keys = backbone_logical_keys | omitted_keys
    missing_source_keys = expected_source_logical_keys - source_keys
    unexpected_source_keys = source_keys - expected_source_logical_keys
    allowed_missing_aliases = {
        alias
        for alias, canonical in aliases.items()
        if canonical in source_keys and alias not in source_keys
    }
    if unexpected_source_keys or missing_source_keys != allowed_missing_aliases:
        raise RuntimeError(
            "source and composed backbone state coverage differ: "
            f"missing={sorted(missing_source_keys - allowed_missing_aliases)}, "
            f"unexpected={sorted(unexpected_source_keys)}"
        )
    stored_backbone_keys = sorted(set(canonical_keys))
    if set(stored_backbone_keys) | set(aliases) != backbone_logical_keys:
        raise RuntimeError("backbone alias partition is incomplete")
    alias_rows = [
        {
            "alias": alias,
            "canonical": canonical,
            "logical_bytes": _tensor_bytes(backbone_state[alias]),
        }
        for alias, canonical in sorted(aliases.items())
    ]
    if tie_embeddings:
        tied = [row for row in alias_rows if row["alias"] == "lm_head.weight"]
        if tied != [
            {
                "alias": "lm_head.weight",
                "canonical": "model.embed_tokens.weight",
                "logical_bytes": _tensor_bytes(backbone_state["lm_head.weight"]),
            }
        ]:
            raise RuntimeError("tied OLMo embeddings were not represented explicitly")

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"hybrid output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o750)
    try:
        asset_rows = _copy_assets(source, temporary)
        _verify_copied_tokenizer_assets(
            asset_rows,
            bound_source.tokenizer_supplement,
        )
        backbone = _write_backbone_shards(
            temporary,
            backbone_state,
            stored_backbone_keys,
            max_shard_size_bytes=max_shard_size_bytes,
        )
        cell_rows = _copy_and_verify_cells(temporary, records, compact_cell_checkpoints)
        cells_by_layer = {int(row["layer_index"]): row for row in cell_rows}
        for architecture in architecture_rows:
            cell = cells_by_layer[int(architecture["layer_index"])]
            architecture.update(
                {
                    "compact_cell": cell["path"],
                    "compact_cell_file_sha256": cell["sha256"],
                    "compiled_plan": cell["compiled_plan"],
                    "compiled_plan_sha256": cell["compiled_plan_sha256"],
                    "selection_manifest": cell["selection_manifest"],
                    "runtime_state_sha256": cell["runtime_state_sha256"],
                    "runtime_state_key_set_sha256": cell[
                        "runtime_state_key_set_sha256"
                    ],
                }
            )

        artifact_rows = _tree_rows(temporary, exclude_metadata=True)
        runtime = _runtime_storage(model)
        omitted_bytes = sum(row["bytes"] for row in omitted_metadata.values())
        reconciliation = _source_relative_runtime_reconciliation(
            model=model,
            records=records,
            source_keys=source_keys,
            source_tensor_metadata=source_tensor_summary["tensor_metadata"],
            omitted_keys=omitted_keys,
            omitted_bytes=omitted_bytes,
            replacement_prefixes=replacement_prefixes,
            expected=expected_runtime_reconciliation,
        )
        asset_bytes = sum(row["bytes"] for row in asset_rows)
        compact_cell_file_bytes = sum(row["bytes"] for row in cell_rows)
        compact_cell_payload_bytes = sum(
            row["serialized_tensor_payload_bytes"] for row in cell_rows
        )
        verified_after = _finalize_bound_export_source(bound_source)
        config_asset = next(row for row in asset_rows if row["path"] == "config.json")
        manifest: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "status": "complete",
            "claim_scope": "physical_artifact_fidelity_not_model_quality",
            "source_identity": {
                "model_type": "olmo3",
                "config_file_sha256": config_asset["sha256"],
                "source_binding": {
                    "schema": _SOURCE_BINDING_SCHEMA,
                    "status": _SOURCE_BINDING_STATUS,
                    "path": str(bound_source.binding_path),
                    "file_sha256": bound_source.binding_file_sha256,
                },
                "model_source": copy.deepcopy(bound_source.declaration),
                "model_artifact": {
                    "manifest": str(verified_after.path),
                    "manifest_file_sha256": verified_after.file_sha256,
                    "artifact_set_sha256": verified_after.artifact_set_sha256,
                    "total_bytes": int(verified_after.total_bytes),
                },
                "capability_tokenizer_supplement": {
                    "schema": _TOKENIZER_SUPPLEMENT_SCHEMA,
                    "path": str(bound_source.tokenizer_supplement_path),
                    "file_sha256": (bound_source.tokenizer_supplement_file_sha256),
                    "tokenizer_artifact_set_sha256": bound_source.tokenizer_supplement[
                        "tokenizer_artifact_set_sha256"
                    ],
                    "total_bytes": int(
                        bound_source.tokenizer_supplement["total_bytes"]
                    ),
                    "chat_template_included": True,
                    "asset_rows_equal_to_bundle": True,
                },
                "export_hash_bracket": {
                    "schema": _SOURCE_HASH_BRACKET_SCHEMA,
                    "full_model_artifact_rehashes": int(
                        bound_source.verification_cache.full_rehashes
                    ),
                    "same_operation_verification_cache": True,
                    "identity_unchanged": True,
                    "binding_and_tokenizer_reverified_after_payload_export": True,
                },
                "physical_safetensors_state_key_count": len(source_keys),
                "physical_safetensors_state_key_set_sha256": _key_set_sha256(
                    source_keys
                ),
                "safetensors_shards": source_shards,
                "physical_tensor_values": int(source_tensor_summary["tensor_values"]),
                "physical_tensor_payload_bytes": int(
                    source_tensor_summary["tensor_payload_bytes"]
                ),
                "physical_tensor_payload_per_dtype_bytes": source_tensor_summary[
                    "per_dtype_bytes"
                ],
                "source_path_required_for_reload": False,
            },
            "construction": {
                "layers_path": layers_path,
                "transformer_replacement_options": _construction_options(
                    transformer_replacement_options
                ),
                "floating_parameter_dtype": "torch.bfloat16",
            },
            "backbone": {
                **backbone,
                "logical_state_key_count_including_aliases": len(backbone_logical_keys),
                "logical_state_key_set_sha256": _key_set_sha256(backbone_logical_keys),
                "aliases": alias_rows,
            },
            "replacements": architecture_rows,
            "omitted_source_ffn": {
                "layer_count": len(records),
                "projection_tensor_count": len(omitted_keys),
                "projection_keys": sorted(omitted_keys),
                "tensor_values": int(
                    sum(row["values"] for row in omitted_metadata.values())
                ),
                "tensor_bytes": int(omitted_bytes),
                "all_omitted_from_backbone_safetensors": True,
                "source_ffn_prefixes_contain_exactly_three_projection_tensors": True,
            },
            "compact_cells": cell_rows,
            "assets": asset_rows,
            "storage": {
                "backbone_tensor_payload_bytes": int(backbone["tensor_payload_bytes"]),
                "backbone_safetensors_file_bytes": int(
                    backbone["safetensors_file_bytes"]
                ),
                "compact_cell_serialized_tensor_payload_bytes": int(
                    compact_cell_payload_bytes
                ),
                "compact_cell_file_bytes": int(compact_cell_file_bytes),
                "configuration_and_tokenizer_asset_bytes": int(asset_bytes),
                "payload_file_bytes_excluding_metadata": int(
                    sum(row["bytes"] for row in artifact_rows)
                ),
                "physical_bundle_file_bytes": 0,
                "metadata_file_bytes": 0,
                "runtime": runtime,
                "source_relative_runtime_reconciliation": reconciliation,
            },
            "artifacts": artifact_rows,
            "artifact_set_sha256": _canonical_sha256(artifact_rows),
            "reload_contract": {
                "source_checkpoint_opened": False,
                "skeleton_constructed_from_bundled_config": True,
                "dense_ffns_removed_before_any_weight_load": True,
                "compact_cells_loaded_separately": True,
                "tied_aliases_reconstructed_explicitly": True,
            },
        }
        _finalize_metadata(temporary, manifest)
        verified_manifest = _verify_bundle(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "schema": BUNDLE_SCHEMA,
        "status": "complete",
        "bundle": str(output),
        "manifest": str(output / MANIFEST_NAME),
        "manifest_file_sha256": _sha256_file(output / MANIFEST_NAME),
        "artifact_set_sha256": verified_manifest["artifact_set_sha256"],
        "source_identity": verified_manifest["source_identity"],
        "storage": verified_manifest["storage"],
        "omitted_source_ffn": verified_manifest["omitted_source_ffn"],
    }


def _verify_bundle(root: Path) -> dict[str, Any]:
    bundle = _real_directory(root, label="hybrid bundle")
    manifest_path = _regular_file(bundle / MANIFEST_NAME, label="hybrid manifest")
    completion_path = _regular_file(
        bundle / COMPLETION_NAME, label="hybrid completion marker"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("hybrid metadata is invalid JSON") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("hybrid manifest schema or status differs")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("hybrid source identity is missing")
    source_binding = source_identity.get("source_binding")
    model_source = source_identity.get("model_source")
    model_artifact = source_identity.get("model_artifact")
    tokenizer = source_identity.get("capability_tokenizer_supplement")
    bracket = source_identity.get("export_hash_bracket")
    if not all(
        isinstance(value, Mapping)
        for value in (source_binding, model_source, model_artifact, tokenizer, bracket)
    ):
        raise ValueError("hybrid bound source identity is incomplete")
    try:
        source_hashes = (
            _lower_sha256(
                source_identity.get("config_file_sha256"),
                label="hybrid source config digest",
            ),
            _lower_sha256(
                source_binding.get("file_sha256"),
                label="hybrid source-binding digest",
            ),
            _lower_sha256(
                model_source.get("artifact_manifest_file_sha256"),
                label="hybrid model-source manifest digest",
            ),
            _lower_sha256(
                model_source.get("artifact_set_sha256"),
                label="hybrid model-source artifact-set digest",
            ),
            _lower_sha256(
                model_artifact.get("manifest_file_sha256"),
                label="hybrid model-artifact manifest digest",
            ),
            _lower_sha256(
                model_artifact.get("artifact_set_sha256"),
                label="hybrid model-artifact-set digest",
            ),
            _lower_sha256(
                tokenizer.get("file_sha256"),
                label="hybrid tokenizer-supplement digest",
            ),
            _lower_sha256(
                tokenizer.get("tokenizer_artifact_set_sha256"),
                label="hybrid tokenizer artifact-set digest",
            ),
        )
    except ValueError as exc:
        raise ValueError("hybrid bound source digest is invalid") from exc
    if (
        source_binding.get("schema") != _SOURCE_BINDING_SCHEMA
        or source_binding.get("status") != _SOURCE_BINDING_STATUS
        or model_source.get("checkpoint") is None
        or model_source.get("artifact_manifest") != model_artifact.get("manifest")
        or source_hashes[2] != source_hashes[4]
        or source_hashes[3] != source_hashes[5]
        or isinstance(model_artifact.get("total_bytes"), bool)
        or not isinstance(model_artifact.get("total_bytes"), int)
        or int(model_artifact["total_bytes"]) < 1
        or tokenizer.get("schema") != _TOKENIZER_SUPPLEMENT_SCHEMA
        or tokenizer.get("chat_template_included") is not True
        or tokenizer.get("asset_rows_equal_to_bundle") is not True
        or isinstance(tokenizer.get("total_bytes"), bool)
        or not isinstance(tokenizer.get("total_bytes"), int)
        or int(tokenizer["total_bytes"]) < 1
        or bracket.get("schema") != _SOURCE_HASH_BRACKET_SCHEMA
        or int(bracket.get("full_model_artifact_rehashes", -1)) != 2
        or bracket.get("same_operation_verification_cache") is not True
        or bracket.get("identity_unchanged") is not True
        or bracket.get("binding_and_tokenizer_reverified_after_payload_export")
        is not True
        or source_identity.get("source_path_required_for_reload") is not False
    ):
        raise ValueError("hybrid bound source export contract differs")
    expected_completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "manifest": MANIFEST_NAME,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "artifact_set_sha256": manifest.get("artifact_set_sha256"),
    }
    if completion != expected_completion:
        raise ValueError("hybrid completion marker differs from its manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("hybrid manifest has no artifact list")
    observed_rows = _tree_rows(bundle, exclude_metadata=True)
    if observed_rows != artifacts or _canonical_sha256(observed_rows) != manifest.get(
        "artifact_set_sha256"
    ):
        raise ValueError("hybrid artifact hashes or coverage differ")
    all_rows = _tree_rows(bundle, exclude_metadata=False)
    physical_bytes = sum(row["bytes"] for row in all_rows)
    metadata_bytes = sum(
        row["bytes"]
        for row in all_rows
        if row["path"] in {MANIFEST_NAME, COMPLETION_NAME}
    )
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping) or (
        int(storage.get("physical_bundle_file_bytes", -1)) != physical_bytes
        or int(storage.get("metadata_file_bytes", -1)) != metadata_bytes
        or int(storage.get("payload_file_bytes_excluding_metadata", -1))
        != physical_bytes - metadata_bytes
    ):
        raise ValueError("hybrid physical byte ledger differs")
    expected_paths = {row["path"] for row in artifacts}
    artifacts_by_path = {row["path"]: row for row in artifacts}
    if len(artifacts_by_path) != len(artifacts):
        raise ValueError("hybrid artifact paths are not unique")
    for section in ("assets", "compact_cells"):
        rows = manifest.get(section)
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) or row.get("path") not in expected_paths
            for row in rows
        ):
            raise ValueError(f"hybrid {section} coverage differs")
    backbone = manifest.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("hybrid backbone manifest is missing")
    _safe_relative(bundle, backbone.get("index"), label="backbone index")
    for shard in backbone.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError("hybrid backbone shard declaration is invalid")
        _safe_relative(bundle, shard.get("path"), label="backbone shard")
        artifact = artifacts_by_path.get(shard.get("path"))
        if (
            artifact is None
            or int(artifact["bytes"]) != int(shard.get("bytes", -1))
            or artifact["sha256"] != shard.get("sha256")
        ):
            raise ValueError("hybrid backbone shard identity differs")
    omitted = set(manifest.get("omitted_source_ffn", {}).get("projection_keys", []))
    try:
        index = json.loads(
            _safe_relative(bundle, backbone["index"], label="backbone index").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("hybrid backbone index is invalid") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or omitted & set(weight_map):
        raise ValueError("hybrid backbone retains an omitted source FFN tensor")
    if len(weight_map) != int(backbone.get("state_key_count", -1)) or _key_set_sha256(
        set(weight_map)
    ) != backbone.get("state_key_set_sha256"):
        raise ValueError("hybrid backbone index coverage differs")
    declared_shard_names = {
        Path(str(shard["path"])).name for shard in backbone.get("shards", [])
    }
    if set(weight_map.values()) != declared_shard_names:
        raise ValueError("hybrid backbone index references undeclared shards")
    aliases = backbone.get("aliases")
    if not isinstance(aliases, list):
        raise ValueError("hybrid backbone aliases are invalid")
    stored_keys = set(weight_map)
    alias_names: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, Mapping):
            raise ValueError("hybrid backbone alias row is invalid")
        alias_name = alias.get("alias")
        canonical = alias.get("canonical")
        if (
            not isinstance(alias_name, str)
            or not isinstance(canonical, str)
            or alias_name in alias_names
            or alias_name in stored_keys
            or canonical not in stored_keys
            or isinstance(alias.get("logical_bytes"), bool)
            or not isinstance(alias.get("logical_bytes"), int)
            or int(alias["logical_bytes"]) < 1
        ):
            raise ValueError("hybrid backbone alias declaration differs")
        alias_names.add(alias_name)
    logical_keys = stored_keys | alias_names
    if len(logical_keys) != int(
        backbone.get("logical_state_key_count_including_aliases", -1)
    ) or _key_set_sha256(logical_keys) != backbone.get("logical_state_key_set_sha256"):
        raise ValueError("hybrid logical backbone key coverage differs")

    replacements = manifest.get("replacements")
    cells = manifest.get("compact_cells")
    if not isinstance(replacements, list) or not isinstance(cells, list):
        raise ValueError("hybrid replacement declarations are invalid")
    cells_by_layer: dict[int, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("hybrid compact cell declaration is invalid")
        layer = int(cell.get("layer_index", -1))
        if layer < 0 or layer in cells_by_layer:
            raise ValueError("hybrid compact cell layers are invalid")
        cells_by_layer[layer] = cell
    replacement_layers: set[int] = set()
    declared_omitted: set[str] = set()
    for row in replacements:
        if not isinstance(row, Mapping):
            raise ValueError("hybrid replacement row is invalid")
        layer = int(row.get("layer_index", -1))
        if layer < 0 or layer in replacement_layers or layer not in cells_by_layer:
            raise ValueError("hybrid replacement layer coverage differs")
        replacement_layers.add(layer)
        cell = cells_by_layer[layer]
        projection_keys = row.get("omitted_source_projection_keys")
        if (
            not isinstance(projection_keys, list)
            or len(projection_keys) != 3
            or len(set(projection_keys)) != 3
        ):
            raise ValueError("hybrid source projection omission row differs")
        declared_omitted.update(str(key) for key in projection_keys)
        if (
            row.get("compact_cell") != cell.get("path")
            or row.get("compact_cell_file_sha256") != cell.get("sha256")
            or row.get("compiled_plan") != cell.get("compiled_plan")
            or row.get("compiled_plan_sha256")
            != _canonical_sha256(row.get("compiled_plan"))
            or row.get("runtime_state_sha256") != cell.get("runtime_state_sha256")
        ):
            raise ValueError("hybrid replacement/cell identity differs")
    if replacement_layers != set(cells_by_layer) or declared_omitted != omitted:
        raise ValueError("hybrid replacement, cell, and omission coverage differ")
    omitted_manifest = manifest.get("omitted_source_ffn", {})
    if (
        int(omitted_manifest.get("layer_count", -1)) != len(replacement_layers)
        or int(omitted_manifest.get("projection_tensor_count", -1)) != len(omitted)
        or len(omitted) != 3 * len(replacement_layers)
    ):
        raise ValueError("hybrid omitted-source tensor count differs")
    return manifest


def _initialize_modelopt_olmo3_skeleton(
    bundle: Path,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct from bundled config without opening any weight checkpoint."""

    config_path = _regular_file(bundle / "config.json", label="bundled config")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if raw_config.get("model_type") != "olmo3":
        raise ValueError("hybrid loader only supports OLMo-3")
    validate_olmo3_attention_contract(raw_config)
    register_olmo3_puzzletron_adapter()
    descriptor, _ = get_registered_olmo3_puzzletron_classes()
    api = _load_recovery_api()
    config = api.load_model_config(bundle, trust_remote_code=False)
    validate_olmo3_attention_contract(config)
    with api.deci_x_patcher(
        model_descriptor=descriptor,
        block_configs=config.block_configs,
    ):
        model = api.auto_model_for_causal_lm.from_config(
            config,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
    if not isinstance(model, nn.Module):
        raise RuntimeError("ModelOpt from_config returned no model")
    # The constructor owns parameter precision. OLMo-3 deliberately computes
    # rotary frequencies in FP32, including non-persistent inv_freq buffers.
    # A blanket dtype conversion rounds those frequencies even though every
    # persistent state tensor and the deployment ledger remain unchanged.
    return model.to(device="cpu")


def _default_replacement_factory(
    row: Mapping[str, Any],
    construction_options: Mapping[str, Any],
    dtype: torch.dtype,
) -> nn.Module:
    plan = compiled_replacement_plan_from_mapping(row["compiled_plan"])
    replacement = build_population_replacement_from_compiled_plan(
        plan,
        hidden_size=int(plan.hidden_size),
        teacher_intermediate_size=int(row["reference_intermediate_size"]),
        transformer_replacement=construction_options,
        selection_manifest=dict(row.get("selection_manifest", {})),
    )
    return replacement.to(device="cpu", dtype=dtype)


def _assign_state_alias(model: nn.Module, alias: str, canonical: str) -> None:
    alias_parent_path, alias_name = alias.rsplit(".", 1)
    canonical_parent_path, canonical_name = canonical.rsplit(".", 1)
    alias_parent = _resolve_attr(model, alias_parent_path)
    canonical_parent = _resolve_attr(model, canonical_parent_path)
    if canonical_name in canonical_parent._parameters:
        value = canonical_parent._parameters[canonical_name]
        if alias_name not in alias_parent._parameters:
            raise RuntimeError("state alias parameter destination is missing")
        alias_parent._parameters[alias_name] = value
    elif canonical_name in canonical_parent._buffers:
        value = canonical_parent._buffers[canonical_name]
        if alias_name not in alias_parent._buffers:
            raise RuntimeError("state alias buffer destination is missing")
        alias_parent._buffers[alias_name] = value
    else:
        raise RuntimeError(f"state alias canonical path is missing: {canonical}")


def load_modelopt_dendritic_hybrid_bundle(
    bundle_dir: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    model_factory: Callable[[Path, torch.dtype], nn.Module] | None = None,
    replacement_factory: (
        Callable[[Mapping[str, Any], Mapping[str, Any], torch.dtype], nn.Module] | None
    ) = None,
) -> LoadedModelOptDendriticHybrid:
    """Load a verified hybrid without resolving or opening a source checkpoint."""

    if dtype is not torch.bfloat16:
        raise ValueError("the v1 hybrid deployment contract requires torch.bfloat16")
    bundle = _real_directory(bundle_dir, label="hybrid bundle")
    manifest = _verify_bundle(bundle)
    build_model = model_factory or _initialize_modelopt_olmo3_skeleton
    build_replacement = replacement_factory or _default_replacement_factory
    model = build_model(bundle, dtype)
    if not isinstance(model, nn.Module):
        raise TypeError("hybrid model factory returned no torch module")
    if any(
        parameter.is_floating_point() and parameter.dtype != dtype
        for parameter in model.parameters()
    ):
        raise RuntimeError("hybrid model factory must construct BF16 parameters")
    model = model.to(device="cpu")
    construction = manifest["construction"]
    layers_path = str(construction["layers_path"])
    layers = _resolve_attr(model, layers_path)
    options = construction["transformer_replacement_options"]

    load_receipts: list[dict[str, Any]] = []
    replacement_modules: dict[int, nn.Module] = {}
    omitted = set(manifest["omitted_source_ffn"]["projection_keys"])
    skeleton_state_keys = set(model.state_dict())
    if not omitted.issubset(skeleton_state_keys):
        raise RuntimeError("fresh skeleton lacks an omitted source FFN tensor")
    for row in manifest["replacements"]:
        layer = int(row["layer_index"])
        prefix = str(row["state_prefix"])
        observed_source = {
            key for key in skeleton_state_keys if key.startswith(prefix + ".")
        }
        if observed_source != set(row["omitted_source_projection_keys"]):
            raise RuntimeError(f"fresh skeleton FFN geometry differs at layer {layer}")
        replacement = build_replacement(row, options, dtype)
        if not isinstance(replacement, nn.Module):
            raise TypeError("hybrid replacement factory returned no torch module")
        replacement = replacement.to(device="cpu", dtype=dtype)
        _set_attr(layers[layer], str(row["mlp_attr"]), replacement)
        replacement_modules[layer] = replacement

    # Compact topology reconstruction replaces train-time modules and can remove
    # training-only keys (for example rewire_step). Validate the realized cells
    # before comparing the complete deployment key set.
    cells_by_layer = {int(row["layer_index"]): row for row in manifest["compact_cells"]}
    for architecture in manifest["replacements"]:
        layer = int(architecture["layer_index"])
        cell = cells_by_layer[layer]
        cell_path = _safe_relative(bundle, cell["path"], label="compact cell")
        payload = torch.load(cell_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict")
        topology = payload.get("sparse_topology_manifest", [])
        if not isinstance(state, Mapping) or not isinstance(topology, list):
            raise RuntimeError("compact cell payload differs during reload")
        embedded_plan = payload.get("compiled_replacement_plan")
        if (
            _tensor_state_sha256(state) != cell["serialized_state_sha256"]
            or _canonical_sha256(topology) != cell["topology_manifest_sha256"]
            or (
                embedded_plan is not None
                and _plain_json(
                    embedded_plan,
                    label=f"compact cell {layer} embedded compiled plan",
                )
                != architecture["compiled_plan"]
            )
        ):
            raise RuntimeError(
                f"compact cell serialized identity differs at layer {layer}"
            )
        receipt = _load_replacement_state_dict(
            replacement_modules[layer],
            state,
            topology,
            verify_semantic_identity=True,
        )
        realized = dict(replacement_modules[layer].state_dict())
        if (
            _tensor_state_sha256(realized) != cell["runtime_state_sha256"]
            or _key_set_sha256(set(realized)) != cell["runtime_state_key_set_sha256"]
        ):
            raise RuntimeError(f"reloaded compact cell differs at layer {layer}")
        load_receipts.append({"layer_index": layer, "semantic_load": receipt})

    for alias in manifest["backbone"]["aliases"]:
        _assign_state_alias(model, str(alias["alias"]), str(alias["canonical"]))
    patched_keys = set(model.state_dict())
    if omitted & patched_keys:
        raise RuntimeError("patched hybrid still contains an omitted dense FFN tensor")
    if (
        _key_set_sha256(patched_keys)
        != manifest["storage"]["runtime"]["state_key_set_sha256"]
    ):
        raise RuntimeError("patched hybrid state-key set differs from export")

    try:
        from safetensors.torch import load_file
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("safetensors is required for hybrid deployment") from exc
    index_path = _safe_relative(
        bundle, manifest["backbone"]["index"], label="backbone index"
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(key): str(value) for key, value in index["weight_map"].items()}
    loaded_backbone: set[str] = set()
    for shard_row in manifest["backbone"]["shards"]:
        shard = _safe_relative(bundle, shard_row["path"], label="backbone shard")
        expected_keys = {key for key, name in weight_map.items() if name == shard.name}
        shard_state = load_file(shard, device="cpu")
        if set(shard_state) != expected_keys:
            raise RuntimeError("hybrid backbone shard key coverage differs")
        incompatible = model.load_state_dict(shard_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError("hybrid backbone has unexpected state keys")
        loaded_backbone.update(shard_state)
        del shard_state
    if loaded_backbone != set(weight_map):
        raise RuntimeError("hybrid backbone did not load every indexed tensor")

    final_state = dict(model.state_dict())
    for alias in manifest["backbone"]["aliases"]:
        alias_name = str(alias["alias"])
        canonical = str(alias["canonical"])
        if _storage_signature(final_state[alias_name]) != _storage_signature(
            final_state[canonical]
        ) or not torch.equal(final_state[alias_name], final_state[canonical]):
            raise RuntimeError(
                f"hybrid state alias was not reconstructed: {alias_name}"
            )
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in model.parameters()
    ):
        raise RuntimeError("reloaded hybrid parameters are not uniformly BF16")
    runtime = _runtime_storage(model)
    if runtime != manifest["storage"]["runtime"]:
        raise RuntimeError("reloaded hybrid runtime-storage ledger differs")
    target_device = torch.device(device)
    model.to(device=target_device)
    model.eval()
    receipt = {
        "schema": LOAD_SCHEMA,
        "status": "verified",
        "bundle_manifest_file_sha256": _sha256_file(bundle / MANIFEST_NAME),
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "source_checkpoint_opened": False,
        "from_pretrained_called": False,
        "dtype": str(dtype),
        "device": str(target_device),
        "omitted_source_projection_tensor_count": len(omitted),
        "backbone_state_key_count": len(loaded_backbone),
        "compact_cell_loads": load_receipts,
        "tied_alias_count": len(manifest["backbone"]["aliases"]),
        "runtime_storage": runtime,
    }
    return LoadedModelOptDendriticHybrid(
        bundle=bundle,
        model=model,
        manifest=manifest,
        receipt=receipt,
    )


def verify_exact_bfloat16_logit_parity(
    reference_model: nn.Module,
    reloaded_model: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Require bitwise-equal short-sequence BF16 logits from two hybrids."""

    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise TypeError("parity input_ids must be a rank-two int64 tensor")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
        raise ValueError("parity input_ids must contain a non-empty token sequence")
    models = (reference_model, reloaded_model)
    devices: list[torch.device] = []
    for model in models:
        floating = {
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
        if floating != {torch.bfloat16}:
            raise RuntimeError(f"parity model is not uniformly BF16: {floating}")
        first = next(model.parameters(), None)
        if first is None:
            raise RuntimeError("parity model has no parameters")
        devices.append(first.device)
        model.eval()
    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for model, device in zip(models, devices, strict=True):
            output = model(input_ids=input_ids.to(device=device), use_cache=False)
            value = getattr(output, "logits", None)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
                raise RuntimeError("parity model did not return BF16 logits")
            if not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError("parity model returned non-finite logits")
            logits.append(value.detach().to(device="cpu").contiguous())
    maximum = float((logits[0].float() - logits[1].float()).abs().max().item())
    if not torch.equal(logits[0], logits[1]):
        raise RuntimeError(f"hybrid BF16 logit parity failed: max_abs={maximum}")
    input_cpu = input_ids.detach().to(device="cpu").contiguous()
    return {
        "schema": PARITY_SCHEMA,
        "status": "verified",
        "input_shape": list(input_cpu.shape),
        "input_sha256": _tensor_state_sha256({"input_ids": input_cpu}),
        "logits_shape": list(logits[0].shape),
        "logits_sha256": _tensor_state_sha256({"logits": logits[0]}),
        "exact_logits_equal": True,
        "maximum_absolute_difference": 0.0,
        "claim_scope": "artifact_fidelity_only_not_model_quality_or_speed",
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "COMPLETION_SCHEMA",
    "LOAD_SCHEMA",
    "PARITY_SCHEMA",
    "LoadedModelOptDendriticHybrid",
    "export_modelopt_dendritic_hybrid_bundle",
    "load_modelopt_dendritic_hybrid_bundle",
    "verify_exact_bfloat16_logit_parity",
]
