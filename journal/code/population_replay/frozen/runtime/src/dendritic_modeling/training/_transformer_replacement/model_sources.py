"""Provenance-bound external model sources for transformer replacement.

The ordinary replacement path reconstructs a homogeneous Hugging Face model
with ``AutoModelForCausalLM.from_pretrained``.  That is not sufficient for a
realized NVIDIA ModelOpt/Puzzletron checkpoint: its OLMo-3 ``block_configs``
encode a different FFN width at each layer.  This module provides an optional,
strict source session that reconstructs that heterogeneous model through the
version-pinned bridge while keeping the historical Hugging Face path unchanged.

A session performs one complete SHA-256 verification before any load, permits
teacher and student to reuse only mutation-checked metadata between loads, and
then performs a second complete SHA-256 verification.  This brackets the model
construction operation without hashing a multi-gigabyte artifact twice per
model copy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dendritic_modeling.scripts.text.model_artifact_identity import (
    ModelArtifactVerificationCache,
    VerifiedModelArtifactIdentity,
    load_and_verify_model_artifact_manifest,
)

MODEL_SOURCE_SCHEMA = "dendritic_transformer_model_source/v1"
MODELOPT_PUZZLETRON_OLMO3_KIND = "modelopt_puzzletron_olmo3"
MODEL_SOURCE_RECEIPT_SCHEMA = "dendritic_transformer_model_source_receipt/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KEYS = {
    "schema",
    "kind",
    "checkpoint",
    "artifact_manifest",
    "artifact_manifest_file_sha256",
    "artifact_set_sha256",
    "expected_structure",
}
_STRUCTURE_KEYS = {
    "model_type",
    "hidden_size",
    "num_layers",
    "teacher_intermediate_size",
    "widths_by_layer",
    "no_op_layers",
    "changed_ffn_layers",
    "unchanged_ffn_layers",
    "ffn_parameter_values_by_layer",
    "total_parameter_values",
}


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


def _lower_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _regular_unsymlinked_path(
    value: object,
    *,
    label: str,
    directory: bool,
) -> Path:
    requested = Path(str(value)).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = requested.resolve(strict=True)
    if directory:
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
    elif not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _normalize_structure(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STRUCTURE_KEYS:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            "model source expected_structure keys drifted: "
            f"{sorted(observed ^ _STRUCTURE_KEYS)}"
        )
    integer_fields = (
        "hidden_size",
        "num_layers",
        "teacher_intermediate_size",
        "total_parameter_values",
    )
    normalized: dict[str, Any] = {"model_type": str(value["model_type"])}
    for name in integer_fields:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"model source expected_structure.{name} is invalid")
        normalized[name] = int(raw)

    widths = value["widths_by_layer"]
    if not isinstance(widths, (list, tuple)):
        raise TypeError("model source widths_by_layer must be a sequence")
    normalized_widths: list[int | None] = []
    for width in widths:
        if width is None:
            normalized_widths.append(None)
        elif isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("model source contains an invalid FFN width")
        else:
            normalized_widths.append(int(width))
    normalized["widths_by_layer"] = normalized_widths

    for name in (
        "no_op_layers",
        "changed_ffn_layers",
        "unchanged_ffn_layers",
        "ffn_parameter_values_by_layer",
    ):
        raw_values = value[name]
        if not isinstance(raw_values, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw_values
        ):
            raise ValueError(f"model source expected_structure.{name} is invalid")
        normalized[name] = [int(item) for item in raw_values]

    n_layers = normalized["num_layers"]
    if (
        len(normalized_widths) != n_layers
        or len(normalized["ffn_parameter_values_by_layer"]) != n_layers
    ):
        raise ValueError("model source expected structure does not cover every layer")
    layer_partition = set(normalized["changed_ffn_layers"]) | set(
        normalized["unchanged_ffn_layers"]
    )
    if (
        set(normalized["changed_ffn_layers"]) & set(normalized["unchanged_ffn_layers"])
        or layer_partition != set(range(n_layers))
        or any(layer >= n_layers for layer in normalized["no_op_layers"])
    ):
        raise ValueError("model source changed/unchanged layer partition is invalid")
    if normalized["model_type"] != "olmo3":
        raise ValueError("ModelOpt model source currently supports only OLMo-3")
    return normalized


def normalize_transformer_model_source(value: object) -> dict[str, Any]:
    """Validate and normalize one external model-source declaration."""

    if not isinstance(value, Mapping) or set(value) != _SOURCE_KEYS:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            "model_source keys drifted: " f"{sorted(observed ^ _SOURCE_KEYS)}"
        )
    if value["schema"] != MODEL_SOURCE_SCHEMA:
        raise ValueError("unsupported transformer model_source schema")
    if value["kind"] != MODELOPT_PUZZLETRON_OLMO3_KIND:
        raise ValueError("unsupported transformer model_source kind")
    checkpoint = _regular_unsymlinked_path(
        value["checkpoint"], label="model source checkpoint", directory=True
    )
    manifest = _regular_unsymlinked_path(
        value["artifact_manifest"],
        label="model source artifact manifest",
        directory=False,
    )
    expected_structure = _normalize_structure(value["expected_structure"])
    return {
        "schema": MODEL_SOURCE_SCHEMA,
        "kind": MODELOPT_PUZZLETRON_OLMO3_KIND,
        "checkpoint": str(checkpoint),
        "artifact_manifest": str(manifest),
        "artifact_manifest_file_sha256": _lower_sha256(
            value["artifact_manifest_file_sha256"],
            label="model source artifact manifest digest",
        ),
        "artifact_set_sha256": _lower_sha256(
            value["artifact_set_sha256"],
            label="model source artifact-set digest",
        ),
        "expected_structure": expected_structure,
    }


def _structure_identity(structure: object) -> dict[str, Any]:
    raw = dataclasses.asdict(structure)
    return _normalize_structure({name: raw[name] for name in _STRUCTURE_KEYS})


class TransformerModelSourceSession:
    """Load immutable external teacher/student copies within one SHA bracket."""

    def __init__(self, declaration: Mapping[str, Any]):
        self.declaration = normalize_transformer_model_source(declaration)
        self._cache = ModelArtifactVerificationCache()
        self._verified_before: VerifiedModelArtifactIdentity | None = None
        self._verified_after: VerifiedModelArtifactIdentity | None = None
        self._structure: dict[str, Any] | None = None
        self._loads = 0
        self._finalized = False

    def _verify(self, *, force_rehash: bool) -> VerifiedModelArtifactIdentity:
        verified = load_and_verify_model_artifact_manifest(
            Path(self.declaration["artifact_manifest"]),
            expected_model_root=Path(self.declaration["checkpoint"]),
            require_tokenizer_assets=False,
            verification_cache=self._cache,
            force_rehash=force_rehash,
        )
        if verified.file_sha256 != self.declaration["artifact_manifest_file_sha256"]:
            raise ValueError("model source artifact-manifest SHA-256 drifted")
        if verified.artifact_set_sha256 != self.declaration["artifact_set_sha256"]:
            raise ValueError("model source artifact-set SHA-256 drifted")
        return verified

    def load(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initialization: str,
    ) -> nn.Module:
        """Reconstruct one exact model copy through the declared backend."""

        if self._finalized:
            raise RuntimeError("model source session was already finalized")
        normalized_initialization = str(initialization).strip().lower()
        if normalized_initialization not in {"pretrained", "checkpoint"}:
            raise ValueError(
                "external transformer model sources require pretrained initialization"
            )
        if self._verified_before is None:
            self._verified_before = self._verify(force_rehash=True)
        else:
            self._verify(force_rehash=False)

        if self.declaration["kind"] != MODELOPT_PUZZLETRON_OLMO3_KIND:
            raise AssertionError("validated external model-source kind drifted")
        from dendritic_modeling.integrations.modelopt_puzzletron_recovery import (
            load_olmo3_puzzletron_checkpoint,
        )

        loaded = load_olmo3_puzzletron_checkpoint(
            Path(self.declaration["checkpoint"]),
            dtype=dtype,
            device=device,
        )
        observed_structure = _structure_identity(loaded.structure)
        if observed_structure != self.declaration["expected_structure"]:
            differing = sorted(
                name
                for name in _STRUCTURE_KEYS
                if observed_structure[name]
                != self.declaration["expected_structure"][name]
            )
            raise RuntimeError(
                "realized external model structure drifted: " + ", ".join(differing)
            )
        if self._structure is not None and observed_structure != self._structure:
            raise RuntimeError("teacher and student external structures differ")
        self._structure = observed_structure
        self._loads += 1
        if hasattr(loaded.model.config, "use_cache"):
            loaded.model.config.use_cache = False
        return loaded.model

    def finalize(self) -> dict[str, Any]:
        """Close the SHA bracket and return a serializable source receipt."""

        if self._finalized:
            raise RuntimeError("model source session was already finalized")
        if self._loads <= 0 or self._verified_before is None or self._structure is None:
            raise RuntimeError("model source session loaded no model")
        self._verified_after = self._verify(force_rehash=True)
        if self._verified_after != self._verified_before:
            raise RuntimeError("model source identity changed across construction")
        self._finalized = True
        return self.receipt

    @property
    def receipt(self) -> dict[str, Any]:
        if not self._finalized or self._verified_after is None:
            raise RuntimeError("model source receipt is unavailable before finalize")
        return {
            "schema": MODEL_SOURCE_RECEIPT_SCHEMA,
            "status": "verified_before_and_after_model_construction",
            "kind": self.declaration["kind"],
            "checkpoint": self.declaration["checkpoint"],
            "artifact_manifest": self.declaration["artifact_manifest"],
            "artifact_manifest_file_sha256": self._verified_after.file_sha256,
            "artifact_set_sha256": self._verified_after.artifact_set_sha256,
            "artifact_total_bytes": int(self._verified_after.total_bytes),
            "structure": dict(self._structure or {}),
            "structure_sha256": _canonical_sha256(self._structure),
            "model_copies_loaded": int(self._loads),
            "full_sha256_boundaries": int(self._cache.full_rehashes),
            "mutation_checked_cache_hits": int(self._cache.cache_hits),
            "claim_boundary": (
                "This receipt proves source identity and realized architecture only; "
                "it contains no recovery-quality, compression-benefit, runtime, "
                "energy, or hardware result."
            ),
        }


__all__ = [
    "MODELOPT_PUZZLETRON_OLMO3_KIND",
    "MODEL_SOURCE_RECEIPT_SCHEMA",
    "MODEL_SOURCE_SCHEMA",
    "TransformerModelSourceSession",
    "normalize_transformer_model_source",
]
