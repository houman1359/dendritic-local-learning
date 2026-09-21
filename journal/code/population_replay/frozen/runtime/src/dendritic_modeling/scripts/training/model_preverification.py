"""Cheap GPU-time model identity checks backed by a CPU full-hash receipt."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dendritic_modeling.scripts.text.model_artifact_identity import (
    load_and_verify_model_artifact_preverification_receipt,
)


def verify_transformer_model_preverification(
    config: Any,
    *,
    model_artifact_manifest: Path | None,
    model_preverification_receipt: Path | None,
    expected_model_preverification_receipt_sha256: str | None,
) -> dict[str, Any] | None:
    """Verify an all-or-none receipt triplet against the configured HF model.

    The CPU receipt already contains a complete SHA-256 pass over every model
    artifact.  This consumer repeats the receipt hash and all persisted
    path/inode/stat guards immediately around model construction or use.
    """

    values = (
        model_artifact_manifest,
        model_preverification_receipt,
        expected_model_preverification_receipt_sha256,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "model artifact manifest, preverification receipt, and receipt "
            "SHA-256 are required together"
        )
    transformer = config.model.transformer_replacement
    model_name = str(transformer.model_name).strip()
    if not model_name:
        raise ValueError("model preverification requires a local model_name")
    model_root = Path(model_name).expanduser().resolve(strict=True)
    if not model_root.is_dir():
        raise NotADirectoryError(model_root)
    receipt = load_and_verify_model_artifact_preverification_receipt(
        Path(model_preverification_receipt),
        expected_receipt_file_sha256=str(expected_model_preverification_receipt_sha256),
        expected_manifest_path=Path(model_artifact_manifest),
        expected_model_root=model_root,
    )
    model = receipt.model_identity
    return {
        "mode": "cpu_full_sha256_receipt_with_complete_stat_guards",
        "receipt_path": str(receipt.path),
        "receipt_file_sha256": receipt.file_sha256,
        "model_artifact_manifest": str(model.path),
        "model_artifact_manifest_sha256": model.file_sha256,
        "model_artifact_set_sha256": model.artifact_set_sha256,
        "model_root": str(model.model_root),
        "model_total_bytes": model.total_bytes,
        "claim_boundary": receipt.record["claim_boundary"],
    }


def require_same_preverification(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> None:
    """Fail closed if the optional receipt boundary changes across an action."""

    if before != after:
        raise RuntimeError("model preverification identity changed across execution")


__all__ = [
    "require_same_preverification",
    "verify_transformer_model_preverification",
]
