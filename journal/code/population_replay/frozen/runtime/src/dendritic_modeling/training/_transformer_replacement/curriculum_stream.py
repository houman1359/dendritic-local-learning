"""Teacher-loss-ordered curriculum streaming for KD recovery corpora.

Puro-2B (arXiv 2608.27370) measured a 1.65x cost-efficiency gain from
ordering pretraining data low-to-high quality.  This module adapts the
ordering ingredient (and only that ingredient) to knowledge-distillation
recovery: the natural per-window quality analog under a fixed teacher is
the TEACHER LOSS, so windows are streamed easy-to-hard (or reversed) in
the exact order recorded by a sealed scoring manifest produced by
``dendritic_modeling.scripts.text.score_recovery_windows``.

The mode is default-off (``text.curriculum_order_manifest`` empty) and
fail-closed: the manifest must seal-verify, its windows tensor must match
both recorded SHA-256 digests, its scores must cover every window exactly
once, and its recorded corpus identity must equal the live text config.
Ordering is deterministic and epoch-stable -- every epoch replays the
same manifest order, and DDP ranks consume disjoint strided columns so
the global consumption order still follows the curriculum.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.networks.checkpoints import sha256_file
from dendritic_modeling.scripts.text.frozen_text_windows import (
    _tensor_content_sha256,
)
from dendritic_modeling.utils.exact_file_inventory import canonical_sha256

from .token_stream import FrozenWindowBatchSource, resolve_data_files

CURRICULUM_MANIFEST_SCHEMA = "dendritic_curriculum_window_manifest/v1"
CURRICULUM_SHARD_RECEIPT_SCHEMA = "dendritic_curriculum_window_shard/v1"
CURRICULUM_DIRECTIONS = ("low_to_high", "high_to_low")

# Corpus-identity fields recorded at scoring time and re-checked against the
# live text config before any curriculum window is streamed.
_CORPUS_IDENTITY_FIELDS = (
    "data_files",
    "split",
    "text_field",
    "tokenizer_name",
    "sequence_length",
    "separator_token_id",
    "max_tokens_per_document",
    "max_documents",
)


def normalized_data_files(value: Any) -> list[str]:
    """Normalize a ``text.data_files`` value to a comparable list of strings."""

    resolved = resolve_data_files(value)
    if resolved is None:
        return []
    if isinstance(resolved, str):
        return [resolved]
    return [str(item) for item in resolved]


def seal_curriculum_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` carrying its canonical-JSON seal."""

    body = {key: payload[key] for key in payload if key != "manifest_sha256"}
    sealed = dict(body)
    sealed["manifest_sha256"] = canonical_sha256(body)
    return sealed


def _verify_manifest_seal(manifest: Mapping[str, Any], *, path: Path) -> None:
    recorded = str(manifest.get("manifest_sha256", "") or "")
    body = {key: manifest[key] for key in manifest if key != "manifest_sha256"}
    try:
        computed = canonical_sha256(body)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"curriculum manifest is not canonically hashable: {path}"
        ) from exc
    if not recorded or computed != recorded:
        raise ValueError(f"curriculum manifest seal differs from content: {path}")


def load_verified_curriculum_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Load one sealed curriculum manifest and its verified windows tensor.

    Fail-closed: any seal, digest, shape, or coverage mismatch raises.
    """

    manifest_path = Path(str(path)).expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read curriculum manifest {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"curriculum manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema") != CURRICULUM_MANIFEST_SCHEMA:
        raise ValueError(
            f"curriculum manifest schema drifted in {manifest_path}: "
            f"{manifest.get('schema')!r}"
        )
    _verify_manifest_seal(manifest, path=manifest_path)

    window_count = int(manifest.get("window_count", 0))
    sequence_length = int(manifest.get("sequence_length", 0))
    if window_count < 1 or sequence_length < 1:
        raise ValueError(f"curriculum manifest geometry is invalid: {manifest_path}")

    tensor_meta = manifest.get("windows_tensor")
    if not isinstance(tensor_meta, Mapping) or not str(
        tensor_meta.get("file_name", "") or ""
    ):
        raise ValueError(
            f"curriculum manifest lacks a windows_tensor entry: {manifest_path}"
        )
    tensor_path = (manifest_path.parent / str(tensor_meta["file_name"])).resolve(
        strict=True
    )
    if sha256_file(tensor_path) != str(tensor_meta.get("file_sha256", "") or ""):
        raise ValueError(
            f"curriculum windows tensor file digest differs: {tensor_path}"
        )
    try:
        input_ids = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"cannot load curriculum windows tensor {tensor_path}"
        ) from exc
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
    ):
        raise TypeError("curriculum windows tensor must be rank-two torch.int64")
    input_ids = input_ids.contiguous()
    if list(input_ids.shape) != [window_count, sequence_length]:
        raise ValueError(
            "curriculum windows tensor shape differs from the manifest: "
            f"{list(input_ids.shape)} != {[window_count, sequence_length]}"
        )
    if _tensor_content_sha256(input_ids) != str(
        tensor_meta.get("content_sha256", "") or ""
    ):
        raise ValueError(
            f"curriculum windows tensor content digest differs: {tensor_path}"
        )

    scores = manifest.get("teacher_loss_by_window")
    if not isinstance(scores, list) or len(scores) != window_count:
        observed = len(scores) if isinstance(scores, list) else None
        raise ValueError(
            "curriculum manifest teacher_loss_by_window does not cover the "
            f"corpus: {observed} scores for {window_count} windows"
        )
    for index, value in enumerate(scores):
        loss = float(value)
        if not math.isfinite(loss):
            raise ValueError(
                f"curriculum manifest window {index} has a non-finite teacher loss"
            )
    return dict(manifest), input_ids


def curriculum_order_from_scores(
    scores: list[float] | list[Any],
    *,
    direction: str,
) -> torch.Tensor:
    """Deterministically order window ids by teacher loss, ids break ties."""

    if direction not in CURRICULUM_DIRECTIONS:
        raise ValueError(
            "text.curriculum_direction must be one of "
            f"{CURRICULUM_DIRECTIONS}, got {direction!r}"
        )
    losses = [float(value) for value in scores]
    sign = 1.0 if direction == "low_to_high" else -1.0
    ordered = sorted(range(len(losses)), key=lambda i: (sign * losses[i], i))
    return torch.tensor(ordered, dtype=torch.long)


class CurriculumOrderedWindowBatchSource(FrozenWindowBatchSource):
    """Manifest-ordered, epoch-stable window stream with strided DDP sharding.

    Unlike the parent, no per-epoch permutation exists: every epoch replays
    the identical curriculum order.  A non-divisible tail is dropped
    deterministically (the last windows of the configured direction).
    """

    _SOURCE_TYPE = "curriculum_ordered_windows"

    def __init__(
        self,
        input_ids: torch.Tensor,
        *,
        order: torch.Tensor,
        direction: str,
        rank: int = 0,
        world_size: int = 1,
        manifest_semantic_sha256: str,
        tensor_content_sha256: str,
    ) -> None:
        super().__init__(
            input_ids,
            seed=0,
            rank=rank,
            world_size=world_size,
            manifest_semantic_sha256=manifest_semantic_sha256,
            tensor_content_sha256=tensor_content_sha256,
        )
        if direction not in CURRICULUM_DIRECTIONS:
            raise ValueError(
                "curriculum direction must be one of "
                f"{CURRICULUM_DIRECTIONS}, got {direction!r}"
            )
        order_tensor = torch.as_tensor(order, dtype=torch.long).reshape(-1)
        if int(order_tensor.numel()) != self.window_count or not torch.equal(
            torch.sort(order_tensor).values,
            torch.arange(self.window_count, dtype=torch.long),
        ):
            raise ValueError(
                "curriculum order must cover every window id exactly once"
            )
        self.direction = str(direction)
        self._order = order_tensor.contiguous()

    def _indices_for_epoch(self) -> torch.Tensor:
        if self._rank_indices is None:
            usable = self.windows_per_rank_epoch * self.world_size
            self._rank_indices = (
                self._order[:usable]
                .reshape(self.windows_per_rank_epoch, self.world_size)[:, self.rank]
                .contiguous()
            )
        return self._rank_indices

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["source_type"] = self._SOURCE_TYPE
        state["curriculum_direction"] = self.direction
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("source_type") != self._SOURCE_TYPE:
            raise ValueError(
                "curriculum stream state source_type differs: "
                f"{state.get('source_type')!r} != {self._SOURCE_TYPE!r}"
            )
        if state.get("curriculum_direction") != self.direction:
            raise ValueError(
                "curriculum stream state direction differs: "
                f"{state.get('curriculum_direction')!r} != {self.direction!r}"
            )
        relabeled = dict(state)
        relabeled["source_type"] = "frozen_text_windows"
        super().load_state_dict(relabeled)


def build_curriculum_window_source(
    *,
    manifest_path: str | Path,
    text_cfg: Mapping[str, Any],
    expected_tokenizer_name: str,
    expected_sequence_length: int,
    expected_separator_token_id: int | None,
    rank: int,
    world_size: int,
) -> CurriculumOrderedWindowBatchSource:
    """Verify one curriculum manifest against the live config and stream it."""

    manifest, input_ids = load_verified_curriculum_manifest(manifest_path)

    identity = manifest.get("corpus_identity")
    if not isinstance(identity, Mapping) or set(identity) != set(
        _CORPUS_IDENTITY_FIELDS
    ):
        observed = set(identity) if isinstance(identity, Mapping) else set()
        raise ValueError(
            "curriculum manifest corpus_identity keys drifted: "
            f"{sorted(observed ^ set(_CORPUS_IDENTITY_FIELDS))}"
        )
    max_documents_raw = text_cfg.get("max_documents")
    live_identity: dict[str, Any] = {
        "data_files": normalized_data_files(text_cfg.get("data_files")),
        "split": str(text_cfg.get("split", "train")),
        "text_field": str(text_cfg.get("text_field", "text")),
        "tokenizer_name": str(expected_tokenizer_name),
        "sequence_length": int(expected_sequence_length),
        "separator_token_id": (
            None
            if expected_separator_token_id is None
            else int(expected_separator_token_id)
        ),
        "max_tokens_per_document": (
            None
            if text_cfg.get("max_tokens_per_document") is None
            else int(text_cfg["max_tokens_per_document"])
        ),
        "max_documents": (
            None if max_documents_raw is None else int(max_documents_raw)
        ),
    }
    mismatches = [
        field
        for field in _CORPUS_IDENTITY_FIELDS
        if identity[field] != live_identity[field]
    ]
    if mismatches:
        raise ValueError(
            "curriculum manifest does not cover the configured corpus; "
            f"identity fields differ: {', '.join(sorted(mismatches))}"
        )

    direction = str(text_cfg.get("curriculum_direction", "low_to_high") or "")
    order = curriculum_order_from_scores(
        manifest["teacher_loss_by_window"], direction=direction
    )
    return CurriculumOrderedWindowBatchSource(
        input_ids,
        order=order,
        direction=direction,
        rank=int(rank),
        world_size=int(world_size),
        manifest_semantic_sha256=str(manifest["manifest_sha256"]),
        tensor_content_sha256=str(manifest["windows_tensor"]["content_sha256"]),
    )


__all__ = [
    "CURRICULUM_DIRECTIONS",
    "CURRICULUM_MANIFEST_SCHEMA",
    "CURRICULUM_SHARD_RECEIPT_SCHEMA",
    "CurriculumOrderedWindowBatchSource",
    "build_curriculum_window_source",
    "curriculum_order_from_scores",
    "load_verified_curriculum_manifest",
    "normalized_data_files",
    "seal_curriculum_manifest",
]
