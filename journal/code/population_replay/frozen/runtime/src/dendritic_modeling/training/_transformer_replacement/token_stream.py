"""Packed, restartable token streams for transformer replacement training."""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class FrozenWindowBatchSource:
    """Restartable, epoch-shuffled batches from verified fixed token windows.

    The source preserves window boundaries.  For DDP, every rank derives the
    same epoch permutation and consumes one disjoint strided column from it.
    A non-divisible tail is dropped *after* shuffling so all ranks have the
    same number of rows and the omitted rows rotate between epochs.
    """

    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        input_ids: torch.Tensor,
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        manifest_semantic_sha256: str,
        tensor_content_sha256: str,
    ) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("frozen input_ids must be a torch.Tensor")
        if input_ids.device.type != "cpu" or input_ids.dtype != torch.long:
            raise TypeError("frozen input_ids must be a CPU torch.int64 tensor")
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError(
                "frozen input_ids must have non-empty [window, token] shape"
            )
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("frozen-window seed must be a non-negative integer")
        if int(world_size) < 1 or int(rank) < 0 or int(rank) >= int(world_size):
            raise ValueError(
                f"invalid frozen-window rank/world_size: {rank}/{world_size}"
            )
        if int(input_ids.shape[0]) < int(world_size):
            raise ValueError("frozen-window count must be at least world_size")
        for label, value in (
            ("manifest_semantic_sha256", manifest_semantic_sha256),
            ("tensor_content_sha256", tensor_content_sha256),
        ):
            normalized = str(value)
            if len(normalized) != 64 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

        self.input_ids = input_ids.contiguous()
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.manifest_semantic_sha256 = str(manifest_semantic_sha256)
        self.tensor_content_sha256 = str(tensor_content_sha256)
        self.sequence_length = int(input_ids.shape[1])
        self.window_count = int(input_ids.shape[0])
        self.epoch = 0
        self.cursor = 0
        self.sequences_emitted = 0
        self._rank_indices: torch.Tensor | None = None

    @property
    def windows_per_rank_epoch(self) -> int:
        return self.window_count // self.world_size

    def _indices_for_epoch(self) -> torch.Tensor:
        if self._rank_indices is None:
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + self.epoch
            )
            permutation = torch.randperm(
                self.window_count,
                generator=generator,
                dtype=torch.long,
            )
            usable = self.windows_per_rank_epoch * self.world_size
            self._rank_indices = permutation[:usable].reshape(
                self.windows_per_rank_epoch,
                self.world_size,
            )[:, self.rank]
        return self._rank_indices

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self._rank_indices = None

    def next_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return a boundary-preserving batch, cycling only at epoch ends."""

        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        remaining = int(batch_size)
        chunks: list[torch.Tensor] = []
        while remaining:
            indices = self._indices_for_epoch()
            available = int(indices.numel()) - self.cursor
            if available == 0:
                self._advance_epoch()
                continue
            take = min(remaining, available)
            selected = indices[self.cursor : self.cursor + take]
            chunks.append(self.input_ids.index_select(0, selected))
            self.cursor += take
            remaining -= take
            if self.cursor == int(indices.numel()) and remaining:
                self._advance_epoch()
        self.sequences_emitted += int(batch_size)
        return torch.cat(chunks, dim=0).to(device=device)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "source_type": "frozen_text_windows",
            "epoch": self.epoch,
            "cursor": self.cursor,
            "sequences_emitted": self.sequences_emitted,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "sequence_length": self.sequence_length,
            "window_count": self.window_count,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "tensor_content_sha256": self.tensor_content_sha256,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "source_type": "frozen_text_windows",
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "sequence_length": self.sequence_length,
            "window_count": self.window_count,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "tensor_content_sha256": self.tensor_content_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"frozen-window stream state {key} differs: "
                    f"{state.get(key)!r} != {value!r}"
                )
        epoch = int(state.get("epoch", -1))
        cursor = int(state.get("cursor", -1))
        emitted = int(state.get("sequences_emitted", -1))
        if epoch < 0 or emitted < 0 or not 0 <= cursor <= self.windows_per_rank_epoch:
            raise ValueError("frozen-window stream state counters are invalid")
        self.epoch = epoch
        self.cursor = cursor
        self.sequences_emitted = emitted
        self._rank_indices = None

    def save_state(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + "\n")
        os.replace(temporary, output)

    def load_state(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text()))


class StreamingTokenBatchSource:
    """Pack a restartable document stream into fixed-length token batches.

    ``document_factory`` receives the zero-based stream epoch.  Recreating the
    factory with the same epoch must produce the same document order; this is
    what makes the compact state dict sufficient for an exact data-stream
    restart.
    """

    def __init__(
        self,
        document_factory: Callable[[int], Iterable[str]],
        tokenizer: Any,
        *,
        sequence_length: int,
        separator_token_id: int | None = None,
        max_tokens_per_document: int | None = None,
    ) -> None:
        if int(sequence_length) < 1:
            raise ValueError("sequence_length must be positive")
        if max_tokens_per_document is not None and int(max_tokens_per_document) < 1:
            raise ValueError("max_tokens_per_document must be positive when set")
        self.document_factory = document_factory
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.separator_token_id = (
            None if separator_token_id is None else int(separator_token_id)
        )
        self.max_tokens_per_document = (
            None if max_tokens_per_document is None else int(max_tokens_per_document)
        )
        self.epoch = 0
        self.documents_seen_in_epoch = 0
        self.sequences_emitted = 0
        self._token_buffer: list[int] = []
        self._document_iterator: Iterator[str] | None = None

    def _new_iterator(self) -> None:
        iterator = iter(self.document_factory(self.epoch))
        if self.documents_seen_in_epoch:
            iterator = itertools.islice(iterator, self.documents_seen_in_epoch, None)
        self._document_iterator = iterator

    def _tokenize_document(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=self.max_tokens_per_document is not None,
            max_length=self.max_tokens_per_document,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("Tokenizer output must contain an input_ids field")
        token_ids = encoded["input_ids"]
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif (
            isinstance(token_ids, Sequence)
            and token_ids
            and isinstance(token_ids[0], Sequence)
        ):
            token_ids = token_ids[0]
        result = [int(token_id) for token_id in token_ids]
        if result and self.separator_token_id is not None:
            result.append(self.separator_token_id)
        return result

    def _append_next_document(self) -> bool:
        """Append one non-empty document, restarting the stream if necessary."""
        empty_epochs = 0
        while True:
            if self._document_iterator is None:
                self._new_iterator()
            assert self._document_iterator is not None
            try:
                text = next(self._document_iterator)
            except StopIteration:
                self.epoch += 1
                self.documents_seen_in_epoch = 0
                self._document_iterator = None
                empty_epochs += 1
                if empty_epochs >= 2:
                    return False
                continue
            self.documents_seen_in_epoch += 1
            token_ids = self._tokenize_document(str(text))
            if token_ids:
                self._token_buffer.extend(token_ids)
                return True

    def next_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return the next packed ``[batch, sequence]`` input-id tensor."""
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        required = int(batch_size) * self.sequence_length
        while len(self._token_buffer) < required:
            if not self._append_next_document():
                raise ValueError("Streaming text source produced no usable tokens")
        batch_tokens = self._token_buffer[:required]
        del self._token_buffer[:required]
        self.sequences_emitted += int(batch_size)
        return (
            torch.tensor(batch_tokens, dtype=torch.long)
            .reshape(int(batch_size), self.sequence_length)
            .to(device=device)
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the minimal exact-restart state for this packed stream."""
        return {
            "schema_version": 1,
            "epoch": self.epoch,
            "documents_seen_in_epoch": self.documents_seen_in_epoch,
            "sequences_emitted": self.sequences_emitted,
            "token_buffer": list(self._token_buffer),
            "sequence_length": self.sequence_length,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        state_sequence_length = int(state.get("sequence_length", self.sequence_length))
        if state_sequence_length != self.sequence_length:
            raise ValueError(
                "Token-stream state sequence_length does not match the current run"
            )
        self.epoch = int(state.get("epoch", 0))
        self.documents_seen_in_epoch = int(state.get("documents_seen_in_epoch", 0))
        self.sequences_emitted = int(state.get("sequences_emitted", 0))
        self._token_buffer = [int(value) for value in state.get("token_buffer", [])]
        self._document_iterator = None

    def save_state(self, path: str | Path) -> None:
        """Atomically save stream progress as portable JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + "\n")
        os.replace(temporary, output)

    def load_state(self, path: str | Path) -> None:
        """Restore stream progress from :meth:`save_state`."""
        self.load_state_dict(json.loads(Path(path).read_text()))


def resolve_data_files(value: Any) -> str | list[str] | None:
    """Normalize a datasets ``data_files`` value without expanding huge globs."""
    if value is None or value == "":
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Sequence):
        return [str(path) for path in value]
    raise TypeError("text.data_files must be a path, glob, or list of paths")


__all__ = [
    "FrozenWindowBatchSource",
    "StreamingTokenBatchSource",
    "resolve_data_files",
]
