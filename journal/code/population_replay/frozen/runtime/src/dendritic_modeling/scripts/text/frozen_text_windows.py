"""Create and verify immutable, content-addressed local text windows.

The default protocol produces one non-overlapping window per source document.
It never joins documents unless ``packing="concatenate"`` is requested
explicitly.  Every output window is bound to the exact local Hugging Face
model/tokenizer artifact manifest, resolved source-file bytes, source row,
document text, and token offsets that produced it.

Only local files are accepted.  Hugging Face snapshot symlinks are followed to
regular files and both their logical and resolved paths are recorded.  The
``verify`` command rehashes the model/tokenizer artifacts, source files, saved
tensor, per-window records, and then rebuilds the tensor from source text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import io
import json
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import torch

from dendritic_modeling.scripts.text.model_artifact_identity import (
    COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA,
    ModelArtifactVerificationCache,
    discover_model_artifacts,
    load_and_verify_model_artifact_manifest,
)

FROZEN_TEXT_WINDOWS_SCHEMA = "dendritic_frozen_text_windows/v1"
TOKENIZER_EQUIVALENCE_SCHEMA = "dendritic_tokenizer_equivalence_audit/v1"
_PACKING_MODES = {"within_document", "concatenate"}
_SELECTION_MODES = {"seeded_shuffle", "input_order", "explicit"}
_TEXT_MODES = {"file", "line"}
_DATA_FORMATS = {"auto", "json", "parquet", "text"}
_TOKENIZER_EQUIVALENCE_SCOPE = "all_scanned_documents_through_last_emitted_window/v1"
_DEFAULT_MAX_SOURCE_RECORD_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_NONSTREAMING_SOURCE_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_SOURCE_FILES = 4096
_DEFAULT_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_RETAINED_TEXT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_DOCUMENT_TOKENS = 1_000_000
_DEFAULT_PARQUET_BATCH_ROWS = 1024
_DEFAULT_MAX_PARQUET_ROW_GROUP_BYTES = 256 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
_TOKENIZER_SNAPSHOT_NAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
}
_EXECUTABLE_CODE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cxx",
    ".fish",
    ".go",
    ".h",
    ".hpp",
    ".ipynb",
    ".java",
    ".jl",
    ".js",
    ".kt",
    ".lua",
    ".mjs",
    ".php",
    ".pl",
    ".py",
    ".pyi",
    ".pyx",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".so",
    ".swift",
    ".ts",
    ".zsh",
}

__all__ = [
    "FROZEN_TEXT_WINDOWS_SCHEMA",
    "TOKENIZER_EQUIVALENCE_SCHEMA",
    "VerifiedFrozenTextWindows",
    "create_frozen_text_windows",
    "verify_frozen_text_windows",
]


@dataclass(frozen=True)
class _DataFile:
    source_index: int
    logical_path: Path
    resolved_path: Path
    is_symlink: bool
    size_bytes: int
    sha256: str
    format: str


@dataclass(frozen=True)
class _Document:
    global_index: int
    source_file_index: int
    source_row_index: int
    text: str
    text_size_bytes: int
    text_sha256: str
    source_group_sha256: str


@dataclass(frozen=True)
class VerifiedFrozenTextWindows:
    """Normalized identity returned after a full fail-closed verification."""

    path: Path
    manifest_file_sha256: str
    semantic_sha256: str
    tensor_path: Path
    tensor_file_sha256: str
    tensor_content_sha256: str
    sequence_length: int
    window_count: int
    primary_tokenizer_artifact_set_sha256: str
    primary_tokenizer_model_root: Path
    data_file_set_sha256: str
    evidentiary: bool
    source_file_count: int
    source_document_count: int
    comparison_tokenizer_equivalence_audited: bool
    comparison_tokenizer_equivalence_scope: str | None
    window_source_document_group_sha256s: tuple[tuple[str, ...], ...]
    record: dict[str, Any]

    def require_group_disjoint_window_prefix(
        self, window_count: int
    ) -> tuple[str, ...]:
        """Require one globally unique source document per consumed row."""

        count = _positive_int(window_count, label="window_count")
        if count > self.window_count:
            raise ValueError(
                f"group-disjoint prefix needs {count} windows; manifest has "
                f"{self.window_count}"
            )
        groups: list[str] = []
        for index, source_groups in enumerate(
            self.window_source_document_group_sha256s[:count]
        ):
            if len(source_groups) != 1:
                raise ValueError(
                    f"frozen window {index} spans {len(source_groups)} source "
                    "documents; one-row/one-group profiling would be invalid"
                )
            groups.append(str(source_groups[0]))
        if len(set(groups)) != len(groups):
            raise ValueError(
                "the consumed frozen-window prefix repeats a source document; "
                "one numeric profiler group per row would leak across partitions"
            )
        return tuple(groups)

    def load_input_ids(self) -> torch.Tensor:
        """Reload the verified tensor without permitting a verification race.

        ``verify_frozen_text_windows`` already reconstructs the tensor from its
        source documents.  Training loads it after that expensive audit, so we
        recheck both the serialized bytes and semantic tensor content before
        returning the exact CPU tensor that will be consumed.
        """

        before = _file_snapshot(self.tensor_path)
        if before[1] != self.tensor_file_sha256:
            raise ValueError("frozen tensor changed after manifest verification")
        input_ids = _load_tensor(self.tensor_path)
        if list(input_ids.shape) != [self.window_count, self.sequence_length]:
            raise ValueError("frozen tensor shape changed after manifest verification")
        if _tensor_content_sha256(input_ids) != self.tensor_content_sha256:
            raise ValueError(
                "frozen tensor content changed after manifest verification"
            )
        if _file_snapshot(self.tensor_path) != before:
            raise ValueError("frozen tensor changed while it was being loaded")
        return input_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "manifest_path": str(self.path),
            "manifest_file_sha256": self.manifest_file_sha256,
            "semantic_sha256": self.semantic_sha256,
            "tensor_path": str(self.tensor_path),
            "tensor_file_sha256": self.tensor_file_sha256,
            "tensor_content_sha256": self.tensor_content_sha256,
            "shape": [self.window_count, self.sequence_length],
            "primary_tokenizer_artifact_set_sha256": (
                self.primary_tokenizer_artifact_set_sha256
            ),
            "primary_tokenizer_model_root": str(self.primary_tokenizer_model_root),
            "data_file_set_sha256": self.data_file_set_sha256,
            "evidentiary": self.evidentiary,
            "source_file_count": self.source_file_count,
            "source_document_count": self.source_document_count,
            "comparison_tokenizer_equivalence_audited": (
                self.comparison_tokenizer_equivalence_audited
            ),
            "comparison_tokenizer_equivalence_scope": (
                self.comparison_tokenizer_equivalence_scope
            ),
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(value.shape)))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _strict_json_loads(value: str, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON constant {constant!r} in {label}")

    return json.loads(
        value,
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )


def _strict_json_load_snapshot(path: Path) -> tuple[dict[str, Any], tuple[int, str]]:
    try:
        payload = path.read_bytes()
        value = _strict_json_loads(payload.decode("utf-8"), label=str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, (len(payload), hashlib.sha256(payload).hexdigest())


def _strict_json_load(path: Path) -> dict[str, Any]:
    value, _snapshot = _strict_json_load_snapshot(path)
    return value


def _file_snapshot(path: Path) -> tuple[int, str]:
    return int(path.stat().st_size), _file_sha256(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _absolute_specification(raw: str | Path) -> str:
    value = os.path.expanduser(str(raw))
    if not os.path.isabs(value):
        value = os.path.join(os.getcwd(), value)
    return os.path.normpath(value)


def _infer_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = path.name.lower()
    if name.endswith(
        (
            ".json",
            ".jsonl",
            ".ndjson",
            ".json.gz",
            ".jsonl.gz",
            ".ndjson.gz",
        )
    ):
        return "json"
    if name.endswith((".parquet", ".pq")):
        return "parquet"
    if name.endswith((".txt", ".text", ".txt.gz", ".text.gz")):
        return "text"
    raise ValueError(
        f"cannot infer data format from {path}; pass --data-format explicitly"
    )


def _resolve_data_files(
    source_specifications: Sequence[str | Path],
    *,
    data_format: str,
    max_source_files: int,
) -> tuple[list[str], list[_DataFile]]:
    if data_format not in _DATA_FORMATS:
        raise ValueError(f"unsupported data format {data_format!r}")
    if not source_specifications:
        raise ValueError("at least one local data source is required")
    maximum_files = _positive_int(max_source_files, label="max_source_files")
    if len(source_specifications) > maximum_files:
        raise ValueError(
            f"data source specification count exceeds max_source_files={maximum_files}"
        )

    import glob

    normalized_specs = [
        _absolute_specification(value) for value in source_specifications
    ]
    logical_paths: list[Path] = []
    seen_logical: set[str] = set()
    for specification in normalized_specs:
        matches: list[str] = []
        for match in glob.iglob(specification, recursive=True):
            if Path(match).is_file():
                matches.append(match)
                if len(matches) > maximum_files:
                    raise ValueError(
                        "one local data specification resolves beyond "
                        f"max_source_files={maximum_files}: {specification}"
                    )
        matches.sort()
        if not matches and not glob.has_magic(specification):
            matches = [specification]
        files = [Path(match).absolute() for match in matches if Path(match).is_file()]
        if not files:
            raise FileNotFoundError(
                f"local data specification resolves to no regular files: {specification}"
            )
        for path in files:
            key = str(path)
            if key not in seen_logical:
                seen_logical.add(key)
                logical_paths.append(path)
                if len(logical_paths) > maximum_files:
                    raise ValueError(
                        "resolved local data file count exceeds "
                        f"max_source_files={maximum_files}"
                    )

    resolved_seen: dict[str, Path] = {}
    records: list[_DataFile] = []
    for index, logical in enumerate(sorted(logical_paths, key=str)):
        try:
            resolved = logical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot safely resolve data file {logical}") from exc
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"resolved data source is not a regular file: {resolved}")
        prior = resolved_seen.get(str(resolved))
        if prior is not None and prior != logical:
            raise ValueError(
                "two logical data paths resolve to the same file; refusing duplicate "
                f"ingestion: {prior} and {logical}"
            )
        resolved_seen[str(resolved)] = logical
        records.append(
            _DataFile(
                source_index=index,
                logical_path=logical,
                resolved_path=resolved,
                is_symlink=logical.is_symlink(),
                size_bytes=int(resolved.stat().st_size),
                sha256=_file_sha256(resolved),
                format=_infer_format(logical, data_format),
            )
        )
    return normalized_specs, records


def _extract_text(value: object, *, text_field: str, label: str) -> str:
    if isinstance(value, str):
        return value
    current = value
    for component in text_field.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"{label} has no string field {text_field!r}")
        current = current[component]
    if not isinstance(current, str):
        raise ValueError(f"{label}.{text_field} must be a string")
    return current


@contextmanager
def _open_binary(path: Path) -> Iterator[BinaryIO]:
    """Open a source as a streaming binary reader, decoding gzip by content.

    Hugging Face snapshots commonly expose a descriptive ``*.json.gz``
    symlink whose resolved content-addressed blob has no suffix.  Compression
    therefore cannot be inferred safely from ``resolved_path.name``.  Probe
    only the two-byte gzip signature, rewind, and keep both the raw file and
    decoder scoped to the caller's context.  File identity continues to hash
    the untouched compressed bytes through :func:`_file_sha256`; all existing
    record and non-streaming limits apply to the decoded stream.
    """

    with Path(path).open("rb") as raw_handle:
        magic = raw_handle.read(len(_GZIP_MAGIC))
        raw_handle.seek(0)
        if magic == _GZIP_MAGIC:
            with gzip.GzipFile(fileobj=raw_handle, mode="rb") as decoded_handle:
                yield decoded_handle
        else:
            yield raw_handle


def _decode_source_bytes(value: bytes, *, label: str) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {label}") from exc


def _read_bounded_source_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    with _open_binary(path) as handle:
        payload = handle.read(int(maximum_bytes) + 1)
    if len(payload) > int(maximum_bytes):
        raise ValueError(
            f"{label} exceeds max_nonstreaming_source_bytes={maximum_bytes}: {path}"
        )
    return payload


def _bounded_binary_line(
    handle: Any,
    *,
    maximum_bytes: int,
    label: str,
    line_number: int,
) -> bytes:
    raw = handle.readline(int(maximum_bytes) + 2)
    if not raw:
        return b""
    content = raw[:-1] if raw.endswith(b"\n") else raw
    if content.endswith(b"\r"):
        content = content[:-1]
    if len(content) > int(maximum_bytes):
        raise ValueError(
            f"{label} at line {line_number} exceeds "
            f"max_source_record_bytes={maximum_bytes}"
        )
    return raw


def _strict_json_value(value: bytes, *, label: str) -> Any:
    text = _decode_source_bytes(value, label=label)
    return _strict_json_loads(text, label=label)


def _first_non_whitespace_byte(path: Path) -> int | None:
    """Inspect a source with constant memory before choosing its JSON reader."""

    with _open_binary(path) as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            for value in chunk:
                if value not in b" \t\r\n":
                    return int(value)
    return None


def _iter_json_values(
    path: Path,
    *,
    max_source_record_bytes: int,
    max_nonstreaming_source_bytes: int,
) -> Iterable[object]:
    """Yield bounded strict JSON arrays, objects, or newline-delimited rows."""

    first_byte = _first_non_whitespace_byte(path)
    if first_byte is None:
        return
    if first_byte == ord("["):
        raw_payload = _read_bounded_source_bytes(
            path,
            maximum_bytes=max_nonstreaming_source_bytes,
            label="JSON array source",
        )
        try:
            payload = _strict_json_value(raw_payload, label=str(path))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON array source {path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"JSON array source expected in {path}")
        yield from payload
        return

    with _open_binary(path) as handle:
        first_nonblank = b""
        first_line_number = 0
        line_number = 0
        while True:
            line_number += 1
            line = _bounded_binary_line(
                handle,
                maximum_bytes=max_source_record_bytes,
                label=str(path),
                line_number=line_number,
            )
            if not line:
                return
            if line.strip():
                first_nonblank = line
                first_line_number = line_number
                break
        try:
            first = _strict_json_value(
                first_nonblank,
                label=f"{path}:{first_line_number}",
            )
        except json.JSONDecodeError:
            raw_payload = _read_bounded_source_bytes(
                path,
                maximum_bytes=max_nonstreaming_source_bytes,
                label="multi-line JSON source",
            )
            try:
                payload = _strict_json_value(raw_payload, label=str(path))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON source {path}") from exc
            if isinstance(payload, list):
                yield from payload
            else:
                yield payload
            return
        yield first
        while True:
            line_number += 1
            line = _bounded_binary_line(
                handle,
                maximum_bytes=max_source_record_bytes,
                label=str(path),
                line_number=line_number,
            )
            if not line:
                break
            if not line.strip():
                continue
            try:
                yield _strict_json_value(line, label=f"{path}:{line_number}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid newline-delimited JSON at {path}:{line_number}"
                ) from exc


def _iter_file_texts(
    source: _DataFile,
    *,
    text_field: str,
    text_mode: str,
    max_source_record_bytes: int,
    max_nonstreaming_source_bytes: int,
    parquet_batch_rows: int,
    max_parquet_row_group_bytes: int,
) -> Iterable[str]:
    if source.format == "json":
        values = _iter_json_values(
            source.resolved_path,
            max_source_record_bytes=max_source_record_bytes,
            max_nonstreaming_source_bytes=max_nonstreaming_source_bytes,
        )
        for row_index, value in enumerate(values):
            yield _extract_text(
                value,
                text_field=text_field,
                label=f"{source.logical_path} row {row_index}",
            )
        return
    if source.format == "text":
        if text_mode == "file":
            payload = _read_bounded_source_bytes(
                source.resolved_path,
                maximum_bytes=max_nonstreaming_source_bytes,
                label="whole-file text source",
            )
            yield _decode_source_bytes(payload, label=str(source.logical_path))
            return
        with _open_binary(source.resolved_path) as handle:
            line_number = 0
            while True:
                line_number += 1
                raw = _bounded_binary_line(
                    handle,
                    maximum_bytes=max_source_record_bytes,
                    label=str(source.logical_path),
                    line_number=line_number,
                )
                if not raw:
                    break
                content = raw[:-1] if raw.endswith(b"\n") else raw
                if content.endswith(b"\r"):
                    content = content[:-1]
                yield _decode_source_bytes(
                    content,
                    label=f"{source.logical_path}:{line_number}",
                )
        return
    if source.format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read Parquet sources") from exc
        top_level = text_field.split(".", 1)[0]
        if source.size_bytes < 8:
            raise ValueError(f"invalid Parquet source {source.logical_path}")
        with source.resolved_path.open("rb") as handle:
            handle.seek(-8, os.SEEK_END)
            footer = handle.read(8)
        if footer[4:] != b"PAR1":
            raise ValueError(f"invalid Parquet footer in {source.logical_path}")
        metadata_bytes = int.from_bytes(footer[:4], byteorder="little")
        if metadata_bytes + 8 > max_nonstreaming_source_bytes:
            raise ValueError(
                f"Parquet metadata needs {metadata_bytes + 8} bytes; limit is "
                "max_nonstreaming_source_bytes="
                f"{max_nonstreaming_source_bytes}"
            )
        parquet = pq.ParquetFile(
            source.resolved_path,
            thrift_string_size_limit=max_nonstreaming_source_bytes,
            thrift_container_size_limit=max(1, max_nonstreaming_source_bytes // 4),
        )
        metadata = parquet.metadata
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            uncompressed_bytes = 0
            matched_column = False
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                column_path = str(column.path_in_schema)
                if column_path == top_level or column_path.startswith(f"{top_level}."):
                    matched_column = True
                    size = int(column.total_uncompressed_size)
                    if size < 0:
                        raise ValueError(
                            "Parquet metadata cannot bound an input column row group"
                        )
                    uncompressed_bytes += size
            if not matched_column:
                raise ValueError(
                    f"Parquet source has no top-level column {top_level!r}"
                )
            if uncompressed_bytes > max_parquet_row_group_bytes:
                raise ValueError(
                    f"Parquet row group {row_group_index} needs "
                    f"{uncompressed_bytes} uncompressed input bytes; limit is "
                    f"max_parquet_row_group_bytes={max_parquet_row_group_bytes}"
                )
            for batch in parquet.iter_batches(
                batch_size=parquet_batch_rows,
                row_groups=[row_group_index],
                columns=[top_level],
            ):
                for value in batch.to_pylist():
                    yield _extract_text(
                        value,
                        text_field=text_field,
                        label=f"{source.logical_path} Parquet row",
                    )
        return
    raise AssertionError(f"unhandled data format {source.format!r}")


def _document_rank(document: _Document, seed: int) -> tuple[int, int]:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(document.global_index).encode("ascii"))
    digest.update(b"\0")
    digest.update(document.text_sha256.encode("ascii"))
    return int.from_bytes(digest.digest(), "big"), document.global_index


def _load_selected_documents(
    sources: Sequence[_DataFile],
    *,
    text_field: str,
    text_mode: str,
    selection_mode: str,
    seed: int,
    explicit_indices: Sequence[int] | None,
    max_documents: int,
    max_source_record_bytes: int,
    max_nonstreaming_source_bytes: int,
    max_document_bytes: int,
    max_retained_text_bytes: int,
    parquet_batch_rows: int,
    max_parquet_row_group_bytes: int,
) -> tuple[dict[int, _Document], list[int], int, list[int], int]:
    """Stream all rows while retaining only the prospectively selected texts."""

    if not text_field or not isinstance(text_field, str):
        raise ValueError("text_field must be a non-empty string")
    if text_mode not in _TEXT_MODES:
        raise ValueError(f"unsupported text mode {text_mode!r}")
    if selection_mode not in _SELECTION_MODES:
        raise ValueError(f"unsupported document selection mode {selection_mode!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    max_documents = _positive_int(max_documents, label="max_documents")

    requested: list[int] | None = None
    requested_set: set[int] | None = None
    if selection_mode == "explicit":
        if explicit_indices is None or not explicit_indices:
            raise ValueError("explicit document selection requires document indices")
        requested = []
        seen: set[int] = set()
        for raw_index in explicit_indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError("document indices must be integers")
            index = int(raw_index)
            if index < 0:
                raise ValueError("document indices must be nonnegative")
            if index in seen:
                raise ValueError(f"duplicate explicit document index {index}")
            seen.add(index)
            requested.append(index)
        if len(requested) > max_documents:
            raise ValueError(
                f"explicit document selection names {len(requested)} indices but "
                f"max_documents={max_documents}; refusing to truncate the protocol"
            )
        requested_set = set(requested)
    elif explicit_indices is not None:
        raise ValueError("document indices are valid only with explicit selection")

    retained: dict[int, _Document] = {}
    # A min-heap of negated ranks keeps the K smallest stable ranks.  Each
    # entry retains the full text only for a document that can be selected.
    ranked_heap: list[tuple[int, int, _Document]] = []
    retained_text_bytes = 0
    per_file_counts: list[int] = []
    global_index = 0
    for source in sources:
        count = 0
        for row_index, text in enumerate(
            _iter_file_texts(
                source,
                text_field=text_field,
                text_mode=text_mode,
                max_source_record_bytes=max_source_record_bytes,
                max_nonstreaming_source_bytes=max_nonstreaming_source_bytes,
                parquet_batch_rows=parquet_batch_rows,
                max_parquet_row_group_bytes=max_parquet_row_group_bytes,
            )
        ):
            text_size_bytes = len(text.encode("utf-8"))
            if text_size_bytes > max_document_bytes:
                raise ValueError(
                    f"source document {global_index} contains {text_size_bytes} "
                    f"UTF-8 bytes; limit is max_document_bytes={max_document_bytes}"
                )
            text_digest = _text_sha256(text)
            group_digest = _canonical_sha256(
                {
                    "source_file_sha256": source.sha256,
                    "source_row_index": row_index,
                    "document_text_sha256": text_digest,
                }
            )
            document = _Document(
                global_index=global_index,
                source_file_index=source.source_index,
                source_row_index=row_index,
                text=text,
                text_size_bytes=text_size_bytes,
                text_sha256=text_digest,
                source_group_sha256=group_digest,
            )
            if selection_mode == "input_order":
                if global_index < max_documents:
                    retained_text_bytes += text_size_bytes
                    if retained_text_bytes > max_retained_text_bytes:
                        raise ValueError(
                            "retained source text exceeds "
                            f"max_retained_text_bytes={max_retained_text_bytes}"
                        )
                    retained[global_index] = document
            elif selection_mode == "explicit":
                if requested_set is not None and global_index in requested_set:
                    retained_text_bytes += text_size_bytes
                    if retained_text_bytes > max_retained_text_bytes:
                        raise ValueError(
                            "retained source text exceeds "
                            f"max_retained_text_bytes={max_retained_text_bytes}"
                        )
                    retained[global_index] = document
            else:
                rank, tie_break = _document_rank(document, seed)
                entry = (-rank, -tie_break, document)
                if len(ranked_heap) < max_documents:
                    retained_text_bytes += text_size_bytes
                    if retained_text_bytes > max_retained_text_bytes:
                        raise ValueError(
                            "seeded top-K source text exceeds "
                            f"max_retained_text_bytes={max_retained_text_bytes}"
                        )
                    heapq.heappush(ranked_heap, entry)
                elif entry[:2] > ranked_heap[0][:2]:
                    replacement_bytes = ranked_heap[0][2].text_size_bytes
                    projected = (
                        retained_text_bytes - replacement_bytes + text_size_bytes
                    )
                    if projected > max_retained_text_bytes:
                        raise ValueError(
                            "seeded top-K source text exceeds "
                            f"max_retained_text_bytes={max_retained_text_bytes}"
                        )
                    heapq.heapreplace(ranked_heap, entry)
                    retained_text_bytes = projected
            global_index += 1
            count += 1
        per_file_counts.append(count)
    if global_index == 0:
        raise ValueError("local data sources contain no documents")
    if selection_mode == "seeded_shuffle":
        retained = {entry[2].global_index: entry[2] for entry in ranked_heap}
        order = [
            document.global_index
            for document in sorted(
                retained.values(), key=lambda value: _document_rank(value, seed)
            )
        ]
    elif selection_mode == "input_order":
        order = sorted(retained)
    else:
        assert requested is not None
        missing = [index for index in requested if index >= global_index]
        if missing:
            raise ValueError(
                f"document indices are outside the source corpus: {missing[:8]}"
            )
        order = list(requested)
    if not order:
        raise ValueError("document selection retained no source documents")
    return retained, per_file_counts, global_index, order, retained_text_bytes


def _tokenize_document(
    tokenizer: Any,
    text: str,
    *,
    max_document_tokens: int,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=True,
        max_length=int(max_document_tokens) + 1,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return an input_ids field")
    raw = encoded["input_ids"]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise TypeError("single-document tokenization returned multiple rows")
        raw = raw[0]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("tokenizer input_ids must be a flat sequence")
    tokens = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("tokenizer input_ids must contain integers")
        if value < 0:
            raise ValueError("tokenizer input_ids must be nonnegative")
        tokens.append(int(value))
    if len(tokens) > max_document_tokens:
        raise ValueError(
            "source document tokenization exceeds "
            f"max_document_tokens={max_document_tokens}"
        )
    return tokens


def _segment(
    document: _Document,
    *,
    source: _DataFile,
    document_token_start: int,
    document_token_end: int,
    window_token_start: int,
    window_token_end: int,
) -> dict[str, Any]:
    return {
        "source_file_index": int(document.source_file_index),
        "source_file_sha256": source.sha256,
        "source_document_index": int(document.global_index),
        "source_row_index": int(document.source_row_index),
        "document_text_sha256": document.text_sha256,
        "source_document_group_sha256": document.source_group_sha256,
        "document_token_start": int(document_token_start),
        "document_token_end": int(document_token_end),
        "window_token_start": int(window_token_start),
        "window_token_end": int(window_token_end),
    }


def _window_record(
    window_index: int,
    tokens: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tensor = torch.tensor(tokens, dtype=torch.long)
    source_groups = list(
        dict.fromkeys(
            str(segment["source_document_group_sha256"]) for segment in segments
        )
    )
    payload = {
        "window_index": int(window_index),
        "input_ids_sha256": _tensor_content_sha256(tensor),
        "source_document_group_sha256s": source_groups,
        "source_group_set_sha256": _canonical_sha256(source_groups),
        "segments": [dict(segment) for segment in segments],
    }
    return {**payload, "semantic_sha256": _canonical_sha256(payload)}


def _scanned_document_record(
    document: _Document,
    tokens: Sequence[int],
    *,
    selected_order: int,
    status: str,
    emitted_token_ranges: Sequence[Sequence[int]],
) -> dict[str, Any]:
    return {
        "selected_order": int(selected_order),
        "source_document_index": int(document.global_index),
        "source_file_index": int(document.source_file_index),
        "source_row_index": int(document.source_row_index),
        "document_text_sha256": document.text_sha256,
        "source_document_group_sha256": document.source_group_sha256,
        "token_count": len(tokens),
        "token_ids_sha256": _tensor_content_sha256(
            torch.tensor(tokens, dtype=torch.long)
        ),
        "status": status,
        "emitted_token_ranges": [list(value) for value in emitted_token_ranges],
    }


def _build_within_document_windows(
    tokenizer: Any,
    documents: Mapping[int, _Document],
    sources: Sequence[_DataFile],
    order: Sequence[int],
    *,
    sequence_length: int,
    window_count: int,
    max_windows_per_document: int | None,
    max_document_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[list[int]] = []
    window_records: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    for selected_order, index in enumerate(order):
        document = documents[index]
        tokens = _tokenize_document(
            tokenizer,
            document.text,
            max_document_tokens=max_document_tokens,
        )
        candidate_count = len(tokens) // sequence_length
        if max_windows_per_document is not None:
            candidate_count = min(candidate_count, max_windows_per_document)
        remaining = window_count - len(rows)
        emitted = min(candidate_count, remaining)
        ranges: list[list[int]] = []
        for candidate in range(emitted):
            start = candidate * sequence_length
            end = start + sequence_length
            window_tokens = tokens[start:end]
            ranges.append([start, end])
            segment = _segment(
                document,
                source=sources[document.source_file_index],
                document_token_start=start,
                document_token_end=end,
                window_token_start=0,
                window_token_end=sequence_length,
            )
            rows.append(window_tokens)
            window_records.append(
                _window_record(len(rows) - 1, window_tokens, [segment])
            )
        status = (
            "used" if emitted else ("blank" if not document.text.strip() else "short")
        )
        scanned.append(
            _scanned_document_record(
                document,
                tokens,
                selected_order=selected_order,
                status=status,
                emitted_token_ranges=ranges,
            )
        )
        if len(rows) == window_count:
            break
    if len(rows) != window_count:
        raise ValueError(
            f"selected documents yielded {len(rows)} complete within-document "
            f"windows; requested {window_count}"
        )
    return torch.tensor(rows, dtype=torch.long), window_records, scanned


def _build_concatenated_windows(
    tokenizer: Any,
    documents: Mapping[int, _Document],
    sources: Sequence[_DataFile],
    order: Sequence[int],
    *,
    sequence_length: int,
    window_count: int,
    max_document_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    needed = sequence_length * window_count
    stream: list[int] = []
    origins: list[tuple[_Document, int]] = []
    scanned: list[dict[str, Any]] = []
    for selected_order, index in enumerate(order):
        document = documents[index]
        tokens = _tokenize_document(
            tokenizer,
            document.text,
            max_document_tokens=max_document_tokens,
        )
        take = min(len(tokens), needed - len(stream))
        if take:
            stream.extend(tokens[:take])
            origins.extend((document, offset) for offset in range(take))
        status = (
            "used" if take else ("blank" if not document.text.strip() else "unused")
        )
        scanned.append(
            _scanned_document_record(
                document,
                tokens,
                selected_order=selected_order,
                status=status,
                emitted_token_ranges=[[0, take]] if take else [],
            )
        )
        if len(stream) == needed:
            break
    if len(stream) != needed:
        raise ValueError(
            f"selected documents yielded {len(stream)} concatenated tokens; "
            f"requested {needed}"
        )
    rows = torch.tensor(stream, dtype=torch.long).view(window_count, sequence_length)
    window_records: list[dict[str, Any]] = []
    for window_index in range(window_count):
        global_start = window_index * sequence_length
        segments: list[dict[str, Any]] = []
        local = 0
        while local < sequence_length:
            document, document_offset = origins[global_start + local]
            run = 1
            while local + run < sequence_length:
                next_document, next_offset = origins[global_start + local + run]
                if next_document.global_index != document.global_index or (
                    next_offset != document_offset + run
                ):
                    break
                run += 1
            segments.append(
                _segment(
                    document,
                    source=sources[document.source_file_index],
                    document_token_start=document_offset,
                    document_token_end=document_offset + run,
                    window_token_start=local,
                    window_token_end=local + run,
                )
            )
            local += run
        window_records.append(
            _window_record(window_index, rows[window_index].tolist(), segments)
        )
    return rows, window_records, scanned


def _load_local_tokenizer(
    root: Path,
    *,
    use_fast: bool,
    trust_local_code: bool,
) -> tuple[Any, dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to freeze text windows") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        use_fast=bool(use_fast),
        trust_remote_code=bool(trust_local_code),
    )
    runtime = {
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocabulary_size": len(tokenizer),
        "transformers_version": str(transformers.__version__),
        "tokenizers_version": str(tokenizers.__version__),
    }
    if use_fast and not runtime["is_fast"]:
        raise RuntimeError("the exact local artifact did not produce a fast tokenizer")
    return tokenizer, runtime


def _tokenizer_artifact_rows(verified: Any) -> list[dict[str, Any]]:
    rows = []
    for raw in verified.record["artifacts"]:
        relative = str(raw["path"])
        path = Path(relative)
        if path.name in _TOKENIZER_SNAPSHOT_NAMES or path.suffix in {".jinja", ".py"}:
            rows.append(
                {
                    "path": relative,
                    "size_bytes": int(raw["size_bytes"]),
                    "sha256": str(raw["sha256"]),
                }
            )
    if not rows:
        raise ValueError("verified model artifact set has no tokenizer assets")
    return rows


def _tokenizer_artifact_set_sha256(verified: Any) -> str:
    return _canonical_sha256(
        {
            "schema": "dendritic_tokenizer_artifact_subset/v1",
            "artifacts": _tokenizer_artifact_rows(verified),
        }
    )


def _verify_tokenizer_snapshot(verified: Any, *, label: str) -> None:
    """Rehash only tokenizer/config/code assets after a tokenizer was used."""

    if _file_sha256(verified.path) != verified.file_sha256:
        raise RuntimeError(f"{label} artifact manifest changed during operation")
    recorded_paths = [str(row["path"]) for row in verified.record["artifacts"]]
    complete_identity = (
        verified.record.get("schema") == COMPLETE_MODEL_ARTIFACT_IDENTITY_SCHEMA
    )
    if (
        discover_model_artifacts(
            verified.model_root,
            complete_consumed_tokenizer_identity=complete_identity,
        )
        != recorded_paths
    ):
        raise RuntimeError(f"{label} discovered artifact set changed during operation")
    for row in _tokenizer_artifact_rows(verified):
        path = verified.model_root / row["path"]
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or _file_sha256(path) != row["sha256"]
        ):
            raise RuntimeError(
                f"{label} tokenizer artifact changed during operation: {row['path']}"
            )


def _is_executable_code_file(path: Path) -> bool:
    if path.suffix.lower() in _EXECUTABLE_CODE_SUFFIXES:
        return True
    try:
        if os.access(path, os.X_OK):
            return True
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return True


def _untracked_executable_rows(root: Path) -> list[dict[str, Any]]:
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError("could not inspect untracked Git paths")
    rows = []
    pending = b""
    try:
        for chunk in iter(lambda: process.stdout.read(64 * 1024), b""):
            pending += chunk
            fields = pending.split(b"\0")
            pending = fields.pop()
            if len(pending) > 1024 * 1024:
                raise RuntimeError("an untracked Git path exceeds one MiB")
            for raw_relative in fields:
                if not raw_relative:
                    continue
                relative = os.fsdecode(raw_relative)
                try:
                    relative.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise RuntimeError(
                        "untracked executable paths must be valid UTF-8"
                    ) from exc
                path = root / relative
                if not path.is_file() or not _is_executable_code_file(path):
                    continue
                snapshot = _file_snapshot(path)
                rows.append(
                    {
                        "path": relative,
                        "size_bytes": snapshot[0],
                        "sha256": snapshot[1],
                    }
                )
                if len(rows) > 100_000:
                    raise RuntimeError(
                        "more than 100000 untracked executable files are present"
                    )
        if pending:
            raise RuntimeError("Git returned an unterminated untracked path")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.wait() != 0:
        raise RuntimeError("could not inspect untracked Git paths")
    return sorted(rows, key=lambda row: row["path"])


def _git_tracked_diff_sha256(root: Path) -> str:
    """Hash tracked Git changes without retaining a potentially large diff."""

    process = subprocess.Popen(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError("could not inspect tracked Git changes")
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError("could not inspect tracked Git changes")
    return digest.hexdigest()


def _git_provenance(repo_root: Path | None) -> dict[str, Any]:
    start = Path(repo_root or Path.cwd()).resolve()
    top = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if top.returncode != 0:
        return {
            "available": False,
            "repo_root": None,
            "commit": None,
            "tracked_worktree_dirty": None,
            "tracked_diff_sha256": None,
            "untracked_executable_files": [],
            "untracked_executable_set_sha256": _canonical_sha256([]),
        }
    root = Path(top.stdout.strip()).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_diff_sha256 = _git_tracked_diff_sha256(root)
    untracked_executables = _untracked_executable_rows(root)
    return {
        "available": True,
        "repo_root": str(root),
        "commit": revision,
        "tracked_worktree_dirty": tracked_diff_sha256
        != hashlib.sha256(b"").hexdigest(),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_executable_files": untracked_executables,
        "untracked_executable_set_sha256": _canonical_sha256(untracked_executables),
    }


def _committed_blob_snapshot(
    repo_root: Path,
    commit: str,
    relative_path: Path,
) -> tuple[int, str]:
    """Return a bounded-memory size/hash snapshot of one committed Git blob."""

    object_name = f"{commit}:{relative_path.as_posix()}"
    size_result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-s", object_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if size_result.returncode != 0:
        raise RuntimeError(
            "executing local module is not tracked at the recorded commit: "
            f"{relative_path}"
        )
    try:
        size_bytes = int(size_result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Git returned an invalid blob size for {relative_path}"
        ) from exc
    process = subprocess.Popen(
        ["git", "-C", str(repo_root), "cat-file", "blob", object_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError(f"could not read committed blob {relative_path}")
    digest = hashlib.sha256()
    observed_size = 0
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        observed_size += len(chunk)
        digest.update(chunk)
    if process.wait() != 0 or observed_size != size_bytes:
        raise RuntimeError(f"could not read committed blob {relative_path}")
    return size_bytes, digest.hexdigest()


def _executed_local_code_paths() -> tuple[tuple[str, Path], ...]:
    """Resolve the local modules that define the frozen-window evidence path."""

    identity_module = sys.modules.get(discover_model_artifacts.__module__)
    identity_file = getattr(identity_module, "__file__", None)
    if identity_file is None:
        raise RuntimeError("cannot resolve the loaded model-artifact identity module")
    return (
        ("frozen_text_windows", Path(__file__).resolve(strict=True)),
        (
            "model_artifact_identity",
            Path(identity_file).resolve(strict=True),
        ),
    )


def _require_evidentiary_execution_identity(git: Mapping[str, Any]) -> None:
    """Bind executing local package code to the recorded clean Git commit."""

    if not git.get("available") or not git.get("repo_root") or not git.get("commit"):
        raise RuntimeError(
            "evidentiary execution identity requires a recorded Git repository"
        )
    repo_root = Path(str(git["repo_root"])).resolve(strict=True)
    commit = str(git["commit"])
    for label, module_path in _executed_local_code_paths():
        try:
            relative_path = module_path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(
                f"executing local module {label} resolves outside the recorded "
                f"repository: module={module_path}, repo_root={repo_root}"
            ) from exc
        actual_snapshot = _file_snapshot(module_path)
        committed_snapshot = _committed_blob_snapshot(
            repo_root,
            commit,
            relative_path,
        )
        if actual_snapshot != committed_snapshot:
            raise RuntimeError(
                f"executing local module {label} does not match its tracked bytes "
                f"at commit {commit}: {relative_path}"
            )


def _require_clean_git_snapshot(
    git: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Require a complete, clean Git snapshot suitable for evidence code."""

    if not git.get("available") or not git.get("repo_root") or not git.get("commit"):
        raise RuntimeError(f"{label} requires an available Git repository")
    if (
        git.get("tracked_worktree_dirty") is not False
        or git.get("tracked_diff_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(f"{label} Git worktree must be clean")
    if git.get("untracked_executable_files"):
        raise RuntimeError(f"{label} forbids untracked executable code")
    if git.get("untracked_executable_set_sha256") != _canonical_sha256([]):
        raise RuntimeError(f"{label} has an inconsistent executable-code snapshot")


def _require_commit_ancestor(
    *,
    repo_root: Path,
    ancestor: str,
    descendant: str,
) -> None:
    """Require ``ancestor`` to be in the history of ``descendant``."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise RuntimeError(
            "current evidence verifier commit is not a descendant of the "
            f"recorded generation commit: {ancestor} -> {descendant}"
        )
    detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
    raise RuntimeError(
        "cannot establish evidence-code ancestry between the recorded generation "
        f"commit and current verifier commit: {detail}"
    )


def _require_evidentiary_verification_identity(
    recorded_git: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the immutable generation witness and a clean descendant verifier.

    Artifact creation remains bound to the exact executing checkout through
    :func:`_require_evidentiary_execution_identity`. Later verification may
    execute from another worktree so bug fixes do not strand existing evidence,
    but only when all evidence-path modules come from one clean repository,
    equal their own HEAD blobs, and that HEAD descends from the recorded
    generation commit. The original worktree is independently revalidated as
    the unchanged generation witness.
    """

    if not recorded_git.get("repo_root") or not recorded_git.get("commit"):
        raise RuntimeError("evidentiary manifest lacks generation Git identity")
    expected_recorded = dict(recorded_git)
    recorded_root = Path(str(recorded_git["repo_root"]))
    observed_recorded = _git_provenance(recorded_root)
    if observed_recorded != expected_recorded:
        raise ValueError(
            "recorded generation Git worktree differs from the evidentiary "
            "frozen-window provenance"
        )
    _require_clean_git_snapshot(
        observed_recorded,
        label="recorded generation evidence",
    )

    executed_paths = _executed_local_code_paths()
    if not executed_paths:
        raise RuntimeError("cannot resolve any executing evidence-path modules")
    first_label, first_path = executed_paths[0]
    current_git = _git_provenance(first_path.parent)
    _require_clean_git_snapshot(current_git, label="current evidence verifier")
    current_root = Path(str(current_git["repo_root"])).resolve(strict=True)
    for label, module_path in executed_paths:
        try:
            module_path.relative_to(current_root)
        except ValueError as exc:
            raise RuntimeError(
                f"executing local module {label} resolves outside the current "
                "verifier repository: "
                f"module={module_path}, repo_root={current_root}; first module="
                f"{first_label}"
            ) from exc
    _require_evidentiary_execution_identity(current_git)
    _require_commit_ancestor(
        repo_root=current_root,
        ancestor=str(recorded_git["commit"]),
        descendant=str(current_git["commit"]),
    )
    return current_git


def _data_file_rows(
    files: Sequence[_DataFile], per_file_counts: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {
            "source_index": source.source_index,
            "logical_path": str(source.logical_path),
            "resolved_path": str(source.resolved_path),
            "is_symlink": source.is_symlink,
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
            "format": source.format,
            "document_count": int(per_file_counts[source.source_index]),
        }
        for source in files
    ]


def _data_file_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                "source_index": row["source_index"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "format": row["format"],
                "document_count": row["document_count"],
            }
            for row in rows
        ]
    )


def _tokenizer_binding(
    verified: Any,
    *,
    use_fast: bool,
    trust_local_code: bool,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_path": str(verified.path),
        "manifest_file_sha256": verified.file_sha256,
        "artifact_set_sha256": verified.artifact_set_sha256,
        "tokenizer_artifact_set_sha256": _tokenizer_artifact_set_sha256(verified),
        "model_root": str(verified.model_root),
        "loading": {
            "local_files_only": True,
            "use_fast": bool(use_fast),
            "trust_local_code": bool(trust_local_code),
        },
        "runtime": dict(runtime),
    }


def _tokenizer_binding_semantics(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_file_sha256": binding["manifest_file_sha256"],
        "artifact_set_sha256": binding["artifact_set_sha256"],
        "tokenizer_artifact_set_sha256": binding["tokenizer_artifact_set_sha256"],
        "loading": binding["loading"],
        "runtime": binding["runtime"],
    }


def _tokenizer_equivalence_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": record["schema"],
        "scope": record["scope"],
        "corpus_wide": record["corpus_wide"],
        "distinct_tokenizer_artifact_identities": record[
            "distinct_tokenizer_artifact_identities"
        ],
        "comparison_tokenizer": _tokenizer_binding_semantics(
            record["comparison_tokenizer"]
        ),
        "audited_documents": record["audited_documents"],
        "audited_document_set_sha256": record["audited_document_set_sha256"],
        "document_count": record["document_count"],
        "exact_document_token_ids_equal": record["exact_document_token_ids_equal"],
        "exact_window_input_ids_equal": record["exact_window_input_ids_equal"],
        "comparison_tensor_content_sha256": record["comparison_tensor_content_sha256"],
    }


def _semantic_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = manifest["tokenizer"]
    data = manifest["data"]
    tensor = manifest["tensor"]
    return {
        "schema": manifest["schema"],
        "protocol": manifest["protocol"],
        "git": manifest["git"],
        "tokenizer": _tokenizer_binding_semantics(tokenizer),
        "tokenizer_equivalence_semantic_sha256": (
            None
            if manifest["tokenizer_equivalence"] is None
            else manifest["tokenizer_equivalence"]["semantic_sha256"]
        ),
        "data": {
            "file_set_sha256": data["file_set_sha256"],
            "document_count_total": data["document_count_total"],
            "retained_document_count": data["retained_document_count"],
            "retained_text_bytes": data["retained_text_bytes"],
            "scanned_documents": data["scanned_documents"],
        },
        "tensor_content_sha256": tensor["content_sha256"],
        "windows": manifest["windows"],
    }


def _audit_comparison_tokenizer(
    comparison_tokenizer: Any,
    *,
    primary_binding: Mapping[str, Any],
    comparison_binding: Mapping[str, Any],
    documents: Mapping[int, _Document],
    sources: Sequence[_DataFile],
    order: Sequence[int],
    protocol: Mapping[str, Any],
    primary_input_ids: torch.Tensor,
    primary_windows: Sequence[Mapping[str, Any]],
    primary_scanned: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        comparison_binding["tokenizer_artifact_set_sha256"]
        == primary_binding["tokenizer_artifact_set_sha256"]
    ):
        raise ValueError(
            "comparison tokenizer must have a distinct tokenizer artifact identity; "
            "a self-comparison is not an equivalence audit"
        )
    audited_documents: list[dict[str, Any]] = []
    for primary in primary_scanned:
        document_index = int(primary["source_document_index"])
        document = documents[document_index]
        comparison_ids = _tokenize_document(
            comparison_tokenizer,
            document.text,
            max_document_tokens=int(protocol["max_document_tokens"]),
        )
        comparison_sha256 = _tensor_content_sha256(
            torch.tensor(comparison_ids, dtype=torch.long)
        )
        if len(comparison_ids) != int(primary["token_count"]) or (
            comparison_sha256 != primary["token_ids_sha256"]
        ):
            raise ValueError(
                "comparison tokenizer token IDs differ for selected source "
                f"document {document_index} (file={document.source_file_index}, "
                f"row={document.source_row_index})"
            )
        audited_documents.append(
            {
                "source_document_index": document_index,
                "source_document_group_sha256": document.source_group_sha256,
                "token_count": len(comparison_ids),
                "token_ids_sha256": comparison_sha256,
            }
        )

    if protocol["packing"] == "within_document":
        comparison_input_ids, comparison_windows, comparison_scanned = (
            _build_within_document_windows(
                comparison_tokenizer,
                documents,
                sources,
                order,
                sequence_length=int(protocol["sequence_length"]),
                window_count=int(protocol["window_count"]),
                max_windows_per_document=protocol["max_windows_per_document"],
                max_document_tokens=int(protocol["max_document_tokens"]),
            )
        )
    else:
        comparison_input_ids, comparison_windows, comparison_scanned = (
            _build_concatenated_windows(
                comparison_tokenizer,
                documents,
                sources,
                order,
                sequence_length=int(protocol["sequence_length"]),
                window_count=int(protocol["window_count"]),
                max_document_tokens=int(protocol["max_document_tokens"]),
            )
        )
    if comparison_scanned != list(primary_scanned):
        raise ValueError("comparison tokenizer changes selected-document provenance")
    if comparison_windows != list(primary_windows) or not torch.equal(
        comparison_input_ids, primary_input_ids
    ):
        raise ValueError("comparison tokenizer changes frozen window token IDs")
    payload = {
        "schema": TOKENIZER_EQUIVALENCE_SCHEMA,
        "scope": _TOKENIZER_EQUIVALENCE_SCOPE,
        "corpus_wide": False,
        "distinct_tokenizer_artifact_identities": True,
        "comparison_tokenizer": dict(comparison_binding),
        "audited_documents": audited_documents,
        "audited_document_set_sha256": _canonical_sha256(audited_documents),
        "document_count": len(audited_documents),
        "exact_document_token_ids_equal": True,
        "exact_window_input_ids_equal": True,
        "comparison_tensor_content_sha256": _tensor_content_sha256(
            comparison_input_ids
        ),
    }
    return {
        **payload,
        "semantic_sha256": _canonical_sha256(_tokenizer_equivalence_payload(payload)),
    }


def create_frozen_text_windows(
    *,
    output_dir: Path,
    tokenizer_manifest_path: Path,
    comparison_tokenizer_manifest_path: Path | None = None,
    data_sources: Sequence[str | Path],
    sequence_length: int,
    window_count: int,
    seed: int,
    text_field: str = "text",
    text_mode: Literal["file", "line"] = "file",
    data_format: Literal["auto", "json", "parquet", "text"] = "auto",
    packing: Literal["within_document", "concatenate"] = "within_document",
    document_selection: Literal[
        "seeded_shuffle", "input_order", "explicit"
    ] = "seeded_shuffle",
    document_indices: Sequence[int] | None = None,
    max_documents: int | None = None,
    max_windows_per_document: int | None = 1,
    max_source_record_bytes: int = _DEFAULT_MAX_SOURCE_RECORD_BYTES,
    max_nonstreaming_source_bytes: int = _DEFAULT_MAX_NONSTREAMING_SOURCE_BYTES,
    max_source_files: int = _DEFAULT_MAX_SOURCE_FILES,
    max_document_bytes: int = _DEFAULT_MAX_DOCUMENT_BYTES,
    max_retained_text_bytes: int = _DEFAULT_MAX_RETAINED_TEXT_BYTES,
    max_document_tokens: int = _DEFAULT_MAX_DOCUMENT_TOKENS,
    parquet_batch_rows: int = _DEFAULT_PARQUET_BATCH_ROWS,
    max_parquet_row_group_bytes: int = _DEFAULT_MAX_PARQUET_ROW_GROUP_BYTES,
    use_fast: bool = True,
    trust_local_code: bool = False,
    evidentiary: bool = True,
    repo_root: Path | None = None,
    model_artifact_verification_cache: ModelArtifactVerificationCache | None = None,
) -> Path:
    """Freeze exact ``[window_count, sequence_length]`` local token windows."""

    sequence_length = _positive_int(sequence_length, label="sequence_length")
    window_count = _positive_int(window_count, label="window_count")
    max_source_record_bytes = _positive_int(
        max_source_record_bytes, label="max_source_record_bytes"
    )
    max_nonstreaming_source_bytes = _positive_int(
        max_nonstreaming_source_bytes, label="max_nonstreaming_source_bytes"
    )
    max_source_files = _positive_int(max_source_files, label="max_source_files")
    max_document_bytes = _positive_int(max_document_bytes, label="max_document_bytes")
    max_retained_text_bytes = _positive_int(
        max_retained_text_bytes, label="max_retained_text_bytes"
    )
    max_document_tokens = _positive_int(
        max_document_tokens, label="max_document_tokens"
    )
    parquet_batch_rows = _positive_int(parquet_batch_rows, label="parquet_batch_rows")
    max_parquet_row_group_bytes = _positive_int(
        max_parquet_row_group_bytes, label="max_parquet_row_group_bytes"
    )
    if not all(
        isinstance(value, bool) for value in (use_fast, trust_local_code, evidentiary)
    ):
        raise TypeError("use_fast, trust_local_code, and evidentiary must be booleans")
    if packing not in _PACKING_MODES:
        raise ValueError(f"unsupported packing mode {packing!r}")
    if max_windows_per_document is not None:
        max_windows_per_document = _positive_int(
            max_windows_per_document, label="max_windows_per_document"
        )
    if packing == "concatenate":
        if max_windows_per_document not in {None, 1}:
            raise ValueError(
                "max_windows_per_document applies only to within_document packing"
            )
        max_windows_per_document = None
    max_documents_origin = "explicit"
    if max_documents is None:
        max_documents_origin = "automatic_window_scaled"
        if document_selection == "explicit" and document_indices:
            max_documents = len(document_indices)
        else:
            # This cap bounds retained source text while the loader still
            # scans/counts every row and computes the exact global seeded top-K.
            max_documents = max(1024, 4 * window_count)
    max_documents = _positive_int(max_documents, label="max_documents")
    git = _git_provenance(repo_root)
    if evidentiary and not git["available"]:
        raise RuntimeError("evidentiary mode requires a Git worktree")
    if evidentiary and (
        git["tracked_worktree_dirty"]
        or git["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "tracked worktree is dirty; evidentiary frozen windows require a clean "
            "commit or an immutable clean worktree"
        )
    if evidentiary and git["untracked_executable_files"]:
        paths = [row["path"] for row in git["untracked_executable_files"][:8]]
        raise RuntimeError(
            "evidentiary frozen windows forbid untracked executable code; "
            f"untracked paths include {paths}"
        )
    if evidentiary:
        _require_evidentiary_execution_identity(git)

    destination = Path(output_dir).resolve()
    destination_preexisted = destination.exists()
    manifest_path = destination / "manifest.json"
    if destination_preexisted:
        if not destination.is_dir():
            raise NotADirectoryError(destination)
        if any(destination.iterdir()):
            raise FileExistsError(
                "refusing to overwrite or mix artifacts in non-empty output "
                f"directory {destination}"
            )

    verified_tokenizer = load_and_verify_model_artifact_manifest(
        tokenizer_manifest_path,
        require_tokenizer_assets=True,
        verification_cache=model_artifact_verification_cache,
    )
    tokenizer, tokenizer_runtime = _load_local_tokenizer(
        verified_tokenizer.model_root,
        use_fast=use_fast,
        trust_local_code=trust_local_code,
    )
    specifications, files = _resolve_data_files(
        data_sources,
        data_format=data_format,
        max_source_files=max_source_files,
    )
    (
        documents,
        per_file_counts,
        document_count_total,
        order,
        retained_text_bytes,
    ) = _load_selected_documents(
        files,
        text_field=text_field,
        text_mode=text_mode,
        selection_mode=document_selection,
        seed=seed,
        explicit_indices=document_indices,
        max_documents=max_documents,
        max_source_record_bytes=max_source_record_bytes,
        max_nonstreaming_source_bytes=max_nonstreaming_source_bytes,
        max_document_bytes=max_document_bytes,
        max_retained_text_bytes=max_retained_text_bytes,
        parquet_batch_rows=parquet_batch_rows,
        max_parquet_row_group_bytes=max_parquet_row_group_bytes,
    )
    if document_selection == "explicit" and order != list(document_indices or []):
        raise AssertionError("effective explicit document order changed")
    if packing == "within_document":
        input_ids, windows, scanned = _build_within_document_windows(
            tokenizer,
            documents,
            files,
            order,
            sequence_length=sequence_length,
            window_count=window_count,
            max_windows_per_document=max_windows_per_document,
            max_document_tokens=max_document_tokens,
        )
    else:
        input_ids, windows, scanned = _build_concatenated_windows(
            tokenizer,
            documents,
            files,
            order,
            sequence_length=sequence_length,
            window_count=window_count,
            max_document_tokens=max_document_tokens,
        )
    if input_ids.dtype != torch.long or tuple(input_ids.shape) != (
        window_count,
        sequence_length,
    ):
        raise AssertionError("internal frozen-window shape/dtype contract failed")

    data_rows = _data_file_rows(files, per_file_counts)
    protocol = {
        "sequence_length": sequence_length,
        "window_count": window_count,
        "seed": int(seed),
        "packing": packing,
        "document_selection": document_selection,
        "document_indices": (list(order) if document_selection == "explicit" else None),
        "max_documents": max_documents,
        "max_documents_origin": max_documents_origin,
        "max_windows_per_document": max_windows_per_document,
        "max_source_record_bytes": max_source_record_bytes,
        "max_nonstreaming_source_bytes": max_nonstreaming_source_bytes,
        "max_source_files": max_source_files,
        "max_document_bytes": max_document_bytes,
        "max_retained_text_bytes": max_retained_text_bytes,
        "max_document_tokens": max_document_tokens,
        "parquet_batch_rows": parquet_batch_rows,
        "max_parquet_row_group_bytes": max_parquet_row_group_bytes,
        "text_field": text_field,
        "text_mode": text_mode,
        "data_format": data_format,
        "add_special_tokens": False,
        "document_separator": None,
        "window_stride_tokens": sequence_length,
        "within_document_window_order": "nonoverlapping_prefix",
    }
    primary_binding = _tokenizer_binding(
        verified_tokenizer,
        use_fast=use_fast,
        trust_local_code=trust_local_code,
        runtime=tokenizer_runtime,
    )
    tokenizer_equivalence = None
    if comparison_tokenizer_manifest_path is not None:
        verified_comparison = load_and_verify_model_artifact_manifest(
            comparison_tokenizer_manifest_path,
            require_tokenizer_assets=True,
            verification_cache=model_artifact_verification_cache,
        )
        comparison_tokenizer, comparison_runtime = _load_local_tokenizer(
            verified_comparison.model_root,
            use_fast=use_fast,
            trust_local_code=trust_local_code,
        )
        comparison_binding = _tokenizer_binding(
            verified_comparison,
            use_fast=use_fast,
            trust_local_code=trust_local_code,
            runtime=comparison_runtime,
        )
        tokenizer_equivalence = _audit_comparison_tokenizer(
            comparison_tokenizer,
            primary_binding=primary_binding,
            comparison_binding=comparison_binding,
            documents=documents,
            sources=files,
            order=order,
            protocol=protocol,
            primary_input_ids=input_ids,
            primary_windows=windows,
            primary_scanned=scanned,
        )
    manifest: dict[str, Any] = {
        "schema": FROZEN_TEXT_WINDOWS_SCHEMA,
        "semantic_sha256": "",
        "protocol": protocol,
        "git": {"evidentiary": bool(evidentiary), **git},
        "tokenizer": primary_binding,
        "tokenizer_equivalence": tokenizer_equivalence,
        "data": {
            "source_specifications": specifications,
            "files": data_rows,
            "file_set_sha256": _data_file_set_sha256(data_rows),
            "document_count_total": document_count_total,
            "retained_document_count": len(documents),
            "retained_text_bytes": retained_text_bytes,
            "scanned_documents": scanned,
        },
        "tensor": {
            "path": "",
            "serialization": "torch.save_tensor",
            "dtype": "torch.int64",
            "shape": [window_count, sequence_length],
            "size_bytes": 0,
            "file_sha256": "",
            "content_sha256": _tensor_content_sha256(input_ids),
        },
        "windows": windows,
        "software": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
        },
    }
    semantic_sha256 = _canonical_sha256(_semantic_payload(manifest))
    manifest["semantic_sha256"] = semantic_sha256
    tensor_name = f"input_ids.{semantic_sha256[:16]}.pt"
    tensor_path = destination / tensor_name
    if tensor_path.exists():
        raise FileExistsError(f"refusing to overwrite {tensor_path}")

    buffer = io.BytesIO()
    torch.save(input_ids.cpu(), buffer)
    tensor_bytes = buffer.getvalue()
    manifest["tensor"].update(
        {
            "path": tensor_name,
            "size_bytes": len(tensor_bytes),
            "file_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
        }
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    stable_specifications, stable_files = _resolve_data_files(
        specifications,
        data_format=data_format,
        max_source_files=max_source_files,
    )
    if stable_specifications != specifications or stable_files != files:
        raise RuntimeError(
            "local data sources changed while frozen windows were being created"
        )
    _verify_tokenizer_snapshot(verified_tokenizer, label="primary tokenizer")
    if comparison_tokenizer_manifest_path is not None:
        _verify_tokenizer_snapshot(
            verified_comparison,
            label="comparison tokenizer",
        )
    if _git_provenance(repo_root) != git:
        raise RuntimeError("Git worktree changed while frozen windows were created")
    if evidentiary:
        _require_evidentiary_execution_identity(git)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            "refusing to overwrite or mix artifacts in non-empty output "
            f"directory {destination}"
        )
    tensor_created = False
    manifest_created = False
    try:
        # Exclusive creation, rather than check-then-rename, makes the
        # no-overwrite contract safe against concurrent producers too.
        with tensor_path.open("xb") as handle:
            tensor_created = True
            handle.write(tensor_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with manifest_path.open("xb") as handle:
            manifest_created = True
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(destination)
    except BaseException:
        # Remove only files this invocation created; pre-existing targets are
        # never unlinked or overwritten.
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        if tensor_created:
            tensor_path.unlink(missing_ok=True)
        if manifest_created or tensor_created:
            try:
                _fsync_directory(destination)
            except OSError:
                pass
        if not destination_preexisted:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    return manifest_path


def _validate_manifest_schema(record: Mapping[str, Any]) -> None:
    _require_exact_keys(
        record,
        {
            "schema",
            "semantic_sha256",
            "protocol",
            "git",
            "tokenizer",
            "tokenizer_equivalence",
            "data",
            "tensor",
            "windows",
            "software",
        },
        label="manifest",
    )
    if record["schema"] != FROZEN_TEXT_WINDOWS_SCHEMA:
        raise ValueError("unsupported frozen-window manifest schema")
    _validate_sha256(record["semantic_sha256"], label="semantic_sha256")
    protocol = _require_exact_keys(
        record["protocol"],
        {
            "sequence_length",
            "window_count",
            "seed",
            "packing",
            "document_selection",
            "document_indices",
            "max_documents",
            "max_documents_origin",
            "max_windows_per_document",
            "max_source_record_bytes",
            "max_nonstreaming_source_bytes",
            "max_source_files",
            "max_document_bytes",
            "max_retained_text_bytes",
            "max_document_tokens",
            "parquet_batch_rows",
            "max_parquet_row_group_bytes",
            "text_field",
            "text_mode",
            "data_format",
            "add_special_tokens",
            "document_separator",
            "window_stride_tokens",
            "within_document_window_order",
        },
        label="protocol",
    )
    _positive_int(protocol["sequence_length"], label="protocol.sequence_length")
    _positive_int(protocol["window_count"], label="protocol.window_count")
    _nonnegative_int(protocol["seed"], label="protocol.seed")
    for key in (
        "max_source_record_bytes",
        "max_nonstreaming_source_bytes",
        "max_source_files",
        "max_document_bytes",
        "max_retained_text_bytes",
        "max_document_tokens",
        "parquet_batch_rows",
        "max_parquet_row_group_bytes",
    ):
        _positive_int(protocol[key], label=f"protocol.{key}")
    if protocol["packing"] not in _PACKING_MODES:
        raise ValueError("protocol.packing is invalid")
    if protocol["document_selection"] not in _SELECTION_MODES:
        raise ValueError("protocol.document_selection is invalid")
    if protocol["text_mode"] not in _TEXT_MODES:
        raise ValueError("protocol.text_mode is invalid")
    if protocol["data_format"] not in _DATA_FORMATS:
        raise ValueError("protocol.data_format is invalid")
    if protocol["add_special_tokens"] is not False:
        raise ValueError("protocol.add_special_tokens must be false")
    if protocol["document_separator"] is not None:
        raise ValueError("protocol.document_separator must be null")
    stride = _positive_int(
        protocol["window_stride_tokens"], label="protocol.window_stride_tokens"
    )
    if stride != protocol["sequence_length"]:
        raise ValueError("protocol.window_stride_tokens must equal sequence_length")
    if protocol["within_document_window_order"] != "nonoverlapping_prefix":
        raise ValueError("protocol.within_document_window_order is invalid")
    if not isinstance(protocol["text_field"], str) or not protocol["text_field"]:
        raise ValueError("protocol.text_field must be a non-empty string")
    indices = protocol["document_indices"]
    if indices is not None:
        if not isinstance(indices, list) or not indices:
            raise ValueError("protocol.document_indices must be null or non-empty")
        normalized_indices = [
            _nonnegative_int(value, label="protocol.document_indices[]")
            for value in indices
        ]
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("protocol.document_indices contains duplicates")
    if protocol["document_selection"] == "explicit" and indices is None:
        raise ValueError("explicit selection requires protocol.document_indices")
    if protocol["document_selection"] != "explicit" and indices is not None:
        raise ValueError("non-explicit selection cannot carry document_indices")
    maximum_documents = _positive_int(
        protocol["max_documents"], label="protocol.max_documents"
    )
    if indices is not None and len(indices) > maximum_documents:
        raise ValueError(
            "protocol.document_indices exceeds max_documents; truncation is invalid"
        )
    if protocol["max_windows_per_document"] is not None:
        _positive_int(
            protocol["max_windows_per_document"],
            label="protocol.max_windows_per_document",
        )
    if protocol["max_documents_origin"] not in {
        "explicit",
        "automatic_window_scaled",
    }:
        raise ValueError("protocol.max_documents_origin is invalid")
    if protocol["max_documents_origin"] == "automatic_window_scaled":
        expected_cap = (
            len(indices)
            if protocol["document_selection"] == "explicit" and indices
            else max(1024, 4 * int(protocol["window_count"]))
        )
        if protocol["max_documents"] != expected_cap:
            raise ValueError("automatic max_documents does not match its protocol")
    if (
        protocol["packing"] == "concatenate"
        and protocol["max_windows_per_document"] is not None
    ):
        raise ValueError("concatenate packing cannot carry max_windows_per_document")

    git = _require_exact_keys(
        record["git"],
        {
            "evidentiary",
            "available",
            "repo_root",
            "commit",
            "tracked_worktree_dirty",
            "tracked_diff_sha256",
            "untracked_executable_files",
            "untracked_executable_set_sha256",
        },
        label="git",
    )
    if not isinstance(git["evidentiary"], bool) or not isinstance(
        git["available"], bool
    ):
        raise ValueError("git evidentiary/available flags must be booleans")
    if git["available"]:
        if (
            not isinstance(git["repo_root"], str)
            or not os.path.isabs(git["repo_root"])
            or not isinstance(git["commit"], str)
            or len(git["commit"]) != 40
            or any(character not in "0123456789abcdef" for character in git["commit"])
        ):
            raise ValueError("available Git provenance requires repo_root and commit")
        if not isinstance(git["tracked_worktree_dirty"], bool):
            raise ValueError("tracked_worktree_dirty must be boolean")
        _validate_sha256(git["tracked_diff_sha256"], label="git.tracked_diff_sha256")
    elif any(
        git[key] is not None
        for key in (
            "repo_root",
            "commit",
            "tracked_worktree_dirty",
            "tracked_diff_sha256",
        )
    ):
        raise ValueError("unavailable Git provenance fields must be null")
    untracked = git["untracked_executable_files"]
    if not isinstance(untracked, list):
        raise ValueError("git.untracked_executable_files must be an array")
    untracked_paths: list[str] = []
    for index, raw in enumerate(untracked):
        row = _require_exact_keys(
            raw,
            {"path", "size_bytes", "sha256"},
            label=f"git.untracked_executable_files[{index}]",
        )
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError("untracked executable path must be non-empty")
        untracked_path = Path(row["path"])
        if untracked_path.is_absolute() or ".." in untracked_path.parts:
            raise ValueError("untracked executable paths must stay inside the repo")
        untracked_paths.append(row["path"])
        _nonnegative_int(row["size_bytes"], label="untracked executable size")
        _validate_sha256(row["sha256"], label="untracked executable sha256")
    if untracked != sorted(untracked, key=lambda row: row["path"]):
        raise ValueError("untracked executable records must be path-sorted")
    if len(set(untracked_paths)) != len(untracked_paths):
        raise ValueError("untracked executable paths must be unique")
    _validate_sha256(
        git["untracked_executable_set_sha256"],
        label="git.untracked_executable_set_sha256",
    )
    if _canonical_sha256(untracked) != git["untracked_executable_set_sha256"]:
        raise ValueError("untracked executable-set sha256 changed")
    if git["evidentiary"] and (
        not git["available"]
        or git["tracked_worktree_dirty"]
        or bool(untracked)
        or git["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("evidentiary manifest does not record a clean Git worktree")
    tokenizer = _require_exact_keys(
        record["tokenizer"],
        {
            "manifest_path",
            "manifest_file_sha256",
            "artifact_set_sha256",
            "tokenizer_artifact_set_sha256",
            "model_root",
            "loading",
            "runtime",
        },
        label="tokenizer",
    )
    _validate_sha256(
        tokenizer["manifest_file_sha256"], label="tokenizer.manifest_file_sha256"
    )
    _validate_sha256(
        tokenizer["artifact_set_sha256"], label="tokenizer.artifact_set_sha256"
    )
    _validate_sha256(
        tokenizer["tokenizer_artifact_set_sha256"],
        label="tokenizer.tokenizer_artifact_set_sha256",
    )
    if not all(
        isinstance(tokenizer[key], str) and tokenizer[key]
        for key in ("manifest_path", "model_root")
    ):
        raise ValueError("tokenizer manifest_path/model_root must be non-empty strings")
    if not all(
        os.path.isabs(tokenizer[key]) for key in ("manifest_path", "model_root")
    ):
        raise ValueError("tokenizer manifest_path/model_root must be absolute")
    loading = _require_exact_keys(
        tokenizer["loading"],
        {"local_files_only", "use_fast", "trust_local_code"},
        label="tokenizer.loading",
    )
    if loading["local_files_only"] is not True:
        raise ValueError("tokenizer.loading.local_files_only must be true")
    if not isinstance(loading["use_fast"], bool) or not isinstance(
        loading["trust_local_code"], bool
    ):
        raise ValueError("tokenizer loading flags must be booleans")
    runtime = _require_exact_keys(
        tokenizer["runtime"],
        {
            "tokenizer_class",
            "is_fast",
            "vocabulary_size",
            "transformers_version",
            "tokenizers_version",
        },
        label="tokenizer.runtime",
    )
    if not all(
        isinstance(runtime[key], str) and runtime[key]
        for key in (
            "tokenizer_class",
            "transformers_version",
            "tokenizers_version",
        )
    ):
        raise ValueError("tokenizer runtime string identities cannot be empty")
    if not isinstance(runtime["is_fast"], bool):
        raise ValueError("tokenizer.runtime.is_fast must be boolean")
    _positive_int(runtime["vocabulary_size"], label="tokenizer.runtime.vocabulary_size")
    if loading["use_fast"] and not runtime["is_fast"]:
        raise ValueError(
            "manifest requested a fast tokenizer but records a slow runtime"
        )
    equivalence = record["tokenizer_equivalence"]
    if equivalence is not None:
        audit = _require_exact_keys(
            equivalence,
            {
                "schema",
                "scope",
                "corpus_wide",
                "distinct_tokenizer_artifact_identities",
                "comparison_tokenizer",
                "audited_documents",
                "audited_document_set_sha256",
                "document_count",
                "exact_document_token_ids_equal",
                "exact_window_input_ids_equal",
                "comparison_tensor_content_sha256",
                "semantic_sha256",
            },
            label="tokenizer_equivalence",
        )
        if audit["schema"] != TOKENIZER_EQUIVALENCE_SCHEMA:
            raise ValueError("unsupported tokenizer-equivalence schema")
        if (
            audit["scope"] != _TOKENIZER_EQUIVALENCE_SCOPE
            or audit["corpus_wide"] is not False
            or audit["distinct_tokenizer_artifact_identities"] is not True
        ):
            raise ValueError("tokenizer-equivalence scope contract differs")
        comparison = _require_exact_keys(
            audit["comparison_tokenizer"],
            {
                "manifest_path",
                "manifest_file_sha256",
                "artifact_set_sha256",
                "tokenizer_artifact_set_sha256",
                "model_root",
                "loading",
                "runtime",
            },
            label="tokenizer_equivalence.comparison_tokenizer",
        )
        for key in (
            "manifest_file_sha256",
            "artifact_set_sha256",
            "tokenizer_artifact_set_sha256",
        ):
            _validate_sha256(comparison[key], label=f"comparison_tokenizer.{key}")
        if (
            comparison["tokenizer_artifact_set_sha256"]
            == tokenizer["tokenizer_artifact_set_sha256"]
        ):
            raise ValueError("tokenizer-equivalence audit is a self-comparison")
        if not all(
            isinstance(comparison[key], str) and comparison[key]
            for key in ("manifest_path", "model_root")
        ):
            raise ValueError("comparison-tokenizer paths must be non-empty strings")
        if not all(
            os.path.isabs(comparison[key]) for key in ("manifest_path", "model_root")
        ):
            raise ValueError("comparison-tokenizer paths must be absolute")
        comparison_loading = _require_exact_keys(
            comparison["loading"],
            {"local_files_only", "use_fast", "trust_local_code"},
            label="comparison_tokenizer.loading",
        )
        if comparison_loading != loading:
            raise ValueError("primary and comparison tokenizer loading policies differ")
        comparison_runtime = _require_exact_keys(
            comparison["runtime"],
            {
                "tokenizer_class",
                "is_fast",
                "vocabulary_size",
                "transformers_version",
                "tokenizers_version",
            },
            label="comparison_tokenizer.runtime",
        )
        if not all(
            isinstance(comparison_runtime[key], str) and comparison_runtime[key]
            for key in (
                "tokenizer_class",
                "transformers_version",
                "tokenizers_version",
            )
        ):
            raise ValueError("comparison tokenizer runtime identity is invalid")
        if not isinstance(comparison_runtime["is_fast"], bool):
            raise ValueError("comparison tokenizer is_fast must be boolean")
        _positive_int(
            comparison_runtime["vocabulary_size"],
            label="comparison_tokenizer.runtime.vocabulary_size",
        )
        audited = audit["audited_documents"]
        if not isinstance(audited, list) or not audited:
            raise ValueError("tokenizer equivalence must audit at least one document")
        audit_document_count = _positive_int(
            audit["document_count"], label="tokenizer_equivalence.document_count"
        )
        if audit_document_count != len(audited):
            raise ValueError("tokenizer-equivalence document_count differs")
        audited_keys = {
            "source_document_index",
            "source_document_group_sha256",
            "token_count",
            "token_ids_sha256",
        }
        for index, value in enumerate(audited):
            document = _require_exact_keys(
                value,
                audited_keys,
                label=f"tokenizer_equivalence.audited_documents[{index}]",
            )
            _nonnegative_int(
                document["source_document_index"],
                label="equivalence.source_document_index",
            )
            _nonnegative_int(document["token_count"], label="equivalence.token_count")
            _validate_sha256(
                document["source_document_group_sha256"],
                label="equivalence.source_document_group_sha256",
            )
            _validate_sha256(
                document["token_ids_sha256"],
                label="equivalence.token_ids_sha256",
            )
        for key in (
            "audited_document_set_sha256",
            "comparison_tensor_content_sha256",
            "semantic_sha256",
        ):
            _validate_sha256(audit[key], label=f"tokenizer_equivalence.{key}")
        if _canonical_sha256(audited) != audit["audited_document_set_sha256"]:
            raise ValueError("tokenizer-equivalence audited-document hash changed")
        if (
            audit["exact_document_token_ids_equal"] is not True
            or audit["exact_window_input_ids_equal"] is not True
        ):
            raise ValueError("tokenizer-equivalence success flags must be true")
        if (
            _canonical_sha256(_tokenizer_equivalence_payload(audit))
            != audit["semantic_sha256"]
        ):
            raise ValueError("tokenizer-equivalence semantic sha256 changed")
    data = _require_exact_keys(
        record["data"],
        {
            "source_specifications",
            "files",
            "file_set_sha256",
            "document_count_total",
            "retained_document_count",
            "retained_text_bytes",
            "scanned_documents",
        },
        label="data",
    )
    _validate_sha256(data["file_set_sha256"], label="data.file_set_sha256")
    _positive_int(data["document_count_total"], label="data.document_count_total")
    retained_document_count = _positive_int(
        data["retained_document_count"], label="data.retained_document_count"
    )
    retained_text_bytes = _nonnegative_int(
        data["retained_text_bytes"], label="data.retained_text_bytes"
    )
    if retained_document_count > protocol["max_documents"]:
        raise ValueError("retained document count exceeds protocol.max_documents")
    if indices is not None and retained_document_count != len(indices):
        raise ValueError(
            "explicit protocol retained-document count differs from document_indices"
        )
    if indices is None and retained_document_count != min(
        data["document_count_total"], protocol["max_documents"]
    ):
        raise ValueError("retained document count differs from the selection cap")
    if retained_text_bytes > protocol["max_retained_text_bytes"]:
        raise ValueError("retained text bytes exceeds its protocol ceiling")
    specifications = data["source_specifications"]
    if (
        not isinstance(specifications, list)
        or not specifications
        or not all(
            isinstance(value, str) and os.path.isabs(value) for value in specifications
        )
    ):
        raise ValueError("data.source_specifications must be non-empty absolute paths")
    file_rows = data["files"]
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("data.files must be a non-empty array")
    if len(file_rows) > protocol["max_source_files"]:
        raise ValueError("data file count exceeds protocol.max_source_files")
    for index, value in enumerate(file_rows):
        row = _require_exact_keys(
            value,
            {
                "source_index",
                "logical_path",
                "resolved_path",
                "is_symlink",
                "size_bytes",
                "sha256",
                "format",
                "document_count",
            },
            label=f"data.files[{index}]",
        )
        source_index = _nonnegative_int(
            row["source_index"], label=f"data.files[{index}].source_index"
        )
        if source_index != index:
            raise ValueError("data file source indices must be contiguous and ordered")
        if not all(
            isinstance(row[key], str) and os.path.isabs(row[key])
            for key in ("logical_path", "resolved_path")
        ):
            raise ValueError("data file paths must be absolute strings")
        if not isinstance(row["is_symlink"], bool):
            raise ValueError("data file is_symlink must be boolean")
        _nonnegative_int(row["size_bytes"], label=f"data.files[{index}].size_bytes")
        _validate_sha256(row["sha256"], label=f"data.files[{index}].sha256")
        if row["format"] not in _DATA_FORMATS - {"auto"}:
            raise ValueError("resolved data file format must be explicit")
        _nonnegative_int(
            row["document_count"], label=f"data.files[{index}].document_count"
        )
    if (
        sum(int(row["document_count"]) for row in file_rows)
        != data["document_count_total"]
    ):
        raise ValueError("data document_count_total differs from per-file counts")
    scanned = data["scanned_documents"]
    if not isinstance(scanned, list) or not scanned:
        raise ValueError("data.scanned_documents must be a non-empty array")
    scanned_keys = {
        "selected_order",
        "source_document_index",
        "source_file_index",
        "source_row_index",
        "document_text_sha256",
        "source_document_group_sha256",
        "token_count",
        "token_ids_sha256",
        "status",
        "emitted_token_ranges",
    }
    for index, value in enumerate(scanned):
        row = _require_exact_keys(
            value, scanned_keys, label=f"data.scanned_documents[{index}]"
        )
        selected_order = _nonnegative_int(
            row["selected_order"], label="scanned.selected_order"
        )
        if selected_order != index:
            raise ValueError("scanned document selected_order must be contiguous")
        _nonnegative_int(
            row["source_document_index"], label="scanned.source_document_index"
        )
        source_file_index = _nonnegative_int(
            row["source_file_index"], label="scanned.source_file_index"
        )
        if source_file_index >= len(file_rows):
            raise ValueError("scanned source_file_index is out of range")
        _nonnegative_int(row["source_row_index"], label="scanned.source_row_index")
        token_count = _nonnegative_int(row["token_count"], label="scanned.token_count")
        if token_count > protocol["max_document_tokens"]:
            raise ValueError("scanned token count exceeds max_document_tokens")
        _validate_sha256(row["document_text_sha256"], label="scanned.document_text")
        _validate_sha256(
            row["source_document_group_sha256"], label="scanned.source_group"
        )
        _validate_sha256(row["token_ids_sha256"], label="scanned.token_ids")
        if row["status"] not in {"used", "blank", "short", "unused"}:
            raise ValueError("scanned document status is invalid")
        ranges = row["emitted_token_ranges"]
        if not isinstance(ranges, list):
            raise ValueError("scanned emitted_token_ranges must be an array")
        for pair in ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each emitted token range must be [start, end]")
            start = _nonnegative_int(pair[0], label="emitted range start")
            end = _nonnegative_int(pair[1], label="emitted range end")
            if end <= start or end > row["token_count"]:
                raise ValueError("emitted token range is invalid")
    tensor = _require_exact_keys(
        record["tensor"],
        {
            "path",
            "serialization",
            "dtype",
            "shape",
            "size_bytes",
            "file_sha256",
            "content_sha256",
        },
        label="tensor",
    )
    _positive_int(tensor["size_bytes"], label="tensor.size_bytes")
    _validate_sha256(tensor["file_sha256"], label="tensor.file_sha256")
    _validate_sha256(tensor["content_sha256"], label="tensor.content_sha256")
    if tensor["serialization"] != "torch.save_tensor" or tensor["dtype"] != (
        "torch.int64"
    ):
        raise ValueError("tensor serialization or dtype contract differs")
    if not isinstance(tensor["shape"], list) or len(tensor["shape"]) != 2:
        raise ValueError("tensor.shape must contain two dimensions")
    normalized_shape = [
        _positive_int(value, label="tensor.shape[]") for value in tensor["shape"]
    ]
    expected_shape = [int(protocol["window_count"]), int(protocol["sequence_length"])]
    if normalized_shape != expected_shape:
        raise ValueError("tensor.shape differs from the protocol")
    if not isinstance(tensor["path"], str) or not tensor["path"]:
        raise ValueError("tensor.path must be a non-empty string")
    windows = record["windows"]
    if not isinstance(windows, list):
        raise ValueError("windows must be an array")
    if len(windows) != int(protocol["window_count"]):
        raise ValueError("window record count differs from protocol")
    window_keys = {
        "window_index",
        "input_ids_sha256",
        "source_document_group_sha256s",
        "source_group_set_sha256",
        "segments",
        "semantic_sha256",
    }
    segment_keys = {
        "source_file_index",
        "source_file_sha256",
        "source_document_index",
        "source_row_index",
        "document_text_sha256",
        "source_document_group_sha256",
        "document_token_start",
        "document_token_end",
        "window_token_start",
        "window_token_end",
    }
    for index, value in enumerate(windows):
        window = _require_exact_keys(value, window_keys, label=f"windows[{index}]")
        window_index = _nonnegative_int(
            window["window_index"], label=f"windows[{index}].window_index"
        )
        if window_index != index:
            raise ValueError("window_index must be contiguous and ordered")
        for key in ("input_ids_sha256", "source_group_set_sha256", "semantic_sha256"):
            _validate_sha256(window[key], label=f"windows[{index}].{key}")
        groups = window["source_document_group_sha256s"]
        if not isinstance(groups, list) or not groups:
            raise ValueError("each window must name at least one source group")
        for group in groups:
            _validate_sha256(group, label=f"windows[{index}].source group")
        if len(set(groups)) != len(groups):
            raise ValueError("window source groups cannot contain duplicates")
        segments = window["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError("each window must contain source segments")
        cursor = 0
        segment_groups: list[str] = []
        for segment_index, raw_segment in enumerate(segments):
            segment = _require_exact_keys(
                raw_segment,
                segment_keys,
                label=f"windows[{index}].segments[{segment_index}]",
            )
            file_index = _nonnegative_int(
                segment["source_file_index"], label="segment.source_file_index"
            )
            if file_index >= len(file_rows):
                raise ValueError("segment source_file_index is out of range")
            for key in (
                "source_file_sha256",
                "document_text_sha256",
                "source_document_group_sha256",
            ):
                _validate_sha256(segment[key], label=f"segment.{key}")
            if segment["source_file_sha256"] != file_rows[file_index]["sha256"]:
                raise ValueError("segment source-file sha256 differs from data.files")
            _nonnegative_int(
                segment["source_document_index"],
                label="segment.source_document_index",
            )
            _nonnegative_int(
                segment["source_row_index"], label="segment.source_row_index"
            )
            document_start = _nonnegative_int(
                segment["document_token_start"], label="segment.document_token_start"
            )
            document_end = _nonnegative_int(
                segment["document_token_end"], label="segment.document_token_end"
            )
            window_start = _nonnegative_int(
                segment["window_token_start"], label="segment.window_token_start"
            )
            window_end = _nonnegative_int(
                segment["window_token_end"], label="segment.window_token_end"
            )
            if window_start != cursor or window_end <= window_start:
                raise ValueError(
                    "window source segments must be contiguous and nonempty"
                )
            if document_end - document_start != window_end - window_start:
                raise ValueError("document/window segment lengths differ")
            cursor = window_end
            group = str(segment["source_document_group_sha256"])
            if group not in segment_groups:
                segment_groups.append(group)
        if cursor != int(protocol["sequence_length"]):
            raise ValueError("window source segments do not cover sequence_length")
        if segment_groups != groups:
            raise ValueError("window source-group list differs from its segments")
        if _canonical_sha256(groups) != window["source_group_set_sha256"]:
            raise ValueError("window source-group set sha256 changed")
        payload = dict(window)
        semantic = payload.pop("semantic_sha256")
        if _canonical_sha256(payload) != semantic:
            raise ValueError("window semantic sha256 changed")
    software = _require_exact_keys(
        record["software"], {"python_version", "torch_version"}, label="software"
    )
    if not all(isinstance(value, str) and value for value in software.values()):
        raise ValueError("software versions must be non-empty strings")


def _safe_tensor_path(manifest_path: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("tensor.path must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise ValueError("tensor.path must name one file beside the manifest")
    path = manifest_path.parent / candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_parent = manifest_path.parent.resolve()
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_parent:
        raise ValueError("tensor path resolves outside the manifest directory")
    return resolved


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot safely load frozen tensor {path}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError("frozen .pt payload must be one tensor")
    if value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 2:
        raise TypeError("frozen .pt tensor must be rank-two CPU torch.int64")
    return value.contiguous()


def verify_frozen_text_windows(
    manifest_path: Path,
    *,
    expected_tensor_path: Path | None = None,
    expected_sequence_length: int | None = None,
    minimum_windows: int | None = None,
    expected_primary_tokenizer_artifact_set_sha256: str | None = None,
    expected_primary_tokenizer_model_root: Path | None = None,
    require_evidentiary: bool = False,
    model_artifact_verification_cache: ModelArtifactVerificationCache | None = None,
    force_model_artifact_rehash: bool = False,
) -> VerifiedFrozenTextWindows:
    """Verify everything and return a normalized, expectation-checked identity.

    The optional model cache avoids rereading multi-shard weights when several
    frozen-window parents bind the same tokenizer/model artifact set.  It never
    caches source-data reconstruction or frozen tensor content verification.
    """

    source = Path(manifest_path).resolve(strict=True)
    record, manifest_snapshot = _strict_json_load_snapshot(source)
    _validate_manifest_schema(record)
    if _canonical_sha256(_semantic_payload(record)) != record["semantic_sha256"]:
        raise ValueError("overall frozen-window semantic sha256 changed")
    if not isinstance(require_evidentiary, bool):
        raise TypeError("require_evidentiary must be boolean")
    if require_evidentiary and record["git"]["evidentiary"] is not True:
        raise ValueError("frozen-window manifest is exploratory, not evidentiary")
    expected_evidentiary_git: dict[str, Any] | None = None
    expected_verifier_git: dict[str, Any] | None = None
    if record["git"]["evidentiary"]:
        git_record = record["git"]
        expected_evidentiary_git = {
            key: value for key, value in git_record.items() if key != "evidentiary"
        }
        expected_verifier_git = _require_evidentiary_verification_identity(
            expected_evidentiary_git
        )
    protocol = record["protocol"]
    actual_sequence_length = int(protocol["sequence_length"])
    actual_window_count = int(protocol["window_count"])
    if expected_sequence_length is not None:
        expected_sequence_length = _positive_int(
            expected_sequence_length, label="expected_sequence_length"
        )
        if actual_sequence_length != expected_sequence_length:
            raise ValueError(
                "frozen-window sequence length differs from the expected value: "
                f"{actual_sequence_length} != {expected_sequence_length}"
            )
    if minimum_windows is not None:
        minimum_windows = _positive_int(minimum_windows, label="minimum_windows")
        if actual_window_count < minimum_windows:
            raise ValueError(
                "frozen-window count is below the required minimum: "
                f"{actual_window_count} < {minimum_windows}"
            )

    tokenizer_record = record["tokenizer"]
    if expected_primary_tokenizer_artifact_set_sha256 is not None:
        expected_artifact_set = _validate_sha256(
            expected_primary_tokenizer_artifact_set_sha256,
            label="expected_primary_tokenizer_artifact_set_sha256",
        )
        if tokenizer_record["artifact_set_sha256"] != expected_artifact_set:
            raise ValueError("primary tokenizer artifact set differs from expectation")
    if (
        expected_primary_tokenizer_model_root is not None
        and Path(tokenizer_record["model_root"]).resolve()
        != Path(expected_primary_tokenizer_model_root).resolve()
    ):
        raise ValueError("primary tokenizer model root differs from expectation")
    tokenizer_manifest = Path(tokenizer_record["manifest_path"]).resolve(strict=True)
    verified_tokenizer = load_and_verify_model_artifact_manifest(
        tokenizer_manifest,
        expected_model_root=Path(tokenizer_record["model_root"]),
        require_tokenizer_assets=True,
        verification_cache=model_artifact_verification_cache,
        force_rehash=force_model_artifact_rehash,
    )
    if (
        verified_tokenizer.file_sha256 != tokenizer_record["manifest_file_sha256"]
        or verified_tokenizer.artifact_set_sha256
        != tokenizer_record["artifact_set_sha256"]
    ):
        raise ValueError("tokenizer/model artifact manifest or set changed")
    if (
        _tokenizer_artifact_set_sha256(verified_tokenizer)
        != tokenizer_record["tokenizer_artifact_set_sha256"]
    ):
        raise ValueError("primary tokenizer artifact subset changed")
    tokenizer, runtime = _load_local_tokenizer(
        verified_tokenizer.model_root,
        use_fast=bool(tokenizer_record["loading"]["use_fast"]),
        trust_local_code=bool(tokenizer_record["loading"]["trust_local_code"]),
    )
    if runtime != tokenizer_record["runtime"]:
        raise ValueError("tokenizer runtime identity changed")
    equivalence_record = record["tokenizer_equivalence"]
    comparison_tokenizer = None
    verified_comparison = None
    if equivalence_record is not None:
        comparison_record = equivalence_record["comparison_tokenizer"]
        comparison_manifest = Path(comparison_record["manifest_path"]).resolve(
            strict=True
        )
        verified_comparison = load_and_verify_model_artifact_manifest(
            comparison_manifest,
            expected_model_root=Path(comparison_record["model_root"]),
            require_tokenizer_assets=True,
            verification_cache=model_artifact_verification_cache,
            force_rehash=force_model_artifact_rehash,
        )
        if (
            verified_comparison.file_sha256 != comparison_record["manifest_file_sha256"]
            or verified_comparison.artifact_set_sha256
            != comparison_record["artifact_set_sha256"]
        ):
            raise ValueError(
                "comparison tokenizer/model artifact manifest or set changed"
            )
        if (
            _tokenizer_artifact_set_sha256(verified_comparison)
            != comparison_record["tokenizer_artifact_set_sha256"]
        ):
            raise ValueError("comparison tokenizer artifact subset changed")
        comparison_tokenizer, comparison_runtime = _load_local_tokenizer(
            verified_comparison.model_root,
            use_fast=bool(comparison_record["loading"]["use_fast"]),
            trust_local_code=bool(comparison_record["loading"]["trust_local_code"]),
        )
        if comparison_runtime != comparison_record["runtime"]:
            raise ValueError("comparison tokenizer runtime identity changed")

    data = record["data"]
    specifications, files = _resolve_data_files(
        data["source_specifications"],
        data_format=protocol["data_format"],
        max_source_files=int(protocol["max_source_files"]),
    )
    if specifications != data["source_specifications"]:
        raise ValueError("normalized data source specifications changed")
    (
        documents,
        per_file_counts,
        document_count_total,
        order,
        retained_text_bytes,
    ) = _load_selected_documents(
        files,
        text_field=protocol["text_field"],
        text_mode=protocol["text_mode"],
        selection_mode=protocol["document_selection"],
        seed=int(protocol["seed"]),
        explicit_indices=protocol["document_indices"],
        max_documents=int(protocol["max_documents"]),
        max_source_record_bytes=int(protocol["max_source_record_bytes"]),
        max_nonstreaming_source_bytes=int(protocol["max_nonstreaming_source_bytes"]),
        max_document_bytes=int(protocol["max_document_bytes"]),
        max_retained_text_bytes=int(protocol["max_retained_text_bytes"]),
        parquet_batch_rows=int(protocol["parquet_batch_rows"]),
        max_parquet_row_group_bytes=int(protocol["max_parquet_row_group_bytes"]),
    )
    if protocol["document_selection"] == "explicit" and order != list(
        protocol["document_indices"]
    ):
        raise ValueError("effective explicit document order differs from protocol")
    actual_data_rows = _data_file_rows(files, per_file_counts)
    if actual_data_rows != data["files"]:
        raise ValueError("resolved data file identity or document count changed")
    if _data_file_set_sha256(actual_data_rows) != data["file_set_sha256"]:
        raise ValueError("data file-set sha256 changed")
    if document_count_total != data["document_count_total"]:
        raise ValueError("source document count changed")
    if len(documents) != data["retained_document_count"]:
        raise ValueError("retained document count changed")
    if retained_text_bytes != data["retained_text_bytes"]:
        raise ValueError("retained source-text byte count changed")

    tensor_record = record["tensor"]
    tensor_path = _safe_tensor_path(source, tensor_record["path"])
    if expected_tensor_path is not None and tensor_path != Path(
        expected_tensor_path
    ).resolve(strict=True):
        raise ValueError("frozen tensor path differs from the expected input tensor")
    if tensor_path.stat().st_size != tensor_record["size_bytes"]:
        raise ValueError("frozen tensor file size changed")
    if _file_sha256(tensor_path) != tensor_record["file_sha256"]:
        raise ValueError("frozen tensor file sha256 changed")
    input_ids = _load_tensor(tensor_path)
    expected_shape = [
        int(protocol["window_count"]),
        int(protocol["sequence_length"]),
    ]
    if (
        list(input_ids.shape) != expected_shape
        or tensor_record["shape"] != expected_shape
    ):
        raise ValueError("frozen tensor shape differs from the manifest protocol")
    if _tensor_content_sha256(input_ids) != tensor_record["content_sha256"]:
        raise ValueError("frozen tensor semantic content sha256 changed")
    for index, window in enumerate(record["windows"]):
        expected = _window_record(index, input_ids[index].tolist(), window["segments"])
        if expected != window:
            raise ValueError(f"window {index} semantic record changed")

    if protocol["packing"] == "within_document":
        rebuilt, rebuilt_windows, rebuilt_scanned = _build_within_document_windows(
            tokenizer,
            documents,
            files,
            order,
            sequence_length=int(protocol["sequence_length"]),
            window_count=int(protocol["window_count"]),
            max_windows_per_document=protocol["max_windows_per_document"],
            max_document_tokens=int(protocol["max_document_tokens"]),
        )
    else:
        rebuilt, rebuilt_windows, rebuilt_scanned = _build_concatenated_windows(
            tokenizer,
            documents,
            files,
            order,
            sequence_length=int(protocol["sequence_length"]),
            window_count=int(protocol["window_count"]),
            max_document_tokens=int(protocol["max_document_tokens"]),
        )
    if not torch.equal(rebuilt, input_ids):
        raise ValueError("retokenized source documents do not reproduce input_ids")
    if rebuilt_windows != record["windows"]:
        raise ValueError("rebuilt window provenance differs from the manifest")
    if rebuilt_scanned != data["scanned_documents"]:
        raise ValueError("rebuilt document-selection provenance differs from manifest")
    if equivalence_record is not None:
        if comparison_tokenizer is None:
            raise AssertionError("comparison tokenizer was not loaded")
        rebuilt_audit = _audit_comparison_tokenizer(
            comparison_tokenizer,
            primary_binding=tokenizer_record,
            comparison_binding=equivalence_record["comparison_tokenizer"],
            documents=documents,
            sources=files,
            order=order,
            protocol=protocol,
            primary_input_ids=input_ids,
            primary_windows=record["windows"],
            primary_scanned=data["scanned_documents"],
        )
        if rebuilt_audit != equivalence_record:
            raise ValueError("rebuilt tokenizer-equivalence audit differs")
    stable_specifications, stable_files = _resolve_data_files(
        specifications,
        data_format=protocol["data_format"],
        max_source_files=int(protocol["max_source_files"]),
    )
    if stable_specifications != specifications or stable_files != files:
        raise ValueError("local data sources changed during verification")
    _verify_tokenizer_snapshot(verified_tokenizer, label="primary tokenizer")
    if verified_comparison is not None:
        _verify_tokenizer_snapshot(
            verified_comparison,
            label="comparison tokenizer",
        )
    if expected_evidentiary_git is not None:
        final_verifier_git = _require_evidentiary_verification_identity(
            expected_evidentiary_git
        )
        if final_verifier_git != expected_verifier_git:
            raise ValueError(
                "current verifier Git worktree changed during evidentiary verification"
            )
    if _file_snapshot(source) != manifest_snapshot:
        raise ValueError("frozen-window manifest changed during verification")
    immutable_window_groups = tuple(
        tuple(str(group) for group in window["source_document_group_sha256s"])
        for window in record["windows"]
    )
    return VerifiedFrozenTextWindows(
        path=source,
        manifest_file_sha256=manifest_snapshot[1],
        semantic_sha256=str(record["semantic_sha256"]),
        tensor_path=tensor_path,
        tensor_file_sha256=str(tensor_record["file_sha256"]),
        tensor_content_sha256=str(tensor_record["content_sha256"]),
        sequence_length=actual_sequence_length,
        window_count=actual_window_count,
        primary_tokenizer_artifact_set_sha256=str(
            tokenizer_record["artifact_set_sha256"]
        ),
        primary_tokenizer_model_root=verified_tokenizer.model_root,
        data_file_set_sha256=str(data["file_set_sha256"]),
        evidentiary=bool(record["git"]["evidentiary"]),
        source_file_count=len(actual_data_rows),
        source_document_count=document_count_total,
        comparison_tokenizer_equivalence_audited=equivalence_record is not None,
        comparison_tokenizer_equivalence_scope=(
            None if equivalence_record is None else str(equivalence_record["scope"])
        ),
        window_source_document_group_sha256s=immutable_window_groups,
        record=dict(record),
    )


def _parse_document_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "document indices cannot contain empty comma-separated fields"
        )
    try:
        indices = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "document indices must be comma-separated integers"
        ) from exc
    if not indices:
        raise argparse.ArgumentTypeError("document indices cannot be empty")
    return indices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create one immutable window set")
    create.add_argument("output_dir", type=Path)
    create.add_argument("--tokenizer-manifest", type=Path, required=True)
    create.add_argument("--comparison-tokenizer-manifest", type=Path)
    create.add_argument("--data", action="append", required=True)
    create.add_argument("--sequence-length", type=int, required=True)
    create.add_argument("--window-count", type=int, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--text-field", default="text")
    create.add_argument("--text-mode", choices=sorted(_TEXT_MODES), default="file")
    create.add_argument("--data-format", choices=sorted(_DATA_FORMATS), default="auto")
    create.add_argument(
        "--packing", choices=sorted(_PACKING_MODES), default="within_document"
    )
    create.add_argument(
        "--document-selection",
        choices=sorted(_SELECTION_MODES),
        default="seeded_shuffle",
    )
    create.add_argument("--document-indices")
    create.add_argument("--max-documents", type=int)
    create.add_argument("--max-windows-per-document", type=int, default=1)
    create.add_argument(
        "--max-source-record-bytes",
        type=int,
        default=_DEFAULT_MAX_SOURCE_RECORD_BYTES,
    )
    create.add_argument(
        "--max-nonstreaming-source-bytes",
        type=int,
        default=_DEFAULT_MAX_NONSTREAMING_SOURCE_BYTES,
    )
    create.add_argument(
        "--max-source-files",
        type=int,
        default=_DEFAULT_MAX_SOURCE_FILES,
    )
    create.add_argument(
        "--max-document-bytes",
        type=int,
        default=_DEFAULT_MAX_DOCUMENT_BYTES,
    )
    create.add_argument(
        "--max-retained-text-bytes",
        type=int,
        default=_DEFAULT_MAX_RETAINED_TEXT_BYTES,
    )
    create.add_argument(
        "--max-document-tokens",
        type=int,
        default=_DEFAULT_MAX_DOCUMENT_TOKENS,
    )
    create.add_argument(
        "--parquet-batch-rows",
        type=int,
        default=_DEFAULT_PARQUET_BATCH_ROWS,
    )
    create.add_argument(
        "--max-parquet-row-group-bytes",
        type=int,
        default=_DEFAULT_MAX_PARQUET_ROW_GROUP_BYTES,
    )
    create.add_argument("--allow-slow-tokenizer", action="store_true")
    create.add_argument("--trust-local-code", action="store_true")
    create.add_argument(
        "--exploratory",
        action="store_true",
        help="record but do not reject a dirty tracked worktree",
    )
    create.add_argument("--repo-root", type=Path)
    verify = subparsers.add_parser("verify", help="rehash and rebuild one window set")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--expected-tensor", type=Path)
    verify.add_argument("--expected-sequence-length", type=int)
    verify.add_argument("--minimum-windows", type=int)
    verify.add_argument("--expected-primary-tokenizer-artifact-set-sha256")
    verify.add_argument("--expected-primary-tokenizer-model-root", type=Path)
    verify.add_argument("--require-evidentiary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        verified = verify_frozen_text_windows(
            args.manifest,
            expected_tensor_path=args.expected_tensor,
            expected_sequence_length=args.expected_sequence_length,
            minimum_windows=args.minimum_windows,
            expected_primary_tokenizer_artifact_set_sha256=(
                args.expected_primary_tokenizer_artifact_set_sha256
            ),
            expected_primary_tokenizer_model_root=(
                args.expected_primary_tokenizer_model_root
            ),
            require_evidentiary=args.require_evidentiary,
        )
        print(json.dumps(verified.as_dict(), indent=2, sort_keys=True))
        return
    try:
        indices = _parse_document_indices(args.document_indices)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    maximum = args.max_windows_per_document
    if maximum == 0:
        maximum = None
    manifest = create_frozen_text_windows(
        output_dir=args.output_dir,
        tokenizer_manifest_path=args.tokenizer_manifest,
        comparison_tokenizer_manifest_path=args.comparison_tokenizer_manifest,
        data_sources=args.data,
        sequence_length=args.sequence_length,
        window_count=args.window_count,
        seed=args.seed,
        text_field=args.text_field,
        text_mode=args.text_mode,
        data_format=args.data_format,
        packing=args.packing,
        document_selection=args.document_selection,
        document_indices=indices,
        max_documents=args.max_documents,
        max_windows_per_document=maximum,
        max_source_record_bytes=args.max_source_record_bytes,
        max_nonstreaming_source_bytes=args.max_nonstreaming_source_bytes,
        max_source_files=args.max_source_files,
        max_document_bytes=args.max_document_bytes,
        max_retained_text_bytes=args.max_retained_text_bytes,
        max_document_tokens=args.max_document_tokens,
        parquet_batch_rows=args.parquet_batch_rows,
        max_parquet_row_group_bytes=args.max_parquet_row_group_bytes,
        use_fast=not args.allow_slow_tokenizer,
        trust_local_code=args.trust_local_code,
        evidentiary=not args.exploratory,
        repo_root=args.repo_root,
    )
    record = _strict_json_load(manifest)
    print(
        json.dumps(
            {
                "status": "created",
                "manifest_path": str(manifest),
                "manifest_file_sha256": _file_sha256(manifest),
                "semantic_sha256": record["semantic_sha256"],
                "tensor_path": str(manifest.parent / record["tensor"]["path"]),
                "shape": record["tensor"]["shape"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
