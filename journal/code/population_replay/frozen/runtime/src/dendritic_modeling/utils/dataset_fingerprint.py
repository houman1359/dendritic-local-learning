"""Content fingerprints for generated training/evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_FINGERPRINT_DEFINITION = "dataset-content-fingerprint-v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _hash_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.layout != torch.strided:
        raise ValueError(f"Dataset tensors must be strided, got {tensor.layout}")
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    header = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "numel": int(cpu.numel()),
    }
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    raw = cpu.view(torch.uint8).reshape(-1)
    for start in range(0, raw.numel(), _HASH_CHUNK_BYTES):
        digest.update(raw[start : start + _HASH_CHUNK_BYTES].numpy().tobytes())
    return {"kind": "tensor", **header, "sha256": digest.hexdigest()}


def _hash_array(array: np.ndarray) -> dict[str, Any]:
    if array.dtype.hasobject:
        raise ValueError("Object-valued dataset arrays cannot be fingerprinted")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    header = {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "size": int(contiguous.size),
    }
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    view = memoryview(contiguous).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])
    return {"kind": "ndarray", **header, "sha256": digest.hexdigest()}


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Dataset metadata must be finite")
        return result
    return None


def _field_record(value: Any, seen: set[int]) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return _hash_tensor(value)
    if isinstance(value, np.ndarray):
        return _hash_array(value)
    scalar = _scalar(value)
    if scalar is not None or value is None:
        return {"kind": "scalar", "value": scalar}
    if isinstance(value, Dataset):
        return {"kind": "dataset", "value": fingerprint_dataset(value, _seen=seen)}
    if isinstance(value, Mapping):
        if id(value) in seen:
            raise ValueError("Dataset metadata contains a reference cycle")
        seen.add(id(value))
        try:
            children = {
                str(key): _field_record(item, seen)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        finally:
            seen.remove(id(value))
        return {"kind": "mapping", "items": children}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if id(value) in seen:
            raise ValueError("Dataset metadata contains a reference cycle")
        seen.add(id(value))
        try:
            children = [_field_record(item, seen) for item in value]
        finally:
            seen.remove(id(value))
        return {"kind": "sequence", "items": children}
    return {
        "kind": "unsupported",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def fingerprint_dataset(
    dataset: Dataset,
    *,
    _seen: set[int] | None = None,
) -> dict[str, Any]:
    """Hash tensor content and stable metadata from one realized dataset."""

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected torch Dataset, got {type(dataset).__name__}")
    seen = set() if _seen is None else _seen
    if id(dataset) in seen:
        raise ValueError("Dataset graph contains a reference cycle")
    seen.add(id(dataset))
    try:
        fields = {
            str(name): _field_record(value, seen)
            for name, value in sorted(vars(dataset).items())
            if not callable(value)
        }
    finally:
        seen.remove(id(dataset))
    record = {
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "length": len(dataset),
        "fields": fields,
    }
    return {
        "definition": DATASET_FINGERPRINT_DEFINITION,
        **record,
        "content_sha256": _sha256_json(record),
    }


def fingerprint_dataset_splits(**splits: Dataset) -> dict[str, Any]:
    """Return a stable content record for named realized dataset splits."""

    if not splits:
        raise ValueError("At least one dataset split is required")
    records = {
        name: fingerprint_dataset(dataset) for name, dataset in sorted(splits.items())
    }
    payload = {
        "definition": DATASET_FINGERPRINT_DEFINITION,
        "splits": records,
    }
    payload["combined_sha256"] = _sha256_json(payload)
    return payload


__all__ = [
    "DATASET_FINGERPRINT_DEFINITION",
    "fingerprint_dataset",
    "fingerprint_dataset_splits",
]
