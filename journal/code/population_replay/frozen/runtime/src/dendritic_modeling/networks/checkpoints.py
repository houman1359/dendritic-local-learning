"""Safe helpers for strict model-state checkpoint loading."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_BITMASK_SUFFIX = ".__bitmask"
_BITMASK_METADATA_SUFFIX = ".__bitmask_metadata"


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    """Atomically save a PyTorch payload without exposing a partial file."""

    resolved = Path(destination).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save_candidates(
    payloads: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    """Serialize candidate payloads and atomically keep the smallest one.

    Topology encodings have different break-even points: integer indices are
    compact at low fan-in, while row-wise bitmasks can win for denser rows.
    Comparing tensor ``numel`` is insufficient because ``torch.save`` adds
    storage and metadata overhead.  This helper serializes every complete
    candidate payload in the destination directory, compares the actual file
    sizes, and moves the smallest byte-for-byte candidate into place.

    Ties are resolved by sorted candidate name so repeated exports are
    deterministic.  Candidate sizes describe the exact payloads considered;
    the selected file is not reserialized after selection.
    """

    if not payloads:
        raise ValueError("payloads must contain at least one candidate")
    resolved = Path(destination).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[str, Path] = {}
    candidate_bytes: dict[str, int] = {}
    try:
        for candidate in sorted(payloads):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{resolved.name}.{candidate}.",
                suffix=".tmp",
                dir=resolved.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporaries[candidate] = temporary
            torch.save(payloads[candidate], temporary)
            candidate_bytes[candidate] = int(temporary.stat().st_size)
        selected = min(
            candidate_bytes,
            key=lambda candidate: (candidate_bytes[candidate], candidate),
        )
        os.replace(temporaries[selected], resolved)
        temporaries.pop(selected)
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    return {
        "selection_basis": "minimum_complete_torch_serialized_payload_bytes",
        "selected": selected,
        "candidate_bytes": candidate_bytes,
        "output_bytes": int(resolved.stat().st_size),
        "output_sha256": sha256_file(resolved),
    }


def compact_sparse_index_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Pack non-negative sparse topology indices for checkpoint storage.

    Runtime indexed layers use signed 32- or 64-bit integer buffers for broad
    operator compatibility. Checkpoints do not need that overhead: non-negative
    topologies can be stored as unsigned 16- or 32-bit integers and are cast
    back to the destination buffer dtype by ``load_state_dict``. Both signed
    input dtypes normalize to the same packed output for equal index values.
    """

    compact: dict[str, torch.Tensor] = {}
    encoding: dict[str, str] = {}
    for key, value in state_dict.items():
        if key.endswith("._last_forward_param_tensor") or key == (
            "_last_forward_param_tensor"
        ):
            continue
        if key.endswith("connection_indices") and value.dtype in (
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
        ):
            # Re-normalize already compact unsigned checkpoints as well as
            # runtime signed buffers. This makes the codec idempotent and lets
            # validators prove that stored uint topology is canonical.
            indices = value.to(dtype=torch.int64)
            max_index = int(indices.max().item()) if indices.numel() else 0
            min_index = int(indices.min().item()) if indices.numel() else 0
            if min_index >= 0 and max_index <= 65_535:
                value = indices.to(dtype=torch.uint16)
            elif min_index >= 0 and max_index <= 4_294_967_295:
                value = indices.to(dtype=torch.uint32)
            encoding[key] = str(value.dtype).removeprefix("torch.")
        compact[key] = value
    return compact, encoding


def compact_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Encode fixed row-wise topology as bitmasks instead of integer indices.

    Sparse weights are reordered into ascending input-index order so the mask
    contains all information required to reconstruct their alignment. This is
    useful for moderately dense fixed topology, where even uint16 costs two
    index bytes per retained weight while a bitmask costs one bit per possible
    connection.
    """

    compact = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("._last_forward_param_tensor")
        and key != "_last_forward_param_tensor"
    }
    encoding: dict[str, str] = {}
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
    for key, value in tuple(compact.items()):
        if not key.endswith("connection_indices"):
            continue
        if value.ndim != 2 or value.dtype not in integer_dtypes:
            continue

        weight_key = f"{key[: -len('connection_indices')]}pre_w"
        weight = compact.get(weight_key)
        if weight is None or tuple(weight.shape) != tuple(value.shape):
            continue

        indices = value.to(dtype=torch.long)
        if indices.numel() and bool((indices < 0).any()):
            continue
        sorted_indices, order = indices.sort(dim=1)
        if sorted_indices.shape[1] > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError(f"Topology indices must be unique within each row: {key}")

        n_rows, k = sorted_indices.shape
        n_inputs = int(sorted_indices.max().item()) + 1 if k else 0
        n_bytes = (n_inputs + 7) // 8
        packed = torch.zeros(
            (n_rows, n_bytes),
            dtype=torch.int16,
            device=sorted_indices.device,
        )
        if k:
            byte_indices = torch.div(sorted_indices, 8, rounding_mode="floor")
            bit_values = torch.bitwise_left_shift(
                torch.ones_like(sorted_indices, dtype=torch.int16),
                torch.remainder(sorted_indices, 8),
            ).to(dtype=torch.int16)
            packed.scatter_add_(1, byte_indices, bit_values)

        compact[weight_key] = weight.gather(1, order.to(weight.device))
        del compact[key]
        compact[f"{key}{_BITMASK_SUFFIX}"] = packed.to(dtype=torch.uint8)
        compact[f"{key}{_BITMASK_METADATA_SUFFIX}"] = torch.tensor(
            [n_rows, k, n_inputs],
            dtype=torch.int64,
            device=indices.device,
        )
        encoding[key] = "bitmask"
    return compact, encoding


def decode_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore row-wise indices from :func:`compact_sparse_bitmask_state_dict`."""

    decoded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(_BITMASK_METADATA_SUFFIX):
            continue
        if not key.endswith(_BITMASK_SUFFIX):
            decoded[key] = value
            continue

        original_key = key.removesuffix(_BITMASK_SUFFIX)
        metadata_key = f"{original_key}{_BITMASK_METADATA_SUFFIX}"
        if metadata_key not in state_dict:
            raise RuntimeError(f"Missing bitmask metadata for {original_key}")
        metadata = state_dict[metadata_key].to(dtype=torch.long).reshape(-1)
        if metadata.numel() != 3:
            raise RuntimeError(f"Invalid bitmask metadata for {original_key}")
        n_rows, k, n_inputs = (int(item) for item in metadata.tolist())
        if tuple(value.shape) != (n_rows, (n_inputs + 7) // 8):
            raise RuntimeError(f"Invalid packed bitmask shape for {original_key}")

        bit_offsets = torch.arange(8, device=value.device, dtype=torch.uint8)
        present = torch.bitwise_and(
            torch.bitwise_right_shift(value.unsqueeze(-1), bit_offsets),
            1,
        ).reshape(n_rows, -1)[:, :n_inputs]
        counts = present.sum(dim=1)
        if bool((counts != k).any()):
            raise RuntimeError(
                f"Packed bitmask row counts do not match K for {original_key}"
            )
        coordinates = torch.nonzero(present, as_tuple=False)
        decoded[original_key] = coordinates[:, 1].reshape(n_rows, k).to(torch.long)
    return decoded


def compact_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    topology_encoding: str = "uint",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return a deployment state dictionary with compact sparse topology."""

    normalized = str(topology_encoding).strip().lower()
    if normalized == "uint":
        return compact_sparse_index_state_dict(state_dict)
    if normalized == "bitmask":
        return compact_sparse_bitmask_state_dict(state_dict)
    raise ValueError("topology_encoding must be 'uint' or 'bitmask'")


def topology_encoding_candidates(requested: str) -> tuple[str, ...]:
    """Resolve an explicit or artifact-size-selected topology policy."""

    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return ("uint", "bitmask")
    if normalized in {"uint", "bitmask"}:
        return (normalized,)
    raise ValueError("topology_encoding must be 'auto', 'uint', or 'bitmask'")


def compact_model_state_dict_candidates(
    state_dict: Mapping[str, torch.Tensor],
    *,
    topology_encoding: str = "auto",
) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, str]]]:
    """Build every state candidate required by a topology encoding policy."""

    return {
        encoding: compact_model_state_dict(
            state_dict,
            topology_encoding=encoding,
        )
        for encoding in topology_encoding_candidates(topology_encoding)
    }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_model_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor-only model state from a standard checkpoint payload."""

    candidate = payload
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidate = nested
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint does not contain a non-empty model state mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Model state must map string keys to tensors only")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in state:
            raise ValueError(
                f"Checkpoint contains duplicate normalized key {normalized!r}"
            )
        state[normalized] = value
    return state


def load_model_state_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dictionary with PyTorch's restricted weights-only loader."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    return extract_model_state_dict(payload)


def load_initial_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Strictly load a model warm start and return an auditable record."""

    resolved = Path(path).expanduser().resolve()
    observed_sha256 = sha256_file(resolved)
    if expected_sha256 is not None:
        normalized_expected = str(expected_sha256).strip().lower()
        if observed_sha256 != normalized_expected:
            raise ValueError(
                "Initial checkpoint SHA-256 mismatch: "
                f"expected {normalized_expected}, observed {observed_sha256}"
            )
    state = load_model_state_checkpoint(resolved)
    incompatibilities = model.load_state_dict(state, strict=bool(strict))
    return {
        "schema_version": "initial_model_checkpoint_v1",
        "path": str(resolved),
        "sha256": observed_sha256,
        "strict": bool(strict),
        "state_dict_key_count": len(state),
        "missing_keys": list(incompatibilities.missing_keys),
        "unexpected_keys": list(incompatibilities.unexpected_keys),
    }
