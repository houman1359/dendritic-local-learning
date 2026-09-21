"""External index manifests for direct sparse dendritic connectivity.

A manifest is a ``torch.save`` file that pins the exact presynaptic support of
every configured branch level, so index-selection *policies* (teacher-weight
top-K, activation-importance sampling, ...) can be computed offline and
consumed through the existing structured-connectivity path without touching
the default seeded-random samplers.

Configured per pathway::

    model.core.connectivity.structured:
      enabled: true
      pathways:
        ee: {method: index_manifest, manifest_path: /abs/path/manifest.pt}

The hook is default-off three times over: no ``structured`` block, or
``enabled: false``, or an unconfigured pathway all resolve to an empty pathway
config before any manifest code runs, leaving the seeded random samplers and
their RNG streams byte-identical.

Manifest schema (``format_version`` 1)::

    {
      "format_version": 1,
      "meta": {...provenance, ignored by the loader...},
      "entries": {
        "<pathway>/layer<layer_idx>/level<level_idx>": {
          "indices": LongTensor[out_features, synapses_per_branch],
          "in_features": int,
        },
        ...
      },
    }

Rows follow DendriNet construction order (owner-major):
``row = owner * branches_per_owner + local_branch`` with the local branch
index decoded root-to-leaf, exactly as ``spatial_morphology`` emits and as
``EfficientBlockLinear`` aggregates children to parents.

The loader fails closed: a configured pathway whose manifest is missing an
entry, or whose entry does not match the requested geometry, raises instead
of silently falling back to random sampling.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _resolve_index_dtype,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    INDEX_MANIFEST_METHODS,
)

__all__ = [
    "INDEX_MANIFEST_METHODS",
    "clear_manifest_cache",
    "load_configured_manifest_indices",
    "manifest_entry_key",
    "save_index_manifest",
]

_MANIFEST_FORMAT_VERSION = 1

# Small per-process cache so one training run does not re-read the manifest
# file for every branch level.  Keyed by absolute path; invalidated by mtime.
_MANIFEST_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def manifest_entry_key(pathway: str, layer_idx: int, level_idx: int) -> str:
    """Canonical entry key for one branch level of one pathway."""

    return f"{pathway}/layer{int(layer_idx)}/level{int(level_idx)}"


def clear_manifest_cache() -> None:
    """Drop all cached manifests (tests and long-lived processes)."""

    _MANIFEST_CACHE.clear()


def _validate_manifest_payload(payload: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Index manifest {path!r} must be a mapping")
    version = payload.get("format_version")
    if version != _MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Index manifest {path!r} has format_version={version!r}; "
            f"this loader supports {_MANIFEST_FORMAT_VERSION}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError(f"Index manifest {path!r} carries no 'entries' mapping")
    return dict(payload)


def _load_manifest(path: str) -> dict[str, Any]:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Index manifest not found: {resolved}")
    mtime = os.path.getmtime(resolved)
    cached = _MANIFEST_CACHE.get(resolved)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    manifest = _validate_manifest_payload(payload, path=resolved)
    _MANIFEST_CACHE[resolved] = (mtime, manifest)
    return manifest


def _validate_entry_indices(
    indices: torch.Tensor,
    *,
    key: str,
    path: str,
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
) -> torch.Tensor:
    if not isinstance(indices, torch.Tensor):
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} must store a tensor under "
            "'indices'"
        )
    if indices.dim() != 2:
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} must be 2D [rows, K], got "
            f"shape {tuple(indices.shape)}"
        )
    if indices.is_floating_point() or indices.dtype == torch.bool:
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} must hold integer indices, "
            f"got dtype {indices.dtype}"
        )
    expected = (int(out_features), int(synapses_per_branch))
    if tuple(indices.shape) != expected:
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} has shape "
            f"{tuple(indices.shape)}; the configured pathway requires {expected}"
        )
    indices = indices.to(torch.long)
    if bool((indices < 0).any()) or bool((indices >= int(in_features)).any()):
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} has indices outside "
            f"[0, {int(in_features) - 1}]"
        )
    sorted_rows = indices.sort(dim=1).values
    if sorted_rows.shape[1] > 1 and bool(
        (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()
    ):
        raise ValueError(
            f"Manifest entry {key!r} in {path!r} repeats an index within a row"
        )
    return indices


def load_configured_manifest_indices(
    pathway_config: Mapping[str, Any],
    *,
    pathway: str,
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
    layer_idx: int = 0,
    level_idx: int = 0,
    index_dtype: str | torch.dtype = "int64",
) -> torch.Tensor:
    """Load one branch level's direct sparse indices from a manifest file.

    Called by :func:`sample_configured_indices` only when the resolved pathway
    config selects an :data:`INDEX_MANIFEST_METHODS` method, so the default
    (random-sampling) path never reaches this module.
    """

    manifest_path = pathway_config.get("manifest_path")
    if not manifest_path:
        raise ValueError(
            f"Structured pathway {pathway!r} selects method='index_manifest' "
            "but provides no 'manifest_path'"
        )
    manifest = _load_manifest(str(manifest_path))
    entries = manifest["entries"]
    key = manifest_entry_key(pathway, layer_idx, level_idx)
    entry = entries.get(key)
    if entry is None:
        known = ", ".join(sorted(str(k) for k in entries))
        raise KeyError(
            f"Index manifest {manifest_path!r} has no entry {key!r} "
            f"(available: {known})"
        )
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"Manifest entry {key!r} in {manifest_path!r} must be a mapping "
            "with 'indices' and 'in_features'"
        )
    entry_in_features = int(entry.get("in_features", -1))
    if entry_in_features != int(in_features):
        raise ValueError(
            f"Manifest entry {key!r} in {manifest_path!r} was built for "
            f"in_features={entry_in_features}, but the pathway has "
            f"in_features={int(in_features)}"
        )
    indices = _validate_entry_indices(
        entry.get("indices"),
        key=key,
        path=str(manifest_path),
        out_features=out_features,
        in_features=in_features,
        synapses_per_branch=synapses_per_branch,
    )
    return indices.to(
        dtype=_resolve_index_dtype(index_dtype, in_features=int(in_features))
    )


def save_index_manifest(
    entries: Mapping[str, Mapping[str, Any]],
    path: str,
    *,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Write a validated manifest file.

    ``entries`` maps :func:`manifest_entry_key` keys to
    ``{"indices": Tensor[rows, K], "in_features": int}`` mappings.
    """

    if not entries:
        raise ValueError("Refusing to write an empty index manifest")
    payload_entries: dict[str, dict[str, Any]] = {}
    for key, entry in entries.items():
        indices = entry.get("indices")
        if not isinstance(indices, torch.Tensor) or indices.dim() != 2:
            raise ValueError(
                f"Entry {key!r} must store a 2D index tensor under 'indices'"
            )
        in_features = int(entry.get("in_features", -1))
        if in_features <= 0:
            raise ValueError(f"Entry {key!r} needs a positive 'in_features'")
        validated = _validate_entry_indices(
            indices,
            key=str(key),
            path=path,
            out_features=int(indices.shape[0]),
            in_features=in_features,
            synapses_per_branch=int(indices.shape[1]),
        )
        payload_entries[str(key)] = {
            "indices": validated,
            "in_features": in_features,
        }
    payload = {
        "format_version": _MANIFEST_FORMAT_VERSION,
        "meta": dict(meta or {}),
        "entries": payload_entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
