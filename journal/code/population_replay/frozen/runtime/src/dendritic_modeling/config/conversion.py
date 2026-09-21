"""Shared config coercion helpers.

These helpers centralize the small but important "config-like object to plain
dict" conversions used by loaders, factories, scripts, and transformer
replacement utilities.  They intentionally accept dicts, OmegaConf objects,
dataclasses, and simple attribute containers so public dict-based usage remains
compatible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from omegaconf import OmegaConf


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert a config-like object into a plain ``dict``.

    Unknown/non-mapping objects return ``{}`` rather than raising; callers use
    this at config boundaries where missing optional blocks are normal.
    """
    if obj is None:
        return {}
    if OmegaConf.is_config(obj):
        converted = OmegaConf.to_container(obj, resolve=True)
        return converted if isinstance(converted, dict) else {}
    if is_dataclass(obj):
        converted = asdict(obj)
        return converted if isinstance(converted, dict) else {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "asdict") and callable(obj.asdict):
        converted = obj.asdict()
        return converted if isinstance(converted, dict) else {}
    try:
        converted = dict(obj)
    except Exception:
        converted = None
    if isinstance(converted, dict):
        return converted
    if hasattr(obj, "__dict__"):
        return {
            key: value for key, value in vars(obj).items() if not key.startswith("_")
        }
    return {}


def deep_merge_dicts(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge two mapping payloads without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def layer_value(
    value: Any,
    layer_idx: int,
    default: Any = 0,
    *,
    allow_empty_list: bool = False,
) -> Any:
    """Read scalar/list per-layer config values with repeat-last semantics."""
    if isinstance(value, list):
        if not value:
            return [] if allow_empty_list else default
        if layer_idx < len(value):
            return value[layer_idx]
        return value[-1]
    if value is None:
        return default
    return value


def normalize_sparsity_type(sparsity_type: Any) -> str:
    """Normalize public sparsity aliases to internal sparse-layer names."""
    normalized = "standard" if sparsity_type is None else str(sparsity_type).lower()
    if normalized == "topk":
        return "standard"
    if normalized in {"indexed-dynamic", "indexed_dynamic_topk"}:
        return "indexed_dynamic"
    if normalized in {"indexed-rewire", "indexed_rewire_topk"}:
        return "indexed_rewire"
    if normalized in {"dense-to-sparse", "dense_to_sparse_topk", "annealed-topk"}:
        return "dense_to_sparse"
    return normalized


def has_enabled_synapse_types(config: Any) -> bool:
    """Return True when a synapse-type block requests typed conductances."""
    synapse_types = to_plain_dict(config)
    if "enabled" in synapse_types:
        return bool(synapse_types["enabled"])
    return bool(
        synapse_types.get("types")
        or synapse_types.get("excitatory")
        or synapse_types.get("inhibitory")
    )


__all__ = [
    "deep_merge_dicts",
    "has_enabled_synapse_types",
    "layer_value",
    "normalize_sparsity_type",
    "to_plain_dict",
]
