"""Lazy public interface for sweep utilities.

The metrics module is intentionally dependency-light.  Avoid importing the
configuration-aware aggregation stack when callers only need labels or plot
styles.
"""

from __future__ import annotations

import importlib
from typing import Any

_ATTRIBUTE_MODULES = {
    "METRIC_CATEGORIES": "metrics",
    "METRIC_MAPPING": "metrics",
    "aggregate_over_seeds": "aggregation",
    "format_metric_name": "formatting",
    "format_value": "formatting",
    "get_metric_label": "metrics",
    "group_by_params": "aggregation",
}


def __getattr__(name: str) -> Any:
    """Import a public sweep utility on first access."""
    module_name = _ATTRIBUTE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_ATTRIBUTE_MODULES))


__all__ = [
    "METRIC_CATEGORIES",
    "METRIC_MAPPING",
    "aggregate_over_seeds",
    "format_metric_name",
    "format_value",
    "get_metric_label",
    "group_by_params",
]
