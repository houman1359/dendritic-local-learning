"""Metric dictionary merge helpers for information analysis."""

from collections.abc import Callable
from typing import Any

import numpy as np

MetricKeyRenamer = Callable[[str, dict[str, str]], str]


def merge_renamed_metric_dicts_in_place(
    *,
    target: dict[str, Any],
    source: dict[str, Any],
    token_map: dict[str, str],
    rename_metric_key_tokens: MetricKeyRenamer,
) -> None:
    """Merge nested metric dictionaries after renaming variable tokens."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target:
            renamed_dict = {}
            for subkey, subval in value.items():
                new_subkey = rename_metric_key_tokens(subkey, token_map)
                renamed_dict[new_subkey] = subval
            target[key].update(renamed_dict)


def merge_average_renamed_numeric_metrics_in_place(
    *,
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    token_map: dict[str, str],
    rename_metric_key_tokens: MetricKeyRenamer,
) -> None:
    """Average numeric nested metric values, rename them, and merge into target."""
    metrics_acc: dict[str, dict[str, list[float]]] = {}
    for source in sources:
        for key, value in source.items():
            if isinstance(value, dict):
                if key not in metrics_acc:
                    metrics_acc[key] = {}
                for subkey, subval in value.items():
                    if isinstance(subval, (int, float)):
                        if subkey not in metrics_acc[key]:
                            metrics_acc[key][subkey] = []
                        metrics_acc[key][subkey].append(subval)

    for key, subdict in metrics_acc.items():
        if key in target and isinstance(target[key], dict):
            for subkey, values_list in subdict.items():
                if values_list:
                    avg_value = np.mean(values_list)
                    new_subkey = rename_metric_key_tokens(subkey, token_map)
                    target[key][new_subkey] = avg_value


__all__ = [
    "merge_average_renamed_numeric_metrics_in_place",
    "merge_renamed_metric_dicts_in_place",
]
