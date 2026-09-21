"""Result aggregation helpers for information analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_helpers import (
    keep_metric_key_given_availability,
    metric_targets_class,
)

_AVERAGED_METRIC_CATEGORIES = (
    "basic_mi",
    "pairwise_mi",
    "conditional_mi",
    "soma_coupling_mi",
    "gaussian_fisher",
)
_LAYER_STATISTIC_CATEGORIES = (
    "basic_mi",
    "pairwise_mi",
    "conditional_mi",
    "soma_coupling_mi",
)

LAYER_INFORMATION_PROXY_SEMANTICS = {
    "scalar_mi_sum_proxy": (
        "Sum of separately estimated scalar branch mutual informations; "
        "this is not joint mutual information."
    ),
    "entropy_clipped_scalar_mi_sum_proxy": (
        "Scalar-MI sum clipped at class entropy; this is not joint mutual information."
    ),
    "independent_evidence_union_proxy": (
        "Heuristic union under an independence assumption; this is not measured union "
        "or joint mutual information."
    ),
    "topk_scalar_mi_sum_proxy": (
        "Sum of the K largest separately estimated scalar branch MIs; this is not "
        "joint mutual information."
    ),
}


def average_mi_results(
    results_list: list[dict[str, dict[str, dict | float]]],
    *,
    method: str,
) -> dict[str, Any]:
    """Average MI result dictionaries across branches or neurons."""
    if not results_list:
        return {"method": method, "n_samples": 0}

    averaged_results: dict[str, Any] = {
        "method": results_list[0].get("method", method),
        "n_samples": results_list[0].get("n_samples", 0),
    }

    for metric_category in _AVERAGED_METRIC_CATEGORIES:
        averaged_category = _average_metric_category(results_list, metric_category)
        if averaged_category:
            averaged_results[metric_category] = averaged_category

    averaged_pid = _average_pid_results(results_list)
    if averaged_pid:
        averaged_results["pid"] = averaged_pid

    return averaged_results


def compute_layer_statistics(
    results_list: list[dict[str, Any]],
    *,
    entropy_C: float | None = None,
    topk: int = 10,
) -> dict[str, float]:
    """Compute per-layer summary stats from per-branch MI results."""
    layer_stats: dict[str, float] = {}
    if not results_list:
        return layer_stats

    for metric_category in _LAYER_STATISTIC_CATEGORIES:
        for key in _metric_keys(results_list, metric_category):
            values = _finite_numeric_values(results_list, metric_category, key)
            if not values:
                continue

            arr = np.asarray(values, dtype=float)
            layer_stats[f"{key}_mean"] = float(np.mean(arr))
            std_val = _metric_std(
                results_list,
                metric_category=metric_category,
                key=key,
                values=arr,
            )
            layer_stats[f"{key}_std"] = std_val
            layer_stats[f"{key}_n"] = float(arr.size)

            if entropy_C is None or entropy_C <= 0 or not metric_targets_class(key):
                continue

            _add_class_targeted_layer_proxies(
                layer_stats,
                key=key,
                values=arr,
                entropy_C=float(entropy_C),
                base_std=std_val,
                topk=topk,
            )

    return layer_stats


def filter_result_metrics_in_place(
    result: dict[str, Any],
    *,
    has_exc_synapses: bool,
    has_inh_synapses: bool,
    has_branch_input: bool,
) -> None:
    """Remove metrics that depend on variables absent from the analyzed layer."""
    for metric_category in _AVERAGED_METRIC_CATEGORIES:
        category = result.get(metric_category)
        if not isinstance(category, dict):
            continue
        result[metric_category] = {
            key: value
            for key, value in category.items()
            if keep_metric_key_given_availability(
                key,
                has_exc_synapses=has_exc_synapses,
                has_inh_synapses=has_inh_synapses,
                has_branch_input=has_branch_input,
            )
        }


def compute_variance_analysis(results: dict[str, Any]) -> dict[str, Any]:
    """Compute variance bookkeeping from information-analysis result summaries."""
    variance_analysis: dict[str, Any] = {}

    if "layer_statistics" in results:
        layer_stats = results["layer_statistics"]
        variance_by_layer: dict[str, dict[str, float]] = {}

        for layer_name, layer_data in layer_stats.items():
            layer_variances: dict[str, float] = {}
            for metric_key, value in layer_data.items():
                if metric_key.endswith("_std"):
                    metric_base = metric_key.replace("_std", "")
                    variance_key = f"{metric_base}_variance"
                    layer_variances[variance_key] = value**2
            variance_by_layer[layer_name] = layer_variances

        variance_analysis["layer_variances"] = variance_by_layer

    variance_analysis["variance_interpretation"] = {
        "layer_variances": "Variance across branches within each layer",
        "computation_level": results.get("computation_level", "unknown"),
        "per_neuron_enabled": results.get("per_neuron_analysis_enabled", False),
        "per_layer_enabled": results.get("per_layer_analysis_enabled", False),
        "per_einet_enabled": results.get("per_einet_analysis_enabled", False),
    }
    return variance_analysis


def _metric_keys(results_list: list[dict[str, Any]], metric_category: str) -> set[str]:
    keys: set[str] = set()
    for result in results_list:
        category = result.get(metric_category)
        if isinstance(category, dict):
            keys.update(category.keys())
    return keys


def _average_metric_category(
    results_list: list[dict[str, Any]],
    metric_category: str,
) -> dict[str, float]:
    averaged_category: dict[str, float] = {}
    for key in _metric_keys(results_list, metric_category):
        values = _finite_numeric_values(results_list, metric_category, key)
        if values:
            averaged_category[key] = float(np.mean(values))
            averaged_category[f"{key}_std"] = float(np.std(values))
    return averaged_category


def _finite_numeric_values(
    results_list: list[dict[str, Any]],
    metric_category: str,
    key: str,
) -> list[float]:
    values: list[float] = []
    for result in results_list:
        category = result.get(metric_category)
        if not isinstance(category, dict) or key not in category:
            continue
        value = category[key]
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            values.append(float(value))
    return values


def _average_pid_results(
    results_list: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    averaged_pid: dict[str, dict[str, float]] = {}
    for pid_key in _pid_keys(results_list):
        metric_keys = _pid_metric_keys(results_list, pid_key)
        if not metric_keys:
            continue

        averaged_pid[pid_key] = {}
        for metric in metric_keys:
            values = _finite_pid_metric_values(results_list, pid_key, metric)
            if values:
                averaged_pid[pid_key][metric] = float(np.mean(values))
                averaged_pid[pid_key][f"{metric}_std"] = float(np.std(values))

    return averaged_pid


def _pid_keys(results_list: list[dict[str, Any]]) -> set[str]:
    pid_keys: set[str] = set()
    for result in results_list:
        pid = result.get("pid")
        if isinstance(pid, dict):
            pid_keys.update(pid.keys())
    return pid_keys


def _pid_metric_keys(results_list: list[dict[str, Any]], pid_key: str) -> set[str]:
    metric_keys: set[str] = set()
    for result in results_list:
        pid_payload = _pid_payload(result, pid_key)
        if isinstance(pid_payload, dict):
            metric_keys.update(pid_payload.keys())
    return metric_keys


def _finite_pid_metric_values(
    results_list: list[dict[str, Any]],
    pid_key: str,
    metric: str,
) -> list[float]:
    values: list[float] = []
    for result in results_list:
        pid_payload = _pid_payload(result, pid_key)
        if not isinstance(pid_payload, dict) or metric not in pid_payload:
            continue
        value = pid_payload[metric]
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            values.append(float(value))
    return values


def _pid_payload(result: dict[str, Any], pid_key: str) -> Any:
    pid = result.get("pid")
    if not isinstance(pid, dict) or pid_key not in pid:
        return None
    return pid[pid_key]


def _metric_std(
    results_list: list[dict[str, Any]],
    *,
    metric_category: str,
    key: str,
    values: np.ndarray,
) -> float:
    std_val = float(np.std(values))
    if not isinstance(key, str) or not key.endswith("_null"):
        return std_val

    null_std_key = f"{key}_std"
    null_stds = _finite_numeric_values(results_list, metric_category, null_std_key)
    if not null_stds:
        return std_val
    null_stds_arr = np.asarray(null_stds, dtype=float)
    return float(np.sqrt(np.mean(null_stds_arr**2)))


def _add_class_targeted_layer_proxies(
    layer_stats: dict[str, float],
    *,
    key: str,
    values: np.ndarray,
    entropy_C: float,
    base_std: float,
    topk: int,
) -> None:
    n = float(values.size)
    total = float(np.sum(values))
    layer_stats[f"{key}_scalar_mi_sum_proxy_mean"] = total
    layer_stats[f"{key}_scalar_mi_sum_proxy_std"] = float(np.sqrt(n) * base_std)
    # Deprecated compatibility aliases retained for existing sweep collectors.
    layer_stats[f"{key}_sum_mean"] = total
    layer_stats[f"{key}_sum_std"] = float(np.sqrt(n) * base_std)

    layer_stats[f"{key}_sum_clipped_mean"] = float(min(total, entropy_C))
    layer_stats[f"{key}_sum_clipped_std"] = (
        layer_stats[f"{key}_sum_std"] if total < entropy_C else 0.0
    )
    layer_stats[f"{key}_entropy_clipped_scalar_mi_sum_proxy_mean"] = layer_stats[
        f"{key}_sum_clipped_mean"
    ]
    layer_stats[f"{key}_entropy_clipped_scalar_mi_sum_proxy_std"] = layer_stats[
        f"{key}_sum_clipped_std"
    ]

    frac = np.clip(values / entropy_C, 0.0, 1.0)
    one_minus = 1.0 - frac
    prod_one_minus = float(np.prod(one_minus))
    union_frac = 1.0 - prod_one_minus
    layer_stats[f"{key}_union_mean"] = float(entropy_C * union_frac)
    layer_stats[f"{key}_independent_evidence_union_proxy_mean"] = layer_stats[
        f"{key}_union_mean"
    ]

    if base_std > 0 and values.size > 0:
        g = np.zeros_like(one_minus)
        mask = (frac > 0.0) & (frac < 1.0) & (one_minus > 0.0)
        if np.any(mask):
            g[mask] = prod_one_minus / one_minus[mask]
            union_std = float(base_std * np.sqrt(np.sum(g**2)))
            layer_stats[f"{key}_union_std"] = float(min(max(union_std, 0.0), entropy_C))
        else:
            layer_stats[f"{key}_union_std"] = 0.0
    else:
        layer_stats[f"{key}_union_std"] = 0.0
    layer_stats[f"{key}_independent_evidence_union_proxy_std"] = layer_stats[
        f"{key}_union_std"
    ]

    layer_stats[f"{key}_max_mean"] = float(np.max(values))
    layer_stats[f"{key}_max_std"] = base_std

    k = int(max(1, topk))
    k = min(k, values.size)
    topk_sum = float(np.sum(np.sort(values)[-k:]))
    layer_stats[f"{key}_top{topk}_sum_mean"] = topk_sum
    layer_stats[f"{key}_top{topk}_sum_std"] = float(np.sqrt(k) * base_std)
    layer_stats[f"{key}_top{topk}_scalar_mi_sum_proxy_mean"] = topk_sum
    layer_stats[f"{key}_top{topk}_scalar_mi_sum_proxy_std"] = float(
        np.sqrt(k) * base_std
    )


__all__ = [
    "LAYER_INFORMATION_PROXY_SEMANTICS",
    "average_mi_results",
    "compute_layer_statistics",
    "compute_variance_analysis",
    "filter_result_metrics_in_place",
]
