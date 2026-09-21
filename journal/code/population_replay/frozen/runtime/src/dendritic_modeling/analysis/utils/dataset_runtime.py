"""Dataset sizing and subsetting policies for analysis runtime."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset

from dendritic_modeling.config.analysis import EvaluationRuntimeConfig


def estimate_dataset_size_mb(dataset: Dataset) -> float:
    """Estimate dataset footprint without full materialization."""
    try:
        n_items = len(dataset)
    except (TypeError, AttributeError):
        return float("inf")
    if n_items == 0:
        return 0.0
    try:
        sample = dataset[0]
        item_bytes = 0
        for value in sample if isinstance(sample, (tuple, list)) else [sample]:
            if isinstance(value, torch.Tensor):
                item_bytes += value.nelement() * value.element_size()
            else:
                item_bytes += 8
        return n_items * item_bytes / (1024 * 1024)
    except Exception:
        return float("inf")


def should_materialize_dataset(
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
) -> bool:
    """Return whether *dataset* should be materialized under *runtime*."""
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    mode = getattr(runtime, "mode", "auto")
    threshold = getattr(runtime, "materialize_threshold_mb", 1024)
    if mode == "materialize":
        return True
    if mode == "stream":
        return False
    return estimate_dataset_size_mb(dataset) <= threshold


def effective_sample_cap(
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None = None,
) -> int | None:
    """Combine runtime and analyzer-local sample caps deterministically."""
    caps: list[int] = []
    if runtime is not None:
        runtime_max_samples = getattr(runtime, "max_samples", None)
        runtime_max_batches = getattr(runtime, "max_batches", None)
        runtime_batch_size = getattr(runtime, "batch_size", 256)
        if runtime_max_samples is not None:
            caps.append(int(runtime_max_samples))
        if runtime_max_batches is not None:
            caps.append(int(runtime_batch_size) * int(runtime_max_batches))
    if explicit_max_samples is not None:
        caps.append(int(explicit_max_samples))
    return min(caps) if caps else None


def subset_dataset_for_runtime(
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None = None,
) -> Dataset:
    """Apply deterministic head truncation for runtime/analyzer sample caps."""
    cap = effective_sample_cap(runtime, explicit_max_samples)
    if cap is None:
        return dataset
    try:
        n_items = len(dataset)
    except (TypeError, AttributeError):
        return dataset
    if cap >= n_items:
        return dataset
    return Subset(dataset, range(cap))


__all__ = [
    "effective_sample_cap",
    "estimate_dataset_size_mb",
    "should_materialize_dataset",
    "subset_dataset_for_runtime",
]
