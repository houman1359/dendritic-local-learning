"""Shared metric scoring wrapper for evaluation utilities."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

import torch

from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.models import Classifier, Regressor

logger = logging.getLogger("dendritic_modeling.training.utils.evaluation")


def _metric_accepts_seq_lengths(computation_func: Callable) -> bool:
    """Return whether a metric computation accepts variable sequence lengths."""
    return "seq_lengths" in inspect.signature(computation_func).parameters


def _call_metric_computation(
    computation_func: Callable,
    model: Classifier | Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int,
    *,
    seq_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Call a metric function with the optional seq_lengths argument when supported."""
    if _metric_accepts_seq_lengths(computation_func):
        return computation_func(
            model,
            inputs,
            labels,
            reduce_dim,
            seq_lengths=seq_lengths,
        )
    return computation_func(model, inputs, labels, reduce_dim)


def _compute_metric_score(
    computation_func: Callable,
    model: Classifier | Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generic function to compute any metric score with common boilerplate.

    Args:
        computation_func: Function that computes the specific metric
        model: The model (classifier or regressor)
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths for variable-length sequences

    Returns:
        Computed metric tensor
    """
    # Early skip for inappropriate metrics on Classifier models
    if isinstance(model, Classifier) and computation_func.__name__ in [
        "_compute_mse",
        "_compute_cosine_similarity",
    ]:
        logger.debug("Skipping %s for Classifier model", computation_func.__name__)
        return torch.tensor(float("nan"))

    try:
        with torch.no_grad():
            if move_device:
                with analysis_device_context(model, device) as analysis_device:
                    inputs = inputs.to(analysis_device)
                    labels = labels.to(analysis_device)
                    if seq_lengths is not None:
                        seq_lengths = seq_lengths.to(analysis_device)
                    return _call_metric_computation(
                        computation_func,
                        model,
                        inputs,
                        labels,
                        reduce_dim,
                        seq_lengths=seq_lengths,
                    )
            return _call_metric_computation(
                computation_func,
                model,
                inputs,
                labels,
                reduce_dim,
                seq_lengths=seq_lengths,
            )
    except Exception as e:
        logger.warning(
            "Error computing metric in %s: %s; input dtypes: inputs=%s, "
            "labels=%s; input shapes: inputs=%s, labels=%s",
            computation_func.__name__,
            e,
            inputs.dtype,
            labels.dtype,
            tuple(inputs.shape),
            tuple(labels.shape),
        )
        nan_tensor = torch.tensor(torch.nan)
        nan_tensor = nan_tensor.to(inputs.device)
        return nan_tensor


__all__ = [
    "_call_metric_computation",
    "_compute_metric_score",
    "_metric_accepts_seq_lengths",
]
