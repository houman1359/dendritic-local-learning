"""Regression metric computations for training evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from dendritic_modeling.models import Regressor
from dendritic_modeling.training.utils._evaluation.prepare import (
    _prepare_regression_predictions_and_labels,
)
from dendritic_modeling.training.utils._evaluation.score import _compute_metric_score


def _compute_mse(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions, labels = _prepare_regression_predictions_and_labels(
        regressor, inputs, labels, seq_lengths=seq_lengths
    )
    return torch.mean((predictions - labels) ** 2, dim=reduce_dim)


def mse_score(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int | None = None,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute mean squared error score for a regressor.

    Args:
        regressor: The regressor model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean (not used in current implementation)
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        MSE tensor
    """
    return _compute_metric_score(
        computation_func=_compute_mse,
        model=regressor,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
    )


def _compute_cosine_similarity(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions, labels = _prepare_regression_predictions_and_labels(
        regressor, inputs, labels, seq_lengths=seq_lengths
    )
    return torch.mean(
        functional.cosine_similarity(predictions, labels, dim=-1),
        dim=reduce_dim,
    )


def cosine_similarity_score(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int | None = None,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute cosine similarity score for a regressor.

    Args:
        regressor: The regressor model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean (not used in current implementation)
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        Cosine similarity tensor
    """
    return _compute_metric_score(
        computation_func=_compute_cosine_similarity,
        model=regressor,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
    )


__all__ = [
    "_compute_cosine_similarity",
    "_compute_mse",
    "cosine_similarity_score",
    "mse_score",
]
