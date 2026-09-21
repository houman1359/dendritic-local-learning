"""
Evaluation Utilities
====================

This module contains utility functions for evaluating model performance
using various metrics including accuracy, log-likelihood, and MSE.

Supports two evaluation modes:

* **materialize** - load the full dataset into memory (fast for small datasets
  like CIFAR-10/MNIST).
* **stream** - iterate batch-by-batch through a ``DataLoader`` (required for
  large datasets like ImageNet that would OOM if materialized).

An ``"auto"`` mode picks the right strategy based on estimated dataset size.
"""

import logging
from typing import Optional, Union

import torch

from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.training.utils._evaluation.classification import (
    _compute_accuracy,
    _compute_auc,
    _compute_loglikelihood,
    _compute_pred_label_mi,
    _pred_label_mi_bits,
    accuracy_score,
    auc_score,
    categorical_loglikelihood_score,
    pred_label_mi_score,
)
from dendritic_modeling.training.utils._evaluation.prepare import (
    _flatten_labels_for_grouped_logits,
    _prepare_classification_logits_and_labels,
    _prepare_regression_predictions_and_labels,
    _reduce_temporal,
    _reduce_temporal_labels,
    _sequence_valid_mask,
)
from dendritic_modeling.training.utils._evaluation.regression import (
    _compute_cosine_similarity,
    _compute_mse,
    cosine_similarity_score,
    mse_score,
)
from dendritic_modeling.training.utils._evaluation.runtime import (
    _CachedOutputModel,
    _evaluate_metric_generic as _runtime_evaluate_metric_generic,
    _make_eval_loader,
    _should_materialize,
    _stream_model_outputs,
)
from dendritic_modeling.training.utils._evaluation.score import (
    _call_metric_computation,
    _compute_metric_score,
    _metric_accepts_seq_lengths,
)

logger = logging.getLogger(__name__)


def _evaluate_metric_generic(*args, **kwargs):
    """Compatibility wrapper around the runtime evaluator.

    The old implementation looked up ``_make_eval_loader`` in this module, so
    external tests and notebooks may monkeypatch that name. Forward the current
    facade binding explicitly to preserve that behavior after extraction.
    """
    return _runtime_evaluate_metric_generic(
        *args,
        eval_loader_factory=_make_eval_loader,
        **kwargs,
    )


def evaluate_accuracy(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "accuracy",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate accuracy on multiple datasets.

    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_acc, valid_acc, test_acc)
    """
    return _evaluate_metric_generic(
        metric_func=accuracy_score,
        metric_name="accuracy",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def evaluate_auc(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "auc",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate AUC on multiple datasets.

    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_auc, valid_auc, test_auc)
    """
    return _evaluate_metric_generic(
        metric_func=auc_score,
        metric_name="auc",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def evaluate_categorical_loglikelihood(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "loglikelihood",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate categorical log-likelihood on multiple datasets.

    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_ll, valid_ll, test_ll)
    """
    return _evaluate_metric_generic(
        metric_func=categorical_loglikelihood_score,
        metric_name="loglikelihood",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def evaluate_pred_label_mi(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "pred_label_mi",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate mutual information between predicted labels and
    ground truth labels on multiple datasets.

    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_mi, valid_mi, test_mi)
    """
    return _evaluate_metric_generic(
        metric_func=pred_label_mi_score,
        metric_name="pred_label_mi",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def evaluate_mse(
    regressor: Regressor,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "mse",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate mean squared error on multiple datasets.

    Args:
        regressor: The regressor model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_mse, valid_mse, test_mse)
    """
    return _evaluate_metric_generic(
        metric_func=mse_score,
        metric_name="mse",
        model=regressor,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def evaluate_cosine_similarity(
    regressor: Regressor,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "cosine_similarity",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate cosine similarity on multiple datasets.

    Args:
        regressor: The regressor model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_cosine_similarity, valid_cosine_similarity, test_cosine_similarity)
    """
    return _evaluate_metric_generic(
        metric_func=cosine_similarity_score,
        metric_name="cosine_similarity",
        model=regressor,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


__all__ = [
    "_CachedOutputModel",
    "_call_metric_computation",
    "_compute_accuracy",
    "_compute_auc",
    "_compute_cosine_similarity",
    "_compute_loglikelihood",
    "_compute_metric_score",
    "_compute_mse",
    "_compute_pred_label_mi",
    "_evaluate_metric_generic",
    "_flatten_labels_for_grouped_logits",
    "_make_eval_loader",
    "_metric_accepts_seq_lengths",
    "_pred_label_mi_bits",
    "_prepare_classification_logits_and_labels",
    "_prepare_regression_predictions_and_labels",
    "_reduce_temporal",
    "_reduce_temporal_labels",
    "_sequence_valid_mask",
    "_should_materialize",
    "_stream_model_outputs",
    "accuracy_score",
    "auc_score",
    "categorical_loglikelihood_score",
    "cosine_similarity_score",
    "evaluate_accuracy",
    "evaluate_auc",
    "evaluate_categorical_loglikelihood",
    "evaluate_cosine_similarity",
    "evaluate_mse",
    "evaluate_pred_label_mi",
    "mse_score",
    "pred_label_mi_score",
]
