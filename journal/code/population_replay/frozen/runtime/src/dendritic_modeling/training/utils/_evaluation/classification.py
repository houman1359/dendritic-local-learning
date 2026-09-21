"""Classification metric computations for training evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from dendritic_modeling.models import Classifier
from dendritic_modeling.training.utils._evaluation.prepare import (
    _prepare_classification_logits_and_labels,
)
from dendritic_modeling.training.utils._evaluation.score import _compute_metric_score
from dendritic_modeling.utils.math.auc import roc_auc_score


def _compute_accuracy(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    yhat = logits.argmax(dim=-1).long().to(labels.device)
    correct = (yhat == labels.long()).float()
    if correct.numel() == 0:
        return torch.tensor(float("nan"), device=logits.device)
    return correct.mean()


def accuracy_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute accuracy score for a classifier.

    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        Accuracy tensor
    """
    return _compute_metric_score(
        computation_func=_compute_accuracy,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
    )


def _compute_auc(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    _ = reduce_dim  # AUC is always computed over flattened sample axis.
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    labels = labels.long()
    unique_labels = torch.unique(labels)

    # AUC is undefined for a single-class target set.
    if unique_labels.numel() < 2:
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)

    # Allow datasets where only a subset of classes is present in labels.
    if logits.dim() == 2 and logits.shape[-1] != unique_labels.numel():
        class_indices = unique_labels.long()
        if class_indices.numel() <= logits.shape[-1]:
            logits = logits.index_select(-1, class_indices)
            label_map = {
                int(old_label.item()): new_idx
                for new_idx, old_label in enumerate(class_indices)
            }
            labels = labels.clone()
            for old_label, new_idx in label_map.items():
                labels[labels == old_label] = new_idx

    return roc_auc_score(labels, logits, dim=0)


def auc_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute AUC score for a classifier.

    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        AUC tensor
    """
    return _compute_metric_score(
        computation_func=_compute_auc,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
    )


def _compute_loglikelihood(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    log_probs = Categorical(logits=logits).log_prob(labels.long())
    if log_probs.numel() == 0:
        return torch.tensor(float("nan"), device=logits.device)
    if log_probs.dim() == 0:
        return log_probs
    if -log_probs.dim() <= reduce_dim < log_probs.dim():
        return torch.mean(log_probs, dim=reduce_dim)
    return torch.mean(log_probs)


def categorical_loglikelihood_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute categorical log-likelihood score for a classifier.

    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        Log-likelihood tensor
    """
    return _compute_metric_score(
        computation_func=_compute_loglikelihood,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
    )


def _pred_label_mi_bits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Discrete mutual information I(C; Ĉ) in bits for predicted classes.

    Notes
    -----
    This uses the empirical joint distribution of (C, Ĉ) and does not require
    any external dependencies (e.g. scikit-learn).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_true.shape[0] != y_pred.shape[0]:
        return 0.0

    # Map labels to contiguous integer IDs
    _, t = np.unique(y_true, return_inverse=True)
    _, p = np.unique(y_pred, return_inverse=True)

    joint = np.zeros((int(t.max()) + 1, int(p.max()) + 1), dtype=np.float64)
    np.add.at(joint, (t, p), 1.0)
    n = float(joint.sum())
    if n <= 0:
        return 0.0

    joint /= n
    p_t = joint.sum(axis=1, keepdims=True)
    p_p = joint.sum(axis=0, keepdims=True)
    denom = p_t @ p_p

    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz])))
    return mi if np.isfinite(mi) else 0.0


def _compute_pred_label_mi(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    logits, normalized_labels = _prepare_classification_logits_and_labels(
        classifier,
        inputs,
        labels,
        seq_lengths=seq_lengths,
    )
    yhat = logits.argmax(dim=-1)
    y_true = normalized_labels.long().detach().cpu().numpy()
    y_pred = yhat.long().detach().cpu().numpy()
    mi = _pred_label_mi_bits(y_true, y_pred)
    return torch.tensor(mi, dtype=torch.float32, device=labels.device)


def pred_label_mi_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute mutual information between predicted labels and
    ground truth labels for a classifier.

    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation

    Returns:
        Mutual information tensor
    """
    return _compute_metric_score(
        computation_func=_compute_pred_label_mi,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
    )


__all__ = [
    "_compute_accuracy",
    "_compute_auc",
    "_compute_loglikelihood",
    "_compute_pred_label_mi",
    "_pred_label_mi_bits",
    "accuracy_score",
    "auc_score",
    "categorical_loglikelihood_score",
    "pred_label_mi_score",
]
