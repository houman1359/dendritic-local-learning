"""Tensor shape normalization helpers for evaluation metrics."""

from __future__ import annotations

import torch

from dendritic_modeling.models import Classifier, Regressor


def _reduce_temporal(output: torch.Tensor) -> torch.Tensor:
    """Reduce [B, T, C] -> [B, C] using last timestep for evaluation metrics.

    No-op when output is already 2D. This ensures evaluation metrics (AUC,
    log-likelihood, MSE, cosine similarity) work with both feedforward and
    recurrent models.
    """
    if output.dim() == 3:
        return output[:, -1, :]
    return output


def _reduce_temporal_labels(labels: torch.Tensor, expected_dims: int) -> torch.Tensor:
    """Reduce temporal labels to match reduced outputs.

    For classification (expected_dims=1): [B, T] -> [B] (last timestep)
    For regression (expected_dims=2): [B, T, C] -> [B, C] (last timestep)
    No-op when labels already have the expected number of dimensions.
    """
    if labels.dim() > expected_dims:
        return labels[:, -1] if labels.dim() == 2 else labels[:, -1, :]
    return labels


def _flatten_labels_for_grouped_logits(
    labels: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Match labels to non-temporal grouped logits and flatten to [N_total]."""
    if labels.dim() == logits.dim() - 1 and labels.shape == logits.shape[:-1]:
        return labels.reshape(-1)

    # Support logits [G, B, C] with labels [B] by repeating labels over G.
    if logits.dim() >= 3 and labels.dim() == 1 and labels.shape[0] == logits.shape[-2]:
        expanded = labels
        for _ in range(logits.dim() - 2):
            expanded = expanded.unsqueeze(0)
        expanded = expanded.expand(*logits.shape[:-2], labels.shape[0])
        return expanded.reshape(-1)

    # Fallback: reduce extra axes in labels and flatten.
    if labels.dim() > 1:
        labels = _reduce_temporal_labels(labels, expected_dims=1)
    return labels.reshape(-1)


def _sequence_valid_mask(
    seq_lengths: torch.Tensor,
    *,
    n_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [B, T] mask selecting valid timesteps for variable-length batches."""
    time_index = torch.arange(n_timesteps, device=device).unsqueeze(0)
    return time_index < seq_lengths.to(device).unsqueeze(1)


def _prepare_classification_logits_and_labels(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seq_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize classifier outputs/labels for metric computation.

    - Recurrent [B, T, C] outputs are flattened over time when labels are
      provided per timestep as [B, T].
    - Otherwise recurrent outputs are reduced to the last timestep.
    - Grouped non-temporal outputs (e.g. [G, B, C]) are flattened to [N_total, C].
    """
    kwargs = {}
    if seq_lengths is not None:
        kwargs["seq_lengths"] = seq_lengths
    logits = classifier(inputs, **kwargs)
    labels = labels.long()

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    is_recurrent_core = bool(getattr(classifier, "_is_recurrent_core", False))
    if logits.dim() == 3 and is_recurrent_core:
        if labels.dim() == 2 and labels.shape[:2] == logits.shape[:2]:
            if seq_lengths is not None:
                valid_mask = _sequence_valid_mask(
                    seq_lengths,
                    n_timesteps=logits.shape[1],
                    device=logits.device,
                )
                logits = logits[valid_mask]
                labels = labels.to(logits.device)[valid_mask]
            else:
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1)
            return logits, labels.long()
        logits = _reduce_temporal(logits)
        labels = _reduce_temporal_labels(labels, expected_dims=1)
        return logits, labels.reshape(-1)

    if logits.dim() >= 3:
        labels_flat = _flatten_labels_for_grouped_logits(labels, logits)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        return logits_flat, labels_flat

    if labels.dim() > 1:
        labels = _reduce_temporal_labels(labels, expected_dims=1)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return logits, labels


def _prepare_regression_predictions_and_labels(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seq_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize regressor outputs/labels for metric computation.

    - Recurrent [B, T, C] outputs are flattened over valid timesteps when labels
      are provided per timestep as [B, T, C].
    - Otherwise recurrent outputs are reduced to the last timestep.
    """
    kwargs = {}
    if seq_lengths is not None:
        kwargs["seq_lengths"] = seq_lengths
    predictions = regressor.predict(inputs, **kwargs)

    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)

    is_recurrent_core = bool(getattr(regressor, "_is_recurrent_core", False))
    if predictions.dim() == 3 and is_recurrent_core:
        if labels.dim() == 3 and labels.shape[:2] == predictions.shape[:2]:
            if seq_lengths is not None:
                valid_mask = _sequence_valid_mask(
                    seq_lengths,
                    n_timesteps=predictions.shape[1],
                    device=predictions.device,
                )
                predictions = predictions[valid_mask]
                labels = labels.to(predictions.device)[valid_mask]
            else:
                predictions = predictions.reshape(-1, predictions.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1])
            return predictions, labels
        predictions = _reduce_temporal(predictions)
        labels = _reduce_temporal_labels(labels, expected_dims=2)
        return predictions, labels

    if labels.dim() > 2:
        labels = _reduce_temporal_labels(labels, expected_dims=2)
    return predictions, labels


__all__ = [
    "_flatten_labels_for_grouped_logits",
    "_prepare_classification_logits_and_labels",
    "_prepare_regression_predictions_and_labels",
    "_reduce_temporal",
    "_reduce_temporal_labels",
    "_sequence_valid_mask",
]
