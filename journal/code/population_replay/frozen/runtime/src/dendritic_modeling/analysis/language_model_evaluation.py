"""Lightweight, reusable causal-language-model evaluation primitives.

This module deliberately depends only on PyTorch and the Python standard
library at import time.  Evaluation-only environments should not need the
training, plotting, dataset, or replacement stacks merely to score an already
loaded causal language model or summarize paired window losses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "paired_difference_stats",
    "paired_student_t_difference_stats",
    "per_window_lm_loss",
]


@torch.no_grad()
def per_window_lm_loss(
    model: Any,
    windows: torch.Tensor,
    device: torch.device | str,
    batch_size: int,
) -> list[float]:
    """Return mean next-token cross-entropy for every input window.

    The returned values retain input-window order, so callers can use each
    window as the pairing unit across teacher, control, and replacement arms.
    Losses are computed in float32 from model logits and are never averaged
    across examples in a batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if windows.ndim != 2:
        raise ValueError("windows must be a two-dimensional token tensor")
    if windows.shape[1] < 2:
        raise ValueError("each window must contain at least two tokens")

    model.eval()
    losses: list[float] = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        logits = model(input_ids=batch).logits.float()
        shifted_logits = logits[:, :-1, :]
        shifted_labels = batch[:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
        )
        losses.extend(token_loss.mean(dim=1).tolist())
    return losses


def paired_difference_stats(
    a: Sequence[float],
    b: Sequence[float],
    *,
    z: float = 1.96,
) -> dict[str, float | int]:
    """Summarize paired ``a - b`` differences with a normal-approximation CI."""

    if len(a) != len(b) or not a:
        raise ValueError("paired stats need two equal-length non-empty lists")
    deltas = [float(x) - float(y) for x, y in zip(a, b)]
    n = len(deltas)
    mean = sum(deltas) / n
    if n > 1:
        variance = sum((delta - mean) ** 2 for delta in deltas) / (n - 1)
        half_width = float(z) * math.sqrt(variance / n)
    else:
        half_width = float("inf")
    return {
        "n_windows": n,
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "fraction_negative": sum(delta < 0 for delta in deltas) / n,
    }


def paired_student_t_difference_stats(
    a: Sequence[float],
    b: Sequence[float],
) -> dict[str, float | int | str]:
    """Summarize paired ``a - b`` differences with a two-sided 95% t CI.

    SciPy is imported only when this function is called, preserving the
    module's lightweight import closure for callers that need only inference
    losses or the normal-approximation interval.
    """

    if len(a) != len(b) or len(a) < 2:
        raise ValueError(
            "paired Student-t stats need equal lists with at least two windows"
        )
    from scipy.stats import t as student_t

    degrees_of_freedom = len(a) - 1
    critical = float(student_t.ppf(0.975, df=degrees_of_freedom))
    if not math.isfinite(critical):
        raise ValueError("Student-t critical value is non-finite")
    result: dict[str, float | int | str] = paired_difference_stats(
        a,
        b,
        z=critical,
    )
    result.update(
        {
            "ci_method": "two-sided_95_percent_Student_t",
            "degrees_of_freedom": degrees_of_freedom,
            "critical_value": critical,
        }
    )
    return result
