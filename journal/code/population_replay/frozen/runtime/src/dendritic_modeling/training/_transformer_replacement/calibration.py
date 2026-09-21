"""Data-driven reactivation calibration for transformer replacements."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    calibrate_reactivation_from_data,
)

from .distributed import is_transformer_main_process


def calibrate_transformer_reactivation(
    module: nn.Module,
    batches: Sequence[torch.Tensor],
    train_cfg: Any,
    *,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Calibrate requested data-driven gates, iterating through deep morphologies."""
    if not bool(getattr(train_cfg, "reactivation_calibration_enabled", True)):
        return {}
    if not batches:
        return {}
    mode_raw = str(
        getattr(train_cfg, "reactivation_calibration_mode", "") or ""
    ).strip()
    mode = mode_raw or None
    iterations = max(
        1,
        int(getattr(train_cfg, "reactivation_calibration_iterations", 3)),
    )
    diagnostics: dict[str, dict[str, Any]] = {}
    for _ in range(iterations):
        current = calibrate_reactivation_from_data(
            module,
            batches,
            k=None,
            device=device,
            mode=mode,
        )
        if not current:
            break
        diagnostics = current
    return diagnostics


def calibration_batch_limit(train_cfg: Any) -> int:
    return max(1, int(getattr(train_cfg, "reactivation_calibration_batches", 3)))


def save_reactivation_calibration(
    diagnostics: Mapping[str, Mapping[str, Any]],
    save_dir: str,
) -> None:
    """Save rank-0 calibration evidence without creating empty reports."""
    if not diagnostics or not is_transformer_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "reactivation_calibration.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")


def prefix_calibration_diagnostics(
    diagnostics: Mapping[str, Mapping[str, Any]],
    prefix: str,
) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}.{name}" if name else prefix: dict(values)
        for name, values in diagnostics.items()
    }


def take_tensor_batches(
    tensors: Iterable[torch.Tensor],
    *,
    limit: int,
) -> list[torch.Tensor]:
    batches = []
    for tensor in tensors:
        batches.append(tensor)
        if len(batches) >= int(limit):
            break
    return batches


__all__ = [
    "calibrate_transformer_reactivation",
    "calibration_batch_limit",
    "prefix_calibration_diagnostics",
    "save_reactivation_calibration",
    "take_tensor_batches",
]
