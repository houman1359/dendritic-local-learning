"""Shared scalar helpers for replacement-training modules."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.training.dataloader_utils import (
    DataLoaderTuning,
    resolve_dataloader_tuning_from_config,
    seeded_dataloader_kwargs,
)

_DTYPE_ALIASES = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}
_MSE_LOSS_NAMES = {"mse", "l2"}
_RELATIVE_MSE_LOSS_NAMES = {"relative_mse", "normalized_mse", "nmse"}
_SMOOTH_L1_LOSS_NAMES = {"smooth_l1", "huber"}


@dataclass
class ReplacementTrainingHistory:
    """Bookkeeping shared by replacement distillation loops."""

    initial_valid_loss: float
    best_valid_loss: float
    best_step: int
    train_losses: list[float]
    valid_losses: list[float]

    @classmethod
    def start(cls, initial_valid_loss: float) -> ReplacementTrainingHistory:
        return cls(
            initial_valid_loss=initial_valid_loss,
            best_valid_loss=initial_valid_loss,
            best_step=0,
            train_losses=[],
            valid_losses=[initial_valid_loss],
        )

    def record_train_loss(self, train_loss: float) -> None:
        self.train_losses.append(train_loss)

    def record_valid_loss(self, step: int, valid_loss: float) -> None:
        self.valid_losses.append(valid_loss)
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_step = step

    @property
    def final_valid_loss(self) -> float:
        return self.valid_losses[-1]

    def as_result_kwargs(self) -> dict[str, Any]:
        """Return common replacement-training result fields."""
        return {
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "initial_valid_loss": self.initial_valid_loss,
            "final_valid_loss": self.final_valid_loss,
            "best_valid_loss": self.best_valid_loss,
            "best_step": self.best_step,
        }


def _should_run_periodic_step(step: int, every: int, max_steps: int) -> bool:
    """Return whether a step should run a periodic action."""
    return step % int(every) == 0 or step == int(max_steps)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    normalized = str(dtype).lower()
    if normalized not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}"
        )
    return _DTYPE_ALIASES[normalized]


def _make_cuda_grad_scaler(use_amp: bool, device: torch.device) -> GradScaler:
    """Create the CUDA GradScaler used by replacement-training loops."""
    return GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))


def _capture_requires_grad_states(
    parameters: Iterable[torch.nn.Parameter],
) -> list[tuple[torch.nn.Parameter, bool]]:
    """Capture parameter trainability before a temporary selection pass."""
    return [(param, bool(param.requires_grad)) for param in parameters]


def _restore_requires_grad_states(
    states: Iterable[tuple[torch.nn.Parameter, bool]],
) -> None:
    """Restore parameter trainability captured by ``_capture_requires_grad_states``."""
    for param, requires_grad in states:
        param.requires_grad_(requires_grad)


def _clip_replacement_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    *,
    max_grad_norm: Any,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Apply optional replacement-training gradient clipping."""
    if max_grad_norm is None:
        return
    max_norm = float(max_grad_norm)
    if max_norm <= 0:
        return
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _base_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    error_context: str,
) -> torch.Tensor:
    """Compute the base distillation loss before optional cosine regularization."""
    normalized = str(loss_name).lower()
    if normalized in _MSE_LOSS_NAMES:
        return F.mse_loss(prediction, target)
    if normalized in _RELATIVE_MSE_LOSS_NAMES:
        mse = F.mse_loss(prediction, target)
        target_power = target.detach().square().mean()
        epsilon = torch.finfo(mse.dtype).eps
        return mse / target_power.clamp_min(epsilon)
    if normalized in _SMOOTH_L1_LOSS_NAMES:
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"{error_context} loss must be mse, relative_mse, or smooth_l1")


def _flatten_for_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_flatten: Literal["batch", "last_dim"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten prediction/target tensors according to a replacement domain."""
    if cosine_flatten == "batch":
        return prediction.reshape(prediction.shape[0], -1), target.reshape(
            target.shape[0], -1
        )
    if cosine_flatten == "last_dim":
        return prediction.reshape(-1, prediction.shape[-1]), target.reshape(
            -1, target.shape[-1]
        )
    raise ValueError("cosine_flatten must be 'batch' or 'last_dim'")


def _compute_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
    cosine_flatten: Literal["batch", "last_dim"] = "last_dim",
    error_context: str = "replacement",
) -> torch.Tensor:
    """Compute the shared MSE/SmoothL1 plus optional cosine distillation loss."""
    loss = _base_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        error_context=error_context,
    )

    if cosine_weight <= 0:
        return loss

    pred_flat, tgt_flat = _flatten_for_cosine(
        prediction,
        target,
        cosine_flatten=cosine_flatten,
    )
    cosine_loss = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    return loss + float(cosine_weight) * cosine_loss


def _compute_transformer_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """Compute transformer replacement loss with token/hidden-dim flattening."""
    return _compute_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        cosine_weight=cosine_weight,
        cosine_flatten="last_dim",
        error_context="transformer replacement",
    )


def _compute_vision_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """Compute vision replacement loss with per-example flattening."""
    return _compute_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        cosine_weight=cosine_weight,
        cosine_flatten="batch",
        error_context="vision replacement",
    )


def _infinite_loader(loader: Any):
    """Yield batches from a loader forever without caching an epoch."""
    while True:
        yield from loader


def _move_replacement_batch_tensors(
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a replacement input/target pair using the shared loop convention."""
    return (
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
    )


def _load_tensor_payload(
    path: str,
    *,
    empty_path_message: str,
) -> Any:
    """Load a tensor-cache payload from disk onto CPU."""
    if not path:
        raise ValueError(empty_path_message)
    return torch.load(path, map_location="cpu")


def _load_required_tensor(
    path: str,
    *,
    empty_path_message: str,
    type_error_message: str,
) -> torch.Tensor:
    """Load a tensor-cache payload that must be a tensor."""
    payload = _load_tensor_payload(path, empty_path_message=empty_path_message)
    if not torch.is_tensor(payload):
        raise TypeError(type_error_message)
    return payload


def _resolve_replacement_loader_tuning(
    dataset: Dataset,
    train_cfg: Any,
    *,
    device: torch.device | str | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning for tensor-cache replacement datasets."""
    return resolve_dataloader_tuning_from_config(dataset, train_cfg, device=device)


def _make_replacement_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    train_cfg: Any | None = None,
    device: torch.device | str | None = None,
    tuning: DataLoaderTuning | None = None,
) -> DataLoader:
    """Build the shared non-dropping DataLoader for replacement training."""
    if tuning is None and train_cfg is not None:
        tuning = _resolve_replacement_loader_tuning(
            dataset,
            train_cfg,
            device=device,
        )
    loader_kwargs = tuning.as_kwargs() if tuning is not None else {}
    configured_seed = (
        getattr(train_cfg, "loader_seed", None) if train_cfg is not None else None
    )
    if configured_seed is None:
        configured_seed = getattr(train_cfg, "seed", 0) if train_cfg is not None else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **seeded_dataloader_kwargs(int(configured_seed or 0)),
        **loader_kwargs,
    )


def _write_metrics_json(
    result: Any,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    filename: str = "metrics.json",
) -> None:
    """Write a replacement-training result object to a metrics JSON file."""
    save_dir = output_dir if output_dir is not None else result.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(result.asdict(), handle, indent=2)


__all__ = [
    "ReplacementTrainingHistory",
    "_capture_requires_grad_states",
    "_clip_replacement_grad_norm",
    "_compute_distillation_loss",
    "_compute_transformer_distillation_loss",
    "_compute_vision_distillation_loss",
    "_infinite_loader",
    "_load_required_tensor",
    "_load_tensor_payload",
    "_make_cuda_grad_scaler",
    "_make_replacement_loader",
    "_move_replacement_batch_tensors",
    "_resolve_device",
    "_resolve_dtype",
    "_resolve_replacement_loader_tuning",
    "_restore_requires_grad_states",
    "_should_run_periodic_step",
    "_to_plain_mapping",
    "_write_metrics_json",
]
