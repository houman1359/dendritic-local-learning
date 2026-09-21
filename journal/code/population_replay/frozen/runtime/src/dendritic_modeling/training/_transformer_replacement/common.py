"""Shared types and scalar helpers for transformer replacement training."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.replacement_common import (
    _compute_transformer_distillation_loss as _compute_distillation_loss,
    _make_replacement_loader,
    _move_replacement_batch_tensors,
    _resolve_device,
    _resolve_dtype,
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def resolve_configured_replacement_layers(
    config: Any,
    *,
    default: Sequence[int] = (),
) -> list[int]:
    """Resolve one unambiguous ordered replacement-layer selection.

    The model block is authoritative for patching, while the training block is
    retained as a convenient explicit mirror. If both are populated they must
    match exactly, including order, because shared stacks assign slots by that
    order.
    """
    model_layers = [int(layer) for layer in config.model.transformer_replacement.layers]
    training_layers = [
        int(layer) for layer in config.training.transformer_replacement.layers
    ]
    if model_layers and training_layers and model_layers != training_layers:
        raise ValueError(
            "model.transformer_replacement.layers and "
            "training.transformer_replacement.layers must match when both are set"
        )
    layers = training_layers or model_layers or [int(layer) for layer in default]
    if len(layers) != len(set(layers)):
        raise ValueError("Transformer replacement layers must not contain duplicates")
    if any(layer < 0 for layer in layers):
        raise ValueError("Transformer replacement layers must be non-negative")
    return layers


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    normalized = str(name).lower()
    if normalized == "silu":
        return F.silu(x)
    if normalized == "gelu":
        return F.gelu(x)
    if normalized == "relu":
        return F.relu(x)
    if normalized in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"Unsupported synthetic teacher activation {name!r}")


class SyntheticTransformerMLP(nn.Module):
    """Small DeepSeek/Llama-style teacher MLP for smoke distillation runs."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        kind: str = "gated_mlp",
        activation: str = "silu",
        bias: bool = False,
        support_fraction: float = 0.25,
        inhibitory_fraction: float = 0.25,
        output_rank: int | None = None,
        branch_factors: Sequence[int] = (2, 2),
        weight_scale: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.kind = str(kind).lower()
        self.activation = activation
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        if self.kind == "linear":
            self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        elif self.kind == "mlp":
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        elif self.kind == "gated_mlp":
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        elif self.kind.startswith("positive_ei_"):
            if bias:
                raise ValueError("planted positive-E/I teachers do not use bias")
            from dendritic_modeling.analysis.fmi.synthetic_teachers import (
                build_planted_teacher,
            )

            self.planted_teacher = build_planted_teacher(
                kind=self.kind,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                support_fraction=support_fraction,
                inhibitory_fraction=inhibitory_fraction,
                output_rank=output_rank,
                branch_factors=branch_factors,
                weight_scale=weight_scale,
                seed=seed,
            )
        else:
            raise ValueError(
                "synthetic_teacher.kind must be linear, mlp, gated_mlp, or a "
                "registered positive_ei planted mechanism"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "planted_teacher"):
            return self.planted_teacher(x)
        if self.kind == "linear":
            return self.proj(x)
        if self.kind == "mlp":
            return self.down_proj(_activation(self.activation, self.up_proj(x)))
        return self.down_proj(
            _activation(self.activation, self.gate_proj(x)) * self.up_proj(x)
        )

    def ground_truth(self) -> dict[str, Any] | None:
        planted = getattr(self, "planted_teacher", None)
        return None if planted is None else planted.ground_truth()

    def parameter_estimate(self) -> dict[str, int]:
        """Return the dense-control count in the replacement estimator shape."""
        total = sum(param.numel() for param in self.parameters())
        return {
            "stored_total": int(total),
            "active_total": int(total),
            "dense_control": int(total),
        }


@dataclass
class DistillationUnit:
    """One teacher/replacement pair."""

    layer_index: int
    replacement: nn.Module
    teacher_mlp: nn.Module | None = None
    tied_group_index: int | None = None
    tied_group_leader: int | None = None
    tied_group_layers: tuple[int, ...] = ()
    is_tied_alias: bool = False
    collapsed_span_index: int | None = None
    collapsed_span_layers: tuple[int, ...] = ()
    collapsed_span_original_mlps: tuple[nn.Module, ...] = ()
    collapsed_span_post_mlp_norm_attr: str = ""
    collapsed_span_original_post_mlp_norms: tuple[nn.Module, ...] = ()


def _replacement_forward_modules(
    units: Sequence[DistillationUnit],
    train_cfg: Any,
    device: torch.device,
) -> list[nn.Module]:
    """Return modules used for forward passes, optionally DataParallel-wrapped."""
    replacements = [unit.replacement for unit in units]
    has_shared_parameters = any(
        hasattr(module, "tied_group_layers") for module in replacements
    )
    distributed_mode = str(
        getattr(train_cfg, "distributed_mode", "none") or "none"
    ).lower()
    if distributed_mode == "ddp":
        if has_shared_parameters:
            raise NotImplementedError(
                "layerwise DDP does not support parameter-tied replacement groups: "
                "wrapping multiple site views of one parameter set creates "
                "duplicate reducer ownership; use single-process training"
            )
        if bool(getattr(train_cfg, "data_parallel", False)):
            raise ValueError("data_parallel and distributed_mode='ddp' are exclusive")
        from .distributed import wrap_transformer_ddp

        return [
            wrap_transformer_ddp(module, train_cfg, device) for module in replacements
        ]
    if not bool(getattr(train_cfg, "data_parallel", False)):
        return replacements
    if has_shared_parameters:
        raise NotImplementedError(
            "DataParallel does not support parameter-tied replacement groups; use "
            "single-process training"
        )
    if device.type != "cuda":
        logger.warning("data_parallel=True ignored because device is %s", device)
        return replacements

    available = torch.cuda.device_count()
    requested = [int(idx) for idx in getattr(train_cfg, "data_parallel_devices", [])]
    device_ids = requested if requested else list(range(available))
    device_ids = [idx for idx in device_ids if 0 <= idx < available]
    if len(device_ids) < 2:
        logger.warning(
            "data_parallel=True requested, but only %s CUDA device(s) are usable",
            available,
        )
        return replacements

    primary = torch.device(f"cuda:{device_ids[0]}")
    for module in replacements:
        module.to(primary)
    logger.info(
        "Using DataParallel for replacement modules on CUDA devices %s", device_ids
    )
    return [
        nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])
        for module in replacements
    ]


@dataclass
class TransformerReplacementTrainingResult:
    """Summary returned by transformer replacement distillation."""

    train_losses: list[float]
    valid_losses: list[float]
    initial_valid_loss: float
    final_valid_loss: float
    best_valid_loss: float
    best_step: int
    save_dir: str
    layer_indices: list[int]
    layer_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None
    # Evidence labeling (audit 2026-08-16): a zero-step run must never be
    # mistakable for a trained one. ``zero_shot_compile`` = init+calibration
    # only; ``trained`` requires steps_executed > 0.
    execution_mode: str = "unknown"
    steps_requested: int = 0
    steps_executed: int = 0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformerReplacementBenchmarkResult:
    """Post-training comparison for a saved transformer replacement."""

    teacher_source: str
    layer_indices: list[int]
    checkpoint_dir: str
    initial: dict[str, Any]
    trained: dict[str, Any]
    teacher_dense_params: int | None
    # Tied replacements count each physical parameter owner once here.
    dendritic_stored_params: int
    # Active terms are summed over logical FFN-site applications, including
    # every use of a tied core; this is not a unique-storage quantity.
    dendritic_active_params: int
    stored_reduction_factor: float | None
    active_reduction_factor: float | None
    teacher_dense_bytes: int | None = None
    dendritic_stored_bytes: int | None = None
    dendritic_index_bytes: int | None = None
    stored_byte_reduction_factor: float | None = None
    # Model-wide accounting is deliberately separate from the reduction over
    # the replaced dense modules.  A large FFN-local factor must never be
    # presented as compression of the full transformer.
    teacher_whole_model_params: int | None = None
    composed_whole_model_stored_params: int | None = None
    whole_model_stored_reduction_factor: float | None = None
    whole_model_parameter_reduction_fraction: float | None = None
    teacher_whole_model_bytes: int | None = None
    composed_whole_model_stored_bytes: int | None = None
    whole_model_stored_byte_reduction_factor: float | None = None
    whole_model_byte_reduction_fraction: float | None = None
    storage_scope: str = "runtime_state_after_checkpoint_load"
    initial_label: str = "initial"
    trained_label: str = "trained_replacement"
    initialization_diagnostics: dict[str, Any] = field(default_factory=dict)
    evaluation_provenance: dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _make_tensor_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    train_cfg: Any | None = None,
    device: torch.device | str | None = None,
) -> DataLoader:
    """Build the tensor-cache loader used by transformer replacement paths."""
    return _make_replacement_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        train_cfg=train_cfg,
        device=device,
    )


__all__ = [
    "DistillationUnit",
    "SyntheticTransformerMLP",
    "TransformerReplacementBenchmarkResult",
    "TransformerReplacementTrainingResult",
    "_activation",
    "_compute_distillation_loss",
    "_make_tensor_loader",
    "_move_replacement_batch_tensors",
    "_replacement_forward_modules",
    "_resolve_device",
    "_resolve_dtype",
    "_to_plain_mapping",
    "resolve_configured_replacement_layers",
]
