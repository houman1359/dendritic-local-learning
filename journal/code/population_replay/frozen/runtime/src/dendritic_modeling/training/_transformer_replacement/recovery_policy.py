"""Opt-in layer-specific updates and parameter retention for joint recovery.

The disabled path returns None and leaves the historical optimizer and loss
unchanged. The first implementation intentionally requires single-process,
replacement-only, fixed-topology recovery without shared parameters or resume:
slot-wise retention is not meaningful when sparse contacts change identity,
and a restart must not silently replace the original proximal reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class JointRecoveryPolicySpec:
    multipliers: dict[int, float]
    proximal_weight: float
    proximal_layers: tuple[int, ...]
    epsilon: float


def validate_joint_recovery_policy_config(
    train_cfg: object,
) -> JointRecoveryPolicySpec | None:
    """Validate opt-in semantics before a model is loaded or mutated."""
    raw = getattr(train_cfg, "joint_layer_lr_multipliers", {})
    if not isinstance(raw, Mapping):
        raise ValueError("joint_layer_lr_multipliers must be a mapping")
    multipliers = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(
                "joint_layer_lr_multipliers keys must be canonical non-negative decimal strings"
            )
        multipliers[int(key)] = _nonnegative_finite(value, f"layer {key} LR multiplier")
    weight = _nonnegative_finite(
        getattr(train_cfg, "joint_proximal_weight", 0.0), "joint_proximal_weight"
    )
    epsilon = _nonnegative_finite(
        getattr(train_cfg, "joint_proximal_epsilon", 1e-8), "joint_proximal_epsilon"
    )
    if epsilon <= 0:
        raise ValueError("joint_proximal_epsilon must be positive")
    raw_layers = getattr(train_cfg, "joint_proximal_layers", [])
    if not isinstance(raw_layers, (list, tuple)) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in raw_layers
    ):
        raise ValueError(
            "joint_proximal_layers must contain non-negative integer layers"
        )
    layers = tuple(raw_layers)
    if len(set(layers)) != len(layers):
        raise ValueError("joint_proximal_layers must be unique")
    if layers and weight == 0:
        raise ValueError(
            "joint_proximal_layers requires positive joint_proximal_weight"
        )
    if not multipliers and weight == 0:
        return None
    if str(getattr(train_cfg, "mode", "")).lower() != "joint_lm_distillation":
        raise ValueError("joint recovery policy requires joint_lm_distillation")
    if str(getattr(train_cfg, "train_target", "replacement_only")).lower() not in {
        "replacement_only",
        "replacements_only",
        "dendritic_only",
    }:
        raise NotImplementedError(
            "joint recovery policy supports replacement_only training"
        )
    distributed = str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
    if distributed not in {"none", "off", "false", ""} or bool(
        getattr(train_cfg, "data_parallel", False)
    ):
        raise NotImplementedError(
            "joint recovery policy supports single-process execution only"
        )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise NotImplementedError(
            "joint recovery policy does not support an active distributed process group"
        )
    if str(getattr(train_cfg, "resume_checkpoint", "") or ""):
        raise NotImplementedError(
            "joint recovery policy resume requires a persisted original proximal reference; use a new warm-start stage"
        )
    return JointRecoveryPolicySpec(multipliers, weight, layers, epsilon)


@dataclass
class JointRecoveryPolicy:
    optimizer_groups: list[dict[str, Any]]
    trainable_parameters: list[nn.Parameter]
    proximal_weight: float
    references: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]
    manifest: dict[str, Any]

    def penalty(self) -> torch.Tensor:
        """Normalized mean per parameter tensor, never a validation term."""
        if not self.references:
            return self.trainable_parameters[0].new_zeros(())
        terms = [
            (parameter.to(dtype=reference.dtype) - reference).square().mean()
            / denominator
            for parameter, reference, denominator in self.references
        ]
        return torch.stack(terms).mean()

    def training_loss(
        self, task_loss: torch.Tensor, components: dict[str, float], *, step: int
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        # Do not add even a floating zero when the penalty is disabled.
        penalty = self.penalty() if self.proximal_weight > 0 else None
        total = (
            task_loss if penalty is None else task_loss + self.proximal_weight * penalty
        )
        penalty_value = 0.0 if penalty is None else float(penalty.detach().item())
        row = {
            "step": int(step),
            "task_objective": float(components["total"]),
            "proximal_penalty": penalty_value,
            "weighted_proximal_penalty": self.proximal_weight * penalty_value,
            "training_total": float(total.detach().item()),
        }
        return total, row

    def write_report(self, save_dir: str, history: Sequence[Mapping[str, Any]]) -> None:
        path = Path(save_dir) / "joint_recovery_policy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {**self.manifest, "training_records": list(history)},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def build_joint_recovery_policy(
    records: Sequence[Any], train_cfg: object, optimizer_cfg: object
) -> JointRecoveryPolicy | None:
    """Snapshot warm-started/calibrated cells and apply validated LR groups.

    Call after the standard replacement-only parameter selection and after
    warm-start loading/calibration. All structural checks precede mutations.
    """
    spec = validate_joint_recovery_policy_config(train_cfg)
    if spec is None:
        return None
    if str(getattr(optimizer_cfg, "name", "")).lower() not in {
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
    }:
        raise NotImplementedError(
            "joint recovery policy requires a standard optimizer with per-group learning rates"
        )
    base_lr = _nonnegative_finite(
        getattr(optimizer_cfg, "lr", None), "base learning rate"
    )
    if base_lr <= 0:
        raise ValueError("joint recovery policy requires a positive base learning rate")
    by_layer = {}
    seen_parameters: set[int] = set()
    for record in records:
        layer = int(record.layer_index)
        if layer in by_layer:
            raise ValueError(
                "joint recovery policy requires one replacement record per layer"
            )
        named = list(record.replacement.named_parameters())
        if not named:
            raise ValueError(f"joint recovery layer {layer} has no parameters")
        for _name, parameter in named:
            if id(parameter) in seen_parameters:
                raise NotImplementedError(
                    "joint recovery policy does not support shared replacement parameters"
                )
            seen_parameters.add(id(parameter))
            if not parameter.requires_grad:
                raise ValueError(
                    "build joint recovery policy after replacement-only trainable selection"
                )
            if parameter.device.type == "meta" or not parameter.is_floating_point():
                raise NotImplementedError(
                    "joint recovery policy requires materialized floating parameters"
                )
        for module in record.replacement.modules():
            if callable(getattr(module, "consume_rewired_mask", None)) or callable(
                getattr(module, "advance_sparsity_schedule", None)
            ):
                raise NotImplementedError(
                    "joint recovery policy requires exported fixed topology; dynamic sparse slots cannot be retained or frozen safely"
                )
        by_layer[layer] = named
    known = set(by_layer)
    unknown = (set(spec.multipliers) | set(spec.proximal_layers)) - known
    if unknown:
        raise ValueError(
            f"joint recovery policy references non-training layers: {sorted(unknown)}"
        )
    multipliers = {layer: spec.multipliers.get(layer, 1.0) for layer in by_layer}
    if any(
        not math.isfinite(base_lr * multiplier) for multiplier in multipliers.values()
    ):
        raise ValueError("joint recovery effective learning rates must be finite")
    train_layers = {
        layer for layer, multiplier in multipliers.items() if multiplier > 0
    }
    if not train_layers:
        raise ValueError("joint recovery policy freezes every replacement")
    proximal_layers = (
        set(spec.proximal_layers) if spec.proximal_layers else train_layers
    )
    if spec.proximal_weight > 0 and not proximal_layers <= train_layers:
        raise ValueError(
            "joint_proximal_layers must be trainable layers with positive LR multipliers"
        )
    devices = {
        parameter.device for named in by_layer.values() for _, parameter in named
    }
    if len(devices) != 1:
        raise NotImplementedError(
            "joint recovery policy requires replacement parameters on one device"
        )
    references = []
    reference_entries = []
    if spec.proximal_weight > 0:
        for layer, named in by_layer.items():
            if layer not in proximal_layers:
                continue
            for name, parameter in named:
                dtype = (
                    torch.float64 if parameter.dtype == torch.float64 else torch.float32
                )
                reference = parameter.detach().to(dtype=dtype).clone()
                if not bool(torch.isfinite(reference).all()):
                    raise ValueError(
                        f"non-finite proximal reference at layer {layer} {name}"
                    )
                denominator = reference.square().mean().clamp_min(spec.epsilon)
                if not bool(torch.isfinite(denominator)):
                    raise ValueError(
                        f"non-finite proximal normalization at layer {layer} {name}"
                    )
                references.append((parameter, reference, denominator))
                reference_entries.append(
                    {
                        "layer": layer,
                        "parameter": name,
                        "shape": list(parameter.shape),
                        "normalization_denominator": float(denominator.item()),
                    }
                )
    groups = []
    trainable = []
    layer_entries = []
    for layer, named in by_layer.items():
        multiplier = multipliers[layer]
        params = [parameter for _, parameter in named]
        if multiplier == 0:
            for parameter in params:
                parameter.requires_grad_(False)
                parameter.grad = None
        else:
            groups.append(
                {"params": params, "lr": base_lr * multiplier, "joint_layer": layer}
            )
            trainable.extend(params)
        layer_entries.append(
            {
                "layer": layer,
                "lr_multiplier": multiplier,
                "effective_lr": base_lr * multiplier,
                "trainable": multiplier > 0,
                "parameter_count": sum(parameter.numel() for parameter in params),
                "proximal": spec.proximal_weight > 0 and layer in proximal_layers,
            }
        )
    manifest = {
        "schema": "dendritic_joint_recovery_policy/v1",
        "base_lr": base_lr,
        "layers": layer_entries,
        "proximal_weight": spec.proximal_weight,
        "proximal_epsilon": spec.epsilon,
        "proximal_reduction": "mean_over_tensors(mean_squared_delta/clamped_reference_mean_square)",
        "proximal_reference": "post_warm_start_or_initial_calibration",
        "proximal_reference_tensors": reference_entries,
        "validation_includes_proximal": False,
        "restart_supported": False,
    }
    return JointRecoveryPolicy(
        groups, trainable, spec.proximal_weight, references, manifest
    )
