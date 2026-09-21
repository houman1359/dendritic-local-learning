"""Observational TRAIN-pass diagnostics for production architecture profiles.

These hooks summarize existing outputs and cotangents. They do not rerun a
forward pass, sample data, change parameters, or enable alternative branch
arithmetic. Zero outputs describe activity; they are not a saturation proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


def tensor_summary(value: torch.Tensor) -> dict:
    """Finite-safe statistics, reduced in FP64 without changing the tensor."""
    value = value.detach()
    flat = value.reshape(-1).double()
    finite = torch.isfinite(flat)
    safe = torch.where(finite, flat, 0.0)
    count = finite.sum()
    denominator = count.clamp_min(1)
    stats = torch.stack(
        (
            count.double(),
            safe.sum() / denominator,
            (safe.square().sum() / denominator).sqrt(),
            safe.abs().max() if flat.numel() else safe.new_zeros(()),
            ((flat == 0) & finite).sum().double() / denominator,
            ((flat <= 0) & finite).sum().double() / denominator,
        )
    ).cpu().tolist()
    result = {
        "shape": list(value.shape),
        "elements": value.numel(),
        "finite_elements": int(stats[0]),
        "mean": stats[1] if stats[0] else None,
        "rms": stats[2] if stats[0] else None,
        "max_abs": stats[3] if stats[0] else None,
        "zero_fraction_of_finite": stats[4] if stats[0] else None,
        "nonpositive_fraction_of_finite": stats[5] if stats[0] else None,
    }
    if value.ndim >= 2 and value.numel():
        result["all_zero_units_fraction"] = float(
            (value.reshape(-1, value.shape[-1]) == 0)
            .all(dim=0).double().mean().item()
        )
        # Across-example variation is distinct from differences between unit
        # means. A layer can have nonzero global RMS while every unit is constant.
        units = value.double().reshape(-1, value.shape[-1])
        unit_finite = torch.isfinite(units)
        counts = unit_finite.sum(dim=0).clamp_min(1)
        unit_safe = torch.where(unit_finite, units, 0.0)
        means = unit_safe.sum(dim=0) / counts
        variance = torch.where(unit_finite, (units - means).square(), 0.0).sum(dim=0) / counts
        valid = unit_finite.all(dim=0)
        if bool(valid.any()):
            std = variance[valid].sqrt()
            result["mean_within_unit_std"] = float(std.mean().item())
            result["constant_units_fraction_of_finite_units"] = float((std == 0).double().mean().item())
        else:
            result["mean_within_unit_std"] = None
            result["constant_units_fraction_of_finite_units"] = None
    return result


def parameter_gradient_summary(model: nn.Module) -> dict:
    """Count each registered parameter once; group by its owning module."""
    groups: dict[str, dict] = {}
    for name, parameter in model.named_parameters():
        owner = name.rpartition(".")[0] or "<root>"
        group = groups.setdefault(owner, {
            "parameters": 0, "trainable_parameters": 0,
            "parameters_missing_gradients": 0, "zero_gradient_elements": 0,
            "nonfinite_parameter_elements": 0, "nonfinite_gradient_elements": 0,
            "parameter_squared_norm": 0.0, "gradient_squared_norm": 0.0,
        })
        group["parameters"] += parameter.numel()
        group["trainable_parameters"] += parameter.numel() if parameter.requires_grad else 0
        p = parameter.detach().double()
        pf = torch.isfinite(p)
        group["nonfinite_parameter_elements"] += int((~pf).sum().item())
        group["parameter_squared_norm"] += float(torch.where(pf, p, 0.0).square().sum().item())
        if parameter.grad is None:
            group["parameters_missing_gradients"] += parameter.numel() if parameter.requires_grad else 0
        else:
            g = parameter.grad.detach().double()
            gf = torch.isfinite(g)
            group["nonfinite_gradient_elements"] += int((~gf).sum().item())
            group["zero_gradient_elements"] += int((g == 0).sum().item())
            group["gradient_squared_norm"] += float(torch.where(gf, g, 0.0).square().sum().item())
    for group in groups.values():
        group["parameter_norm"] = group.pop("parameter_squared_norm") ** 0.5
        group["gradient_norm"] = group.pop("gradient_squared_norm") ** 0.5
        group["gradient_to_parameter_norm"] = (
            group["gradient_norm"] / group["parameter_norm"]
            if group["parameter_norm"] else None
        )
    return groups


class TrainingDiagnostics:
    """Record first/last/every-k TRAIN step, before gradient clipping/update."""

    def __init__(self, model: nn.Module, path: Path, *, every: int, steps: int):
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import DendriticBranchLayer

        if every < 1 or steps < 1:
            raise ValueError("Diagnostic cadence and horizon must be positive")
        self.model, self.every, self.steps = model, every, steps
        self.handle = path.open("x", buffering=1)
        self.handles = []
        self.active = False
        self.record: dict = {}
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer) or name == "readout" or (
                name.startswith("hidden.") and name.count(".") == 1
            ):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name: str, module: nn.Module):
        def observe(_module, _inputs, output):
            if not self.active or not isinstance(output, torch.Tensor):
                return
            row = {"module_type": type(module).__name__, "output": tensor_summary(output)}
            self.record["activations"].setdefault(name, []).append(row)
            if output.requires_grad:
                def cotangent(gradient):
                    row["output_gradient"] = tensor_summary(gradient)
                output.register_hook(cotangent)
        return observe

    def begin_step(self, step: int) -> None:
        self.active = step in {1, self.steps} or step % self.every == 0
        if self.active:
            self.record = {
                "step": step, "split": "train", "gradient_stage": "pre_clipping",
                "parameter_stage": "pre_update", "activations": {},
            }

    def finish_step(self) -> None:
        if self.active:
            self.record["status"] = "backward_completed"
            self.record["parameter_groups"] = parameter_gradient_summary(self.model)
            self.handle.write(json.dumps(self.record, sort_keys=True, allow_nan=False) + "\n")
        self.active = False
        self.record = {}

    def record_failure(self, error: BaseException, step: int) -> None:
        """Preserve available observations and partial gradients after a failure."""
        captured = self.active
        if not captured:
            self.record = {"step": step, "split": "train" if step else "initialization_or_validation", "activations": {}}
        self.record.update({
            "status": "failed", "activation_capture_available": captured,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "parameter_groups": parameter_gradient_summary(self.model),
            "gradient_stage": "failure_state_may_be_partial",
        })
        self.handle.write(json.dumps(self.record, sort_keys=True, allow_nan=False) + "\n")
        self.active = False
        self.record = {}

    def close(self) -> None:
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.handle.close()
