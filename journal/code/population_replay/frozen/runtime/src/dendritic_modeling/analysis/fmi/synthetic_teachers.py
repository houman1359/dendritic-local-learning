"""Planted teacher mechanisms for FMI and replacement validation.

These teachers are deliberately small, frozen functions with recorded ground
truth. They test whether the profiler and selector recover known support,
sign, gating, shunting, and hierarchical-interaction structure before FMI is
trusted on a natural network where the correct morphology is unknown.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANTED_TEACHER_KINDS = (
    "positive_ei_additive",
    "positive_ei_shunting",
    "positive_ei_gated",
    "positive_ei_hierarchical",
)


@dataclass(frozen=True)
class PlantedTeacherSpec:
    """Serializable definition of one planted positive-E/I teacher."""

    kind: str
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    support_fraction: float = 0.25
    inhibitory_fraction: float = 0.25
    output_rank: int | None = None
    branch_factors: tuple[int, ...] = (2, 2)
    weight_scale: float = 1.0
    seed: int = 0


def _positive_activation(name: str, values: torch.Tensor) -> torch.Tensor:
    normalized = str(name).lower()
    if normalized in {"silu", "swish"}:
        return F.silu(values)
    if normalized == "gelu":
        return F.gelu(values)
    if normalized == "relu":
        return F.relu(values)
    if normalized == "softplus":
        return F.softplus(values)
    if normalized in {"identity", "linear", "none"}:
        return values
    raise ValueError(f"Unsupported planted-teacher activation {name!r}")


def _signed_split(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((F.relu(values), F.relu(-values)), dim=-1)


def _plant_positive_sparse_linear_(
    layer: nn.Linear,
    *,
    support_fraction: float,
    weight_scale: float,
    generator: torch.Generator,
) -> None:
    fan_in = int(layer.in_features)
    contacts = max(1, min(fan_in, round(float(support_fraction) * fan_in)))
    with torch.no_grad():
        weights = torch.zeros_like(layer.weight)
        for row in range(layer.out_features):
            support = torch.randperm(fan_in, generator=generator)[:contacts]
            values = torch.rand(contacts, generator=generator)
            values = values * (float(weight_scale) / math.sqrt(contacts))
            weights[row, support] = values.to(weights)
        layer.weight.copy_(weights)
        if layer.bias is not None:
            layer.bias.zero_()


def _plant_positive_low_rank_readout_(
    layer: nn.Linear,
    *,
    rank: int,
    weight_scale: float,
    generator: torch.Generator,
) -> None:
    realized_rank = max(1, min(int(rank), layer.in_features, layer.out_features))
    left = torch.rand(layer.out_features, realized_rank, generator=generator)
    right = torch.rand(realized_rank, layer.in_features, generator=generator)
    weights = left @ right
    weights = weights / weights.square().sum(dim=1, keepdim=True).sqrt().clamp_min(
        1e-12
    )
    with torch.no_grad():
        layer.weight.copy_((weights * float(weight_scale)).to(layer.weight))
        if layer.bias is not None:
            layer.bias.zero_()


class PlantedPositiveEITeacher(nn.Module):
    """Frozen positive-weight E/I teacher with a known integration mechanism.

    Signed transformer states are converted to two nonnegative channels. Every
    learned edge after that transform is nonnegative; inhibition is represented
    by a separate pathway rather than negative synaptic weights.
    """

    def __init__(self, spec: PlantedTeacherSpec):
        super().__init__()
        kind = str(spec.kind).lower()
        if kind not in PLANTED_TEACHER_KINDS:
            raise ValueError(
                f"kind must be one of {PLANTED_TEACHER_KINDS}, got {spec.kind!r}"
            )
        if spec.hidden_size < 1 or spec.intermediate_size < 1:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if not 0.0 < float(spec.support_fraction) <= 1.0:
            raise ValueError("support_fraction must be in (0, 1]")
        if not 0.0 < float(spec.inhibitory_fraction) <= 1.0:
            raise ValueError("inhibitory_fraction must be in (0, 1]")
        if any(int(value) < 2 for value in spec.branch_factors):
            raise ValueError("branch_factors must contain integers >= 2")

        self.spec = spec
        self.kind = kind
        self.hidden_size = int(spec.hidden_size)
        self.intermediate_size = int(spec.intermediate_size)
        self.activation = str(spec.activation)
        self.input_transform = "signed_split"
        self.biological_neuron = True
        self.explicit_ei = True
        self.integration_rule = {
            "positive_ei_additive": "raw_additive",
            "positive_ei_shunting": "shunting",
            "positive_ei_gated": "raw_additive",
            "positive_ei_hierarchical": "raw_additive",
        }[kind]
        self.gated = kind == "positive_ei_gated"
        self.branch_factors = (
            tuple(int(value) for value in spec.branch_factors)
            if kind == "positive_ei_hierarchical"
            else ()
        )

        source_width = 2 * self.hidden_size
        inhibitory_width = max(
            1, round(self.intermediate_size * float(spec.inhibitory_fraction))
        )
        generator = torch.Generator(device="cpu").manual_seed(int(spec.seed))

        if kind == "positive_ei_hierarchical":
            leaf_count = math.prod(self.branch_factors)
            branch_width = max(1, math.ceil(self.intermediate_size / leaf_count))
            self.excitatory_branches = nn.ModuleList(
                nn.Linear(source_width, branch_width, bias=False)
                for _ in range(leaf_count)
            )
            self.inhibitory_branches = nn.ModuleList(
                nn.Linear(source_width, branch_width, bias=False)
                for _ in range(leaf_count)
            )
            for layer in (*self.excitatory_branches, *self.inhibitory_branches):
                _plant_positive_sparse_linear_(
                    layer,
                    support_fraction=spec.support_fraction,
                    weight_scale=spec.weight_scale,
                    generator=generator,
                )
            readout_width = branch_width
        else:
            self.excitatory_proj = nn.Linear(
                source_width, self.intermediate_size, bias=False
            )
            self.inhibitory_proj = nn.Linear(source_width, inhibitory_width, bias=False)
            _plant_positive_sparse_linear_(
                self.excitatory_proj,
                support_fraction=spec.support_fraction,
                weight_scale=spec.weight_scale,
                generator=generator,
            )
            _plant_positive_sparse_linear_(
                self.inhibitory_proj,
                support_fraction=spec.support_fraction,
                weight_scale=spec.weight_scale,
                generator=generator,
            )
            if self.gated:
                self.excitatory_gate = nn.Linear(
                    source_width, self.intermediate_size, bias=False
                )
                self.inhibitory_gate = nn.Linear(
                    source_width, inhibitory_width, bias=False
                )
                _plant_positive_sparse_linear_(
                    self.excitatory_gate,
                    support_fraction=spec.support_fraction,
                    weight_scale=spec.weight_scale,
                    generator=generator,
                )
                _plant_positive_sparse_linear_(
                    self.inhibitory_gate,
                    support_fraction=spec.support_fraction,
                    weight_scale=spec.weight_scale,
                    generator=generator,
                )
            readout_width = self.intermediate_size

        inhibitory_readout_width = (
            readout_width if kind == "positive_ei_hierarchical" else inhibitory_width
        )
        self.excitatory_readout = nn.Linear(readout_width, self.hidden_size, bias=False)
        self.inhibitory_readout = nn.Linear(
            inhibitory_readout_width, self.hidden_size, bias=False
        )
        rank = (
            min(readout_width, self.hidden_size)
            if spec.output_rank is None
            else int(spec.output_rank)
        )
        _plant_positive_low_rank_readout_(
            self.excitatory_readout,
            rank=rank,
            weight_scale=spec.weight_scale,
            generator=generator,
        )
        _plant_positive_low_rank_readout_(
            self.inhibitory_readout,
            rank=min(rank, inhibitory_readout_width),
            weight_scale=spec.weight_scale,
            generator=generator,
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _hierarchical_path(
        self,
        inputs: torch.Tensor,
        branches: nn.ModuleList,
    ) -> torch.Tensor:
        values = torch.stack(
            [
                _positive_activation(self.activation, layer(inputs))
                for layer in branches
            ],
            dim=-2,
        )
        remaining = len(branches)
        for factor in reversed(self.branch_factors):
            groups = remaining // factor
            values = values.reshape(
                *values.shape[:-2], groups, factor, values.shape[-1]
            )
            # Geometric composition provides a known non-additive interaction
            # tree while preserving nonnegative activity.
            values = (values + 1e-6).log().mean(dim=-2).exp()
            remaining = groups
        return values.squeeze(-2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        inputs = _signed_split(values)
        if self.kind == "positive_ei_hierarchical":
            excitation = self._hierarchical_path(inputs, self.excitatory_branches)
            inhibition = self._hierarchical_path(inputs, self.inhibitory_branches)
        else:
            excitation = _positive_activation(
                self.activation, self.excitatory_proj(inputs)
            )
            inhibition = _positive_activation(
                self.activation, self.inhibitory_proj(inputs)
            )
            if self.gated:
                excitation = excitation * _positive_activation(
                    self.activation, self.excitatory_gate(inputs)
                )
                inhibition = inhibition * _positive_activation(
                    self.activation, self.inhibitory_gate(inputs)
                )

        excitatory_output = self.excitatory_readout(excitation)
        inhibitory_output = self.inhibitory_readout(inhibition)
        if self.kind == "positive_ei_shunting":
            return excitatory_output / (1.0 + inhibitory_output)
        return excitatory_output - inhibitory_output

    def ground_truth(self) -> dict[str, Any]:
        return {
            "schema": "dendritic_fmi_planted_teacher/v1",
            **asdict(self.spec),
            "biological_neuron": self.biological_neuron,
            "explicit_ei": self.explicit_ei,
            "input_transform": self.input_transform,
            "integration_rule": self.integration_rule,
            "gated": self.gated,
            "branch_factors": list(self.branch_factors),
        }

    def parameter_estimate(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        active = sum(
            int(torch.count_nonzero(parameter)) for parameter in self.parameters()
        )
        return {
            "stored_total": int(total),
            "active_total": int(active),
            "dense_control": int(total),
        }


def build_planted_teacher(**kwargs: Any) -> PlantedPositiveEITeacher:
    """Build a planted teacher from JSON/YAML-compatible keyword arguments."""

    branch_factors = kwargs.get("branch_factors", (2, 2))
    kwargs["branch_factors"] = tuple(int(value) for value in branch_factors)
    return PlantedPositiveEITeacher(PlantedTeacherSpec(**kwargs))


__all__ = [
    "PLANTED_TEACHER_KINDS",
    "PlantedPositiveEITeacher",
    "PlantedTeacherSpec",
    "build_planted_teacher",
]
