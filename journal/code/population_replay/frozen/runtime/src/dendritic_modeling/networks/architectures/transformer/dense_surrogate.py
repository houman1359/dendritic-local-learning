"""Conventional dense SwiGLU controls at explicitly chosen widths."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.replacement.cells import (
    RUNTIME_TENSOR_CONTRACT_SCHEMA,
    preserve_runtime_tensor_contract,
)


def dense_swiglu_options(options: dict[str, Any]) -> dict[str, Any]:
    """Require an explicit width; never inherit a sparse compiler plan."""
    options = dict(options)
    options.pop("kind", None)
    options.pop("type", None)
    unknown = set(options) - {"intermediate_size", "init_std"}
    if unknown:
        raise ValueError(f"Unsupported dense SwiGLU options: {sorted(unknown)}")
    if "intermediate_size" not in options:
        raise ValueError("dense_swiglu_surrogate requires explicit intermediate_size")
    return options


class DenseSwiGLUSurrogate(nn.Module):
    """Fresh bias-free down(silu(gate(x)) * up(x)); no internal residual/norm.

    Width is a resource choice rather than a copy of the teacher's expansion.
    Initialization uses independent normal weights, with an explicit standard
    deviation. Fitting this module does not modify the retained decoder norm.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(init_std)) or float(init_std) <= 0:
            raise ValueError("init_std must be finite and positive")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.init_std = float(init_std)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=self.init_std)

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        *,
        intermediate_size: int,
        init_std: float = 0.02,
    ) -> DenseSwiGLUSurrogate:
        # Restrict the first control to the architecture whose exact boundary
        # and gating order are exercised by the compression campaign.
        if type(mlp).__name__ != "Olmo2MLP" or not type(mlp).__module__.startswith(
            "transformers.models.olmo2."
        ):
            raise TypeError("Dense SwiGLU surrogate currently requires an OLMo2 MLP")
        projections = [
            getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(not isinstance(module, nn.Linear) for module in projections):
            raise TypeError("Expected three conventional dense OLMo2 projections")
        gate, up, down = projections
        if any(module.bias is not None for module in projections):
            raise ValueError(
                "Dense SwiGLU surrogate requires bias-free teacher projections"
            )
        if (
            gate.in_features != up.in_features
            or gate.out_features != up.out_features
            or down.in_features != up.out_features
            or down.out_features != up.in_features
        ):
            raise ValueError("Teacher SwiGLU projection dimensions disagree")
        activation = getattr(mlp, "act_fn", None)
        from transformers.activations import SiLUActivation

        if not isinstance(activation, (nn.SiLU, SiLUActivation)):
            raise ValueError(
                "Dense SwiGLU surrogate requires the teacher's exact SiLU activation"
            )
        return cls(
            hidden_size=up.in_features,
            intermediate_size=intermediate_size,
            init_std=init_std,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_floating_point():
            raise TypeError("Dense SwiGLU input must be floating point")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("Dense SwiGLU input hidden dimension differs")
        inputs = hidden_states.to(
            device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype
        )
        output = self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
        return preserve_runtime_tensor_contract(
            output, hidden_states, boundary=type(self).__name__
        )

    def parameter_estimate(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"stored_total": total, "active_total": total, "dense_control": total}
