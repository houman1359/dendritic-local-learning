"""Direct active-parameter-matched point bottleneck control.

The control intentionally has one learned operation only::

    Linear(input_dim, bottleneck_width) -> ReLU -> zero-pad(output_dim)

The bottleneck width is derived from the effective parameter count reported by
the structured dendritic reference.  Padding is parameter free, so storage and
active parameter counts are identical and auditable.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)


def matched_direct_bottleneck_width(
    target_active_parameters: int,
    input_dim: int,
    output_dim: int,
    *,
    max_relative_error: float = 0.01,
) -> int:
    """Return the smallest affine width that meets an active-parameter target.

    A ``Linear(input_dim, width)`` layer has ``(input_dim + 1) * width``
    trainable scalars, including bias.  The returned width is required to fit
    inside ``output_dim`` because this control expands only by zero-padding.
    """

    if isinstance(target_active_parameters, bool) or int(target_active_parameters) < 1:
        raise ValueError("target_active_parameters must be a positive integer")
    if isinstance(input_dim, bool) or int(input_dim) < 1:
        raise ValueError("input_dim must be a positive integer")
    if isinstance(output_dim, bool) or int(output_dim) < 1:
        raise ValueError("output_dim must be a positive integer")
    if not math.isfinite(float(max_relative_error)) or not (
        0.0 <= float(max_relative_error) < 1.0
    ):
        raise ValueError("max_relative_error must be finite and in [0, 1)")

    target = int(target_active_parameters)
    input_width = int(input_dim)
    interface_width = int(output_dim)
    bottleneck_width = math.ceil(target / (input_width + 1))
    if bottleneck_width > interface_width:
        raise ValueError(
            "The matched direct bottleneck does not fit the output interface: "
            f"width={bottleneck_width}, output_dim={interface_width}"
        )

    actual = (input_width + 1) * bottleneck_width
    relative_error = abs(actual - target) / target
    if relative_error > float(max_relative_error):
        raise ValueError(
            "Direct bottleneck cannot match the requested active parameter count "
            f"within tolerance: target={target}, actual={actual}, "
            f"relative_error={relative_error:.6g}, "
            f"max_relative_error={float(max_relative_error):.6g}"
        )
    return bottleneck_width


class DirectActiveMatchedPointBottleneck(nn.Module):
    """One-layer point control matched to a dendritic core's active count.

    The structured E/I arguments define the reference model only.  They do not
    add dendritic operations or buffers to the returned point control.
    ``target_active_parameters`` is an explicit override intended for focused
    tests and diagnostics; normal configs leave it unset and derive the target
    from :meth:`ExcitationInhibitionNetwork.get_effective_params`.
    """

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: list[int],
        inhibitory_layer_sizes,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=None,
        reactivate: bool = False,
        somatic_synapses: bool = True,
        activation: str = "relu",
        target_active_parameters: int | None = None,
        match_relative_tolerance: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not excitatory_layer_sizes:
            raise ValueError("excitatory_layer_sizes cannot be empty")
        if str(activation).lower() != "relu":
            raise ValueError(
                "DirectActiveMatchedPointBottleneck has a fixed ReLU activation"
            )
        input_dim = int(input_dim)
        output_dim = int(excitatory_layer_sizes[-1])
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")

        transfer_params = dict(kwargs.get("transfer_params") or {})
        output_activation = transfer_params.get("output_activation")
        normalized_output_activation = (
            "none" if output_activation is None else str(output_activation).lower()
        )
        if normalized_output_activation not in {"none", "relu"}:
            raise ValueError(
                "DirectActiveMatchedPointBottleneck supports only a ReLU or "
                "identity structured output activation so padded channels remain zero"
            )

        if target_active_parameters is None:
            if ii_synapses_per_branch_per_layer is None:
                ii_synapses_per_branch_per_layer = []
            reference_kwargs = dict(kwargs)
            reference_kwargs["transfer_params"] = {
                "input_mode": 0,
                "independent_pathways": False,
                "excitatory_dim": [],
                "inhibitory_dim": [],
                "output_activation": "none",
                **transfer_params,
            }
            reference = ExcitationInhibitionNetwork(
                input_dim=input_dim,
                excitatory_layer_sizes=excitatory_layer_sizes,
                inhibitory_layer_sizes=inhibitory_layer_sizes,
                excitatory_branch_factors=excitatory_branch_factors,
                inhibitory_branch_factors=inhibitory_branch_factors,
                ee_synapses_per_branch_per_layer=(ee_synapses_per_branch_per_layer),
                ei_synapses_per_branch_per_layer=(ei_synapses_per_branch_per_layer),
                ie_synapses_per_branch_per_layer=(ie_synapses_per_branch_per_layer),
                ii_synapses_per_branch_per_layer=(ii_synapses_per_branch_per_layer),
                reactivate=reactivate,
                somatic_synapses=somatic_synapses,
                **reference_kwargs,
            )
            target_active_parameters = int(reference.get_effective_params())
            del reference

        target = int(target_active_parameters)
        bottleneck_width = matched_direct_bottleneck_width(
            target,
            input_dim,
            output_dim,
            max_relative_error=match_relative_tolerance,
        )
        linear = nn.Linear(input_dim, bottleneck_width)
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        nn.init.zeros_(linear.bias)

        self.linear = linear
        self.activation = nn.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_width = bottleneck_width
        self.padding_width = output_dim - bottleneck_width
        self.target_active_parameters = target
        self.actual_active_parameters = (input_dim + 1) * bottleneck_width
        self.active_parameter_relative_error = (
            abs(self.actual_active_parameters - target) / target
        )
        self.output_activation_name = normalized_output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.activation(self.linear(x))
        return F.pad(active, (0, self.padding_width), mode="constant", value=0.0)

    def get_effective_params(self) -> int:
        """Return the exact number of learned scalars used in the forward pass."""

        return self.actual_active_parameters


__all__ = [
    "DirectActiveMatchedPointBottleneck",
    "matched_direct_bottleneck_width",
]
