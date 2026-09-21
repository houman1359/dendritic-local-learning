"""
Baseline MLP models.

This module contains simple MLP implementations for baseline comparisons.
"""

from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.base import BaseNetwork


class MLP(BaseNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        output_dim: Optional[int] = None,
        input_normalization: str = "none",
        input_norm_eps: float = 1e-5,
        output_init: str = "default",
        **kwargs,
    ):
        super().__init__()

        activation_factory = ActivationFactory()
        normalized_input_mode = (
            str(input_normalization).strip().lower().replace("-", "_")
        )
        if normalized_input_mode in {"none", "identity"}:
            self.input_normalization = nn.Identity()
        elif normalized_input_mode in {"layer_norm", "layernorm"}:
            self.input_normalization = nn.LayerNorm(
                input_dim,
                eps=float(input_norm_eps),
                elementwise_affine=False,
            )
        else:
            raise ValueError(
                "input_normalization must be one of "
                "{'none', 'identity', 'layer_norm', 'layernorm'}"
            )

        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)

        layers = []

        # Create hidden layers
        for i in range(1, len(hidden_dims)):
            fc_linear = torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            if activation == "relu":
                nn.init.kaiming_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            else:
                nn.init.xavier_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            layers.append(fc_linear)

            layers.append(
                activation_factory.create(
                    act_type=activation, output_dim=hidden_dims[i]
                )
            )

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            output_layer = nn.Linear(self.output_dim, output_dim)
            normalized_output_init = str(output_init).strip().lower().replace("-", "_")
            if normalized_output_init == "zero":
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
            elif normalized_output_init in {"xavier_zero_bias", "xavier"}:
                nn.init.xavier_normal_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
            elif normalized_output_init != "default":
                raise ValueError(
                    "output_init must be one of "
                    "{'default', 'zero', 'xavier', 'xavier_zero_bias'}"
                )
            layers.append(output_layer)
            self.output_dim = output_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.input_normalization(x))


__all__ = ["MLP"]
