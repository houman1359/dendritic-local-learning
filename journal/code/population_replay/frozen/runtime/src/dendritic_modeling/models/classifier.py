"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.base import BaseNetwork


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(
        self,
        encoder_network: Optional[BaseNetwork],
        core_network: BaseNetwork,
        decoder_network: Optional[BaseNetwork],
        *,
        learned_output_scale: bool = True,
        fixed_output_scale: float = 1.0,
        output_scale_mode: Optional[str] = None,
    ):
        super().__init__(encoder_network, core_network, decoder_network)

        self.learned_output_scale = bool(learned_output_scale)
        self.fixed_output_scale = float(fixed_output_scale)
        if not math.isfinite(self.fixed_output_scale) or self.fixed_output_scale <= 0:
            raise ValueError(
                "fixed_output_scale must be finite and positive, "
                f"got {self.fixed_output_scale!r}"
            )
        if output_scale_mode is None:
            output_scale_mode = "per_class" if self.learned_output_scale else "fixed"
        self.output_scale_mode = str(output_scale_mode).strip().lower()
        if self.output_scale_mode not in {"per_class", "global", "fixed"}:
            raise ValueError(
                "output_scale_mode must be 'per_class', 'global', or 'fixed', "
                f"got {output_scale_mode!r}"
            )
        if isinstance(self.decoder_network, Identity):
            if self.output_scale_mode == "per_class":
                self.log_output_scale = nn.Parameter(
                    torch.zeros((self.core_network.output_dim,)), requires_grad=True
                )
            elif self.output_scale_mode == "global":
                self.log_global_output_scale = nn.Parameter(
                    torch.zeros(()), requires_grad=True
                )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = super().forward(x, **kwargs)
        x = self.fixed_output_scale * x

        if hasattr(self, "log_output_scale"):
            x = self.log_output_scale.exp() * x
        elif hasattr(self, "log_global_output_scale"):
            x = self.log_global_output_scale.exp() * x

        return x

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, **kwargs)
            return logits.argmax(dim=-1)


__all__ = ["Classifier"]
