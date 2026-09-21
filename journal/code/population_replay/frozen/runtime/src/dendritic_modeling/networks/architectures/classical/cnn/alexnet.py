"""
AlexNet-based Input Transform for neural networks.

This module implements AlexNetInputTransform which uses a pre-trained AlexNet
to transform inputs into excitatory and inhibitory pathways.
"""

from typing import Optional

import torch
from torch import nn
from torchvision import models

from dendritic_modeling.networks.base import BaseNetwork


class AlexNet(BaseNetwork):
    def __init__(self, output_dim: Optional[int] = None, **kwargs):
        super().__init__()

        # Load AlexNet
        alexnet = models.alexnet(pretrained=True)
        layers = list(alexnet.features)
        layers.append(nn.AdaptiveAvgPool2d((6, 6)))
        layers.append(nn.Flatten(start_dim=-3))

        self.output_dim = 9216
        if output_dim is not None:
            layers.append(nn.Linear(self.output_dim, output_dim))
            self.output_dim = output_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (C, H, W)
            x = x[None, ...]  # Add batch dimension: (1, C, H, W)

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        output: torch.Tensor = self.layers(x)
        return output.reshape(*batch_shape, -1)


__all__ = ["AlexNet"]
