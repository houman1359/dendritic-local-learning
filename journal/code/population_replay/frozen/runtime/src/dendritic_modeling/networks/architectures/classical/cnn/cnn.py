"""
CNN-based Input Transform for neural networks.

This module implements CNNInputTransform which uses convolutional neural networks
to transform inputs into excitatory and inhibitory pathways.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.base import BaseNetwork

# change nomenclature to encoder and decoder


class CNNDownsample(BaseNetwork):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        hidden_dims: list[int],  # List of channel sizes for each conv layer
        kernel_sizes: list[int],  # Kernel size for all conv layers
        strides: list[int],  # Stride for all conv layers
        activation: str = "relu",
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Input validation
        input_dim, height, width = input_shape

        # Build CNN backbone for excitatory pathway
        layers = []
        in_dim = input_dim
        current_h, current_w = height, width

        activation_factory = ActivationFactory()

        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            # Add convolutional layer
            conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            if activation == "relu":
                nn.init.kaiming_normal_(conv.weight)
                nn.init.zeros_(conv.bias)
            else:
                nn.init.xavier_normal_(conv.weight)
                nn.init.zeros_(conv.bias)
            layers.append(conv)

            layers.append(
                activation_factory.create(act_type=activation, output_dim=hidden_dim)
            )

            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1

        final_hidden_flat = hidden_dims[-1] * current_h * current_w

        if output_dim is not None:
            layers.append(nn.Flatten(start_dim=-3))
            layers.append(nn.Linear(final_hidden_flat, output_dim))
            self.output_dim = output_dim
        else:
            self.output_shape = (hidden_dims[-1], current_h, current_w)
            self.output_dim = final_hidden_flat

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct shape (add batch dimension if needed)
        if x.dim() == 3:  # (C, H, W)
            x = x[None, ...]  # Add batch dimension: (1, C, H, W)

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        output: torch.Tensor = self.layers(x)
        if hasattr(self, "output_shape"):
            return output.reshape(*batch_shape, *self.output_shape)
        else:
            return output.reshape(*batch_shape, self.output_dim)


class CNNUpsample(BaseNetwork):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (C, H, W) - starting shape
        target_shape: tuple[int, int, int],  # (C, H, W) - target output shape
        hidden_dims: list[int],
        activation: str = "relu",
    ):
        super().__init__()

        input_c, input_h, input_w = input_shape
        target_c, target_h, target_w = target_shape

        if input_h != input_w or target_h != target_w:
            raise ValueError("Currently only supports square spatial dimensions")

        if input_h > target_h:
            raise ValueError("Input spatial size must be <= target spatial size")

        # Auto-calculate transpose conv parameters from input_shape to target_shape
        kernels, strides, paddings = self._calculate_transpose_params(
            input_h, target_h, len(hidden_dims) + 1
        )

        # Build transpose convolution layers
        layers = []
        in_channels = input_c  # Start with input channels
        activation_factory = ActivationFactory()

        for i, out_channels in enumerate([*hidden_dims, target_c]):
            conv_transpose = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i],
            )

            # Initialize weights
            if activation == "relu":
                nn.init.kaiming_normal_(conv_transpose.weight)
                nn.init.zeros_(conv_transpose.bias)
            else:
                nn.init.xavier_normal_(conv_transpose.weight)
                nn.init.zeros_(conv_transpose.bias)

            layers.append(conv_transpose)

            # Add activation (except for final layer)
            if i < len(hidden_dims):
                layers.append(activation_factory.create(activation, out_channels))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.output_shape = (target_c, target_h, target_w)
        self.output_dim = target_c * target_h * target_w

    def _calculate_transpose_params(
        self, input_size: int, target_size: int, num_layers: int
    ) -> tuple[list[int], list[int], list[int]]:
        """Calculate transpose convolution parameters from input_size to target_size.

        This method GUARANTEES exact target_size by doing exhaustive search.
        """

        if input_size == target_size:
            kernels = [1] * num_layers
            strides = [1] * num_layers
            paddings = [0] * num_layers
            return kernels, strides, paddings

        if input_size > target_size:
            raise ValueError(
                f"Input size {input_size} must be <= target size {target_size}"
            )

        # Find exact solution by working forward and doing exhaustive search
        def find_exact_params(
            current_size: int, remaining_target: int, remaining_layers: int
        ) -> list[tuple[int, int, int]]:
            """Recursively find kernel, stride, padding combinations that reach exact target."""
            if remaining_layers == 0:
                return [] if current_size == remaining_target else None

            if remaining_layers == 1:
                # Last layer - must reach exact target
                for stride in range(1, 9):  # Try strides 1-8
                    for kernel in range(1, 21):  # Try kernels 1-20
                        for padding in range(
                            min(kernel, 11)
                        ):  # Try reasonable paddings
                            # Formula: output = (input - 1) * stride - 2 * padding + kernel
                            output = (current_size - 1) * stride - 2 * padding + kernel
                            if output == remaining_target:
                                return [(kernel, stride, padding)]
                return None

            # Try different parameters for this layer
            for stride in range(1, 9):
                for kernel in range(1, 21):
                    for padding in range(min(kernel, 11)):
                        output = (current_size - 1) * stride - 2 * padding + kernel

                        # Must be reasonable step toward target
                        if output > current_size and output <= remaining_target:
                            # Recursively try to reach target from this output
                            rest = find_exact_params(
                                output, remaining_target, remaining_layers - 1
                            )
                            if rest is not None:
                                return [(kernel, stride, padding), *rest]

            return None

        # Find exact parameters
        exact_params = find_exact_params(input_size, target_size, num_layers)

        if exact_params is None:
            raise ValueError(
                f"Cannot find exact upsampling from {input_size} to {target_size} "
                f"in {num_layers} layers. Try different latent_shape spatial dimensions "
                f"or decoder_hidden_dims length."
            )

        kernels, strides, paddings = zip(*exact_params)
        return list(kernels), list(strides), list(paddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[None, ...]
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        output: torch.Tensor = self.layers(x)
        return output.reshape(*batch_shape, *self.output_shape)


__all__ = ["CNNDownsample", "CNNUpsample"]
