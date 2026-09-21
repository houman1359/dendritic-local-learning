from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from dendritic_modeling.networks.architectures.classical.autoencoder.base import (
    BaseAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn.cnn import (
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.base import BaseNetwork


class VariationalEncoder(BaseNetwork):
    def __init__(self, net: MLP, latent_dim: int):
        super().__init__()

        self.net = net

        fc_mean = []
        fc_std = []

        fc_mean.append(nn.Linear(net.output_dim, latent_dim))
        fc_std.append(nn.Linear(net.output_dim, latent_dim))
        fc_std.append(nn.Softplus())

        self.fc_mean = nn.Sequential(*fc_mean)
        self.fc_std = nn.Sequential(*fc_std)

        self.output_dim = latent_dim

    def latent_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        z_mu = self.fc_mean(x)
        z_sigma = self.fc_std(x)
        return z_mu, z_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.latent_distribution(x)
        z = torch.sigmoid(Normal(z_mu, z_sigma).rsample())
        return z


class BaseVariationalAutoencoder(BaseAutoencoder):
    def __init__(self):
        super().__init__()
        self.encoder: VariationalEncoder


class MLPVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        decoder_hidden_dims: list[int],
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        encoder_net = MLP(
            input_dim=input_dim, hidden_dims=encoder_hidden_dims, activation=activation
        )

        self.encoder = VariationalEncoder(encoder_net, latent_dim)

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim,
        )

        self.output_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


class CNNVariationalEncoder(BaseNetwork):
    def __init__(self, net: CNNDownsample):
        super().__init__()

        self.net = net

        # Get the output shape from the base network
        net_c, net_h, net_w = net.output_shape
        # Use the encoder network's actual output shape as the latent shape
        latent_shape = net.output_shape

        # Create Conv2d layers for mean and std
        # These will transform from net output shape to latent shape (same in this case)
        self.conv_mean = nn.Conv2d(
            in_channels=net_c,
            out_channels=net_c,  # Keep same number of channels
            kernel_size=1,  # 1x1 conv
            stride=1,
            padding=0,
        )

        self.conv_std = nn.Sequential(
            nn.Conv2d(
                in_channels=net_c,
                out_channels=net_c,  # Keep same number of channels
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softplus(),  # Ensure positive std
        )

        self.output_shape = latent_shape
        self.output_dim = net_c * net_h * net_w

    def latent_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std tensors for the latent distribution."""
        x = self.net(x)
        z_mu = self.conv_mean(x)
        z_sigma = self.conv_std(x)
        return z_mu, z_sigma

    def forward(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Sample from the latent distribution and return spatial latent tensor."""
        z_mu, z_sigma = self.latent_distribution(x)
        z = torch.sigmoid(Normal(z_mu, z_sigma).rsample())
        if flatten:
            z = torch.flatten(z, start_dim=-3)
        return z


class CNNVariationalAutoencoder(BaseVariationalAutoencoder):
    """
    CNN Variational Autoencoder that maintains spatial structure in the latent space.
    Uses CNNVariationalEncoder instead of flattening to preserve spatial relationships.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        encoder_hidden_dims: list[int],
        latent_shape: tuple[int, int, int],
        decoder_hidden_dims: list[int],
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        encoder_hidden_dims = deepcopy(encoder_hidden_dims)
        encoder_hidden_dims.append(latent_shape[0])

        # Calculate downsampling parameters from input_shape to latent_shape
        kernel_sizes, strides = self._calculate_downsample_params(
            input_shape, latent_shape, len(encoder_hidden_dims)
        )

        encoder_net = CNNDownsample(
            input_shape=input_shape,
            hidden_dims=encoder_hidden_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=activation,
        )
        # VALIDATION: Check if parameter calculation worked
        actual_output_shape = encoder_net.output_shape
        if actual_output_shape != tuple(latent_shape):
            raise ValueError(
                f"Parameter calculation failed! "
                f"Encoder produces shape {actual_output_shape} "
                f"but expected {latent_shape}. "
                f"Try adjusting latent_shape spatial dimensions or encoder_hidden_dims."
            )

        self.encoder = CNNVariationalEncoder(encoder_net)

        self.decoder = CNNUpsample(
            input_shape=encoder_net.output_shape,
            target_shape=input_shape,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

        self.output_shape = input_shape
        self.output_dim = 1
        for size in input_shape:
            self.output_dim *= size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, flatten=False)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def _calculate_downsample_params(
        self,
        input_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
        num_layers: int,
    ) -> tuple[list[int], list[int]]:
        """Calculate convolution parameters to downsample from input_shape to latent_shape.

        CNNDownsample uses padding=0, so we work with that constraint.
        Formula: output_size = (input_size - kernel_size) // stride + 1
        """

        _, input_h, input_w = input_shape
        _, latent_h, latent_w = latent_shape

        if input_h != input_w or latent_h != latent_w:
            raise ValueError("Currently only supports square spatial dimensions")

        if input_h < latent_h:
            raise ValueError("Input spatial size must be >= latent spatial size")

        if input_h == latent_h:
            # No downsampling needed spatially, just use 1x1 convs
            kernels = [1] * num_layers
            strides = [1] * num_layers
            return kernels, strides

        # Calculate the total downsampling factor needed
        input_h / latent_h

        # Distribute downsampling across layers
        # Try to use stride=2 as much as possible, then adjust for remaining factor
        kernels = []
        strides = []

        current_size = input_h
        remaining_layers = num_layers

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer - must reach exact target
                target_size = latent_h
            else:
                # Intermediate layer - try to downsample by factor of 2
                if current_size >= 2 * latent_h:
                    target_size = current_size // 2
                else:
                    # Calculate remaining factor needed
                    remaining_factor = current_size / latent_h
                    per_layer_factor = remaining_factor ** (1.0 / remaining_layers)
                    target_size = max(latent_h, int(current_size / per_layer_factor))

            # Find kernel_size and stride that work
            # Formula: target_size = (current_size - kernel_size) // stride + 1
            # Rearranged: kernel_size = current_size - stride * (target_size - 1)

            if target_size == current_size:
                # No spatial change needed
                kernel = 1
                stride = 1
            else:
                # Try common stride values
                best_kernel = None
                best_stride = None

                for test_stride in [2, 3, 4, 5, 8]:
                    test_kernel = current_size - test_stride * (target_size - 1)
                    if test_kernel >= 1 and test_kernel <= current_size:
                        # Check if this gives exact target
                        computed_output = (
                            current_size - test_kernel
                        ) // test_stride + 1
                        if computed_output == target_size:
                            best_kernel = test_kernel
                            best_stride = test_stride
                            break

                if best_kernel is None:
                    # Fallback: use simple approach
                    stride = max(1, current_size // target_size)
                    kernel = current_size - stride * (target_size - 1)
                    kernel = max(1, min(kernel, current_size))
                else:
                    kernel = best_kernel
                    stride = best_stride

            kernels.append(kernel)
            strides.append(stride)

            # Update current size for next iteration
            current_size = (current_size - kernel) // stride + 1
            remaining_layers -= 1

        return kernels, strides
