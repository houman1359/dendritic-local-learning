"""
Base Input Transform for neural networks.

InputNetTransform maps raw inputs to excitatory and inhibitory pathways
for dendritic network architectures.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import (
    IndexInputTransform,
)

logger = logging.getLogger(__name__)


class InputNetTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input).
    This is a generalized version that can support different network architectures.
    By default, it uses an MLP with configurable hidden dimensions.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        independent_heads: bool = False,
        excitatory_dim: Optional[int] = None,
        inhibitory_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        input_shape: Optional[tuple[int, int, int]] = None,
        excitatory_hidden_dims: Optional[list[int]] = None,
        inhibitory_hidden_dims: Optional[list[int]] = None,
        excitatory_kernel_sizes: Optional[list[int]] = None,
        inhibitory_kernel_sizes: Optional[list[int]] = None,
        excitatory_strides: Optional[list[int]] = None,
        inhibitory_strides: Optional[list[int]] = None,
        network_type: str = "MLP",
        split_seed: Optional[int] = None,
    ):
        super().__init__()
        self.network_type = network_type

        if network_type == "IndexInputTransform":
            self.input_transform = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_dim,
                inhibitory_dim=inhibitory_dim,
                split_seed=split_seed,
            )

        elif network_type == "MLP":
            self.input_transform = MLP(
                input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim
            )
            self.input_transform.set_readout(False)
            self.input_transform.disable_grad()

        elif network_type == "FeatureExtractorMLP":
            from ..mlp import FeatureExtractorMLP

            self.input_transform = FeatureExtractorMLP(
                input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim
            )
            self.input_transform.set_readout(False)
            self.input_transform.disable_grad()

        elif network_type == "AlexNetTransform":
            # AlexNet feature extraction requires implementation of AlexNetInputTransform class
            raise NotImplementedError(
                "AlexNetInputTransform not implemented. To use AlexNet features, "
                "implement the AlexNetInputTransform class in networks.architectures.classical.cnn.alexnet"
            )

        elif network_type == "CNNInputTransform":
            # CNN-based input transformation requires implementation of CNNInputTransform class
            raise NotImplementedError(
                "CNNInputTransform not implemented. To use CNN-based input transformation, "
                "implement the CNNInputTransform class in networks.architectures.classical.cnn.cnn"
            )

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, x):
        excitatory_input, inhibitory_input = self.input_transform(x)
        return excitatory_input, inhibitory_input

    def load_pretrained_weights(self, source, weights_path=None, pytorch_model=None):
        """
        Load pre-trained weights for the input network.

        Args:
            source (str): Source of weights - 'file', 'pytorch', or 'none'
            weights_path (str, optional): Path to weights file if source is 'file'
            pytorch_model (str, optional): Name of PyTorch model if source is 'pytorch'
        """
        if source == "none":
            logger.info("No pre-trained weights to load.")
            return

        elif source == "file":
            if not weights_path:
                raise ValueError("weights_path must be provided when source is 'file'")
            logger.info(f"Loading weights from file: {weights_path}")
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)

        elif source == "pytorch":
            if not pytorch_model:
                raise ValueError(
                    "pytorch_model must be provided when source is 'pytorch'"
                )
            logger.info(f"Loading pre-trained PyTorch model: {pytorch_model}")

            import torchvision.models as models

            if pytorch_model == "alexnet":
                pretrained_model = models.alexnet(pretrained=True)
                # Copy features to shared backbone
                if self.network_type == "AlexNetTransform":
                    # For AlexNet, we can directly copy the features
                    for i, layer in enumerate(list(pretrained_model.features)):
                        if i < len(list(self.shared)):
                            self.shared[i].load_state_dict(layer.state_dict())
            else:
                raise ValueError(f"Unsupported PyTorch model: {pytorch_model}")
        else:
            raise ValueError(f"Unknown source: {source}")


__all__ = ["InputNetTransform"]
