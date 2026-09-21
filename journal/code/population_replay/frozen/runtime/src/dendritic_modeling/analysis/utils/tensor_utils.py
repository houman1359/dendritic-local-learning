"""
Tensor utilities for analysis.

This module provides utilities for handling tensor operations and shape
compatibility issues during analysis, particularly with efficient_blocklinear.
"""

import torch
import torch.nn as nn

from dendritic_modeling.utils.hooks import iter_modules_of_type


def ensure_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is 2D for layer compatibility.

    Args:
        x: Input tensor of any dimension

    Returns:
        2D tensor with shape [batch_size, features]
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    elif x.dim() == 2:
        return x
    elif x.dim() == 3:
        # Common case in analysis: [batch, seq_len, features]
        # Flatten to [batch * seq_len, features]
        batch_size, seq_len, features = x.shape
        return x.reshape(batch_size * seq_len, features)
    else:
        # For higher dimensions, flatten all but the last dimension
        return x.reshape(-1, x.shape[-1])


def wrap_model_for_analysis(model: nn.Module) -> nn.Module:
    """
    Wrap a model to ensure compatibility with analysis tools.

    This wrapper ensures that inputs to EfficientBlockLinear layers
    are properly shaped during analysis forward passes.

    Args:
        model: The model to wrap

    Returns:
        Wrapped model with shape compatibility
    """

    class AnalysisWrapper(nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model
            self._wrap_efficient_blocklinear_layers()

        def _wrap_efficient_blocklinear_layers(self):
            """Add input shape correction hooks to EfficientBlockLinear layers."""
            from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
                EfficientBlockLinear,
            )

            for module in iter_modules_of_type(
                self.wrapped_model, EfficientBlockLinear
            ):
                original_forward = module.forward

                def corrected_forward(x, original_forward=original_forward):
                    # Ensure input is properly shaped
                    if x.dim() > 2:
                        B = x.shape[0]
                        x = x.reshape(B, -1)
                    return original_forward(x)

                # Replace forward method
                module.forward = corrected_forward

        def forward(self, x):
            return self.wrapped_model(x)

        def __getattr__(self, name):
            # Delegate attribute access to wrapped model
            if name == "wrapped_model" or name == "_modules":
                return super().__getattr__(name)
            return getattr(self.wrapped_model, name)

    return AnalysisWrapper(model)


def fix_tensor_shapes_for_blocklinear(
    tensors: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    """
    Fix tensor shapes for compatibility with BlockLinear layers.

    This function checks if the model uses EfficientBlockLinear and adjusts
    tensor shapes accordingly for analysis.

    Args:
        tensors: Dictionary of tensors from layer activations
        model: The model being analyzed

    Returns:
        Dictionary with corrected tensor shapes
    """
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
        EfficientBlockLinear,
    )

    # Check if model uses EfficientBlockLinear
    uses_efficient = any(iter_modules_of_type(model, EfficientBlockLinear))

    if not uses_efficient:
        return tensors

    # Fix shapes for each tensor
    fixed_tensors = {}
    for key, tensor in tensors.items():
        if tensor.dim() > 2:
            # Flatten extra dimensions for compatibility
            batch_size = tensor.shape[0]
            fixed_tensors[key] = tensor.reshape(batch_size, -1)
        else:
            fixed_tensors[key] = tensor

    return fixed_tensors


__all__ = [
    "ensure_2d_tensor",
    "fix_tensor_shapes_for_blocklinear",
    "wrap_model_for_analysis",
]
