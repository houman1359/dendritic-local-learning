"""Shared projection helpers for information-analysis signal variants."""

from __future__ import annotations

import numpy as np
import torch


def compute_projected_branch_output(
    module,
    excitation: np.ndarray,
    inhibition: np.ndarray,
    branch_input: np.ndarray,
) -> np.ndarray:
    """Apply the layer's dendritic voltage equation and reactivation to projected signals."""
    use_shunting = getattr(module, "use_shunting", True)
    epsilon = getattr(module, "epsilon", 1e-8)

    if use_shunting:
        numerator = excitation + branch_input
        denominator = 1.0 + excitation + inhibition + branch_input
        voltage = numerator / (denominator + epsilon)
    else:
        voltage = excitation + branch_input - inhibition

    if hasattr(module, "reactivation") and module.reactivation is not None:
        with torch.no_grad():
            voltage_tensor = torch.tensor(voltage, dtype=torch.float32)
            if next(module.reactivation.parameters(), None) is not None:
                device = next(module.reactivation.parameters()).device
                voltage_tensor = voltage_tensor.to(device)
            return module.reactivation(voltage_tensor).cpu().numpy()

    return voltage


__all__ = ["compute_projected_branch_output"]
