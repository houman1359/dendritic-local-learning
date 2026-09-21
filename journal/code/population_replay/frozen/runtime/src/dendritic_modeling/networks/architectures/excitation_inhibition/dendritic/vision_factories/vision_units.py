"""Factory functions that build reusable dendritic visual units used by higher-level
architectures. Each function returns a configured ``DendriteConv2d`` whose
E- and I-weights implement a classical visual receptive field (Gabor, centre
surround, motion, end-stopped)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

__all__ = [
    "create_center_surround_unit",
    "create_end_stopped_unit",
    "create_gabor_unit",
    "create_motion_unit",
    "rf_gabor",
    "rf_off_center",
    "rf_on_center",
]


def _gabor_kernel(
    size: int,
    wavelength: float,
    orientation: float,
    phase: float,
    sigma: float,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    g = np.exp(-(Xr**2 + (Yr / aspect_ratio) ** 2) / (2 * sigma**2))
    return g * np.cos(2 * np.pi * Xr / wavelength + phase)


def _push_pull_layer(E: np.ndarray, I_var: np.ndarray) -> DendriteConv2d:
    layer = DendriteConv2d(1, 1, E.shape[0], padding=E.shape[0] // 2, bias=False)
    with torch.no_grad():
        layer.weight_E_raw.data[0, 0, 0] = torch.log(
            torch.tensor(E, dtype=torch.float32) + 1e-6
        )
        layer.weight_I_raw.data[0, 0, 0] = torch.log(
            torch.tensor(I_var, dtype=torch.float32) + 1e-6
        )
    return layer


# Target RF templates for fitting
def rf_on_center(size: int = 15, sigma: float = 2.0) -> torch.Tensor:
    """Generate ON-center receptive field template."""
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    center_x, center_y = size // 2, size // 2

    # Gaussian center
    center = torch.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma**2))

    # Surround (wider Gaussian)
    surround = torch.exp(
        -((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * (2 * sigma) ** 2)
    )

    # ON-center: positive center, negative surround
    rf = center - 0.5 * surround
    return rf / rf.abs().max()  # Normalize


def rf_off_center(size: int = 15, sigma: float = 2.0) -> torch.Tensor:
    """Generate OFF-center receptive field template."""
    return -rf_on_center(size, sigma)


def rf_gabor(
    size: int = 15, sigma: float = 2.0, freq: float = 0.3, theta: float = 0.0
) -> torch.Tensor:
    """Generate Gabor receptive field template."""
    x = torch.arange(size, dtype=torch.float32) - size // 2
    y = torch.arange(size, dtype=torch.float32) - size // 2
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Convert theta to tensor if needed
    theta = torch.tensor(theta, dtype=torch.float32)

    # Rotate coordinates
    X_rot = X * torch.cos(theta) + Y * torch.sin(theta)
    Y_rot = -X * torch.sin(theta) + Y * torch.cos(theta)

    # Gabor function
    gaussian = torch.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
    sinusoid = torch.cos(2 * torch.pi * freq * X_rot)

    rf = gaussian * sinusoid
    return rf / rf.abs().max()  # Normalize


# -----------------------------------------------------------------------------
# Public factory functions
# -----------------------------------------------------------------------------


def create_gabor_unit(
    orientation: float,
    *,
    kernel_size: int = 11,
    wavelength: float = 8.0,
    sigma: float = 3.0,
) -> DendriteConv2d:
    gabor = _gabor_kernel(kernel_size, wavelength, orientation, 0.0, sigma)
    E = np.maximum(gabor, 0.0)
    I_var = np.maximum(-gabor, 0.0) + 0.1
    return _push_pull_layer(E, I_var)


def create_center_surround_unit(
    kind: str = "on",
    *,
    kernel_size: int = 15,
    center_size: int = 5,
    surround_size: int = 11,
) -> DendriteConv2d:
    x = np.arange(kernel_size) - kernel_size // 2
    y = np.arange(kernel_size) - kernel_size // 2
    X, Y = np.meshgrid(x, y)
    center = np.exp(-(X**2 + Y**2) / (2 * (center_size / 3) ** 2))
    center /= center.sum()
    surround = np.exp(-(X**2 + Y**2) / (2 * (surround_size / 3) ** 2))
    surround /= surround.sum()
    if kind.lower().startswith("on"):
        E, I_var = center * 10.0, surround * 8.0 + 0.1
    else:
        E, I_var = surround * 10.0, center * 8.0 + 0.1
    return _push_pull_layer(E, I_var)


def create_motion_unit(
    direction: str = "rightward", *, kernel_size: int = 15
) -> DendriteConv2d:
    E = np.zeros((kernel_size, kernel_size))
    I_var = np.zeros_like(E)
    midpoint = kernel_size // 2
    left_width = midpoint
    right_width = kernel_size - midpoint
    if direction == "rightward":
        E[:, :midpoint] = gaussian_filter(np.ones((kernel_size, left_width)), 2)
        I_var[:, midpoint:] = gaussian_filter(np.ones((kernel_size, right_width)), 2)
    else:
        E[:, midpoint:] = gaussian_filter(np.ones((kernel_size, right_width)), 2)
        I_var[:, :midpoint] = gaussian_filter(np.ones((kernel_size, left_width)), 2)
    E = E / (E.max() + 1e-6)
    I_var = I_var / (I_var.max() + 1e-6) + 0.1
    return _push_pull_layer(E, I_var)


def create_end_stopped_unit(
    preferred_length: int = 20, *, orientation: float = 90.0
) -> DendriteConv2d:
    k = 25
    x = np.arange(k) - k // 2
    y = np.arange(k) - k // 2
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    E_center = np.exp(-(Xr**2) / (2 * 3**2)) * np.exp(
        -(Yr**2) / (2 * (preferred_length / 4) ** 2)
    )
    I_ends = np.exp(-(Xr**2) / (2 * 5**2)) * (
        np.exp(-((Yr + preferred_length / 2 + 3) ** 2) / (2 * 3**2))
        + np.exp(-((Yr - preferred_length / 2 - 3) ** 2) / (2 * 3**2))
    )
    I_sides = np.exp(-((Xr + 6) ** 2) / (2 * 3**2)) + np.exp(
        -((Xr - 6) ** 2) / (2 * 3**2)
    )
    E = E_center * 5
    I_var = I_ends * 3 + I_sides * 0.5 + 0.1
    return _push_pull_layer(E, I_var)
