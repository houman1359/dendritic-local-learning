"""
Noise Utilities
===============

This module contains utility functions for adding various types of noise
to input data, useful for robustness testing and data augmentation.
"""

import torch

ClampRange = tuple[float, float] | None


def _maybe_clamp(x: torch.Tensor, clamp_range: ClampRange) -> torch.Tensor:
    if clamp_range is None:
        return x
    lo, hi = clamp_range
    return torch.clamp(x, lo, hi)


def add_gaussian_noise(
    x: torch.Tensor,
    std: float = 0.1,
    n_samples: int = 1,
    generator: torch.Generator | None = None,
    clamp_range: ClampRange = (0.0, 1.0),
) -> torch.Tensor:
    """
    Add Gaussian noise to input tensor.

    This function is useful for testing model robustness and for data augmentation.
    The noise is added to multiple samples of the input.

    Args:
        x: Input tensor of shape [batch, features]
        std: Standard deviation of the Gaussian noise
        n_samples: Number of noise samples to generate
        generator: Optional random generator
        clamp_range: Optional ``(min, max)`` output clamp. Use ``None`` for
            unnormalized neural activity.

    Returns:
        Perturbed tensor of shape [n_samples, batch, features]
    """
    # Generate noise samples
    noise = torch.randn(n_samples, *x.shape, device=x.device, generator=generator) * std
    # Add noise to expanded input
    noisy_x = x[None, ...] + noise
    return _maybe_clamp(noisy_x, clamp_range)


def add_uniform_noise(
    x: torch.Tensor,
    spread: float = 0.1,
    n_samples: int = 1,
    generator: torch.Generator | None = None,
    clamp_range: ClampRange = (0.0, 1.0),
) -> torch.Tensor:
    """
    Add uniform noise to input tensor.

    Adds noise sampled from a uniform distribution with specified spread.

    Args:
        x: Input tensor of shape [batch, features]
        spread: Width of uniform distribution [-spread, spread]
        n_samples: Number of noise samples to generate
        generator: Optional random generator
        clamp_range: Optional ``(min, max)`` output clamp. Use ``None`` for
            unnormalized neural activity.

    Returns:
        Perturbed tensor of shape [n_samples, batch, features]
    """
    # Generate uniform noise in [-spread, spread]
    noise = (
        (torch.rand(n_samples, *x.shape, device=x.device, generator=generator) - 0.5)
        * 2
        * spread
    )
    # Add noise to expanded input
    noisy_x = x[None, ...] + noise
    return _maybe_clamp(noisy_x, clamp_range)


def add_poisson_noise(
    x: torch.Tensor,
    rate: float = 1.0,
    n_samples: int = 1,
    clamp_range: ClampRange = (0.0, 1.0),
) -> torch.Tensor:
    """
    Add Poisson noise to input tensor, simulating photon noise.

    This type of noise is particularly relevant for modeling biological
    neural systems and optical imaging scenarios.

    Args:
        x: Input tensor of shape [batch, features]
        rate: Rate parameter for Poisson distribution
        n_samples: Number of noise samples to generate
        clamp_range: Optional ``(min, max)`` output clamp. Use ``None`` for
            unnormalized neural activity.

    Returns:
        Perturbed tensor of shape [n_samples, batch, features]
    """
    # Scale input by rate parameter
    scaled_x = x * rate
    # Generate Poisson samples
    noisy_scaled_x = torch.poisson(scaled_x[None, ...].expand(n_samples, *x.shape))
    # Scale back to original range
    noisy_x = noisy_scaled_x / rate
    return _maybe_clamp(noisy_x, clamp_range)


def add_frequency_noise(
    x: torch.Tensor,
    std: float = 0.1,
    n_samples: int = 1,
    n_channels: int = 1,
    image_shape: tuple[int, int] = (28, 28),
    clamp_range: ClampRange = (0.0, 1.0),
) -> torch.Tensor:
    """
    Add noise in the frequency domain with separate real and imaginary components.

    This function applies noise in the Fourier domain, which can create
    different types of spatial corruption patterns.

    Args:
        x: Input tensor of shape [batch, features]
        std: Standard deviation for noise components
        n_samples: Number of noise samples to generate
        n_channels: Number of channels in the image
        image_shape: Original image shape (height, width)
        clamp_range: Optional ``(min, max)`` output clamp. Use ``None`` for
            unnormalized neural activity.

    Returns:
        Perturbed tensor of shape [n_samples, batch, features]
    """
    batch = x.shape[:-1]
    height, width = image_shape

    # Reshape to image format for FFT
    images = x.view(*batch, n_channels, height, width)

    # Compute 2D FFT
    fft = torch.fft.fft2(images)

    # Generate complex noise
    real_noise = (
        torch.randn(n_samples, *batch, n_channels, height, width, device=x.device) * std
    )
    imag_noise = (
        torch.randn(n_samples, *batch, n_channels, height, width, device=x.device) * std
    )
    freq_noise = torch.complex(real_noise, imag_noise)

    # Add noise in frequency domain
    noisy_fft = fft[None, ...] + freq_noise

    # Inverse FFT to get back to spatial domain
    noisy_images = torch.fft.ifft2(noisy_fft).real

    # Reshape back to original format and clamp
    noisy_x = noisy_images.reshape(n_samples, *batch, -1)
    return _maybe_clamp(noisy_x, clamp_range)
