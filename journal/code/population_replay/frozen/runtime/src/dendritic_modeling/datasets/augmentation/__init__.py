"""
Data augmentation utilities.

This module provides various noise and augmentation functions for data preprocessing.
"""

from .noise import (
    add_frequency_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
)

__all__ = [
    "add_frequency_noise",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_uniform_noise",
]
