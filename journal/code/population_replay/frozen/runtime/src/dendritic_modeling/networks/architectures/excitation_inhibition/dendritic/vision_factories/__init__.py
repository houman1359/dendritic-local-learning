"""
Vision neuron factories for dendritic modeling.

This module provides factory functions for creating biologically-inspired
vision neurons using dendritic computation, including:
- Gabor orientation-selective units
- Center-surround receptive fields
- Motion-selective units
- End-stopped cells
"""

from .neuron_factory import (
    build_neuron_from_yaml,
    get_criterion,
    get_stimulus_params,
    load_properties_config,
)
from .vision_units import (
    create_center_surround_unit,
    create_end_stopped_unit,
    create_gabor_unit,
    create_motion_unit,
    rf_gabor,
    rf_off_center,
    rf_on_center,
)

__all__ = [
    # Factory functions
    "build_neuron_from_yaml",
    "create_center_surround_unit",
    "create_end_stopped_unit",
    # Vision units
    "create_gabor_unit",
    "create_motion_unit",
    "get_criterion",
    "get_stimulus_params",
    "load_properties_config",
    "rf_gabor",
    "rf_off_center",
    "rf_on_center",
]
