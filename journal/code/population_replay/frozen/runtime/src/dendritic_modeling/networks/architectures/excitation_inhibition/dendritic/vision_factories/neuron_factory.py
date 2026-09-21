"""Factory for building dendritic neurons from configuration."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import yaml

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

from .vision_units import (
    create_center_surround_unit,
    create_end_stopped_unit,
    create_gabor_unit,
    create_motion_unit,
)

TemplateBuilder = Callable[[dict[str, Any]], DendriteConv2d]

__all__ = ["build_neuron_from_yaml", "load_properties_config"]


def load_properties_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """Load visual properties configuration.

    Parameters
    ----------
    config_path : str or None
        Path to properties.yaml. If None, uses default.

    Returns
    -------
    config : dict
        Properties configuration
    """
    if config_path is None:
        # Use default config - go up to project root then into configs
        config_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
            / "configs"
            / "properties.yaml"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def _build_gabor_template(params: dict[str, Any]) -> DendriteConv2d:
    return create_gabor_unit(
        orientation=params.get("orientation", 0),
        wavelength=params.get("wavelength", 8),
        sigma=params.get("sigma", 3.0),
    )


def _build_center_surround_template(params: dict[str, Any]) -> DendriteConv2d:
    return create_center_surround_unit(
        kind=params.get("cell_type", "on-center"),
        center_size=params.get("center_size", 5),
        surround_size=params.get("surround_size", 11),
    )


def _build_motion_template(params: dict[str, Any]) -> DendriteConv2d:
    return create_motion_unit(
        direction=params.get("preferred_direction", "rightward"),
        kernel_size=params.get("kernel_size", 15),
    )


def _build_end_stopped_template(params: dict[str, Any]) -> DendriteConv2d:
    return create_end_stopped_unit(
        preferred_length=params.get("preferred_length", 20),
        orientation=params.get("orientation", 90),
    )


_TEMPLATE_BUILDERS: dict[str, TemplateBuilder] = {
    "center_surround": _build_center_surround_template,
    "end_stopped": _build_end_stopped_template,
    "gabor": _build_gabor_template,
    # Complex cell with multiple phase Gabors. For now, preserve legacy behavior
    # by returning the same single Gabor unit as the previous branch.
    "gabor_pair": _build_gabor_template,
    "motion_ds": _build_motion_template,
}


def _build_neuron_from_template(
    template: str,
    params: dict[str, Any],
) -> DendriteConv2d:
    builder = _TEMPLATE_BUILDERS.get(template)
    if builder is None:
        raise ValueError(f"Unknown weight template: {template}")
    return builder(params)


def build_neuron_from_yaml(
    property_tag: str, config: Optional[dict[str, Any]] = None
) -> DendriteConv2d:
    """Build a dendritic neuron from property configuration.

    Parameters
    ----------
    property_tag : str
        Property tag from properties.yaml (e.g., 'V1_OSI', 'retina_dog')
    config : dict or None
        Properties config dict. If None, loads default.

    Returns
    -------
    neuron : DendriteConv2d
        Configured dendritic neuron
    """
    if config is None:
        config = load_properties_config()

    if property_tag not in config:
        raise ValueError(f"Unknown property tag: {property_tag}")

    prop_config = config[property_tag]
    template = prop_config["weight_template"]
    params = prop_config.get("params", {})

    return _build_neuron_from_template(template, params)


def get_stimulus_params(
    property_tag: str, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Get stimulus parameters for a property.

    Parameters
    ----------
    property_tag : str
        Property tag
    config : dict or None
        Properties config

    Returns
    -------
    stim_params : dict
        Stimulus type and parameters
    """
    if config is None:
        config = load_properties_config()

    prop_config = config[property_tag]
    return prop_config["stimulus"]


def get_criterion(
    property_tag: str, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Get success criterion for a property.

    Parameters
    ----------
    property_tag : str
        Property tag
    config : dict or None
        Properties config

    Returns
    -------
    criterion : dict
        Metric name and threshold
    """
    if config is None:
        config = load_properties_config()

    prop_config = config[property_tag]
    return prop_config["criterion"]
