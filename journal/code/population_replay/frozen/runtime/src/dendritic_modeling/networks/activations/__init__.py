"""
Activation functions module.

This module provides various activation/reactivation functions and a factory
to build the desired activation function from config.
"""

from dendritic_modeling.networks.activations.factory import (
    ActivationFactory,
    get_activation_builder,
    get_registered_activation_types,
    register_activation_builder,
    unregister_activation_builder,
)
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.activations.resolution import (
    resolve_dendritic_activation,
)

__all__ = [
    "ActivationFactory",
    "ParametricActivation",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "get_activation_builder",
    "get_registered_activation_types",
    "register_activation_builder",
    "resolve_dendritic_activation",
    "unregister_activation_builder",
]
