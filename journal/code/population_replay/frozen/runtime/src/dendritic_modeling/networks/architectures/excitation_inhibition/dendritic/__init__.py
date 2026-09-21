"""
Dendritic network architectures.

This module provides complete dendritic network implementations including
DendriNet and DendriNetWithOutputs, as well as vision neuron factories.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
    DendriNetConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    analytical_expectation_dbl_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    AdaptiveInitConfig,
    DeepSTConfig,
    DendriticSynapseConfig,
    DenseToSparseConfig,
    IndexedSynapseConfig,
    MorphologyConfig,
    ReactivationConfig,
    TopKConfig,
)

# Vision neuron factories
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories import (
    build_neuron_from_yaml,
    create_center_surround_unit,
    create_end_stopped_unit,
    create_gabor_unit,
    create_motion_unit,
)

__all__ = [
    "AdaptiveInitConfig",
    "BlockLinear",
    "DeepSTConfig",
    "DendriNet",
    "DendriNetConfig",
    "DendriticBranchConfig",
    "DendriticBranchLayer",
    "DendriticSynapseConfig",
    "DenseToSparseConfig",
    "GradientScaler",
    "IndexedSynapseConfig",
    "MorphologyConfig",
    "ReactivationConfig",
    "TopKConfig",
    "analytical_expectation_dbl_init",
    "build_neuron_from_yaml",
    "create_center_surround_unit",
    "create_end_stopped_unit",
    "create_gabor_unit",
    "create_motion_unit",
]
