"""
Excitation-Inhibition Network architectures.

This module contains network architectures based on excitation-inhibition principles.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    AdaptiveInitConfig,
    BlockLinear,
    DeepSTConfig,
    DendriNet,
    DendriNetConfig,
    DendriticBranchConfig,
    DendriticBranchLayer,
    DendriticSynapseConfig,
    DenseToSparseConfig,
    GradientScaler,
    IndexedSynapseConfig,
    MorphologyConfig,
    ReactivationConfig,
    TopKConfig,
    analytical_expectation_dbl_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform import (
    IdentityInputTransform,
    IndexInputTransform,
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    CreditGatedDeepstLinear,
    DeepstLinear,
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
    distance_kernel_probabilities,
    normalize_pathway_name,
    sample_configured_mask,
    sample_distance_kernel_mask,
    sample_fixed_indegree_mask,
    sample_probability_mask,
    validate_connection_mask,
)

__all__ = [
    "AdaptiveInitConfig",
    "BlockLinear",
    "ConfigurableEINetwork",
    "CreditGatedDeepstLinear",
    "DeepSTConfig",
    "DeepstLinear",
    "DendriNet",
    "DendriNetConfig",
    "DendriticBranchConfig",
    "DendriticBranchLayer",
    "DendriticSynapseConfig",
    "DenseToSparseConfig",
    "DenseToSparseLinear",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "IdentityInputTransform",
    "IndexInputTransform",
    "IndexedDynamicTopKLinear",
    "IndexedRewireLinear",
    "IndexedSparseLinear",
    "IndexedSynapseConfig",
    "MorphologyConfig",
    "ReactivationConfig",
    "StochasticTopKLinear",
    "TopKConfig",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
    "distance_kernel_probabilities",
    "normalize_pathway_name",
    "sample_configured_mask",
    "sample_distance_kernel_mask",
    "sample_fixed_indegree_mask",
    "sample_probability_mask",
    "validate_connection_mask",
]
