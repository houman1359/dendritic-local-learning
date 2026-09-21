"""
Recurrent architecture module for dendritic modeling.

Unified E-I recurrent architecture with hierarchical dendrites, per-level
temporal traces, and proper 4-compartment synaptic structure (FF E, FF I,
REC E, REC I) per neuron.
"""

from dendritic_modeling.networks.architectures.recurrent.baseline_rnn import (
    BaselineRNN,
    BaselineRNNConfig,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    compute_input_branch_factors,
    compute_level_sizes,
    compute_output_owner_index_per_level,
    resolve_synapses_by_level,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_spikes import (
    DendriticSpikeDynamics,
    previous_dendritic_spike_state,
    resolve_dendritic_spike_levels,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    EILayerConfig,
    EINetworkConfig,
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_layer import EILayer
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.networks.architectures.recurrent.ei_state import (
    DendriNetState,
    EIState,
)
from dendritic_modeling.networks.architectures.recurrent.heterogeneous_leak_ctrnn import (
    HeterogeneousLeakCTRNN,
    HeterogeneousLeakCTRNNConfig,
    HeterogeneousLeakCTRNNState,
    TauFeatureInputRoutingConfig,
    TauOwnerPatternConfig,
)
from dendritic_modeling.networks.architectures.recurrent.legendre_memory import (
    FixedLegendreMemory,
    FixedLegendreMemoryConfig,
    LegendreMemoryState,
    legendre_delay_continuous_matrices,
    zero_order_hold_discretization,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationDefinitionConfig,
    PopulationLayer,
    PopulationLayerConfig,
    PopulationLayerState,
    PopulationNetwork,
    PopulationNetworkConfig,
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.protocol import (
    RecurrentCore,
    SteppableRecurrentCore,
)
from dendritic_modeling.networks.architectures.recurrent.spiking_soma import (
    LIFSoma,
    SurrogateSpike,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    SynapseTypeConfig,
    SynapseTypeGroup,
    SynapseTypeSet,
    build_synapse_type_set,
    validate_additive_synapse_reversals,
)

__all__ = [
    "BaselineRNN",
    "BaselineRNNConfig",
    "DendriNetState",
    "DendriticSpikeDynamics",
    "EILayer",
    "EILayerConfig",
    "EINetwork",
    "EINetworkConfig",
    "EIState",
    "FixedLegendreMemory",
    "FixedLegendreMemoryConfig",
    "HeterogeneousLeakCTRNN",
    "HeterogeneousLeakCTRNNConfig",
    "HeterogeneousLeakCTRNNState",
    "LIFSoma",
    "LegendreMemoryState",
    "PopulationConfig",
    "PopulationDefinitionConfig",
    "PopulationLayer",
    "PopulationLayerConfig",
    "PopulationLayerState",
    "PopulationNetwork",
    "PopulationNetworkConfig",
    "PopulationProjectionConfig",
    "RecurrentCore",
    "StatefulDendriNet",
    "SteppableRecurrentCore",
    "SurrogateSpike",
    "SynapseTypeConfig",
    "SynapseTypeGroup",
    "SynapseTypeSet",
    "TauFeatureInputRoutingConfig",
    "TauOwnerPatternConfig",
    "build_synapse_type_set",
    "compute_input_branch_factors",
    "compute_level_sizes",
    "compute_output_owner_index_per_level",
    "legendre_delay_continuous_matrices",
    "previous_dendritic_spike_state",
    "resolve_dendritic_spike_levels",
    "resolve_synapses_by_level",
    "validate_additive_synapse_reversals",
    "zero_order_hold_discretization",
]
