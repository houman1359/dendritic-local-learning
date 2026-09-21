"""
DEPRECATED: Legacy recurrent network configuration.

These configs (BranchConfig, RecurrentConfig) were used by the old DendriticRNNCell/
DendriticRNNNetwork implementation which has been removed. Kept for backward
compatibility with existing config files.

For new code, use the unified EI config classes:
  - PopulationConfig, EILayerConfig, EINetworkConfig
  from dendritic_modeling.networks.architectures.recurrent.ei_config
"""

import warnings
from dataclasses import dataclass, field

from dendritic_modeling.config.base import BaseConfig


@dataclass
class BranchConfig(BaseConfig):
    """Configuration for a single dendritic branch in a recurrent cell.

    Each branch independently specifies its temporal dynamics (tau),
    input routing (ff/rec synapse budgets), and compartment complexity.
    """

    tau: float = 50.0
    ff_synapses: int = 8
    rec_synapses: int = 8
    ie_synapses: int = 4
    rec_i_synapses: int = 0  # recurrent inhibitory synapses (from local I pool)
    compartment: str = "topk"  # "topk" | "dendritic_branch" | "dendrinet" | "linear"
    activation: str = "relu"
    # Only used when compartment="dendrinet":
    dendrinet_branch_factors: list[int] = field(default_factory=lambda: [2])
    # Only used when compartment="dendritic_branch":
    use_shunting: bool = True
    reactivate: bool = False
    reactivation_type: str = "param_tanh"
    reactivation_init_m: float = 1.0
    reactivation_init_b: float = 0.5
    topk_init_method: str = "xavier_normal"
    epsilon: float = 1e-8
    weight_transform: str = "exp"

    def __post_init__(self):
        for name in ("ff_synapses", "rec_synapses", "ie_synapses", "rec_i_synapses"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if self.ff_synapses == 0 and self.rec_synapses == 0:
            raise ValueError(
                f"Branch with compartment='{self.compartment}' must have "
                f"at least one of ff_synapses or rec_synapses > 0 "
                f"(otherwise the branch receives no input)"
            )
        valid_compartments = ("topk", "dendritic_branch", "dendrinet", "linear")
        if self.compartment not in valid_compartments:
            raise ValueError(
                f"compartment must be one of {valid_compartments}, "
                f"got '{self.compartment}'"
            )
        if self.compartment != "dendritic_branch" and self.rec_i_synapses > 0:
            raise ValueError(
                "rec_i_synapses is only valid when compartment='dendritic_branch' "
                f"(got compartment='{self.compartment}', rec_i_synapses={self.rec_i_synapses})"
            )


@dataclass
class RecurrentConfig(BaseConfig):
    """Master configuration for recurrent networks.

    When enabled=False (default), the recurrent system is completely inert
    and has zero impact on the feedforward pipeline.
    """

    enabled: bool = False  # DEPRECATED: use unified EI config classes instead
    cell_type: str = "dendritic"  # "dendritic" | "gru" | "lstm"
    n_excitatory: int = 64
    n_inhibitory: int = 16
    num_layers: int = 1
    branches: list = field(
        default_factory=lambda: [
            BranchConfig(tau=10.0, ff_synapses=12, rec_synapses=0),
            BranchConfig(tau=50.0, ff_synapses=6, rec_synapses=6),
            BranchConfig(tau=200.0, ff_synapses=0, rec_synapses=10),
        ]
    )
    i_tau: float = 10.0
    i_compartment: str = "topk"
    dt: float = 1.0
    learn_taus: bool = False
    g_leak: float = 1.0
    soma_mode: str = "shunting"  # "shunting" | "additive" | "sum"
    output_mode: str = "last"  # "last" | "all" | "mean"
    input_projection_layers: list[int] = field(default_factory=lambda: [64])
    dropout: float = 0.0
    grad_clip: float = 1.0
    store_routing: bool = False

    def __post_init__(self):
        if self.enabled:
            warnings.warn(
                "RecurrentConfig is deprecated. Use the unified EI config classes: "
                "PopulationConfig, EILayerConfig, EINetworkConfig from "
                "dendritic_modeling.networks.architectures.recurrent.ei_config",
                DeprecationWarning,
                stacklevel=2,
            )
