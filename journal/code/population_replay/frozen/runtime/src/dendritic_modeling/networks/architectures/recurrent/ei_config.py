"""
Configuration dataclasses for unified E-I layers/networks with optional recurrence.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_backend,
)


@dataclass
class PopulationConfig(BaseConfig):
    """Configuration for one population (E or I)."""

    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])

    ff_excitatory_synapses: int = 8
    ff_inhibitory_synapses: int = 4
    rec_excitatory_synapses: int = 0
    rec_inhibitory_synapses: int = 0
    ff_excitatory_synapses_by_level: list[int] = field(default_factory=list)
    ff_inhibitory_synapses_by_level: list[int] = field(default_factory=list)
    rec_excitatory_synapses_by_level: list[int] = field(default_factory=list)
    rec_inhibitory_synapses_by_level: list[int] = field(default_factory=list)

    topk_init_method: str = "xavier_normal"
    topk_noise_level: float = 0.0
    topk_type: str = "standard"
    topk_weight_norm_order: Optional[int] = None
    topk_gamma: float = 1.0
    topk_temperature: float = 0.0
    topk_ultrafast: bool = False
    topk_strategy: str = "none"
    indexed_seed: Optional[int] = None
    indexed_candidate_size: Optional[int] = None
    indexed_selection: str = "standard"
    indexed_output_chunk_size: int = 2048
    indexed_rewire_frequency: int = 100
    indexed_rewire_quantile: float = 0.05
    indexed_rewire_until_step: Optional[int] = None
    dense_to_sparse_initial_density: float = 1.0
    dense_to_sparse_initial_k: Optional[int] = None
    dense_to_sparse_start_step: int = 0
    dense_to_sparse_end_step: int = 1000
    dense_to_sparse_update_interval: int = 1
    dense_to_sparse_schedule: str = "cubic"
    dense_to_sparse_freeze_on_end: bool = False
    dense_to_sparse_advance_on_forward: bool = True
    dense_to_sparse_prune_metric: str = "weight"
    weight_transform: str = "exp"
    use_shunting: bool = True
    use_additive_normalization: bool = False
    additive_mode: str = "raw"
    additive_tangent_n0: Optional[float] = None
    additive_tangent_t0: Optional[float] = None
    # Values are ordered by soma-relative depth: soma=0, proximal=1, ... .
    additive_tangent_n0_by_depth: list[float] = field(default_factory=list)
    additive_tangent_t0_by_depth: list[float] = field(default_factory=list)
    reactivate: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_init_m: float = 1.0
    reactivation_init_b: float = 0.5
    reactivation_init_policy: str = "analytical"
    reactivation_occupancy_quantile_low: Optional[float] = None
    reactivation_occupancy_quantile_high: Optional[float] = None
    reactivation_occupancy_target_low: Optional[float] = None
    reactivation_occupancy_target_high: Optional[float] = None
    reactivation_calibration_min_quantile_width: float = 1e-3
    reactivation_calibration_max_m: float = 50.0
    reactivation_calibration_revert_on_invalid: bool = True
    reactivation_sigma_aware_k: float = 0.25
    reactivation_strategy: str = "none"
    blocklinear_strategy: str = "none"

    # Per-level taus. Empty list -> auto-generated (distal slower, proximal faster).
    level_taus: list[float] = field(default_factory=list)
    tau_base: float = 50.0
    tau_ratio: float = 3.0  # Geometric ratio between adjacent dendritic levels
    learnable_tau: bool = False  # If True, tau values are learned during training
    allow_self_recurrence: bool = True
    epsilon: float = 1e-8

    # Additional DendriticBranchLayer parameters
    somatic_synapses: bool = True  # Whether soma level gets direct synaptic input
    dbl_init_method: str = "analytical_expectation"
    efficient_blocklinear: bool = False
    print_hooks: bool = False

    # DeepST / implementation options
    excitatory_target_density: float = 0.1
    inhibitory_target_density: float = 0.1
    use_noise: bool = True
    sigma: float = 0.05
    rewiring_mode: str = "global"
    rewire_frequency: int = 1
    synapses_per_branch: Optional[int] = None
    freeze_excitatory_connectivity: bool = False
    freeze_inhibitory_connectivity: bool = False
    init_method: str = "xavier_normal"
    weight_threshold: float = 1e-6
    credit_trace_decay: float = 0.95
    credit_candidate_pool_size: int = 8
    credit_turnover_fraction: float = 0.1
    credit_weak_active_pool_fraction: float = 0.5
    credit_min_observations: int = 5
    credit_swap_margin: float = 0.0
    credit_force_turnover: bool = False
    credit_selection: str = "credit"
    credit_warmup_steps: int = 5
    adaptive_initialization: bool = True
    adaptive_initialization_policy: str = "preserve_shunting_center"
    adaptive_target_conductance: float = 5.0
    initial_child_conductance: float = 1.0
    # When configured, every population/path/depth receives a stable keyed RNG
    # substream so changing an unrelated path does not perturb matched weights.
    initialization_seed: Optional[int] = None
    initialization_namespace: str = ""

    # Optional structured topology constraints. Pathway names are interpreted
    # from the target population's point of view (ee/ie for E targets,
    # ei/ii for I targets, and rec_* for recurrent inputs).
    structured_connectivity: dict[str, Any] = field(default_factory=dict)
    structured_layer_idx: int = 0
    ff_excitatory_pathway: str = "ee"
    ff_inhibitory_pathway: str = "ie"
    rec_excitatory_pathway: str = "rec_ee"
    rec_inhibitory_pathway: str = "rec_ie"

    # Optional receptor/conductance-style synapse types for recurrent temporal
    # integration. Disabled by default, preserving the legacy E/I trace path.
    synapse_types: dict[str, Any] = field(default_factory=dict)

    # Optional soma dynamics. "rate" is the legacy output; "spike" routes the
    # dendritic soma voltage through a LIF soma with surrogate-gradient spikes.
    dynamics_mode: str = "rate"
    spike_threshold: float = 1.0
    spike_reset: float = 0.0
    spike_tau: float = 20.0
    spike_refractory_steps: int = 0
    spike_surrogate_beta: float = 10.0
    spike_readout: str = "spikes"  # spikes | rate | membrane
    spike_readout_tau: float = 20.0

    # Optional dendritic branch spike/plateau events. These act on non-soma
    # dendritic levels before their output is propagated toward the soma.
    dendritic_spikes_enabled: bool = False
    dendritic_spike_mode: str = "plateau"  # plateau | threshold
    dendritic_spike_levels: str | list[int] = "non_soma"
    dendritic_spike_threshold: float = 0.7
    dendritic_spike_plateau_amplitude: float = 1.0
    dendritic_spike_plateau_tau: float = 20.0
    dendritic_spike_refractory_steps: int = 0
    dendritic_spike_surrogate_beta: float = 10.0
    dendritic_spike_propagation: str = "additive"  # additive | replace | gated

    # Optional delayed soma-to-dendrite feedback. Disabled by default, so the
    # original dendrite-to-soma computation is preserved unless explicitly
    # requested. Feedback uses the previous timestep's soma signal.
    soma_feedback_enabled: bool = False
    soma_feedback_mode: str = "additive"  # additive | shunting | gated
    soma_feedback_source: str = "output"  # output | voltage | spike_readout
    soma_feedback_levels: str | list[int] = "non_soma"
    soma_feedback_strength: float = 0.1
    soma_feedback_learnable_strength: bool = False
    soma_feedback_per_level: bool = True
    soma_feedback_init_std: float = 0.0
    soma_feedback_reversal: float = 1.0

    # Appended to retain the positional constructor order of all legacy fields.
    # ``parallel_readout`` removes serial child-to-parent drive and pools every
    # non-soma level directly into the soma readout while retaining the same
    # per-level state banks. ``all_active_star`` retains the serial child
    # aggregators as independent, leaky star-arm drives, but prevents one arm's
    # state from entering the next level's local computation.
    cross_level_mode: str = "hierarchical"

    # Standard autograd uses the exact loss derivative arriving at every
    # compartment. ``soma_broadcast`` retains autograd eligibility derivatives
    # but replaces each non-somatic output derivative by the derivative of its
    # owning soma. This feedforward control isolates the cost of a neuron-level
    # teaching coordinate without changing the forward graph.
    autograd_credit_mode: str = "exact"

    # Appended indexed-synapse scaling controls. Defaults preserve legacy
    # topology storage, checkpoint, and initialization behavior.
    indexed_index_dtype: str = "int64"
    indexed_workspace_mb: Optional[float] = None
    indexed_cache_transformed_weights: bool = False
    indexed_recompute_backward: bool = False
    indexed_persistent_indices: bool = True
    indexed_init_mode: str = "per_rank"
    indexed_projection_backend: str = "eager"
    reactivation_memory_efficient: bool = False

    def __post_init__(self):
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        self.indexed_index_dtype = str(self.indexed_index_dtype).lower()
        if self.indexed_index_dtype not in {"auto", "int32", "int64", "long"}:
            raise ValueError(
                "indexed_index_dtype must be one of 'auto', 'int32', or 'int64', "
                f"got {self.indexed_index_dtype!r}"
            )
        if self.indexed_workspace_mb is not None and self.indexed_workspace_mb <= 0:
            raise ValueError(
                "indexed_workspace_mb must be > 0 when provided, got "
                f"{self.indexed_workspace_mb}"
            )
        self.indexed_projection_backend = normalize_indexed_projection_backend(
            self.indexed_projection_backend
        )
        if self.indexed_recompute_backward and self.indexed_projection_backend not in {
            "eager",
            "recompute",
        }:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={self.indexed_projection_backend!r}"
            )
        self.indexed_init_mode = str(self.indexed_init_mode).lower()
        if self.indexed_init_mode not in {"per_rank", "rank0_broadcast"}:
            raise ValueError(
                "indexed_init_mode must be 'per_rank' or 'rank0_broadcast', "
                f"got {self.indexed_init_mode!r}"
            )
        self.cross_level_mode = str(self.cross_level_mode).lower()
        if self.cross_level_mode not in {
            "hierarchical",
            "parallel_readout",
            "all_active_star",
        }:
            raise ValueError(
                "cross_level_mode must be 'hierarchical', 'parallel_readout', "
                "or 'all_active_star', "
                f"got {self.cross_level_mode!r}"
            )
        self.autograd_credit_mode = str(self.autograd_credit_mode).lower()
        if self.autograd_credit_mode not in {"exact", "soma_broadcast"}:
            raise ValueError(
                "autograd_credit_mode must be 'exact' or 'soma_broadcast', "
                f"got {self.autograd_credit_mode!r}"
            )
        if any(b < 1 for b in self.branch_factors):
            raise ValueError(
                f"branch_factors must all be >= 1, got {self.branch_factors}"
            )
        for name in (
            "ff_excitatory_synapses",
            "ff_inhibitory_synapses",
            "rec_excitatory_synapses",
            "rec_inhibitory_synapses",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in (
            "ff_excitatory_synapses_by_level",
            "ff_inhibitory_synapses_by_level",
            "rec_excitatory_synapses_by_level",
            "rec_inhibitory_synapses_by_level",
        ):
            values = getattr(self, name)
            if any(v < 0 for v in values):
                raise ValueError(f"{name} must contain only >= 0 values, got {values}")
        if self.tau_base <= 0:
            raise ValueError(f"tau_base must be > 0, got {self.tau_base}")
        if self.tau_ratio <= 0:
            raise ValueError(f"tau_ratio must be > 0, got {self.tau_ratio}")
        if self.level_taus and any(t <= 0 for t in self.level_taus):
            raise ValueError(f"level_taus must all be > 0, got {self.level_taus}")
        if (
            not math.isfinite(self.initial_child_conductance)
            or self.initial_child_conductance <= 0
        ):
            raise ValueError(
                "initial_child_conductance must be finite and positive, got "
                f"{self.initial_child_conductance}"
            )
        self.dynamics_mode = str(self.dynamics_mode).lower()
        if self.dynamics_mode not in {"rate", "spike"}:
            raise ValueError(
                f"dynamics_mode must be 'rate' or 'spike', got {self.dynamics_mode!r}"
            )
        if self.spike_tau <= 0:
            raise ValueError(f"spike_tau must be > 0, got {self.spike_tau}")
        if self.spike_readout_tau <= 0:
            raise ValueError(
                f"spike_readout_tau must be > 0, got {self.spike_readout_tau}"
            )
        if self.spike_refractory_steps < 0:
            raise ValueError(
                "spike_refractory_steps must be >= 0, "
                f"got {self.spike_refractory_steps}"
            )
        self.spike_readout = str(self.spike_readout).lower()
        if self.spike_readout not in {"spikes", "rate", "membrane"}:
            raise ValueError(
                "spike_readout must be one of 'spikes', 'rate', or 'membrane', "
                f"got {self.spike_readout!r}"
            )
        self.dendritic_spike_mode = str(self.dendritic_spike_mode).lower()
        if self.dendritic_spike_mode not in {"plateau", "threshold"}:
            raise ValueError(
                "dendritic_spike_mode must be 'plateau' or 'threshold', "
                f"got {self.dendritic_spike_mode!r}"
            )
        self.dendritic_spike_propagation = str(self.dendritic_spike_propagation).lower()
        if self.dendritic_spike_propagation not in {"additive", "replace", "gated"}:
            raise ValueError(
                "dendritic_spike_propagation must be one of "
                "'additive', 'replace', or 'gated', "
                f"got {self.dendritic_spike_propagation!r}"
            )
        if self.dendritic_spike_plateau_tau <= 0:
            raise ValueError(
                "dendritic_spike_plateau_tau must be > 0, "
                f"got {self.dendritic_spike_plateau_tau}"
            )
        if self.dendritic_spike_refractory_steps < 0:
            raise ValueError(
                "dendritic_spike_refractory_steps must be >= 0, "
                f"got {self.dendritic_spike_refractory_steps}"
            )
        if self.dendritic_spike_surrogate_beta <= 0:
            raise ValueError(
                "dendritic_spike_surrogate_beta must be > 0, "
                f"got {self.dendritic_spike_surrogate_beta}"
            )
        self.soma_feedback_mode = str(self.soma_feedback_mode).lower()
        if self.soma_feedback_mode not in {"additive", "shunting", "gated"}:
            raise ValueError(
                "soma_feedback_mode must be one of 'additive', 'shunting', "
                f"or 'gated', got {self.soma_feedback_mode!r}"
            )
        self.soma_feedback_source = str(self.soma_feedback_source).lower()
        if self.soma_feedback_source in {"soma_output", "readout"}:
            self.soma_feedback_source = "output"
        elif self.soma_feedback_source in {"soma_voltage", "membrane", "v_soma"}:
            self.soma_feedback_source = "voltage"
        elif self.soma_feedback_source in {"spike", "spikes", "rate"}:
            self.soma_feedback_source = "spike_readout"
        if self.soma_feedback_source not in {"output", "voltage", "spike_readout"}:
            raise ValueError(
                "soma_feedback_source must be one of 'output', 'voltage', "
                f"or 'spike_readout', got {self.soma_feedback_source!r}"
            )
        if (
            self.soma_feedback_enabled
            and self.soma_feedback_mode == "shunting"
            and not self.use_shunting
        ):
            raise ValueError(
                "soma_feedback_mode='shunting' requires use_shunting=True. "
                "Use soma_feedback_mode='additive' or 'gated' for additive "
                "populations."
            )
        if self.soma_feedback_init_std < 0:
            raise ValueError(
                "soma_feedback_init_std must be >= 0, "
                f"got {self.soma_feedback_init_std}"
            )


@dataclass
class EILayerConfig(BaseConfig):
    """Configuration for one unified E-I layer."""

    excitatory: PopulationConfig = field(default_factory=PopulationConfig)
    inhibitory: Optional[PopulationConfig] = field(
        default_factory=lambda: PopulationConfig(
            n_neurons=16,
            branch_factors=[1],
            ff_excitatory_synapses=8,
            ff_inhibitory_synapses=4,
            rec_excitatory_synapses=0,
            rec_inhibitory_synapses=0,
        )
    )

    # Set/overwritten by network at build time.
    excitatory_input_dim: int = 64
    inhibitory_input_dim: Optional[int] = None
    # If True, excitatory population can consume external inhibitory FF input
    # directly even when no inhibitory population is built in this layer.
    direct_ff_inhibitory_to_excitatory: bool = False

    recurrent: bool = False
    dt: float = 1.0

    def __post_init__(self):
        if self.excitatory_input_dim <= 0:
            raise ValueError(
                f"excitatory_input_dim must be > 0, got {self.excitatory_input_dim}"
            )
        if self.inhibitory_input_dim is not None and self.inhibitory_input_dim <= 0:
            raise ValueError(
                f"inhibitory_input_dim must be > 0 or None, got {self.inhibitory_input_dim}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass
class EINetworkConfig(BaseConfig):
    """Configuration for a stack of unified E-I layers."""

    layers: list[EILayerConfig] = field(default_factory=lambda: [EILayerConfig()])
    input_dim: int = 64
    use_transfer: bool = False
    transfer_params: dict[str, Any] = field(default_factory=dict)
    input_projection_dims: list[int] = field(default_factory=list)
    output_mode: str = "last"  # recurrent only: "last" | "all" | "mean"
    store_routing: bool = False

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if self.output_mode not in ("last", "all", "mean"):
            raise ValueError(
                f"output_mode must be one of ('last','all','mean'), got {self.output_mode}"
            )
