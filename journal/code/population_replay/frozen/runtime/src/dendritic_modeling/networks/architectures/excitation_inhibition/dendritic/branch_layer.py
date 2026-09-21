"""
Dendritic Branch Layer for neural networks.

This module implements ``DendriticBranchLayer``, an abstract
multi-compartment computation over excitatory and inhibitory inputs.  The
module is inspired by dendritic integration but is not a biophysical neuron
simulation.
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
    compute_branch_raw_currents,
    compute_branch_voltage_from_currents,
    forward_branch_dynamics,
    normalize_additive_mode,
    normalize_branch_additive_voltage,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_grad_scales import (
    compute_branch_grad_scales,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_maintenance import (
    apply_branch_rewiring,
    collect_from_branch_synapse_layers,
    decay_branch_synapse_weights,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_options import (
    resolve_branch_config,
    resolve_branch_layer_synapse_config,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_synapses import (
    create_branch_sparse_layer,
    create_registered_branch_sparse_layer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    analytical_expectation_dbl_init,
    default_dbl_init,
    ei_equivalence_dbl_init,
    identity_weighttransform_dbl_init,
    mechanism_neutral_dbl_init,
    naive_dbl_init,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset


class DendriticBranchLayer(nn.Module):
    """
    Implements an abstract multi-compartment E/I computation along branches at
    one tree depth.

    The layer aggregates converging child-branch values and optional direct
    excitatory and inhibitory inputs using a configured integration rule.  It
    is a trainable computational model inspired by dendritic integration, not
    a biophysical simulation of a biological neuron.

    Parameters
    ----------
    output_dim : int
        The number of output features.

    excitatory_input_dim : int
        The number of input features from excitatory cells.

    excitatory_synapses_per_branch : int
        The number of excitatory synapses per branch.

    inhibitory_input_dim : int, optional
        The number of input features from inhibitory cells. Defaults to `None`.

    inhibitory_synapses_per_branch : int, optional
        The number of inhibitory synapses per branch. Defaults to `None`.

    input_branch_factor : int, optional
        The number of branches converging onto a single branch. Defaults to
        `None`.
    """

    def __init__(
        self,
        output_dim=None,
        excitatory_input_dim=None,
        excitatory_synapses_per_branch=None,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        input_branch_factor=None,
        recurrent_input_dim=None,
        recurrent_synapses_per_branch=None,
        recurrent_forbidden_input_index_per_output=None,
        rec_inhibitory_input_dim=None,
        rec_inhibitory_synapses_per_branch=None,
        rec_inhibitory_forbidden_input_index_per_output=None,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # TopK implementation strategy.
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        topk_temperature=0.5,  # Stochastic TopK temperature.
        topk_ultrafast=True,  # ultrafast stochastic-topk default (set False for rank-probabilistic)
        topk_strategy="none",
        dense_to_sparse_initial_density=1.0,
        dense_to_sparse_initial_k=None,
        dense_to_sparse_start_step=0,
        dense_to_sparse_end_step=1000,
        dense_to_sparse_update_interval=1,
        dense_to_sparse_schedule="cubic",
        dense_to_sparse_freeze_on_end=False,
        dense_to_sparse_advance_on_forward=True,
        dense_to_sparse_prune_metric="weight",
        indexed_rewire_frequency=100,
        indexed_rewire_quantile=0.05,
        indexed_rewire_until_step=None,
        use_shunting=True,
        reactivate=False,
        reactivation_type="param_tanh",
        dendritic_activation=None,
        reactivation_init_m=1,
        reactivation_init_b=0.5,
        reactivation_init_policy="analytical",
        reactivation_occupancy_quantile_low=None,
        reactivation_occupancy_quantile_high=None,
        reactivation_occupancy_target_low=None,
        reactivation_occupancy_target_high=None,
        reactivation_calibration_min_quantile_width=1e-3,
        reactivation_calibration_max_m=50.0,
        reactivation_calibration_revert_on_invalid=True,
        reactivation_sigma_aware_k=0.25,
        dbl_init_method="analytical_expectation",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        efficient_blocklinear: bool = False,
        layer_idx=0,
        print_hooks=False,
        # DeepST-compatible sparse-rewiring parameters.
        use_noise=True,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        sigma=0.05,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        adaptive_initialization=True,
        adaptive_initialization_policy="preserve_shunting_center",
        adaptive_target_conductance=5.0,
        initial_child_conductance=1.0,
        branch_factors=None,  # Branch count per dendritic level.
        initialization_seed=None,
        initialization_namespace="",
        epsilon=1e-8,  # Numerical stability constant.
        weight_transform="exp",  # Weight transformation for positive weights
        excitatory_connection_indices=None,
        inhibitory_connection_indices=None,
        excitatory_connection_mask=None,
        inhibitory_connection_mask=None,
        recurrent_connection_mask=None,
        rec_inhibitory_connection_mask=None,
        branch_config=None,
        synapse_config=None,
        **kwargs,
    ):
        super().__init__()
        provided_branch_config = resolve_branch_config(branch_config)
        if provided_branch_config is not None:
            output_dim = provided_branch_config.output_dim
            excitatory_input_dim = provided_branch_config.excitatory_input_dim
            excitatory_synapses_per_branch = (
                provided_branch_config.excitatory_synapses_per_branch
            )
            inhibitory_input_dim = provided_branch_config.inhibitory_input_dim
            inhibitory_synapses_per_branch = (
                provided_branch_config.inhibitory_synapses_per_branch
            )
            input_branch_factor = provided_branch_config.input_branch_factor
            recurrent_input_dim = provided_branch_config.recurrent_input_dim
            recurrent_synapses_per_branch = (
                provided_branch_config.recurrent_synapses_per_branch
            )
            recurrent_forbidden_input_index_per_output = (
                provided_branch_config.recurrent_forbidden_input_index_per_output
            )
            rec_inhibitory_input_dim = provided_branch_config.rec_inhibitory_input_dim
            rec_inhibitory_synapses_per_branch = (
                provided_branch_config.rec_inhibitory_synapses_per_branch
            )
            rec_inhibitory_forbidden_input_index_per_output = (
                provided_branch_config.rec_inhibitory_forbidden_input_index_per_output
            )
            layer_idx = provided_branch_config.layer_idx
            excitatory_connection_mask = (
                provided_branch_config.excitatory_connection_mask
            )
            inhibitory_connection_mask = (
                provided_branch_config.inhibitory_connection_mask
            )
            recurrent_connection_mask = provided_branch_config.recurrent_connection_mask
            rec_inhibitory_connection_mask = (
                provided_branch_config.rec_inhibitory_connection_mask
            )
            excitatory_connection_indices = (
                provided_branch_config.excitatory_connection_indices
            )
            inhibitory_connection_indices = (
                provided_branch_config.inhibitory_connection_indices
            )
        if output_dim is None:
            raise ValueError("DendriticBranchLayer requires output_dim")
        self.branch_config = DendriticBranchConfig.from_kwargs(
            output_dim=output_dim,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            input_branch_factor=input_branch_factor,
            recurrent_input_dim=recurrent_input_dim,
            recurrent_synapses_per_branch=recurrent_synapses_per_branch,
            recurrent_forbidden_input_index_per_output=(
                recurrent_forbidden_input_index_per_output
            ),
            rec_inhibitory_input_dim=rec_inhibitory_input_dim,
            rec_inhibitory_synapses_per_branch=rec_inhibitory_synapses_per_branch,
            rec_inhibitory_forbidden_input_index_per_output=(
                rec_inhibitory_forbidden_input_index_per_output
            ),
            layer_idx=layer_idx,
            excitatory_connection_indices=excitatory_connection_indices,
            inhibitory_connection_indices=inhibitory_connection_indices,
            excitatory_connection_mask=excitatory_connection_mask,
            inhibitory_connection_mask=inhibitory_connection_mask,
            recurrent_connection_mask=recurrent_connection_mask,
            rec_inhibitory_connection_mask=rec_inhibitory_connection_mask,
        )
        formal_synapse_kwargs = DendriticSynapseConfig.legacy_kwargs_from_mapping(
            locals()
        )
        self.synapse_config, synapse_options, kwargs = (
            resolve_branch_layer_synapse_config(
                formal_synapse_kwargs=formal_synapse_kwargs,
                extra_kwargs=kwargs,
                synapse_config=synapse_config,
            )
        )

        self._apply_synapse_option_attributes(
            synapse_options=synapse_options,
            layer_idx=layer_idx,
            output_dim=output_dim,
            excitatory_input_dim=excitatory_input_dim,
        )
        gradient_scaler = self._create_gradient_scaler()
        self._init_feedforward_synapse_paths(
            synapse_options=synapse_options,
            gradient_scaler=gradient_scaler,
            output_dim=output_dim,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            excitatory_connection_indices=excitatory_connection_indices,
            inhibitory_connection_indices=inhibitory_connection_indices,
            excitatory_connection_mask=excitatory_connection_mask,
            inhibitory_connection_mask=inhibitory_connection_mask,
        )
        self._init_recurrent_synapse_paths(
            synapse_options=synapse_options,
            gradient_scaler=gradient_scaler,
            output_dim=output_dim,
            recurrent_input_dim=recurrent_input_dim,
            recurrent_synapses_per_branch=recurrent_synapses_per_branch,
            recurrent_forbidden_input_index_per_output=(
                recurrent_forbidden_input_index_per_output
            ),
            recurrent_connection_mask=recurrent_connection_mask,
            rec_inhibitory_input_dim=rec_inhibitory_input_dim,
            rec_inhibitory_synapses_per_branch=rec_inhibitory_synapses_per_branch,
            rec_inhibitory_forbidden_input_index_per_output=(
                rec_inhibitory_forbidden_input_index_per_output
            ),
            rec_inhibitory_connection_mask=rec_inhibitory_connection_mask,
        )
        self._init_branch_aggregation(
            input_branch_factor=input_branch_factor,
            output_dim=output_dim,
            gradient_scaler=gradient_scaler,
        )
        self._init_reactivation(
            synapse_options=synapse_options,
            output_dim=output_dim,
            gradient_scaler=gradient_scaler,
            activation_kwargs=kwargs,
        )

        self.set_forward_dynamic_grad_scaling(False)
        self.initialize()

    def _apply_synapse_option_attributes(
        self,
        *,
        synapse_options: dict,
        layer_idx: int,
        output_dim: int,
        excitatory_input_dim: int | None,
    ) -> None:
        """Store normalized synapse options on the layer."""
        self.layer_idx = layer_idx
        self.n_branches = output_dim
        self.reactivation_strategy = synapse_options["reactivation_strategy"]
        self.blocklinear_strategy = synapse_options["blocklinear_strategy"]
        self.efficient_blocklinear = synapse_options["efficient_blocklinear"]
        self.use_shunting = synapse_options["use_shunting"]
        self.use_additive_normalization = synapse_options["use_additive_normalization"]
        self.additive_mode = normalize_additive_mode(synapse_options["additive_mode"])
        if (
            not self.use_shunting
            and self.additive_mode != "raw"
            and self.use_additive_normalization
        ):
            raise ValueError(
                "use_additive_normalization is the legacy per-sample, "
                "across-output z-score "
                "and can only be combined with additive_mode='raw'. Set it to "
                "false for tangent_matched or conductance_normalized controls."
            )
        self.dbl_init_method = synapse_options["dbl_init_method"]
        self.epsilon = synapse_options["epsilon"]
        self.register_buffer("_additive_tangent_n0", torch.tensor(0.0), persistent=True)
        self.register_buffer("_additive_tangent_t0", torch.tensor(0.0), persistent=True)
        self.register_buffer(
            "_additive_tangent_is_set", torch.tensor(False), persistent=True
        )
        tangent_n0 = synapse_options["additive_tangent_n0"]
        tangent_t0 = synapse_options["additive_tangent_t0"]
        if (tangent_n0 is None) != (tangent_t0 is None):
            raise ValueError(
                "additive_tangent_n0 and additive_tangent_t0 must be provided together"
            )
        if tangent_n0 is not None:
            self.set_additive_operating_point(tangent_n0, tangent_t0)
        self._store_diagnostics = False
        self._last_branch_diagnostics: dict = {}
        self._diag_g_tot = None
        self._diag_numerator = None
        self.topk_type = synapse_options["topk_type"]
        self.topk_weight_norm_order = synapse_options["topk_weight_norm_order"]
        self.topk_gamma = synapse_options["topk_gamma"]
        self.topk_temperature = synapse_options["topk_temperature"]
        self.topk_ultrafast = synapse_options["topk_ultrafast"]
        self.topk_strategy = synapse_options["topk_strategy"]
        self.dense_to_sparse_initial_density = synapse_options[
            "dense_to_sparse_initial_density"
        ]
        self.dense_to_sparse_initial_k = synapse_options["dense_to_sparse_initial_k"]
        self.dense_to_sparse_start_step = synapse_options["dense_to_sparse_start_step"]
        self.dense_to_sparse_end_step = synapse_options["dense_to_sparse_end_step"]
        self.dense_to_sparse_update_interval = synapse_options[
            "dense_to_sparse_update_interval"
        ]
        self.dense_to_sparse_schedule = synapse_options["dense_to_sparse_schedule"]
        self.dense_to_sparse_freeze_on_end = synapse_options[
            "dense_to_sparse_freeze_on_end"
        ]
        self.dense_to_sparse_advance_on_forward = synapse_options[
            "dense_to_sparse_advance_on_forward"
        ]
        self.dense_to_sparse_prune_metric = synapse_options[
            "dense_to_sparse_prune_metric"
        ]
        self.indexed_seed = synapse_options["indexed_seed"]
        self.indexed_output_chunk_size = synapse_options["indexed_output_chunk_size"]
        self.indexed_candidate_size = synapse_options["indexed_candidate_size"]
        self.indexed_selection = synapse_options["indexed_selection"]
        self.indexed_rewire_frequency = int(synapse_options["indexed_rewire_frequency"])
        self.indexed_rewire_quantile = float(synapse_options["indexed_rewire_quantile"])
        self.indexed_rewire_until_step = synapse_options["indexed_rewire_until_step"]
        self.indexed_index_dtype = synapse_options["indexed_index_dtype"]
        self.indexed_workspace_mb = synapse_options["indexed_workspace_mb"]
        self.indexed_cache_transformed_weights = synapse_options[
            "indexed_cache_transformed_weights"
        ]
        self.indexed_recompute_backward = synapse_options["indexed_recompute_backward"]
        self.indexed_support_group_rows = synapse_options.get(
            "indexed_support_group_rows", 1
        )
        self.indexed_support_col_block = synapse_options.get(
            "indexed_support_col_block", 1
        )
        self.indexed_projection_backend = synapse_options["indexed_projection_backend"]
        self.indexed_persistent_indices = synapse_options["indexed_persistent_indices"]
        self.indexed_init_mode = synapse_options["indexed_init_mode"]
        self.input_excitatory = excitatory_input_dim is not None
        self.print_hooks = synapse_options["print_hooks"]

        # DeepST-compatible sparse-rewiring parameters.
        self.use_noise = synapse_options["use_noise"]
        self.rewiring_mode = synapse_options["rewiring_mode"]
        self.rewire_frequency = synapse_options["rewire_frequency"]
        self.synapses_per_branch = synapse_options["synapses_per_branch"]
        self.sigma = synapse_options["sigma"]
        self.excitatory_target_density = synapse_options["excitatory_target_density"]
        self.inhibitory_target_density = synapse_options["inhibitory_target_density"]
        self.freeze_excitatory_connectivity = synapse_options[
            "freeze_excitatory_connectivity"
        ]
        self.freeze_inhibitory_connectivity = synapse_options[
            "freeze_inhibitory_connectivity"
        ]
        self.init_method = synapse_options["init_method"]
        self.weight_threshold = synapse_options["weight_threshold"]
        self.credit_trace_decay = synapse_options["credit_trace_decay"]
        self.credit_candidate_pool_size = synapse_options["credit_candidate_pool_size"]
        self.credit_turnover_fraction = synapse_options["credit_turnover_fraction"]
        self.credit_weak_active_pool_fraction = synapse_options[
            "credit_weak_active_pool_fraction"
        ]
        self.credit_min_observations = synapse_options["credit_min_observations"]
        self.credit_swap_margin = synapse_options["credit_swap_margin"]
        self.credit_force_turnover = synapse_options["credit_force_turnover"]
        self.credit_selection = synapse_options["credit_selection"]
        self.credit_warmup_steps = synapse_options["credit_warmup_steps"]
        self.adaptive_initialization = synapse_options["adaptive_initialization"]
        self.adaptive_initialization_policy = synapse_options[
            "adaptive_initialization_policy"
        ]
        self.adaptive_target_conductance = synapse_options[
            "adaptive_target_conductance"
        ]
        self.initial_child_conductance = float(
            synapse_options["initial_child_conductance"]
        )
        if (
            not math.isfinite(self.initial_child_conductance)
            or self.initial_child_conductance <= 0.0
        ):
            raise ValueError(
                "initial_child_conductance must be finite and positive, got "
                f"{self.initial_child_conductance!r}"
            )
        self.branch_factors = synapse_options["branch_factors"]
        self.initialization_seed = synapse_options["initialization_seed"]
        self.initialization_namespace = synapse_options["initialization_namespace"]
        self.weight_transform = synapse_options["weight_transform"]
        self.dendritic_activation = (
            synapse_options["reactivation_type"]
            if synapse_options["reactivate"]
            else "none"
        )

    def _create_gradient_scaler(self) -> GradientScaler:
        """Create the branch-layer gradient-scaling coordinator."""
        return GradientScaler(
            topk_strategy=self.topk_strategy,
            reactivation_strategy=self.reactivation_strategy,
            blocklinear_strategy=self.blocklinear_strategy,
            layer_idx=self.layer_idx,
            print_hooks=self.print_hooks,
        )

    @contextmanager
    def initialization_seed_scope(self, component: str):
        """Run one initialization component in a stable, isolated RNG stream.

        The keyed stream is opt-in. With ``initialization_seed=None`` the code
        follows the legacy global RNG exactly. The scope restores the caller's
        CPU RNG state, so constructing a larger path cannot perturb the next
        population, pathway, or decoder.
        """
        if self.initialization_seed is None:
            yield
            return
        seed = (
            int(self.initialization_seed)
            + stable_seed_offset(
                "dendritic_parameter_initialization",
                self.initialization_namespace,
                int(self.layer_idx),
                component,
            )
        ) % ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield

    def _init_feedforward_synapse_paths(
        self,
        *,
        synapse_options: dict,
        gradient_scaler: GradientScaler,
        output_dim: int,
        excitatory_input_dim: int | None,
        excitatory_synapses_per_branch: int | None,
        inhibitory_input_dim: int | None,
        inhibitory_synapses_per_branch: int | None,
        excitatory_connection_indices,
        inhibitory_connection_indices,
        excitatory_connection_mask,
        inhibitory_connection_mask,
    ) -> None:
        """Create feedforward excitatory and inhibitory sparse synapse paths."""
        if self.input_excitatory:
            self.branch_excitation = self._create_registered_topk_layer(
                in_features=excitatory_input_dim,
                out_features=output_dim,
                K=excitatory_synapses_per_branch,
                init_method=synapse_options["topk_init_method"],
                noise_level=synapse_options["topk_noise_level"],
                gradient_scaler=gradient_scaler,
                synapse_type="exc",
                pathway="ff_excitatory",
                connection_indices=excitatory_connection_indices,
                connection_mask=excitatory_connection_mask,
            )
            self.excitatory_input_dim = excitatory_input_dim
        else:
            self.branch_excitation = None

        # inhibitory_input_dim can be set even when this layer does not consume it.
        self.input_inhibitory = (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
            and inhibitory_synapses_per_branch > 0
        )
        if self.input_inhibitory:
            self.branch_inhibition = self._create_registered_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=synapse_options["topk_init_method"],
                noise_level=synapse_options["topk_noise_level"],
                gradient_scaler=gradient_scaler,
                synapse_type="inh",
                pathway="ff_inhibitory",
                connection_indices=inhibitory_connection_indices,
                connection_mask=inhibitory_connection_mask,
            )
            self.inhibitory_input_dim = inhibitory_input_dim
        else:
            self.branch_inhibition = None

    def _init_recurrent_synapse_paths(
        self,
        *,
        synapse_options: dict,
        gradient_scaler: GradientScaler,
        output_dim: int,
        recurrent_input_dim: int | None,
        recurrent_synapses_per_branch: int | None,
        recurrent_forbidden_input_index_per_output,
        recurrent_connection_mask,
        rec_inhibitory_input_dim: int | None,
        rec_inhibitory_synapses_per_branch: int | None,
        rec_inhibitory_forbidden_input_index_per_output,
        rec_inhibitory_connection_mask,
    ) -> None:
        """Create optional recurrent excitatory and inhibitory sparse paths."""
        self.input_recurrent = (
            recurrent_input_dim is not None
            and recurrent_synapses_per_branch is not None
            and recurrent_synapses_per_branch > 0
        )
        if self.input_recurrent:
            self.branch_recurrent = self._create_registered_topk_layer(
                in_features=recurrent_input_dim,
                out_features=output_dim,
                K=recurrent_synapses_per_branch,
                init_method=synapse_options["topk_init_method"],
                noise_level=synapse_options["topk_noise_level"],
                gradient_scaler=gradient_scaler,
                synapse_type="exc",
                pathway="rec_excitatory",
                forbidden_input_index_per_output=recurrent_forbidden_input_index_per_output,
                connection_mask=recurrent_connection_mask,
            )
            self.recurrent_input_dim = recurrent_input_dim
        else:
            self.branch_recurrent = None

        self.input_rec_inhibitory = (
            rec_inhibitory_input_dim is not None
            and rec_inhibitory_synapses_per_branch is not None
            and rec_inhibitory_synapses_per_branch > 0
        )
        if self.input_rec_inhibitory:
            self.branch_rec_inhibition = self._create_registered_topk_layer(
                in_features=rec_inhibitory_input_dim,
                out_features=output_dim,
                K=rec_inhibitory_synapses_per_branch,
                init_method=synapse_options["topk_init_method"],
                noise_level=synapse_options["topk_noise_level"],
                gradient_scaler=gradient_scaler,
                synapse_type="inh",
                pathway="rec_inhibitory",
                forbidden_input_index_per_output=(
                    rec_inhibitory_forbidden_input_index_per_output
                ),
                connection_mask=rec_inhibitory_connection_mask,
            )
            self.rec_inhibitory_input_dim = rec_inhibitory_input_dim
        else:
            self.branch_rec_inhibition = None

    def _init_branch_aggregation(
        self,
        *,
        input_branch_factor: int | None,
        output_dim: int,
        gradient_scaler: GradientScaler,
    ) -> None:
        """Create optional branch-to-output aggregation."""
        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            BlockCls = (
                EfficientBlockLinear if self.efficient_blocklinear else BlockLinear
            )
            with self.initialization_seed_scope("constructor.branch_aggregation"):
                self.branches_to_output = BlockCls(
                    output_dim * input_branch_factor,
                    output_dim,
                    weight_transform=self.weight_transform,
                )
            gradient_scaler.register_block_linear_dynamic(self.branches_to_output)
        self.input_branch_factor = input_branch_factor

    def _init_reactivation(
        self,
        *,
        synapse_options: dict,
        output_dim: int,
        gradient_scaler: GradientScaler,
        activation_kwargs: dict,
    ) -> None:
        """Create dendritic reactivation and store its initialization options."""
        self.reactivation_init_m = synapse_options["reactivation_init_m"]
        self.reactivation_init_b = synapse_options["reactivation_init_b"]
        self.reactivation_init_policy = normalize_reactivation_init_policy(
            synapse_options["reactivation_init_policy"]
        )
        self.reactivation_occupancy_quantile_low = synapse_options[
            "reactivation_occupancy_quantile_low"
        ]
        self.reactivation_occupancy_quantile_high = synapse_options[
            "reactivation_occupancy_quantile_high"
        ]
        self.reactivation_occupancy_target_low = synapse_options[
            "reactivation_occupancy_target_low"
        ]
        self.reactivation_occupancy_target_high = synapse_options[
            "reactivation_occupancy_target_high"
        ]
        self.reactivation_calibration_min_quantile_width = synapse_options[
            "reactivation_calibration_min_quantile_width"
        ]
        self.reactivation_calibration_max_m = synapse_options[
            "reactivation_calibration_max_m"
        ]
        self.reactivation_calibration_revert_on_invalid = synapse_options[
            "reactivation_calibration_revert_on_invalid"
        ]
        self.reactivation_sigma_aware_k = synapse_options["reactivation_sigma_aware_k"]

        if synapse_options["reactivate"]:
            # initialize() may later overwrite these values based on policy.
            activation_kwargs.setdefault(
                "memory_efficient",
                synapse_options["reactivation_memory_efficient"],
            )
            with self.initialization_seed_scope("constructor.reactivation"):
                self.reactivation = ActivationFactory.create(
                    act_type=synapse_options["reactivation_type"],
                    output_dim=output_dim,
                    init_m=synapse_options["reactivation_init_m"],
                    init_b=synapse_options["reactivation_init_b"],
                    **activation_kwargs,
                )
            if self.use_shunting:
                gradient_scaler.register_reactivation_inverse(self.reactivation)
        else:
            self.reactivation = nn.Identity()
        self.reactivate = synapse_options["reactivate"]

    def set_forward_dynamic_grad_scaling(self, enabled: bool) -> None:
        """Opt TopK sublayers into per-forward parameter-gradient hooks."""

        for attr in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        ):
            layer = getattr(self, attr, None)
            if layer is not None and hasattr(
                layer, "_use_forward_dynamic_grad_scaling"
            ):
                layer._use_forward_dynamic_grad_scaling = bool(enabled)

    def _create_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        synapse_type="exc",
        pathway=None,
        forbidden_input_index_per_output=None,
        connection_indices=None,
        connection_mask=None,
    ):
        """Helper function that picks which topk class to use based on strategy."""
        component = pathway or synapse_type
        with self.initialization_seed_scope(f"constructor.{component}"):
            return create_branch_sparse_layer(
                self,
                in_features=in_features,
                out_features=out_features,
                K=K,
                init_method=init_method,
                noise_level=noise_level,
                synapse_type=synapse_type,
                pathway=pathway,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_indices=connection_indices,
                connection_mask=connection_mask,
            )

    def _create_registered_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        gradient_scaler,
        synapse_type="exc",
        pathway=None,
        forbidden_input_index_per_output=None,
        connection_indices=None,
        connection_mask=None,
    ):
        """Create a sparse synapse layer and register gradient scaling hooks."""
        component = pathway or synapse_type
        with self.initialization_seed_scope(f"constructor.{component}"):
            return create_registered_branch_sparse_layer(
                self,
                gradient_scaler,
                in_features=in_features,
                out_features=out_features,
                K=K,
                init_method=init_method,
                noise_level=noise_level,
                synapse_type=synapse_type,
                pathway=pathway,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_indices=connection_indices,
                connection_mask=connection_mask,
            )

    def initialize(self):
        if (self.weight_transform or "").lower() == "identity":
            identity_weighttransform_dbl_init(self)

        elif self.dbl_init_method == "analytical_expectation":
            analytical_expectation_dbl_init(
                self,
                adaptive=self.adaptive_initialization,
                branch_factors=self.branch_factors,
            )
        elif self.dbl_init_method == "ei_equivalence":
            ei_equivalence_dbl_init(self)
        elif self.dbl_init_method == "naive":
            naive_dbl_init(self)
        elif self.dbl_init_method == "default":
            default_dbl_init(self)
        elif self.dbl_init_method == "mechanism_neutral":
            mechanism_neutral_dbl_init(self)
        else:
            raise ValueError(f"Unknown dbl_init_method: {self.dbl_init_method}")

    def decay_weights(self, weight_decay, weight_boosting=False):
        decay_branch_synapse_weights(self, weight_decay, weight_boosting)

    # Sparse-rewiring maintenance API.
    def apply_rewiring(self):
        """
        Apply rewiring to all synaptic connections.
        """
        apply_branch_rewiring(self)

    @property
    def has_additive_operating_point(self) -> bool:
        """Whether a finite tangent-matching anchor has been frozen."""
        return bool(self._additive_tangent_is_set.item())

    @property
    def additive_operating_point(self) -> tuple[float, float] | None:
        """Return the frozen ``(N0, T0)`` tangent anchor, when configured."""
        if not self.has_additive_operating_point:
            return None
        return (
            float(self._additive_tangent_n0.item()),
            float(self._additive_tangent_t0.item()),
        )

    def set_additive_operating_point(self, n0, t0) -> None:
        """Freeze the scalar operating point used by ``tangent_matched``.

        This method never runs implicitly during ``forward``. Callers can use
        a fixed analytical anchor or explicitly calibrate a shunting reference
        with :meth:`calibrate_additive_operating_point`.
        """

        def _finite_scalar(name, value) -> float:
            tensor = torch.as_tensor(value).detach()
            if tensor.numel() != 1:
                raise ValueError(f"{name} must be a scalar, got shape {tensor.shape}")
            scalar = float(tensor.item())
            if not torch.isfinite(torch.tensor(scalar)):
                raise ValueError(f"{name} must be finite, got {scalar}")
            return scalar

        n0_value = _finite_scalar("n0", n0)
        t0_value = _finite_scalar("t0", t0)
        if 1 + t0_value + self.epsilon <= 0:
            raise ValueError(
                "The tangent operating point requires "
                f"1 + t0 + epsilon > 0, got t0={t0_value}"
            )
        self._additive_tangent_n0.fill_(n0_value)
        self._additive_tangent_t0.fill_(t0_value)
        self._additive_tangent_is_set.fill_(True)

    def calibrate_additive_operating_point(
        self, diagnostics: dict | None = None
    ) -> tuple[float, float]:
        """Freeze ``(N0, T0)`` as means from explicit shunting diagnostics."""
        diagnostics = diagnostics or self._last_branch_diagnostics
        if "N" not in diagnostics or "T" not in diagnostics:
            raise RuntimeError(
                "Calibration requires diagnostics containing N and T. Enable "
                "branch diagnostics on a shunting reference and run it first."
            )
        n_values = diagnostics["N"]
        t_values = diagnostics["T"]
        if not torch.is_tensor(n_values) or not torch.is_tensor(t_values):
            raise TypeError("Calibration diagnostics N and T must be tensors")
        self.set_additive_operating_point(n_values.mean(), t_values.mean())
        operating_point = self.additive_operating_point
        assert operating_point is not None
        return operating_point

    def clear_additive_operating_point(self) -> None:
        """Clear a calibrated tangent anchor."""
        self._additive_tangent_n0.zero_()
        self._additive_tangent_t0.zero_()
        self._additive_tangent_is_set.fill_(False)

    def set_branch_diagnostics(self, enabled: bool = True) -> None:
        """Enable or disable detached E/I/C/G/N/T/V forward diagnostics."""
        self._store_diagnostics = bool(enabled)
        if not enabled:
            self._pending_realized_k = None

    def clear_branch_diagnostics(self) -> None:
        """Discard the most recently stored branch decomposition."""
        self._last_branch_diagnostics = {}
        self._pending_realized_k = None
        self._diag_g_tot = None
        self._diag_numerator = None

    def get_branch_diagnostics(self) -> dict:
        """Return a shallow copy of the most recent detached diagnostics."""
        return dict(self._last_branch_diagnostics)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load old checkpoints while preserving configured tangent anchors."""
        is_set_key = prefix + "_additive_tangent_is_set"
        incoming_is_set = state_dict.get(is_set_key)
        preserve_current = self.has_additive_operating_point and (
            incoming_is_set is None or not bool(incoming_is_set.item())
        )
        for name in (
            "_additive_tangent_n0",
            "_additive_tangent_t0",
            "_additive_tangent_is_set",
        ):
            key = prefix + name
            current = getattr(self, name).detach()
            incoming = state_dict.get(key)
            if incoming is None or preserve_current:
                state_dict[key] = current.clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x,
        inhibitory_input=None,
        branch_input=None,
        recurrent_input=None,
        rec_inhibitory_input=None,
    ):
        return forward_branch_dynamics(
            self,
            x,
            inhibitory_input=inhibitory_input,
            branch_input=branch_input,
            recurrent_input=recurrent_input,
            rec_inhibitory_input=rec_inhibitory_input,
        )

    def compute_raw_currents(
        self, x, inhibitory_input=None, recurrent_input=None, rec_inhibitory_input=None
    ):
        """Compute raw synaptic currents without shunting or reactivation.

        Used by the RNN cell to separate current computation from temporal
        integration and shunting. Feedforward code should use forward() instead.

        Returns:
            raw_E: FF excitatory current [batch, output_dim] or 0
            raw_E_rec: Recurrent excitatory current [batch, output_dim] or 0
            raw_I: FF inhibitory current [batch, output_dim] or 0
            raw_I_rec: Recurrent inhibitory current [batch, output_dim] or 0
        """
        return compute_branch_raw_currents(
            self,
            x,
            inhibitory_input=inhibitory_input,
            recurrent_input=recurrent_input,
            rec_inhibitory_input=rec_inhibitory_input,
        )

    def normalize_additive_voltage(self, voltage):
        """Normalize additive voltages safely across output features."""
        return normalize_branch_additive_voltage(self, voltage)

    def voltage_from_currents(
        self,
        trace_E,
        trace_E_rec,
        trace_I,
        trace_I_rec=0,
        trace_branch=0,
        branch_conductance=None,
    ):
        """Compute pre-reactivation voltage from integrated recurrent traces.

        Branch aggregation mirrors the feedforward shunting equation: branch
        current contributes to the numerator, while branch coupling conductance
        contributes separately to the denominator when provided.
        """
        return compute_branch_voltage_from_currents(
            self,
            trace_E=trace_E,
            trace_E_rec=trace_E_rec,
            trace_I=trace_I,
            trace_I_rec=trace_I_rec,
            trace_branch=trace_branch,
            branch_conductance=branch_conductance,
        )

    def shunt_from_currents(
        self,
        trace_E,
        trace_E_rec,
        trace_I,
        trace_I_rec=0,
        trace_branch=0,
        branch_conductance=None,
    ):
        """Apply shunting (or additive) computation on pre-integrated currents.

        Used by the RNN cell after temporal integration of traces.
        The traces are the leaky-integrated versions of raw currents.

        Args:
            trace_E: Integrated FF excitatory trace [batch, output_dim]
            trace_E_rec: Integrated recurrent excitatory trace (can be 0)
            trace_I: Integrated FF inhibitory trace (can be 0)
            trace_I_rec: Integrated recurrent inhibitory trace (can be 0)

        Returns:
            voltage: Post-shunting, post-reactivation output [batch, output_dim]
        """
        voltage, _denominator = self.voltage_from_currents(
            trace_E=trace_E,
            trace_E_rec=trace_E_rec,
            trace_I=trace_I,
            trace_I_rec=trace_I_rec,
            trace_branch=trace_branch,
            branch_conductance=branch_conductance,
        )
        return self.reactivation(voltage)

    def compute_grad_scales(
        self,
        g_total: torch.Tensor | None = None,
        *,
        use_forward_hooks: bool = False,
    ):
        """
        A conduction-based dynamic scale, returning a flattened vector so each weight
        can get a distinct multiplier. shape [out_features * block_size].
        """
        return compute_branch_grad_scales(
            self,
            g_total=g_total,
            use_forward_hooks=use_forward_hooks,
        )

    def _collect_from_layers(self, method_name):
        """Helper to collect a named attribute from all active TopK layers."""
        return collect_from_branch_synapse_layers(self, method_name)

    def get_weights(self):
        return self._collect_from_layers("weight")

    def get_log_weights(self):
        return self._collect_from_layers("log_weight")

    def get_mask(self):
        return self._collect_from_layers("weight_mask")

    def get_pruned_weights(self):
        return self._collect_from_layers("pruned_weight")

    def get_log_pruned_weights(self):
        return self._collect_from_layers("log_pruned_weight")


__all__ = ["DendriticBranchLayer"]
