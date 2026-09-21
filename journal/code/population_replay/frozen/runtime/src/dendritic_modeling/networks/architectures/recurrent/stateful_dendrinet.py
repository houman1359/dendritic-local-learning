"""
Stateful DendriNet with optional recurrent compartments and per-level traces.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    compute_level_sizes,
    compute_output_owner_index_per_level,
    sum_level_values_by_output_owner,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.stateful_feedback import (
    _apply_stateful_soma_feedback_effect,
    _build_stateful_soma_feedback,
    _previous_stateful_soma_feedback_source,
    _stateful_soma_feedback_for_level,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_layers import (
    _build_stateful_branch_layers,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_setup import (
    _active_input_dim as _active_input_dim,
    _active_synapse_count as _active_synapse_count,
    _build_stateful_spiking_soma,
    _resolve_stateful_branch_synapse_layout as _resolve_stateful_branch_synapse_layout,
    _resolve_stateful_dendritic_spike_setup,
    _resolve_stateful_level_taus,
    _resolve_stateful_soma_feedback_levels,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_step import (
    _apply_stateful_branch_voltage_modulators,
    _apply_stateful_dendritic_spike_dynamics,
    _apply_stateful_keep_mask_to_level_tensors,
    _apply_stateful_legacy_voltage,
    _apply_stateful_soma_dynamics,
    _build_stateful_next_state,
    _build_stateful_routing_info,
    _init_stateful_dendrinet_state,
    _init_stateful_step_buffers,
    _integrate_stateful_branch_trace,
    _integrate_stateful_legacy_traces,
    _stateful_current_taus,
    _stateful_decay_factors,
    _stateful_level_keep_mask,
    _stateful_previous_level_tensors,
    _stateful_raw_branch_drive,
    _stateful_raw_currents,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_typed_dynamics import (
    StatefulTypedDynamicsMixin,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    build_synapse_type_set,
    validate_additive_synapse_reversals,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
)


class StatefulDendriNet(StatefulTypedDynamicsMixin, nn.Module):
    """Dendritic population module for feedforward and recurrent operation.

    Constructs a hierarchy of DendriticBranchLayers with all 4 synaptic
    compartments (FF E, FF I, REC E, REC I). In feedforward mode, recurrent
    compartments are not constructed (None dims). In recurrent mode, per-level
    temporal traces are maintained with geometrically spaced time constants.

    When ``store_routing`` is True, each call to :meth:`step` populates
    ``_last_routing_info`` with per-level contribution and FF/REC decomposition.
    """

    def __init__(
        self,
        pop_config: PopulationConfig,
        excitatory_input_dim: int,
        inhibitory_input_dim: Optional[int] = None,
        recurrent_excitatory_input_dim: Optional[int] = None,
        recurrent_inhibitory_input_dim: Optional[int] = None,
        recurrent_excitatory_is_self_population: bool = False,
        recurrent_inhibitory_is_self_population: bool = False,
        connection_masks_by_pathway=None,
        dt: float = 1.0,
    ):
        super().__init__()
        self.pop_config = pop_config
        self.dt = dt
        self.synapse_types = build_synapse_type_set(pop_config.synapse_types)
        self.synapse_types_enabled = bool(self.synapse_types.enabled)
        if self.synapse_types_enabled and not bool(pop_config.use_shunting):
            if str(pop_config.additive_mode).lower() != "raw":
                raise ValueError(
                    "Typed reversal-potential synapses currently support only "
                    "additive_mode='raw' when use_shunting=false; their signed "
                    "current decomposition does not share the legacy N/T control "
                    "equations."
                )
            self._validate_additive_synapse_reversals()
        self.dynamics_mode = str(pop_config.dynamics_mode).lower()
        self.spiking_soma = _build_stateful_spiking_soma(
            pop_config,
            dynamics_mode=self.dynamics_mode,
            dt=dt,
        )

        # Routing analysis flags (set externally or via EINetwork)
        self.store_routing: bool = False
        self._last_routing_info: dict = {}
        self._active_level_silencing: Optional[torch.Tensor] = None
        self._n_soma = pop_config.n_neurons
        self._level_dims = compute_level_sizes(
            n_soma=pop_config.n_neurons,
            branch_factors=pop_config.branch_factors,
        )
        self._output_owner_index_per_level = compute_output_owner_index_per_level(
            self._level_dims, self._n_soma
        )
        self.n_levels = len(self._level_dims)
        self.cross_level_mode = str(pop_config.cross_level_mode).lower()
        self.autograd_credit_mode = str(pop_config.autograd_credit_mode).lower()
        if self.is_parallel_readout and self.n_levels < 2:
            raise ValueError(
                "cross_level_mode='parallel_readout' requires at least two "
                "levels (a non-soma level and a soma)"
            )
        if self.is_all_active_star and self.n_levels < 2:
            raise ValueError(
                "cross_level_mode='all_active_star' requires at least two "
                "levels (a non-soma level and a soma)"
            )
        dendritic_spike_setup = _resolve_stateful_dendritic_spike_setup(
            pop_config,
            n_levels=self.n_levels,
            dt=self.dt,
        )
        self.dendritic_spikes_enabled = dendritic_spike_setup.enabled
        self.dendritic_spike_mode = dendritic_spike_setup.mode
        self.dendritic_spike_threshold = dendritic_spike_setup.threshold
        self.dendritic_spike_plateau_amplitude = dendritic_spike_setup.plateau_amplitude
        self.dendritic_spike_plateau_tau = dendritic_spike_setup.plateau_tau
        self.dendritic_spike_refractory_steps = dendritic_spike_setup.refractory_steps
        self.dendritic_spike_surrogate_beta = dendritic_spike_setup.surrogate_beta
        self.dendritic_spike_propagation = dendritic_spike_setup.propagation
        self.dendritic_spike_level_indices = dendritic_spike_setup.level_indices
        self.dendritic_spike_dynamics = dendritic_spike_setup.dynamics
        self.soma_feedback_enabled = bool(pop_config.soma_feedback_enabled)
        self.soma_feedback_mode = str(pop_config.soma_feedback_mode).lower()
        self.soma_feedback_source = str(pop_config.soma_feedback_source).lower()
        self.soma_feedback_per_level = bool(pop_config.soma_feedback_per_level)
        self.soma_feedback_reversal = float(pop_config.soma_feedback_reversal)
        self.soma_feedback_level_indices = _resolve_stateful_soma_feedback_levels(
            self.soma_feedback_enabled,
            pop_config.soma_feedback_levels,
            self.n_levels,
        )
        soma_feedback_build = _build_stateful_soma_feedback(
            pop_config=pop_config,
            level_dims=self._level_dims,
            n_soma=self._n_soma,
            level_indices=self.soma_feedback_level_indices,
            per_level=self.soma_feedback_per_level,
            enabled=self.soma_feedback_enabled,
        )
        self._soma_feedback_strength_index = soma_feedback_build.strength_index
        self.soma_feedback_projections = soma_feedback_build.projections
        if soma_feedback_build.strength_is_parameter:
            self.soma_feedback_strength = nn.Parameter(soma_feedback_build.strength)
        else:
            self.register_buffer("soma_feedback_strength", soma_feedback_build.strength)

        level_taus = _resolve_stateful_level_taus(pop_config, self.n_levels)

        self.learnable_tau = bool(pop_config.learnable_tau)
        if self.learnable_tau:
            # Store log(tau) as learnable parameter; tau = exp(log_tau) > 0.
            # decay = exp(-dt / tau) is computed dynamically in step().
            self.log_tau = nn.Parameter(
                torch.tensor([math.log(tau) for tau in level_taus], dtype=torch.float32)
            )
        else:
            self.log_tau = None
            self.register_buffer(
                "decays",
                torch.tensor(
                    [math.exp(-dt / tau) for tau in level_taus], dtype=torch.float32
                ),
            )

        branch_layer_build = _build_stateful_branch_layers(
            pop_config=pop_config,
            n_levels=self.n_levels,
            level_dims=self._level_dims,
            output_owner_index_per_level=self._output_owner_index_per_level,
            excitatory_input_dim=excitatory_input_dim,
            inhibitory_input_dim=inhibitory_input_dim,
            recurrent_excitatory_input_dim=recurrent_excitatory_input_dim,
            recurrent_inhibitory_input_dim=recurrent_inhibitory_input_dim,
            recurrent_excitatory_is_self_population=(
                recurrent_excitatory_is_self_population
            ),
            recurrent_inhibitory_is_self_population=(
                recurrent_inhibitory_is_self_population
            ),
            connection_masks_by_pathway=connection_masks_by_pathway,
        )
        self.branch_layers = branch_layer_build.branch_layers
        self.synapse_config = branch_layer_build.synapse_config
        self.synapse_configs = tuple(
            branch_layer.synapse_config for branch_layer in self.branch_layers
        )
        self.branch_configs = tuple(
            branch_layer.branch_config for branch_layer in self.branch_layers
        )
        if self.cross_level_mode == "parallel_readout":
            self._configure_parallel_readout()
        elif self.cross_level_mode == "all_active_star":
            self._configure_all_active_star()

    @property
    def n_soma(self) -> int:
        return self._n_soma

    @property
    def output_dim(self) -> int:
        return self._n_soma

    @property
    def is_parallel_readout(self) -> bool:
        """Whether levels update independently and pool directly to the soma."""
        return self.cross_level_mode == "parallel_readout"

    @property
    def is_all_active_star(self) -> bool:
        """Whether serial-shaped branch states form independent soma arms."""
        return self.cross_level_mode == "all_active_star"

    @property
    def level_dims(self) -> list[int]:
        return self._level_dims

    def _configure_parallel_readout(self) -> None:
        """Replace serial child aggregators with parameter-matched direct pooling.

        For local pre-reactivation voltages ``v_l``, reactivations ``phi_l``,
        soma-local shunting numerator ``N_s``, denominator ``D_s``, and direct
        projection conductance ``G``, the rate-mode control computes

        ``C = sum_{l < s} P_l phi_l(v_l)``

        ``G = sum_{l < s} sum_conductances(P_l)``

        ``y = phi_s((N_s + C) / (D_s + G + epsilon))``.

        Thus every non-soma level and the soma are reactivated exactly once. Each
        ``P_l`` is a fixed-support positive block projection from the level width
        directly to ``n_soma``. Its number of trainable scalars equals that level
        width, so replacing the serial aggregators preserves total parameters.
        """
        unsupported: list[str] = []
        if self.dynamics_mode != "rate":
            unsupported.append("spiking soma dynamics")
        if self.synapse_types_enabled:
            unsupported.append("typed synapses")
        if self.dendritic_spikes_enabled:
            unsupported.append("dendritic spikes")
        if self.soma_feedback_enabled:
            unsupported.append("soma feedback")
        if not self.pop_config.use_shunting:
            unsupported.append("additive/non-shunting dynamics")
        if self.pop_config.blocklinear_strategy != "none":
            unsupported.append(
                "non-default blocklinear gradient scaling "
                f"({self.pop_config.blocklinear_strategy!r})"
            )
        if unsupported:
            raise ValueError(
                "cross_level_mode='parallel_readout' does not yet support "
                + ", ".join(unsupported)
            )
        if self.pop_config.weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
            raise ValueError(
                "cross_level_mode='parallel_readout' requires a positive "
                "weight_transform, got "
                f"{self.pop_config.weight_transform!r}"
            )
        projection_cls = (
            EfficientBlockLinear
            if self.pop_config.efficient_blocklinear
            else BlockLinear
        )
        projections = nn.ModuleList()
        for level_idx, level_dim in enumerate(self._level_dims[:-1]):
            projection = projection_cls(
                level_dim,
                self._n_soma,
                weight_transform=self.pop_config.weight_transform,
            )
            # Each direct projection replaces exactly one serial child
            # aggregator and therefore has the same number of trainable
            # scalars.  Copy the initialized raw conductances before deleting
            # the serial module.  This makes the one-stage control exactly
            # equal to the hierarchy at initialization and avoids granting the
            # grouped-point control a different axial operating point.
            source = self.branch_layers[level_idx + 1].branches_to_output
            if projection.log_weight.numel() != source.log_weight.numel():
                raise RuntimeError(
                    "parallel readout projection and replaced child "
                    "aggregator must contain the same number of parameters"
                )
            with torch.no_grad():
                projection.log_weight.copy_(
                    source.log_weight.reshape_as(projection.log_weight)
                )
            projections.append(projection)
        self.parallel_readout_projections = projections

        # Child aggregators have the same total scalar count as the direct
        # projections. Remove them rather than retaining unused capacity.
        for layer in self.branch_layers[1:]:
            if hasattr(layer, "branches_to_output"):
                del layer.branches_to_output
            layer.input_branches = False

    def _configure_all_active_star(self) -> None:
        """Validate an all-active, resource-identical independent-arm control.

        Unlike ``parallel_readout``, this mode retains every serial child
        aggregator.  Aggregator outputs drive the receiver-width branch traces,
        with the same receiver tau as the hierarchy, but those traces do not
        enter receiver-local voltages.  They are instead pooled independently
        at the soma through a parameter-free owner sum.
        """

        unsupported: list[str] = []
        if self.dynamics_mode != "rate":
            unsupported.append("spiking soma dynamics")
        if self.synapse_types_enabled:
            unsupported.append("typed synapses")
        if self.dendritic_spikes_enabled:
            unsupported.append("dendritic spikes")
        if self.soma_feedback_enabled:
            unsupported.append("soma feedback")
        if not self.pop_config.use_shunting:
            unsupported.append("additive/non-shunting dynamics")
        if self.pop_config.blocklinear_strategy != "none":
            unsupported.append(
                "non-default blocklinear gradient scaling "
                f"({self.pop_config.blocklinear_strategy!r})"
            )
        if unsupported:
            raise ValueError(
                "cross_level_mode='all_active_star' does not yet support "
                + ", ".join(unsupported)
            )
        if self.pop_config.weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
            raise ValueError(
                "cross_level_mode='all_active_star' requires a positive "
                "weight_transform, got "
                f"{self.pop_config.weight_transform!r}"
            )
        for level_idx, layer in enumerate(self.branch_layers[1:], start=1):
            if not layer.input_branches or not hasattr(layer, "branches_to_output"):
                raise ValueError(
                    "cross_level_mode='all_active_star' requires a child "
                    f"aggregator at level {level_idx}"
                )

    def _parallel_readout_soma_voltage(
        self,
        *,
        layer: nn.Module,
        local_numerator: torch.Tensor,
        local_denominator: torch.Tensor,
        level_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pool independent level outputs through the soma shunting equation.

        Each positive projection contributes its activity-weighted current to
        the numerator and its activity-independent transformed conductance sum
        to the denominator, matching the primary branch shunting law while
        replacing only the serial cross-level graph.
        """
        direct_current = torch.zeros_like(local_numerator)
        direct_conductance = torch.zeros_like(local_numerator[:1])
        for projection, level_output in zip(
            self.parallel_readout_projections,
            level_outputs,
            strict=True,
        ):
            direct_current = direct_current + projection(level_output)
            projection_conductance = projection.sum_conductances().to(
                device=direct_current.device,
                dtype=direct_current.dtype,
            )
            direct_conductance = direct_conductance + projection_conductance.unsqueeze(
                0
            )
        return (local_numerator + direct_current) / (
            local_denominator + direct_conductance + layer.epsilon
        )

    def _all_active_star_soma_voltage(
        self,
        *,
        layer: nn.Module,
        local_numerator: torch.Tensor,
        local_denominator: torch.Tensor,
        branch_traces: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pool every serial-reachable child trace as an independent soma arm.

        A level lesion removes both parts of that arm's shunting contribution:
        its dynamic trace current is already masked during state integration,
        and its static coupling conductance is masked here with the same
        sample-specific keep mask.
        """

        if len(branch_traces) != self.n_levels:
            raise ValueError(
                f"expected {self.n_levels} branch-trace levels, "
                f"got {len(branch_traces)}"
            )
        direct_current = torch.zeros_like(local_numerator)
        direct_conductance = torch.zeros_like(local_numerator[:1])
        for level_idx in range(1, self.n_levels):
            branch_layer = self.branch_layers[level_idx]
            aggregation = branch_layer.branches_to_output
            direct_current = direct_current + sum_level_values_by_output_owner(
                branch_traces[level_idx],
                self._n_soma,
            )
            conductance = aggregation.sum_conductances().to(
                device=direct_current.device,
                dtype=direct_current.dtype,
            )
            owner_conductance = sum_level_values_by_output_owner(
                conductance.unsqueeze(0),
                self._n_soma,
            )
            keep_mask = self._level_keep_mask(
                level_idx=level_idx,
                batch_size=local_numerator.shape[0],
                device=direct_current.device,
                dtype=direct_current.dtype,
            )
            if keep_mask is not None:
                owner_conductance = owner_conductance * keep_mask
            direct_conductance = direct_conductance + owner_conductance
        return (local_numerator + direct_current) / (
            local_denominator + direct_conductance + layer.epsilon
        )

    def get_decays(self) -> torch.Tensor:
        """Return per-level decay factors, computing from log_tau if learnable."""
        return _stateful_decay_factors(
            learnable_tau=self.learnable_tau,
            log_tau=self.log_tau,
            decays=getattr(self, "decays", None) if self.learnable_tau else self.decays,
            dt=self.dt,
        )

    def set_level_silencing(self, mask: Optional[torch.Tensor]) -> None:
        """Set an optional per-level lesion mask used during recurrent analysis.

        Supported shapes:
        - [n_levels]: same silencing pattern for every sample in the batch
        - [batch, n_levels]: sample-specific silencing per level
        Values are interpreted as 1=silence, 0=keep.
        """
        self._active_level_silencing = mask

    def clear_level_silencing(self) -> None:
        self._active_level_silencing = None

    def _level_keep_mask(
        self,
        level_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return _stateful_level_keep_mask(
            self._active_level_silencing,
            level_idx=level_idx,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    @property
    def current_taus(self) -> torch.Tensor:
        """Current tau values (useful for inspection/logging)."""
        return _stateful_current_taus(
            learnable_tau=self.learnable_tau,
            log_tau=self.log_tau,
            decays=getattr(self, "decays", None) if self.learnable_tau else self.decays,
            dt=self.dt,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> DendriNetState:
        return _init_stateful_dendrinet_state(
            level_dims=self._level_dims,
            n_soma=self._n_soma,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            spiking_soma_enabled=self.spiking_soma is not None,
            soma_feedback_enabled=self.soma_feedback_enabled,
            synapse_types_enabled=self.synapse_types_enabled,
            dendritic_spikes_enabled=self.dendritic_spikes_enabled,
        )

    def _resolve_soma_feedback_levels(self, levels) -> set[int]:
        return _resolve_stateful_soma_feedback_levels(
            self.soma_feedback_enabled,
            levels,
            self.n_levels,
        )

    def _previous_soma_feedback_source(
        self,
        state: DendriNetState,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return _previous_stateful_soma_feedback_source(
            enabled=self.soma_feedback_enabled,
            source_mode=self.soma_feedback_source,
            spiking_soma_enabled=self.spiking_soma is not None,
            state=state,
            batch_size=batch_size,
            n_soma=self._n_soma,
            device=device,
            dtype=dtype,
        )

    def _soma_feedback_for_level(
        self,
        level_idx: int,
        source: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        return _stateful_soma_feedback_for_level(
            enabled=self.soma_feedback_enabled,
            per_level=self.soma_feedback_per_level,
            level_indices=self.soma_feedback_level_indices,
            strength_index=self._soma_feedback_strength_index,
            projections=self.soma_feedback_projections,
            strength=self.soma_feedback_strength,
            level_idx=level_idx,
            source=source,
            reference=reference,
        )

    def _validate_additive_synapse_reversals(self) -> None:
        validate_additive_synapse_reversals(self.synapse_types)

    def _apply_soma_dynamics(
        self,
        soma_voltage: torch.Tensor,
        state: DendriNetState,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        return _apply_stateful_soma_dynamics(
            spiking_soma=self.spiking_soma,
            soma_feedback_enabled=self.soma_feedback_enabled,
            soma_voltage=soma_voltage,
            state=state,
        )

    def _apply_dendritic_spike_dynamics(
        self,
        *,
        level_idx: int,
        branch_voltage: torch.Tensor,
        state: DendriNetState,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _apply_stateful_dendritic_spike_dynamics(
            dynamics=self.dendritic_spike_dynamics,
            level_idx=level_idx,
            branch_voltage=branch_voltage,
            state=state,
            keep_mask=keep_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Feedforward pass, using one stateful step when opt-in dynamics need state."""
        if self.is_parallel_readout:
            return self._forward_parallel_readout(x, inhibitory_input)
        if (
            self.synapse_types_enabled
            or self.spiking_soma is not None
            or self.dendritic_spikes_enabled
            or self.soma_feedback_enabled
        ):
            output, _ = self.step(x=x, inhibitory_input=inhibitory_input)
            return output
        if self.is_all_active_star:
            return self._forward_all_active_star(x, inhibitory_input)
        branch_output = None
        soma_gradient: dict[str, torch.Tensor] = {}
        for level_idx, layer in enumerate(self.branch_layers):
            branch_output = layer(
                x=x,
                inhibitory_input=inhibitory_input,
                branch_input=branch_output,
                recurrent_input=None,
                rec_inhibitory_input=None,
            )
            if (
                self.autograd_credit_mode == "soma_broadcast"
                and torch.is_grad_enabled()
                and branch_output.requires_grad
            ):
                if level_idx == self.n_levels - 1:

                    def store_soma_gradient(gradient: torch.Tensor) -> torch.Tensor:
                        soma_gradient["value"] = gradient
                        return gradient

                    branch_output.register_hook(store_soma_gradient)
                else:
                    owner_index = self._output_owner_index_per_level[level_idx]

                    def broadcast_soma_gradient(
                        gradient: torch.Tensor,
                        *,
                        owner: torch.Tensor = owner_index,
                    ) -> torch.Tensor:
                        del gradient
                        if "value" not in soma_gradient:
                            raise RuntimeError(
                                "soma-broadcast autograd did not receive the "
                                "soma derivative before a compartment derivative"
                            )
                        soma = soma_gradient["value"]
                        return soma.index_select(
                            1,
                            owner.to(device=soma.device, dtype=torch.long),
                        )

                    branch_output.register_hook(broadcast_soma_gradient)
        return branch_output

    def _forward_parallel_readout(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate a grouped-point emulation with direct soma readout.

        Every non-somatic level retains its original input mask, synaptic
        parameters and local shunting computation, but it receives no child
        output.  Its once-reactivated state is projected directly to the soma
        by the parameter-matched projection that replaced its serial child
        aggregator.  This is the feedforward counterpart of the recurrent
        parallel-readout control and supplies the literal grouped-subunit
        comparison required by the physical-depth experiment.
        """

        level_outputs: list[torch.Tensor] = []
        soma_layer = self.branch_layers[-1]
        soma_local_numerator: torch.Tensor | None = None
        soma_local_denominator: torch.Tensor | None = None

        for level_idx, layer in enumerate(self.branch_layers):
            raw_E, raw_E_rec, raw_I, raw_I_rec = layer.compute_raw_currents(
                x=x,
                inhibitory_input=inhibitory_input,
                recurrent_input=None,
                rec_inhibitory_input=None,
            )
            zeros = torch.zeros(
                x.shape[0],
                self._level_dims[level_idx],
                device=x.device,
                dtype=x.dtype,
            )
            raw_E = raw_E if torch.is_tensor(raw_E) else zeros
            raw_E_rec = raw_E_rec if torch.is_tensor(raw_E_rec) else zeros
            raw_I = raw_I if torch.is_tensor(raw_I) else zeros
            raw_I_rec = raw_I_rec if torch.is_tensor(raw_I_rec) else zeros
            local_voltage, denominator = layer.voltage_from_currents(
                trace_E=raw_E,
                trace_E_rec=raw_E_rec,
                trace_I=raw_I,
                trace_I_rec=raw_I_rec,
                trace_branch=zeros,
                branch_conductance=zeros,
            )
            if level_idx == self.n_levels - 1:
                if denominator is None:
                    raise RuntimeError(
                        "feedforward parallel_readout requires a soma shunting "
                        "denominator"
                    )
                soma_local_numerator = raw_E + raw_E_rec
                soma_local_denominator = denominator
            else:
                level_outputs.append(layer.reactivation(local_voltage))

        if soma_local_numerator is None or soma_local_denominator is None:
            raise RuntimeError("feedforward parallel_readout soma was not initialized")
        pooled_voltage = self._parallel_readout_soma_voltage(
            layer=soma_layer,
            local_numerator=soma_local_numerator,
            local_denominator=soma_local_denominator,
            level_outputs=level_outputs,
        )
        return soma_layer.reactivation(pooled_voltage)

    def _forward_all_active_star(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate a static resource-identical independent-arm control.

        Each level computes its local synaptic voltage without receiving the
        preceding level.  The retained serial-shaped child aggregators instead
        form independent arms that are pooled once at the soma.  Consequently
        the state dictionary, masks, parameter count and active contacts match
        the hierarchy exactly; only serial child-to-parent composition changes.
        With one non-soma level this graph is equivalent to the hierarchy.
        """

        local_outputs: list[torch.Tensor] = []
        arm_currents: list[torch.Tensor] = [
            torch.zeros(
                x.shape[0],
                self._level_dims[0],
                device=x.device,
                dtype=x.dtype,
            )
        ]
        soma_layer = self.branch_layers[-1]
        soma_local_numerator: torch.Tensor | None = None
        soma_local_denominator: torch.Tensor | None = None

        for level_idx, layer in enumerate(self.branch_layers):
            raw_E, raw_E_rec, raw_I, raw_I_rec = layer.compute_raw_currents(
                x=x,
                inhibitory_input=inhibitory_input,
                recurrent_input=None,
                rec_inhibitory_input=None,
            )
            zeros = torch.zeros(
                x.shape[0],
                self._level_dims[level_idx],
                device=x.device,
                dtype=x.dtype,
            )
            raw_E = raw_E if torch.is_tensor(raw_E) else zeros
            raw_E_rec = raw_E_rec if torch.is_tensor(raw_E_rec) else zeros
            raw_I = raw_I if torch.is_tensor(raw_I) else zeros
            raw_I_rec = raw_I_rec if torch.is_tensor(raw_I_rec) else zeros

            if level_idx > 0:
                arm_currents.append(layer.branches_to_output(local_outputs[-1]))

            local_voltage, denominator = layer.voltage_from_currents(
                trace_E=raw_E,
                trace_E_rec=raw_E_rec,
                trace_I=raw_I,
                trace_I_rec=raw_I_rec,
                trace_branch=zeros,
                branch_conductance=zeros,
            )
            if level_idx == self.n_levels - 1:
                if denominator is None:
                    raise RuntimeError(
                        "feedforward all_active_star requires a soma shunting "
                        "denominator"
                    )
                soma_local_numerator = raw_E + raw_E_rec
                soma_local_denominator = denominator
            else:
                local_outputs.append(layer.reactivation(local_voltage))

        if soma_local_numerator is None or soma_local_denominator is None:
            raise RuntimeError("feedforward all_active_star soma was not initialized")
        pooled_voltage = self._all_active_star_soma_voltage(
            layer=soma_layer,
            local_numerator=soma_local_numerator,
            local_denominator=soma_local_denominator,
            branch_traces=arm_currents,
        )
        return soma_layer.reactivation(pooled_voltage)

    def step(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor] = None,
        recurrent_input: Optional[torch.Tensor] = None,
        rec_inhibitory_input: Optional[torch.Tensor] = None,
        state: Optional[DendriNetState] = None,
    ) -> tuple[torch.Tensor, DendriNetState]:
        """Recurrent step with per-level trace integration.

        Args:
            x: FF excitatory input [batch, excitatory_input_dim].
            inhibitory_input: FF inhibitory input [batch, inhibitory_input_dim].
            recurrent_input: Recurrent excitatory input [batch, rec_e_dim].
            rec_inhibitory_input: Recurrent inhibitory input [batch, rec_i_dim].
            state: Previous DendriNetState (auto-initialized if None).

        Returns:
            output: Soma-level voltage [batch, n_soma].
            new_state: Updated DendriNetState with integrated traces.
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if state is None:
            state = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        # Precompute all decay factors (differentiable when learnable_tau=True)
        all_decays = self.get_decays()

        buffers = _init_stateful_step_buffers(
            synapse_types_enabled=self.synapse_types_enabled,
            dendritic_spikes_enabled=self.dendritic_spikes_enabled,
        )

        branch_voltage = None
        parallel_level_outputs: list[torch.Tensor] = []
        soma_local_numerator: torch.Tensor | None = None
        soma_local_denominator: torch.Tensor | None = None
        soma_feedback_source = self._previous_soma_feedback_source(
            state,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        for level_idx, layer in enumerate(self.branch_layers):
            # 1. Compute raw synaptic currents from all 4 compartments
            raw_E, raw_E_rec, raw_I, raw_I_rec = _stateful_raw_currents(
                layer=layer,
                x=x,
                inhibitory_input=inhibitory_input,
                recurrent_input=recurrent_input,
                rec_inhibitory_input=rec_inhibitory_input,
                batch_size=batch_size,
                level_dim=self._level_dims[level_idx],
                device=device,
                dtype=dtype,
            )

            # 2. Branch aggregation from child level
            raw_branch = _stateful_raw_branch_drive(
                layer=layer,
                branch_voltage=(None if self.is_parallel_readout else branch_voltage),
                batch_size=batch_size,
                level_dim=self._level_dims[level_idx],
                device=device,
                dtype=dtype,
            )
            branch_conductance = self._branch_conductance_tensor(layer, raw_branch)

            keep_mask = self._level_keep_mask(
                level_idx=level_idx,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            previous = _stateful_previous_level_tensors(state, level_idx=level_idx)
            masked = _apply_stateful_keep_mask_to_level_tensors(
                keep_mask=keep_mask,
                raw_E=raw_E,
                raw_E_rec=raw_E_rec,
                raw_I=raw_I,
                raw_I_rec=raw_I_rec,
                raw_branch=raw_branch,
                branch_conductance=branch_conductance,
                prev_trace_E=previous.trace_E,
                prev_trace_E_rec=previous.trace_E_rec,
                prev_trace_I=previous.trace_I,
                prev_trace_I_rec=previous.trace_I_rec,
                prev_trace_branch=(
                    torch.zeros_like(previous.trace_branch)
                    if self.is_parallel_readout
                    else previous.trace_branch
                ),
                prev_branch_voltage=previous.branch_voltage,
            )

            # 3. Leaky-integrate all 5 trace types
            decay = all_decays[level_idx].to(device=device, dtype=dtype)
            one_minus_decay = 1 - decay

            trace_branch = _integrate_stateful_branch_trace(
                decay=decay,
                one_minus_decay=one_minus_decay,
                prev_trace_branch=masked.prev_trace_branch,
                raw_branch=masked.raw_branch,
            )
            buffers.trace_branch.append(trace_branch)
            soma_feedback = self._soma_feedback_for_level(
                level_idx,
                soma_feedback_source,
                masked.raw_branch,
            )
            soma_feedback_effect = _apply_stateful_soma_feedback_effect(
                feedback=soma_feedback,
                keep_mask=keep_mask,
                trace_branch=trace_branch,
                branch_conductance=masked.branch_conductance,
                mode=self.soma_feedback_mode,
                reversal=self.soma_feedback_reversal,
            )
            effective_trace_branch = soma_feedback_effect.trace_branch
            effective_branch_conductance = soma_feedback_effect.branch_conductance
            soma_feedback_gate = soma_feedback_effect.gate

            if self.synapse_types_enabled:
                assert buffers.typed_traces is not None
                typed_level = self._integrate_typed_level_traces(
                    raw_E=masked.raw_E,
                    raw_E_rec=masked.raw_E_rec,
                    raw_I=masked.raw_I,
                    raw_I_rec=masked.raw_I_rec,
                    prev_typed_traces=state.typed_traces,
                    new_typed_traces=buffers.typed_traces,
                    level_idx=level_idx,
                    legacy_decay=decay,
                    voltage_reference=masked.prev_branch_voltage,
                )
                buffers.trace_E.append(typed_level.trace_E)
                buffers.trace_E_rec.append(typed_level.trace_E_rec)
                buffers.trace_I.append(typed_level.trace_I)
                buffers.trace_I_rec.append(typed_level.trace_I_rec)

                # 4. Shunting/additive computation from typed conductances.
                branch_voltage = self._apply_typed_voltage(
                    layer,
                    trace_E=typed_level.trace_E,
                    trace_E_rec=typed_level.trace_E_rec,
                    trace_I=typed_level.trace_I,
                    trace_I_rec=typed_level.trace_I_rec,
                    current_E=typed_level.current_E,
                    current_E_rec=typed_level.current_E_rec,
                    current_I=typed_level.current_I,
                    current_I_rec=typed_level.current_I_rec,
                    drive_E=typed_level.drive_E,
                    drive_E_rec=typed_level.drive_E_rec,
                    drive_I=typed_level.drive_I,
                    drive_I_rec=typed_level.drive_I_rec,
                    trace_branch=effective_trace_branch,
                    branch_conductance=effective_branch_conductance,
                )
            else:
                trace_E, trace_E_rec, trace_I, trace_I_rec = (
                    _integrate_stateful_legacy_traces(
                        decay=decay,
                        one_minus_decay=one_minus_decay,
                        prev_trace_E=masked.prev_trace_E,
                        prev_trace_E_rec=masked.prev_trace_E_rec,
                        prev_trace_I=masked.prev_trace_I,
                        prev_trace_I_rec=masked.prev_trace_I_rec,
                        raw_E=masked.raw_E,
                        raw_E_rec=masked.raw_E_rec,
                        raw_I=masked.raw_I,
                        raw_I_rec=masked.raw_I_rec,
                    )
                )
                buffers.trace_E.append(trace_E)
                buffers.trace_E_rec.append(trace_E_rec)
                buffers.trace_I.append(trace_I)
                buffers.trace_I_rec.append(trace_I_rec)

                # 4. Shunting/additive computation from integrated traces.
                if self.is_parallel_readout:
                    local_voltage, denominator = layer.voltage_from_currents(
                        trace_E=trace_E,
                        trace_E_rec=trace_E_rec,
                        trace_I=trace_I,
                        trace_I_rec=trace_I_rec,
                        trace_branch=effective_trace_branch,
                        branch_conductance=effective_branch_conductance,
                    )
                    # Non-soma levels feed their once-reactivated local outputs
                    # to direct projections. The soma remains pre-reactivation
                    # until all direct contributions have been added.
                    branch_voltage = (
                        local_voltage
                        if level_idx == self.n_levels - 1
                        else layer.reactivation(local_voltage)
                    )
                    if level_idx == self.n_levels - 1:
                        if denominator is None:
                            raise RuntimeError(
                                "parallel_readout requires a soma shunting denominator"
                            )
                        soma_local_numerator = trace_E + trace_E_rec
                        soma_local_denominator = denominator
                elif self.is_all_active_star:
                    # The child trace is a live star-arm state, not an input to
                    # the receiver-local computation. This is the single graph
                    # distinction from the serial hierarchy.
                    zero_branch = torch.zeros_like(effective_trace_branch)
                    local_voltage, denominator = layer.voltage_from_currents(
                        trace_E=trace_E,
                        trace_E_rec=trace_E_rec,
                        trace_I=trace_I,
                        trace_I_rec=trace_I_rec,
                        trace_branch=zero_branch,
                        branch_conductance=zero_branch,
                    )
                    branch_voltage = (
                        local_voltage
                        if level_idx == self.n_levels - 1
                        else layer.reactivation(local_voltage)
                    )
                    if level_idx == self.n_levels - 1:
                        if denominator is None:
                            raise RuntimeError(
                                "all_active_star requires a soma shunting denominator"
                            )
                        soma_local_numerator = trace_E + trace_E_rec
                        soma_local_denominator = denominator
                else:
                    branch_voltage, denominator = _apply_stateful_legacy_voltage(
                        layer=layer,
                        trace_E=trace_E,
                        trace_E_rec=trace_E_rec,
                        trace_I=trace_I,
                        trace_I_rec=trace_I_rec,
                        trace_branch=effective_trace_branch,
                        branch_conductance=effective_branch_conductance,
                    )
                self._maybe_update_dynamic_grad_scales(layer, denominator)
            branch_voltage = _apply_stateful_branch_voltage_modulators(
                branch_voltage=branch_voltage,
                soma_feedback_gate=soma_feedback_gate,
                keep_mask=keep_mask,
            )
            if self.is_parallel_readout:
                if level_idx < self.n_levels - 1:
                    parallel_level_outputs.append(branch_voltage)
                else:
                    if soma_local_numerator is None or soma_local_denominator is None:
                        raise RuntimeError(
                            "parallel_readout soma currents were not initialized"
                        )
                    pooled_voltage = self._parallel_readout_soma_voltage(
                        layer=layer,
                        local_numerator=soma_local_numerator,
                        local_denominator=soma_local_denominator,
                        level_outputs=parallel_level_outputs,
                    )
                    # The soma reactivation is applied here, once, and nowhere
                    # earlier on the soma path.
                    branch_voltage = layer.reactivation(pooled_voltage)
                    if keep_mask is not None:
                        branch_voltage = branch_voltage * keep_mask
            elif self.is_all_active_star and level_idx == self.n_levels - 1:
                if soma_local_numerator is None or soma_local_denominator is None:
                    raise RuntimeError(
                        "all_active_star soma currents were not initialized"
                    )
                pooled_voltage = self._all_active_star_soma_voltage(
                    layer=layer,
                    local_numerator=soma_local_numerator,
                    local_denominator=soma_local_denominator,
                    branch_traces=buffers.trace_branch,
                )
                branch_voltage = layer.reactivation(pooled_voltage)
                if keep_mask is not None:
                    branch_voltage = branch_voltage * keep_mask
            if self.dendritic_spikes_enabled:
                branch_voltage, plateau, refractory, event = (
                    self._apply_dendritic_spike_dynamics(
                        level_idx=level_idx,
                        branch_voltage=branch_voltage,
                        state=state,
                        keep_mask=keep_mask,
                    )
                )
                assert buffers.dendritic_spike_plateau is not None
                assert buffers.dendritic_spike_refractory is not None
                assert buffers.dendritic_spike_events is not None
                buffers.dendritic_spike_plateau.append(plateau)
                buffers.dendritic_spike_refractory.append(refractory)
                buffers.dendritic_spike_events.append(event)
            buffers.level_voltage_cache.append(branch_voltage)

        output, v_soma, refractory_counter, spike_readout = self._apply_soma_dynamics(
            branch_voltage,
            state,
        )

        new_state = _build_stateful_next_state(
            trace_E=buffers.trace_E,
            trace_E_rec=buffers.trace_E_rec,
            trace_I=buffers.trace_I,
            trace_I_rec=buffers.trace_I_rec,
            trace_branch=buffers.trace_branch,
            level_voltage_cache=buffers.level_voltage_cache,
            typed_traces=buffers.typed_traces,
            output=output,
            v_soma=v_soma,
            refractory_counter=refractory_counter,
            spike_readout=spike_readout,
            dendritic_spike_plateau=buffers.dendritic_spike_plateau,
            dendritic_spike_refractory=buffers.dendritic_spike_refractory,
            dendritic_spike_events=buffers.dendritic_spike_events,
            track_branch_voltage=(
                self.synapse_types_enabled or self.dendritic_spikes_enabled
            ),
            track_soma_output=self.soma_feedback_enabled,
        )

        if self.store_routing:
            self._last_routing_info = _build_stateful_routing_info(
                trace_E=buffers.trace_E,
                trace_E_rec=buffers.trace_E_rec,
                trace_I=buffers.trace_I,
                trace_I_rec=buffers.trace_I_rec,
                level_voltage_cache=buffers.level_voltage_cache,
                n_levels=self.n_levels,
            )

        return output, new_state
