"""Step-result helpers for stateful recurrent dendritic populations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dendritic_modeling.networks.architectures.recurrent.dendritic_spikes import (
    DendriticSpikeDynamics,
    previous_dendritic_spike_state,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState


@dataclass(frozen=True)
class _StatefulPreviousLevelTensors:
    """Previous traces and branch voltage for one dendritic level."""

    trace_E: Tensor
    trace_E_rec: Tensor
    trace_I: Tensor
    trace_I_rec: Tensor
    trace_branch: Tensor
    branch_voltage: Tensor


@dataclass(frozen=True)
class _StatefulMaskedLevelTensors:
    """Level tensors after applying an optional keep mask."""

    raw_E: Tensor
    raw_E_rec: Tensor
    raw_I: Tensor
    raw_I_rec: Tensor
    raw_branch: Tensor
    branch_conductance: Tensor
    prev_trace_E: Tensor
    prev_trace_E_rec: Tensor
    prev_trace_I: Tensor
    prev_trace_I_rec: Tensor
    prev_trace_branch: Tensor
    prev_branch_voltage: Tensor


@dataclass
class _StatefulStepBuffers:
    """Mutable trace buffers populated during one recurrent step."""

    trace_E: list[Tensor]
    trace_E_rec: list[Tensor]
    trace_I: list[Tensor]
    trace_I_rec: list[Tensor]
    trace_branch: list[Tensor]
    level_voltage_cache: list[Tensor]
    typed_traces: dict[str, list[Tensor]] | None
    dendritic_spike_plateau: list[Tensor] | None
    dendritic_spike_refractory: list[Tensor] | None
    dendritic_spike_events: list[Tensor] | None


def _init_stateful_step_buffers(
    *,
    synapse_types_enabled: bool,
    dendritic_spikes_enabled: bool,
) -> _StatefulStepBuffers:
    """Create empty mutable buffers for one recurrent step."""
    return _StatefulStepBuffers(
        trace_E=[],
        trace_E_rec=[],
        trace_I=[],
        trace_I_rec=[],
        trace_branch=[],
        level_voltage_cache=[],
        typed_traces={} if synapse_types_enabled else None,
        dendritic_spike_plateau=[] if dendritic_spikes_enabled else None,
        dendritic_spike_refractory=[] if dendritic_spikes_enabled else None,
        dendritic_spike_events=[] if dendritic_spikes_enabled else None,
    )


def _stateful_level_keep_mask(
    active_level_silencing: Tensor | None,
    *,
    level_idx: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve a per-level keep mask from the optional lesion mask."""
    if active_level_silencing is None:
        return None

    mask = active_level_silencing.to(device=device)
    if mask.dim() == 1:
        level_mask = mask[level_idx].expand(batch_size)
    elif mask.dim() == 2:
        level_mask = mask[:, level_idx]
    else:
        raise ValueError(
            "Level silencing mask must have shape [n_levels] or [batch, n_levels]"
        )

    keep = 1.0 - level_mask.to(dtype=dtype).clamp(0.0, 1.0)
    return keep.unsqueeze(-1)


def _apply_stateful_dendritic_spike_dynamics(
    *,
    dynamics: DendriticSpikeDynamics,
    level_idx: int,
    branch_voltage: Tensor,
    state: DendriNetState,
    keep_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply configured dendritic spike dynamics to one level."""
    prev_plateau = previous_dendritic_spike_state(
        state.dendritic_spike_plateau,
        level_idx,
        branch_voltage,
    )
    prev_refractory = previous_dendritic_spike_state(
        state.dendritic_spike_refractory,
        level_idx,
        branch_voltage,
    )
    return dynamics.apply(
        level_idx=level_idx,
        branch_voltage=branch_voltage,
        prev_plateau=prev_plateau,
        prev_refractory=prev_refractory,
        keep_mask=keep_mask,
    )


def _apply_stateful_soma_dynamics(
    *,
    spiking_soma: torch.nn.Module | None,
    soma_feedback_enabled: bool,
    soma_voltage: Tensor,
    state: DendriNetState,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Apply optional spiking soma dynamics to one stateful step."""
    if spiking_soma is None:
        v_soma = soma_voltage if soma_feedback_enabled else state.v_soma
        return (
            soma_voltage,
            v_soma,
            state.refractory_counter,
            state.spike_readout,
        )
    return spiking_soma(
        soma_voltage,
        state.v_soma,
        state.refractory_counter,
        state.spike_readout,
    )[:4]


def _stateful_decay_factors(
    *,
    learnable_tau: bool,
    log_tau: Tensor | None,
    decays: Tensor | None,
    dt: float,
) -> Tensor:
    """Return per-level decay factors for fixed or learnable time constants."""
    if learnable_tau:
        tau = log_tau.exp()
        return torch.exp(-dt / tau)
    return decays


def _stateful_current_taus(
    *,
    learnable_tau: bool,
    log_tau: Tensor | None,
    decays: Tensor | None,
    dt: float,
) -> Tensor:
    """Return inspectable per-level tau values for fixed or learnable taus."""
    if learnable_tau:
        return log_tau.exp().detach()
    return -dt / decays.clamp(min=1e-8).log()


def _init_stateful_dendrinet_state(
    *,
    level_dims: list[int],
    n_soma: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    spiking_soma_enabled: bool,
    soma_feedback_enabled: bool,
    synapse_types_enabled: bool,
    dendritic_spikes_enabled: bool,
) -> DendriNetState:
    """Create the zero state for one stateful dendritic population."""
    return DendriNetState.zeros(
        level_dims=level_dims,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        soma_dim=n_soma if spiking_soma_enabled or soma_feedback_enabled else None,
        track_branch_voltage=synapse_types_enabled or dendritic_spikes_enabled,
        track_dendritic_spikes=dendritic_spikes_enabled,
        track_soma_output=soma_feedback_enabled,
    )


def _stateful_raw_branch_drive(
    *,
    layer: torch.nn.Module,
    branch_voltage: Tensor | None,
    batch_size: int,
    level_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Resolve branch input drive from child branches or a zero fallback."""
    if layer.input_branches and branch_voltage is not None:
        return layer.branches_to_output(branch_voltage)
    return torch.zeros(batch_size, level_dim, device=device, dtype=dtype)


def _ensure_stateful_level_tensor(
    value,
    *,
    batch_size: int,
    level_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a level-shaped zero tensor when an absent pathway returns scalar 0."""
    if torch.is_tensor(value):
        return value
    return torch.zeros(batch_size, level_dim, device=device, dtype=dtype)


def _stateful_raw_currents(
    *,
    layer: torch.nn.Module,
    x: Tensor,
    inhibitory_input: Tensor | None,
    recurrent_input: Tensor | None,
    rec_inhibitory_input: Tensor | None,
    batch_size: int,
    level_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute and normalize the four raw synaptic current pathways."""
    raw_E, raw_E_rec, raw_I, raw_I_rec = layer.compute_raw_currents(
        x=x,
        inhibitory_input=inhibitory_input,
        recurrent_input=recurrent_input,
        rec_inhibitory_input=rec_inhibitory_input,
    )
    return (
        _ensure_stateful_level_tensor(
            raw_E,
            batch_size=batch_size,
            level_dim=level_dim,
            device=device,
            dtype=dtype,
        ),
        _ensure_stateful_level_tensor(
            raw_E_rec,
            batch_size=batch_size,
            level_dim=level_dim,
            device=device,
            dtype=dtype,
        ),
        _ensure_stateful_level_tensor(
            raw_I,
            batch_size=batch_size,
            level_dim=level_dim,
            device=device,
            dtype=dtype,
        ),
        _ensure_stateful_level_tensor(
            raw_I_rec,
            batch_size=batch_size,
            level_dim=level_dim,
            device=device,
            dtype=dtype,
        ),
    )


def _stateful_previous_level_tensors(
    state: DendriNetState,
    *,
    level_idx: int,
) -> _StatefulPreviousLevelTensors:
    """Return previous traces and branch voltage for one dendritic level."""
    prev_trace_E = state.trace_E[level_idx]
    prev_trace_E_rec = state.trace_E_rec[level_idx]
    prev_trace_I = state.trace_I[level_idx]
    prev_trace_I_rec = state.trace_I_rec[level_idx]
    prev_trace_branch = state.trace_branch[level_idx]
    prev_branch_voltage = (
        state.branch_voltage[level_idx]
        if state.branch_voltage is not None and level_idx < len(state.branch_voltage)
        else prev_trace_branch
    )
    return _StatefulPreviousLevelTensors(
        trace_E=prev_trace_E,
        trace_E_rec=prev_trace_E_rec,
        trace_I=prev_trace_I,
        trace_I_rec=prev_trace_I_rec,
        trace_branch=prev_trace_branch,
        branch_voltage=prev_branch_voltage,
    )


def _integrate_stateful_legacy_traces(
    *,
    decay: Tensor,
    one_minus_decay: Tensor,
    prev_trace_E: Tensor,
    prev_trace_E_rec: Tensor,
    prev_trace_I: Tensor,
    prev_trace_I_rec: Tensor,
    raw_E: Tensor,
    raw_E_rec: Tensor,
    raw_I: Tensor,
    raw_I_rec: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Integrate legacy excitatory/inhibitory traces for one stateful level."""
    return (
        decay * prev_trace_E + one_minus_decay * raw_E,
        decay * prev_trace_E_rec + one_minus_decay * raw_E_rec,
        decay * prev_trace_I + one_minus_decay * raw_I,
        decay * prev_trace_I_rec + one_minus_decay * raw_I_rec,
    )


def _integrate_stateful_branch_trace(
    *,
    decay: Tensor,
    one_minus_decay: Tensor,
    prev_trace_branch: Tensor,
    raw_branch: Tensor,
) -> Tensor:
    """Integrate the child-branch trace for one stateful level."""
    return decay * prev_trace_branch + one_minus_decay * raw_branch


def _apply_stateful_keep_mask_to_level_tensors(
    *,
    keep_mask: Tensor | None,
    raw_E: Tensor,
    raw_E_rec: Tensor,
    raw_I: Tensor,
    raw_I_rec: Tensor,
    raw_branch: Tensor,
    branch_conductance: Tensor,
    prev_trace_E: Tensor,
    prev_trace_E_rec: Tensor,
    prev_trace_I: Tensor,
    prev_trace_I_rec: Tensor,
    prev_trace_branch: Tensor,
    prev_branch_voltage: Tensor,
) -> _StatefulMaskedLevelTensors:
    """Apply a resolved level keep mask to all tensors that carry level state."""
    if keep_mask is None:
        return _StatefulMaskedLevelTensors(
            raw_E=raw_E,
            raw_E_rec=raw_E_rec,
            raw_I=raw_I,
            raw_I_rec=raw_I_rec,
            raw_branch=raw_branch,
            branch_conductance=branch_conductance,
            prev_trace_E=prev_trace_E,
            prev_trace_E_rec=prev_trace_E_rec,
            prev_trace_I=prev_trace_I,
            prev_trace_I_rec=prev_trace_I_rec,
            prev_trace_branch=prev_trace_branch,
            prev_branch_voltage=prev_branch_voltage,
        )
    return _StatefulMaskedLevelTensors(
        raw_E=raw_E * keep_mask,
        raw_E_rec=raw_E_rec * keep_mask,
        raw_I=raw_I * keep_mask,
        raw_I_rec=raw_I_rec * keep_mask,
        raw_branch=raw_branch * keep_mask,
        branch_conductance=branch_conductance * keep_mask,
        prev_trace_E=prev_trace_E * keep_mask,
        prev_trace_E_rec=prev_trace_E_rec * keep_mask,
        prev_trace_I=prev_trace_I * keep_mask,
        prev_trace_I_rec=prev_trace_I_rec * keep_mask,
        prev_trace_branch=prev_trace_branch * keep_mask,
        prev_branch_voltage=prev_branch_voltage * keep_mask,
    )


def _apply_stateful_branch_voltage_modulators(
    *,
    branch_voltage: Tensor,
    soma_feedback_gate: Tensor | None,
    keep_mask: Tensor | None,
) -> Tensor:
    """Apply post-voltage soma feedback and level keep-mask modulation."""
    if soma_feedback_gate is not None:
        branch_voltage = branch_voltage * (1.0 + soma_feedback_gate)
    if keep_mask is not None:
        branch_voltage = branch_voltage * keep_mask
    return branch_voltage


def _apply_stateful_legacy_voltage(
    *,
    layer: torch.nn.Module,
    trace_E: Tensor,
    trace_E_rec: Tensor,
    trace_I: Tensor,
    trace_I_rec: Tensor,
    trace_branch: Tensor,
    branch_conductance: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Compute legacy branch voltage from integrated E/I traces."""
    voltage, denominator = layer.voltage_from_currents(
        trace_E=trace_E,
        trace_E_rec=trace_E_rec,
        trace_I=trace_I,
        trace_I_rec=trace_I_rec,
        trace_branch=trace_branch,
        branch_conductance=branch_conductance,
    )
    return layer.reactivation(voltage), denominator


def _build_stateful_next_state(
    *,
    trace_E: list[Tensor],
    trace_E_rec: list[Tensor],
    trace_I: list[Tensor],
    trace_I_rec: list[Tensor],
    trace_branch: list[Tensor],
    level_voltage_cache: list[Tensor],
    typed_traces: dict[str, list[Tensor]] | None,
    output: Tensor,
    v_soma: Tensor | None,
    refractory_counter: Tensor | None,
    spike_readout: Tensor | None,
    dendritic_spike_plateau: list[Tensor] | None,
    dendritic_spike_refractory: list[Tensor] | None,
    dendritic_spike_events: list[Tensor] | None,
    track_branch_voltage: bool,
    track_soma_output: bool,
) -> DendriNetState:
    """Build the next recurrent state from the tensors produced by one step."""
    return DendriNetState(
        trace_E=trace_E,
        trace_E_rec=trace_E_rec,
        trace_I=trace_I,
        trace_I_rec=trace_I_rec,
        trace_branch=trace_branch,
        branch_voltage=level_voltage_cache if track_branch_voltage else None,
        typed_traces=typed_traces,
        soma_output=output if track_soma_output else None,
        v_soma=v_soma,
        refractory_counter=refractory_counter,
        spike_readout=spike_readout,
        dendritic_spike_plateau=dendritic_spike_plateau,
        dendritic_spike_refractory=dendritic_spike_refractory,
        dendritic_spike_events=dendritic_spike_events,
    )


def _build_stateful_routing_info(
    *,
    trace_E: list[Tensor],
    trace_E_rec: list[Tensor],
    trace_I: list[Tensor],
    trace_I_rec: list[Tensor],
    level_voltage_cache: list[Tensor],
    n_levels: int,
) -> dict[str, Tensor]:
    """Build per-level routing diagnostics from completed step tensors."""
    eps = 1e-8
    level_voltage_l2 = []
    level_voltage_rms = []
    ff_rec_ratios = []
    for level_idx in range(n_levels):
        voltage = level_voltage_cache[level_idx]
        l2 = voltage.norm(dim=-1)
        level_voltage_l2.append(l2)
        level_voltage_rms.append(l2 / float(voltage.shape[-1]) ** 0.5)

        ff_mag = trace_E[level_idx].abs().sum(dim=-1) + trace_I[level_idx].abs().sum(
            dim=-1
        )
        rec_mag = trace_E_rec[level_idx].abs().sum(dim=-1) + trace_I_rec[
            level_idx
        ].abs().sum(dim=-1)
        ff_rec_ratios.append(ff_mag / (ff_mag + rec_mag + eps))

    return {
        # Historical alias retained for old consumers. Raw L2 magnitude is not
        # dimension-invariant and should not be interpreted as causal routing.
        "level_contributions": torch.stack(level_voltage_l2, dim=-1),
        "level_activity_l2": torch.stack(level_voltage_l2, dim=-1),
        "level_activity_rms": torch.stack(level_voltage_rms, dim=-1),
        "ff_rec_ratios": torch.stack(ff_rec_ratios, dim=-1),
    }


__all__ = [
    "_StatefulMaskedLevelTensors",
    "_StatefulPreviousLevelTensors",
    "_StatefulStepBuffers",
    "_apply_stateful_branch_voltage_modulators",
    "_apply_stateful_dendritic_spike_dynamics",
    "_apply_stateful_keep_mask_to_level_tensors",
    "_apply_stateful_legacy_voltage",
    "_apply_stateful_soma_dynamics",
    "_build_stateful_next_state",
    "_build_stateful_routing_info",
    "_init_stateful_dendrinet_state",
    "_init_stateful_step_buffers",
    "_integrate_stateful_branch_trace",
    "_integrate_stateful_legacy_traces",
    "_stateful_current_taus",
    "_stateful_decay_factors",
    "_stateful_level_keep_mask",
    "_stateful_previous_level_tensors",
    "_stateful_raw_branch_drive",
    "_stateful_raw_currents",
]
