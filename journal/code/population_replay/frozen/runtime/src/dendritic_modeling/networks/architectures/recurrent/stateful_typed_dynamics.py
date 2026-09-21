"""Typed synapse trace dynamics for stateful dendritic recurrent modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    SynapseTypeConfig,
)


@dataclass(frozen=True)
class _StatefulTypedLevelTraces:
    """Integrated typed traces, conductances, and additive drives for one level."""

    trace_E: torch.Tensor
    trace_E_rec: torch.Tensor
    trace_I: torch.Tensor
    trace_I_rec: torch.Tensor
    current_E: torch.Tensor
    current_E_rec: torch.Tensor
    current_I: torch.Tensor
    current_I_rec: torch.Tensor
    drive_E: torch.Tensor
    drive_E_rec: torch.Tensor
    drive_I: torch.Tensor
    drive_I_rec: torch.Tensor


class StatefulTypedDynamicsMixin:
    """Helpers for integrating configured synapse-type dynamics."""

    def _typed_trace_key(self, compartment: str, synapse_type: str) -> str:
        return f"{compartment}:{synapse_type}"

    def _previous_typed_trace(
        self,
        traces: dict[str, list[torch.Tensor]] | None,
        key: str,
        level_idx: int,
        reference: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        if traces is None or key not in traces or level_idx >= len(traces[key]):
            return torch.full_like(reference, float(default))
        return traces[key][level_idx]

    def _decay_for_tau(
        self, tau: float | None, legacy_decay: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if tau is None:
            return legacy_decay.to(device=reference.device, dtype=reference.dtype)
        return torch.exp(
            torch.as_tensor(
                -self.dt / tau, device=reference.device, dtype=reference.dtype
            )
        )

    def _synapse_decay(
        self,
        synapse_type: SynapseTypeConfig,
        legacy_decay: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if synapse_type.tau_decay is None:
            return legacy_decay
        return torch.exp(
            torch.as_tensor(
                -self.dt / synapse_type.tau_decay, device=device, dtype=dtype
            )
        )

    def _apply_synapse_nonlinearity(
        self, raw: torch.Tensor, synapse_type: SynapseTypeConfig
    ) -> torch.Tensor:
        mode = synapse_type.nonlinearity
        if mode == "relu":
            raw = torch.relu(raw)
        elif mode == "sigmoid":
            raw = torch.sigmoid(raw)
        elif mode == "tanh":
            raw = torch.tanh(raw)
        elif mode == "softplus":
            raw = torch.nn.functional.softplus(raw)

        power = float(synapse_type.conductance_power)
        if power != 1.0:
            magnitude = raw.abs()
            if power < 1.0 and torch.is_floating_point(magnitude):
                zero = magnitude == 0
                safe_magnitude = torch.where(
                    zero, torch.ones_like(magnitude), magnitude
                )
                powered = safe_magnitude.pow(power)
                powered = torch.where(zero, torch.zeros_like(powered), powered)
            else:
                powered = magnitude.pow(power)
            raw = raw.sign() * powered
        return raw

    def _apply_short_term_plasticity(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        key: str,
        level_idx: int,
        synapse_type: SynapseTypeConfig,
    ) -> torch.Tensor:
        if not synapse_type.stp_enabled:
            return raw

        u_key = f"{key}:stp_u"
        x_key = f"{key}:stp_x"
        prev_u = self._previous_typed_trace(
            prev_typed_traces,
            u_key,
            level_idx,
            raw,
            default=synapse_type.stp_u0,
        )
        prev_x = self._previous_typed_trace(
            prev_typed_traces, x_key, level_idx, raw, default=1.0
        )
        # Aggregate STP uses the local raw current as a presynaptic-rate proxy.
        activity_source = raw if synapse_type.stp_differentiable_state else raw.detach()
        activity = torch.sigmoid(activity_source)
        dt_over_tau_u = self.dt / float(synapse_type.stp_tau_u)
        dt_over_tau_x = self.dt / float(synapse_type.stp_tau_x)
        next_u = prev_u + dt_over_tau_u * (synapse_type.stp_u0 - prev_u)
        next_u = (
            next_u + float(synapse_type.stp_facilitation) * (1.0 - prev_u) * activity
        )
        next_x = prev_x + dt_over_tau_x * (1.0 - prev_x)
        next_x = (
            next_x - float(synapse_type.stp_depression) * next_u * prev_x * activity
        )
        next_u = next_u.clamp(0.0, 1.0)
        next_x = next_x.clamp(0.0, 1.0)
        new_typed_traces.setdefault(u_key, []).append(next_u)
        new_typed_traces.setdefault(x_key, []).append(next_x)
        return raw * next_u * next_x

    def _apply_voltage_gate(
        self,
        trace: torch.Tensor,
        synapse_type: SynapseTypeConfig,
        voltage_reference: torch.Tensor,
    ) -> torch.Tensor:
        gate_mode = synapse_type.voltage_gate
        if gate_mode == "none":
            return trace
        v_ref = voltage_reference.to(device=trace.device, dtype=trace.dtype)
        v_gate = v_ref * float(synapse_type.voltage_scale_mv) + float(
            synapse_type.mg_v_offset
        )
        if gate_mode == "nmda_magnesium":
            exponent = -v_gate / float(synapse_type.mg_slope)
            gate = 1.0 / (
                1.0
                + float(synapse_type.mg_concentration)
                * torch.exp(exponent)
                / float(synapse_type.mg_scale)
            )
            return trace * gate
        if gate_mode == "sigmoid":
            gate = torch.sigmoid(v_gate / float(synapse_type.mg_slope))
            return trace * gate
        return trace

    def _integrate_single_or_double_exponential(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        key: str,
        level_idx: int,
        synapse_type: SynapseTypeConfig,
        legacy_decay: torch.Tensor,
    ) -> torch.Tensor:
        if synapse_type.tau_rise is None:
            prev_trace = self._previous_typed_trace(
                prev_typed_traces, key, level_idx, raw
            )
            decay = self._synapse_decay(
                synapse_type,
                legacy_decay,
                device=raw.device,
                dtype=raw.dtype,
            )
            return decay * prev_trace + (1 - decay) * (
                raw * float(synapse_type.fraction)
            )

        drive = raw * float(synapse_type.fraction)
        rise_key = f"{key}:rise"
        decay_key = f"{key}:decay"
        prev_rise = self._previous_typed_trace(
            prev_typed_traces, rise_key, level_idx, raw
        )
        prev_decay = self._previous_typed_trace(
            prev_typed_traces, decay_key, level_idx, raw
        )
        rise_decay = self._decay_for_tau(
            synapse_type.tau_rise, legacy_decay, reference=raw
        )
        decay_decay = self._decay_for_tau(
            synapse_type.tau_decay, legacy_decay, reference=raw
        )
        rise_trace = rise_decay * prev_rise + drive
        decay_trace = decay_decay * prev_decay + drive
        new_typed_traces.setdefault(rise_key, []).append(rise_trace)
        new_typed_traces.setdefault(decay_key, []).append(decay_trace)
        return (decay_trace - rise_trace).clamp_min(0.0)

    def _integrate_typed_compartment(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        level_idx: int,
        compartment: str,
        synapse_types: list[SynapseTypeConfig],
        legacy_decay: torch.Tensor,
        voltage_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conductance = torch.zeros_like(raw)
        shunting_current = torch.zeros_like(raw)
        additive_drive = torch.zeros_like(raw)

        for synapse_type in synapse_types:
            key = self._typed_trace_key(compartment, synapse_type.name)
            raw_type = self._apply_synapse_nonlinearity(raw, synapse_type)
            raw_type = self._apply_short_term_plasticity(
                raw=raw_type,
                prev_typed_traces=prev_typed_traces,
                new_typed_traces=new_typed_traces,
                key=key,
                level_idx=level_idx,
                synapse_type=synapse_type,
            )
            trace = self._integrate_single_or_double_exponential(
                raw=raw_type,
                prev_typed_traces=prev_typed_traces,
                new_typed_traces=new_typed_traces,
                key=key,
                level_idx=level_idx,
                synapse_type=synapse_type,
                legacy_decay=legacy_decay,
            )
            trace = self._apply_voltage_gate(trace, synapse_type, voltage_reference)
            new_typed_traces.setdefault(key, []).append(trace)
            conductance = conductance + trace
            reversal = float(synapse_type.reversal_potential)
            shunting_current = shunting_current + trace * reversal
            if synapse_type.polarity == "inhibitory":
                additive_drive = additive_drive + trace * (1.0 - reversal)
            else:
                additive_drive = additive_drive + trace * reversal

        return conductance, shunting_current, additive_drive

    def _integrate_typed_level_traces(
        self,
        *,
        raw_E: torch.Tensor,
        raw_E_rec: torch.Tensor,
        raw_I: torch.Tensor,
        raw_I_rec: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        level_idx: int,
        legacy_decay: torch.Tensor,
        voltage_reference: torch.Tensor,
    ) -> _StatefulTypedLevelTraces:
        """Integrate all typed synaptic compartments for one recurrent level."""
        trace_E, current_E, drive_E = self._integrate_typed_compartment(
            raw=raw_E,
            prev_typed_traces=prev_typed_traces,
            new_typed_traces=new_typed_traces,
            level_idx=level_idx,
            compartment="ff_exc",
            synapse_types=self.synapse_types.excitatory.types,
            legacy_decay=legacy_decay,
            voltage_reference=voltage_reference,
        )
        trace_E_rec, current_E_rec, drive_E_rec = self._integrate_typed_compartment(
            raw=raw_E_rec,
            prev_typed_traces=prev_typed_traces,
            new_typed_traces=new_typed_traces,
            level_idx=level_idx,
            compartment="rec_exc",
            synapse_types=self.synapse_types.excitatory.types,
            legacy_decay=legacy_decay,
            voltage_reference=voltage_reference,
        )
        trace_I, current_I, drive_I = self._integrate_typed_compartment(
            raw=raw_I,
            prev_typed_traces=prev_typed_traces,
            new_typed_traces=new_typed_traces,
            level_idx=level_idx,
            compartment="ff_inh",
            synapse_types=self.synapse_types.inhibitory.types,
            legacy_decay=legacy_decay,
            voltage_reference=voltage_reference,
        )
        trace_I_rec, current_I_rec, drive_I_rec = self._integrate_typed_compartment(
            raw=raw_I_rec,
            prev_typed_traces=prev_typed_traces,
            new_typed_traces=new_typed_traces,
            level_idx=level_idx,
            compartment="rec_inh",
            synapse_types=self.synapse_types.inhibitory.types,
            legacy_decay=legacy_decay,
            voltage_reference=voltage_reference,
        )
        return _StatefulTypedLevelTraces(
            trace_E=trace_E,
            trace_E_rec=trace_E_rec,
            trace_I=trace_I,
            trace_I_rec=trace_I_rec,
            current_E=current_E,
            current_E_rec=current_E_rec,
            current_I=current_I,
            current_I_rec=current_I_rec,
            drive_E=drive_E,
            drive_E_rec=drive_E_rec,
            drive_I=drive_I,
            drive_I_rec=drive_I_rec,
        )

    def _branch_conductance_tensor(
        self,
        layer: DendriticBranchLayer,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not layer.input_branches or not hasattr(layer, "branches_to_output"):
            return torch.zeros_like(reference)
        conductance = layer.branches_to_output.sum_conductances()
        return conductance.to(device=reference.device, dtype=reference.dtype).unsqueeze(
            0
        )

    def _maybe_update_dynamic_grad_scales(
        self,
        layer: DendriticBranchLayer,
        denominator: torch.Tensor | None,
    ) -> None:
        if not self.training:
            return
        if "conductance_dynamic" not in [
            layer.topk_strategy,
            layer.blocklinear_strategy,
        ]:
            return
        layer.compute_grad_scales(
            g_total=denominator if layer.use_shunting else None,
            use_forward_hooks=True,
        )

    def _apply_typed_voltage(
        self,
        layer: DendriticBranchLayer,
        *,
        trace_E: torch.Tensor,
        trace_E_rec: torch.Tensor,
        trace_I: torch.Tensor,
        trace_I_rec: torch.Tensor,
        current_E: torch.Tensor,
        current_E_rec: torch.Tensor,
        current_I: torch.Tensor,
        current_I_rec: torch.Tensor,
        drive_E: torch.Tensor,
        drive_E_rec: torch.Tensor,
        drive_I: torch.Tensor,
        drive_I_rec: torch.Tensor,
        trace_branch: torch.Tensor,
        branch_conductance: torch.Tensor,
    ) -> torch.Tensor:
        denominator = None
        if layer.use_shunting:
            numerator = current_E + current_E_rec + current_I + current_I_rec
            numerator = numerator + trace_branch
            denominator = 1 + trace_E + trace_E_rec + trace_I + trace_I_rec
            denominator = denominator + branch_conductance
            voltage = numerator / (denominator + layer.epsilon)
        else:
            voltage = drive_E + drive_E_rec - drive_I - drive_I_rec + trace_branch
            voltage = layer.normalize_additive_voltage(voltage)
        self._maybe_update_dynamic_grad_scales(layer, denominator)
        return layer.reactivation(voltage)


__all__ = ["StatefulTypedDynamicsMixin"]
