"""
State containers for unified E-I recurrent layers.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class DendriNetState:
    """Per-level traces for one population."""

    trace_E: list[Tensor]
    trace_E_rec: list[Tensor]
    trace_I: list[Tensor]
    trace_I_rec: list[Tensor]
    trace_branch: list[Tensor]
    branch_voltage: Optional[list[Tensor]] = None
    typed_traces: Optional[dict[str, list[Tensor]]] = None
    soma_output: Optional[Tensor] = None
    v_soma: Optional[Tensor] = None
    refractory_counter: Optional[Tensor] = None
    spike_readout: Optional[Tensor] = None
    dendritic_spike_plateau: Optional[list[Tensor]] = None
    dendritic_spike_refractory: Optional[list[Tensor]] = None
    dendritic_spike_events: Optional[list[Tensor]] = None

    @staticmethod
    def zeros(
        level_dims: list[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        soma_dim: Optional[int] = None,
        track_branch_voltage: bool = False,
        track_dendritic_spikes: bool = False,
        track_soma_output: bool = False,
    ) -> "DendriNetState":
        def _mk():
            return [
                torch.zeros(batch_size, dim, device=device, dtype=dtype)
                for dim in level_dims
            ]

        soma_state = (
            torch.zeros(batch_size, soma_dim, device=device, dtype=dtype)
            if soma_dim is not None
            else None
        )
        return DendriNetState(
            trace_E=_mk(),
            trace_E_rec=_mk(),
            trace_I=_mk(),
            trace_I_rec=_mk(),
            trace_branch=_mk(),
            branch_voltage=_mk() if track_branch_voltage else None,
            typed_traces=None,
            soma_output=(
                soma_state.clone()
                if track_soma_output and soma_state is not None
                else None
            ),
            v_soma=soma_state,
            refractory_counter=(
                torch.zeros(batch_size, soma_dim, device=device, dtype=dtype)
                if soma_dim is not None
                else None
            ),
            spike_readout=(
                torch.zeros(batch_size, soma_dim, device=device, dtype=dtype)
                if soma_dim is not None
                else None
            ),
            dendritic_spike_plateau=_mk() if track_dendritic_spikes else None,
            dendritic_spike_refractory=_mk() if track_dendritic_spikes else None,
            dendritic_spike_events=_mk() if track_dendritic_spikes else None,
        )

    def detach(self) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.detach() for t in self.trace_E],
            trace_E_rec=[t.detach() for t in self.trace_E_rec],
            trace_I=[t.detach() for t in self.trace_I],
            trace_I_rec=[t.detach() for t in self.trace_I_rec],
            trace_branch=[t.detach() for t in self.trace_branch],
            branch_voltage=(
                [t.detach() for t in self.branch_voltage]
                if self.branch_voltage is not None
                else None
            ),
            typed_traces=(
                {
                    key: [trace.detach() for trace in traces]
                    for key, traces in self.typed_traces.items()
                }
                if self.typed_traces is not None
                else None
            ),
            soma_output=(
                self.soma_output.detach() if self.soma_output is not None else None
            ),
            v_soma=self.v_soma.detach() if self.v_soma is not None else None,
            refractory_counter=(
                self.refractory_counter.detach()
                if self.refractory_counter is not None
                else None
            ),
            spike_readout=(
                self.spike_readout.detach() if self.spike_readout is not None else None
            ),
            dendritic_spike_plateau=(
                [t.detach() for t in self.dendritic_spike_plateau]
                if self.dendritic_spike_plateau is not None
                else None
            ),
            dendritic_spike_refractory=(
                [t.detach() for t in self.dendritic_spike_refractory]
                if self.dendritic_spike_refractory is not None
                else None
            ),
            dendritic_spike_events=(
                [t.detach() for t in self.dendritic_spike_events]
                if self.dendritic_spike_events is not None
                else None
            ),
        )

    def clone(self) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.clone() for t in self.trace_E],
            trace_E_rec=[t.clone() for t in self.trace_E_rec],
            trace_I=[t.clone() for t in self.trace_I],
            trace_I_rec=[t.clone() for t in self.trace_I_rec],
            trace_branch=[t.clone() for t in self.trace_branch],
            branch_voltage=(
                [t.clone() for t in self.branch_voltage]
                if self.branch_voltage is not None
                else None
            ),
            typed_traces=(
                {
                    key: [trace.clone() for trace in traces]
                    for key, traces in self.typed_traces.items()
                }
                if self.typed_traces is not None
                else None
            ),
            soma_output=(
                self.soma_output.clone() if self.soma_output is not None else None
            ),
            v_soma=self.v_soma.clone() if self.v_soma is not None else None,
            refractory_counter=(
                self.refractory_counter.clone()
                if self.refractory_counter is not None
                else None
            ),
            spike_readout=(
                self.spike_readout.clone() if self.spike_readout is not None else None
            ),
            dendritic_spike_plateau=(
                [t.clone() for t in self.dendritic_spike_plateau]
                if self.dendritic_spike_plateau is not None
                else None
            ),
            dendritic_spike_refractory=(
                [t.clone() for t in self.dendritic_spike_refractory]
                if self.dendritic_spike_refractory is not None
                else None
            ),
            dendritic_spike_events=(
                [t.clone() for t in self.dendritic_spike_events]
                if self.dendritic_spike_events is not None
                else None
            ),
        )

    def to(self, device: torch.device) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.to(device) for t in self.trace_E],
            trace_E_rec=[t.to(device) for t in self.trace_E_rec],
            trace_I=[t.to(device) for t in self.trace_I],
            trace_I_rec=[t.to(device) for t in self.trace_I_rec],
            trace_branch=[t.to(device) for t in self.trace_branch],
            branch_voltage=(
                [t.to(device) for t in self.branch_voltage]
                if self.branch_voltage is not None
                else None
            ),
            typed_traces=(
                {
                    key: [trace.to(device) for trace in traces]
                    for key, traces in self.typed_traces.items()
                }
                if self.typed_traces is not None
                else None
            ),
            soma_output=(
                self.soma_output.to(device) if self.soma_output is not None else None
            ),
            v_soma=self.v_soma.to(device) if self.v_soma is not None else None,
            refractory_counter=(
                self.refractory_counter.to(device)
                if self.refractory_counter is not None
                else None
            ),
            spike_readout=(
                self.spike_readout.to(device)
                if self.spike_readout is not None
                else None
            ),
            dendritic_spike_plateau=(
                [t.to(device) for t in self.dendritic_spike_plateau]
                if self.dendritic_spike_plateau is not None
                else None
            ),
            dendritic_spike_refractory=(
                [t.to(device) for t in self.dendritic_spike_refractory]
                if self.dendritic_spike_refractory is not None
                else None
            ),
            dendritic_spike_events=(
                [t.to(device) for t in self.dendritic_spike_events]
                if self.dendritic_spike_events is not None
                else None
            ),
        )


@dataclass
class EIState:
    """State for one unified E-I layer."""

    e_state: DendriNetState
    i_state: Optional[DendriNetState]
    s_E: Tensor
    h_I: Optional[Tensor]

    def detach(self) -> "EIState":
        return EIState(
            e_state=self.e_state.detach(),
            i_state=self.i_state.detach() if self.i_state is not None else None,
            s_E=self.s_E.detach(),
            h_I=self.h_I.detach() if self.h_I is not None else None,
        )

    def clone(self) -> "EIState":
        return EIState(
            e_state=self.e_state.clone(),
            i_state=self.i_state.clone() if self.i_state is not None else None,
            s_E=self.s_E.clone(),
            h_I=self.h_I.clone() if self.h_I is not None else None,
        )

    def to(self, device: torch.device) -> "EIState":
        return EIState(
            e_state=self.e_state.to(device),
            i_state=self.i_state.to(device) if self.i_state is not None else None,
            s_E=self.s_E.to(device),
            h_I=self.h_I.to(device) if self.h_I is not None else None,
        )
