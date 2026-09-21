"""
Unified E-I layer: feedforward special case and recurrent mode in one class.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.recurrent.ei_config import EILayerConfig
from dendritic_modeling.networks.architectures.recurrent.ei_state import EIState
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)


class EILayer(nn.Module):
    """Unified E-I layer with optional recurrence."""

    def __init__(self, config: EILayerConfig):
        super().__init__()
        for population_name in ("excitatory", "inhibitory"):
            population = getattr(config, population_name, None)
            if (
                population is not None
                and population.initialization_seed is not None
                and not population.initialization_namespace
            ):
                population.initialization_namespace = (
                    f"ei.layer.{population.structured_layer_idx}.{population_name}"
                )
        self.config = config
        self.recurrent = config.recurrent
        self._last_routing_info: dict = {}

        self.n_exc = config.excitatory.n_neurons
        self.n_inh = config.inhibitory.n_neurons if config.inhibitory is not None else 0
        self.direct_ff_inhibitory_to_excitatory = bool(
            config.direct_ff_inhibitory_to_excitatory
        )

        rec_e_dim = self.n_exc if self.recurrent else None
        rec_i_dim = self.n_inh if self.recurrent and self.n_inh > 0 else None

        self.i_population: Optional[StatefulDendriNet]
        if config.inhibitory is not None and self.n_inh > 0:
            self.i_population = StatefulDendriNet(
                pop_config=config.inhibitory,
                excitatory_input_dim=config.excitatory_input_dim,
                inhibitory_input_dim=config.inhibitory_input_dim,
                recurrent_excitatory_input_dim=rec_e_dim,
                recurrent_inhibitory_input_dim=rec_i_dim,
                recurrent_excitatory_is_self_population=False,
                recurrent_inhibitory_is_self_population=True,
                dt=config.dt,
            )
        else:
            self.i_population = None

        # E receives FF inhibitory from same-layer I population (if present).
        e_ff_inhibitory_dim = (
            self.n_inh
            if self.n_inh > 0
            else (
                config.inhibitory_input_dim
                if self.direct_ff_inhibitory_to_excitatory
                else None
            )
        )
        self.e_population = StatefulDendriNet(
            pop_config=config.excitatory,
            excitatory_input_dim=config.excitatory_input_dim,
            inhibitory_input_dim=e_ff_inhibitory_dim,
            recurrent_excitatory_input_dim=rec_e_dim,
            recurrent_inhibitory_input_dim=rec_i_dim,
            recurrent_excitatory_is_self_population=True,
            recurrent_inhibitory_is_self_population=False,
            dt=config.dt,
        )

        self.output_dim = self.n_exc

    @property
    def store_routing(self) -> bool:
        return self.e_population.store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self.e_population.store_routing = value
        if self.i_population is not None:
            self.i_population.store_routing = value

    def set_population_level_silencing(
        self,
        population: str,
        mask: Optional[torch.Tensor],
    ) -> None:
        if population == "excitatory":
            self.e_population.set_level_silencing(mask)
            return
        if population == "inhibitory":
            if self.i_population is None:
                raise ValueError(
                    "Cannot silence inhibitory levels: no inhibitory population"
                )
            self.i_population.set_level_silencing(mask)
            return
        if population == "both":
            self.e_population.set_level_silencing(mask)
            if self.i_population is not None:
                self.i_population.set_level_silencing(mask)
            return
        raise ValueError(f"Unknown population '{population}'")

    def clear_population_level_silencing(self) -> None:
        self.e_population.clear_level_silencing()
        if self.i_population is not None:
            self.i_population.clear_level_silencing()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> EIState:
        e_state = self.e_population.init_state(
            batch_size=batch_size, device=device, dtype=dtype
        )
        i_state = (
            self.i_population.init_state(
                batch_size=batch_size, device=device, dtype=dtype
            )
            if self.i_population is not None
            else None
        )
        s_E = torch.zeros(batch_size, self.n_exc, device=device, dtype=dtype)
        h_I = (
            torch.zeros(batch_size, self.n_inh, device=device, dtype=dtype)
            if self.n_inh > 0
            else None
        )
        return EIState(e_state=e_state, i_state=i_state, s_E=s_E, h_I=h_I)

    def forward(
        self,
        x_ff: torch.Tensor,
        ff_inhibitory: Optional[torch.Tensor] = None,
        state: Optional[EIState] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[EIState]]:
        """Forward pass through the E-I layer.

        Args:
            x_ff: FF excitatory input from upstream [batch, excitatory_input_dim].
            ff_inhibitory: FF inhibitory input from upstream (for I pop) [batch, inhibitory_input_dim].
            state: Recurrent state (only used when recurrent=True).

        Returns:
            s_E: Excitatory output [batch, n_exc].
            h_I: Inhibitory output [batch, n_inh] or None.
            new_state: Updated EIState (None when recurrent=False).
        """
        if not self.recurrent:
            h_I = (
                self.i_population(x_ff, ff_inhibitory)
                if self.i_population is not None
                else None
            )
            e_ff_inhibitory = (
                h_I
                if h_I is not None
                else (
                    ff_inhibitory if self.direct_ff_inhibitory_to_excitatory else None
                )
            )
            s_E = self.e_population(x_ff, e_ff_inhibitory)
            return s_E, h_I, None

        batch_size = x_ff.shape[0]
        device = x_ff.device
        dtype = x_ff.dtype

        if state is None:
            state = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        s_E_prev = state.s_E
        h_I_prev = state.h_I

        # Update I first (faster dynamics).
        if self.i_population is not None:
            h_I, i_state = self.i_population.step(
                x=x_ff,
                inhibitory_input=ff_inhibitory,
                recurrent_input=s_E_prev,
                rec_inhibitory_input=h_I_prev,
                state=state.i_state,
            )
        else:
            h_I, i_state = None, None

        # E uses current same-layer I as FF inhibitory and previous I for REC inhibitory.
        e_ff_inhibitory = (
            h_I
            if h_I is not None
            else (ff_inhibitory if self.direct_ff_inhibitory_to_excitatory else None)
        )
        s_E, e_state = self.e_population.step(
            x=x_ff,
            inhibitory_input=e_ff_inhibitory,
            recurrent_input=s_E_prev,
            rec_inhibitory_input=h_I_prev,
            state=state.e_state,
        )

        new_state = EIState(e_state=e_state, i_state=i_state, s_E=s_E, h_I=h_I)

        if self.e_population.store_routing:
            self._last_routing_info = {
                "excitatory": self.e_population._last_routing_info,
            }
            if self.i_population is not None:
                self._last_routing_info["inhibitory"] = (
                    self.i_population._last_routing_info
                )

        return s_E, h_I, new_state
