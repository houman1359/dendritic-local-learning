"""
Unified multi-layer E-I network with optional recurrent unrolling.
"""

import copy
from typing import Optional, Union

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    EILayerConfig,
    EINetworkConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_layer import EILayer
from dendritic_modeling.networks.architectures.recurrent.ei_state import EIState
from dendritic_modeling.networks.base import BaseNetwork


class EINetwork(BaseNetwork):
    """Stack of unified E-I layers.

    Feedforward mode:
      input [B, D] -> output [B, n_exc_last]
    Recurrent mode:
      input [B, T, D] -> output based on output_mode
    """

    def __init__(self, config: EINetworkConfig):
        super().__init__()
        self.config = config
        self.output_mode = config.output_mode
        self._store_routing = config.store_routing
        self._routing_info: list[dict] = []
        self._use_transfer = bool(config.use_transfer)
        self.transfer_fn: TransferLayer | None = None

        transfer_exc_dim = config.input_dim
        transfer_inh_dim: int | None = None
        if self._use_transfer:
            self.transfer_fn = TransferLayer(
                input_dim=config.input_dim, transfer_params=config.transfer_params
            )
            transfer_exc_dim = self.transfer_fn.excitatory_dim
            transfer_inh_dim = self.transfer_fn.inhibitory_dim

        proj_layers = []
        prev_dim = transfer_exc_dim
        for hidden_dim in config.input_projection_dims:
            proj_layers.append(nn.Linear(prev_dim, hidden_dim))
            proj_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )
        self.proj_output_dim = prev_dim

        self.layers = nn.ModuleList()
        prev_exc_dim = self.proj_output_dim
        prev_inh_dim = transfer_inh_dim if self._use_transfer else None
        self._is_recurrent = False
        for layer_idx, base_cfg in enumerate(config.layers):
            layer_cfg: EILayerConfig = copy.deepcopy(base_cfg)
            for population_name in ("excitatory", "inhibitory"):
                population = getattr(layer_cfg, population_name, None)
                if (
                    population is not None
                    and population.initialization_seed is not None
                    and not population.initialization_namespace
                ):
                    population.initialization_namespace = (
                        f"unified.layer.{layer_idx}.{population_name}"
                    )
            layer_cfg.excitatory_input_dim = prev_exc_dim
            if self._use_transfer and layer_cfg.direct_ff_inhibitory_to_excitatory:
                layer_cfg.inhibitory_input_dim = transfer_inh_dim
            else:
                layer_cfg.inhibitory_input_dim = prev_inh_dim
            layer = EILayer(layer_cfg)
            self.layers.append(layer)

            self._is_recurrent = self._is_recurrent or layer_cfg.recurrent
            prev_exc_dim = layer_cfg.excitatory.n_neurons
            prev_inh_dim = (
                layer_cfg.inhibitory.n_neurons
                if layer_cfg.inhibitory is not None
                else None
            )

        self.output_dim = prev_exc_dim

        # Propagate store_routing to all layers
        if self._store_routing:
            for layer in self.layers:
                layer.store_routing = True

    @property
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @property
    def store_routing(self) -> bool:
        return self._store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._store_routing = value
        for layer in self.layers:
            layer.store_routing = value

    def set_layer_population_silencing(
        self,
        layer_idx: int,
        population: str,
        mask: Optional[torch.Tensor],
    ) -> None:
        self.layers[layer_idx].set_population_level_silencing(population, mask)

    def clear_all_population_silencing(self) -> None:
        for layer in self.layers:
            layer.clear_population_level_silencing()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> list[EIState]:
        return [
            layer.init_state(batch_size=batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_mode == "last":
            if seq_lengths is not None:
                B, T, _ = outputs.shape
                idx = (seq_lengths - 1).long().clamp(0, T - 1)
                return outputs[torch.arange(B, device=outputs.device), idx]
            return outputs[:, -1, :]
        if self.output_mode == "mean":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                masked = outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
                denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=outputs.dtype)
                return masked.sum(dim=1) / denom
            return outputs.mean(dim=1)
        if self.output_mode == "all":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                return outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
            return outputs
        raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def _select_layer_inhibitory_input(
        self,
        layer: EILayer,
        ff_inhibitory: Optional[torch.Tensor],
        transfer_inhibitory: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Select inhibitory input for one layer.

        Priority:
        1) ``direct_ff_inhibitory_to_excitatory`` forces transfer inhibitory
           (used by ``input_mode=1`` with ``inhibitory_mode`` "first"/"all").
        2) Propagated inhibitory from the previous layer (``ff_inhibitory``).
        3) If no upstream inhibitory and the layer has an I population,
           fall back to transfer inhibitory (``input_mode=0`` first-layer case).
        """
        if (
            layer.config.direct_ff_inhibitory_to_excitatory
            and transfer_inhibitory is not None
        ):
            return transfer_inhibitory
        if ff_inhibitory is not None:
            return ff_inhibitory
        # First layer with I population but no upstream: use transfer stream
        if layer.i_population is not None and transfer_inhibitory is not None:
            return transfer_inhibitory
        return None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[list[EIState]] = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[EIState]]]:
        """Forward pass through the network.

        Args:
            x: Input tensor.
                Feedforward: [batch, input_dim]
                Recurrent: [batch, seq_len, input_dim]
            hidden: Optional list of EIState (one per layer), for recurrent mode.
            return_hidden: If True, return (output, hidden_state) tuple.
            seq_lengths: Optional per-sample sequence lengths [batch].
                When provided and output_mode="last", reads output at position
                seq_lengths[i]-1 instead of the last timestep.

        Returns:
            output: Network output tensor.
            hidden: Final hidden state (only when return_hidden=True).
        """
        if self.is_recurrent:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, hidden = self._forward_recurrent(
                x, hidden=hidden, seq_lengths=seq_lengths
            )
        else:
            out = self._forward_feedforward(x)

        if return_hidden:
            return out, hidden
        return out

    def _forward_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        transfer_inhibitory: torch.Tensor | None = None
        if self.transfer_fn is not None:
            h, transfer_inhibitory = self.transfer_fn(x)
            h = self.input_projection(h)
        else:
            h = self.input_projection(x)

        ff_I: Optional[torch.Tensor] = None
        for layer in self.layers:
            layer_inhibitory = self._select_layer_inhibitory_input(
                layer=layer,
                ff_inhibitory=ff_I,
                transfer_inhibitory=transfer_inhibitory,
            )
            h, ff_I, _ = layer(h, layer_inhibitory)
        return h

    def _forward_recurrent(
        self,
        x: torch.Tensor,
        hidden: Optional[list[EIState]] = None,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[EIState]]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        if hidden is None:
            hidden = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        outputs = []
        self._routing_info = []
        # Reuse deterministic TopK selection across timesteps of this unroll.
        # Always release it, including on shape/config errors.
        apply_recurrent_weight_cache(self, True)
        try:
            for t in range(seq_len):
                transfer_inhibitory: torch.Tensor | None = None
                if self.transfer_fn is not None:
                    h, transfer_inhibitory = self.transfer_fn(x[:, t, :])
                    h = self.input_projection(h)
                else:
                    h = self.input_projection(x[:, t, :])
                ff_I = None
                for layer_idx, layer in enumerate(self.layers):
                    layer_inhibitory = self._select_layer_inhibitory_input(
                        layer=layer,
                        ff_inhibitory=ff_I,
                        transfer_inhibitory=transfer_inhibitory,
                    )
                    h, ff_I, hidden[layer_idx] = layer(
                        h, layer_inhibitory, hidden[layer_idx]
                    )
                outputs.append(h)

                if self._store_routing:
                    timestep_info = {}
                    for layer_idx, layer in enumerate(self.layers):
                        timestep_info[f"layer_{layer_idx}"] = layer._last_routing_info
                    self._routing_info.append(timestep_info)
        finally:
            apply_recurrent_weight_cache(self, False)
        stacked = torch.stack(outputs, dim=1)
        return self._reduce_outputs(stacked, seq_lengths=seq_lengths), hidden
