"""Recurrent architecture factory helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.config.legacy import normalize_transfer_config

_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}

_POPULATION_NETWORK_TYPES: set[str] = {
    "population_network",
}

_HETEROGENEOUS_LEAK_CTRNN_TYPES: set[str] = {
    "heterogeneous_leak_ctrnn",
    "heterogeneous_ctrnn",
}

_LEGENDRE_MEMORY_TYPES: set[str] = {
    "legendre_memory",
}


def _build_baseline_rnn_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build baseline GRU/LSTM/vanilla RNN cores."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        BaselineRNN,
        BaselineRNNConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating baseline RNN (type='{type}')"
        )

    raw_params = _to_plain_mapping(parameters)
    baseline_cfg = raw_params.get("baseline_rnn", raw_params)
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {}
    baseline_cfg = dict(baseline_cfg)
    baseline_cfg["cell_type"] = _BASELINE_RNN_TYPES[type]
    baseline_cfg["input_dim"] = input_dim
    return BaselineRNN(BaselineRNNConfig(**baseline_cfg))


def _build_population_network_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build canonical population-network layers."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        PopulationLayerConfig,
        PopulationNetwork,
        PopulationNetworkConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating population-network architecture (type='{type}')"
        )

    raw_params = _to_plain_mapping(parameters)
    mp_cfg = raw_params.get("population_network") or raw_params
    if not isinstance(mp_cfg, dict):
        mp_cfg = {}
    mp_cfg = dict(mp_cfg)
    if "transfer_params" in mp_cfg:
        mp_cfg["transfer_params"] = normalize_transfer_config(mp_cfg["transfer_params"])
    mp_cfg["input_dim"] = input_dim
    mp_cfg["layers"] = [
        (
            layer
            if isinstance(layer, PopulationLayerConfig)
            else PopulationLayerConfig(**layer)
        )
        for layer in mp_cfg.get("layers", [])
    ]
    return PopulationNetwork(PopulationNetworkConfig(**mp_cfg))


def _build_heterogeneous_leak_ctrnn_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build the sparse heterogeneous-leak point recurrent control."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        HeterogeneousLeakCTRNN,
        HeterogeneousLeakCTRNNConfig,
    )

    if input_dim is None:
        raise ValueError(f"input_dim required when creating CTRNN (type='{type}')")
    raw_params = _to_plain_mapping(parameters)
    ctrnn_cfg = raw_params.get("heterogeneous_leak_ctrnn", raw_params)
    if not isinstance(ctrnn_cfg, dict):
        ctrnn_cfg = {}
    ctrnn_cfg = dict(ctrnn_cfg)
    ctrnn_cfg["input_dim"] = input_dim
    return HeterogeneousLeakCTRNN(HeterogeneousLeakCTRNNConfig(**ctrnn_cfg))


def _build_legendre_memory_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build a fixed Legendre Delay Network memory core."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        FixedLegendreMemory,
        FixedLegendreMemoryConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating fixed Legendre memory (type='{type}')"
        )
    raw_params = _to_plain_mapping(parameters)
    memory_cfg = raw_params.get("legendre_memory", raw_params)
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
    memory_cfg = dict(memory_cfg)
    memory_cfg["input_dim"] = input_dim
    return FixedLegendreMemory(FixedLegendreMemoryConfig(**memory_cfg))


__all__ = [
    "_BASELINE_RNN_TYPES",
    "_HETEROGENEOUS_LEAK_CTRNN_TYPES",
    "_LEGENDRE_MEMORY_TYPES",
    "_POPULATION_NETWORK_TYPES",
    "_build_baseline_rnn_architecture",
    "_build_heterogeneous_leak_ctrnn_architecture",
    "_build_legendre_memory_architecture",
    "_build_population_network_architecture",
]
