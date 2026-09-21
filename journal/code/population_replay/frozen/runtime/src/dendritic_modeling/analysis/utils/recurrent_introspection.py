"""Duck-typed introspection helpers for recurrent dendritic analysis.

These utilities keep the analysis stack compatible with both the legacy
``EINetwork``/``EILayer`` layout and the canonical population-network backend.
They intentionally avoid importing concrete architecture classes so analyzers
can use them on lightweight fakes in tests and on checkpoint-loaded modules.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PopulationRecord:
    """One recurrent dendritic population discovered in a model core."""

    key: str
    population: torch.nn.Module
    polarity: str
    n_neurons: int
    layer_index: int | None = None
    population_name: str | None = None


def _as_bool(value: Any) -> bool:
    return bool(value() if callable(value) else value)


def _n_soma(population: Any) -> int:
    for attr in ("n_soma", "_n_soma", "output_dim"):
        value = getattr(population, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


def _population_network_polarities(layer: Any) -> dict[str, str]:
    polarities: dict[str, str] = {}
    for population in getattr(layer, "population_definitions", []) or []:
        name = getattr(population, "name", None)
        polarity = getattr(population, "polarity", None)
        if name is not None and polarity is not None:
            polarities[str(name)] = str(polarity).lower()
    return polarities


def iter_recurrent_populations(core: Any) -> Iterator[PopulationRecord]:
    """Yield recurrent dendritic populations from known core layouts.

    Supported layouts:
    - ``EINetwork.layers[i].e_population`` / ``i_population``
    - ``PopulationNetwork.layers[i].populations[name]``

    Unknown cores simply yield nothing.
    """
    layers = getattr(core, "layers", None)
    if layers is not None:
        for idx, layer in enumerate(layers):
            named_populations = getattr(layer, "populations", None)
            if named_populations is not None:
                layer_name = str(getattr(getattr(layer, "config", None), "name", idx))
                polarities = _population_network_polarities(layer)
                for name, population in named_populations.items():
                    name_str = str(name)
                    polarity = polarities.get(name_str, "excitatory")
                    yield PopulationRecord(
                        key=f"population_network/{layer_name}.{name_str}",
                        population=population,
                        polarity=polarity,
                        n_neurons=_n_soma(population),
                        layer_index=idx,
                        population_name=name_str,
                    )
                continue

            e_pop = getattr(layer, "e_population", None)
            if e_pop is not None:
                yield PopulationRecord(
                    key=f"layer_{idx}/excitatory",
                    population=e_pop,
                    polarity="excitatory",
                    n_neurons=_n_soma(e_pop),
                    layer_index=idx,
                    population_name="excitatory",
                )
            i_pop = getattr(layer, "i_population", None)
            if i_pop is not None:
                yield PopulationRecord(
                    key=f"layer_{idx}/inhibitory",
                    population=i_pop,
                    polarity="inhibitory",
                    n_neurons=_n_soma(i_pop),
                    layer_index=idx,
                    population_name="inhibitory",
                )


def recurrent_population_summary(core: Any) -> dict[str, dict[str, Any]]:
    """Summarize discovered recurrent populations for diagnostics/docs tests."""
    out: dict[str, dict[str, Any]] = {}
    for record in iter_recurrent_populations(core):
        population = record.population
        synapse_types = getattr(population, "synapse_types", None)
        spiking_soma = getattr(population, "spiking_soma", None)
        out[record.key] = {
            "polarity": record.polarity,
            "n_neurons": record.n_neurons,
            "n_levels": len(getattr(population, "branch_layers", []) or []),
            "synapse_types_enabled": bool(getattr(synapse_types, "enabled", False)),
            "spiking_enabled": spiking_soma is not None,
            "spike_readout": getattr(spiking_soma, "output_mode", None),
        }
    return out


def iter_routing_entries(
    timestep_info: dict[str, Any],
) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
    """Yield normalized routing entries from legacy and named-population cores."""
    for outer_key, value in timestep_info.items():
        if not isinstance(value, dict):
            continue
        if "level_contributions" in value or "ff_rec_ratios" in value:
            yield str(outer_key), value
            continue
        for pop_key, pop_info in value.items():
            if not isinstance(pop_info, dict):
                continue
            if "level_contributions" in pop_info or "ff_rec_ratios" in pop_info:
                yield f"{outer_key}/{pop_key}", pop_info


def is_recurrent_core(core: Any) -> bool:
    """Return True for recurrent cores while handling method/property variants."""
    return _as_bool(getattr(core, "is_recurrent", False))


__all__ = [
    "PopulationRecord",
    "is_recurrent_core",
    "iter_recurrent_populations",
    "iter_routing_entries",
    "recurrent_population_summary",
]
