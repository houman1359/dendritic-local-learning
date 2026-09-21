"""Helpers for analysis tools that operate on dendritic E/I cores."""

from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork


def has_einet_core(model) -> bool:
    """Return whether a model exposes a dendritic E/I analysis core."""
    core = getattr(model, "core_network", None)
    if isinstance(core, ExcitationInhibitionNetwork):
        return True

    wrapped = getattr(core, "einet", None)
    if isinstance(wrapped, ExcitationInhibitionNetwork):
        return True

    for candidate in (core, wrapped):
        if candidate is None:
            continue
        if any(
            hasattr(record.population, "branch_layers")
            for record in iter_recurrent_populations(candidate)
        ):
            return True

    return False
