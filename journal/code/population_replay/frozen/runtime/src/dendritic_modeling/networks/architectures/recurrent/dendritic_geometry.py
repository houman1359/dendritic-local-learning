"""Geometry helpers for recurrent dendritic population modules."""

from __future__ import annotations

import torch


def compute_level_sizes(n_soma: int, branch_factors: list[int]) -> list[int]:
    """Compute output dimensions per level in distal-to-soma order."""
    sizes = [n_soma]
    for bf in branch_factors:
        sizes.append(sizes[-1] * bf)
    sizes.reverse()
    return sizes


def compute_input_branch_factors(branch_factors: list[int]) -> list[int | None]:
    """Compute child aggregation factors per level."""
    rev_bf = list(reversed(branch_factors))
    return [None, *rev_bf]


def compute_output_owner_index_per_level(
    level_dims: list[int], n_soma: int
) -> list[torch.Tensor]:
    """Map each branch output at a level to its parent soma-neuron index."""
    owner_indices = []
    for level_dim in level_dims:
        if level_dim % n_soma != 0:
            raise ValueError(
                f"Level dimension {level_dim} must be divisible by soma count {n_soma}"
            )
        repeats_per_soma = level_dim // n_soma
        owner_indices.append(
            torch.arange(level_dim, dtype=torch.long) // repeats_per_soma
        )
    return owner_indices


def sum_level_values_by_output_owner(
    values: torch.Tensor,
    n_soma: int,
) -> torch.Tensor:
    """Sum contiguous level coordinates belonging to the same soma owner.

    Recurrent dendritic levels are laid out owner-major: every soma owns one
    contiguous block of coordinates at every level.  This reduction exposes an
    independently evolving level-width state at soma width without adding a
    trainable projection or a data-dependent routing choice.
    """

    if values.ndim < 1:
        raise ValueError("level values must include a feature dimension")
    if isinstance(n_soma, bool) or not isinstance(n_soma, int) or n_soma <= 0:
        raise ValueError("n_soma must be a positive integer")
    level_dim = int(values.shape[-1])
    if level_dim % n_soma != 0:
        raise ValueError(
            f"level width {level_dim} must be divisible by soma count {n_soma}"
        )
    coordinates_per_owner = level_dim // n_soma
    return values.reshape(
        *values.shape[:-1],
        n_soma,
        coordinates_per_owner,
    ).sum(dim=-1)


def resolve_synapses_by_level(
    *,
    per_level: list[int],
    fallback: int,
    n_levels: int,
    field_name: str,
) -> list[int]:
    """Resolve either a scalar fallback or an explicit per-level synapse list."""
    if not per_level:
        return [int(fallback)] * n_levels
    if len(per_level) != n_levels:
        raise ValueError(
            f"{field_name} length must match number of dendritic levels "
            f"({n_levels}), got {len(per_level)}"
        )
    return [int(v) for v in per_level]


__all__ = [
    "compute_input_branch_factors",
    "compute_level_sizes",
    "compute_output_owner_index_per_level",
    "resolve_synapses_by_level",
    "sum_level_values_by_output_owner",
]
