"""Dendritic spike and plateau dynamics for recurrent populations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dendritic_modeling.networks.architectures.recurrent.spiking_soma import (
    SurrogateSpike,
)


def resolve_dendritic_spike_levels(enabled: bool, levels, n_levels: int) -> set[int]:
    """Resolve distal-to-soma non-soma level selectors."""
    if not enabled:
        return set()
    dendritic_levels = list(range(max(n_levels - 1, 0)))
    if isinstance(levels, str):
        key = levels.lower()
        if key in {"all", "dendritic", "non_soma", "non-soma"}:
            return set(dendritic_levels)
        if key == "distal":
            return {0} if dendritic_levels else set()
        if key == "proximal":
            return {dendritic_levels[-1]} if dendritic_levels else set()
        if key in {"none", ""}:
            return set()
        raise ValueError(
            "dendritic_spike_levels must be one of 'non_soma', 'all', "
            "'distal', 'proximal', 'none', or a list of non-soma level indices, "
            f"got {levels!r}"
        )

    resolved = set()
    for value in levels:
        level_idx = int(value)
        if level_idx < 0 or level_idx >= n_levels - 1:
            raise ValueError(
                "dendritic_spike_levels entries are distal-to-soma non-soma "
                f"indices in [0, {n_levels - 2}], got {level_idx}"
            )
        resolved.add(level_idx)
    return resolved


def previous_dendritic_spike_state(
    values: list[torch.Tensor] | None,
    level_idx: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return the previous level state or zeros matching ``reference``."""
    if values is None or level_idx >= len(values):
        return torch.zeros_like(reference)
    return values[level_idx]


@dataclass(frozen=True)
class DendriticSpikeDynamics:
    """Opt-in local branch spike/plateau dynamics."""

    enabled: bool
    level_indices: set[int]
    mode: str = "plateau"
    threshold: float = 1.0
    plateau_amplitude: float = 1.0
    plateau_tau: float = 10.0
    refractory_steps: int = 0
    surrogate_beta: float = 10.0
    propagation: str = "additive"
    dt: float = 1.0

    def apply(
        self,
        *,
        level_idx: int,
        branch_voltage: torch.Tensor,
        prev_plateau: torch.Tensor,
        prev_refractory: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one local dendritic spike update for a branch level."""
        if keep_mask is not None:
            prev_plateau = prev_plateau * keep_mask
            prev_refractory = prev_refractory * keep_mask

        if not self.enabled or level_idx not in self.level_indices:
            zeros = torch.zeros_like(branch_voltage)
            return branch_voltage, zeros, zeros, zeros

        active = (prev_refractory <= 0).to(dtype=branch_voltage.dtype)
        event = SurrogateSpike.apply(
            branch_voltage - self.threshold,
            self.surrogate_beta,
        )
        event = event * active

        if self.mode == "plateau":
            plateau_decay = torch.as_tensor(
                math.exp(-self.dt / self.plateau_tau),
                device=branch_voltage.device,
                dtype=branch_voltage.dtype,
            )
            plateau = plateau_decay * prev_plateau + (1 - plateau_decay) * event
        else:
            plateau = event

        next_refractory = torch.clamp(prev_refractory - 1, min=0)
        if self.refractory_steps > 0:
            next_refractory = torch.where(
                event.detach() > 0,
                torch.full_like(next_refractory, float(self.refractory_steps)),
                next_refractory,
            )

        if self.propagation == "additive":
            modified_voltage = branch_voltage + self.plateau_amplitude * plateau
        elif self.propagation == "replace":
            modified_voltage = (
                branch_voltage * (1 - plateau) + self.plateau_amplitude * plateau
            )
        else:
            modified_voltage = branch_voltage * (1 + self.plateau_amplitude * plateau)

        if keep_mask is not None:
            modified_voltage = modified_voltage * keep_mask
            plateau = plateau * keep_mask
            next_refractory = next_refractory * keep_mask
            event = event * keep_mask

        return modified_voltage, plateau, next_refractory, event


__all__ = [
    "DendriticSpikeDynamics",
    "previous_dendritic_spike_state",
    "resolve_dendritic_spike_levels",
]
