"""Soma-feedback helpers for stateful recurrent dendritic populations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState


@dataclass(frozen=True)
class _StatefulSomaFeedbackBuild:
    """Constructed soma-feedback modules and strength state."""

    strength_index: dict[int, int]
    projections: nn.ModuleList
    strength: Tensor
    strength_is_parameter: bool


@dataclass(frozen=True)
class _StatefulSomaFeedbackEffect:
    """Soma-feedback effect applied to one dendritic level."""

    trace_branch: Tensor
    branch_conductance: Tensor
    gate: Tensor | None


def _build_stateful_soma_feedback(
    *,
    pop_config: PopulationConfig,
    level_dims: list[int],
    n_soma: int,
    level_indices: set[int],
    per_level: bool,
    enabled: bool,
) -> _StatefulSomaFeedbackBuild:
    """Build soma-feedback projections and strength state."""
    strength_index = {
        level_idx: strength_idx
        for strength_idx, level_idx in enumerate(sorted(level_indices))
    }
    projections = nn.ModuleList()
    if enabled:
        # Keep opt-in feedback modules from shifting the initialization stream
        # for the existing dendritic branches and downstream layers.
        with torch.random.fork_rng(devices=[]):
            for level_idx, level_dim in enumerate(level_dims):
                if level_idx in level_indices:
                    projection = nn.Linear(n_soma, level_dim, bias=False)
                    init_std = float(pop_config.soma_feedback_init_std)
                    if init_std == 0.0:
                        nn.init.zeros_(projection.weight)
                    else:
                        nn.init.normal_(projection.weight, mean=0.0, std=init_std)
                else:
                    # Preserve projection indexing by level while avoiding
                    # trainable parameters for feedback-disabled levels.
                    projection = nn.Identity()
                projections.append(projection)

    strength_shape = len(level_indices) if per_level else 1
    strength = torch.full(
        (strength_shape,),
        float(pop_config.soma_feedback_strength),
        dtype=torch.float32,
    )
    return _StatefulSomaFeedbackBuild(
        strength_index=strength_index,
        projections=projections,
        strength=strength,
        strength_is_parameter=enabled
        and bool(pop_config.soma_feedback_learnable_strength),
    )


def _previous_stateful_soma_feedback_source(
    *,
    enabled: bool,
    source_mode: str,
    spiking_soma_enabled: bool,
    state: DendriNetState,
    batch_size: int,
    n_soma: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve the previous soma-feedback source tensor for a step."""
    if not enabled:
        return None
    if source_mode == "voltage":
        source = state.v_soma if state.v_soma is not None else state.soma_output
    elif source_mode == "spike_readout":
        source = (
            state.spike_readout
            if spiking_soma_enabled and state.spike_readout is not None
            else state.soma_output
        )
    else:
        source = state.soma_output
    if source is None:
        return torch.zeros(batch_size, n_soma, device=device, dtype=dtype)
    return source.to(device=device, dtype=dtype)


def _stateful_soma_feedback_for_level(
    *,
    enabled: bool,
    per_level: bool,
    level_indices: set[int],
    strength_index: dict[int, int],
    projections: nn.ModuleList,
    strength: Tensor,
    level_idx: int,
    source: Tensor | None,
    reference: Tensor,
) -> Tensor | None:
    """Project soma feedback into one dendritic level."""
    if not enabled or source is None or level_idx not in level_indices:
        return None
    strength_idx = strength_index[level_idx] if per_level else 0
    level_strength = strength[strength_idx].to(
        device=reference.device,
        dtype=reference.dtype,
    )
    feedback = projections[level_idx](source)
    return feedback.to(device=reference.device, dtype=reference.dtype) * level_strength


def _apply_stateful_soma_feedback_effect(
    *,
    feedback: Tensor | None,
    keep_mask: Tensor | None,
    trace_branch: Tensor,
    branch_conductance: Tensor,
    mode: str,
    reversal: float,
) -> _StatefulSomaFeedbackEffect:
    """Apply soma feedback to branch trace/conductance tensors."""
    if feedback is not None and keep_mask is not None:
        feedback = feedback * keep_mask
    effective_trace_branch = trace_branch
    effective_branch_conductance = branch_conductance
    gate = None
    if feedback is not None:
        if mode == "additive":
            effective_trace_branch = effective_trace_branch + feedback
        elif mode == "shunting":
            feedback_conductance = torch.relu(feedback)
            effective_trace_branch = (
                effective_trace_branch + feedback_conductance * reversal
            )
            effective_branch_conductance = (
                effective_branch_conductance + feedback_conductance
            )
        else:
            gate = torch.tanh(feedback)
    return _StatefulSomaFeedbackEffect(
        trace_branch=effective_trace_branch,
        branch_conductance=effective_branch_conductance,
        gate=gate,
    )


__all__ = [
    "_StatefulSomaFeedbackBuild",
    "_StatefulSomaFeedbackEffect",
    "_apply_stateful_soma_feedback_effect",
    "_build_stateful_soma_feedback",
    "_previous_stateful_soma_feedback_source",
    "_stateful_soma_feedback_for_level",
]
