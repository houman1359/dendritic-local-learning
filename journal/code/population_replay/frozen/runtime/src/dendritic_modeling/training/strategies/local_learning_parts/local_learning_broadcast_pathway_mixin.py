"""Pathway-role broadcast helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
    compute_pathway_activity,
    compute_pathway_seed_projection,
    compute_role_selectivity,
    effective_topk_weight,
    normalize_role_mass,
    resolve_recent_role_profile,
    resolve_router_output_roles,
    role_mass_from_weight_matrix,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)


class LocalLearningPathwayBroadcastMixin:
    """Pathway-role inference and pathway-vector broadcast helpers."""

    @staticmethod
    def _normalize_role_mass(
        role_mass: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Normalize non-negative role mass into a role profile."""
        return normalize_role_mass(role_mass, eps)

    @staticmethod
    def _compute_role_selectivity(
        role_profile: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Measure pathway selectivity from normalized role profiles."""
        return compute_role_selectivity(role_profile, eps)

    @staticmethod
    def _resolve_recent_role_profile(
        input_dim: int,
        base_input_roles: torch.Tensor | None,
        prior_output_roles: list[torch.Tensor],
    ) -> torch.Tensor | None:
        """Resolve the most plausible role basis for a layer input size."""
        return resolve_recent_role_profile(
            input_dim, base_input_roles, prior_output_roles
        )

    @staticmethod
    def _effective_topk_weight(
        layer: TopKLinear, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return masked effective synaptic weights."""
        return effective_topk_weight(layer, mask)

    @staticmethod
    def _role_mass_from_weight_matrix(
        weight_matrix: torch.Tensor,
        input_roles: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Aggregate pathway-role mass through a synaptic weight matrix."""
        return role_mass_from_weight_matrix(weight_matrix, input_roles)

    def _resolve_router_output_roles(self, model: BaseModel) -> torch.Tensor | None:
        """Return pathway ownership for encoder outputs when available."""
        return resolve_router_output_roles(self.local_cfg, model)

    def _compute_pathway_activity(
        self,
        rec: dict[str, Any],
        stats: _LayerStats,
        v_n: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor | None:
        """Compute a sample-wise local pathway activity gate."""
        return compute_pathway_activity(rec, stats, v_n, eps)

    def _compute_pathway_vector_broadcast(
        self,
        seed_error: torch.Tensor | None,
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        rec: dict[str, Any],
        v_n: torch.Tensor,
    ) -> torch.Tensor:
        """Project soma error into a pathway-aligned apical feedback vector."""
        stats = self._layer_stats.get(id(rec.get("layer")))

        if stats is None or not isinstance(stats.branch_role_profile, torch.Tensor):
            return delta_scalar.expand(-1, v_n.size(1))

        role_profile = stats.branch_role_profile.to(device=v_n.device, dtype=v_n.dtype)

        if role_profile.dim() != 2 or role_profile.size(0) != v_n.size(1):
            return delta_scalar.expand(-1, v_n.size(1))

        pathway_error, base = compute_pathway_seed_projection(
            seed_error=seed_error,
            delta_scalar=delta_scalar,
            role_profile=role_profile,
            out_features=v_n.size(1),
        )

        pathway_activity = self._compute_pathway_activity(rec, stats, v_n)
        gate_strength = float(
            getattr(self.local_cfg, "pathway_activity_gate_strength", 1.0)
        )

        if pathway_activity is not None and gate_strength > 0.0:
            pathway_activity = pathway_activity.to(device=v_n.device, dtype=v_n.dtype)
            pathway_gate = 1.0 + gate_strength * (pathway_activity - 1.0)
            pathway_error = pathway_error * pathway_gate

        projected = pathway_error @ role_profile.t()
        residual = float(
            max(
                0.0,
                min(1.0, getattr(self.local_cfg, "pathway_broadcast_residual", 0.25)),
            )
        )
        return (1.0 - residual) * projected + residual * base


__all__ = ["LocalLearningPathwayBroadcastMixin"]
