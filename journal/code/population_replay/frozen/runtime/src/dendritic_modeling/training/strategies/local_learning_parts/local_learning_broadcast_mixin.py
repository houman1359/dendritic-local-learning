"""Broadcast-error helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
    apply_broadcast_bandwidth,
    compute_layer_total_conductance,
    compute_local_mismatch_broadcast,
    expand_soma_error_to_compartments,
    reduce_error_to_scalar,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_low_rank_mixin import (
    LocalLearningLowRankBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_pathway_mixin import (
    LocalLearningPathwayBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_transport_mixin import (
    LocalLearningPathTransportBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _BroadcastState,
)


class LocalLearningBroadcastMixin(
    LocalLearningPathTransportBroadcastMixin,
    LocalLearningPathwayBroadcastMixin,
    LocalLearningLowRankBroadcastMixin,
):
    @staticmethod
    def _reduce_error_to_scalar(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Reduce output error to a per-sample scalar for broadcast modes.

        For CE-like errors, per-sample mean can be near zero due to probability
        simplex constraints. In that case we fall back to the max-magnitude
        component to keep a meaningful signed modulatory signal.
        """
        return reduce_error_to_scalar(delta, eps=eps)

    @staticmethod
    def _compute_local_mismatch_broadcast(
        delta_scalar: torch.Tensor,
        v_n: torch.Tensor,
        parent: torch.Tensor | None,
        residual_fraction: float = 0.2,
        clip_value: float = 3.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute a stabilized local-mismatch broadcast signal.

        The raw local mismatch signal can collapse when parent and soma voltages are
        close or poorly scaled. We stabilize it by:
        1) centering mismatch across batch,
        2) normalizing mismatch per sample by RMS magnitude,
        3) clipping outliers,
        4) blending with a residual scalar broadcast path.
        """
        return compute_local_mismatch_broadcast(
            delta_scalar=delta_scalar,
            v_n=v_n,
            parent=parent,
            residual_fraction=residual_fraction,
            clip_value=clip_value,
            eps=eps,
        )

    def _apply_broadcast_bandwidth(
        self, e_n: torch.Tensor, bandwidth: str
    ) -> torch.Tensor:
        """Apply bandwidth reduction to the broadcast error signal."""
        return apply_broadcast_bandwidth(self.local_cfg, e_n, bandwidth)

    def _shuffle_soma_error(self, delta: torch.Tensor) -> torch.Tensor:
        """Reassign soma errors through one fixed, seed-controlled derangement.

        This preserves the number and marginal distribution of feedback values
        while sending each neuron's value to the wrong anatomical subtree.
        """
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        elif delta.dim() != 2:
            delta = delta.reshape(delta.size(0), -1)

        n_soma = int(delta.size(1))
        if n_soma <= 1:
            return delta

        seed = int(getattr(self.local_cfg, "broadcast_seed", 0))
        cache = getattr(self, "_broadcast_cache", None)
        if cache is None:
            cache = {}
            self._broadcast_cache = cache
        cache_key = ("per_soma_shuffled", seed, n_soma)
        permutation = cache.get(cache_key)
        if permutation is None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            order = torch.randperm(n_soma, generator=generator)
            permutation = torch.empty_like(order)
            permutation[order] = order.roll(-1)
            cache[cache_key] = permutation

        return delta.index_select(1, permutation.to(device=delta.device))

    @staticmethod
    def _compute_layer_total_conductance(
        rec: dict[str, Any], v_n: torch.Tensor
    ) -> torch.Tensor:
        """Compute total conductance used by conductance-aware local rules."""
        return compute_layer_total_conductance(rec, v_n)

    def _prepare_local_rule_broadcast_state(
        self,
        *,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
    ) -> _BroadcastState:
        """Prepare per-batch broadcast state shared across local-rule layers."""
        bmode = (self.local_cfg.error_broadcast_mode or "scalar").lower()

        if (
            bmode == "pathway_vector"
            or bmode == "path_transport"
            or self.local_cfg.morphology_aware.use_path_propagation
            or self.local_cfg.morphology_aware.morphology_modulator_mode != "none"
            or self.local_cfg.morphology_aware.use_branch_type_rules
            or self.local_cfg.morphology_aware.use_branch_role_rules
            or self.local_cfg.morphology_aware.use_branch_length_modulation
        ):
            self._initialize_branch_morphology(model, layer_records)

        if self.local_cfg.morphology_aware.use_path_propagation:
            path_factors = self._precompute_path_propagation_factors(
                layer_records,
                include_parent_activation_derivative=True,
            )
            for rec, path_factor in zip(layer_records, path_factors):
                rec["path_factor"] = path_factor

        transported_errors: list[torch.Tensor | None] = []
        if bmode == "path_transport":
            transported_errors = self._precompute_path_transport_errors(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        feedback_seeds: list[torch.Tensor | None] = []
        if bmode == "pathway_vector":
            feedback_seeds = self._precompute_feedback_seeds(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        return _BroadcastState(
            mode=bmode,
            transported_errors=transported_errors,
            feedback_seeds=feedback_seeds,
        )

    def _compute_local_broadcast_error(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        out_features: int,
        v_n: torch.Tensor,
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        broadcast_state: _BroadcastState,
    ) -> torch.Tensor:
        """Compute the compartment-level local broadcast error for one layer."""
        bmode = broadcast_state.mode
        if bmode == "per_soma" and delta.dim() == 2 and delta.size(1) == out_features:
            e_n = delta
        elif bmode in {"per_soma_shared", "per_soma_tree"}:
            expanded = expand_soma_error_to_compartments(delta, out_features)
            if isinstance(expanded, torch.Tensor):
                e_n = expanded.to(device=v_n.device, dtype=v_n.dtype)
            else:
                e_n = delta_scalar.expand(-1, out_features)
        elif bmode == "per_soma_shuffled":
            shuffled = self._shuffle_soma_error(delta)
            expanded = expand_soma_error_to_compartments(shuffled, out_features)
            if isinstance(expanded, torch.Tensor):
                e_n = expanded.to(device=v_n.device, dtype=v_n.dtype)
            else:
                e_n = delta_scalar.expand(-1, out_features)
        elif bmode == "low_rank":
            e_n = self._compute_low_rank_broadcast(
                delta=delta,
                layer_idx=layer_idx,
                out_features=out_features,
                device=v_n.device,
                dtype=v_n.dtype,
            )
        elif bmode == "pathway_vector":
            e_n = self._compute_pathway_vector_broadcast(
                seed_error=(
                    broadcast_state.feedback_seeds[layer_idx]
                    if layer_idx < len(broadcast_state.feedback_seeds)
                    else None
                ),
                delta=delta,
                delta_scalar=delta_scalar,
                rec=rec,
                v_n=v_n,
            )
        elif bmode == "path_transport":
            transported = (
                broadcast_state.transported_errors[layer_idx]
                if layer_idx < len(broadcast_state.transported_errors)
                else None
            )
            if isinstance(transported, torch.Tensor):
                e_n = transported.to(device=v_n.device, dtype=v_n.dtype)
            else:
                e_n = delta_scalar.expand(-1, out_features)
        elif bmode == "local_mismatch":
            parent = rec.get("blk_out")
            if parent is None:
                parent = rec.get("exc_out")
            if parent is None:
                parent = rec.get("inh_out")
            e_n = self._compute_local_mismatch_broadcast(
                delta_scalar=delta_scalar,
                v_n=v_n,
                parent=parent,
            )
        else:
            e_n = delta_scalar.expand(-1, out_features)

        noise_sigma = getattr(self.local_cfg, "error_noise_sigma", 0.0)
        if noise_sigma > 0.0:
            e_n = e_n + noise_sigma * torch.randn_like(e_n)

        bandwidth = getattr(self.local_cfg, "broadcast_bandwidth", "full")
        if bandwidth != "full":
            e_n = self._apply_broadcast_bandwidth(e_n, bandwidth)
        return e_n

    def _compute_local_input_resistance(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor | float:
        """Compute conductance-mode input resistance for local-rule updates."""
        if layer_dynamics_mode != "conductance":
            return 1.0
        if not self.local_cfg.three_factor.use_conductance_scaling:
            return 1.0
        g_tot = self._compute_layer_total_conductance(rec, v_n)
        return 1.0 / (g_tot + 1e-8)


__all__ = ["LocalLearningBroadcastMixin"]
