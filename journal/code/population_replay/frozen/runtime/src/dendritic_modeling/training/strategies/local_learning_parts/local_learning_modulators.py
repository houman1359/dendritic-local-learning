"""Layer-wise modulators for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_post_factors_mixin import (
    LocalLearningPostFactorsMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
)


def _apply_layerwise_scale(
    value: Any,
    *,
    layer_depth: int,
    scale: float,
) -> Any:
    """Apply legacy layer-depth scaling when the configured scale is positive."""
    if scale <= 0:
        return value
    return value / (layer_depth**0.5 * scale)


class LocalLearningModulatorMixin(LocalLearningPostFactorsMixin):
    def _compute_local_rho_modulator(
        self,
        *,
        rec: dict[str, Any],
        v0: torch.Tensor,
        layer_depth: int,
    ) -> Any:
        """Compute rho and optional morphology-specific modulation."""
        rho = 1.0
        if self.local_cfg.rule_variant not in {"4f", "5f"}:
            return rho

        rho = self._compute_layer_rho(rec, v0)
        rho = _apply_layerwise_scale(
            rho,
            layer_depth=layer_depth,
            scale=self.local_cfg.four_factor.layer_wise_rho_scale,
        )
        return self._apply_local_rho_morphology_modulator(rec, rho)

    def _apply_local_rho_morphology_modulator(
        self,
        rec: dict[str, Any],
        rho: Any,
    ) -> Any:
        """Apply optional morphology-specific rho modulation."""
        morphology_mode = self.local_cfg.morphology_aware.morphology_modulator_mode
        if morphology_mode == "depth":
            return self._compute_branch_depth_modulator(rec, rho)
        if morphology_mode == "centrality":
            return self._compute_branch_centrality_modulator(rec, rho)
        return rho

    def _compute_local_phi_modulator(
        self,
        *,
        rec: dict[str, Any],
        layer_depth: int,
    ) -> Any:
        """Compute phi and optional layer-depth scaling."""
        phi = 1.0
        if self.local_cfg.rule_variant != "5f":
            return phi

        phi = self._compute_local_phi_value(rec)

        return _apply_layerwise_scale(
            phi,
            layer_depth=layer_depth,
            scale=self.local_cfg.five_factor.layer_wise_phi_scale,
        )

    def _compute_local_phi_value(self, rec: dict[str, Any]) -> Any:
        """Compute the unscaled phi value using the configured estimator."""
        phi_mode = self.local_cfg.five_factor.phi_mode.lower()
        phi_estimator = self.local_cfg.five_factor.phi_estimator.lower()
        if phi_mode == "conditional":
            if phi_estimator == "variance_ema":
                return self._compute_layer_phi(rec)
            return self._compute_layer_phi_conditional(rec)
        return self._compute_layer_phi(rec)

    def _compute_local_branch_scales(
        self,
        rec: dict[str, Any],
    ) -> tuple[
        torch.Tensor | float,
        torch.Tensor | float,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute branch scale and role-alignment modulators."""
        if self.local_cfg.morphology_aware.use_branch_role_rules:
            return self._compute_local_branch_role_scales(rec)
        if self._uses_local_branch_type_scale():
            return self._compute_local_branch_type_scales(rec)
        return 1.0, 1.0, None, None

    def _uses_local_branch_type_scale(self) -> bool:
        """Return whether branch type/length rules should set branch scales."""
        cfg = self.local_cfg.morphology_aware
        return cfg.use_branch_type_rules or cfg.use_branch_length_modulation

    def _compute_local_branch_role_scales(
        self,
        rec: dict[str, Any],
    ) -> tuple[
        torch.Tensor | float,
        torch.Tensor | float,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute role-aware branch scales and alignment factors."""
        branch_scale, dendritic_branch_scale = self._get_branch_role_scales(rec)
        (
            role_synaptic_alignment,
            role_block_alignment,
        ) = self._get_branch_role_alignment_factors(rec)
        return (
            branch_scale,
            dendritic_branch_scale,
            role_synaptic_alignment,
            role_block_alignment,
        )

    def _compute_local_branch_type_scales(
        self,
        rec: dict[str, Any],
    ) -> tuple[
        torch.Tensor | float,
        torch.Tensor | float,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute branch type/length scales without role alignment."""
        branch_scale = self._get_branch_type_scale(rec)
        return branch_scale, branch_scale, None, None

    def _compute_local_layer_modulators(
        self,
        *,
        rec: dict[str, Any],
        v0: torch.Tensor,
        layer_depth: int,
    ) -> _LayerModulators:
        rho = self._compute_local_rho_modulator(
            rec=rec,
            v0=v0,
            layer_depth=layer_depth,
        )
        phi = self._compute_local_phi_modulator(
            rec=rec,
            layer_depth=layer_depth,
        )
        (
            branch_scale,
            dendritic_branch_scale,
            role_synaptic_alignment,
            role_block_alignment,
        ) = self._compute_local_branch_scales(rec)

        return _LayerModulators(
            rho=rho,
            phi=phi,
            branch_scale=branch_scale,
            dendritic_branch_scale=dendritic_branch_scale,
            role_synaptic_alignment=role_synaptic_alignment,
            role_block_alignment=role_block_alignment,
        )


__all__ = ["LocalLearningModulatorMixin"]
