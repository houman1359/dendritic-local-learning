"""Local gradient-application methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_dendritic_block_mixin import (
    LocalLearningDendriticBlockGradientMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradient_utils import (
    _accumulate_parameter_grad,
    _apply_optional_alignment,
    _factor_pre_weight_grad,
    _normalize_batch_gradient,
    _reactivation_output_slope,
    _reduce_batch_mean_gradient,
    _scale_reactivation_error,
    _topk_pre_weight_grad,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_reactivation_mixin import (
    LocalLearningReactivationMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_topk_mixin import (
    LocalLearningTopKGradientMixin,
)


def _resolve_stdp_post_signal(
    rec: dict[str, Any],
    v_n: torch.Tensor,
) -> torch.Tensor:
    post_signal = rec.get("v_out")
    if isinstance(post_signal, torch.Tensor):
        return post_signal
    return v_n


class LocalLearningGradientMixin(
    LocalLearningTopKGradientMixin,
    LocalLearningDendriticBlockGradientMixin,
    LocalLearningReactivationMixin,
):
    """Gradient writeback helpers used by ``LocalCreditAssignment``."""

    def _apply_synaptic_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        batch_size: int,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
    ) -> Any:
        """Apply synaptic and dendritic-block local gradients for one layer."""
        exc_layer = self._apply_excitatory_topk_pathways(
            rec=rec,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            modulators=modulators,
            post_factors=post_factors,
        )

        self._apply_inhibitory_topk_pathways(
            rec=rec,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            modulators=modulators,
            post_factors=post_factors,
        )

        self._apply_dendritic_block_local_grad(
            rec=rec,
            e_v=post_factors.e_v,
            post_factor_raw=post_factors.post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=modulators.rho,
            phi=modulators.phi,
            dendritic_branch_scale=modulators.dendritic_branch_scale,
            role_block_alignment=modulators.role_block_alignment,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
        )
        return exc_layer

    def _apply_auxiliary_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        exc_layer: Any,
        layer_idx: int,
        num_layers: int,
        y_target: torch.Tensor | None,
        batch_size: int,
        e_n: torch.Tensor,
        stdp_error_signal: torch.Tensor,
        v_n: torch.Tensor,
        modulators: _LayerModulators,
        update_reactivation: bool,
    ) -> None:
        """Apply reactivation, STDP, and HSIC local-gradient phases."""
        self._apply_reactivation_local_grad(
            rec=rec,
            e_n=e_n,
            v_n=v_n,
            rho=modulators.rho,
            phi=modulators.phi,
            branch_scale=modulators.branch_scale,
            update_reactivation=update_reactivation,
        )

        self._apply_stdp_local_gradients(
            rec=rec,
            v_n=v_n,
            stdp_error_signal=stdp_error_signal,
        )

        self._apply_hsic_auxiliary_gradients(
            rec=rec,
            exc_layer=exc_layer,
            layer_idx=layer_idx,
            num_layers=num_layers,
            y_target=y_target,
            batch_size=batch_size,
        )

    def _apply_layer_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        num_layers: int,
        y_target: torch.Tensor | None,
        batch_size: int,
        e_n: torch.Tensor,
        stdp_error_signal: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
        update_reactivation: bool,
    ) -> None:
        exc_layer = self._apply_synaptic_local_gradients(
            rec=rec,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            modulators=modulators,
            post_factors=post_factors,
        )

        self._apply_auxiliary_local_gradients(
            rec=rec,
            exc_layer=exc_layer,
            layer_idx=layer_idx,
            num_layers=num_layers,
            y_target=y_target,
            batch_size=batch_size,
            e_n=e_n,
            stdp_error_signal=stdp_error_signal,
            v_n=v_n,
            modulators=modulators,
            update_reactivation=update_reactivation,
        )

    def _apply_stdp_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        stdp_error_signal: torch.Tensor,
    ) -> None:
        """Apply configured STDP gradients for one local-learning record."""
        if not self._stdp_enabled():
            return

        post_signal = _resolve_stdp_post_signal(rec, v_n)
        self._apply_stdp_pathways(
            rec=rec,
            post_signal=post_signal,
            error_signal=stdp_error_signal,
        )


__all__ = [
    "LocalLearningGradientMixin",
    "_accumulate_parameter_grad",
    "_apply_optional_alignment",
    "_factor_pre_weight_grad",
    "_normalize_batch_gradient",
    "_reactivation_output_slope",
    "_reduce_batch_mean_gradient",
    "_resolve_stdp_post_signal",
    "_scale_reactivation_error",
    "_topk_pre_weight_grad",
]
