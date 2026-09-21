"""Post-synaptic factor assembly for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_homeostasis_mixin import (
    LocalLearningHomeostasisMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


def _activation_derivative_tensor(
    activation_derivative: Any,
    v_n: torch.Tensor,
) -> torch.Tensor:
    """Return an activation derivative tensor on the layer voltage device/dtype."""
    if not isinstance(activation_derivative, torch.Tensor):
        activation_derivative = torch.ones_like(v_n)
    return activation_derivative.to(device=v_n.device, dtype=v_n.dtype)


def _conductance_post_factor(
    *,
    e_v: torch.Tensor,
    v_n: torch.Tensor,
    r_tot: torch.Tensor | float,
    use_driving_force: bool,
    e_rev_exc: float,
    theta: float,
) -> torch.Tensor:
    """Compute the conductance-mode post factor using the legacy formula."""
    if use_driving_force:
        post_mod = e_rev_exc - v_n
    else:
        post_mod = v_n - theta
    return e_v * r_tot * post_mod


def _apply_output_modulator(
    post_factor: torch.Tensor,
    modulator: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a neuron-wise or scalar output modulator."""
    if isinstance(modulator, torch.Tensor):
        return post_factor * modulator.unsqueeze(0)
    return post_factor * float(modulator)


def _apply_post_factor_modulators(
    post_factor: torch.Tensor,
    modulators: _LayerModulators,
) -> torch.Tensor:
    """Apply rho, phi, and branch-scale modulators in the historical order."""
    post_factor = _apply_output_modulator(post_factor, modulators.rho)
    post_factor = post_factor * float(modulators.phi)
    return _apply_output_modulator(post_factor, modulators.branch_scale)


class LocalLearningPostFactorsMixin(LocalLearningHomeostasisMixin):
    """Build local post-synaptic factors from error, voltage, and modulators."""

    def _compute_local_voltage_error(
        self,
        *,
        rec: dict[str, Any],
        e_n: torch.Tensor,
        v_n: torch.Tensor,
    ) -> torch.Tensor:
        """Map neuron errors through the cached activation derivative."""
        activation_derivative = self._get_layer_activation_derivative(
            rec=rec,
            v_n=v_n,
            v_out=rec.get("v_out"),
        )
        activation_derivative = _activation_derivative_tensor(
            activation_derivative, v_n
        )
        return e_n * activation_derivative

    def _compute_additive_post_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        v_n: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute additive-mode post factors using the configured gain path."""
        additive_mode = getattr(
            self.local_cfg.three_factor, "additive_gain_mode", "none"
        )
        if additive_mode == "learned_gain":
            gain_params = self._ensure_additive_gain_params(v_n.size(1), v_n.device)
            gain_n = torch.sigmoid(gain_params)
            return e_v * gain_n.unsqueeze(0)
        if additive_mode == "input_dependent":
            pseudo_R, pseudo_drive = self._compute_additive_pseudo_signals(rec, v_n)
            return e_v * pseudo_R * pseudo_drive
        if additive_mode == "running_stats":
            inv_std = self._get_additive_running_inv_std(layer_idx, v_n)
            return e_v * inv_std
        return e_v

    def _compute_base_post_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute the post factor before rho/phi/branch modulation."""
        if layer_dynamics_mode == "conductance":
            cfg = self.local_cfg.three_factor
            return _conductance_post_factor(
                e_v=e_v,
                v_n=v_n,
                r_tot=r_tot,
                use_driving_force=cfg.use_driving_force,
                e_rev_exc=cfg.e_rev_exc,
                theta=cfg.theta,
            )
        return self._compute_additive_post_factor(
            rec=rec,
            e_v=e_v,
            v_n=v_n,
            layer_idx=layer_idx,
        )

    def _compute_local_post_factors(
        self,
        *,
        rec: dict[str, Any],
        e_n: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        layer_idx: int,
        modulators: _LayerModulators,
    ) -> _PostFactors:
        e_v = self._compute_local_voltage_error(rec=rec, e_n=e_n, v_n=v_n)

        post_factor = self._compute_base_post_factor(
            rec=rec,
            e_v=e_v,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            layer_idx=layer_idx,
        )
        post_factor_raw = post_factor
        post_factor = _apply_post_factor_modulators(post_factor, modulators)

        return _PostFactors(
            e_v=e_v,
            post_factor=post_factor,
            post_factor_raw=post_factor_raw,
        )


__all__ = ["LocalLearningPostFactorsMixin"]
