"""Dendritic block-gradient helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradient_utils import (
    _accumulate_parameter_grad,
    _apply_optional_alignment,
    _reduce_batch_mean_gradient,
)


def _expand_resistance_for_blocks(
    r_tot: torch.Tensor | float,
    v_n: torch.Tensor,
) -> torch.Tensor:
    """Broadcast scalar or per-neuron resistance over dendritic blocks."""
    if isinstance(r_tot, torch.Tensor):
        return r_tot.unsqueeze(-1)
    return torch.ones_like(v_n).unsqueeze(-1) * float(r_tot)


def _apply_dendritic_rho_modulator(
    factor: torch.Tensor,
    rho: torch.Tensor | float,
    *,
    rule_variant: str,
) -> torch.Tensor:
    """Apply the dendritic-block rho rule for 4f/5f variants."""
    if rule_variant not in {"4f", "5f"}:
        return factor
    if isinstance(rho, torch.Tensor):
        return factor * rho.unsqueeze(0).unsqueeze(-1)
    return factor * float(rho)


def _apply_dendritic_phi_modulator(
    factor: torch.Tensor,
    phi: float,
    *,
    rule_variant: str,
) -> torch.Tensor:
    """Apply the dendritic-block phi rule for 5f variants."""
    if rule_variant == "5f":
        return factor * float(phi)
    return factor


def _apply_dendritic_branch_scale(
    factor: torch.Tensor,
    dendritic_branch_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Apply scalar, per-neuron, or per-block dendritic branch scaling."""
    if isinstance(dendritic_branch_scale, torch.Tensor):
        if dendritic_branch_scale.dim() == 2:
            return factor * dendritic_branch_scale.unsqueeze(0)
        return factor * dendritic_branch_scale.unsqueeze(0).unsqueeze(-1)
    return factor * float(dendritic_branch_scale)


def _apply_dendritic_block_modulators(
    factor: torch.Tensor,
    *,
    rho: torch.Tensor | float,
    phi: float,
    dendritic_branch_scale: torch.Tensor | float,
    rule_variant: str,
) -> torch.Tensor:
    """Apply dendritic-block modulators in the legacy order."""
    factor = _apply_dendritic_rho_modulator(factor, rho, rule_variant=rule_variant)
    factor = _apply_dendritic_phi_modulator(factor, phi, rule_variant=rule_variant)
    return _apply_dendritic_branch_scale(factor, dendritic_branch_scale)


def _conductance_dendritic_block_factor(
    *,
    e_v: torch.Tensor,
    v_n: torch.Tensor,
    r_tot: torch.Tensor | float,
    x_blk: torch.Tensor,
) -> torch.Tensor:
    """Compute dendritic-block factors for conductance dynamics."""
    diff = x_blk - v_n.unsqueeze(-1)
    r_exp = _expand_resistance_for_blocks(r_tot, v_n)
    return e_v.unsqueeze(-1) * r_exp * diff


def _additive_dendritic_block_factor(
    *,
    e_v: torch.Tensor,
    v_n: torch.Tensor,
    post_factor_raw: torch.Tensor,
    x_blk: torch.Tensor,
    additive_mode: str,
    pseudo_R: torch.Tensor | None,
) -> torch.Tensor:
    """Compute dendritic-block factors for additive dynamics."""
    if additive_mode == "input_dependent":
        if pseudo_R is None:
            raise ValueError("pseudo_R is required for input_dependent additive mode.")
        diff = x_blk - v_n.unsqueeze(-1)
        return e_v.unsqueeze(-1) * pseudo_R.unsqueeze(-1) * diff
    if additive_mode in ("learned_gain", "running_stats"):
        return post_factor_raw.unsqueeze(-1) * x_blk
    return e_v.unsqueeze(-1) * x_blk


def _dendritic_block_homeostasis_factor(
    *,
    voltage_homeo_error: torch.Tensor,
    v_n: torch.Tensor,
    r_tot: torch.Tensor | float,
    x_blk: torch.Tensor,
    layer_dynamics_mode: str,
) -> torch.Tensor:
    """Compute dendritic-block voltage homeostasis factors."""
    if layer_dynamics_mode == "conductance":
        diff_homeo = x_blk - v_n.unsqueeze(-1)
        r_homeo = _expand_resistance_for_blocks(r_tot, v_n)
        return voltage_homeo_error.unsqueeze(-1) * r_homeo * diff_homeo
    return voltage_homeo_error.unsqueeze(-1) * x_blk


class LocalLearningDendriticBlockGradientMixin:
    """Gradient writeback for dendritic block conductances."""

    def _apply_dendritic_block_local_grad(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        dendritic_branch_scale: Any,
        role_block_alignment: torch.Tensor | None,
        layer_dynamics_mode: str,
        batch_size: int,
    ) -> None:
        blk_layer = rec.get("blk_module")
        x_blk_raw = rec.get("x_blk_raw")
        if blk_layer is None or x_blk_raw is None:
            return

        block_size = blk_layer.block_size
        x_blk = x_blk_raw.view(x_blk_raw.size(0), blk_layer.out_features, block_size)
        den_factor = self._compute_dendritic_block_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=rho,
            phi=phi,
            dendritic_branch_scale=dendritic_branch_scale,
            x_blk=x_blk,
            layer_dynamics_mode=layer_dynamics_mode,
        )

        grad_g_den = _reduce_batch_mean_gradient(
            den_factor,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is not None:
            den_homeo_factor = self._compute_dendritic_block_homeostasis_factor(
                voltage_homeo_error=voltage_homeo_error,
                v_n=v_n,
                r_tot=r_tot,
                x_blk=x_blk,
                layer_dynamics_mode=layer_dynamics_mode,
            )
            grad_homeo_den = _reduce_batch_mean_gradient(
                den_homeo_factor,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g_den = grad_g_den + grad_homeo_den

        grad_g_den = _apply_optional_alignment(grad_g_den, role_block_alignment)

        if self.local_cfg.morphology_aware.use_dendritic_normalization:
            grad_g_den = self._compute_dendritic_normalization(rec, grad_g_den)

        self._add_dendritic_block_transformed_grad(blk_layer, grad_g_den)

    def _add_dendritic_block_transformed_grad(
        self,
        blk_layer: Any,
        local_grad: torch.Tensor,
    ) -> None:
        den_chain = self._weight_transform_derivative(
            blk_layer.log_weight.detach(),
            getattr(blk_layer, "weight_transform", "exp"),
        )
        grad_param_den = local_grad * den_chain
        _accumulate_parameter_grad(blk_layer.log_weight, grad_param_den)

    def _compute_dendritic_block_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        dendritic_branch_scale: Any,
        x_blk: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        den_factor = self._compute_base_dendritic_block_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            x_blk=x_blk,
            layer_dynamics_mode=layer_dynamics_mode,
        )
        return _apply_dendritic_block_modulators(
            den_factor,
            rho=rho,
            phi=phi,
            dendritic_branch_scale=dendritic_branch_scale,
            rule_variant=self.local_cfg.rule_variant,
        )

    def _compute_base_dendritic_block_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        x_blk: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        """Compute dendritic-block factors before rho/phi/branch modulation."""
        if layer_dynamics_mode == "conductance":
            return _conductance_dendritic_block_factor(
                e_v=e_v,
                v_n=v_n,
                r_tot=r_tot,
                x_blk=x_blk,
            )

        additive_mode = getattr(
            self.local_cfg.three_factor, "additive_gain_mode", "none"
        )
        pseudo_R = None
        if additive_mode == "input_dependent":
            pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
        return _additive_dendritic_block_factor(
            e_v=e_v,
            v_n=v_n,
            post_factor_raw=post_factor_raw,
            x_blk=x_blk,
            additive_mode=additive_mode,
            pseudo_R=pseudo_R,
        )

    def _compute_dendritic_block_homeostasis_factor(
        self,
        *,
        voltage_homeo_error: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        x_blk: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        return _dendritic_block_homeostasis_factor(
            voltage_homeo_error=voltage_homeo_error,
            v_n=v_n,
            r_tot=r_tot,
            x_blk=x_blk,
            layer_dynamics_mode=layer_dynamics_mode,
        )


__all__ = ["LocalLearningDendriticBlockGradientMixin"]
