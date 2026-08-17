"""Five-factor phi modulators for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic import (
    clamp_phi,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)

_PHI_EPS = 1e-8
_MIN_CONDITIONAL_VARIANCE = 1e-12


def _update_scalar_ema(
    current: float | None,
    value: float,
    *,
    alpha: float,
) -> float:
    """Update scalar EMA values with the historical phi convention."""
    if current is None:
        return value
    return (1.0 - alpha) * current + alpha * value


def _update_variance_ratio_phi(
    stats: _LayerStats,
    *,
    var_batch: float,
    alpha: float,
) -> float:
    """Update variance EMA and return the unclamped variance-ratio phi."""
    stats.var_ema = _update_scalar_ema(stats.var_ema, var_batch, alpha=alpha)
    return var_batch / (stats.var_ema + _PHI_EPS)


def _resolve_parent_proxy(rec: dict[str, Any]) -> torch.Tensor | None:
    """Return the parent proxy using the legacy blk/exc/inh priority."""
    parent = rec.get("blk_out")
    if parent is None:
        parent = rec.get("exc_out")
    if parent is None:
        parent = rec.get("inh_out")
    return parent


def _center_for_conditional_phi(
    v_n: torch.Tensor,
    parent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center child and parent activations batch-wise."""
    return (
        v_n - v_n.mean(dim=0, keepdim=True),
        parent - parent.mean(dim=0, keepdim=True),
    )


def _conditional_phi_from_moments(
    *,
    var_y: float,
    var_x: float,
    cov_xy: float,
    ridge_lambda: float,
) -> float:
    """Return the historical conditional SNR-like phi from EMA moments."""
    if var_x <= _MIN_CONDITIONAL_VARIANCE:
        return 1.0

    beta = cov_xy / (var_x + ridge_lambda)
    residual_var = var_y - beta * cov_xy
    if residual_var <= _MIN_CONDITIONAL_VARIANCE:
        residual_var = _MIN_CONDITIONAL_VARIANCE
    return var_y / residual_var


def _update_conditional_phi_moments(
    stats: _LayerStats,
    *,
    y: torch.Tensor,
    x: torch.Tensor,
    alpha: float,
) -> tuple[float, float, float]:
    """Update conditional phi EMA moments and return them."""
    decay = 1.0 - alpha
    stats.ema_var_y = decay * (stats.ema_var_y or 0.0) + alpha * float(
        y.pow(2).mean().item()
    )
    stats.ema_var_x = decay * (stats.ema_var_x or 0.0) + alpha * float(
        x.pow(2).mean().item()
    )
    stats.ema_cov_xy = decay * (stats.ema_cov_xy or 0.0) + alpha * float(
        (y * x).mean().item()
    )
    return stats.ema_var_y, stats.ema_var_x, stats.ema_cov_xy


def _mean_scalar_pair(
    v_n: torch.Tensor,
    parent: torch.Tensor,
) -> tuple[float, float]:
    """Return scalar child/parent means for single-sample conditional EMA."""
    return v_n.mean().item(), parent.mean().item()


def _update_single_sample_conditional_moments(
    stats: _LayerStats,
    *,
    y_sample: float,
    x_sample: float,
    alpha: float,
) -> bool:
    """Update single-sample conditional moments; return False on initialization."""
    if stats.ema_mean_y is None:
        stats.ema_mean_y = y_sample
        stats.ema_mean_x = x_sample
        stats.ema_var_y = 0.0
        stats.ema_var_x = 0.0
        stats.ema_cov_xy = 0.0
        return False

    decay = 1.0 - alpha
    old_mean_y = stats.ema_mean_y
    old_mean_x = stats.ema_mean_x
    stats.ema_mean_y = decay * stats.ema_mean_y + alpha * y_sample
    stats.ema_mean_x = decay * stats.ema_mean_x + alpha * x_sample

    delta_y = y_sample - old_mean_y
    delta_x = x_sample - old_mean_x
    stats.ema_var_y = decay * stats.ema_var_y + alpha * (delta_y**2)
    stats.ema_var_x = decay * stats.ema_var_x + alpha * (delta_x**2)
    stats.ema_cov_xy = decay * stats.ema_cov_xy + alpha * (delta_y * delta_x)
    return True


class LocalLearningPhiMixin:
    """Information-confidence modulators for 5-factor local rules."""

    def _compute_layer_phi(self, rec: dict[str, Any]) -> float:
        layer_id = id(rec.get("layer"))
        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        v_n: torch.Tensor = rec.get("v_n")
        if v_n is None:
            return 1.0

        if v_n.size(0) < 2:
            return 1.0

        var_batch = float(v_n.var(unbiased=False).item())
        alpha = self.local_cfg.four_factor.ema_alpha
        phi = _update_variance_ratio_phi(
            stats,
            var_batch=var_batch,
            alpha=alpha,
        )
        phi = self._clamp_phi(phi)
        return float(phi)

    def _compute_layer_phi_conditional(self, rec: dict[str, Any]) -> float:
        """Compute conditional information proxy using parent voltages.

        Approximates I(S; V_n | V_parent) by the fraction of variance in V_n
        explained by its residual after regressing on a parent proxy (blk_out if present).
        """
        v_n: torch.Tensor | None = rec.get("v_n")
        if v_n is None:
            return 1.0

        parent = _resolve_parent_proxy(rec)
        if parent is None:
            return self._compute_layer_phi(rec)

        if (
            v_n.size(0) < 2
            and self.local_cfg.five_factor.phi_estimator == "conditional_ema"
        ):
            return self._compute_layer_phi_conditional_ema(rec, v_n, parent)

        y, x = _center_for_conditional_phi(v_n, parent)
        lam = self.local_cfg.five_factor.phi_ridge_lambda
        layer_id = id(rec.get("layer"))
        stats_local = self._layer_stats.setdefault(layer_id, _LayerStats())
        var_y, var_x, cov_xy = _update_conditional_phi_moments(
            stats_local,
            y=y,
            x=x,
            alpha=self.local_cfg.four_factor.ema_alpha,
        )
        phi = _conditional_phi_from_moments(
            var_y=var_y,
            var_x=var_x,
            cov_xy=cov_xy,
            ridge_lambda=lam,
        )
        return float(self._clamp_phi(phi))

    def _compute_layer_phi_conditional_ema(
        self, rec: dict[str, Any], v_n: torch.Tensor, parent: torch.Tensor
    ) -> float:
        """Compute conditional phi for single samples using online EMA.

        Uses Welford-like updates to maintain running statistics of conditional variance.
        """
        layer_id = id(rec.get("layer"))
        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        y_sample, x_sample = _mean_scalar_pair(v_n, parent)
        alpha = self.local_cfg.four_factor.ema_alpha
        lam = self.local_cfg.five_factor.phi_ridge_lambda

        if not _update_single_sample_conditional_moments(
            stats,
            y_sample=y_sample,
            x_sample=x_sample,
            alpha=alpha,
        ):
            return 1.0

        phi = _conditional_phi_from_moments(
            var_y=stats.ema_var_y,
            var_x=stats.ema_var_x,
            cov_xy=stats.ema_cov_xy,
            ridge_lambda=lam,
        )
        return float(self._clamp_phi(phi))

    def _clamp_phi(self, phi: float) -> float:
        """Clamp the 5F confidence factor using configured stability bounds."""

        return clamp_phi(self.local_cfg, phi)


__all__ = ["LocalLearningPhiMixin"]
