"""Four-factor rho modulators for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_phi_mixin import (
    LocalLearningPhiMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)

logger = logging.getLogger(__name__)

_RHO_EPS = 1e-8


def _clamp_rho(rho: float) -> float:
    """Clamp rho to the legacy stability range."""

    return float(max(0.1, min(2.0, rho)))


def _ema_update(
    current: float | None,
    value: float,
    *,
    alpha: float,
    decay: float,
) -> float:
    """Update a scalar EMA using the historical rho rule."""
    if current is None:
        return value
    return decay * current + alpha * value


def _mean_activity_pair(
    v_n: torch.Tensor,
    v0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample mean output/input activities used by rho estimators."""
    return v_n.mean(dim=1), v0.mean(dim=1)


def _pearson_proxy_from_samples(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the rho Pearson proxy used by batch and augment modes."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = a_centered.std(unbiased=False) * b_centered.std(unbiased=False) + _RHO_EPS
    return float((a_centered * b_centered).mean().item() / denom.item())


def _noisy_mean_samples(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    n_samples: int,
    noise_sigma: float,
) -> tuple[list[float], list[float]]:
    """Return noisy scalar sample means while preserving RNG call order."""
    a_samples = []
    b_samples = []
    a_orig = a.clone()
    b_orig = b.clone()

    for _ in range(n_samples):
        noise_a = torch.randn_like(a_orig) * noise_sigma
        noise_b = torch.randn_like(b_orig) * noise_sigma
        a_noisy = a_orig + noise_a
        b_noisy = b_orig + noise_b
        a_samples.append(a_noisy.mean().item())
        b_samples.append(b_noisy.mean().item())

    return a_samples, b_samples


def _mean_augmented_activity_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    n_samples: int,
    noise_sigma: float,
) -> tuple[float, float]:
    """Return the mean noisy scalar samples used by ``ema+augment``."""
    a_samples, b_samples = _noisy_mean_samples(
        a,
        b,
        n_samples=n_samples,
        noise_sigma=noise_sigma,
    )
    return sum(a_samples) / n_samples, sum(b_samples) / n_samples


def _augmented_rho_sample(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    n_samples: int,
    noise_sigma: float,
) -> float:
    """Compute the historical micro-ensemble augmentation rho sample."""
    a_samples, b_samples = _noisy_mean_samples(
        a,
        b,
        n_samples=n_samples,
        noise_sigma=noise_sigma,
    )
    return _pearson_proxy_from_samples(torch.tensor(a_samples), torch.tensor(b_samples))


def _update_single_sample_ema_rho(
    stats: _LayerStats,
    *,
    a_sample: float,
    b_sample: float,
    alpha: float,
    decay: float,
) -> float:
    """Update single-sample EMA covariance statistics and return unclamped rho."""
    if stats.ema_mean_y is None:
        stats.ema_mean_y = a_sample
        stats.ema_mean_x = b_sample
        stats.ema_var_y = 0.0
        stats.ema_var_x = 0.0
        stats.ema_cov_xy = 0.0

    old_mean_y = stats.ema_mean_y
    old_mean_x = stats.ema_mean_x
    stats.ema_mean_y = decay * stats.ema_mean_y + alpha * a_sample
    stats.ema_mean_x = decay * stats.ema_mean_x + alpha * b_sample

    delta_y = a_sample - old_mean_y
    delta_x = b_sample - old_mean_x
    stats.ema_var_y = decay * stats.ema_var_y + alpha * (delta_y**2)
    stats.ema_var_x = decay * stats.ema_var_x + alpha * (delta_x**2)
    stats.ema_cov_xy = decay * stats.ema_cov_xy + alpha * (delta_y * delta_x)

    denom = (stats.ema_var_y * stats.ema_var_x) ** 0.5 + _RHO_EPS
    rho_ema = stats.ema_cov_xy / denom
    stats.rho_ema = rho_ema
    return rho_ema


class LocalLearningRhoMixin(LocalLearningPhiMixin):
    """Input-output correlation modulators for 4-factor and 5-factor rules."""

    def _compute_layer_rho(self, rec: dict[str, Any], v0: torch.Tensor) -> float:
        layer_id = id(rec.get("layer"))
        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        v_n: torch.Tensor = rec.get("v_n")
        if v_n is None:
            return 1.0

        batch_size = v_n.size(0)
        alpha = self.local_cfg.four_factor.ema_alpha
        decay = 1.0 - alpha
        a, b = _mean_activity_pair(v_n, v0)

        rho_mode = self.local_cfg.four_factor.rho_mode.lower()

        if rho_mode == "dot":
            rho_val = float((a * b).mean().item())
            stats.rho_ema = _ema_update(
                stats.rho_ema,
                rho_val,
                alpha=alpha,
                decay=decay,
            )
            return _clamp_rho(stats.rho_ema)

        if rho_mode == "none":
            return 1.0

        if batch_size >= 2:
            rho_batch = _pearson_proxy_from_samples(a, b)
            stats.rho_ema = _ema_update(
                stats.rho_ema,
                rho_batch,
                alpha=alpha,
                decay=decay,
            )
            return _clamp_rho(stats.rho_ema)

        estimator = self.local_cfg.four_factor.rho_estimator.lower()

        if estimator in ["ema", "ema+augment"]:
            if estimator == "ema+augment":
                a_sample, b_sample = _mean_augmented_activity_pair(
                    a,
                    b,
                    n_samples=self.local_cfg.four_factor.augment_k,
                    noise_sigma=self.local_cfg.four_factor.augment_noise_sigma,
                )
            else:
                a_sample = a.mean().item()
                b_sample = b.mean().item()

            rho_ema = _update_single_sample_ema_rho(
                stats,
                a_sample=a_sample,
                b_sample=b_sample,
                alpha=alpha,
                decay=decay,
            )
            return _clamp_rho(rho_ema)

        if estimator == "augment":
            rho_augment = _augmented_rho_sample(
                a,
                b,
                n_samples=self.local_cfg.four_factor.augment_k,
                noise_sigma=self.local_cfg.four_factor.augment_noise_sigma,
            )
            stats.rho_ema = _ema_update(
                stats.rho_ema,
                rho_augment,
                alpha=alpha,
                decay=decay,
            )
            return _clamp_rho(rho_augment)

        logger.warning(f"Unknown rho_estimator '{estimator}'; returning 1.0")
        return 1.0


__all__ = ["LocalLearningRhoMixin"]
