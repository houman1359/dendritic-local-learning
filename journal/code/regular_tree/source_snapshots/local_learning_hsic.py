"""Pure HSIC / additive-gain helpers for local credit assignment.

Extracted from ``LocalCreditAssignment``. Stateless: config passed explicitly as
``local_cfg``, per-layer state as ``rec``. The stateful additive-gain parameter
cache and the ``_layer_stats``-coupled rho/phi accumulators stay on the trainer.
"""

from __future__ import annotations

from typing import Any

import torch


def compute_kernel_matrix(local_cfg, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute kernel matrix K(X, Y) based on the configured kernel type."""
    kernel_type = local_cfg.hsic.kernel.lower()

    if kernel_type == "linear":
        return X @ Y.t()
    elif kernel_type == "rbf" or kernel_type == "gaussian":
        gamma = (
            1.0 / (2.0 * local_cfg.hsic.sigma**2) if local_cfg.hsic.sigma > 0 else 1.0
        )
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        dist_sq = X_norm + Y_norm.t() - 2.0 * (X @ Y.t())
        return torch.exp(-gamma * dist_sq)
    elif kernel_type == "polynomial" or kernel_type == "poly":
        dot_product = X @ Y.t()
        return (local_cfg.hsic.coef0 + dot_product) ** local_cfg.hsic.degree
    else:
        return X @ Y.t()


def compute_hsic_gradient(
    local_cfg, z: torch.Tensor, y: torch.Tensor, weight: float, grad_type: str
) -> torch.Tensor:
    """Compute the HSIC gradient w.r.t. ``z`` using the configured kernel.

    ``grad_type`` is "self" (self-dependence) or "target" (target dependence).
    """
    batch_size = z.size(0)
    kernel_type = local_cfg.hsic.kernel.lower()

    if kernel_type == "linear":
        if grad_type == "self":
            return (2.0 / float(batch_size)) * z * local_cfg.hsic.weight * weight
        else:  # target
            cov_zy = (z.t() @ y) / float(batch_size)
            grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
            return grad_z * local_cfg.hsic.weight * weight

    elif kernel_type in ["rbf", "gaussian"]:
        gamma = (
            1.0 / (2.0 * local_cfg.hsic.sigma**2) if local_cfg.hsic.sigma > 0 else 1.0
        )
        if grad_type == "self":
            K_zz = compute_kernel_matrix(local_cfg, z, z)
            z_expanded = z.unsqueeze(1)
            z_diff = z_expanded - z.unsqueeze(0)
            K_expanded = K_zz.unsqueeze(-1)
            grad_z = -2.0 * gamma * (K_expanded * z_diff).sum(dim=1)
            return grad_z * local_cfg.hsic.weight * weight / float(batch_size)
        else:  # target
            K_zz = compute_kernel_matrix(local_cfg, z, z)
            L_yy = compute_kernel_matrix(local_cfg, y, y)
            Lc = (
                L_yy
                - L_yy.mean(dim=0, keepdim=True)
                - L_yy.mean(dim=1, keepdim=True)
                + L_yy.mean()
            )
            z_expanded = z.unsqueeze(1)
            z_diff = z_expanded - z.unsqueeze(0)
            weight_mat = (K_zz * Lc).unsqueeze(-1)
            grad_z = 2.0 * gamma * (weight_mat * z_diff).sum(dim=1)
            return grad_z * local_cfg.hsic.weight * weight / float(batch_size)

    elif kernel_type in ["polynomial", "poly"]:
        degree = local_cfg.hsic.degree
        coef0 = local_cfg.hsic.coef0
        if grad_type == "self":
            dot_zz = z @ z.t()
            poly_base = coef0 + dot_zz
            poly_grad_coef = degree * (poly_base ** (degree - 1))
            grad_z = (poly_grad_coef @ z) * 2.0
            return grad_z * local_cfg.hsic.weight * weight / float(batch_size)
        else:  # target
            dot_zz = z @ z.t()
            K_base = coef0 + dot_zz
            K_coef = degree * (K_base ** (degree - 1))
            dot_yy = y @ y.t()
            L_base = coef0 + dot_yy
            L_yy = L_base**degree
            Lc = (
                L_yy
                - L_yy.mean(dim=0, keepdim=True)
                - L_yy.mean(dim=1, keepdim=True)
                + L_yy.mean()
            )
            weight_mat = K_coef * Lc
            grad_z = (weight_mat @ z) * 2.0
            return grad_z * local_cfg.hsic.weight * weight / float(batch_size)

    else:
        if grad_type == "self":
            return (2.0 / float(batch_size)) * z * local_cfg.hsic.weight * weight
        else:
            cov_zy = (z.t() @ y) / float(batch_size)
            grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
            return grad_z * local_cfg.hsic.weight * weight


def compute_additive_pseudo_signals(
    rec: dict[str, Any], v_n: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pseudo-R_tot and pseudo-driving-force for additive networks.

    pseudo_R: analogous to 1/g_tot; normalizes by total activation magnitude.
    pseudo_drive: analogous to (E_rev - V); uses voltage deviation from mean.
    """
    total_activation = torch.ones_like(v_n)
    exc_out = rec.get("exc_out")
    if exc_out is not None:
        total_activation = total_activation + exc_out.abs()
    inh_out = rec.get("inh_out")
    if inh_out is not None:
        total_activation = total_activation + inh_out.abs()
    pseudo_R = 1.0 / (total_activation + 1e-8)

    v_mean = v_n.mean(dim=0, keepdim=True)
    pseudo_drive = v_mean - v_n + 1.0
    return pseudo_R, pseudo_drive


def clamp_phi(local_cfg, phi: float) -> float:
    """Clamp the 5F confidence factor using configured stability bounds."""
    phi_cfg = getattr(local_cfg, "five_factor", None)
    phi_min = float(getattr(phi_cfg, "phi_clamp_min", 0.25))
    phi_max = float(getattr(phi_cfg, "phi_clamp_max", 4.0))
    if phi_min > phi_max:
        phi_min, phi_max = phi_max, phi_min
    return max(phi_min, min(phi_max, phi))


def compute_dendritic_normalization(
    rec: dict[str, Any], grad_g_den: torch.Tensor
) -> torch.Tensor:
    """Normalize branch conductance updates by total branch conductance."""
    blk_layer = rec.get("blk_module")
    if blk_layer is None or not hasattr(blk_layer, "sum_conductances"):
        return grad_g_den

    g_den_total = blk_layer.sum_conductances().detach()
    norm_factor = 1.0 / (g_den_total.unsqueeze(-1) + 1e-8)

    if grad_g_den.dim() == 3:
        norm_factor = norm_factor.unsqueeze(0)

    return grad_g_den * norm_factor


__all__ = [
    "clamp_phi",
    "compute_additive_pseudo_signals",
    "compute_dendritic_normalization",
    "compute_hsic_gradient",
    "compute_kernel_matrix",
]
