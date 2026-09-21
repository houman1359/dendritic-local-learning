"""Pure homeostasis helpers for local credit assignment.

Extracted from ``LocalCreditAssignment``. Each function computes and returns a
local auxiliary signal/gradient (no trainer state mutation); the trainer applies
them. Config is passed in explicitly as ``local_cfg``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricTanh,
    ParametricTanhOnlyM,
)

logger = logging.getLogger(__name__)


def compute_inhibitory_homeostasis_factor(
    local_cfg,
    rec: dict[str, Any],
    R_tot: torch.Tensor | float,
    v_n: torch.Tensor,
) -> torch.Tensor | None:
    """Compute an auxiliary local inhibitory-homeostasis gradient factor."""
    cfg = getattr(local_cfg, "inhibitory_homeostasis", None)
    if cfg is None or not getattr(cfg, "enabled", False) or cfg.weight <= 0.0:
        return None

    layer = rec.get("layer")
    if layer is not None and not getattr(layer, "use_shunting", True):
        return None

    mode = str(getattr(cfg, "mode", "r_tot")).lower()
    weight = float(cfg.weight)

    if mode == "r_tot":
        if not isinstance(R_tot, torch.Tensor):
            return None
        target_r = float(getattr(cfg, "target_r_tot", 0.35))
        return -weight * (R_tot - target_r) * (R_tot**2)

    if mode == "voltage":
        r_tensor = (
            R_tot
            if isinstance(R_tot, torch.Tensor)
            else torch.ones_like(v_n) * float(R_tot)
        )
        target_v = float(getattr(cfg, "target_voltage", 0.15))
        e_inh = float(getattr(local_cfg.three_factor, "e_rev_inh", 0.0))
        return weight * (v_n - target_v) * r_tensor * (e_inh - v_n)

    logger.warning(
        "Unknown inhibitory_homeostasis.mode='%s'; disabling auxiliary term.",
        mode,
    )
    return None


def compute_voltage_homeostasis_error(
    local_cfg, v_n: torch.Tensor
) -> torch.Tensor | None:
    """Return a local voltage-centering error signal.

    This term is intentionally local: it depends only on the compartment voltage
    and a fixed target midpoint, and it is applied through the same local
    sensitivity factors used by the task-driven updates.
    """
    cfg = getattr(local_cfg, "voltage_homeostasis", None)
    if cfg is None or not getattr(cfg, "enabled", False) or cfg.weight <= 0.0:
        return None

    target_v = float(getattr(cfg, "target_voltage", 0.5))
    weight = float(cfg.weight)
    return weight * (target_v - v_n)


def compute_gate_homeostasis_aux_grads(
    local_cfg,
    react_module: nn.Module | None,
    v_n: torch.Tensor | None,
    v_out: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return strictly local auxiliary grads for reactivation parameters.

    This is the local counterpart to gate-occupancy warmup: it uses only each
    layer's own gate outputs and local derivatives, never a global backpropagated
    signal. The returned tensors already live in parameter coordinates
    (``log_m``, ``b``) and can be added directly to ``.grad``.
    """
    cfg = getattr(local_cfg, "gate_homeostasis", None)
    if (
        cfg is None
        or not getattr(cfg, "enabled", False)
        or v_n is None
        or v_out is None
        or not isinstance(react_module, ParametricActivation)
        or not hasattr(react_module, "log_m")
    ):
        return None, None

    if v_n.dim() != 2 or v_out.dim() != 2:
        return None, None

    if not isinstance(react_module, (ParametricTanh, ParametricTanhOnlyM)):
        return None, None

    target_mean = float(getattr(cfg, "target_mean", 0.5))
    target_saturation = float(getattr(cfg, "target_saturation", 0.2))
    low_thr = float(getattr(cfg, "low_threshold", 0.1))
    high_thr = float(getattr(cfg, "high_threshold", 0.9))
    sat_k = float(getattr(cfg, "saturation_slope", 20.0))
    center_weight = float(getattr(cfg, "center_weight", 0.0))
    saturation_weight = float(getattr(cfg, "saturation_weight", 0.0))

    if center_weight <= 0.0 and saturation_weight <= 0.0:
        return None, None

    y = v_out
    base = 0.5 * (1.0 - (2.0 * y - 1.0).square())
    m_vec = react_module.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)

    if hasattr(react_module, "b"):
        b_vec = react_module.b.detach().to(device=v_n.device, dtype=v_n.dtype)
    else:
        b_vec = torch.full_like(m_vec, float(getattr(react_module, "fixed_b", 0.5)))

    dr_dlogm = base * (v_n - b_vec.unsqueeze(0)) * m_vec.unsqueeze(0)
    dr_db: torch.Tensor | None
    if hasattr(react_module, "b"):
        dr_db = -base * m_vec.unsqueeze(0)
    else:
        dr_db = None

    grad_log_m = torch.zeros_like(m_vec)
    grad_b = torch.zeros_like(b_vec) if dr_db is not None else None

    if center_weight > 0.0:
        mean_r = y.mean(dim=0)
        mean_err = mean_r - target_mean
        dmean_dlogm = dr_dlogm.mean(dim=0)
        grad_log_m = grad_log_m + (2.0 * center_weight) * mean_err * dmean_dlogm
        if dr_db is not None and grad_b is not None:
            dmean_db = dr_db.mean(dim=0)
            grad_b = grad_b + (2.0 * center_weight) * mean_err * dmean_db

    if saturation_weight > 0.0:
        soft_lo = torch.sigmoid(sat_k * (low_thr - y))
        soft_hi = torch.sigmoid(sat_k * (y - high_thr))
        total_sat = soft_lo.mean(dim=0) + soft_hi.mean(dim=0)
        sat_err = total_sat - target_saturation
        dsat_dr = -sat_k * soft_lo * (1.0 - soft_lo) + sat_k * soft_hi * (1.0 - soft_hi)
        dsat_dlogm = (dsat_dr * dr_dlogm).mean(dim=0)
        grad_log_m = grad_log_m + (2.0 * saturation_weight) * sat_err * dsat_dlogm
        if dr_db is not None and grad_b is not None:
            dsat_db = (dsat_dr * dr_db).mean(dim=0)
            grad_b = grad_b + (2.0 * saturation_weight) * sat_err * dsat_db

    return grad_log_m, grad_b


__all__ = [
    "compute_gate_homeostasis_aux_grads",
    "compute_inhibitory_homeostasis_factor",
    "compute_voltage_homeostasis_error",
]
