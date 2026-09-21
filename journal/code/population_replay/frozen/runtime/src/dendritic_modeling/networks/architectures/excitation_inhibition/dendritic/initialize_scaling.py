from __future__ import annotations

from math import log, sqrt


def _center_preserving_shunting_conductances(
    E_g_exc: float,
    E_g_inh: float,
    target_conductance: float,
) -> tuple[float, float]:
    """Reduce total conductance while preserving the shunting voltage center."""

    E_g_exc = max(float(E_g_exc), 0.0)
    E_g_inh = max(float(E_g_inh), 0.0)
    total = E_g_exc + E_g_inh
    if E_g_exc <= 0.0 or total <= 0.0:
        return E_g_exc, E_g_inh

    center = E_g_exc / (total + 1.0)
    if center <= 0.0 or center >= 1.0:
        return E_g_exc, E_g_inh

    target_total = min(total, max(float(target_conductance), 1e-12))
    min_total_for_center = center / (1.0 - center)
    target_total = max(target_total, min_total_for_center)
    target_total = min(target_total, total)

    target_exc = center * (target_total + 1.0)
    target_inh = target_total - target_exc
    return max(target_exc, 0.0), max(target_inh, 0.0)


def compute_network_scale_factor(dbl, branch_factors=None, target_conductance=5.0):
    """
    Compute a scale factor based on the total network architecture.
    This prevents overflow when many branches accumulate in shunting computation.

    Args:
        dbl: DendriticBranchLayer to initialize
        branch_factors: List of branch factors from the parent network
        target_conductance: Conductance budget used by legacy scale mode

    Returns a scale factor to reduce initial weights in log space.
    """
    # Calculate total branches
    total_branches = 1

    if branch_factors:
        for bf in branch_factors:
            total_branches *= bf
    else:
        # Conservative estimate if branch factors not provided
        total_branches = 100

    # Calculate approximate total conductance
    k_exc = dbl.branch_excitation.K if dbl.input_excitatory else 0
    k_rec = dbl.branch_recurrent.K if getattr(dbl, "input_recurrent", False) else 0
    k_inh = dbl.branch_inhibition.K if dbl.input_inhibitory else 0
    k_rec_inh = (
        dbl.branch_rec_inhibition.K
        if getattr(dbl, "input_rec_inhibitory", False)
        else 0
    )

    # Total approximate conductance with all weights at 1
    # Scale by sqrt of branches for more conservative scaling
    total_conductance = (k_exc + k_rec + k_inh + k_rec_inh) * sqrt(total_branches)

    # Target a reasonable initial total conductance
    if total_conductance > target_conductance:
        scale_factor = log(total_conductance / target_conductance)
    else:
        scale_factor = 0.0

    return scale_factor


__all__ = [
    "_center_preserving_shunting_conductances",
    "compute_network_scale_factor",
]
