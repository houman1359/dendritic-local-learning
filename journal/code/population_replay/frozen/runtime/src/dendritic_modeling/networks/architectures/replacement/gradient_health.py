"""Gradient-flow diagnostics and the positive-pathway live-region re-init.

Measured failure (2026-08-24, Pythia-L2 budget): strict-positive cells
initialize their input pathways at ``pre_w`` means of -6.6 (input-to-E) and
-10.9 (input-to-I) under the softplus transform.  Two consequences follow
from ``d softplus(p)/dp = sigmoid(p) ~ w`` for small effective weights:

* the transform derivative at init is ~1e-3 to ~1e-5, so input pathways
  receive orders of magnitude less gradient than signed pathways (measured
  ~300x below ``signed_flat`` at matched budget); and
* even with Adam's per-coordinate rescaling, the init sits 4-9 units of
  ``pre_w`` from the live region, which at lr 3e-4 is a ~15k-step traverse —
  far beyond screen budgets.  Raising the learning rate does not rescue it:
  the readout (initialized IN the live region) destabilizes first, which is
  exactly what the lr-rescue screens measured.

VALIDATION (2026-08-24 evening): the init-only re-balance under RAW
ADDITIVE integration was FALSIFIED end-to-end — every E/I family trained
WORSE at the frozen L2 budget (+126% to +960%), because (a) the function is
genuinely flat in dead-zone parameters, so no forward-preserving move helps,
and (b) a common live-region floor destroys the analytical E-to-I weight
ratio under additive integration.  Signed-E/I families are affected too:
their Dale-law pathways are softplus.  Keep this flag OFF for raw-additive
cells; the surviving hypothesis pairs the re-balance with divisive
(conductance-normalized) integration, where the drive is scale-invariant.

The original design rationale, retained for the record: an INIT-ONLY change: re-initialize positive-transform
pathways inside the transform's live region, preserving topology and each
synapse's relative magnitude, and let the standard reactivation calibration
and local fit restore forward scale.  It is opt-in and recorded in the run
config, so frozen campaigns remain exactly reproducible.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import inverse_softplus

#: sigmoid(pre_w) at the live-region floor: softplus'(-1.1) ~ 0.25.
LIVE_REGION_MIN_DERIVATIVE = 0.25


def _transform_stats(
    module: IndexedSparseLinear,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pre = module.pre_w.detach()
    if module.weight_transform == "softplus":
        return F.softplus(pre), torch.sigmoid(pre)
    if module.weight_transform == "exp":
        weight = pre.exp()
        return weight, weight
    return None


def pathway_gradient_report(
    cell: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Per-indexed-layer gradient-flow report from one forward/backward.

    ``grad_over_param`` is the ratio of gradient norm to parameter norm — the
    effective relative step a unit learning rate would take — and
    ``mean_transform_derivative`` is the factor a positive transform applies
    to every gradient entering ``pre_w``.
    """

    was_training = cell.training
    cell.train()
    cell.zero_grad(set_to_none=True)
    loss = F.mse_loss(cell(inputs), targets)
    loss.backward()
    report: dict[str, dict[str, float]] = {}
    for name, module in cell.named_modules():
        if not isinstance(module, IndexedSparseLinear):
            continue
        grad = module.pre_w.grad
        entry: dict[str, float] = {
            "pre_w_mean": float(module.pre_w.detach().mean()),
            "grad_over_param": (
                float(grad.norm() / module.pre_w.detach().norm().clamp_min(1e-12))
                if grad is not None
                else 0.0
            ),
        }
        stats = _transform_stats(module)
        if stats is not None:
            weight, derivative = stats
            entry["effective_weight_mean"] = float(weight.mean())
            entry["mean_transform_derivative"] = float(derivative.mean())
        report[name] = entry
    cell.zero_grad(set_to_none=True)
    cell.train(was_training)
    return report


def rebalance_positive_pathways_(
    cell: nn.Module,
    *,
    target_weight: float = 0.5,
    relative_jitter: float = 0.2,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Re-initialize softplus pathways inside the transform's live region.

    For every ``IndexedSparseLinear`` whose transform is ``softplus`` and
    whose current mean derivative is below :data:`LIVE_REGION_MIN_DERIVATIVE`:
    each synapse's new effective weight is ``target_weight`` scaled by the
    synapse's CURRENT weight relative to its row mean (so any structure the
    analytical init encoded in relative magnitudes survives), clamped to a
    factor-of-four band, plus seeded multiplicative jitter; ``pre_w`` is set
    to the inverse transform of that target.  Topology is untouched.

    This deliberately changes the pathway's forward SCALE.  Callers must run
    the standard calibration (reactivation moments, teacher-conditioned
    readout fit) or local fit afterwards, which every trainer in this
    repository already does for fresh cells.
    """

    generator = torch.Generator().manual_seed(int(seed))
    report: dict[str, dict[str, Any]] = {}
    for name, module in cell.named_modules():
        if not isinstance(module, IndexedSparseLinear):
            continue
        if module.weight_transform != "softplus":
            continue
        stats = _transform_stats(module)
        assert stats is not None
        weight, derivative = stats
        before = float(derivative.mean())
        if before >= LIVE_REGION_MIN_DERIVATIVE:
            report[name] = {"rebalanced": False, "mean_derivative": before}
            continue
        row_mean = weight.mean(dim=1, keepdim=True).clamp_min(1e-12)
        relative = (weight / row_mean).clamp(0.25, 4.0)
        # The seeded generator lives on CPU so the draw is identical on any
        # device; move the jitter to the weights afterwards.
        jitter = 1.0 + relative_jitter * torch.randn(
            weight.shape, generator=generator
        ).to(weight.device)
        target = (float(target_weight) * relative * jitter).clamp_min(1e-3)
        with torch.no_grad():
            module.pre_w.copy_(inverse_softplus(target.to(module.pre_w.dtype)))
        after_stats = _transform_stats(module)
        assert after_stats is not None
        report[name] = {
            "rebalanced": True,
            "mean_derivative_before": before,
            "mean_derivative_after": float(after_stats[1].mean()),
            "pre_w_mean_before": float(inverse_softplus(weight).mean()),
            "pre_w_mean_after": float(module.pre_w.detach().mean()),
        }
    return report


__all__ = [
    "LIVE_REGION_MIN_DERIVATIVE",
    "pathway_gradient_report",
    "rebalance_positive_pathways_",
]
