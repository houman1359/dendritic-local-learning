"""Pure STDP helper functions for local credit assignment.

Extracted from ``LocalCreditAssignment`` so the trace/activity math and the
pathway-enable predicates are unit-testable in isolation. These functions hold
no trainer state: config is passed in explicitly as ``local_cfg`` and tensors
are detached where the originals detached.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
from torch.nn import functional

logger = logging.getLogger(__name__)


def stdp_enabled(local_cfg) -> bool:
    """Return whether trace-based STDP updates should be applied."""
    stdp_cfg = getattr(local_cfg, "stdp", None)
    return bool(
        getattr(stdp_cfg, "enabled", False)
        or str(getattr(local_cfg, "rule_variant", "")).lower() == "stdp"
    )


def stdp_pathway_enabled(local_cfg, pathway: str) -> bool:
    """Return whether STDP is configured for a recorded synaptic pathway."""
    stdp_cfg = getattr(local_cfg, "stdp", None)
    apply_to = getattr(stdp_cfg, "apply_to", ("exc", "rec_exc"))
    if isinstance(apply_to, str):
        apply_to = (apply_to,)
    normalized = {str(item).strip().lower().replace("-", "_") for item in apply_to}
    return "all" in normalized or pathway in normalized


def stdp_activity(values: torch.Tensor, threshold: float, mode: str) -> torch.Tensor:
    """Convert raw local values to non-backpropagating STDP activities."""
    activity = values.detach()
    if activity.dim() != 2:
        activity = activity.reshape(activity.size(0), -1)
    centered = activity - float(threshold)
    mode = (mode or "relu").lower()
    if mode == "identity":
        return centered
    if mode == "binary":
        return (centered > 0).to(dtype=activity.dtype)
    if mode in {"abs", "absolute"}:
        return centered.abs()
    if mode != "relu":
        logger.warning("Unknown stdp.activity_mode=%r; using relu.", mode)
    return functional.relu(centered)


def stdp_decay(tau: float) -> float:
    """Single-step exponential decay for an STDP eligibility trace."""
    tau = max(float(tau), 1e-6)
    return math.exp(-1.0 / tau)


def stdp_trace_like(
    previous_trace: torch.Tensor | None,
    *,
    features: int,
    activity: torch.Tensor,
) -> torch.Tensor:
    """Return a compatible previous trace or initialize a zero trace."""
    expected_shape = (features,)
    if (
        previous_trace is None
        or previous_trace.shape != expected_shape
        or previous_trace.device != activity.device
        or previous_trace.dtype != activity.dtype
    ):
        return torch.zeros(features, device=activity.device, dtype=activity.dtype)
    return previous_trace


def stdp_weight_delta(
    *,
    pre_batch: torch.Tensor,
    post_batch: torch.Tensor,
    prev_pre: torch.Tensor,
    prev_post: torch.Tensor,
    a_plus: float,
    a_minus: float,
) -> torch.Tensor:
    """Compute the STDP weight update before pathway/error scaling."""
    potentiation = post_batch[:, None] * prev_pre[None, :]
    depression = prev_post[:, None] * pre_batch[None, :]
    return float(a_plus) * potentiation - float(a_minus) * depression


def stdp_apply_pathway_sign(
    delta_w: torch.Tensor,
    *,
    pathway: str,
    inhibitory_update_sign: float,
) -> torch.Tensor:
    """Apply the historical inhibitory pathway sign convention."""
    if pathway in {"inh", "rec_inh"}:
        return delta_w * float(inhibitory_update_sign)
    return delta_w


def stdp_error_scale(
    error_signal: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Reduce an error signal to the scalar STDP modulation factor."""
    mode = str(mode).lower()
    if mode == "scalar_signed":
        return error_signal.detach().mean()
    return error_signal.detach().abs().mean()


def stdp_apply_error_modulation(
    delta_w: torch.Tensor,
    *,
    error_signal: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Apply optional scalar error modulation to an STDP update."""
    if not isinstance(error_signal, torch.Tensor):
        return delta_w
    scale = stdp_error_scale(error_signal, mode=mode)
    return delta_w * scale.to(device=delta_w.device, dtype=delta_w.dtype)


def stdp_clamp_update(delta_w: torch.Tensor, clamp_update: float) -> torch.Tensor:
    """Clamp an STDP update when a positive clamp value is configured."""
    clamp_update = float(clamp_update)
    if clamp_update > 0.0:
        return torch.clamp(delta_w, -clamp_update, clamp_update)
    return delta_w


def stdp_next_trace(
    previous_trace: torch.Tensor,
    batch_trace: torch.Tensor,
    *,
    tau: float,
    detach: bool,
) -> torch.Tensor:
    """Advance one STDP eligibility trace by one batch."""
    decay = stdp_decay(tau)
    next_trace = decay * previous_trace + (1.0 - decay) * batch_trace
    if detach:
        return next_trace.detach()
    return next_trace


def weight_transform_derivative(
    raw_param: torch.Tensor,
    transform_type: str,
    *,
    warn_unknown: Callable[[str], None] | None = None,
) -> torch.Tensor:
    """Derivative of transformed conductance with respect to a raw parameter."""
    transform = (transform_type or "exp").lower()
    if transform == "exp":
        return raw_param.exp()
    if transform == "relu":
        return (raw_param > 0).to(raw_param.dtype)
    if transform == "softplus":
        return torch.sigmoid(raw_param)
    if transform == "identity":
        return torch.ones_like(raw_param)

    if warn_unknown is not None:
        warn_unknown(transform_type)
    return raw_param.exp()


__all__ = [
    "stdp_activity",
    "stdp_apply_error_modulation",
    "stdp_apply_pathway_sign",
    "stdp_clamp_update",
    "stdp_decay",
    "stdp_enabled",
    "stdp_error_scale",
    "stdp_next_trace",
    "stdp_pathway_enabled",
    "stdp_trace_like",
    "stdp_weight_delta",
    "weight_transform_derivative",
]
