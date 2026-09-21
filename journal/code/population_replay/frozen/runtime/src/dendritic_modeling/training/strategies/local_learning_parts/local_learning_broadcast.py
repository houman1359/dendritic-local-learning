"""Pure broadcast / pathway-role helpers for local credit assignment.

Extracted from ``LocalCreditAssignment``. These are stateless: config is passed
explicitly as ``local_cfg`` and per-layer state as ``rec``/``stats`` arguments.
The stateful pieces (low-rank broadcast cache, path-propagation precompute) stay
on the trainer.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.nn import functional

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer  # noqa: F401  (parity)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)

logger = logging.getLogger(__name__)


def apply_broadcast_bandwidth(
    local_cfg, e_n: torch.Tensor, bandwidth: str
) -> torch.Tensor:
    """Apply bandwidth reduction to the broadcast error signal.

    Simulates biologically plausible low-bandwidth feedback channels.
    ``bandwidth`` is one of "sign_only", "quantized", "sparse_topk".
    """
    if bandwidth == "sign_only":
        magnitude = e_n.abs().mean()
        return torch.sign(e_n) * magnitude

    elif bandwidth == "quantized":
        bits = getattr(local_cfg, "broadcast_bits", 8)
        n_levels = 2**bits
        e_min = e_n.min()
        e_max = e_n.max()
        e_range = e_max - e_min + 1e-8
        quantized = torch.round((e_n - e_min) / e_range * (n_levels - 1))
        return quantized / (n_levels - 1) * e_range + e_min

    elif bandwidth == "sparse_topk":
        frac = getattr(local_cfg, "broadcast_topk_fraction", 0.3)
        k = max(1, int(e_n.shape[-1] * frac))
        _, topk_idx = torch.topk(e_n.abs(), k, dim=-1)
        mask = torch.zeros_like(e_n)
        mask.scatter_(-1, topk_idx, 1.0)
        return e_n * mask

    else:
        return e_n


def reduce_error_to_scalar(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reduce output error to a robust per-sample scalar broadcast signal."""
    if delta.dim() == 1:
        return delta.unsqueeze(-1)

    if delta.dim() != 2:
        delta = delta.view(delta.size(0), -1)

    delta_scalar = delta.mean(dim=1, keepdim=True)
    near_zero = delta_scalar.abs() <= eps
    if near_zero.any():
        max_idx = delta.abs().argmax(dim=1, keepdim=True)
        fallback = delta.gather(dim=1, index=max_idx)
        delta_scalar = torch.where(near_zero, fallback, delta_scalar)

    return delta_scalar


def expand_soma_error_to_compartments(
    delta: torch.Tensor,
    out_features: int,
) -> torch.Tensor | None:
    """Repeat each soma error over the compartments owned by that soma.

    Dendritic branch tensors are ordered soma-major: all descendants of soma
    ``u`` occupy one contiguous block. When the compartment width is an integer
    multiple of the soma-error width, repeating each coordinate preserves that
    ancestry. A non-divisible width is ambiguous and returns ``None`` so callers
    can use an explicit fallback rather than silently tile unrelated neurons.
    """
    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)
    elif delta.dim() != 2:
        delta = delta.reshape(delta.size(0), -1)

    n_soma = int(delta.size(1))
    if n_soma <= 0 or out_features <= 0 or out_features % n_soma != 0:
        return None
    compartments_per_soma = out_features // n_soma
    return delta.repeat_interleave(compartments_per_soma, dim=1)


def compute_local_mismatch_broadcast(
    *,
    delta_scalar: torch.Tensor,
    v_n: torch.Tensor,
    parent: torch.Tensor | None,
    residual_fraction: float = 0.2,
    clip_value: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a stabilized local-mismatch broadcast signal."""
    base = delta_scalar.expand(-1, v_n.size(1))
    if parent is None:
        return base

    p = parent
    if p.size() != v_n.size():
        try:
            p = p.view_as(v_n)
        except Exception:
            return base

    mismatch = p - v_n
    mismatch = mismatch - mismatch.mean(dim=0, keepdim=True)
    rms = mismatch.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    mismatch_norm = mismatch / rms
    if clip_value > 0.0:
        mismatch_norm = torch.clamp(mismatch_norm, -clip_value, clip_value)

    modulated = base * mismatch_norm
    blend = float(max(0.0, min(1.0, residual_fraction)))
    return (1.0 - blend) * modulated + blend * base


def compute_layer_total_conductance(
    rec: dict[str, Any], v_n: torch.Tensor
) -> torch.Tensor:
    """Compute total conductance used by conductance-aware local rules."""
    g_tot = torch.ones_like(v_n)

    exc_out = rec.get("exc_out")
    if exc_out is not None:
        g_tot = g_tot + functional.relu(exc_out)

    inh_out = rec.get("inh_out")
    if inh_out is not None:
        g_tot = g_tot + functional.relu(inh_out)

    blk_module = rec.get("blk_module")
    if blk_module is not None and hasattr(blk_module, "sum_conductances"):
        g_blk = blk_module.sum_conductances().detach()[None, :].expand_as(v_n)
        g_tot = g_tot + functional.relu(g_blk)

    return g_tot


def expand_parent_signal_to_children(
    signal: torch.Tensor, block_size: int, child_out_features: int
) -> torch.Tensor:
    """Broadcast a parent-layer signal to its grouped child branches."""
    expanded = signal.repeat_interleave(block_size, dim=1)
    if expanded.size(1) == child_out_features:
        return expanded

    if expanded.size(1) > child_out_features:
        return expanded[:, :child_out_features]

    repeats = math.ceil(float(child_out_features) / float(max(1, expanded.size(1))))
    tiled = expanded.repeat(1, repeats)
    return tiled[:, :child_out_features]


def normalize_role_mass(role_mass: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize non-negative role mass into a role profile."""
    denom = role_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    return role_mass / denom


def compute_role_selectivity(
    role_profile: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Measure pathway selectivity from normalized role profiles."""
    n_roles = role_profile.size(-1)
    if n_roles <= 1:
        return torch.ones(
            role_profile.size(0),
            device=role_profile.device,
            dtype=role_profile.dtype,
        )
    probs = role_profile.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    max_entropy = math.log(float(n_roles))
    return 1.0 - entropy / max(max_entropy, eps)


def resolve_recent_role_profile(
    input_dim: int,
    base_input_roles: torch.Tensor | None,
    prior_output_roles: list[torch.Tensor],
) -> torch.Tensor | None:
    """Resolve the most plausible role basis for a layer input size."""
    if (
        isinstance(base_input_roles, torch.Tensor)
        and base_input_roles.dim() == 2
        and base_input_roles.size(0) == input_dim
    ):
        return base_input_roles

    for role_profile in reversed(prior_output_roles):
        if role_profile.dim() == 2 and role_profile.size(0) == input_dim:
            return role_profile
    return None


def effective_topk_weight(layer: TopKLinear, mask: torch.Tensor | None) -> torch.Tensor:
    """Return masked effective synaptic weights."""
    eff_weight = layer.weight().detach()
    if mask is None:
        mask = layer.weight_mask().detach()
    return eff_weight * mask.to(device=eff_weight.device, dtype=eff_weight.dtype)


def role_mass_from_weight_matrix(
    weight_matrix: torch.Tensor,
    input_roles: torch.Tensor | None,
) -> torch.Tensor | None:
    """Aggregate pathway-role mass through a synaptic weight matrix."""
    if input_roles is None or input_roles.dim() != 2:
        return None
    if weight_matrix.dim() != 2 or weight_matrix.size(1) != input_roles.size(0):
        return None
    roles = input_roles.to(device=weight_matrix.device, dtype=weight_matrix.dtype)
    return weight_matrix.abs() @ roles


def resolve_router_output_roles(local_cfg, model: BaseModel) -> torch.Tensor | None:
    """Return pathway ownership for encoder outputs when available."""
    source = str(
        getattr(local_cfg.morphology_aware, "branch_role_source", "router_pathways")
    ).lower()
    if source != "router_pathways":
        logger.warning(
            "Unknown branch_role_source='%s'; disabling pathway-role inference.",
            source,
        )
        return None

    encoder = getattr(model, "encoder_network", None)
    if not isinstance(encoder, PathwayRouter):
        return None
    return encoder.get_output_role_matrix().detach()


def compute_pathway_activity(
    rec: dict[str, Any],
    stats: _LayerStats,
    v_n: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """Compute a sample-wise local pathway activity gate."""
    candidates: list[torch.Tensor] = []

    x_exc = rec.get("x_exc")
    if isinstance(x_exc, torch.Tensor) and isinstance(
        stats.input_role_profile, torch.Tensor
    ):
        if x_exc.size(1) == stats.input_role_profile.size(0):
            act_exc = x_exc.abs() @ stats.input_role_profile.to(
                device=x_exc.device, dtype=x_exc.dtype
            )
            candidates.append(act_exc)

    x_blk_raw = rec.get("x_blk_raw")
    if (
        isinstance(x_blk_raw, torch.Tensor)
        and isinstance(stats.block_input_role_profile, torch.Tensor)
        and x_blk_raw.size(1) == stats.block_input_role_profile.size(0)
    ):
        act_blk = x_blk_raw.abs() @ stats.block_input_role_profile.to(
            device=x_blk_raw.device, dtype=x_blk_raw.dtype
        )
        candidates.append(act_blk)

    if isinstance(stats.branch_role_profile, torch.Tensor):
        denom = stats.branch_role_profile.sum(dim=0, keepdim=True).clamp_min(eps)
        act_v = functional.relu(v_n) @ stats.branch_role_profile.to(
            device=v_n.device, dtype=v_n.dtype
        )
        candidates.append(act_v / denom.to(device=v_n.device, dtype=v_n.dtype))

    if not candidates:
        return None

    normalized: list[torch.Tensor] = []
    for candidate in candidates:
        denom = candidate.mean(dim=1, keepdim=True).clamp_min(eps)
        normalized.append(candidate / denom)
    return torch.stack(normalized, dim=0).mean(dim=0)


def compute_pathway_seed_projection(
    seed_error: torch.Tensor | None,
    delta_scalar: torch.Tensor,
    role_profile: torch.Tensor,
    out_features: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve pathway-space error and residual base for pathway broadcasts."""
    role_mass = role_profile.sum(dim=0, keepdim=True).clamp_min(eps)

    seed = seed_error
    if isinstance(seed, torch.Tensor):
        if seed.dim() == 1:
            seed = seed.unsqueeze(-1)
        elif seed.dim() != 2:
            seed = seed.view(seed.size(0), -1)

    if isinstance(seed, torch.Tensor) and seed.size(1) == out_features:
        seed = seed.to(device=role_profile.device, dtype=role_profile.dtype)
        pathway_error = seed @ role_profile
        pathway_error = pathway_error / role_mass
        return pathway_error, seed

    scalar = delta_scalar.to(device=role_profile.device, dtype=role_profile.dtype)
    pathway_error = scalar.expand(-1, role_profile.size(1))
    base = scalar.expand(-1, out_features)
    return pathway_error, base


__all__ = [
    "apply_broadcast_bandwidth",
    "compute_layer_total_conductance",
    "compute_local_mismatch_broadcast",
    "compute_pathway_activity",
    "compute_pathway_seed_projection",
    "compute_role_selectivity",
    "effective_topk_weight",
    "expand_parent_signal_to_children",
    "normalize_role_mass",
    "reduce_error_to_scalar",
    "resolve_recent_role_profile",
    "resolve_router_output_roles",
    "role_mass_from_weight_matrix",
]
