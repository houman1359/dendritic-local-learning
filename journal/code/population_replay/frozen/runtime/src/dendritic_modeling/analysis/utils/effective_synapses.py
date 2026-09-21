"""Snapshots of the synaptic operator actually used by a sparse forward pass.

Analysis code must not confuse a layer's dense candidate parameters with its
realized sparse operator.  The helpers in this module centralize that contract
for standard, indexed, dynamic, and stochastic Top-K implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EffectiveSynapseSnapshot:
    """One internally consistent view of a sparse synaptic layer.

    ``candidate_weight`` is the transformed dense compatibility view,
    ``active_mask`` is the exact binary realization used to construct
    ``effective_weight``, and ``realized_k`` counts active inputs per output.
    For stochastic selectors the mask must come from a cached forward pass;
    sampling a fresh mask during analysis would not describe the analyzed
    computation.
    """

    candidate_weight: torch.Tensor
    candidate_mask: torch.Tensor
    active_mask: torch.Tensor
    effective_weight: torch.Tensor
    realized_k: torch.Tensor
    selection_policy: str
    mask_source: str


def _selection_is_stochastic(module: torch.nn.Module) -> bool:
    selection = str(getattr(module, "selection", "")).lower()
    class_name = type(module).__name__.lower()
    return (
        "stochastic" in class_name
        or selection in {"stochastic", "rank_probabilistic"}
        or float(getattr(module, "noise_level", 0.0) or 0.0) > 0.0
    )


def _candidate_mask(module: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    candidate_mask_fn = getattr(module, "candidate_mask", None)
    if callable(candidate_mask_fn):
        return candidate_mask_fn().to(device=reference.device, dtype=reference.dtype)

    mask = torch.ones_like(reference)
    connection_mask = getattr(module, "connection_mask", None)
    if isinstance(connection_mask, torch.Tensor) and connection_mask.numel() > 0:
        mask = mask * connection_mask.to(device=reference.device, dtype=reference.dtype)

    forbidden = getattr(module, "_forbidden_input_index_per_output", None)
    if isinstance(forbidden, torch.Tensor) and forbidden.numel() > 0:
        forbidden = forbidden.to(device=reference.device)
        rows = torch.arange(reference.shape[0], device=reference.device)
        valid = forbidden >= 0
        if bool(valid.any()):
            mask[rows[valid], forbidden[valid]] = 0
    return mask


def _effective_weight_from_mask(
    module: torch.nn.Module,
    candidate_weight: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    from_mask = getattr(module, "_pruned_weight_from_mask", None)
    if callable(from_mask):
        return from_mask(active_mask)

    effective = candidate_weight * active_mask
    norm_order = getattr(module, "weight_norm_order", None)
    if norm_order is None:
        return effective

    norm = torch.norm(effective, p=norm_order, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(effective.dtype).eps)
    gamma = torch.as_tensor(
        getattr(module, "gamma", 1.0),
        device=effective.device,
        dtype=effective.dtype,
    )
    normalized = (effective / safe_norm) * gamma
    return torch.where(norm > 0, normalized, effective)


def effective_synapse_snapshot(
    module: torch.nn.Module,
    *,
    prefer_cached_forward_mask: bool = True,
    detach: bool = True,
) -> EffectiveSynapseSnapshot:
    """Return the realized sparse operator for ``module``.

    Deterministic selectors may compute their current mask directly.  A
    stochastic selector must expose ``_last_forward_weight_mask`` from a
    forward executed with ``cache_mask=True``; otherwise this function raises
    rather than silently analyzing a different random realization.
    """

    weight_fn = getattr(module, "weight", None)
    mask_fn = getattr(module, "weight_mask", None)
    if not callable(weight_fn) or not callable(mask_fn):
        raise TypeError(
            f"{type(module).__name__} does not expose weight() and weight_mask()"
        )

    candidate_weight = weight_fn()
    cached_mask = getattr(module, "_last_forward_weight_mask", None)
    if prefer_cached_forward_mask and isinstance(cached_mask, torch.Tensor):
        active_mask = cached_mask.to(
            device=candidate_weight.device, dtype=candidate_weight.dtype
        )
        mask_source = "cached_forward"
    else:
        if _selection_is_stochastic(module):
            raise RuntimeError(
                "Stochastic synapse analysis requires the realized forward mask. "
                "Set cache_mask=True, execute the analyzed forward pass, then "
                "request the snapshot."
            )
        active_mask = mask_fn().to(
            device=candidate_weight.device, dtype=candidate_weight.dtype
        )
        mask_source = "deterministic_current"

    if active_mask.shape != candidate_weight.shape:
        raise ValueError(
            "Synaptic mask and candidate weight shapes differ: "
            f"{tuple(active_mask.shape)} vs {tuple(candidate_weight.shape)}"
        )

    active_mask = active_mask.clamp(0, 1)
    effective_weight = _effective_weight_from_mask(
        module,
        candidate_weight=candidate_weight,
        active_mask=active_mask,
    )
    candidate_mask = _candidate_mask(module, candidate_weight)
    realized_k = active_mask.ne(0).sum(dim=-1)
    selection_policy = str(getattr(module, "selection", type(module).__name__)).lower()

    if detach:
        candidate_weight = candidate_weight.detach()
        candidate_mask = candidate_mask.detach()
        active_mask = active_mask.detach()
        effective_weight = effective_weight.detach()
        realized_k = realized_k.detach()

    return EffectiveSynapseSnapshot(
        candidate_weight=candidate_weight,
        candidate_mask=candidate_mask,
        active_mask=active_mask,
        effective_weight=effective_weight,
        realized_k=realized_k,
        selection_policy=selection_policy,
        mask_source=mask_source,
    )


__all__ = ["EffectiveSynapseSnapshot", "effective_synapse_snapshot"]
