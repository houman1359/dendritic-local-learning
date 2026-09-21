"""Connectivity-motif analysis for recurrent E/I dendritic networks.

This module extends the recurrent inhibitory-specialization tooling to a more
general view of dendritic routing motifs across the local E/I circuit. It reads
trained synaptic weights from branch-layer checkpoints and summarizes how each
route distributes its weight across distal, middle, and somatic compartments of
the target population.

The core structural quantity is a route-level targeting index for each
source-target pair::

    B(src, tgt) = f_soma(src, tgt) - f_distal(src, tgt)

where ``f_level`` is the within-pair fraction of per-compartment mean synaptic
weight assigned to that dendritic level. Positive values indicate soma-biased
targeting; negative values indicate distal-biased targeting.

This is a structural routing descriptor. It does not by itself imply causal
dynamic gating, but it can reveal motifs that are consistent with output-gain
control, distal context modulation, or mixed target-dependent inhibition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

TARGETING_CLASS_ORDER = [
    "soma-biased",
    "mixed-targeting",
    "diffuse",
    "distal-biased",
]


@dataclass(frozen=True)
class ConnectivityRouteSpec:
    """Description of one dendritic connectivity route."""

    key: str
    population: str
    synapse_attr: str
    label: str
    source_label: str
    target_label: str
    temporal_role: str


ROUTE_SPECS: dict[str, ConnectivityRouteSpec] = {
    "e_ff_exc": ConnectivityRouteSpec(
        key="e_ff_exc",
        population="e_population",
        synapse_attr="branch_excitation",
        label="input→E",
        source_label="input",
        target_label="E",
        temporal_role="feedforward excitatory",
    ),
    "e_ff_inh": ConnectivityRouteSpec(
        key="e_ff_inh",
        population="e_population",
        synapse_attr="branch_inhibition",
        label="ff inh→E",
        source_label="inh_ff",
        target_label="E",
        temporal_role="feedforward inhibitory",
    ),
    "e_rec_exc": ConnectivityRouteSpec(
        key="e_rec_exc",
        population="e_population",
        synapse_attr="branch_recurrent",
        label="E(t-1)→E",
        source_label="E",
        target_label="E",
        temporal_role="recurrent excitatory",
    ),
    "e_rec_inh": ConnectivityRouteSpec(
        key="e_rec_inh",
        population="e_population",
        synapse_attr="branch_rec_inhibition",
        label="I(t-1)→E",
        source_label="I",
        target_label="E",
        temporal_role="recurrent inhibitory",
    ),
    "i_ff_exc": ConnectivityRouteSpec(
        key="i_ff_exc",
        population="i_population",
        synapse_attr="branch_excitation",
        label="input→I",
        source_label="input",
        target_label="I",
        temporal_role="feedforward excitatory",
    ),
    "i_ff_inh": ConnectivityRouteSpec(
        key="i_ff_inh",
        population="i_population",
        synapse_attr="branch_inhibition",
        label="ff inh→I",
        source_label="inh_ff",
        target_label="I",
        temporal_role="feedforward inhibitory",
    ),
    "i_rec_exc": ConnectivityRouteSpec(
        key="i_rec_exc",
        population="i_population",
        synapse_attr="branch_recurrent",
        label="E(t-1)→I",
        source_label="E",
        target_label="I",
        temporal_role="recurrent excitatory",
    ),
    "i_rec_inh": ConnectivityRouteSpec(
        key="i_rec_inh",
        population="i_population",
        synapse_attr="branch_rec_inhibition",
        label="I(t-1)→I",
        source_label="I",
        target_label="I",
        temporal_role="recurrent inhibitory",
    ),
}


LOCAL_RECURRENT_ROUTE_KEYS = [
    "e_rec_exc",
    "e_rec_inh",
    "i_rec_exc",
    "i_rec_inh",
]


def _apply_weight_transform(
    pre_w: torch.Tensor,
    weight_transform: str,
) -> torch.Tensor:
    transform = weight_transform.lower()
    if transform == "exp":
        return torch.exp(pre_w)
    if transform == "softplus":
        return torch.nn.functional.softplus(pre_w)
    if transform == "relu":
        return torch.relu(pre_w)
    if transform == "identity":
        return pre_w
    raise ValueError(f"Unsupported weight_transform: {weight_transform}")


def _reshape_weight_matrix_by_compartment(
    w: torch.Tensor,
    n_target: int,
) -> torch.Tensor | None:
    """Reshape ``(n_target * K_l, n_source)`` to ``(n_target, K_l, n_source)``."""
    out_features, _n_source = w.shape
    if n_target <= 0 or out_features % n_target != 0:
        return None
    branch_factor = out_features // n_target
    if branch_factor == 0:
        return None
    return w.reshape(n_target, branch_factor, -1)


def _compartment_mean_level_weights(
    compartment_weights: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Average each level over its branch count to compare levels fairly."""
    return [w.sum(dim=1) / float(w.shape[1]) for w in compartment_weights]


def _compute_targeting_index(
    per_level: list[torch.Tensor],
    eps: float = 1e-12,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute per-pair soma-minus-distal targeting index and level fractions."""
    if len(per_level) < 2:
        raise ValueError("Need at least two levels to compute targeting index")
    stacked = torch.stack(per_level, dim=0)  # (L, n_target, n_source)
    total = stacked.sum(dim=0).clamp(min=eps)
    level_fracs = [stacked[i] / total for i in range(stacked.shape[0])]
    f_distal = level_fracs[0]
    f_soma = level_fracs[-1]
    bias = (f_soma - f_distal).T  # (n_source, n_target)
    return bias, level_fracs


def _classify_sources(
    bias: torch.Tensor,
    hi: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Classify each source unit by how it distributes targeting across targets."""
    mean_bias = bias.mean(dim=1)
    std_bias = bias.std(dim=1, unbiased=False)
    classes: list[str] = []
    for m, s in zip(mean_bias.tolist(), std_bias.tolist()):
        if s > sigma:
            classes.append("mixed-targeting")
        elif m > hi:
            classes.append("soma-biased")
        elif m < -hi:
            classes.append("distal-biased")
        else:
            classes.append("diffuse")
    return mean_bias, std_bias, classes


def _pair_level_stats(
    bias: torch.Tensor,
    hi: float,
) -> dict[str, float]:
    n = bias.numel()
    if n == 0:
        return {
            "strong_soma_pct": 0.0,
            "strong_distal_pct": 0.0,
            "balanced_pct": 0.0,
            "any_soma_bias_pct": 0.0,
            "n_pairs": 0,
        }
    return {
        "strong_soma_pct": float((bias >= hi).float().mean().item() * 100.0),
        "strong_distal_pct": float((bias <= -hi).float().mean().item() * 100.0),
        "balanced_pct": float(
            ((bias > -hi) & (bias < hi)).float().mean().item() * 100.0
        ),
        "any_soma_bias_pct": float((bias > 0).float().mean().item() * 100.0),
        "n_pairs": int(n),
    }


def pair_stats_by_class(
    bias: torch.Tensor,
    classes: list[str],
    hi: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Per-class summary of the pair-level targeting-index distribution."""
    out: dict[str, dict[str, float]] = {}
    for cname in TARGETING_CLASS_ORDER:
        idx = [i for i, c in enumerate(classes) if c == cname]
        if not idx:
            out[cname] = {"n_source": 0, "n_pairs": 0}
            continue
        vals = bias[idx].reshape(-1)
        out[cname] = {
            "n_source": len(idx),
            "n_pairs": int(vals.numel()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "q10": float(torch.quantile(vals, 0.10).item()),
            "q50": float(torch.quantile(vals, 0.50).item()),
            "q90": float(torch.quantile(vals, 0.90).item()),
            "strong_soma_pct": float((vals >= hi).float().mean().item() * 100.0),
            "strong_distal_pct": float((vals <= -hi).float().mean().item() * 100.0),
            "balanced_pct": float(
                ((vals > -hi) & (vals < hi)).float().mean().item() * 100.0
            ),
        }
    return out


def extract_compartment_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    synapse_attr: str,
    n_target: int,
    weight_transform: str = "exp",
    active_masks: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Extract realized per-level weights as ``(n_target, K_l, n_source)``.

    A state dict stores candidate parameters, so callers must supply the exact
    active masks from the analyzed model/forward realization.
    """
    if active_masks is None:
        raise ValueError(
            "active_masks are required: checkpoint pre_w tensors are dense "
            "candidate parameters, not realized synapses"
        )
    level_tensors: list[torch.Tensor] = []
    level = 0
    while True:
        key = f"{prefix}.branch_layers.{level}.{synapse_attr}.pre_w"
        if key not in state_dict:
            break
        pre_w = state_dict[key].detach().cpu().float()
        if level >= len(active_masks):
            raise ValueError(f"Missing active mask for route level {level}")
        mask = active_masks[level].detach().cpu().to(dtype=pre_w.dtype)
        if mask.shape != pre_w.shape:
            raise ValueError(
                f"Mask shape at level {level} is {tuple(mask.shape)}, expected "
                f"{tuple(pre_w.shape)}"
            )
        w = _apply_weight_transform(pre_w, weight_transform) * mask
        compartment_w = _reshape_weight_matrix_by_compartment(w, n_target=n_target)
        if compartment_w is None:
            raise ValueError(
                f"Could not reshape weights at level {level} for n_target={n_target}"
            )
        level_tensors.append(compartment_w)
        level += 1
    if not level_tensors:
        raise ValueError(
            f"No weights found under prefix '{prefix}' and synapse '{synapse_attr}'"
        )
    return level_tensors


def summarize_compartment_targeting_from_compartment_weights(
    compartment_weights: list[torch.Tensor],
    hi: float = 0.1,
    sigma: float = 0.15,
) -> dict[str, Any]:
    """Summarize generic soma-vs-distal targeting from compartment weights."""
    if not compartment_weights:
        raise ValueError("Expected at least one level of compartment weights")
    n_target = int(compartment_weights[0].shape[0])
    n_source = int(compartment_weights[0].shape[2])
    per_level = _compartment_mean_level_weights(compartment_weights)
    bias, level_fracs = _compute_targeting_index(per_level)
    mean_bias, std_bias, classes = _classify_sources(bias, hi=hi, sigma=sigma)
    class_counts = {c: int(classes.count(c)) for c in TARGETING_CLASS_ORDER}
    frac_by_level = [lvl.T for lvl in level_fracs]
    return {
        "n_target": n_target,
        "n_source": n_source,
        "n_levels": len(level_fracs),
        "targeting_index": bias,
        "mean_bias": mean_bias,
        "std_bias": std_bias,
        "classes": classes,
        "class_counts": class_counts,
        "pair_level": _pair_level_stats(bias, hi=hi),
        "level_fracs": level_fracs,
        "frac_by_level_src_target": frac_by_level,
        "frac_distal": frac_by_level[0],
        "frac_middle": (
            torch.stack(frac_by_level[1:-1], dim=0).mean(dim=0)
            if len(frac_by_level) > 2
            else torch.zeros_like(frac_by_level[0])
        ),
        "frac_soma": frac_by_level[-1],
        "frac_distal_pop": float(level_fracs[0].mean().item()),
        "frac_middle_pop": (
            float(torch.stack(level_fracs[1:-1], dim=0).mean().item())
            if len(level_fracs) > 2
            else 0.0
        ),
        "frac_soma_pop": float(level_fracs[-1].mean().item()),
    }


def analyze_ei_connectivity_motifs_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    layer_prefix: str,
    n_E: int,
    n_I: int,
    weight_transform: str = "exp",
    route_keys: list[str] | None = None,
    hi: float = 0.1,
    sigma: float = 0.15,
    active_masks_by_route: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Analyze all requested routes present in a checkpoint state dict."""
    if active_masks_by_route is None:
        raise ValueError(
            "active_masks_by_route is required for realized connectivity analysis"
        )
    if route_keys is None:
        route_keys = list(ROUTE_SPECS.keys())

    summaries: dict[str, dict[str, Any]] = {}
    for route_key in route_keys:
        spec = ROUTE_SPECS[route_key]
        n_target = n_E if spec.target_label == "E" else n_I
        prefix = f"{layer_prefix}.{spec.population}"
        try:
            route_masks = active_masks_by_route.get(route_key)
            if route_masks is None:
                continue
            compartment_weights = extract_compartment_weights_from_state_dict(
                state_dict,
                prefix=prefix,
                synapse_attr=spec.synapse_attr,
                n_target=n_target,
                weight_transform=weight_transform,
                active_masks=route_masks,
            )
        except ValueError:
            continue
        summary = summarize_compartment_targeting_from_compartment_weights(
            compartment_weights,
            hi=hi,
            sigma=sigma,
        )
        summaries[route_key] = {
            "route_key": route_key,
            "label": spec.label,
            "source_label": spec.source_label,
            "target_label": spec.target_label,
            "temporal_role": spec.temporal_role,
            "n_compartments_per_level": [int(w.shape[1]) for w in compartment_weights],
            **summary,
        }
    return summaries


__all__ = [
    "LOCAL_RECURRENT_ROUTE_KEYS",
    "ROUTE_SPECS",
    "TARGETING_CLASS_ORDER",
    "ConnectivityRouteSpec",
    "analyze_ei_connectivity_motifs_from_state_dict",
    "extract_compartment_weights_from_state_dict",
    "pair_stats_by_class",
    "summarize_compartment_targeting_from_compartment_weights",
]
