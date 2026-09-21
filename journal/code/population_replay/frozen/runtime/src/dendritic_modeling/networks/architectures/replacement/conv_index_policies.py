"""Index-selection policies for conv-boundary dendritic replacement cores.

A :class:`HierarchicalDendriticConv` evaluates one shared E/I network on every
unfolded patch, so its only spatial structure is the unfold window; *which*
patch coordinates ``(c, ky, kx)`` each branch contacts is, by default, a
seeded uniform draw.  The policies here condition that support on the teacher
conv layer instead, and emit external index manifests consumed through the
default-off ``method: index_manifest`` structured-connectivity hook
(``synapse/index_manifest.py``).

Patch coordinate layout matches ``F.unfold`` and ``Conv2d.weight.flatten(1)``:
``flat = c * (kh * kw) + ky * kw + kx``.

Row layout matches DendriNet construction order (owner-major):
``row = owner * branches_per_owner + local_branch``, where ``owner`` is the
soma index (one excitatory soma per output channel) and levels are constructed
distal-first (``level_idx = 0`` is the widest level).

Policies
--------
``teacher_topk``
    Row-conditioned hard top-K.  For each excitatory soma (output channel
    ``o``), rank patch coordinates by ``|W_teacher[o]|`` and deal the top
    ``branches * K`` coordinates round-robin across that soma's branches, so
    the per-soma union is exactly the teacher's strongest support and branches
    stay disjoint.  Inhibitory somas have no teacher row and share the
    aggregate ranking ``sum_o |W_teacher[o]|``.

``importance_sampled``
    Channel-importance-aware random support.  Coordinates are drawn without
    replacement, per row, with probability proportional to
    ``(sum_o |W[o, c, ky, kx]|) * E[|x_c|]``, where ``E[|x_c|]`` comes from a
    short calibration pass over the teacher's input feature map.  A soft prior
    rather than a hard cut: rows stay diverse and every coordinate keeps an
    epsilon floor.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.index_manifest import (
    manifest_entry_key,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset

__all__ = [
    "CONV_INDEX_POLICIES",
    "build_conv_dendritic_index_manifest",
    "dendrinet_feedforward_levels",
    "importance_sampled_level_indices",
    "teacher_topk_level_indices",
]

CONV_INDEX_POLICIES = ("teacher_topk", "importance_sampled")


def dendrinet_feedforward_levels(
    n_soma: int,
    branch_factors: list[int] | tuple[int, ...],
) -> list[dict[str, int]]:
    """Describe the feedforward branch levels of one DendriNet population.

    Mirrors ``_plan_dendrinet_branch_layout``: construction order is
    distal-first, level ``i`` has ``n_soma * prod(branch_factors) /
    prod(branch_factors[:i])`` rows.  The soma level is excluded (the
    validated conv recipes run ``somatic_synapses: false``).
    """

    factors = [int(f) for f in branch_factors]
    if int(n_soma) < 1 or any(f < 1 for f in factors):
        raise ValueError("n_soma and branch_factors must be positive")
    total = int(n_soma) * math.prod(factors)
    levels = []
    consumed = 1
    for level_idx in range(len(factors)):
        rows = total // consumed
        levels.append(
            {
                "level_idx": level_idx,
                "rows": rows,
                "branches_per_owner": rows // int(n_soma),
            }
        )
        consumed *= factors[level_idx]
    return levels


def teacher_topk_level_indices(
    scores_per_owner: torch.Tensor,
    *,
    branches_per_owner: int,
    synapses_per_branch: int,
) -> torch.Tensor:
    """Deal each owner's top ``branches * K`` coordinates across its branches.

    Args:
        scores_per_owner: ``[owners, in_features]`` non-negative scores.
        branches_per_owner: branches each owner holds at this level.
        synapses_per_branch: contacts per branch (K).

    Returns:
        ``[owners * branches_per_owner, synapses_per_branch]`` long tensor in
        owner-major row order; rank ``r`` of an owner's descending score order
        goes to branch ``r % branches`` (so branch supports partition the
        owner's top set and remain disjoint), sorted ascending within rows.
    """

    if scores_per_owner.dim() != 2:
        raise ValueError("scores_per_owner must be [owners, in_features]")
    owners, in_features = scores_per_owner.shape
    branches = int(branches_per_owner)
    k = int(synapses_per_branch)
    if branches < 1 or k < 1:
        raise ValueError("branches_per_owner and synapses_per_branch must be >= 1")
    top = branches * k
    if top > in_features:
        raise ValueError(
            f"branches_per_owner * synapses_per_branch = {top} exceeds "
            f"in_features = {in_features}"
        )
    # Deterministic ranking with index-ascending tie-break.
    order = torch.argsort(
        scores_per_owner.to(torch.float64), dim=1, descending=True, stable=True
    )[:, :top]
    # order: [owners, branches * k]; rank r -> branch r % branches.
    dealt = order.view(owners, k, branches).permute(0, 2, 1)
    dealt = dealt.reshape(owners * branches, k)
    return dealt.sort(dim=1).values.to(torch.long)


def importance_sampled_level_indices(
    scores: torch.Tensor,
    *,
    rows: int,
    synapses_per_branch: int,
    generator: torch.Generator,
    epsilon_fraction: float = 1e-6,
) -> torch.Tensor:
    """Sample per-row unique coordinates with probability proportional to score.

    Args:
        scores: ``[in_features]`` non-negative importance scores.
        rows: number of branch rows to sample.
        synapses_per_branch: contacts per row (K), sampled without replacement.
        generator: seeded CPU generator (determinism contract).
        epsilon_fraction: floor added as ``epsilon_fraction * max(score)`` so
            zero-scored coordinates stay reachable and the draw is well posed.

    Returns:
        ``[rows, synapses_per_branch]`` long tensor, sorted ascending per row.
    """

    if scores.dim() != 1:
        raise ValueError("scores must be a 1D [in_features] tensor")
    if bool((scores < 0).any()):
        raise ValueError("scores must be non-negative")
    in_features = scores.shape[0]
    k = int(synapses_per_branch)
    if k > in_features:
        raise ValueError(
            f"synapses_per_branch = {k} exceeds in_features = {in_features}"
        )
    probs = scores.to(torch.float64)
    peak = float(probs.max())
    if peak <= 0.0:
        probs = torch.ones_like(probs)
    else:
        probs = probs + peak * float(epsilon_fraction)
    matrix = probs.unsqueeze(0).expand(int(rows), in_features)
    drawn = torch.multinomial(matrix, k, replacement=False, generator=generator)
    return drawn.sort(dim=1).values.to(torch.long)


def _flat_abs_teacher_weight(teacher_weight: torch.Tensor) -> torch.Tensor:
    if teacher_weight.dim() != 4:
        raise ValueError(
            "teacher_weight must be a Conv2d weight [out, in, kh, kw], got "
            f"shape {tuple(teacher_weight.shape)}"
        )
    return teacher_weight.detach().abs().flatten(1).to(torch.float64).cpu()


def build_conv_dendritic_index_manifest(
    teacher_weight: torch.Tensor,
    *,
    policy: str,
    excitatory_somas: int,
    excitatory_branch_factors: list[int] | tuple[int, ...],
    ee_synapses_per_branch: int,
    inhibitory_somas: int = 0,
    inhibitory_branch_factors: list[int] | tuple[int, ...] = (),
    ei_synapses_per_branch: int = 0,
    channel_activation: torch.Tensor | None = None,
    seed: int = 0,
    layer_idx: int = 0,
) -> dict[str, dict[str, Any]]:
    """Build manifest entries for one conv-boundary dendritic core.

    Covers the two feedforward-patch pathways of the replacement recipe:
    ``ee`` (excitatory somas <- patch) and, when ``inhibitory_somas > 0`` with
    ``ei_synapses_per_branch > 0``, ``ei`` (inhibitory somas <- patch).  The
    ``ie`` pathway samples the local inhibitory population, which has no
    teacher analogue, and is deliberately left to the default random sampler.

    Returns entries suitable for
    :func:`dendritic_modeling...index_manifest.save_index_manifest`.
    """

    if policy not in CONV_INDEX_POLICIES:
        raise ValueError(
            f"Unknown policy {policy!r}; choose from {CONV_INDEX_POLICIES}"
        )
    flat_abs = _flat_abs_teacher_weight(teacher_weight)
    out_channels, in_features = flat_abs.shape
    if int(excitatory_somas) != out_channels:
        raise ValueError(
            f"excitatory_somas = {excitatory_somas} must equal the teacher's "
            f"out_channels = {out_channels} (one soma per output channel)"
        )
    aggregate = flat_abs.sum(dim=0)

    if policy == "importance_sampled":
        if channel_activation is None:
            raise ValueError(
                "policy='importance_sampled' requires channel_activation "
                "statistics (E[|x_c|]) from a calibration pass"
            )
        channel_activation = channel_activation.detach().to(torch.float64).cpu()
        in_channels = int(teacher_weight.shape[1])
        if channel_activation.shape != (in_channels,):
            raise ValueError(
                f"channel_activation must be [{in_channels}], got "
                f"{tuple(channel_activation.shape)}"
            )
        if bool((channel_activation < 0).any()):
            raise ValueError("channel_activation must be non-negative")
        window = in_features // in_channels
        importance = (
            aggregate.view(in_channels, window) * channel_activation.unsqueeze(1)
        ).flatten()

    entries: dict[str, dict[str, Any]] = {}

    def _fill_pathway(
        pathway: str,
        n_soma: int,
        branch_factors,
        synapses_per_branch: int,
        owner_scores: torch.Tensor | None,
    ) -> None:
        for level in dendrinet_feedforward_levels(n_soma, branch_factors):
            key = manifest_entry_key(pathway, layer_idx, level["level_idx"])
            if policy == "teacher_topk":
                indices = teacher_topk_level_indices(
                    owner_scores,
                    branches_per_owner=level["branches_per_owner"],
                    synapses_per_branch=synapses_per_branch,
                )
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    (
                        int(seed)
                        + stable_seed_offset(
                            "conv_index_manifest",
                            pathway,
                            int(layer_idx),
                            int(level["level_idx"]),
                        )
                    )
                    % ((1 << 63) - 1)
                )
                indices = importance_sampled_level_indices(
                    importance,
                    rows=level["rows"],
                    synapses_per_branch=synapses_per_branch,
                    generator=generator,
                )
            entries[key] = {"indices": indices, "in_features": int(in_features)}

    _fill_pathway(
        "ee",
        int(excitatory_somas),
        excitatory_branch_factors,
        int(ee_synapses_per_branch),
        flat_abs if policy == "teacher_topk" else None,
    )
    if int(inhibitory_somas) > 0 and int(ei_synapses_per_branch) > 0:
        _fill_pathway(
            "ei",
            int(inhibitory_somas),
            inhibitory_branch_factors,
            int(ei_synapses_per_branch),
            (
                aggregate.unsqueeze(0).expand(int(inhibitory_somas), -1)
                if policy == "teacher_topk"
                else None
            ),
        )
    return entries
