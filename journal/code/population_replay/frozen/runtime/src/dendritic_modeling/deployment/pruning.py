"""Post-training pruning of dendritic indexed synapses.

Indexed dendritic layers store only active contacts, so classical mask-based
pruning does not apply; pruning here means reducing each output row's fixed
contact count K to a smaller K' by keeping the largest-magnitude *applied*
weights (after the positive/signed weight transform — magnitude in applied
space is the quantity that enters the forward sum). Per-row uniform K'
preserves the padded-dense ``[out, K']`` layout, so every kernel, exporter,
and resource-accounting path keeps working unchanged.

The intended recipe mirrors the classical magnitude-prune literature
(prune -> short recovery fine-tune -> repeat):

1. ``prune_model_indexed_(model, keep_fraction)`` or the
   ``dendritic-prune-checkpoint`` CLI produces a pruned checkpoint plus a
   config patch (the new per-pathway contact counts).
2. Recovery-train from the pruned checkpoint via ``model.initial_checkpoint``
   with the patched connectivity counts.
3. Repeat at a lower keep fraction while quality holds.

Group-shared supports (``support_group_rows`` > 1) are pruned per group so
tile structure survives.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import floor
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import apply_weight_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedPruningTarget:
    """Frozen structural target for one indexed projection.

    ``new_k`` is the exact retained contact count per output row.  The target
    is resolved before any module is mutated, which makes a pruning rung
    inspectable and prevents earlier replacements from changing how later
    path selectors are interpreted.
    """

    path: str
    k_before: int
    new_k: int
    in_features: int
    out_features: int
    target_density: float
    selector: str


def _structured_keep_columns(
    magnitudes: torch.Tensor,
    *,
    new_k: int,
    support_group_rows: int,
    support_col_block: int,
) -> torch.Tensor:
    """Select topology columns without violating row or column-block sharing."""

    row_group = max(1, int(support_group_rows))
    col_block = max(1, int(support_col_block))
    if col_block > 1 and new_k % col_block:
        raise ValueError(
            f"new_k={new_k} must be divisible by support_col_block={col_block}"
        )
    n_groups = (magnitudes.shape[0] + row_group - 1) // row_group
    keep_rows: list[torch.Tensor] = []
    for group_index in range(n_groups):
        rows = magnitudes[group_index * row_group : (group_index + 1) * row_group]
        scores = rows.sum(dim=0)
        if col_block > 1:
            if scores.numel() % col_block:
                raise ValueError(
                    "Existing K must be divisible by support_col_block before "
                    "block-preserving pruning"
                )
            block_scores = scores.reshape(-1, col_block).sum(dim=1)
            block_ids = block_scores.topk(new_k // col_block).indices.sort().values
            offsets = torch.arange(col_block, device=block_ids.device)
            keep = (block_ids[:, None] * col_block + offsets).reshape(-1)
        else:
            keep = scores.topk(new_k).indices.sort().values
        keep_rows.append(keep)
    return torch.stack(keep_rows).repeat_interleave(row_group, dim=0)[
        : magnitudes.shape[0]
    ]


def prune_indexed_layer(
    layer: IndexedSparseLinear,
    new_k: int,
) -> tuple[IndexedSparseLinear, dict[str, float]]:
    """Return a copy of ``layer`` keeping the top-``new_k`` applied weights per row."""

    if new_k < 1 or new_k > layer.K:
        raise ValueError(f"new_k must be in [1, {layer.K}], got {new_k}")

    with torch.no_grad():
        applied = apply_weight_transform(layer.pre_w, layer.weight_transform)
        magnitudes = applied.abs()
        group = max(1, int(getattr(layer, "support_group_rows", 1)))
        col_block = max(1, int(getattr(layer, "support_col_block", 1)))
        keep = _structured_keep_columns(
            magnitudes,
            new_k=new_k,
            support_group_rows=group,
            support_col_block=col_block,
        )

        kept_indices = layer.connection_indices.gather(1, keep.to(torch.long))
        kept_pre_w = layer.pre_w.gather(1, keep.to(torch.long))
        kept_energy = applied.gather(1, keep.to(torch.long)).pow(2).sum()
        total_energy = applied.pow(2).sum().clamp_min(1e-12)

    pruned = IndexedSparseLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        K=new_k,
        param_space=layer.param_space,
        init_method=layer.init_method,
        init_gain=layer.init_gain,
        weight_transform=layer.weight_transform,
        weight_norm_order=layer.weight_norm_order,
        gamma=layer.gamma,
        connection_indices=kept_indices.detach().cpu(),
        output_chunk_size=layer.output_chunk_size,
        index_dtype="auto",
        workspace_mb=layer.workspace_mb,
        cache_transformed_weights=layer.cache_transformed_weights,
        recompute_backward=layer.recompute_backward,
        projection_backend=layer.projection_backend,
        persistent_indices=layer.persistent_indices,
        init_mode=layer.init_mode,
        support_group_rows=getattr(layer, "support_group_rows", 1),
        support_col_block=getattr(layer, "support_col_block", 1),
    ).to(device=layer.pre_w.device, dtype=layer.pre_w.dtype)
    with torch.no_grad():
        pruned.connection_indices.copy_(
            kept_indices.to(pruned.connection_indices.dtype)
        )
        pruned.pre_w.copy_(kept_pre_w)
    pruned.pre_w.requires_grad_(layer.pre_w.requires_grad)

    diagnostics = {
        "k_before": int(layer.K),
        "k_after": int(new_k),
        "applied_weight_energy_retained": float(kept_energy / total_energy),
    }
    return pruned, diagnostics


def resolve_indexed_pruning_targets(
    model: nn.Module,
    *,
    default_density: float | None = None,
    path_densities: Mapping[str, float] | None = None,
    path_contacts: Mapping[str, int] | None = None,
    min_k: int = 1,
) -> list[IndexedPruningTarget]:
    """Resolve one prospective, path-aware pruning rung without changing ``model``.

    Selectors use shell-style paths (for example ``"*.gate_core.*"``). Exact
    contact selectors take precedence over density selectors, which take
    precedence over ``default_density``.  A selected target may only remove
    contacts; increasing K would require topology growth and is rejected.
    """

    if min_k < 1:
        raise ValueError("min_k must be >= 1")
    density_rules = dict(path_densities or {})
    contact_rules = dict(path_contacts or {})
    for selector, density in density_rules.items():
        if not 0.0 < float(density) <= 1.0:
            raise ValueError(f"Density for selector {selector!r} must be in (0, 1]")
    if default_density is not None and not 0.0 < float(default_density) <= 1.0:
        raise ValueError("default_density must be in (0, 1]")

    def _single_match(path: str, rules: Mapping[str, Any]) -> tuple[str, Any] | None:
        matches = [
            (selector, value)
            for selector, value in rules.items()
            if fnmatch.fnmatchcase(path, selector)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Projection {path!r} matches multiple pruning selectors: "
                f"{[selector for selector, _ in matches]}"
            )
        return matches[0] if matches else None

    targets: list[IndexedPruningTarget] = []
    matched_selectors: set[str] = set()
    for path, module in model.named_modules():
        if not isinstance(module, IndexedSparseLinear):
            continue
        contact_match = _single_match(path, contact_rules)
        density_match = _single_match(path, density_rules)
        if contact_match is not None:
            selector, value = contact_match
            matched_selectors.add(selector)
            requested = int(value)
        elif density_match is not None:
            selector, value = density_match
            matched_selectors.add(selector)
            requested = floor(float(value) * int(module.in_features))
        elif default_density is not None:
            selector = "<default_density>"
            requested = floor(float(default_density) * int(module.in_features))
        else:
            continue
        col_block = max(1, int(getattr(module, "support_col_block", 1)))
        requested = max(int(min_k), int(requested))
        if col_block > 1:
            requested = max(col_block, (requested // col_block) * col_block)
        if requested > int(module.K):
            raise ValueError(
                f"Pruning target for {path!r} requests K={requested}, but current "
                f"K={module.K}; pruning cannot grow topology"
            )
        targets.append(
            IndexedPruningTarget(
                path=path,
                k_before=int(module.K),
                new_k=int(requested),
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                target_density=float(requested / module.in_features),
                selector=selector,
            )
        )
    unmatched = (set(density_rules) | set(contact_rules)) - matched_selectors
    if unmatched:
        raise ValueError(
            f"Pruning selectors matched no indexed projection: {sorted(unmatched)}"
        )
    if not targets:
        raise ValueError("Pruning rung selected no IndexedSparseLinear projections")
    return targets


def apply_indexed_pruning_targets_(
    model: nn.Module,
    targets: Sequence[IndexedPruningTarget],
) -> dict[str, dict[str, float]]:
    """Apply a previously resolved pruning rung exactly and return diagnostics."""

    target_by_path = {target.path: target for target in targets}
    if len(target_by_path) != len(targets):
        raise ValueError("Indexed pruning targets must have unique module paths")
    modules = dict(model.named_modules())
    report: dict[str, dict[str, float]] = {}
    for path, target in target_by_path.items():
        current = modules.get(path)
        if not isinstance(current, IndexedSparseLinear):
            raise ValueError(
                f"Pruning target {path!r} no longer resolves to an indexed layer"
            )
        if int(current.K) != int(target.k_before):
            raise ValueError(
                f"Pruning target {path!r} expected K={target.k_before}, "
                f"found K={current.K}"
            )
        if target.new_k == current.K:
            diagnostics = {
                "k_before": int(current.K),
                "k_after": int(current.K),
                "applied_weight_energy_retained": 1.0,
            }
        else:
            pruned, diagnostics = prune_indexed_layer(current, int(target.new_k))
            parent_path, _, child_name = path.rpartition(".")
            parent = modules[parent_path] if parent_path else model
            setattr(parent, child_name, pruned)
        diagnostics.update(
            {
                "in_features": int(target.in_features),
                "out_features": int(target.out_features),
                "target_density": float(target.target_density),
                "selector": target.selector,
            }
        )
        report[path] = diagnostics
    return report


def pruning_targets_manifest(
    targets: Sequence[IndexedPruningTarget],
) -> list[dict[str, Any]]:
    """Return a JSON-serializable prospective manifest for one rung."""

    return [asdict(target) for target in targets]


def prune_model_indexed_(
    model: nn.Module,
    keep_fraction: float,
    *,
    min_k: int = 1,
) -> dict[str, dict[str, float]]:
    """Prune every IndexedSparseLinear in ``model`` to ``keep_fraction`` of K.

    Returns per-module diagnostics keyed by module path. The model is edited
    in place (modules are replaced), so save a checkpoint afterwards and
    recovery-train with the patched contact counts.
    """

    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    report: dict[str, dict[str, float]] = {}
    for path, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, IndexedSparseLinear):
                new_k = max(min_k, round(child.K * keep_fraction))
                if new_k >= child.K:
                    continue
                pruned, diagnostics = prune_indexed_layer(child, new_k)
                setattr(module, child_name, pruned)
                full_path = f"{path}.{child_name}" if path else child_name
                report[full_path] = diagnostics
    logger.info(
        "Pruned %d indexed layers to keep_fraction=%.3f",
        len(report),
        keep_fraction,
    )
    return report


__all__ = [
    "IndexedPruningTarget",
    "apply_indexed_pruning_targets_",
    "prune_indexed_layer",
    "prune_model_indexed_",
    "pruning_targets_manifest",
    "resolve_indexed_pruning_targets",
]
