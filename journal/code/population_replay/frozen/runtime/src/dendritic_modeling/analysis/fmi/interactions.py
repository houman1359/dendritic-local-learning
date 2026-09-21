"""Feature-interaction structure: branch grouping and tree depth.

Estimates the pairwise interaction matrix ``H_pq`` of one scalar latent via
structured finite differences, scores candidate partitions by captured
off-diagonal interaction energy (the branch-modularity ``Q_j(B)`` of the FMI
paper), and infers blocks by recursive spectral bipartition. Blocks with
vanishing cross-interaction correspond exactly to additive branch
decompositions (paper, separability proposition).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "infer_blocks",
    "infer_tree",
    "interaction_matrix",
    "partition_modularity",
]


def interaction_matrix(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    coordinates: list[int] | None = None,
    probe_points: int = 32,
    step: float = 0.5,
) -> torch.Tensor:
    """RMS pair-interaction matrix over ``coordinates`` (default: all).

    Uses the second-order finite difference
    ``z(x+d_p+d_q) - z(x+d_p) - z(x+d_q) + z(x)`` averaged over probe points,
    with steps scaled by each coordinate's standard deviation. The RMS delta
    is normalized by the product of the two perturbation sizes, so entries
    are in mixed-second-derivative (Hessian) units and comparable across
    coordinates with different scales (audit fix 2026-08-16).
    """
    d_in = inputs.shape[1]
    coords = list(range(d_in)) if coordinates is None else list(coordinates)
    n = len(coords)
    points = inputs[:probe_points].detach()
    # Anchor near-constant coordinates to the typical scale so their probe
    # steps neither underflow (spurious zeros) nor amplify estimator noise.
    stds = inputs.std(dim=0)
    scale = stds.clamp_min(0.05 * stds.mean().clamp_min(1e-6)) * step

    base = z_fn(points)
    singles = []
    for p in coords:
        shifted = points.clone()
        shifted[:, p] += scale[p]
        singles.append(z_fn(shifted))
    interactions = torch.zeros(n, n)
    for i, p in enumerate(coords):
        for j in range(i + 1, n):
            q = coords[j]
            both = points.clone()
            both[:, p] += scale[p]
            both[:, q] += scale[q]
            delta = z_fn(both) - singles[i] - singles[j] + base
            value = float(delta.pow(2).mean().sqrt()) / float(
                (scale[p] * scale[q]).clamp_min(1e-12)
            )
            interactions[i, j] = value
            interactions[j, i] = value
    return interactions


def partition_modularity(
    interactions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Fraction of off-diagonal interaction energy captured within blocks."""
    off_diagonal = interactions - torch.diag(torch.diagonal(interactions))
    total = float(off_diagonal.sum())
    if total <= 0.0:
        return 1.0
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    return float(off_diagonal[same].sum()) / total


def _fiedler_split(interactions: torch.Tensor) -> torch.Tensor:
    """Two-way spectral partition labels from the graph Laplacian.

    For a near-disconnected graph the two smallest Laplacian eigenvectors
    span {constant, component separator} and ``eigh`` may rotate freely
    within that near-degenerate pair, so the separator is recovered as the
    centered candidate with the most variation, thresholded at zero (a
    median threshold would force balanced splits and misassign members of
    unequal blocks).
    """
    weights = interactions.clamp_min(0.0)
    degree = weights.sum(dim=1)
    laplacian = torch.diag(degree) - weights
    _, vecs = torch.linalg.eigh(laplacian)
    count = min(2, vecs.shape[1])
    candidates = [vecs[:, i] - vecs[:, i].mean() for i in range(count)]
    fiedler = max(candidates, key=lambda vec: float(vec.abs().sum()))
    return (fiedler >= 0).long()


def infer_blocks(
    interactions: torch.Tensor,
    max_blocks: int = 8,
    min_gain: float = 0.02,
) -> torch.Tensor:
    """Greedy recursive bipartition while within-block modularity is retained.

    Splits the block whose bipartition loses the least interaction energy,
    stopping when any further split would drop captured energy by more than
    ``min_gain`` or ``max_blocks`` is reached. Returns integer labels.
    """
    n = interactions.shape[0]
    labels = torch.zeros(n, dtype=torch.long)
    while int(labels.max()) + 1 < max_blocks:
        best = None
        for block in range(int(labels.max()) + 1):
            members = torch.where(labels == block)[0]
            if len(members) < 2:
                continue
            sub = interactions[members][:, members]
            split = _fiedler_split(sub)
            if split.min() == split.max():
                continue
            trial = labels.clone()
            trial[members[split == 1]] = int(labels.max()) + 1
            score = partition_modularity(interactions, trial)
            if best is None or score > best[0]:
                best = (score, trial)
        if best is None:
            break
        current = partition_modularity(interactions, labels)
        if current - best[0] > min_gain:
            break
        labels = best[1]
    # Renumber labels densely in first-appearance order.
    remap = {int(v): i for i, v in enumerate(dict.fromkeys(labels.tolist()))}
    return torch.tensor([remap[int(v)] for v in labels], dtype=torch.long)


def infer_tree(
    interactions: torch.Tensor,
    max_depth: int = 2,
    min_gain: float = 0.02,
    eta_tree: float = 0.65,
    min_block: int = 2,
) -> tuple[int, list[torch.Tensor]]:
    """Hierarchical block recovery: nested splits while locally separable.

    Level 1 uses :func:`infer_blocks` (global modularity retention). Each
    deeper level bipartitions a block only when the split retains at least
    ``eta_tree`` of that block's own off-diagonal interaction energy — the
    local analogue of the paper's depth criterion: depth must be earned by
    nested near-separability, and a uniformly coupled block refuses to
    split. Returns ``(depth_used, labels_per_level)`` where level ``i``
    labels refine level ``i-1``.
    """
    levels = [infer_blocks(interactions, min_gain=min_gain)]
    for _ in range(1, max_depth):
        labels = levels[-1]
        refined = labels.clone()
        next_label = int(labels.max()) + 1
        changed = False
        for block in range(int(labels.max()) + 1):
            members = torch.where(labels == block)[0]
            if len(members) < 2 * min_block:
                continue
            sub = interactions[members][:, members]
            off_total = float(sub.sum() - sub.diagonal().sum())
            if off_total <= 0.0:
                continue
            split = _fiedler_split(sub)
            if split.min() == split.max():
                continue
            local = torch.zeros(len(members), dtype=torch.long)
            local[split == 1] = 1
            retained = partition_modularity(sub, local)
            if retained >= eta_tree:
                refined[members[split == 1]] = next_label
                next_label += 1
                changed = True
        if not changed:
            break
        levels.append(refined)
    depth_used = len(levels) - (1 if int(levels[0].max()) == 0 else 0)
    return max(depth_used, 0), levels
