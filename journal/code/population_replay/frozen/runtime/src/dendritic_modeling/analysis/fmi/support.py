"""Gradient-based support, rank, stability, and sign estimators.

Level-2 estimators of the FMI theory operating on one scalar latent target
``z(x)``: coordinate attribution energy (synapse count), gradient second
moment and participation ratio (intrinsic active rank vs. coordinate
sparsity), support-stability Jaccard (sparsity class), and sign consistency
(E/I allocation).

Every estimator takes a callable ``z_fn`` mapping a batch ``[M, d_in]``
(requires_grad handled internally) to a scalar per example ``[M]``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "attribution_energy",
    "gradient_participation_ratio",
    "gradient_second_moment",
    "participation_ratio",
    "sign_consistency",
    "support_jaccard",
    "support_size",
]


def _batch_gradients(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Per-example input gradients ``[M, d_in]`` of a scalar latent."""
    x = inputs.detach().clone().requires_grad_(True)
    z = z_fn(x)
    if z.ndim != 1:
        raise ValueError(
            f"z_fn must return one scalar per example, got {tuple(z.shape)}"
        )
    (grads,) = torch.autograd.grad(z.sum(), x)
    return grads.detach()


def attribution_energy(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Input-scaled attribution energy ``a_p = E[(x_p dz/dx_p)^2]`` per coordinate."""
    grads = _batch_gradients(z_fn, inputs)
    return (inputs.detach() * grads).pow(2).mean(dim=0)


def support_size(attributions: torch.Tensor, eta: float) -> int:
    """Smallest K whose top-K attributions cover an ``eta`` energy fraction."""
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    total = float(attributions.sum())
    if total <= 0.0:
        return 0
    sorted_energy = torch.sort(attributions, descending=True).values
    cumulative = torch.cumsum(sorted_energy, dim=0) / total
    return min(int((cumulative < eta).sum()) + 1, attributions.numel())


def gradient_second_moment(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Gradient second moment ``Gamma = E[grad z grad z^T]`` ``[d_in, d_in]``."""
    grads = _batch_gradients(z_fn, inputs)
    return grads.T @ grads / grads.shape[0]


def gradient_participation_ratio(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> float:
    """Effective rank of the gradient second moment without a width-square matrix.

    If ``G`` contains per-example gradients, ``tr(G'G)^2`` can be computed as
    ``||GG'||_F^2``. The sample-space form is exact and changes the dominant
    storage from ``d_in^2`` to ``M^2`` when calibration examples are fewer than
    input coordinates, as is typical for large-model teacher profiling.
    """

    grads = _batch_gradients(z_fn, inputs)
    trace = float(grads.square().sum() / grads.shape[0])
    if grads.shape[0] <= grads.shape[1]:
        gram = grads @ grads.T / grads.shape[0]
    else:
        gram = grads.T @ grads / grads.shape[0]
    trace_sq = float(gram.square().sum())
    if trace_sq <= 0.0:
        return 0.0
    return trace * trace / trace_sq


def participation_ratio(matrix: torch.Tensor) -> float:
    """Effective rank ``tr(Gamma)^2 / tr(Gamma^2)`` of a PSD matrix."""
    trace = float(torch.diagonal(matrix).sum())
    trace_sq = float((matrix * matrix.T).sum())
    if trace_sq <= 0.0:
        return 0.0
    return trace * trace / trace_sq


def support_jaccard(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    k: int,
    pairs: int = 64,
    generator: torch.Generator | None = None,
) -> float:
    """Mean pairwise Jaccard overlap of per-example top-``k`` attribution supports."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    grads = _batch_gradients(z_fn, inputs)
    scores = (inputs.detach() * grads).abs()
    top = scores.topk(min(k, scores.shape[1]), dim=1).indices
    m = top.shape[0]
    if m < 2:
        raise ValueError("support_jaccard needs at least two examples")
    idx_a = torch.randint(m, (pairs,), generator=generator)
    idx_b = torch.randint(m, (pairs,), generator=generator)
    idx_b = torch.where(idx_b == idx_a, (idx_b + 1) % m, idx_b)
    overlaps = []
    for a, b in zip(idx_a.tolist(), idx_b.tolist()):
        sa, sb = set(top[a].tolist()), set(top[b].tolist())
        overlaps.append(len(sa & sb) / len(sa | sb))
    return float(sum(overlaps) / len(overlaps))


def sign_consistency(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Per-coordinate sign consistency ``s_p in [-1, 1]`` of ``dz/dx_p``.

    ``+1``: consistently excitatory; ``-1``: consistently inhibitory;
    near ``0``: context-dependent (a gating / signed-control signal).
    """
    grads = _batch_gradients(z_fn, inputs)
    return grads.mean(dim=0) / (grads.pow(2).mean(dim=0).sqrt() + epsilon)
