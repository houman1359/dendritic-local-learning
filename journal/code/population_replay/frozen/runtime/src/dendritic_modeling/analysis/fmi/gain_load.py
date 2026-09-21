"""Gain-load diagnostics: the finite-amplitude shunting decision.

Shunting and additive integration agree to first order at any operating
point, so the shunting decision requires finite-amplitude challenge probes
(paper §9, gain-load theory). This module provides the teacher-side gain
fingerprint and the challenge-evaluated mechanism comparison:

- ``gain_sensitivity``: how strongly the teacher latent responds to common
  input gain (a divisively normalized target saturates; an additive target
  scales linearly).
- ``challenge_scores``: mechanism probes fitted on natural data, evaluated
  on a common-gain challenge sweep — the regime where the tangent
  equivalence breaks.
- ``gain_load_score``: the v0 explanatory composite S_GL — the shunting
  advantage at finite amplitude minus any shunting penalty on natural data.
- ``paired_shunting_win_probability``: the registered decision statistic,
  computed by paired bootstrap over challenge examples.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from dendritic_modeling.analysis.fmi.mechanism import (
    MECHANISMS,
    delta_shunt,
    fit_probe_module,
)

__all__ = [
    "challenge_score_distribution",
    "challenge_scores",
    "gain_load_score",
    "gain_sensitivity",
    "paired_shunting_win_probability",
]

_DEFAULT_SCALES = (0.5, 0.8, 1.25, 2.0)


def gain_sensitivity(
    z_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    scales: tuple[float, ...] = _DEFAULT_SCALES,
) -> float:
    """Normalized variance of the latent under common input-gain scaling.

    ``1.0`` is the affine reference: the gain-responsive part of the latent,
    ``z(x) - z(0)``, scales linearly with input gain while the offset
    ``z(0)`` does not. Anchoring the reference at ``z(0)`` keeps the
    statistic offset-invariant — without it, a purely additive teacher with
    a DC offset scores far below one and is misclassified as gain-invariant
    (nominated for the opposite mechanism). Values well below one indicate
    genuine gain invariance and nominate the latent for divisive
    normalization — or for saturation, which is also gain-compressive; the
    mechanism probes of :func:`challenge_scores` disambiguate the two.
    """
    with torch.no_grad():
        base = z_fn(inputs)
        anchor = z_fn(torch.zeros_like(inputs[:1]))
        responses = torch.stack([z_fn(inputs * s) for s in scales])
        linear = torch.stack([anchor + s * (base - anchor) for s in scales])
        observed = responses.var(dim=0).mean()
        reference = linear.var(dim=0).mean().clamp_min(1e-12)
    return float(observed / reference)


def challenge_score_distribution(
    inputs: torch.Tensor,
    targets_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    scales: tuple[float, ...] = _DEFAULT_SCALES,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Fit on natural data and retain paired challenge-example losses.

    ``targets_fn`` must be the teacher latent closure (challenge targets are
    teacher responses to the swept inputs, so this is on-manifold in the
    gain direction by construction). The returned residual vectors share the
    same challenge examples, which makes a paired uncertainty calculation
    possible instead of treating point-risk differences as decisions.
    """
    with torch.no_grad():
        natural_targets = targets_fn(inputs)
        challenge_inputs = torch.cat([inputs * s for s in scales])
        challenge_targets = targets_fn(challenge_inputs)
        variance = challenge_targets.var().clamp_min(1e-12)
    scores: dict[str, float] = {}
    residuals: dict[str, torch.Tensor] = {}
    for kind in MECHANISMS:
        probe, _ = fit_probe_module(
            kind, inputs, natural_targets, steps=steps, lr=lr, seed=seed
        )
        with torch.no_grad():
            per_example = (probe(challenge_inputs) - challenge_targets).square()
            per_example = (per_example / variance).detach().float().cpu()
        residuals[kind] = per_example
        scores[kind] = float(per_example.mean())
    return scores, residuals


def challenge_scores(
    inputs: torch.Tensor,
    targets_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    scales: tuple[float, ...] = _DEFAULT_SCALES,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Fit each mechanism on natural data and return challenge point risks."""

    scores, _ = challenge_score_distribution(
        inputs,
        targets_fn,
        scales=scales,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    return scores


def paired_shunting_win_probability(
    residuals: dict[str, torch.Tensor],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Estimate ``P(R_shunt < R_best_additive)`` by paired bootstrap.

    Every mechanism must be evaluated on the same challenge examples. The
    additive comparator is reselected inside each bootstrap replicate so the
    probability includes uncertainty about which matched additive control is
    best. This statistic, rather than ``gain_load_score``, is suitable for a
    preregistered shunting decision threshold.
    """

    required = set(MECHANISMS)
    missing = sorted(required - set(residuals))
    if missing:
        raise ValueError(f"paired shunting bootstrap is missing mechanisms {missing}")
    vectors = {
        name: torch.as_tensor(residuals[name]).detach().float().cpu().reshape(-1)
        for name in MECHANISMS
    }
    sizes = {int(vector.numel()) for vector in vectors.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) < 2:
        raise ValueError("paired residual vectors must have one shared length >= 2")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors.values()):
        raise ValueError("paired residual vectors must be finite")

    count = next(iter(sizes))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    wins = 0.0
    processed = 0
    while processed < int(bootstrap_samples):
        current = min(256, int(bootstrap_samples) - processed)
        indices = torch.randint(
            count,
            (current, count),
            generator=generator,
            device="cpu",
        )
        means = {name: vector[indices].mean(dim=1) for name, vector in vectors.items()}
        best_additive = torch.minimum(
            means["raw_additive"],
            torch.minimum(means["tangent_additive"], means["normalized_additive"]),
        )
        difference = best_additive - means["shunting"]
        wins += float((difference > 0).sum())
        wins += 0.5 * float((difference == 0).sum())
        processed += current
    return wins / float(bootstrap_samples)


def gain_load_score(
    natural_scores: dict[str, float],
    challenge: dict[str, float],
) -> float:
    """v0 explanatory composite S_GL.

    Positive values explain when shunting beats the matched controls on the
    finite-amplitude sweep without paying for it on natural data. This mixed-
    unit score is not a decision rule; use
    :func:`paired_shunting_win_probability` for selection.
    """
    benefit = delta_shunt(challenge)
    load = max(0.0, -delta_shunt(natural_scores))
    return benefit - load
