"""Mechanism selection probes: additive vs normalized vs shunting vs gated.

Implements the matched scalar probes of the FMI theory (paper §9): four
branch mechanisms sharing the same nonnegative drive parameterization and
parameter count, fitted to one scalar latent target on nonnegative inputs,
compared on held-out risk. Because shunting and additive integration agree
to first order at any operating point (tangent indistinguishability), the
decision statistic is the held-out advantage at finite amplitude,
``delta_shunt``, plus the product-vs-additive advantage ``r_mult`` for the
gating decision. Also provides the architecture-family error-curve fit
``fit_error_curve`` for q_A(c) = q_inf + a * c^(-alpha).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MECHANISMS",
    "delta_shunt",
    "fit_error_curve",
    "fit_probe",
    "fit_probe_module",
    "mechanism_scores",
    "multiplicative_advantage",
]

MECHANISMS = ("raw_additive", "tangent_additive", "normalized_additive", "shunting")


class _BranchProbe(nn.Module):
    """One scalar branch mechanism over shared nonnegative E/I drives.

    All four mechanisms use identical drive parameterizations
    ``E = u . softplus(p_E)``, ``I = u . softplus(p_I)`` plus two scalars,
    so held-out risk differences reflect the integration mechanism alone.
    """

    def __init__(self, dim: int, kind: str, generator: torch.Generator | None = None):
        super().__init__()
        if kind not in MECHANISMS:
            raise ValueError(f"kind must be one of {MECHANISMS}, got {kind!r}")
        self.kind = kind
        self.p_exc = nn.Parameter(torch.randn(dim, generator=generator) * 0.1 - 2.0)
        self.p_inh = nn.Parameter(torch.randn(dim, generator=generator) * 0.1 - 2.0)
        self.scalar_a = nn.Parameter(torch.zeros(()))
        self.scalar_b = nn.Parameter(torch.zeros(()))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        excitation = u @ F.softplus(self.p_exc)
        inhibition = u @ F.softplus(self.p_inh)
        if self.kind == "raw_additive":
            return excitation - inhibition + self.scalar_a
        if self.kind == "tangent_additive":
            # Affine in the drives: the exact first-order family.
            return (
                excitation * torch.sigmoid(self.scalar_b) * 2.0
                - inhibition * torch.sigmoid(self.scalar_b) * 2.0
                + self.scalar_a
            )
        denominator = 1.0 + excitation + inhibition + F.softplus(self.scalar_b)
        if self.kind == "normalized_additive":
            return (excitation - inhibition + self.scalar_a) / denominator
        return (excitation + self.scalar_a) / denominator  # shunting


class _ProductProbe(nn.Module):
    """Gated product vs matched additive alternatives.

    Modes: ``additive`` (linear sum), ``saturating`` (tanh of the sum with a
    learned amplitude — included so a merely saturating additive teacher is
    not misread as multiplicative), and ``product``
    (``phi(u . g + b) * (u . v) + c``).
    """

    MODES = ("additive", "saturating", "product")

    def __init__(self, dim: int, mode: str, generator: torch.Generator | None = None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.gate = nn.Parameter(torch.randn(dim, generator=generator) * 0.3)
        self.gate_bias = nn.Parameter(torch.zeros(()))
        self.value = nn.Parameter(torch.randn(dim, generator=generator) * 0.3)
        self.offset = nn.Parameter(torch.zeros(()))
        self.amplitude = nn.Parameter(torch.ones(()))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "product":
            return (
                torch.sigmoid(u @ self.gate + self.gate_bias) * (u @ self.value)
                + self.offset
            )
        if self.mode == "saturating":
            return (
                self.amplitude
                * torch.tanh(u @ self.gate + u @ self.value + self.gate_bias)
                + self.offset
            )
        return u @ self.gate + u @ self.value + self.offset


def _split(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    holdout_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized train/holdout split (a contiguous split biases relative
    risks whenever the calibration set carries any ordering)."""
    permutation = torch.randperm(inputs.shape[0], generator=generator)
    split = int(inputs.shape[0] * (1.0 - holdout_fraction))
    train, heldout = permutation[:split], permutation[split:]
    return inputs[train], targets[train], inputs[heldout], targets[heldout]


def _fit(
    probe: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    heldout_inputs: torch.Tensor,
    heldout_targets: torch.Tensor,
    steps: int,
    lr: float,
) -> float:
    # Probes must never backpropagate into whatever produced the inputs or
    # targets (e.g. a teacher forward left graph-attached by the caller).
    inputs, targets = inputs.detach(), targets.detach()
    heldout_inputs, heldout_targets = heldout_inputs.detach(), heldout_targets.detach()
    probe = probe.to(inputs.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(probe(inputs), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        residual = F.mse_loss(probe(heldout_inputs), heldout_targets)
        variance = heldout_targets.var().clamp_min(1e-12)
    return float(residual / variance)  # relative held-out risk


def fit_probe_module(
    kind: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.05,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[nn.Module, float]:
    """Fit one mechanism probe; return the trained probe and held-out risk.

    Uses a local generator (never the global RNG) so probe fits stay
    independent of each other and of caller state.
    """
    generator = torch.Generator().manual_seed(seed)
    probe = _BranchProbe(inputs.shape[1], kind, generator=generator)
    risk = _fit(
        probe,
        *_split(inputs, targets, holdout_fraction, generator),
        steps,
        lr,
    )
    return probe, risk


def fit_probe(
    kind: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.05,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> float:
    """Relative held-out risk of one mechanism probe on one latent target."""
    _, risk = fit_probe_module(
        kind,
        inputs,
        targets,
        steps=steps,
        lr=lr,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    return risk


def mechanism_scores(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out relative risk of all four matched branch mechanisms."""
    return {
        kind: fit_probe(kind, inputs, targets, steps=steps, lr=lr, seed=seed)
        for kind in MECHANISMS
    }


def delta_shunt(scores: dict[str, float]) -> float:
    """Relative held-out advantage of shunting over the best additive control.

    Clamped to ``[-1, 1]``: when an additive control fits (near) perfectly
    the raw ratio diverges, but no decision needs more than "shunting is
    at least twice as bad". When BOTH sides fit essentially perfectly
    (relative risks below ``1e-4``) the ratio carries no mechanistic
    evidence at all and ``0.0`` is returned.
    """
    best_additive = min(
        scores["raw_additive"],
        scores["tangent_additive"],
        scores["normalized_additive"],
    )
    if best_additive < 1e-4 and scores["shunting"] < 1e-4:
        return 0.0
    best_additive = max(best_additive, 1e-9)
    advantage = (best_additive - scores["shunting"]) / best_additive
    return float(max(-1.0, min(1.0, advantage)))


def multiplicative_advantage(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 800,
    lr: float = 0.1,
    seed: int = 0,
    restarts: int = 3,
) -> float:
    """R_mult: relative held-out advantage of a product probe over additive.

    The product probe is compared against the better of two matched
    non-product alternatives — a linear sum AND a saturating (tanh)
    additive form — so a merely saturating additive teacher is not misread
    as multiplicative. The product and saturating landscapes are
    multimodal, so those probes are fitted from ``restarts``
    initializations keeping the best held-out risk. Local generators only;
    the global RNG is never touched. The advantage is clamped to
    ``[-1, 1]``, and ``0.0`` is returned when every probe fits essentially
    perfectly (no evidence either way).
    """
    generator = torch.Generator().manual_seed(seed)
    split_data = _split(inputs, targets, 0.25, generator)
    risks: dict[str, float] = {}
    for mode in _ProductProbe.MODES:
        attempts = 1 if mode == "additive" else restarts
        for attempt in range(attempts):
            init = torch.Generator().manual_seed(seed + attempt)
            probe = _ProductProbe(inputs.shape[1], mode, generator=init)
            risk = _fit(probe, *split_data, steps, lr)
            if mode not in risks or risk < risks[mode]:
                risks[mode] = risk
    baseline = min(risks["additive"], risks["saturating"])
    if baseline < 1e-4 and risks["product"] < 1e-4:
        return 0.0
    floor = max(baseline, 1e-6)
    return float(max(-1.0, min(1.0, (baseline - risks["product"]) / floor)))


def fit_error_curve(
    resources: torch.Tensor,
    errors: torch.Tensor,
    grid_points: int = 64,
) -> tuple[float, float, float]:
    """Fit ``q(c) = q_inf + a * c^(-alpha)`` by grid search over the floor.

    For each candidate floor the remaining curve is linear in log-log space.
    Two properties are essential for saturated sweeps (floor dominating the
    decaying term): floor candidates are spaced geometrically *below*
    ``errors.min()`` — the true floor always sits a residual-scale below
    the smallest observed error, so a linear grid from zero misses it —
    and candidates are compared by reconstruction error in the ORIGINAL
    error space, because log-space residuals are not comparable across
    floors and systematically prefer floors far below the truth. Floors
    producing non-positive residuals are rejected, never clamped. Returns
    ``(q_inf, a, alpha)``.
    """
    if (resources <= 0).any():
        raise ValueError("resources must be strictly positive")
    if torch.unique(resources).numel() != resources.numel():
        raise ValueError("resources must be distinct")
    if (errors < 0).any():
        raise ValueError("errors must be non-negative")

    if resources.numel() != errors.numel() or resources.numel() < 3:
        raise ValueError("need matching resource/error vectors of length >= 3")
    if bool((errors <= 0).any()):
        raise ValueError("errors must be strictly positive")
    log_c = torch.log(resources.double())
    errors = errors.double()
    minimum = float(errors.min())
    span = max(float(errors.max()) - minimum, minimum * 1e-3)
    offsets = torch.logspace(
        float(torch.log10(torch.tensor(span * 1e-9))),
        float(torch.log10(torch.tensor(span))),
        max(grid_points - 1, 2),
        dtype=torch.float64,
    )
    candidates = [0.0] + [
        minimum - float(offset) for offset in offsets if minimum - float(offset) >= 0.0
    ]
    best = None
    for floor in candidates:
        residual = errors - floor
        if bool((residual <= 0).any()):
            continue
        log_r = torch.log(residual)
        centered = log_c - log_c.mean()
        slope = float((centered * (log_r - log_r.mean())).sum() / (centered**2).sum())
        intercept = float(log_r.mean() - slope * log_c.mean())
        reconstructed = floor + torch.exp(intercept + slope * log_c)
        sse = float(((reconstructed - errors) ** 2).sum())
        if best is None or sse < best[0]:
            best = (
                sse,
                float(floor),
                float(torch.exp(torch.tensor(intercept))),
                -slope,
            )
    _, q_inf, amplitude, alpha = best
    return q_inf, amplitude, alpha
