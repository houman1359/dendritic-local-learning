#!/usr/bin/env python3
"""Numerically verify the projected-gradient descent proposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = PROJECT / "results" / "credit_capture_bound_verification"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    min_slack = float("inf")
    max_capture_identity_error = 0.0
    captures = []
    realized_to_bound = []
    for _ in range(int(args.trials)):
        n = int(rng.integers(4, 33))
        m = int(rng.integers(1, n))
        eigenvalues = rng.uniform(0.02, 1.0, size=n)
        eigenvalues[-1] = 1.0
        rotation, _ = np.linalg.qr(rng.standard_normal((n, n)))
        hessian = (rotation * eigenvalues[None, :]) @ rotation.T
        w = rng.standard_normal(n)
        target = rng.standard_normal(n)
        gradient = hessian @ (w - target)
        dictionary, _ = np.linalg.qr(rng.standard_normal((n, m)))
        projected = dictionary @ (dictionary.T @ gradient)
        residual = np.linalg.norm(gradient - projected) / np.linalg.norm(gradient)
        capture = float(np.dot(projected, projected) / np.dot(gradient, gradient))
        max_capture_identity_error = max(
            max_capture_identity_error, abs(capture - (1.0 - residual * residual))
        )
        loss_before = 0.5 * float((w - target) @ hessian @ (w - target))
        updated = w - projected  # eta=1/L and L=1 by construction
        loss_after = 0.5 * float((updated - target) @ hessian @ (updated - target))
        realized_decrease = loss_before - loss_after
        guaranteed_decrease = 0.5 * float(np.dot(projected, projected))
        min_slack = min(min_slack, realized_decrease - guaranteed_decrease)
        captures.append(capture)
        if guaranteed_decrease > 0:
            realized_to_bound.append(realized_decrease / guaranteed_decrease)
    tolerance = 1e-10
    passed = min_slack >= -tolerance and max_capture_identity_error <= tolerance
    summary = {
        "status": "passed" if passed else "failed",
        "n_random_quadratic_trials": int(args.trials),
        "seed": int(args.seed),
        "minimum_descent_bound_slack": float(min_slack),
        "maximum_capture_identity_absolute_error": float(max_capture_identity_error),
        "mean_credit_capture": float(np.mean(captures)),
        "minimum_realized_decrease_over_guarantee": float(np.min(realized_to_bound)),
        "statement": (
            "For L-smooth quadratics at eta=1/L, projected-gradient decrease was never below "
            "||P_D g||^2/(2L), and capture equaled 1-residual^2 to numerical precision."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise AssertionError(summary)


if __name__ == "__main__":
    main()
