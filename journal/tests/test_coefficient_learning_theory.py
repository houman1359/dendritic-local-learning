from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "code" / "theory" / "coefficient_learning.py"
SPEC = importlib.util.spec_from_file_location("coefficient_learning", PATH)
assert SPEC is not None and SPEC.loader is not None
THEORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THEORY)


def test_exact_risk_matches_monte_carlo() -> None:
    rng = np.random.default_rng(41)
    operator = np.diag([1.0, 0.4, 0.0])
    target = np.array([0.7, -0.3, 0.0])
    noise_variance = 0.2
    step = 0.25
    iterations = 12
    expected = THEORY.linear_field_learning_risk(
        operator, target, noise_variance, step, iterations
    )["total_expected_loss"]
    fields = np.zeros((100_000, 3))
    for _ in range(iterations):
        noise = rng.normal(scale=np.sqrt(noise_variance), size=fields.shape)
        fields -= step * ((fields - target - noise) @ operator.T)
    observed = 0.5 * np.mean(np.sum((fields - target) ** 2, axis=1))
    assert abs(observed - expected) < 8e-4


def test_slow_mode_has_bias_variance_crossover() -> None:
    target = np.array([1.0, 0.4])
    fast = np.eye(2)
    slow = np.diag([1.0, 0.05])
    crossover = THEORY.pairwise_effective_sample_crossover(
        slow, fast, target, 4.0, 0.2, 80
    )
    assert np.isfinite(crossover) and crossover > 0
    below = max(crossover / 4.0, 0.1)
    above = crossover * 4.0
    slow_below = THEORY.linear_field_learning_risk(slow, target, 4.0 / below, 0.2, 80)
    fast_below = THEORY.linear_field_learning_risk(fast, target, 4.0 / below, 0.2, 80)
    slow_above = THEORY.linear_field_learning_risk(slow, target, 4.0 / above, 0.2, 80)
    fast_above = THEORY.linear_field_learning_risk(fast, target, 4.0 / above, 0.2, 80)
    assert slow_below["total_expected_loss"] < fast_below["total_expected_loss"]
    assert slow_above["total_expected_loss"] > fast_above["total_expected_loss"]
