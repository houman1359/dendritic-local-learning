from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_adaptive_conductance_reliability.py"
SPEC = importlib.util.spec_from_file_location("adaptive_conductance_reliability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)
CFG = json.loads(MODEL.CONFIG.read_text(encoding="utf-8"))


def small_state(seed: int = 37):
    rng = np.random.default_rng(seed)
    branches = int(CFG["task"]["branches"])
    features = int(CFG["task"]["features_per_branch"])
    teacher = np.ones((branches, features))
    x, labels = MODEL.BASE.make_data(rng, 96, CFG, teacher)
    q = np.full(
        (branches, features),
        MODEL.BASE.inverse_softplus(CFG["task"]["initial_excitatory_conductance"]),
    )
    center = float(MODEL.BASE.base_state(q, x, CFG)[3].mean())
    return rng, q, x, labels, center


def test_paired_gradient_moments_recover_signal_and_noise_in_expectation() -> None:
    rng, q, x, labels, center = small_state()
    clean = MODEL.BASE.loss_and_gradient(q, x, labels, center, CFG)["gradient"]
    coefficient_sd = np.linspace(0.05, 0.3, q.shape[0])
    signals, noises = [], []
    for _ in range(300):
        left = rng.normal(size=(len(x), q.shape[0])) * coefficient_sd
        right = rng.normal(size=(len(x), q.shape[0])) * coefficient_sd
        signal, noise = MODEL.observed_moments(q, x, labels, center, CFG, left, right)
        signals.append(signal)
        noises.append(noise)
    observed = np.mean(signals, axis=0)
    expected = np.sum(clean * clean, axis=1)
    assert np.allclose(observed, expected, rtol=0.35, atol=2e-5)
    assert np.all(np.mean(noises, axis=0) > 0)


def test_gain_from_equal_moments_is_branch_constant() -> None:
    signal = np.linspace(0.2, 2.0, 8)
    noise = signal.copy()
    gain = MODEL.gain_from_moments(signal, noise, CFG)
    assert np.allclose(gain, gain[0])
    assert np.all((gain >= 0.03) & (gain <= 1.0))


def test_independent_point_gate_matches_physical_shunt_for_adaptive_gain() -> None:
    rng, q, x, labels, center = small_state(43)
    gain = MODEL.gain_from_moments(
        np.linspace(0.1, 1.0, q.shape[0]),
        np.linspace(1.0, 0.1, q.shape[0]),
        CFG,
    )
    total = MODEL.BASE.base_state(q, x, CFG)[2].mean(axis=0)
    shunt = MODEL.BASE.shunt_for_gain(gain, total, CFG)
    noise = rng.normal(scale=0.1, size=(len(x), q.shape[0]))
    physical = MODEL.BASE.loss_and_gradient(
        q, x, labels, center, CFG, shunt=shunt,
        coefficient_noise=noise, update_mode="physical_shunt",
    )
    point = MODEL.BASE.loss_and_gradient(
        q, x, labels, center, CFG, shunt=shunt,
        coefficient_noise=noise, update_mode="point_gate",
    )
    assert np.allclose(physical["gradient"], point["gradient"], atol=1e-13)
