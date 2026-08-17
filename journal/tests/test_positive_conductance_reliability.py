from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_positive_conductance_reliability.py"
SPEC = importlib.util.spec_from_file_location("positive_reliability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
CFG = json.loads(MODEL.CONFIG.read_text(encoding="utf-8"))


def small_state(seed: int = 13):
    rng = np.random.default_rng(seed)
    branches = int(CFG["task"]["branches"])
    features = int(CFG["task"]["features_per_branch"])
    teacher = np.ones((branches, features))
    x, labels = MODEL.make_data(rng, 96, CFG, teacher)
    q = np.full(
        (branches, features),
        MODEL.inverse_softplus(CFG["task"]["initial_excitatory_conductance"]),
    )
    voltage = MODEL.base_state(q, x, CFG)[3]
    return q, x, labels, float(voltage.mean())


def test_point_gate_is_independent_and_matches_physical_shunt() -> None:
    q, x, labels, center = small_state()
    total = MODEL.base_state(q, x, CFG)[2].mean(axis=0)
    gains = np.linspace(0.15, 0.95, q.shape[0])
    shunt = MODEL.shunt_for_gain(gains, total, CFG)
    physical = MODEL.loss_and_gradient(
        q, x, labels, center, CFG, shunt=shunt, update_mode="physical_shunt"
    )
    point = MODEL.loss_and_gradient(
        q, x, labels, center, CFG, shunt=shunt, update_mode="point_gate"
    )
    assert np.allclose(physical["gradient"], point["gradient"], atol=1e-13)
    assert np.allclose(physical["matched_voltage"], physical["voltage"], atol=1e-14)


def test_state_clamped_unattenuated_control_matches_no_shunt_update() -> None:
    q, x, labels, center = small_state(17)
    total = MODEL.base_state(q, x, CFG)[2].mean(axis=0)
    shunt = MODEL.shunt_for_gain(np.linspace(0.2, 0.9, q.shape[0]), total, CFG)
    noise = np.random.default_rng(19).normal(scale=0.1, size=(len(x), q.shape[0]))
    unshunted = MODEL.loss_and_gradient(
        q, x, labels, center, CFG, coefficient_noise=noise, update_mode="unshunted"
    )
    control = MODEL.loss_and_gradient(
        q,
        x,
        labels,
        center,
        CFG,
        shunt=shunt,
        coefficient_noise=noise,
        update_mode="state_clamped_unattenuated",
    )
    assert np.allclose(unshunted["gradient"], control["gradient"], atol=1e-13)


def test_physical_shunt_gradient_matches_frozen_clamp_derivative() -> None:
    q, x, labels, center = small_state(23)
    total = MODEL.base_state(q, x, CFG)[2].mean(axis=0)
    shunt = MODEL.shunt_for_gain(np.linspace(0.1, 0.9, q.shape[0]), total, CFG)
    error = MODEL.physical_shunt_finite_difference_error(
        q, x, labels, center, CFG, shunt
    )
    assert error < 1e-5
