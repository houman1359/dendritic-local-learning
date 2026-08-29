from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_path_necessity_fashion.py"
)
SPEC = importlib.util.spec_from_file_location("path_necessity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_balanced_context_label_cells() -> None:
    context, label = MODULE.balanced_context_labels(
        np.random.default_rng(4), n=160, branches=8
    )
    counts = np.zeros((8, 2), dtype=int)
    for branch, target in zip(context, label):
        counts[int(branch), int(target)] += 1
    assert np.ptp(counts) == 0
    assert abs(float(np.corrcoef(context, label)[0, 1])) < 1e-12


def test_conflict_masks_are_nested_and_never_replace_selected_path() -> None:
    n, branches, features = 16, 4, 3
    context = np.tile(np.arange(branches), n // branches)
    label = np.tile(np.asarray([0, 1]), n // 2).astype(np.float32)
    selected = np.arange(n * features, dtype=np.float32).reshape(n, features)
    opposite = -np.ones((n, branches, features), dtype=np.float32)
    uniforms = np.linspace(0.01, 0.99, n * branches).reshape(n, branches)
    base = MODULE.BaseTrials(
        context=context,
        label=label,
        selected=selected,
        opposite=opposite,
        uniforms=uniforms,
        selected_source_label=label.astype(int),
        opposite_source_label=np.repeat((1 - label.astype(int))[:, None], branches, axis=1),
    )
    low_values, low_mask = MODULE.materialize_trials(base, 0.25)
    high_values, high_mask = MODULE.materialize_trials(base, 0.75)
    assert np.all(~low_mask | high_mask)
    rows = np.arange(n)
    assert not np.any(low_mask[rows, context])
    assert not np.any(high_mask[rows, context])
    np.testing.assert_array_equal(low_values[rows, context], selected)
    np.testing.assert_array_equal(high_values[rows, context], selected)


def test_route_matrix_rank_assignment_and_equivalence() -> None:
    routes = MODULE.route_matrices(4)
    assert routes.shape == (len(MODULE.CONDITIONS), 4, 4)
    np.testing.assert_allclose(routes[0], np.ones((4, 4)) / 4)
    np.testing.assert_allclose(routes[1], np.eye(4))
    np.testing.assert_allclose(routes[2], np.roll(np.eye(4), 1, axis=1))
    np.testing.assert_array_equal(routes[1], routes[3])
    np.testing.assert_array_equal(routes[1], routes[4])
    assert np.linalg.matrix_rank(routes[0]) == 1
    assert np.linalg.matrix_rank(routes[1]) == 4
    assert np.linalg.matrix_rank(routes[2]) == 4


def test_predicted_branch_count_dependent_boundaries() -> None:
    for branches, boundary in ((2, 1.0), (4, 2 / 3), (8, 4 / 7)):
        shared, _ = MODULE.theoretical_relative_signals(branches, boundary)
        assert abs(shared) < 1e-12
        below, _ = MODULE.theoretical_relative_signals(branches, boundary - 0.01)
        assert below > 0
        if boundary < 1:
            above, _ = MODULE.theoretical_relative_signals(branches, boundary + 0.01)
            assert above < 0
    _, deranged = MODULE.theoretical_relative_signals(8, 0.5)
    assert abs(deranged) < 1e-12


def test_exact_analytic_gradient_matches_autograd() -> None:
    torch.manual_seed(5)
    n, branches, features = 32, 4, 7
    values = torch.randn(n, branches, features)
    context = torch.arange(n) % branches
    labels = (torch.arange(n) % 2).float()
    initial = torch.randn(branches, features) * 0.01
    difference = MODULE.autograd_exact_difference(
        initial, values, context, labels
    )
    assert difference < 1e-6


def test_exact_equivalent_conditions_train_identically() -> None:
    torch.manual_seed(6)
    n, branches, features = 32, 4, 5
    values = torch.randn(n, branches, features)
    context = torch.arange(n) % branches
    labels = (torch.arange(n) % 2).float()
    initial = torch.randn(branches, features) * 0.01
    cfg = {"training": {"learning_rate": 0.02, "epochs": 3}}
    fitted = MODULE.train_models(
        initial,
        values,
        context,
        labels,
        torch.as_tensor(MODULE.route_matrices(branches)),
        cfg,
    )
    exact = fitted[MODULE.CONDITIONS.index("correct_path")]
    torch.testing.assert_close(
        exact, fitted[MODULE.CONDITIONS.index("backpropagation")], rtol=0, atol=0
    )
    torch.testing.assert_close(
        exact,
        fitted[MODULE.CONDITIONS.index("gated_point_emulation")],
        rtol=0,
        atol=0,
    )
