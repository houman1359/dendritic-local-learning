from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


JOURNAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))

import run_reconstructed_tree_task_learning as analysis  # noqa: E402


def synthetic_tree() -> analysis.PassiveTree:
    matrix = np.asarray(
        [
            [2.4, -0.35, -0.20],
            [-0.35, 1.8, -0.15],
            [-0.20, -0.15, 1.5],
        ],
        dtype=float,
    )
    inverse = np.linalg.inv(matrix)
    rhs = np.asarray([0.05, 0.12, -0.03], dtype=float)
    dictionary = np.eye(2, dtype=float)
    return analysis.PassiveTree(
        root_id=1,
        n_segments=3,
        root_index=0,
        site_indices=np.asarray([1, 2], dtype=int),
        site_segment_ids=np.asarray([11, 12], dtype=int),
        base_voltage=inverse @ rhs,
        inverse_base=inverse,
        excitatory_reversal=1.0,
        morphology_dictionary=dictionary,
        shuffled_dictionary=dictionary[:, ::-1],
        random_dictionary=dictionary,
        requested_channels=2,
        used_channels=2,
        dictionary_rank=2,
        selected_route_segments=[11, 12],
        random_route_segments=[11, 12],
    )


def test_exact_conductance_gradient_matches_finite_difference() -> None:
    tree = synthetic_tree()
    x = np.asarray(
        [
            [0.2, 1.1],
            [1.3, 0.4],
            [0.7, 0.9],
            [1.8, 0.1],
        ],
        dtype=float,
    )
    y = np.asarray([-0.4, 0.7, 0.1, -0.2], dtype=float)
    state = analysis.State(
        q=np.asarray([-1.2, -0.5], dtype=float),
        readout=1.7,
        bias=-0.15,
    )
    checks = [
        analysis.finite_difference_check(
            tree,
            x,
            y,
            state,
            coordinate,
            weight_decay=1e-3,
            readout_decay=1e-6,
        )
        for coordinate in range(x.shape[1])
    ]
    assert max(float(check["relative_error"]) for check in checks) < 1e-7


def test_site_shuffle_preserves_column_values_nonzeros_and_rank() -> None:
    dictionary = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 1.0, 0.0],
            [0.6, 0.7, 1.0],
            [0.0, 0.5, 0.9],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    rank = int(np.linalg.matrix_rank(dictionary))
    shuffled = analysis.rank_matched_shuffle(
        dictionary,
        rank,
        np.random.default_rng(20260731),
    )
    assert np.linalg.matrix_rank(shuffled) == rank
    assert np.count_nonzero(shuffled) == np.count_nonzero(dictionary)
    assert not np.array_equal(shuffled, dictionary)
    for column in range(dictionary.shape[1]):
        np.testing.assert_array_equal(
            np.sort(shuffled[:, column]),
            np.sort(dictionary[:, column]),
        )
