"""Analysis invariants: normalization, nesting and preservation of model state."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import CreditNet, dataset
from run import diagnose
from functional import cross_covariance, partial_rank


def test_common_state_diagnostics_do_not_change_parameters():
    net = CreditNet(2, 2, 2).double()
    data = dataset(5, 'diagnostic', 23, 2)
    before = {k:v.clone() for k,v in net.state_dict().items()}
    rows = diagnose(net, data)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert rows[0]['exact_core_cosine'] == pytest.approx(1.)
    assert len({row['baseline_nmse'] for row in rows}) == 1
    assert len({row['displacement_norm'] for row in rows}) == 1


def test_split_covariance_is_symmetric_and_recovers_correlation_without_noise():
    x = np.random.default_rng(1).normal(size=(100, 6))
    cov = cross_covariance(x, x)
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)
    np.testing.assert_allclose(cov, np.corrcoef(x.T), atol=1e-14)


def test_partial_rank_degenerate_covariates_return_no_association():
    x = np.arange(12.)
    assert partial_rank(x, x, x[:, None]) is None


def test_rank_one_transport_and_fixed_current_derivative():
    # Two-way coupling, unlike the directed DendriNet training model.
    matrix = np.array([[3., -1., 0.], [-1., 4., -1.], [0., -1., 2.]])
    inverse = np.linalg.inv(matrix)
    k, soma, eta = 1, 0, 2.
    q = inverse[:, soma]
    column = inverse[:, k]
    prediction = q-eta*column*q[k]/(1+eta*inverse[k, k])
    changed = matrix.copy(); changed[k, k] += eta
    np.testing.assert_allclose(prediction, np.linalg.inv(changed)[:, soma])
    # Current injection changes the RHS only; transport is exactly unchanged.
    rhs = np.array([.1, .2, .3])
    current_rhs = rhs.copy(); current_rhs[k] -= .4
    eps = 1e-6
    perturb = np.eye(3)[soma]*eps
    finite = (np.linalg.solve(matrix, current_rhs+perturb)-np.linalg.solve(matrix, current_rhs-perturb))/(2*eps)
    np.testing.assert_allclose(finite, q, rtol=1e-9)
