"""Independent hand-derived checks, exact certificates, and bound falsification checks."""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("boolean_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_all_labeled_unordered_four_leaf_shapes_once():
    trees = audit.all_trees()
    assert len(trees) == 15
    assert len({audit.label(t) for t in trees}) == 15
    assert sum(audit.depth(t) == 2 for t in trees) == 3
    assert sum(audit.depth(t) == 3 for t in trees) == 12
    assert all(audit.support(t) == (0, 1, 2, 3) for t in trees)


def test_xor_of_and_crossed_matrix_exact_hand_derivation():
    vals = audit.truth("xor_of_ands")
    coeff = audit.walsh(vals)
    assert coeff[0] == Q(3, 8)
    assert sum(v * v for v in coeff[1:]) == Q(15, 64)
    matrix = audit.centered_matrix(coeff, (0, 2))
    expected = [[1, 1, -1, -1], [1, -1, 1, -1], [-1, -1, -1, -1]]
    assert matrix == [[Q(v, 8) for v in row] for row in expected]
    assert audit.exact_rank(matrix) == 3
    cut = audit.cut_audit(coeff, (0, 2), Q(15, 64))
    np.testing.assert_allclose(cut["raw_mse_lower_bound"], 1 / 8, atol=1e-14)
    np.testing.assert_allclose(cut["normalized_mse_lower_bound"], 8 / 15, atol=1e-14)


def test_three_distinct_grouping_obstructions_and_depth_control():
    expected = {"or_of_ands": ((15 - np.sqrt(33)) / 128, (30 - 2*np.sqrt(33)) / 63),
                "xor_of_ands": (1/8, 8/15), "and_of_xors": (1/8, 2/3)}
    for family, (raw, normalized) in expected.items():
        coeff = audit.walsh(audit.truth(family))
        variance = sum(v * v for v in coeff[1:])
        assert audit.exact_rank(audit.centered_matrix(coeff, (0, 1))) == 1
        for cut in ((0, 2), (0, 3)):
            result = audit.cut_audit(coeff, cut, variance)
            assert result["centered_rank"] == 3
            np.testing.assert_allclose(result["raw_mse_lower_bound"], raw, atol=1e-14)
            np.testing.assert_allclose(result["normalized_mse_lower_bound"], normalized, atol=1e-14)
    nested = audit.truth("nested")
    coeff = audit.walsh(nested)
    balanced_bounds = []
    for tree in audit.all_trees():
        if audit.depth(tree) == 2:
            assert audit.exact_construction(nested, tree) is None
            balanced_bounds.append(max(audit.cut_audit(coeff, audit.support(node), Q(55, 256))["normalized_mse_lower_bound"]
                                       for node in audit.nodes(tree)[:-1]))
    np.testing.assert_allclose(min(balanced_bounds), .1254605659901545, atol=1e-14)
    assert audit.exact_construction(nested, (0, (1, (2, 3)))) is not None


def test_all_zero_rank_obstructions_have_exact_box_bounded_certificates():
    for family in audit.FAMILIES:
        values = audit.truth(family)
        coeff = audit.walsh(values)
        for tree in audit.all_trees():
            rank_ok = all(audit.exact_rank(audit.centered_matrix(coeff, audit.support(node))) <= 1
                          for node in audit.nodes(tree)[:-1])
            cert = audit.exact_construction(values, tree)
            assert (cert is not None) == rank_ok
            if rank_ok:
                assert cert["raw_truth_table_verified_exactly"]
                assert Q(cert["max_normalized_coefficient_squared_rational"]) <= 4
        if family in ("and4", "or4", "parity4"):
            assert all(audit.exact_construction(values, t) is not None for t in audit.all_trees())


def test_lower_bounds_hold_for_independently_evaluated_random_polynomial_trees():
    # Not a capacity proof: this catches normalization, cut orientation, and
    # accidental summing of overlapping cuts in the exported bound computation.
    rng = np.random.default_rng(190505)
    def forward(tree, weights, row):
        if isinstance(tree, int):
            return row[tree]
        l, r = forward(tree[0], weights, row), forward(tree[1], weights, row)
        a, b, c, d = weights[audit.label(tree)]
        return a + b*l + c*r + d*l*r
    for family in audit.FAMILIES:
        y = np.asarray(audit.truth(family), dtype=float)
        coeff = audit.walsh(audit.truth(family))
        variance = np.var(y)
        z = (y-y.mean()) / np.sqrt(variance)
        for tree in audit.all_trees():
            bound = max(audit.cut_audit(coeff, audit.support(node), Q(float(variance)))["normalized_mse_lower_bound"]
                        for node in audit.nodes(tree)[:-1])
            for _ in range(5):
                weights = {audit.label(n): rng.uniform(-2, 2, 4) for n in audit.nodes(tree)}
                pred = np.array([forward(tree, weights, row) for row in audit.X])
                assert np.mean((pred-z)**2) + 1e-12 >= bound


def test_canonical_gate_derivative_sign_and_finite_difference():
    for gate in ("AND", "OR", "XOR"):
        for u, v in ((.2, .3), (.4, .9)):
            _, du, dv = audit.gate_values(gate, u, v)
            eps = 1e-6
            numeric_u = (audit.gate_values(gate, u+eps, v)[0] - audit.gate_values(gate, u-eps, v)[0]) / (2*eps)
            numeric_v = (audit.gate_values(gate, u, v+eps)[0] - audit.gate_values(gate, u, v-eps)[0]) / (2*eps)
            np.testing.assert_allclose([du, dv], [numeric_u, numeric_v], atol=1e-9)
    assert audit.gate_values("XOR", .3, 0)[1] == 1
    assert audit.gate_values("XOR", .3, 1)[1] == -1
    assert audit.gate_values("OR", .3, 1)[1] == 0
    assert audit.gate_values("AND", .3, 0)[1] == 0
