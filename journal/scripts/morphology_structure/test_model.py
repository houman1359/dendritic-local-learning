import unittest
import numpy as np
from model import (candidates, tasks, domain, fourier_design, input_gradient_covariance,
                   cut_scores, forward, sensitivity, stabilize_gauge, fit)


class StructureTests(unittest.TestCase):
    def test_isospectral_complete_task_enumeration(self):
        records = tasks()
        self.assertEqual(len(records), 140)
        self.assertEqual(len({tuple(t['coefficients']) for t in records}), 140)
        for task in records:
            np.testing.assert_allclose(input_gradient_covariance(task['coefficients']), np.eye(8)/4, atol=1e-14)
        x = domain()
        f = fourier_design(x)
        np.testing.assert_allclose(f.T @ f/256, np.eye(256), atol=1e-14)

    def test_realized_trees_obey_scalar_cut_constraint(self):
        x, rng = domain(), np.random.default_rng(134)
        f = fourier_design(x)
        for tree in candidates():
            w = rng.normal(size=(7,4))
            coeff = f.T @ forward(x,tree,w)[tree.root]/256
            self.assertLess(cut_scores(coeff,tree)['centered_cut_bound'], 1e-25)

    def test_coordinate_design_matches_finite_perturbation(self):
        x, rng = domain(), np.random.default_rng(135)
        for tree in candidates():
            w = rng.normal(size=(7,4))
            values = forward(x,tree,w)
            deriv = sensitivity(values,tree,w)
            for node,(left,right) in tree.children.items():
                features = np.array([np.ones(256),values[left],values[right],values[left]*values[right]])
                delta = rng.normal(size=4)
                changed = w.copy()
                changed[node-8] += delta
                np.testing.assert_allclose(forward(x,tree,changed)[tree.root]-values[tree.root],
                    deriv[node] * (delta @ features), rtol=1e-10, atol=1e-10)

    def test_gauge_preserves_output(self):
        x, rng = domain(), np.random.default_rng(136)
        for tree in candidates():
            w = rng.normal(size=(7,4))
            before = forward(x,tree,w)[tree.root].copy()
            for node in tree.children:
                stabilize_gauge(x,tree,w,node)
            np.testing.assert_allclose(forward(x,tree,w)[tree.root],before,rtol=1e-10,atol=1e-10)

    def test_known_representable_matching_and_als_monotonicity(self):
        x, tree = domain(), candidates()[0]
        y = fourier_design(x) @ tasks()[0]['coefficients']
        w = np.zeros((7,4))
        for node,(left,right) in tree.children.items():
            w[node-8] = (0,0,0,1) if left<8 and right<8 else (0,1,1,0)
        w[tree.root-8] *= 0.5
        np.testing.assert_allclose(forward(x,tree,w)[tree.root],y,atol=1e-14)
        _,curve = fit(x,y,tree,100,sweeps=8,checkpoints=tuple(range(9)))
        self.assertTrue(np.all(np.diff(list(curve.values())) <= 1e-10))
        self.assertLess(curve[8],1e-8)


if __name__ == '__main__':
    unittest.main()
