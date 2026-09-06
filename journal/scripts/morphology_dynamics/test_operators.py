"""Focused mathematical tests: explicit updates, not a second copy of formulas."""
import itertools
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss

from operators import empirical_fullbatch_spectral, gaussian_fullbatch_spectral, gaussian_moments


class OperatorTests(unittest.TestCase):
    def test_empirical_spectral_matches_direct_updates_and_heldout_risk(self):
        rng = np.random.default_rng(70)
        z = rng.normal(size=(40, 5)); z[:, 4] = z[:, 3]  # Singular Hessian.
        y = rng.normal(size=40)
        ev = rng.normal(size=(17, 5)); ey = rng.normal(size=17)
        predicted = empirical_fullbatch_spectral(z, y, ev, ey, .07, [1, 16, 256])
        w = np.zeros(5)
        for step in range(1, 257):
            w -= .07*z.T @ (z @ w-y)/len(z)
            if step in predicted:
                np.testing.assert_allclose(predicted[step]['weights'], w, atol=2e-13)
                self.assertAlmostEqual(predicted[step]['loss'], .5*np.mean((ev @ w-ey)**2), places=13)

    def test_population_fullbatch_spectral_matches_recurrence(self):
        rng = np.random.default_rng(71)
        c = rng.normal(size=(3, 2, 4))*.3  # K>r and nontrivial zero modes.
        targets = rng.normal(size=(2, 3)); p = np.array([.4, .6]); noise = np.array([.2, .7])
        spectral = gaussian_fullbatch_spectral(c, targets, p, noise, .2, [1, 16, 256])
        direct = gaussian_moments(c, targets, p, noise, .2, np.inf, [1, 16, 256])
        for step in spectral:
            np.testing.assert_allclose(spectral[step]['mean'], direct[step]['mean'], atol=3e-13)
            np.testing.assert_allclose(spectral[step]['loss'], direct[step]['loss'], atol=3e-13)

    def test_sgd_second_moment_matches_exact_gaussian_quadrature(self):
        # Three-node Gauss-Hermite quadrature is exact for the degree-four input
        # polynomials in the update Gram. Enumerate all B=2 batches explicitly.
        c = np.array([[.7, -.2], [.3, .8]])
        target = np.array([[.6, -.5], [.1, .4]])
        probs = np.array([.35, .65]); noise = np.array([.2, .3]); eta = .13
        initial = np.array([[[.1, .2], [-.3, .7]], [[-.4, .2], [.3, -.1]]])
        mean = initial.mean(axis=0)
        cov = np.mean((initial-mean) @ (initial-mean).transpose(0, 2, 1), axis=0)
        nodes, weights = hermgauss(3); nodes *= np.sqrt(2); weights /= np.sqrt(np.pi)
        samples = []
        for context in range(2):
            for inds in itertools.product(range(3), repeat=3):
                x = nodes[list(inds[:2])]
                eps = np.sqrt(noise[context])*nodes[inds[2]]
                y = target[context] @ x+eps
                weight = probs[context]*np.prod(weights[list(inds)])
                samples.append((c[context], x, y, weight))
        expected_mean = np.zeros_like(mean); expected_gram = np.zeros_like(cov)
        for v in initial:
            for first, second in itertools.product(samples, repeat=2):
                grad = np.zeros_like(v)
                for cc, xx, yy, _ in [first, second]:
                    grad += np.outer(cc, xx)*(cc @ v @ xx-yy)/2
                updated = v-eta*grad
                weight = .5*first[3]*second[3]
                expected_mean += weight*updated
                expected_gram += weight*updated @ updated.T
        got = gaussian_moments(c[None], target, probs, noise, eta, 2, [1], mean[None], cov[None])[1]
        np.testing.assert_allclose(got['mean'][0], expected_mean, atol=2e-14)
        np.testing.assert_allclose(got['covariance_gram'][0]+expected_mean @ expected_mean.T, expected_gram, atol=2e-14)


if __name__ == '__main__':
    unittest.main()
