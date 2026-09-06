"""Finite, noisy observation bridge for the eight-leaf algebraic tree model.

Selectors receive noisy input/output samples only; full target coefficients are
owned by the task generator and are used for sealed retrospective evaluation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

STRUCTURE = Path(__file__).resolve().parents[1] / "morphology_structure"
sys.path.insert(0, str(STRUCTURE))
from model import (candidates, fourier_design, input_gradient_covariance,
                   cut_scores, fit, forward, domain, Tree)
from constructive_dp_v2 import tree_from_coeff

FAMILIES = ("matching", "quartet", "nested_prefix", "random_interactions")
CALIBRATION_SIZES = (64, 256, 1024)
CALIBRATION_NOISE = (0.0, 0.5)
ALPHA_GRID = (0.15, 0.35, 0.65)
TIE_TOLERANCE = 1e-10
FAILURE_LOSS = 1e6


def rng(seed, family, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(family), int(stream)]))


def task(seed, family):
    generator = rng(seed, family, 1)
    permutation = generator.permutation(8)
    if family == 0:
        supports = [tuple(permutation[i:i+2]) for i in range(0, 8, 2)]
    elif family == 1:
        supports = [tuple(permutation[:4]), tuple(permutation[4:])]
    elif family == 2:
        supports = [tuple(permutation[:k]) for k in (2, 4, 6, 8)]
    elif family == 3:
        eligible = [mask for mask in range(1, 256) if mask.bit_count() >= 2]
        masks = generator.choice(eligible, size=4, replace=False)
        supports = [tuple(i for i in range(8) if mask & (1 << i)) for mask in masks]
    else:
        raise ValueError(family)
    values = generator.uniform(0.5, 1.5, len(supports))
    values *= generator.choice((-1.0, 1.0), len(supports))
    values /= np.linalg.norm(values)
    coeff = np.zeros(256)
    for support, value in zip(supports, values):
        coeff[sum(1 << int(i) for i in support)] = value
    assert np.isclose(coeff @ coeff, 1.0) and coeff[0] == 0
    return coeff


def sample(coeff, seed, family, stream, count, noise):
    generator = rng(seed, family, stream)
    patterns = generator.integers(0, 256, count)
    x = domain()[patterns]
    clean = fourier_design(x) @ coeff
    y = clean + float(noise) * generator.normal(size=count)
    return x, y, clean, patterns


def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str(a.shape).encode()); digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def estimate_coefficients(x, y, alpha_scale):
    """Generic 255-monomial Lasso; no family, supports or true coefficients."""
    start = time.perf_counter()
    design = fourier_design(x)[:, 1:]
    alpha = float(alpha_scale * max(np.std(y), 1e-12)
                  * np.sqrt(2 * np.log(256) / len(y)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000,
                       tol=1e-8, selection="cyclic").fit(design, y)
    coeff = np.r_[0.0, fitted.coef_]  # global intercept does not affect cut compatibility
    eigenvalues = np.linalg.eigvalsh(input_gradient_covariance(coeff))[::-1]
    energy = float(eigenvalues.sum())
    rank = int(np.searchsorted(np.cumsum(eigenvalues), 0.995 * energy) + 1) if energy > 1e-18 else 0
    return coeff, dict(alpha=alpha, intercept=float(fitted.intercept_),
        nonzero_coefficients=int(np.count_nonzero(coeff)), observed_gradient_rank995=rank,
        coefficient_energy=float(coeff @ coeff), converged=not bool(caught),
        iterations=int(fitted.n_iter_), seconds=time.perf_counter()-start)


def score_coefficients(coeff, trees):
    scores = []
    for tree in trees:
        if float(coeff @ coeff) <= 1e-18:
            scores.append((tree.name, 0.0, 0.0))
        else:
            item = cut_scores(coeff, tree)
            scores.append((tree.name, item["centered_cut_bound"], item["centered_cut_sum"]))
    return scores


def select_scores(scores):
    """Primary max cut tail; summed overlapping tails only break max-score ties."""
    minimum = min(row[1] for row in scores)
    remaining = [row for row in scores if row[1] <= minimum + TIE_TOLERANCE]
    secondary = min(row[2] for row in remaining)
    return min(row[0] for row in remaining if row[2] <= secondary + TIE_TOLERANCE)


def tree_payload(tree):
    return dict(name=tree.name, shape=tree.shape, permutation=list(map(int, tree.permutation)),
        children={str(k): list(v) for k, v in tree.children.items()})


def tree_from_payload(payload):
    children = {int(k): tuple(v) for k, v in payload["children"].items()}
    descendants, parent = {i: (i,) for i in range(8)}, {}
    for node, (left, right) in children.items():
        descendants[node] = descendants[left] + descendants[right]
        parent[left], parent[right] = (node, 0), (node, 1)
    return Tree(payload["name"], payload["shape"], tuple(payload["permutation"]),
                children, descendants, parent, max(children))


def adaptive_tree(coeff, name):
    if float(coeff @ coeff) <= 1e-18:
        tree = candidates()[0]
        payload = tree_payload(tree); payload["name"] = name
        return tree_from_payload(payload), 0.0
    tree, bound, _ = tree_from_coeff(coeff, name)
    return tree, bound


def pilot(x_fit, y_fit, x_gate, y_gate, seed, trees):
    start = time.perf_counter()
    rows = []
    for tree in trees:
        try:
            weights, _ = fit(x_fit, y_fit, tree, seed, sweeps=2, checkpoints=(2,))
            value = float(np.mean((forward(x_gate, tree, weights)[tree.root] - y_gate) ** 2))
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite pilot")
            status = "completed"
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            value, status = FAILURE_LOSS, type(error).__name__
        rows.append(dict(candidate_id=tree.name, pilot_gate_mse=value, pilot_status=status))
    selected = min(rows, key=lambda row: (round(row["pilot_gate_mse"], 10), row["candidate_id"]))["candidate_id"]
    return selected, rows, time.perf_counter()-start


def train_candidate(tree, coeff, seed, family, config):
    """All final restarts selected using independent noisy validation only."""
    x, y, _, _ = sample(coeff, seed, family, 10, config["training_rows"], config["training_noise"])
    xv, yv, _, _ = sample(coeff, seed, family, 11, config["validation_rows"], config["training_noise"])
    xt, yt, clean, _ = sample(coeff, seed, family, 12, config["test_rows"], config["training_noise"])
    full = domain(); exact_target = fourier_design(full) @ coeff
    results = []
    for restart in range(config["restarts"]):
        tick = time.perf_counter()
        fit_seed = int(seed * 100 + family * 10 + restart)
        try:
            weights, curve, diagnostics = fit(x, y, tree, fit_seed, sweeps=config["sweeps"],
                checkpoints=(0, 2, 8, config["sweeps"]), diagnostics=True)
            valid = float(np.mean((forward(xv, tree, weights)[tree.root] - yv) ** 2))
            pred = forward(xt, tree, weights)[tree.root]
            test = float(np.mean((pred - clean) ** 2))
            noisy_test = float(np.mean((pred - yt) ** 2))
            exact = float(np.mean((forward(full, tree, weights)[tree.root] - exact_target) ** 2))
            if not np.all(np.isfinite((valid, test, noisy_test, exact))):
                raise FloatingPointError("nonfinite endpoint")
            status = "completed"
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            valid = test = noisy_test = exact = FAILURE_LOSS
            status = type(error).__name__; curve = {}; diagnostics = {}
        results.append(dict(candidate_id=tree.name, shape=tree.shape, restart=restart,
            fit_seed=fit_seed, validation_mse=valid, test_nmse=test, noisy_test_mse=noisy_test,
            exact_population_nmse=exact, status=status, fit_seconds=time.perf_counter()-tick,
            training_curve=curve, **diagnostics))
    selected = min(results, key=lambda row: (row["validation_mse"], row["restart"]))["restart"]
    for row in results:
        row["selected_by_validation"] = row["restart"] == selected
    return results
