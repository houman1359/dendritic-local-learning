"""Exact finite-domain diagnostic of scalar nonlinear tree bottlenecks.

Every internal node has u=a+b*l+c*r+d*l*r; leaves see one named input.
This is an algebraic tree surrogate, not a conductance or spiking neuron.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class Tree:
    name: str
    shape: str
    permutation: tuple
    children: dict
    descendants: dict
    parent: dict
    root: int


def build_tree(name, shape, permutation):
    children, descendants, parent = {}, {i: (i,) for i in range(8)}, {}

    def join(leaves, is_root=False):
        if len(leaves) == 1:
            return int(leaves[0])
        cut = len(leaves) // 2 if shape == "balanced" else 1
        if shape == "mixed":
            cut = 3 if is_root else len(leaves) // 2
        left, right = join(leaves[:cut]), join(leaves[cut:])
        node = 8 + len(children)
        children[node] = (left, right)
        descendants[node] = descendants[left] + descendants[right]
        parent[left], parent[right] = (node, 0), (node, 1)
        return node

    root = join(list(permutation), True)
    return Tree(name, shape, tuple(permutation), children, descendants, parent, root)


def candidates():
    rng = np.random.default_rng(901_60905)
    permutations = [tuple(range(8)), (0, 4, 2, 6, 1, 5, 3, 7)]
    permutations += [tuple(rng.permutation(8)) for _ in range(2)]
    return [build_tree(f"{shape}_p{i}", shape, p)
            for shape in ("balanced", "comb", "mixed")
            for i, p in enumerate(permutations)]


def domain():
    bits = (np.arange(256)[:, None] >> np.arange(8)) & 1
    return 2.0 * bits - 1.0


def fourier_design(x):
    result = np.ones((len(x), 256))
    for mask in range(1, 256):
        leaf_bit = mask & -mask
        leaf = leaf_bit.bit_length() - 1
        result[:, mask] = result[:, mask ^ leaf_bit] * x[:, leaf]
    return result


def matchings(items):
    if not items:
        yield ()
        return
    first = items[0]
    for j in range(1, len(items)):
        pair = (first, items[j])
        rest = items[1:j] + items[j + 1:]
        for remaining in matchings(rest):
            yield (pair,) + remaining


def tasks():
    records = []
    for i, pairs in enumerate(matchings(tuple(range(8)))):
        coeff = np.zeros(256)
        for pair in pairs:
            coeff[sum(1 << j for j in pair)] = 0.5
        records.append(dict(task_id=f"matching_{i:03d}", family="quadratic_matching",
                            coefficients=coeff, supports=pairs))
    # Every unordered 4+4 partition exactly once: the first set contains input0.
    for i, other in enumerate(itertools.combinations(range(1, 8), 3)):
        first = (0,) + other
        second = tuple(j for j in range(8) if j not in first)
        coeff = np.zeros(256)
        for subset in (first, second):
            coeff[sum(1 << j for j in subset)] = 0.5
        records.append(dict(task_id=f"quartet_{i:03d}", family="quartic_partition",
                            coefficients=coeff, supports=(first, second)))
    return records


def input_gradient_covariance(coeff):
    deriv = np.zeros((8, 256))
    for mask in np.flatnonzero(coeff):
        for leaf in range(8):
            if mask & (1 << leaf):
                deriv[leaf, mask ^ (1 << leaf)] = coeff[mask]
    return deriv @ deriv.T


def matricize(coeff, subset):
    rest = tuple(j for j in range(8) if j not in subset)
    result = np.zeros((1 << len(subset), 1 << len(rest)))
    for mask, value in enumerate(coeff):
        row = sum(((mask >> leaf) & 1) << j for j, leaf in enumerate(subset))
        col = sum(((mask >> leaf) & 1) << j for j, leaf in enumerate(rest))
        result[row, col] = value
    return result


def cut_scores(coeff, tree):
    """Valid lower bounds on population MSE; sum is only a heuristic score.

    A scalar subtree enters its ancestors affinely: f=A(rest)+h(subtree)B(rest).
    After removing the constant subtree row, its Fourier matrix has rank <=1.
    Orthogonal projection and Eckart-Young give each tail-energy lower bound.
    Taking the maximum preserves a bound; summing overlapping cuts does not.
    """
    centered, general, rows = [], [], []
    for node in tree.children:
        if node == tree.root:
            continue
        matrix = matricize(coeff, tree.descendants[node])
        singular = np.linalg.svd(matrix, compute_uv=False)
        c_singular = np.linalg.svd(matrix[1:], compute_uv=False)
        full_tail = float(np.sum(singular[2:] ** 2))
        tail = float(np.sum(c_singular[1:] ** 2))
        general.append(full_tail)
        centered.append(tail)
        rows.append(dict(node=node, support=list(tree.descendants[node]),
                         full_rank2_tail=full_tail, centered_rank1_tail=tail))
    energy = float(coeff @ coeff)
    return dict(centered_cut_bound=max(centered) / energy,
                centered_cut_sum=sum(centered) / energy,
                full_cut_bound=max(general) / energy,
                cut_details=rows)


def forward(x, tree, weights):
    values = np.zeros((15, len(x)))
    values[:8] = x.T
    for node, (left, right) in tree.children.items():
        a, b, c, d = weights[node - 8]
        values[node] = a + b * values[left] + c * values[right] + d * values[left] * values[right]
    return values


def sensitivity(values, tree, weights):
    deriv = np.zeros_like(values)
    deriv[tree.root] = 1.0
    for node in reversed(tree.children):
        left, right = tree.children[node]
        _, b, c, d = weights[node - 8]
        deriv[left] = deriv[node] * (b + d * values[right])
        deriv[right] = deriv[node] * (c + d * values[left])
    return deriv


def stabilize_gauge(x, tree, weights, node):
    if node == tree.root:
        return
    values = forward(x, tree, weights)[node]
    mean, scale = float(values.mean()), float(values.std())
    if scale < 1e-10:
        return
    weights[node - 8, 0] -= mean
    weights[node - 8] /= scale
    parent, side = tree.parent[node]
    a, b, c, d = weights[parent - 8].copy()
    if side == 0:
        weights[parent - 8] = (a + b * mean, b * scale, c + d * mean, d * scale)
    else:
        weights[parent - 8] = (a + c * mean, b + d * mean, c * scale, d * scale)


def fit(x, y, tree, seed, sweeps=32, checkpoints=(0, 2, 8, 32), diagnostics=False):
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.5, (7, 4))
    weights[:, 0] = 0.0
    for node in tree.children:
        stabilize_gauge(x, tree, weights, node)
    target_energy = float(y @ y / len(y))
    curve = {0: float(np.mean((forward(x, tree, weights)[tree.root] - y) ** 2) / target_energy)}
    dead_blocks, minimum_rank, max_norm = 0, 4, float(np.linalg.norm(weights))
    for sweep in range(1, sweeps + 1):
        order = list(tree.children)
        if sweep % 2 == 0:
            order.reverse()
        for node in order:
            values = forward(x, tree, weights)
            deriv = sensitivity(values, tree, weights)[node]
            left, right = tree.children[node]
            design = np.column_stack((np.ones(len(y)), values[left], values[right],
                                      values[left] * values[right])) * deriv[:, None]
            offset = values[tree.root] - deriv * values[node]
            solution, _, rank, singular = np.linalg.lstsq(design, y - offset, rcond=1e-12)
            minimum_rank = min(minimum_rank, int(rank))
            dead_blocks += int(singular[0] < 1e-12)
            # An exact coordinate minimizer. No fitting hyperparameter or held-out outcomes.
            weights[node - 8] = solution
            stabilize_gauge(x, tree, weights, node)
            max_norm = max(max_norm, float(np.linalg.norm(weights)))
        if sweep in checkpoints or sweep == sweeps:
            curve[sweep] = float(np.mean((forward(x, tree, weights)[tree.root] - y) ** 2) / target_energy)
            if not np.isfinite(curve[sweep]):
                raise FloatingPointError(f"Nonfinite fit: {tree.name}, seed={seed}, sweep={sweep}")
    if diagnostics:
        return weights, curve, dict(dead_coordinate_blocks=dead_blocks, minimum_coordinate_rank=minimum_rank,
            final_parameter_norm=float(np.linalg.norm(weights)), maximum_parameter_norm=max_norm)
    return weights, curve
