"""Four-input scalar multi-affine trees, without target-dependent weights."""
from __future__ import annotations
import itertools
import numpy as np

FAMILIES = ('and4', 'or4', 'parity4', 'or_of_ands', 'xor_of_ands', 'and_of_xors', 'nested')
TREES = {
    'balanced_ab_cd': ((0, 1), (2, 3)),
    'balanced_ac_bd': ((0, 2), (1, 3)),
    'balanced_ad_bc': ((0, 3), (1, 2)),
    'comb_a_b_cd': (0, (1, (2, 3))),
}


def domain():
    return np.array(list(itertools.product((-1., 1.), repeat=4)))


def raw_target(x, family, permutation=(0, 1, 2, 3)):
    bits = (np.asarray(x)[..., permutation] + 1.) / 2.
    a, b, c, d = [bits[..., i].astype(bool) for i in range(4)]
    values = {
        'and4': a & b & c & d,
        'or4': a | b | c | d,
        'parity4': a ^ b ^ c ^ d,
        'or_of_ands': (a & b) | (c & d),
        'xor_of_ands': (a & b) ^ (c & d),
        'and_of_xors': (a ^ b) & (c ^ d),
        'nested': a & (b | (c & d)),
    }
    return values[family].astype(float)


def normalization(family):
    raw = raw_target(domain(), family)
    return float(raw.mean()), float(raw.std())


def target(x, family, permutation=(0, 1, 2, 3)):
    mean, sd = normalization(family)
    return (raw_target(x, family, permutation) - mean) / sd


def pack_tree(tree, permutation=(0, 1, 2, 3)):
    children = []
    def visit(node):
        if isinstance(node, int):
            return int(permutation[node])
        left, right = visit(node[0]), visit(node[1])
        idx = 4 + len(children)
        children.append((left, right))
        return idx
    assert visit(tree) == 6 and len(children) == 3
    return np.array(children, dtype=int)


def forward(x, weights, children):
    """x is common (batch,4) or condition-specific (condition,batch,4)."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.broadcast_to(x, (len(weights), *x.shape))
    count, batch, n_inputs = x.shape
    assert n_inputs == 4 and weights.shape == (count, 3, 4)
    values = np.empty((count, 7, batch))
    values[:, :4] = x.transpose(0, 2, 1)
    ix = np.arange(count)
    for node in range(3):
        left = values[ix, children[:, node, 0]]
        right = values[ix, children[:, node, 1]]
        a, b, c, d = weights[:, node, :].T
        values[:, 4 + node] = a[:, None] + b[:, None]*left + c[:, None]*right + d[:, None]*left*right
    return values


def gradient(x, y, weights, children, broadcast):
    """Gradient of half mean squared normalized-label error.

    All nonroot sensitivities are one under unit broadcast. The root is one
    under both rules; local feature derivatives remain unchanged.
    """
    values = forward(x, weights, children)
    count, _, batch = values.shape
    ix = np.arange(count)
    q = np.zeros_like(values)
    q[:, 6] = 1.
    features = np.empty((count, 3, batch, 4))
    for node in range(2, -1, -1):
        left_idx, right_idx = children[:, node, 0], children[:, node, 1]
        left, right = values[ix, left_idx], values[ix, right_idx]
        a, b, c, d = weights[:, node, :].T
        q[ix, left_idx] = q[:, node+4] * (b[:, None]+d[:, None]*right)
        q[ix, right_idx] = q[:, node+4] * (c[:, None]+d[:, None]*left)
        features[:, node] = np.stack((np.ones_like(left), left, right, left*right), axis=-1)
    routed = q[:, 4:].copy()
    routed[np.asarray(broadcast, dtype=bool), :2] = 1.
    residual = values[:, 6] - np.asarray(y)
    delivered = np.einsum('cb,cjb,cjbf->cjf', residual, routed, features)/batch
    exact = np.einsum('cb,cjb,cjbf->cjf', residual, q[:, 4:], features)/batch
    return delivered, exact, values[:, 6], q[:, 4:]


def classification(prediction, raw_labels, mean, sd):
    decision = prediction >= (0.5-mean)/sd
    truth = raw_labels.astype(bool)
    accuracy = np.mean(decision == truth, axis=1)
    tpr = np.sum(decision & truth, axis=1) / np.sum(truth, axis=1)
    tnr = np.sum(~decision & ~truth, axis=1) / np.sum(~truth, axis=1)
    return accuracy, (tpr+tnr)/2
