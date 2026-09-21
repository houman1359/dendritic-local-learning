"""Support-only lower bounds; unrestricted functions on each output support.

These are population approximation bounds for affine sums. They are not the
capacity or learning curve of a particular local activation or nonlinear soma.
"""
import math

import numpy as np


def union_inclusion_probability(d, s, order, contacts):
    """Probability a fixed order-subset is in the union of iid s-subsets.

    A stable occupancy recursion avoids cancellation in inclusion-exclusion.
    """
    if not (0 <= s <= d and 0 <= order <= d and contacts >= 0 and int(contacts) == contacts):
        raise ValueError("Invalid support or contact count")
    state = np.zeros(order + 1, dtype=np.float64)
    state[0] = 1
    denominator = math.comb(d, s)
    transition = np.zeros((order + 1, order + 1), dtype=np.float64)
    for covered in range(order + 1):
        missing = order - covered
        for hit in range(max(0, s - (d - missing)), min(missing, s) + 1):
            transition[covered, covered + hit] = (math.comb(missing, hit)
                                                       * math.comb(d - missing, s - hit) / denominator)
    for _ in range(int(contacts)):
        state = state @ transition
    return float(state[-1])


def support_counts(raw_supports, masks, group_size=1):
    """Number of independent output supports containing each target monomial."""
    raw = np.asarray(raw_supports)
    masks = np.asarray(masks)
    if raw.ndim != 2 or masks.ndim != 2 or group_size < 1 or len(raw) % group_size:
        raise ValueError("Invalid support grouping")
    if (not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0)
            or np.any(raw >= masks.shape[1]) or not np.isin(masks, [0, 1]).all()):
        raise ValueError("Invalid raw indices or masks")
    indicator = np.zeros((len(raw), masks.shape[1]), dtype=np.int16)
    indicator[np.arange(len(raw))[:, None], raw] = 1
    unions = indicator.reshape(-1, group_size, masks.shape[1]).max(1)
    counts = np.zeros(len(masks), dtype=np.int64)
    # Avoid a large sites x teacher-terms temporary at wide local banks.
    for block in np.array_split(unions, max(1, math.ceil(len(unions) / 1024))):
        counts += ((block @ masks.T) == masks.sum(1)[None, :]).sum(0)
    return counts


def affine_support_floor(raw_supports, packet, group_size=1):
    counts = support_counts(raw_supports, packet["masks"], group_size)
    return float(np.square(packet["coefficients"])[counts == 0].sum())
