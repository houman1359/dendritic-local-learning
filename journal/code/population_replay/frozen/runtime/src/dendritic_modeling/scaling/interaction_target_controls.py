"""Teacher organization controls with identical energy at every Walsh degree.

These one-term-per-order diagnostic targets complement, and do not replace,
the eight-term primary pilot. No model receives the teacher permutation.
"""
from copy import deepcopy

import numpy as np

from dendritic_modeling.scaling.order_spectrum import target_packet


def organization_pair(d=64, rho=0.5, seed=62001, k_min=2):
    """Independent supports versus prefixes of a hidden random permutation.

    Both targets have exactly the same signed coefficient at each degree. Every
    individual nested support is marginally uniform among subsets of that size,
    while adjacent supports share structure. This does not match higher moments
    of the label distribution, finite-data kernel risk, or neural trainability.
    """
    independent = target_packet(d=d, rho=rho, seed=seed, k_min=k_min, terms_per_order=1)
    independent["organization"] = "independent_orders"
    nested = deepcopy(independent)
    nested["organization"] = "nested_orders"
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, 1771]))
    permutation = rng.permutation(d)
    nested["masks"][:] = 0
    for row, order in enumerate(nested["orders"]):
        nested["masks"][row, permutation[:order]] = 1
    # Teacher-only metadata, never an architecture or routing argument.
    nested["teacher_permutation"] = permutation
    return independent, nested


def support_organization(packet):
    """Descriptive orderwise overlaps; these are not minimum leap complexity."""
    seen, previous, rows = set(), set(), []
    for mask, order in zip(packet["masks"], packet["orders"]):
        support = set(np.flatnonzero(mask).tolist())
        rows.append({"order": int(order),
                     "new_coordinates_vs_preceding_support": len(support - previous),
                     "new_coordinates_vs_all_lower_orders": len(support - seen),
                     "preceding_support_is_subset": bool(previous <= support)})
        seen |= support
        previous = support
    return rows
