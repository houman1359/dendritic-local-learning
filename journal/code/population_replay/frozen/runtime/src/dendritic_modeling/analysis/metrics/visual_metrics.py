from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

"""Metrics for characterising visual neuron responses.

Functions follow definitions commonly used in V1 literature.
They operate on *numpy arrays* or *PyTorch tensors*; tensors are detached
and moved to CPU automatically.
"""

__all__ = [
    "direction_selectivity_index",
    "length_selectivity_index",
    "orientation_selectivity_index",
    "size_tuning_index",
]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def orientation_selectivity_index(
    responses: Sequence[float], orientations: Sequence[float]
) -> float:
    """Classic OSI = (R_pref - R_orth) / (R_pref + R_orth).

    Parameters
    ----------
    responses
        Response magnitude for each test orientation (same order as *orientations*).
    orientations
        Stimulus orientations in **degrees**.
    """
    r = _to_numpy(responses).ravel()
    orientations = np.deg2rad(_to_numpy(orientations).ravel())
    idx_pref = r.argmax()

    # orientation orthogonal to preferred
    orth_angle = (orientations[idx_pref] + np.pi / 2) % np.pi
    idx_orth = int(
        np.argmin(np.abs(((orientations - orth_angle + np.pi) % np.pi) - np.pi / 2))
    )
    return float((r[idx_pref] - r[idx_orth]) / (r[idx_pref] + r[idx_orth] + 1e-9))


def direction_selectivity_index(
    responses: Sequence[float], directions: Sequence[float]
) -> float:
    """DSI = (R_pref - R_null) / (R_pref + R_null)."""
    r = _to_numpy(responses).ravel()
    directions = np.deg2rad(_to_numpy(directions).ravel())
    idx_pref = r.argmax()
    null_angle = (directions[idx_pref] + np.pi) % (2 * np.pi)
    idx_null = int(np.argmin(np.abs((directions - null_angle) % (2 * np.pi))))
    return float((r[idx_pref] - r[idx_null]) / (r[idx_pref] + r[idx_null] + 1e-9))


def size_tuning_index(responses: Sequence[float]) -> float:
    """Suppression index for centre-surround size tuning.

    SI = 1 - R_large / R_peak  (Sceniak 1999)
    """
    responses = _to_numpy(responses).ravel()
    return float(1.0 - responses[-1] / (responses.max() + 1e-9))


def length_selectivity_index(responses: Sequence[float]) -> float:
    """LSI = (R_pref - R_long) / (R_pref + R_long)."""
    responses = _to_numpy(responses).ravel()
    return float(
        (responses.max() - responses[-1]) / (responses.max() + responses[-1] + 1e-9)
    )
