"""Utility functions for Partial Information Decomposition (PID)."""

from __future__ import annotations

import logging

import numpy as np

try:
    from external import PID
except Exception as e:
    PID = None
    pid_error: Exception = e
else:
    pid_error = None


def check_pid_available() -> None:
    """Raise an ImportError if the PID package is not available."""
    if PID is None:
        raise ImportError("PID package not found.") from pid_error


def pid_synergy_redundancy(
    sources: list[np.ndarray], target: np.ndarray
) -> tuple[float, float]:
    """Compute synergy and redundancy using the external PID library.

    Parameters
    ----------
    sources : list of ndarrays
        List of source variables with shape ``(n_samples, n_features)``.
    target : ndarray
        Target variable with shape ``(n_samples, n_features)``.

    Returns
    -------
    tuple of floats
        ``(synergy, redundancy)`` from the chosen PID method.
    """

    check_pid_available()

    # The exact API will depend on the external library.  Here we
    # demonstrate a generic usage pattern expected from typical PID
    # implementations.
    data = np.hstack([*sources, target])
    pid_obj = PID(data)

    try:
        synergy = pid_obj.synergy()
        redundancy = pid_obj.redundancy()
    except AttributeError as exc:  # pragma: no cover - depends on PID API
        logging.error("Unexpected PID API: %s", exc)
        raise

    return float(synergy), float(redundancy)
