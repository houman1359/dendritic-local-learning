"""Deterministic inferential summaries for replicated model experiments.

The experimental unit in most model-comparison studies is a trained-model
seed, not an input example, branch, synapse, or perturbation draw.  This module
provides the small, explicit primitives used by publication analyses after
those lower-level observations have been reduced within seed.

The functions intentionally do not guess a grouping or pairing scheme.  A
caller must first construct the exact per-seed estimand and then pass the
resulting one-dimensional values here.  This keeps paired and unpaired claims
scientifically distinguishable while sharing one deterministic bootstrap
implementation across analyzers and sweep reports.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class BootstrapMeanInterval:
    """A percentile-bootstrap interval over independent experimental units."""

    mean: float
    lower: float
    upper: float
    standard_deviation: float
    n_units: int
    draws: int
    seed: int
    confidence: float
    method: str = "experimental_unit_percentile"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe record."""

        return asdict(self)


def stable_seed(
    base_seed: int,
    *identity: Any,
    combine: Literal["xor", "add"] = "xor",
    representation: Literal["repr", "pipe"] = "repr",
    digest_bytes: int = 8,
) -> int:
    """Derive a reproducible RNG seed from an explicit analysis identity.

    ``repr`` preserves tuple/type boundaries and is suitable for grouped
    pandas keys. ``pipe`` matches older sweep analyses that joined simple
    scalar identities with ``|``.  The combination rule is explicit so a
    migrated analysis can retain its historical random stream exactly.
    """

    if not 1 <= int(digest_bytes) <= 8:
        raise ValueError("digest_bytes must be between 1 and 8")
    if representation == "repr":
        payload = repr(identity[0] if len(identity) == 1 else identity).encode()
    elif representation == "pipe":
        payload = "|".join(map(str, identity)).encode()
    else:  # pragma: no cover - guarded by the Literal type for typed callers
        raise ValueError(f"Unsupported seed representation: {representation}")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:digest_bytes], "little")
    if combine == "xor":
        return int(base_seed) ^ offset
    if combine == "add":
        return int(base_seed) + offset
    raise ValueError(f"Unsupported seed combination: {combine}")


def bootstrap_mean_interval(
    values: Iterable[float] | np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
    require_n: int | None = None,
    finite_policy: Literal["raise", "drop"] = "raise",
) -> BootstrapMeanInterval:
    """Summarize a mean by resampling independent units with replacement.

    Parameters
    ----------
    values:
        One scalar per independent experimental unit, normally one trained
        model seed or one already-paired seed difference.
    draws:
        Number of bootstrap resamples.  A non-positive value returns a
        degenerate interval at the observed mean for diagnostic use.
    seed:
        Concrete RNG seed.  Use :func:`stable_seed` when each group needs an
        independent deterministic stream.
    require_n:
        Optional fail-closed expected unit count.
    finite_policy:
        Publication analyses should use ``"raise"``. ``"drop"`` is retained
        for exploratory sweep summaries and never changes ``require_n`` after
        filtering.
    """

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    array = array.astype(np.float64, copy=False).reshape(-1)
    original_n = int(array.size)
    if require_n is not None and original_n != int(require_n):
        raise ValueError(
            f"Expected {int(require_n)} experimental units, observed {original_n}"
        )
    finite = np.isfinite(array)
    if finite_policy == "raise" and not finite.all():
        raise ValueError("Bootstrap values must all be finite")
    if finite_policy == "drop":
        array = array[finite]
    elif finite_policy != "raise":
        raise ValueError(f"Unsupported finite policy: {finite_policy}")
    if array.size == 0:
        raise ValueError("Bootstrap values cannot be empty")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size == 1 or int(draws) <= 0:
        lower = upper = mean
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(
            0,
            array.size,
            size=(int(draws), int(array.size)),
        )
        bootstrap_means = array[indices].mean(axis=1)
        tail = (1.0 - float(confidence)) / 2.0
        lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
        lower = float(lower)
        upper = float(upper)

    if not all(math.isfinite(value) for value in (mean, lower, upper)):
        raise FloatingPointError("Bootstrap summary produced a non-finite value")
    return BootstrapMeanInterval(
        mean=mean,
        lower=lower,
        upper=upper,
        standard_deviation=standard_deviation,
        n_units=int(array.size),
        draws=max(0, int(draws)),
        seed=int(seed),
        confidence=float(confidence),
    )


__all__ = [
    "BootstrapMeanInterval",
    "bootstrap_mean_interval",
    "stable_seed",
]
