"""Exact linear response of serial and parallel leaky state banks.

The helpers in this module describe the small-signal temporal kernel implied by
the leaky trace update used by recurrent dendritic populations.  For a unit
input ``u[t]`` one first-order stage evolves as

``x[t] = rho * x[t - 1] + (1 - rho) * u[t]``,

where ``rho = exp(-dt / tau)``.  A serial dendritic path composes these stages;
a parallel multiscale bank forms a non-negative weighted sum of them.  The two
constructions have different temporal geometry even when they use the same
time constants: an equal-tau serial path has a negative-binomial impulse
kernel with a depth-dependent delayed peak, whereas a positive parallel bank
is a monotone sum of decays.

These functions are architecture-independent and intentionally do not inspect
or mutate a trained model.  They provide theory-side predictions that can be
tested against full nonlinear recurrent simulations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class KernelSummary:
    """Normalized temporal moments and peak location of a non-negative kernel."""

    mass: float
    peak_index: int
    peak_value: float
    mean_delay: float
    standard_deviation: float
    coefficient_of_variation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CascadeStateSpace:
    """State-space realization of a same-step serial leaky cascade.

    With previous state ``x_prev`` and scalar input ``u``, the current state is
    ``x = transition @ x_prev + input_vector * u``. The scalar output is
    ``readout_vector @ x``. This convention matches
    :func:`serial_cascade_impulse`.
    """

    transition: torch.Tensor
    input_vector: torch.Tensor
    readout_vector: torch.Tensor


@dataclass(frozen=True)
class AnalyticMomentSummary:
    """Exact normalized moments of a heterogeneous serial delay kernel."""

    mean_delay: float
    variance: float
    standard_deviation: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a real floating-point torch.dtype")


def decay_from_tau(tau: float, *, dt: float = 1.0) -> float:
    """Return ``exp(-dt / tau)`` after validating the time discretization."""

    tau = float(tau)
    dt = float(dt)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be finite and positive, got {tau!r}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    return math.exp(-dt / tau)


def first_order_impulse(
    tau: float,
    *,
    n_steps: int,
    dt: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the unit-mass impulse response of one discrete leaky stage."""

    _validate_floating_dtype(dtype)
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    rho = decay_from_tau(tau, dt=dt)
    time = torch.arange(int(n_steps), dtype=dtype)
    return (1.0 - rho) * torch.pow(torch.as_tensor(rho, dtype=dtype), time)


def serial_cascade_impulse(
    taus: Sequence[float],
    *,
    n_steps: int,
    dt: float = 1.0,
    gains: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the impulse response of same-step serially composed leaky stages.

    The implementation directly follows the recurrent update and is stable for
    heterogeneous time constants.  ``gains`` optionally multiplies the output
    of each stage before it drives the next stage.
    """

    _validate_floating_dtype(dtype)
    taus = tuple(float(tau) for tau in taus)
    if not taus:
        raise ValueError("taus must contain at least one time constant")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if gains is None:
        gains = (1.0,) * len(taus)
    else:
        gains = tuple(float(gain) for gain in gains)
        if len(gains) != len(taus):
            raise ValueError("gains must have the same length as taus")
        if any(not math.isfinite(gain) for gain in gains):
            raise ValueError("gains must be finite")

    decays = [decay_from_tau(tau, dt=dt) for tau in taus]
    states = torch.zeros(len(taus), dtype=dtype)
    response = torch.zeros(int(n_steps), dtype=dtype)
    for timestep in range(int(n_steps)):
        drive = torch.as_tensor(1.0 if timestep == 0 else 0.0, dtype=dtype)
        for level, (rho, gain) in enumerate(zip(decays, gains, strict=True)):
            states[level] = rho * states[level] + (1.0 - rho) * drive
            drive = float(gain) * states[level]
        response[timestep] = drive
    return response


def serial_cascade_moments(
    taus: Sequence[float],
    *,
    dt: float = 1.0,
) -> AnalyticMomentSummary:
    """Return exact moments of a normalized heterogeneous serial cascade.

    Each stage contributes an independent geometric waiting time with support
    on non-negative integer timesteps.  The cascade delay is their sum, so its
    mean and variance are the sums of the stage moments.  The result is
    invariant to permutations of ``taus`` even though intermediate-state
    responses and local interventions generally are not.
    """

    resolved_taus = tuple(float(tau) for tau in taus)
    if not resolved_taus:
        raise ValueError("taus must contain at least one time constant")
    decays = tuple(decay_from_tau(tau, dt=dt) for tau in resolved_taus)
    mean = sum(rho / (1.0 - rho) for rho in decays)
    variance = sum(rho / (1.0 - rho) ** 2 for rho in decays)
    return AnalyticMomentSummary(
        mean_delay=mean,
        variance=variance,
        standard_deviation=math.sqrt(variance),
    )


def positive_tree_path_mixture_impulse(
    paths: Sequence[Sequence[float]],
    *,
    n_steps: int,
    dt: float = 1.0,
    path_weights: Sequence[float] | None = None,
    normalize_weights: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the positive mixture of leaf-to-root serial path kernels.

    In a linear rooted tree, superposition writes the root response to inputs
    on multiple leaves as a sum over their unique leaf-to-root paths.  Each
    path is a serial cascade; ``path_weights`` collects its input amplitude,
    child-to-parent couplings, and readout gain.  Repeated shared suffixes are
    valid because this function describes the transfer from the declared leaf
    inputs to the root, not an allocation of independent state per path.

    Non-negative weights make the result a positive temporal-kernel basis.
    Set ``normalize_weights=False`` to retain the physical summed DC gain.
    """

    _validate_floating_dtype(dtype)
    resolved_paths = tuple(tuple(float(tau) for tau in path) for path in paths)
    if not resolved_paths or any(not path for path in resolved_paths):
        raise ValueError("paths must contain at least one non-empty tau path")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if path_weights is None:
        weights = (1.0,) * len(resolved_paths)
    else:
        weights = tuple(float(weight) for weight in path_weights)
    if len(weights) != len(resolved_paths):
        raise ValueError("path_weights must have the same length as paths")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("path weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("at least one path weight must be positive")
    if normalize_weights:
        weights = tuple(weight / total for weight in weights)

    response = torch.zeros(int(n_steps), dtype=dtype)
    for path, weight in zip(resolved_paths, weights, strict=True):
        response = response + weight * serial_cascade_impulse(
            path,
            n_steps=int(n_steps),
            dt=dt,
            dtype=dtype,
        )
    return response


def serial_cascade_state_space(
    taus: Sequence[float],
    *,
    dt: float = 1.0,
    gains: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float64,
) -> CascadeStateSpace:
    """Construct the exact finite-dimensional state realization of a cascade.

    Same-step child-to-parent propagation makes the transition matrix
    triangular. Its eigenvalues are the individual decay factors, while its
    off-diagonal entries make it non-normal whenever at least two coupled
    stages are present.
    """

    _validate_floating_dtype(dtype)
    taus = tuple(float(tau) for tau in taus)
    if not taus:
        raise ValueError("taus must contain at least one time constant")
    if gains is None:
        resolved_gains = (1.0,) * len(taus)
    else:
        resolved_gains = tuple(float(gain) for gain in gains)
        if len(resolved_gains) != len(taus):
            raise ValueError("gains must have the same length as taus")
        if any(not math.isfinite(gain) for gain in resolved_gains):
            raise ValueError("gains must be finite")

    decays = tuple(decay_from_tau(tau, dt=dt) for tau in taus)
    depth = len(taus)

    def advance(state: torch.Tensor, input_value: float) -> torch.Tensor:
        drive = torch.as_tensor(input_value, dtype=dtype)
        current: list[torch.Tensor] = []
        for level, (rho, gain) in enumerate(zip(decays, resolved_gains, strict=True)):
            value = rho * state[level] + (1.0 - rho) * drive
            current.append(value)
            drive = gain * value
        return torch.stack(current)

    zero = torch.zeros(depth, dtype=dtype)
    columns = []
    for state_index in range(depth):
        basis = zero.clone()
        basis[state_index] = 1.0
        columns.append(advance(basis, 0.0))
    transition = torch.stack(columns, dim=1)
    input_vector = advance(zero, 1.0)
    readout_vector = torch.zeros(depth, dtype=dtype)
    readout_vector[-1] = resolved_gains[-1]
    return CascadeStateSpace(
        transition=transition,
        input_vector=input_vector,
        readout_vector=readout_vector,
    )


def similarity_transform_transition_intervention(
    intervention: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """Map a transition intervention under ``x_target = T x_source``.

    If nominal realizations satisfy ``A_target = T A_source T^{-1}``, then a
    source-coordinate change ``intervention`` is reproduced exactly by
    ``T intervention T^{-1}`` in the target coordinates.  A one-coordinate
    source intervention can therefore become dense in a generic point-state
    basis while retaining the same rank.
    """

    if not isinstance(intervention, torch.Tensor) or not isinstance(
        transform, torch.Tensor
    ):
        raise TypeError("intervention and transform must be torch.Tensor objects")
    if intervention.ndim != 2 or transform.ndim != 2:
        raise ValueError("intervention and transform must be two-dimensional")
    if (
        intervention.shape[0] != intervention.shape[1]
        or transform.shape[0] != transform.shape[1]
        or intervention.shape != transform.shape
        or intervention.shape[0] == 0
    ):
        raise ValueError("intervention and transform must be non-empty matched squares")
    if not intervention.is_floating_point() or not transform.is_floating_point():
        raise TypeError("intervention and transform must use floating-point dtypes")
    if intervention.device != transform.device:
        raise ValueError("intervention and transform must be on the same device")
    if intervention.dtype != transform.dtype:
        raise ValueError("intervention and transform must have the same dtype")
    if not bool(torch.isfinite(intervention).all()) or not bool(
        torch.isfinite(transform).all()
    ):
        raise ValueError("intervention and transform must be finite")
    if int(torch.linalg.matrix_rank(transform).item()) != transform.shape[0]:
        raise ValueError("transform must be invertible")
    left_product = transform @ intervention
    return torch.linalg.solve(transform.T, left_product.T).T


def normalized_nonnormality(matrix: torch.Tensor) -> float:
    """Return a dimensionless Frobenius norm of the normality commutator.

    The value is zero exactly for a normal matrix up to numerical precision.
    It is a structural diagnostic, not a finite-time amplification or Lyapunov
    exponent.
    """

    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("matrix must be a two-dimensional torch.Tensor")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix must be non-empty and square")
    if not matrix.is_floating_point() and not matrix.is_complex():
        raise TypeError("matrix must have a floating-point or complex dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    adjoint = matrix.mH
    commutator = adjoint @ matrix - matrix @ adjoint
    scale = torch.linalg.matrix_norm(matrix, ord="fro").square()
    if float(scale.item()) == 0.0:
        return 0.0
    return float((torch.linalg.matrix_norm(commutator, ord="fro") / scale).item())


def equal_tau_serial_closed_form(
    *,
    tau: float,
    depth: int,
    n_steps: int,
    dt: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the negative-binomial impulse kernel for equal-tau serial stages."""

    _validate_floating_dtype(dtype)
    if isinstance(depth, bool) or int(depth) != depth or depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    rho = decay_from_tau(tau, dt=dt)
    values = [
        math.comb(timestep + int(depth) - 1, int(depth) - 1)
        * (1.0 - rho) ** int(depth)
        * rho**timestep
        for timestep in range(int(n_steps))
    ]
    return torch.tensor(values, dtype=dtype)


def parallel_bank_impulse(
    taus: Sequence[float],
    *,
    n_steps: int,
    dt: float = 1.0,
    weights: Sequence[float] | None = None,
    normalize_weights: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a non-negative weighted sum of first-order impulse responses."""

    _validate_floating_dtype(dtype)
    taus = tuple(float(tau) for tau in taus)
    if not taus:
        raise ValueError("taus must contain at least one time constant")
    if weights is None:
        weights = (1.0,) * len(taus)
    else:
        weights = tuple(float(weight) for weight in weights)
    if len(weights) != len(taus):
        raise ValueError("weights must have the same length as taus")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("parallel weights must be finite and non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one parallel weight must be positive")
    if normalize_weights:
        weights = tuple(weight / weight_sum for weight in weights)

    response = torch.zeros(int(n_steps), dtype=dtype)
    for tau, weight in zip(taus, weights, strict=True):
        response = response + float(weight) * first_order_impulse(
            tau,
            n_steps=n_steps,
            dt=dt,
            dtype=dtype,
        )
    return response


def frequency_response(
    taus: Sequence[float],
    angular_frequencies: torch.Tensor,
    *,
    topology: str,
    dt: float = 1.0,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Evaluate the discrete transfer function on angular frequencies.

    ``angular_frequencies`` uses radians per timestep.  ``topology`` is either
    ``"serial"`` (product of filters) or ``"parallel"`` (weighted sum).
    """

    taus = tuple(float(tau) for tau in taus)
    if not taus:
        raise ValueError("taus must contain at least one time constant")
    if not isinstance(angular_frequencies, torch.Tensor):
        raise TypeError("angular_frequencies must be a torch.Tensor")
    if not angular_frequencies.is_floating_point():
        raise TypeError("angular_frequencies must use a real floating-point dtype")
    if not bool(torch.isfinite(angular_frequencies).all()):
        raise ValueError("angular_frequencies must be finite")
    complex_dtype = (
        torch.complex128
        if angular_frequencies.dtype == torch.float64
        else torch.complex64
    )
    z_inverse = torch.exp(-1j * angular_frequencies.to(dtype=complex_dtype))
    stages = []
    for tau in taus:
        rho = decay_from_tau(tau, dt=dt)
        stages.append((1.0 - rho) / (1.0 - rho * z_inverse))

    topology = str(topology).lower()
    if topology == "serial":
        result = torch.ones_like(stages[0])
        for stage in stages:
            result = result * stage
        return result
    if topology != "parallel":
        raise ValueError("topology must be 'serial' or 'parallel'")

    if weights is None:
        normalized = (1.0 / len(stages),) * len(stages)
    else:
        normalized = tuple(float(weight) for weight in weights)
        if len(normalized) != len(stages):
            raise ValueError("weights must have the same length as taus")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in normalized):
            raise ValueError("parallel weights must be finite and non-negative")
        total = sum(normalized)
        if total <= 0.0:
            raise ValueError("at least one parallel weight must be positive")
        normalized = tuple(weight / total for weight in normalized)
    result = torch.zeros_like(stages[0])
    for weight, stage in zip(normalized, stages, strict=True):
        result = result + weight * stage
    return result


def summarize_kernel(kernel: torch.Tensor) -> KernelSummary:
    """Summarize a finite non-negative kernel after normalizing its mass."""

    if not isinstance(kernel, torch.Tensor) or kernel.ndim != 1 or not kernel.numel():
        raise ValueError("kernel must be a non-empty one-dimensional tensor")
    if not torch.isfinite(kernel).all() or bool((kernel < 0).any()):
        raise ValueError("kernel must be finite and non-negative")
    mass_tensor = kernel.sum()
    mass = float(mass_tensor.item())
    if mass <= 0.0:
        raise ValueError("kernel must have positive mass")
    probability = kernel / mass_tensor
    time = torch.arange(kernel.numel(), device=kernel.device, dtype=kernel.dtype)
    mean = (probability * time).sum()
    variance = (probability * (time - mean).square()).sum()
    standard_deviation = variance.clamp_min(0.0).sqrt()
    mean_float = float(mean.item())
    sd_float = float(standard_deviation.item())
    return KernelSummary(
        mass=mass,
        peak_index=int(torch.argmax(kernel).item()),
        peak_value=float(kernel.max().item()),
        mean_delay=mean_float,
        standard_deviation=sd_float,
        coefficient_of_variation=(
            sd_float / mean_float if mean_float > 0.0 else math.inf
        ),
    )


__all__ = [
    "AnalyticMomentSummary",
    "CascadeStateSpace",
    "KernelSummary",
    "decay_from_tau",
    "equal_tau_serial_closed_form",
    "first_order_impulse",
    "frequency_response",
    "normalized_nonnormality",
    "parallel_bank_impulse",
    "positive_tree_path_mixture_impulse",
    "serial_cascade_impulse",
    "serial_cascade_moments",
    "serial_cascade_state_space",
    "similarity_transform_transition_intervention",
    "summarize_kernel",
]
