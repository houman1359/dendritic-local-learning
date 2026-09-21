"""Held-out linear memory curves from recorded state trajectories.

This module measures how well a single linear readout of the current state can
recover past values of an exogenous drive.  It deliberately accepts separate
training and test trajectories: the utility does not split one trajectory or
select hyperparameters from the test data.

All requested lags are fit as one multi-target ridge regression.  Every lag
uses the same state rows, starting at the maximum requested lag, so differences
between lag scores cannot be caused by different sample counts.  State and
target centering is estimated on training rows only.  The primary integrated
capacity is the sum over lags of held-out coefficient-of-determination scores,
clipped componentwise to ``[0, 1]`` and averaged uniformly across drive
coordinates.  This out-of-sample, calibration-sensitive quantity is distinct
from the classic in-sample sum of squared correlations; held-out correlations
are returned separately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LinearMemoryCurveMetadata:
    """Explicit sampling, fitting, and metric contract for one estimate."""

    lags: tuple[int, ...]
    ridge_alpha: float
    ridge_penalty_scale: float
    horizon_threshold: float
    train_trajectories: int
    test_trajectories: int
    train_steps_per_trajectory: int
    test_steps_per_trajectory: int
    train_rows: int
    test_rows: int
    state_dimension: int
    drive_dimension: int
    internal_dtype: str = "float64"
    estimator: str = "single_solve_multi_target_ridge_with_unpenalized_intercept"
    ridge_objective: str = (
        "sum_squared_error_plus_alpha_times_mean_centered_state_gram_diagonal_"
        "times_weight_l2_squared"
    )
    ridge_invariance: str = (
        "invariant_to_one_uniform_global_rescaling_of_all_state_coordinates;_"
        "not_invariant_to_coordinatewise_or_general_linear_reparameterization"
    )
    centering: str = "state_and_lagged_targets_centered_with_training_means_only"
    row_window: str = "t_from_maximum_lag_through_final_step_for_every_lag"
    target_layout: str = "lag_major_then_drive_coordinate"
    r2_definition: str = "one_minus_test_sse_over_test_mean_baseline_sse"
    correlation_definition: str = "test_sample_pearson; zero_for_constant_prediction"
    lag_aggregation: str = "uniform_arithmetic_mean_over_drive_coordinates"
    capacity_definition: str = (
        "sum_over_lags_of_mean_over_drive_coordinates_of_clip(test_r2,0,1)"
    )
    horizon_definition: str = (
        "largest_evaluated_lag_with_clipped_mean_test_r2_at_least_threshold"
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe metadata record."""

        return asdict(self)


@dataclass(frozen=True)
class LinearMemoryCurveResult:
    """Fit parameters and held-out metrics for a linear memory curve.

    Arrays with suffix ``_by_lag_and_drive`` have shape
    ``(n_lags, drive_dimension)``.  ``coefficients`` has shape
    ``(state_dimension, n_lags, drive_dimension)`` and ``intercept`` has shape
    ``(n_lags, drive_dimension)``.  All returned arrays are read-only copies.
    """

    metadata: LinearMemoryCurveMetadata
    r2_by_lag: FloatArray
    correlation_by_lag: FloatArray
    r2_by_lag_and_drive: FloatArray
    correlation_by_lag_and_drive: FloatArray
    clipped_capacity_by_lag: FloatArray
    capacity_by_drive: FloatArray
    integrated_memory_capacity: float
    threshold_horizon: int | None
    coefficients: FloatArray
    intercept: FloatArray
    state_train_mean: FloatArray
    target_train_mean: FloatArray

    def to_dict(self, *, include_fit: bool = False) -> dict[str, object]:
        """Return a JSON-safe summary, optionally including fitted readouts."""

        payload: dict[str, object] = {
            "metadata": self.metadata.to_dict(),
            "r2_by_lag": self.r2_by_lag.tolist(),
            "correlation_by_lag": self.correlation_by_lag.tolist(),
            "r2_by_lag_and_drive": self.r2_by_lag_and_drive.tolist(),
            "correlation_by_lag_and_drive": (
                self.correlation_by_lag_and_drive.tolist()
            ),
            "clipped_capacity_by_lag": self.clipped_capacity_by_lag.tolist(),
            "capacity_by_drive": self.capacity_by_drive.tolist(),
            "integrated_memory_capacity": self.integrated_memory_capacity,
            "threshold_horizon": self.threshold_horizon,
        }
        if include_fit:
            payload.update(
                {
                    "coefficients": self.coefficients.tolist(),
                    "intercept": self.intercept.tolist(),
                    "state_train_mean": self.state_train_mean.tolist(),
                    "target_train_mean": self.target_train_mean.tolist(),
                }
            )
        return payload


def _as_finite_numeric_array(value: np.ndarray, *, name: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if np.issubdtype(value.dtype, np.bool_) or not np.issubdtype(
        value.dtype, np.number
    ):
        raise TypeError(f"{name} must have a real numeric dtype")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a real numeric dtype")
    array = value.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalize_states(
    value: np.ndarray,
    *,
    name: str,
) -> tuple[FloatArray, bool]:
    array = _as_finite_numeric_array(value, name=name)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
        originally_batched = False
    elif array.ndim == 3:
        originally_batched = True
    else:
        raise ValueError(
            f"{name} must have shape (time, state) or (trajectory, time, state)"
        )
    if any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} dimensions must all be non-empty")
    return array, originally_batched


def _normalize_drive(
    value: np.ndarray,
    *,
    name: str,
    trajectories: int,
    steps: int,
    states_were_batched: bool,
) -> FloatArray:
    array = _as_finite_numeric_array(value, name=name)
    if states_were_batched:
        if array.ndim == 2:
            array = array[..., np.newaxis]
        elif array.ndim != 3:
            raise ValueError(
                f"{name} must have shape (trajectory, time) or "
                "(trajectory, time, drive) for batched states"
            )
    else:
        if array.ndim == 1:
            array = array[np.newaxis, :, np.newaxis]
        elif array.ndim == 2:
            array = array[np.newaxis, ...]
        else:
            raise ValueError(
                f"{name} must have shape (time,) or (time, drive) "
                "for one state trajectory"
            )
    if array.shape[:2] != (trajectories, steps):
        raise ValueError(
            f"{name} trajectory/time shape {array.shape[:2]} does not match "
            f"states shape {(trajectories, steps)}"
        )
    if array.shape[2] <= 0:
        raise ValueError(f"{name} drive dimension must be non-empty")
    return array


def _validated_lags(lags: Sequence[int]) -> tuple[int, ...]:
    if isinstance(lags, (str, bytes)):
        raise TypeError("lags must be a sequence of non-negative integers")
    values = tuple(lags)
    if not values:
        raise ValueError("lags must contain at least one value")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("lags must contain only non-negative integers")
    resolved = tuple(int(value) for value in values)
    if any(value < 0 for value in resolved):
        raise ValueError("lags must contain only non-negative integers")
    if any(right <= left for left, right in pairwise(resolved)):
        raise ValueError("lags must be unique and strictly increasing")
    return resolved


def _validated_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return resolved


def _lagged_rows(
    states: FloatArray,
    drive: FloatArray,
    lags: tuple[int, ...],
) -> tuple[FloatArray, FloatArray]:
    maximum_lag = lags[-1]
    steps = states.shape[1]
    usable_steps = steps - maximum_lag
    state_rows = states[:, maximum_lag:, :].reshape(
        states.shape[0] * usable_steps,
        states.shape[2],
    )
    targets = np.stack(
        [
            drive[
                :,
                maximum_lag - lag : (steps - lag if lag > 0 else steps),
                :,
            ]
            for lag in lags
        ],
        axis=2,
    )
    target_rows = targets.reshape(
        states.shape[0] * usable_steps,
        len(lags) * drive.shape[2],
    )
    return state_rows, target_rows


def _centered_sum_squares(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    centered = values - values.mean(axis=0, keepdims=True)
    return centered, np.sum(centered * centered, axis=0)


def _variance_tolerance(values: FloatArray) -> FloatArray:
    maximum = np.max(np.abs(values), axis=0)
    return np.finfo(np.float64).eps * values.shape[0] * maximum * maximum


def _require_variable_targets(values: FloatArray, *, split: str) -> None:
    _, sum_squares = _centered_sum_squares(values)
    invalid = sum_squares <= _variance_tolerance(values)
    if invalid.any():
        indices = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"{split} lagged targets must vary in every lag/drive column; "
            f"constant or numerically constant columns: {indices}"
        )


def _readonly(value: np.ndarray) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def estimate_linear_memory_curve(
    *,
    train_states: np.ndarray,
    train_drive: np.ndarray,
    test_states: np.ndarray,
    test_drive: np.ndarray,
    lags: Sequence[int],
    ridge_alpha: float = 1e-6,
    horizon_threshold: float = 0.1,
) -> LinearMemoryCurveResult:
    """Fit a joint ridge readout and evaluate a held-out memory curve.

    Args:
        train_states: ``(time, state)`` or ``(trajectory, time, state)`` array.
        train_drive: Scalar or vector drive aligned with ``train_states``.
        test_states: Independent state trajectories with the same state width.
        test_drive: Drive aligned with ``test_states`` and the same drive width.
        lags: Unique, strictly increasing non-negative integer lags.  Target
            column ``lag`` is the drive at ``t - lag`` and the state is at ``t``.
        ridge_alpha: Positive dimensionless multiplier on the summed squared
            weight norm. The executable penalty coefficient is this value
            times the mean diagonal of the centered training-state Gram matrix,
            making predictions invariant to uniform rescaling of state units.
        horizon_threshold: Prespecified clipped mean test-R2 threshold in
            ``(0, 1]``.  The horizon is the largest evaluated lag meeting it.

    Returns:
        An immutable-contract result with fitted parameters and held-out scores.

    Notes:
        Separate arguments and a no-shared-memory check prevent accidental
        train-on-test reuse.  Statistical independence of copied trajectories
        remains a property the caller must establish through data provenance.
    """

    raw_train = (train_states, train_drive)
    raw_test = (test_states, test_drive)
    if not all(isinstance(value, np.ndarray) for value in (*raw_train, *raw_test)):
        raise TypeError("train/test states and drives must be separate numpy arrays")
    for train_value in raw_train:
        for test_value in raw_test:
            if np.shares_memory(train_value, test_value):
                raise ValueError(
                    "train and test states/drives must be separate, non-overlapping "
                    "arrays"
                )

    resolved_lags = _validated_lags(lags)
    alpha = _validated_positive_float(ridge_alpha, name="ridge_alpha")
    threshold = _validated_positive_float(horizon_threshold, name="horizon_threshold")
    if threshold > 1.0:
        raise ValueError("horizon_threshold must be at most 1")

    train_state_array, train_batched = _normalize_states(
        train_states, name="train_states"
    )
    test_state_array, test_batched = _normalize_states(test_states, name="test_states")
    if train_state_array.shape[2] != test_state_array.shape[2]:
        raise ValueError("train and test state dimensions must match")

    train_drive_array = _normalize_drive(
        train_drive,
        name="train_drive",
        trajectories=train_state_array.shape[0],
        steps=train_state_array.shape[1],
        states_were_batched=train_batched,
    )
    test_drive_array = _normalize_drive(
        test_drive,
        name="test_drive",
        trajectories=test_state_array.shape[0],
        steps=test_state_array.shape[1],
        states_were_batched=test_batched,
    )
    if train_drive_array.shape[2] != test_drive_array.shape[2]:
        raise ValueError("train and test drive dimensions must match")

    maximum_lag = resolved_lags[-1]
    for name, array in (
        ("train_states", train_state_array),
        ("test_states", test_state_array),
    ):
        if array.shape[1] <= maximum_lag:
            raise ValueError(
                f"{name} needs more than maximum lag {maximum_lag} timesteps"
            )
        row_count = array.shape[0] * (array.shape[1] - maximum_lag)
        if row_count < 2:
            raise ValueError(f"{name} must provide at least two usable rows")

    train_x, train_y = _lagged_rows(train_state_array, train_drive_array, resolved_lags)
    test_x, test_y = _lagged_rows(test_state_array, test_drive_array, resolved_lags)
    _require_variable_targets(train_y, split="training")
    _require_variable_targets(test_y, split="test")

    state_mean = train_x.mean(axis=0)
    target_mean = train_y.mean(axis=0)
    centered_x = train_x - state_mean
    centered_y = train_y - target_mean

    gram = centered_x.T @ centered_x
    right_hand_side = centered_x.T @ centered_y
    ridge_scale = float(np.trace(gram) / gram.shape[0])
    if not math.isfinite(ridge_scale) or ridge_scale <= 0.0:
        raise ValueError("training states must have nonzero centered variation")
    effective_alpha = alpha * ridge_scale
    gram.flat[:: gram.shape[0] + 1] += effective_alpha
    if not np.isfinite(gram).all() or not np.isfinite(right_hand_side).all():
        raise FloatingPointError("ridge normal equations overflowed")
    try:
        flat_coefficients = np.linalg.solve(gram, right_hand_side)
    except np.linalg.LinAlgError as error:  # pragma: no cover - alpha makes this rare
        raise FloatingPointError("ridge solve failed") from error
    if not np.isfinite(flat_coefficients).all():
        raise FloatingPointError("ridge solve produced non-finite coefficients")

    flat_intercept = target_mean - state_mean @ flat_coefficients
    prediction = test_x @ flat_coefficients + flat_intercept
    if not np.isfinite(prediction).all():
        raise FloatingPointError("ridge prediction produced non-finite values")

    target_centered, target_sum_squares = _centered_sum_squares(test_y)
    prediction_centered, prediction_sum_squares = _centered_sum_squares(prediction)
    residual_sum_squares = np.sum((test_y - prediction) ** 2, axis=0)
    r2 = 1.0 - residual_sum_squares / target_sum_squares

    covariance = np.sum(target_centered * prediction_centered, axis=0)
    prediction_variable = prediction_sum_squares > _variance_tolerance(prediction)
    correlation = np.zeros_like(covariance)
    correlation[prediction_variable] = covariance[prediction_variable] / np.sqrt(
        target_sum_squares[prediction_variable]
        * prediction_sum_squares[prediction_variable]
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    if not np.isfinite(r2).all() or not np.isfinite(correlation).all():
        raise FloatingPointError("memory-curve metrics produced non-finite values")

    n_lags = len(resolved_lags)
    drive_dimension = train_drive_array.shape[2]
    component_r2 = r2.reshape(n_lags, drive_dimension)
    component_correlation = correlation.reshape(n_lags, drive_dimension)
    clipped_components = np.clip(component_r2, 0.0, 1.0)
    r2_by_lag = component_r2.mean(axis=1)
    correlation_by_lag = component_correlation.mean(axis=1)
    capacity_by_lag = clipped_components.mean(axis=1)
    capacity_by_drive = clipped_components.sum(axis=0)
    integrated_capacity = float(capacity_by_lag.sum())

    qualifying = np.flatnonzero(capacity_by_lag >= threshold)
    threshold_horizon = (
        int(resolved_lags[int(qualifying[-1])]) if qualifying.size else None
    )
    coefficients = flat_coefficients.reshape(
        train_state_array.shape[2], n_lags, drive_dimension
    )
    intercept = flat_intercept.reshape(n_lags, drive_dimension)
    target_train_mean = target_mean.reshape(n_lags, drive_dimension)

    metadata = LinearMemoryCurveMetadata(
        lags=resolved_lags,
        ridge_alpha=alpha,
        ridge_penalty_scale=ridge_scale,
        horizon_threshold=threshold,
        train_trajectories=int(train_state_array.shape[0]),
        test_trajectories=int(test_state_array.shape[0]),
        train_steps_per_trajectory=int(train_state_array.shape[1]),
        test_steps_per_trajectory=int(test_state_array.shape[1]),
        train_rows=int(train_x.shape[0]),
        test_rows=int(test_x.shape[0]),
        state_dimension=int(train_state_array.shape[2]),
        drive_dimension=int(drive_dimension),
    )
    return LinearMemoryCurveResult(
        metadata=metadata,
        r2_by_lag=_readonly(r2_by_lag),
        correlation_by_lag=_readonly(correlation_by_lag),
        r2_by_lag_and_drive=_readonly(component_r2),
        correlation_by_lag_and_drive=_readonly(component_correlation),
        clipped_capacity_by_lag=_readonly(capacity_by_lag),
        capacity_by_drive=_readonly(capacity_by_drive),
        integrated_memory_capacity=integrated_capacity,
        threshold_horizon=threshold_horizon,
        coefficients=_readonly(coefficients),
        intercept=_readonly(intercept),
        state_train_mean=_readonly(state_mean),
        target_train_mean=_readonly(target_train_mean),
    )


__all__ = [
    "LinearMemoryCurveMetadata",
    "LinearMemoryCurveResult",
    "estimate_linear_memory_curve",
]
