"""Bounded ordinary polynomial controls for nonpolynomial development tasks.

The dictionary uses observed supplied blocks only, with every within-block
monomial through a declared total degree and one global intercept. It consumes
no teacher state, intrinsic coordinates or held-out TEST labels. TRAIN feature
centering/scaling is folded into the deployed coefficients and intercept.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq


class ReferenceBudgetError(ValueError):
    """A dictionary is rejected before combinatorial materialization or fitting."""


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        valid = int(value) == value and value >= minimum
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _payload_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def polynomial_inventory(degree, blocks, ambient_dim):
    degree, blocks, ambient_dim = (
        _integer(degree, "Degree"),
        _integer(blocks, "Blocks"),
        _integer(ambient_dim, "Ambient dimension"),
    )
    per_block = math.comb(ambient_dim + degree, degree) - 1
    features = blocks * per_block
    return {
        "degree": degree,
        "blocks": blocks,
        "ambient_dim": ambient_dim,
        "nonconstant_features_per_block": per_block,
        "nonconstant_features": features,
        "stored_parameters": features + 1,
        "trainable_coefficients": features + 1,
        "intercept_parameters": 1,
        "stored_float64_bytes": 8 * (features + 1),
        "fixed_dictionary_scope": "Complete monomials reconstructed from public degree/dimension/block recipe, with repeated-variable index tuples; no stored learned input projection.",
        "support_scope": "Supplied observed raw blocks only. All models receive the same block prior; scrambled true support is not restored.",
        "deployed_normalization_slots": 0,
    }


def _validate_xy(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or len(x) < 1 or min(x.shape[1:]) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty raw inputs [N,blocks,ambient_dim]")
    if y is None:
        return x
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Finite scalar labels must match the input rows")
    return x, y


def _check_budget(
    inventory, n, parameter_ceiling, max_features, max_workspace_bytes, *, fitting
):
    features = inventory["nonconstant_features"]
    if parameter_ceiling is not None and inventory["stored_parameters"] > _integer(
        parameter_ceiling, "Parameter ceiling"
    ):
        raise ReferenceBudgetError(
            f"Dictionary needs {inventory['stored_parameters']} coefficient slots, above ceiling {parameter_ceiling}"
        )
    if features > _integer(max_features, "Feature limit"):
        raise ReferenceBudgetError(
            f"Dictionary has {features} nonconstant features, above safety cap {max_features}"
        )
    # A conservative preflight for explicitly allocated design/augmented arrays.
    # LAPACK's implementation-specific temporary storage is not a guaranteed RSS bound.
    workspace = (
        8 * (4 * n * features + 3 * features * features + 4 * n + 10 * features)
        if fitting
        else 8 * n * features
    )
    if workspace > _integer(max_workspace_bytes, "Workspace byte limit"):
        raise ReferenceBudgetError(
            f"Estimated explicit workspace {workspace} bytes exceeds cap {max_workspace_bytes}"
        )
    return workspace


def _terms(ambient_dim, degree):
    return tuple(
        term
        for order in range(1, degree + 1)
        for term in itertools.combinations_with_replacement(range(ambient_dim), order)
    )


def polynomial_features(
    x,
    degree,
    *,
    parameter_ceiling=None,
    max_features=4096,
    max_workspace_bytes=512 * 1024**2,
):
    """Construct a complete block dictionary only after exact count/memory checks."""
    x = _validate_xy(x)
    inventory = polynomial_inventory(degree, x.shape[1], x.shape[2])
    _check_budget(
        inventory,
        len(x),
        parameter_ceiling,
        max_features,
        max_workspace_bytes,
        fitting=False,
    )
    terms = _terms(x.shape[2], inventory["degree"])
    design = np.empty((len(x), inventory["nonconstant_features"]), dtype=np.float64)
    column = 0
    with np.errstate(over="raise", invalid="raise"):
        for block in range(x.shape[1]):
            for term in terms:
                design[:, column] = np.prod(x[:, block, term], axis=1)
                column += 1
    if column != design.shape[1] or not np.isfinite(design).all():
        raise FloatingPointError(
            "Polynomial dictionary has nonfinite values or inconsistent count"
        )
    return design


@dataclass(frozen=True)
class PolynomialReference:
    degree: int
    blocks: int
    ambient_dim: int
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self):
        inventory = polynomial_inventory(self.degree, self.blocks, self.ambient_dim)
        for name in ("degree", "blocks", "ambient_dim"):
            object.__setattr__(self, name, inventory[name])
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        if (
            coefficients.shape != (inventory["nonconstant_features"],)
            or not np.isfinite(coefficients).all()
            or not math.isfinite(self.intercept)
        ):
            raise ValueError("Polynomial state has invalid coefficient shape or values")
        coefficients.flags.writeable = False
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))

    @property
    def parameter_count(self):
        return len(self.coefficients) + 1

    def predict(self, x, *, batch_size=512, max_workspace_bytes=512 * 1024**2):
        x = _validate_xy(x)
        if x.shape[1:] != (self.blocks, self.ambient_dim):
            raise ValueError("Polynomial state and observed input shape disagree")
        batch_size = _integer(batch_size, "Prediction batch size")
        output = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), batch_size):
            design = polynomial_features(
                x[start : start + batch_size],
                self.degree,
                max_features=len(self.coefficients),
                max_workspace_bytes=max_workspace_bytes,
            )
            output[start : start + batch_size] = (
                design @ self.coefficients + self.intercept
            )
        return output

    def specification(self):
        payload = {
            "schema": "nonpolynomial_sharing_polynomial_reference_v1",
            "degree": self.degree,
            "blocks": self.blocks,
            "ambient_dim": self.ambient_dim,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "parameter_count": self.parameter_count,
        }
        return {**payload, "payload_sha256": _payload_hash(payload)}

    @classmethod
    def from_specification(cls, specification):
        if (
            specification.get("schema")
            != "nonpolynomial_sharing_polynomial_reference_v1"
        ):
            raise ValueError("Unknown polynomial state schema")
        payload = {
            key: value
            for key, value in specification.items()
            if key != "payload_sha256"
        }
        if _payload_hash(payload) != specification.get("payload_sha256"):
            raise ValueError("Polynomial state hash mismatch")
        model = cls(
            specification["degree"],
            specification["blocks"],
            specification["ambient_dim"],
            specification["coefficients"],
            specification["intercept"],
        )
        if model.parameter_count != specification["parameter_count"]:
            raise ValueError("Polynomial state count mismatch")
        return model


def fit_polynomial_reference(
    x_train,
    y_train,
    *,
    degree,
    ridge,
    parameter_ceiling,
    max_features=4096,
    max_workspace_bytes=512 * 1024**2,
    rcond=1e-12,
):
    """TRAIN-only SVD solve, with positive ridge or declared zero-ridge OLS.

    Minimize TRAIN MSE + ridge*||beta||^2 in TRAIN-centered, RMS-standardized
    monomials, with an unpenalized intercept. All coefficients are counted,
    including columns numerically null on TRAIN. The SVD cutoff/rank are recorded.
    """
    x_train, y_train = _validate_xy(x_train, y_train)
    if len(x_train) < 2:
        raise ValueError("At least two TRAIN rows required")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge must be finite and nonnegative")
    if not math.isfinite(rcond) or rcond <= 0:
        raise ValueError("SVD rcond must be positive and finite")
    inventory = polynomial_inventory(degree, x_train.shape[1], x_train.shape[2])
    workspace = _check_budget(
        inventory,
        len(x_train),
        parameter_ceiling,
        max_features,
        max_workspace_bytes,
        fitting=True,
    )
    design = polynomial_features(
        x_train,
        degree,
        parameter_ceiling=parameter_ceiling,
        max_features=max_features,
        max_workspace_bytes=max_workspace_bytes,
    )
    means = design.mean(axis=0)
    centered = design - means
    scales = np.sqrt(np.mean(np.square(centered), axis=0))
    # A constant TRAIN column contributes zero centered feature and stays counted.
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = centered / scales
    y_mean = float(y_train.mean())
    centered_y = y_train - y_mean
    if ridge > 0:
        matrix = np.concatenate(
            (standardized, math.sqrt(len(x_train) * ridge) * np.eye(design.shape[1])),
            axis=0,
        )
        rhs = np.concatenate((centered_y, np.zeros(design.shape[1])))
    else:
        matrix, rhs = standardized, centered_y
    beta, _, rank, singular_values = lstsq(
        matrix, rhs, cond=rcond, lapack_driver="gelsd", check_finite=True
    )
    coefficients = beta / scales
    intercept = y_mean - float(means @ coefficients)
    model = PolynomialReference(
        degree, x_train.shape[1], x_train.shape[2], coefficients, intercept
    )
    deployed = design @ coefficients + intercept
    standardized_prediction = standardized @ beta + y_mean
    if not np.isfinite(deployed).all():
        raise FloatingPointError("Nonfinite deployed polynomial prediction")
    receipt = {
        "schema": "nonpolynomial_sharing_polynomial_fit_v1",
        "inventory": inventory,
        "ridge": float(ridge),
        "svd_rcond": float(rcond),
        "svd_rank": int(rank),
        "svd_matrix_shape": list(matrix.shape),
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "train_rows": len(x_train),
        "train_x_sha256": _array_hash(x_train),
        "train_y_sha256": _array_hash(y_train),
        "train_mse": float(np.mean(np.square(deployed - y_train))),
        "standardization_fold_max_error": float(
            np.max(np.abs(deployed - standardized_prediction))
        ),
        "zero_centered_train_columns": int(
            np.count_nonzero(np.all(centered == 0, axis=0))
        ),
        "estimated_explicit_workspace_bytes": workspace,
        "workspace_scope": "Conservative bound for explicit design/solve arrays; LAPACK internal workspaces and process RSS require measured profiling.",
        "deployed_model_sha256": model.specification()["payload_sha256"],
        "objective": "mean TRAIN squared error + ridge*sum(standardized feature coefficient squared); intercept unpenalized.",
        "scope": "Fit consumes TRAIN only. Feature centering/scaling is fitted on TRAIN and folded into existing deployed coefficient/intercept slots; no validation fitting or private subspace.",
    }
    return model, receipt


def select_polynomial_reference(
    x_train,
    y_train,
    x_validation,
    y_validation,
    *,
    parameter_ceiling,
    degrees=(1, 2, 3, 4),
    ridges=(1e-8, 1e-5),
    max_features=4096,
    max_workspace_bytes=512 * 1024**2,
    rcond=1e-12,
):
    """Choose a TRAIN-fitted reference on development validation; retain all tries.

    A skipped capacity/workspace candidate is explicitly recorded. Numerical
    failure propagates rather than silently discarding a difficult competitor.
    Degree/ridge ties retain the first declared candidate. No TEST argument exists.
    """
    x_train, y_train = _validate_xy(x_train, y_train)
    x_validation, y_validation = _validate_xy(x_validation, y_validation)
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise ValueError("TRAIN and validation observed dimensions disagree")
    degrees, ridges = tuple(degrees), tuple(ridges)
    if (
        not degrees
        or not ridges
        or len(set(degrees)) != len(degrees)
        or len(set(ridges)) != len(ridges)
    ):
        raise ValueError("Nonempty unique degree and ridge choices required")
    records, best_model, best_error, best_index = [], None, math.inf, None
    for degree, ridge in itertools.product(degrees, ridges):
        inventory = polynomial_inventory(degree, x_train.shape[1], x_train.shape[2])
        try:
            model, receipt = fit_polynomial_reference(
                x_train,
                y_train,
                degree=degree,
                ridge=ridge,
                parameter_ceiling=parameter_ceiling,
                max_features=max_features,
                max_workspace_bytes=max_workspace_bytes,
                rcond=rcond,
            )
        except ReferenceBudgetError as error:
            records.append(
                {
                    "degree": degree,
                    "ridge": ridge,
                    "inventory": inventory,
                    "status": "skipped_before_materialization",
                    "reason": str(error),
                }
            )
            continue
        prediction = model.predict(
            x_validation, max_workspace_bytes=max_workspace_bytes
        )
        mse = float(np.mean(np.square(prediction - y_validation)))
        if not math.isfinite(mse):
            raise FloatingPointError("Nonfinite polynomial validation MSE")
        records.append(
            {
                "degree": degree,
                "ridge": ridge,
                "status": "complete",
                "validation_mse": mse,
                "validation_predictions_sha256": _array_hash(prediction),
                "fit": receipt,
                "model": model.specification(),
            }
        )
        if mse < best_error:
            best_model, best_error, best_index = model, mse, len(records) - 1
    if best_model is None:
        raise ReferenceBudgetError(
            "No declared polynomial candidate fits the parameter/workspace budget"
        )
    return best_model, {
        "schema": "nonpolynomial_sharing_polynomial_selection_v1",
        "degrees": list(degrees),
        "ridges": list(ridges),
        "parameter_ceiling": parameter_ceiling,
        "validation_rows": len(x_validation),
        "validation_x_sha256": _array_hash(x_validation),
        "validation_y_sha256": _array_hash(y_validation),
        "selected_candidate_index": best_index,
        "selected_degree": best_model.degree,
        "selected_ridge": records[best_index]["ridge"],
        "selected_parameter_count": best_model.parameter_count,
        "selected_validation_mse": best_error,
        "candidates": records,
        "scope": "Development validation selects among TRAIN-only coefficient fits. This is a tuned development envelope, not confirmation evidence. No TEST labels consumed.",
    }
