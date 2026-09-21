"""Weight transformations for signed and nonnegative synaptic parameters.

``relu`` guarantees only nonnegativity because it can return zero.  ``exp``
and ``softplus`` are strictly positive as real-valued functions, although
finite-precision underflow can still produce zero.  The historical
``POSITIVE_WEIGHT_TRANSFORMS`` name remains as a compatibility alias.
"""

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as functional

WeightTransformType = Literal["identity", "exp", "relu", "softplus"]
NONNEGATIVE_WEIGHT_TRANSFORMS = {"exp", "relu", "softplus"}
# Backward-compatible public name used by existing validation and model code.
POSITIVE_WEIGHT_TRANSFORMS = NONNEGATIVE_WEIGHT_TRANSFORMS
STRICTLY_POSITIVE_WEIGHT_TRANSFORMS = {"exp", "softplus"}
NONNEGATIVE_TRANSFER_ACTIVATIONS = {"relu", "sigmoid", "softplus"}
_SUPPORTED_WEIGHT_TRANSFORMS: tuple[WeightTransformType, ...] = (
    "identity",
    "exp",
    "relu",
    "softplus",
)
_TensorTransform = Callable[[torch.Tensor], torch.Tensor]
DEFAULT_SAFE_EXP_MARGIN = 1.0
INVERSE_EXP_EPS = 1e-8
INVERSE_SOFTPLUS_EPS = 1e-8


def safe_exp(
    weights: torch.Tensor,
    margin: float = DEFAULT_SAFE_EXP_MARGIN,
    max_exponent: float | None = None,
) -> torch.Tensor:
    """Exponentiate without overflow or a second full-size output allocation."""
    if not torch.is_floating_point(weights):
        return torch.exp(weights)
    max_log = math.log(torch.finfo(weights.dtype).max) - float(margin)
    if max_exponent is not None:
        max_log = min(max_log, float(max_exponent))
    # clamp() creates a fresh tensor, so exponentiating that buffer in-place
    # does not mutate the parameter. This avoids simultaneously materializing
    # both clamp and exp outputs for multi-billion-contact sparse layers.
    return weights.clamp(max=max_log).exp_()


def inverse_softplus(
    positive_weights: torch.Tensor,
    *,
    eps: float = INVERSE_SOFTPLUS_EPS,
) -> torch.Tensor:
    """Numerically stable inverse of ``softplus`` for positive values.

    Uses ``x + log(-expm1(-x))`` instead of ``log(exp(x) - 1)`` so large
    weights map back to raw parameters without overflowing.
    """
    x = positive_weights.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def _identity(weights: torch.Tensor) -> torch.Tensor:
    return weights


def _inverse_exp(positive_weights: torch.Tensor) -> torch.Tensor:
    return torch.log(positive_weights + INVERSE_EXP_EPS)


_WEIGHT_TRANSFORMS: dict[WeightTransformType, _TensorTransform] = {
    "exp": safe_exp,
    "identity": _identity,
    "relu": functional.relu,
    "softplus": functional.softplus,
}
_INVERSE_WEIGHT_TRANSFORMS: dict[WeightTransformType, _TensorTransform] = {
    "exp": _inverse_exp,
    "identity": _identity,
    "relu": _identity,
    "softplus": inverse_softplus,
}


def _unsupported_transform_error(transform_type: str) -> ValueError:
    supported = ", ".join(f"'{name}'" for name in _SUPPORTED_WEIGHT_TRANSFORMS)
    return ValueError(
        f"Unsupported weight transform type: {transform_type}. "
        f"Supported types: {supported}"
    )


def _lookup_transform(
    registry: dict[WeightTransformType, _TensorTransform],
    transform_type: str,
) -> _TensorTransform:
    """Return a transform from a registry or raise a supported-name error."""
    transform = registry.get(transform_type)
    if transform is None:
        raise _unsupported_transform_error(transform_type)
    return transform


def apply_weight_transform(
    weights: torch.Tensor, transform_type: WeightTransformType = "exp"
) -> torch.Tensor:
    """
    Apply the specified transformation to constrain the stored weights.

    Args:
        weights: Raw weight parameters (can be negative)
        transform_type: Type of transformation to apply

    Returns:
        Transformed weights. ``relu`` is nonnegative; ``exp`` and ``softplus``
        are strictly positive in real arithmetic.

    Raises:
        ValueError: If transform_type is not supported
    """
    return _lookup_transform(_WEIGHT_TRANSFORMS, transform_type)(weights)


def get_weight_transform_fn(transform_type: WeightTransformType) -> _TensorTransform:
    """
    Get a transformation function for the specified type.

    Args:
        transform_type: Type of transformation

    Returns:
        Function that applies the transformation
    """
    return _lookup_transform(_WEIGHT_TRANSFORMS, transform_type)


def inverse_weight_transform(
    positive_weights: torch.Tensor, transform_type: WeightTransformType = "exp"
) -> torch.Tensor:
    """
    Apply the inverse transformation to get raw parameters from positive weights.

    This is useful for initialization or analysis.

    Args:
        positive_weights: Positive weight values
        transform_type: Type of transformation that was applied

    Returns:
        Raw parameters that would produce the given positive weights
    """
    return _lookup_transform(_INVERSE_WEIGHT_TRANSFORMS, transform_type)(
        positive_weights
    )


__all__ = [
    "NONNEGATIVE_TRANSFER_ACTIVATIONS",
    "NONNEGATIVE_WEIGHT_TRANSFORMS",
    "POSITIVE_WEIGHT_TRANSFORMS",
    "STRICTLY_POSITIVE_WEIGHT_TRANSFORMS",
    "WeightTransformType",
    "apply_weight_transform",
    "get_weight_transform_fn",
    "inverse_softplus",
    "inverse_weight_transform",
    "safe_exp",
]
