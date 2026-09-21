from __future__ import annotations

from math import erf, exp, isfinite, log, sqrt

import torch
from scipy.stats import norm

MAX_CALIBRATION_QUANTILE_SAMPLES = 1_000_000
ANALYTICAL_INITIALIZATION_GRID_STEP = 0.01
MAX_ANALYTICAL_INITIALIZATION_GRID_STEPS = 1_000_000


ADAPTIVE_INITIALIZATION_POLICY_ALIASES = {
    None: "preserve_shunting_center",
    "preserve": "preserve_shunting_center",
    "preserve_center": "preserve_shunting_center",
    "preserve_shunting": "preserve_shunting_center",
    "preserve_shunting_center": "preserve_shunting_center",
    "center_preserving": "preserve_shunting_center",
    "legacy": "legacy_scale",
    "legacy_scale": "legacy_scale",
    "scale": "legacy_scale",
    "scale_conductance": "legacy_scale",
}


def normalize_adaptive_initialization_policy(policy: str | None) -> str:
    """Return the canonical adaptive-initialization policy name."""

    key = None if policy is None else str(policy).strip().lower()
    canonical = ADAPTIVE_INITIALIZATION_POLICY_ALIASES.get(key, key)
    if canonical not in {"preserve_shunting_center", "legacy_scale"}:
        raise ValueError(
            "Unknown adaptive_initialization_policy "
            f"{policy!r}; expected 'preserve_shunting_center' or 'legacy_scale'."
        )
    return canonical


def _safe_quantile(
    values: torch.Tensor,
    probs: torch.Tensor,
    *,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute quantiles on a capped deterministic subsample when needed."""

    if values.numel() == 0:
        raise ValueError("Cannot compute quantiles of an empty tensor")

    if max_samples is None:
        max_samples = MAX_CALIBRATION_QUANTILE_SAMPLES

    sample_n = int(values.numel())
    if sample_n <= max_samples:
        return torch.quantile(values, probs), sample_n

    idx = torch.linspace(
        0,
        sample_n - 1,
        steps=max_samples,
        dtype=torch.long,
        device=values.device,
    )
    sampled = values.index_select(0, idx)
    return torch.quantile(sampled, probs), int(sampled.numel())


def _aggregate_quantiles_from_chunks(
    chunks: list[torch.Tensor],
    probs: torch.Tensor,
    *,
    aggregation: str = "global",
) -> tuple[torch.Tensor, int]:
    """Aggregate quantiles from voltage chunks.

    ``global`` matches the original behavior by pooling all chunks before
    fitting quantiles. ``chunk_median`` computes the requested quantiles for
    each chunk independently and then takes the median across chunks. The
    latter is useful for recurrent models where each hook invocation typically
    corresponds to a different timestep and we want a calibration that reflects
    typical temporal slices rather than the pooled multimodal distribution.
    """

    valid_chunks = []
    for chunk in chunks:
        flat = chunk.reshape(-1).double()
        flat = flat[torch.isfinite(flat)]
        if flat.numel() > 0:
            valid_chunks.append(flat)

    if not valid_chunks:
        raise ValueError("Cannot compute quantiles from empty calibration chunks")

    if aggregation == "global":
        pooled = torch.cat(valid_chunks)
        return _safe_quantile(pooled, probs)

    if aggregation == "chunk_median":
        per_chunk = []
        total_n = 0
        for chunk in valid_chunks:
            q_vals, sample_n = _safe_quantile(chunk, probs)
            per_chunk.append(q_vals)
            total_n += sample_n
        return torch.median(torch.stack(per_chunk, dim=0), dim=0).values, total_n

    raise ValueError(
        f"Unknown quantile aggregation mode {aggregation!r} "
        f"(expected 'global' or 'chunk_median')"
    )


def compute_expectation_truncated_log_normal(
    mean,
    std,
    upper_quantile,
):
    log_alpha = mean + std * norm.ppf(1 - upper_quantile).item()

    numerator = exp(mean + 0.5 * std**2) * (
        1 - erf((log_alpha - mean - std**2) / (std * sqrt(2)))
    )
    denominator = 1 - erf((log_alpha - mean) / (std * sqrt(2)))

    expectation = numerator / denominator
    return expectation


def compute_expectation_truncated_inverse_softplus_normal(
    mean,
    std,
    upper_quantile,
):
    z = mean + std * norm.ppf(1 - upper_quantile).item()

    normal = torch.distributions.Normal(mean, std)

    truncated_samples_list = []
    max_iter = 100
    iter_counter = 0
    done = False
    while not done:
        iter_counter += 1
        samples = normal.sample((10000,))
        truncated_samples = samples[samples > z]
        truncated_samples_list.append(truncated_samples)
        all_truncated_samples = torch.cat(truncated_samples_list)
        if (all_truncated_samples.shape[0] >= 10000) or (iter_counter >= max_iter):
            done = True

    transformed_samples = torch.log(1 + torch.exp(all_truncated_samples))
    return torch.mean(transformed_samples).item()


def _truncated_normal_moments(
    mean: float, std: float, upper_quantile: float
) -> tuple[float, float, float]:
    """
    Compute moments of X | X > alpha where X ~ Normal(mean, std^2),
    and alpha is the threshold corresponding to the given upper_quantile (top proportion kept).

    Returns:
        alpha: threshold
        mean_c: E[X | X > alpha]
        var_c: Var[X | X > alpha]
    """
    # Determine threshold alpha such that P(X > alpha) = upper_quantile.
    # Clamp to open interval (0, 1) to avoid +/-inf from norm.ppf at the edges
    # (e.g., K == in_features -> upper_quantile == 1.0).
    upper_quantile = min(max(float(upper_quantile), 1e-12), 1 - 1e-12)

    # Determine threshold alpha such that P(X > alpha) = upper_quantile
    # gamma = (alpha - mean) / std
    gamma = norm.ppf(1 - upper_quantile)
    alpha = mean + std * gamma

    # Avoid numerical issues for extreme quantiles
    Q = max(1e-12, 1 - norm.cdf(gamma))
    phi = norm.pdf(gamma)
    lam = phi / Q

    mean_c = mean + std * lam
    var_c = (std**2) * (1 + gamma * lam - lam**2)

    # Ensure variance is non-negative (can be slightly negative due to numerical errors)
    var_c = max(0.0, var_c)

    return alpha, mean_c, var_c


def _compute_expectation_truncated_softplus(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[softplus(X) | X > alpha] via delta method using truncated normal moments.
    softplus(x) = ln(1 + e^x),
    g'(x) = sigmoid(x), g''(x) = sigmoid(x) * (1 - sigmoid(x)).
    """
    _, mu_c, var_c = _truncated_normal_moments(mean, std, upper_quantile)

    # The lower clamp avoids underflow in the small-value branch.  Do not cap
    # the upper tail: for ``mu_c > 20`` the implementation below already uses
    # the overflow-safe asymptote ``softplus(mu_c) ~= mu_c``.  The former
    # upper cap at 50 made some valid E/I balance targets mathematically
    # unreachable and left constructor-time grid searches in an infinite loop.
    mu_c = max(-50.0, mu_c)

    # Safe computation of softplus to avoid overflow
    if mu_c > 20:
        sp = mu_c  # For large x, softplus(x) ≈ x
    else:
        sp = log(1 + exp(mu_c))

    # Safe computation of sigmoid to avoid overflow
    if mu_c > 0:
        sig = 1.0 / (1.0 + exp(-mu_c))
    else:
        exp_mu = exp(mu_c)
        sig = exp_mu / (1.0 + exp_mu)

    sp_corr = sp + 0.5 * sig * (1 - sig) * var_c
    return sp_corr


def _compute_expectation_truncated_relu(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[ReLU(X) | X > alpha]. Since conditioning ensures X > alpha,
    and for typical TopK alpha is positive, we use E[X | X > alpha] as a good approximation.
    """
    _, mu_c, _ = _truncated_normal_moments(mean, std, upper_quantile)
    # If alpha <= 0 but mu_c still small, ensure non-negativity
    return max(0.0, mu_c)


def _expected_weight_for_transform(
    transform: str, mean: float, std: float, upper_quantile: float
) -> float:
    """
    Compute/approximate E[g(X) | X > alpha] where g is the transform and X ~ N(mean, std^2).

    For 'exp' uses the existing closed-form truncated log-normal expectation.
    For 'softplus' and 'relu' uses truncated normal moments with a delta-method approximation.
    """
    t = (transform or "exp").lower()
    if t == "exp":
        return compute_expectation_truncated_log_normal(
            mean=mean, std=std, upper_quantile=upper_quantile
        )
    if t == "softplus":
        return _compute_expectation_truncated_softplus(mean, std, upper_quantile)
    if t == "relu":
        return _compute_expectation_truncated_relu(mean, std, upper_quantile)
    # Fallback to exp-style expectation if unknown
    return compute_expectation_truncated_log_normal(
        mean=mean, std=std, upper_quantile=upper_quantile
    )


def _mean_for_expected_weight(
    transform: str,
    std: float,
    upper_quantile: float,
    target_weight: float,
    *,
    reference_mean: float = 0.0,
) -> float:
    """Find a raw-weight mean whose transformed top-K expectation hits target."""

    target = max(float(target_weight), 1e-12)
    t = (transform or "exp").lower()

    if t == "exp":
        reference = _expected_weight_for_transform(
            transform=t,
            mean=reference_mean,
            std=std,
            upper_quantile=upper_quantile,
        )
        return reference_mean + log(target / max(reference, 1e-12))

    low, high = -80.0, 80.0
    for _ in range(96):
        mid = 0.5 * (low + high)
        value = _expected_weight_for_transform(
            transform=t,
            mean=mid,
            std=std,
            upper_quantile=upper_quantile,
        )
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _solve_first_crossing_expected_weight_mean(
    transform: str,
    std: float,
    upper_quantile: float,
    target_weight: float,
    *,
    direction: int,
    reference_mean: float = 0.0,
    step: float = ANALYTICAL_INITIALIZATION_GRID_STEP,
    max_steps: int = MAX_ANALYTICAL_INITIALIZATION_GRID_STEPS,
) -> float:
    """Return the first monotone mean-grid point that crosses ``target_weight``.

    Historical analytical initialization moved the raw-weight mean by 0.01 in
    an unbounded Python loop.  This solver preserves that first-crossing grid
    rule while using exponential bracketing plus integer binary search.  It
    validates both the crossing and its immediate predecessor before returning,
    and fails closed for non-finite or unreachable targets.

    ``direction=1`` searches upward for the first expectation greater than or
    equal to the target.  ``direction=-1`` searches downward for the first
    expectation less than or equal to the target.
    """

    if direction not in {-1, 1}:
        raise ValueError("analytical initialization direction must be -1 or 1")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("analytical initialization max_steps must be a positive int")
    numeric = {
        "std": float(std),
        "upper_quantile": float(upper_quantile),
        "target_weight": float(target_weight),
        "reference_mean": float(reference_mean),
        "step": float(step),
    }
    if any(not isfinite(value) for value in numeric.values()):
        raise ValueError("analytical initialization grid inputs must be finite")
    if numeric["std"] <= 0.0:
        raise ValueError("analytical initialization std must be positive")
    if not 0.0 < numeric["upper_quantile"] <= 1.0:
        raise ValueError("analytical initialization upper_quantile must be in (0, 1]")
    if numeric["target_weight"] < 0.0:
        raise ValueError("analytical initialization target weight must be non-negative")
    if numeric["step"] <= 0.0:
        raise ValueError("analytical initialization grid step must be positive")

    def mean_at(tick: int) -> float:
        return numeric["reference_mean"] + direction * numeric["step"] * tick

    def expectation_at(tick: int) -> float:
        mean = mean_at(tick)
        try:
            value = float(
                _expected_weight_for_transform(
                    transform=transform,
                    mean=mean,
                    std=numeric["std"],
                    upper_quantile=numeric["upper_quantile"],
                )
            )
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise RuntimeError(
                "analytical initialization expectation failed at "
                f"mean={mean}, tick={tick}"
            ) from exc
        if not isfinite(value):
            raise RuntimeError(
                "analytical initialization expectation became non-finite at "
                f"mean={mean}, tick={tick}"
            )
        return value

    def crossed(value: float) -> bool:
        if direction > 0:
            return value >= numeric["target_weight"]
        return value <= numeric["target_weight"]

    initial = expectation_at(0)
    if crossed(initial):
        return numeric["reference_mean"]

    lower_tick = 0
    upper_tick = 1
    upper_value = expectation_at(upper_tick)
    while not crossed(upper_value):
        lower_tick = upper_tick
        if upper_tick == max_steps:
            raise RuntimeError(
                "analytical initialization target is unreachable on the bounded "
                f"{numeric['step']}-spaced grid after {max_steps} steps"
            )
        upper_tick = min(max_steps, upper_tick * 2)
        upper_value = expectation_at(upper_tick)

    while upper_tick - lower_tick > 1:
        middle_tick = (lower_tick + upper_tick) // 2
        if crossed(expectation_at(middle_tick)):
            upper_tick = middle_tick
        else:
            lower_tick = middle_tick

    solution_value = expectation_at(upper_tick)
    predecessor_value = expectation_at(upper_tick - 1)
    if not crossed(solution_value) or crossed(predecessor_value):
        raise RuntimeError(
            "analytical initialization grid solver did not isolate the exact "
            "first crossing"
        )
    return mean_at(upper_tick)


__all__ = [
    "ADAPTIVE_INITIALIZATION_POLICY_ALIASES",
    "ANALYTICAL_INITIALIZATION_GRID_STEP",
    "MAX_ANALYTICAL_INITIALIZATION_GRID_STEPS",
    "MAX_CALIBRATION_QUANTILE_SAMPLES",
    "_aggregate_quantiles_from_chunks",
    "_compute_expectation_truncated_relu",
    "_compute_expectation_truncated_softplus",
    "_expected_weight_for_transform",
    "_mean_for_expected_weight",
    "_safe_quantile",
    "_solve_first_crossing_expected_weight_mean",
    "_truncated_normal_moments",
    "compute_expectation_truncated_inverse_softplus_normal",
    "compute_expectation_truncated_log_normal",
    "normalize_adaptive_initialization_policy",
]
