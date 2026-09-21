"""Fail-closed validation of sample-limited FMI output-spectrum ranks.

An output-spectrum rank is a candidate morphology width until it is shown to
be below the centered sample ceiling, stable as the calibration set grows,
and predictive on examples that were not used to fit the eigenspace.  This
module performs those checks on deterministic nested subsets sampled without
replacement and a fixed, disjoint holdout.

Validation remains tied to one prospectively selected energy target. Separate
multi-energy curves, full-empirical-span holdout residuals, rank-growth slopes,
and leading-subspace principal angles diagnose *why* a gate failed. In
particular, they distinguish a stable lower-energy core from a rank that keeps
tracking the sample ceiling, and show when no rank available from the current
training capture can reconstruct the holdout. These diagnostics never promote
an unresolved primary rank.

The validator deliberately returns ``validated_rank=None`` whenever any gate
fails.  The diagnostic candidate and its full trajectory remain in the report
so an unresolved estimate cannot be mistaken for a theory-derived width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.analysis.fmi.spectrum import (
    SPECTRUM_ESTIMATORS,
    fit_robust_task_weighted_spectrum,
    n_min,
)

__all__ = [
    "DEFAULT_DIAGNOSTIC_ENERGY_TARGETS",
    "build_parser",
    "main",
    "nested_disjoint_split_indices",
    "nested_group_disjoint_split_indices",
    "rank_capacity_diagnostics",
    "subspace_convergence_diagnostics",
    "validate_rank_estimate",
]


DEFAULT_DIAGNOSTIC_ENERGY_TARGETS = (0.9, 0.95, 0.99, 0.995)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(payload: Any) -> Any:
    """Replace non-finite floats so artifacts remain strict JSON."""
    if isinstance(payload, float) and not math.isfinite(payload):
        if math.isnan(payload):
            return "NaN"
        return "Infinity" if payload > 0 else "-Infinity"
    if isinstance(payload, dict):
        return {key: _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(value) for value in payload]
    return payload


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and value bytes without one large host copy."""
    detached = tensor.detach()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(detached.dtype), "shape": list(detached.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if detached.numel() == 0:
        return digest.hexdigest()
    rows = detached.reshape(-1, detached.shape[-1]) if detached.ndim else detached[None]
    # Bound transfers for captured LLM boundaries while retaining byte-exact
    # provenance. Viewing as uint8 supports bfloat16, unlike NumPy conversion.
    bytes_per_row = max(1, rows[0].numel() * rows.element_size())
    rows_per_chunk = max(1, (16 * 1024 * 1024) // bytes_per_row)
    for start in range(0, rows.shape[0], rows_per_chunk):
        block = rows[start : start + rows_per_chunk].contiguous().cpu()
        digest.update(block.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _index_sha256(indices: torch.Tensor) -> str:
    canonical = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def nested_disjoint_split_indices(
    total_rows: int,
    *,
    sample_sizes: Sequence[int],
    holdout_rows: int,
    seed: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Return a fixed holdout and nested train prefixes without replacement.

    A single seeded CPU permutation defines the complete split.  Holdout rows
    are the first prefix; every train subset is a prefix of the remaining
    permutation.  Consequently, smaller train subsets are exact subsets of
    larger ones and no row can appear in both train and holdout.
    """
    rows = int(total_rows)
    holdout = int(holdout_rows)
    sizes = sorted({int(size) for size in sample_sizes})
    if rows < 4:
        raise ValueError(f"total_rows must be >= 4, got {rows}")
    if holdout < 1 or holdout >= rows - 1:
        raise ValueError(
            "holdout_rows must leave at least two training rows, "
            f"got total_rows={rows}, holdout_rows={holdout}"
        )
    if not sizes or sizes[0] < 2:
        raise ValueError("sample_sizes must contain positive sizes >= 2")
    train_rows = rows - holdout
    if sizes[-1] > train_rows:
        raise ValueError(
            f"largest sample size {sizes[-1]} exceeds training pool {train_rows}"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(rows, generator=generator)
    holdout_indices = permutation[:holdout]
    train_order = permutation[holdout:]
    subsets = {size: train_order[:size].clone() for size in sizes}
    return holdout_indices, subsets


def nested_group_disjoint_split_indices(
    group_ids: torch.Tensor,
    *,
    sample_sizes: Sequence[int],
    holdout_rows: int,
    seed: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Return exact-size nested splits with disjoint source groups.

    Groups are permuted once. Complete groups are assigned to the holdout
    until it contains at least ``holdout_rows`` rows; only the requested
    prefix is used and any remaining rows from those groups are discarded.
    Training rows are then prefixes of a separate permutation of all rows in
    the remaining groups. Thus row counts remain exact, train subsets remain
    nested, and no source group can cross the train/holdout boundary.
    """

    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    if group_ids.dtype.is_floating_point or group_ids.dtype.is_complex:
        raise ValueError("group_ids must use an integer dtype")
    canonical = group_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    rows = int(canonical.numel())
    holdout = int(holdout_rows)
    sizes = sorted({int(size) for size in sample_sizes})
    if rows < 4:
        raise ValueError(f"group_ids must contain >= 4 rows, got {rows}")
    if holdout < 1 or holdout >= rows - 1:
        raise ValueError(
            "holdout_rows must leave at least two training rows, "
            f"got total_rows={rows}, holdout_rows={holdout}"
        )
    if not sizes or sizes[0] < 2:
        raise ValueError("sample_sizes must contain positive sizes >= 2")
    unique_groups = torch.unique(canonical, sorted=True)
    if unique_groups.numel() < 2:
        raise ValueError("group-disjoint validation requires at least two groups")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    group_order = unique_groups[
        torch.randperm(int(unique_groups.numel()), generator=generator)
    ]
    grouped_rows: list[torch.Tensor] = []
    for group in group_order:
        indices = torch.nonzero(canonical == group, as_tuple=False).flatten()
        indices = indices[torch.randperm(int(indices.numel()), generator=generator)]
        grouped_rows.append(indices)

    holdout_groups = 0
    heldout_pool_rows = 0
    while holdout_groups < len(grouped_rows) and heldout_pool_rows < holdout:
        heldout_pool_rows += int(grouped_rows[holdout_groups].numel())
        holdout_groups += 1
    if holdout_groups == len(grouped_rows):
        raise ValueError("holdout group assignment leaves no independent train group")
    holdout_indices = torch.cat(grouped_rows[:holdout_groups])[:holdout].clone()
    train_pool = torch.cat(grouped_rows[holdout_groups:])
    # Once source groups have been isolated from the holdout, mix their rows
    # globally so a small nested training rung is not accidentally dominated
    # by the first one or two groups in the group permutation.
    train_order = train_pool[
        torch.randperm(int(train_pool.numel()), generator=generator)
    ]
    if sizes[-1] > int(train_order.numel()):
        raise ValueError(
            f"largest sample size {sizes[-1]} exceeds group-disjoint training "
            f"pool {int(train_order.numel())}; {heldout_pool_rows - holdout} rows "
            "were discarded to keep holdout groups disjoint"
        )
    subsets = {size: train_order[:size].clone() for size in sizes}
    return holdout_indices, subsets


def rank_capacity_diagnostics(
    rank: int,
    *,
    sample_size: int,
    feature_dimension: int,
    center: bool = True,
) -> dict[str, float | int | bool]:
    """Diagnose proximity to sample-limited and feature-limited capacity."""
    candidate = int(rank)
    rows = int(sample_size)
    dimension = int(feature_dimension)
    if candidate < 0 or rows < 1 or dimension < 1:
        raise ValueError("rank must be non-negative and dimensions must be positive")
    sample_ceiling = min(dimension, rows - (1 if center else 0))
    fraction = float(candidate / sample_ceiling) if sample_ceiling > 0 else math.inf
    return {
        "centered_sample_ceiling": sample_ceiling,
        "capacity_fraction": fraction,
        "feature_capacity_fraction": float(candidate / dimension),
        "sample_limited": bool(rows - (1 if center else 0) < dimension),
    }


def subspace_convergence_diagnostics(
    previous_basis: torch.Tensor,
    current_basis: torch.Tensor,
    *,
    rank_cap: int = 64,
) -> dict[str, float | int | bool]:
    """Compare leading fitted subspaces without assuming eigenvector signs.

    The two bases must live in the same feature coordinates and have
    orthonormal columns.  Comparison is deliberately capped: a dense overlap
    of two thousand-dimensional bases would add cubic work to a validator
    whose spectrum fit is already expensive.  The report states both fitted
    ranks and the exact leading rank that was compared, so a capped leading-
    subspace diagnostic cannot be mistaken for whole-subspace convergence.

    ``rms_principal_sine`` and ``max_principal_sine`` are zero for identical
    subspaces and one for orthogonal subspaces. ``previous_unexplained`` is
    the fraction of the compared previous basis outside the compared current
    basis. All quantities are invariant to column signs and rotations within
    either compared subspace.
    """

    if previous_basis.ndim != 2 or current_basis.ndim != 2:
        raise ValueError("subspace bases must be two-dimensional")
    if previous_basis.shape[0] != current_basis.shape[0]:
        raise ValueError("subspace bases must share the feature dimension")
    cap = int(rank_cap)
    if cap < 1:
        raise ValueError("rank_cap must be positive")
    previous_rank = int(previous_basis.shape[1])
    current_rank = int(current_basis.shape[1])
    compared_rank = min(previous_rank, current_rank, cap)
    if compared_rank == 0:
        return {
            "previous_rank": previous_rank,
            "current_rank": current_rank,
            "rank_cap": cap,
            "compared_leading_rank": 0,
            "comparison_was_capped": False,
            "rms_principal_sine": 0.0 if previous_rank == current_rank else 1.0,
            "max_principal_sine": 0.0 if previous_rank == current_rank else 1.0,
            "previous_unexplained_fraction": (0.0 if previous_rank == 0 else 1.0),
        }

    previous = previous_basis[:, :compared_rank]
    current = current_basis[:, :compared_rank].to(
        device=previous.device, dtype=previous.dtype
    )
    singular_values = torch.linalg.svdvals(previous.T @ current).clamp(0.0, 1.0)
    sine_squared = (1.0 - singular_values.square()).clamp(0.0, 1.0)
    rms_sine = float(sine_squared.mean().sqrt().detach().cpu())
    max_sine = float(sine_squared.max().sqrt().detach().cpu())
    return {
        "previous_rank": previous_rank,
        "current_rank": current_rank,
        "rank_cap": cap,
        "compared_leading_rank": compared_rank,
        "comparison_was_capped": bool(min(previous_rank, current_rank) > compared_rank),
        "rms_principal_sine": rms_sine,
        "max_principal_sine": max_sine,
        "previous_unexplained_fraction": float(sine_squared.mean().detach().cpu()),
    }


def _training_reconstruction_residual(eigenvalues: torch.Tensor, rank: int) -> float:
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return 0.0 if float(total) == 0.0 else math.inf
    retained = eigenvalues[: int(rank)].sum()
    value = float((1.0 - retained / total).detach().cpu())
    return min(1.0, max(0.0, value)) if math.isfinite(value) else value


def _numerical_spectrum_rank(
    eigenvalues: torch.Tensor, *, sample_size: int, feature_dimension: int
) -> int:
    """Numerical rank using the standard matrix-rank relative tolerance."""

    if eigenvalues.numel() == 0 or float(eigenvalues[0]) <= 0.0:
        return 0
    dtype = eigenvalues.dtype
    epsilon = torch.finfo(dtype).eps if dtype.is_floating_point else 0.0
    singular_rtol = max(int(sample_size), int(feature_dimension)) * float(epsilon)
    eigenvalue_threshold = float(eigenvalues[0]) * singular_rtol**2
    return int((eigenvalues > eigenvalue_threshold).sum().detach().cpu())


def _trajectory_power_slope(
    sample_sizes: Sequence[int], values: Sequence[float]
) -> float | None:
    """Log-log slope, or ``None`` when a positive trajectory is unavailable."""

    if len(sample_sizes) < 2 or len(sample_sizes) != len(values):
        return None
    x = torch.tensor(sample_sizes, dtype=torch.float64).log()
    y_values = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(y_values).all()) or bool((y_values <= 0).any()):
        return None
    y = y_values.log()
    centered = x - x.mean()
    denominator = centered.square().sum()
    if float(denominator) <= 0.0:
        return None
    return float((centered * (y - y.mean())).sum() / denominator)


def _heldout_reconstruction_residual(
    transformed_holdout: torch.Tensor,
    eigenvectors: torch.Tensor,
    rank: int,
) -> float:
    total = transformed_holdout.square().sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return 0.0 if float(total) == 0.0 else math.inf
    if int(rank) >= int(eigenvectors.shape[0]):
        # A complete orthonormal feature basis reconstructs identically and
        # should not trigger an unnecessary O(M d^2) projection in the
        # above-feature-dimension diagnostic rung.
        return 0.0
    basis = eigenvectors[:, : int(rank)]
    if basis.numel() == 0:
        residual = total
    else:
        projected = transformed_holdout @ basis
        # For an orthonormal basis, Pythagoras gives the residual energy
        # directly. This is exactly the same projection diagnostic without a
        # second matrix multiply back into feature space.
        residual = (total - projected.square().sum()).clamp_min(0.0)
    value = float((residual / total).detach().cpu())
    # Roundoff can put a projection residual infinitesimally outside [0, 1].
    return min(1.0, max(0.0, value)) if math.isfinite(value) else value


def _sign_canonical_basis_sha256(
    eigenvectors: torch.Tensor,
    rank: int,
) -> str:
    basis = eigenvectors[:, : int(rank)].detach().clone()
    if basis.numel():
        # Eigenvector signs are arbitrary. Canonicalizing each column makes
        # provenance stable across equivalent sign choices.
        pivots = basis.abs().argmax(dim=0)
        columns = torch.arange(basis.shape[1], device=basis.device)
        signs = torch.sign(basis[pivots, columns])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs.unsqueeze(0)
    return _tensor_sha256(basis)


def _failure(
    code: str,
    message: str,
    *,
    observed: Any = None,
    limit: Any = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {"code": code, "message": message}
    if observed is not None:
        failure["observed"] = observed
    if limit is not None:
        failure["limit"] = limit
    return failure


def validate_rank_estimate(
    matrix: torch.Tensor,
    *,
    sample_sizes: Sequence[int],
    holdout_rows: int,
    epsilon: float = 0.005,
    estimator: str = "ordinary",
    winsorize_quantile: float = 0.995,
    huber_c: float = 1.345,
    huber_max_iters: int = 5,
    mom_blocks: int = 16,
    seed: int = 0,
    center: bool = True,
    weight: torch.Tensor | None = None,
    max_capacity_fraction: float = 0.8,
    max_relative_rank_drift: float = 0.1,
    max_heldout_residual: float = 0.02,
    max_heldout_residual_drift: float = 0.01,
    tail_points: int = 2,
    diagnostic_energy_targets: Sequence[float] = DEFAULT_DIAGNOSTIC_ENERGY_TARGETS,
    subspace_rank_cap: int = 64,
    group_ids: torch.Tensor | None = None,
    require_group_disjoint: bool = False,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an FMI spectrum rank on nested train sets and one holdout.

    The final candidate passes only if all of the following are true over the
    requested tail of the sample-size trajectory:

    * rank/sample-ceiling is strictly below ``max_capacity_fraction``;
    * relative rank drift between adjacent fits is no larger than
      ``max_relative_rank_drift``;
    * final estimator-aware held-out reconstruction residual is no larger
      than ``max_heldout_residual``;
    * adjacent held-out residual drift is no larger than
      ``max_heldout_residual_drift``.

    Configuration errors raise immediately. Scientific non-validation is
    encoded in a complete report with explicit failure reasons and
    ``validated_rank=None``. The gate defaults are conservative operational
    starting points, not universal constants; a campaign should preregister
    and justify them before inspecting its validation trajectory. Diagnostic
    energy targets and the subspace rank cap likewise must be fixed before a
    confirmatory capture; they characterize resolution but never alter the
    four primary acceptance checks.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be [M, d], got {tuple(matrix.shape)}")
    if matrix.shape[0] < 4 or matrix.shape[1] < 1:
        raise ValueError("matrix must have at least four rows and one column")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix contains non-finite values")
    if require_group_disjoint and group_ids is None:
        raise ValueError(
            "require_group_disjoint=True requires captured source group IDs"
        )
    canonical_group_ids: torch.Tensor | None = None
    if group_ids is not None:
        if group_ids.ndim != 1 or int(group_ids.numel()) != int(matrix.shape[0]):
            raise ValueError("group_ids must have exactly one value per matrix row")
        if group_ids.dtype.is_floating_point or group_ids.dtype.is_complex:
            raise ValueError("group_ids must use an integer dtype")
        canonical_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64)
    name = str(estimator).strip().lower()
    if name not in SPECTRUM_ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {SPECTRUM_ESTIMATORS}, got {estimator!r}"
        )
    sizes = sorted({int(size) for size in sample_sizes})
    if int(tail_points) < 2:
        raise ValueError(f"tail_points must be >= 2, got {tail_points}")
    if len(sizes) < int(tail_points):
        raise ValueError(
            f"need at least tail_points={tail_points} distinct sample sizes, "
            f"got {sizes}"
        )
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    energy_targets = sorted(
        {
            *(float(value) for value in diagnostic_energy_targets),
            float(1.0 - epsilon),
        }
    )
    if not energy_targets or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in energy_targets
    ):
        raise ValueError("diagnostic energy targets must be finite values in (0, 1)")
    if int(subspace_rank_cap) < 1:
        raise ValueError("subspace_rank_cap must be positive")
    if not 0.0 < float(max_capacity_fraction) <= 1.0:
        raise ValueError("max_capacity_fraction must be in (0, 1]")
    for label, value in (
        ("max_relative_rank_drift", max_relative_rank_drift),
        ("max_heldout_residual", max_heldout_residual),
        ("max_heldout_residual_drift", max_heldout_residual_drift),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{label} must be non-negative, got {value}")

    if canonical_group_ids is None:
        holdout_indices, train_subsets = nested_disjoint_split_indices(
            int(matrix.shape[0]),
            sample_sizes=sizes,
            holdout_rows=int(holdout_rows),
            seed=int(seed),
        )
    else:
        holdout_indices, train_subsets = nested_group_disjoint_split_indices(
            canonical_group_ids,
            sample_sizes=sizes,
            holdout_rows=int(holdout_rows),
            seed=int(seed),
        )
    holdout = matrix[holdout_indices.to(matrix.device)]
    trajectory: list[dict[str, Any]] = []
    previous_bases: dict[float | str, torch.Tensor] = {}
    for size in sizes:
        indices = train_subsets[size]
        train = matrix[indices.to(matrix.device)]
        fit = fit_robust_task_weighted_spectrum(
            train,
            weight,
            estimator=name,
            winsorize_quantile=float(winsorize_quantile),
            huber_c=float(huber_c),
            huber_max_iters=int(huber_max_iters),
            mom_blocks=int(mom_blocks),
            # Keep estimator randomness fixed so the trajectory isolates the
            # effect of adding rows rather than changing MoM partitions too.
            seed=int(seed),
            center=bool(center),
        )
        previous = trajectory[-1] if trajectory else None
        estimator_centers = bool(center or name != "ordinary")
        heldout_rows_transformed = fit.transform_rows(holdout)
        energy_curve: list[dict[str, Any]] = []
        current_bases: dict[float | str, torch.Tensor] = {}
        subspace_cache: dict[tuple[int, int], dict[str, float | int | bool]] = {}
        for retained_energy in energy_targets:
            energy_epsilon = float(1.0 - retained_energy)
            energy_rank = int(n_min(fit.eigenvalues, energy_epsilon))
            energy_capacity = rank_capacity_diagnostics(
                energy_rank,
                sample_size=int(size),
                feature_dimension=int(matrix.shape[1]),
                center=estimator_centers,
            )
            energy_residual = _heldout_reconstruction_residual(
                heldout_rows_transformed,
                fit.eigenvectors,
                energy_rank,
            )
            training_residual = _training_reconstruction_residual(
                fit.eigenvalues, energy_rank
            )
            basis = fit.eigenvectors[:, :energy_rank].detach()
            current_bases[retained_energy] = basis
            previous_basis = previous_bases.get(retained_energy)
            previous_energy_point = (
                None
                if previous is None
                else next(
                    point
                    for point in previous["multi_energy_rank_curve"]
                    if math.isclose(
                        float(point["retained_energy_target"]),
                        retained_energy,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            )
            energy_rank_drift = (
                None
                if previous_energy_point is None
                else abs(energy_rank - int(previous_energy_point["rank"]))
                / max(energy_rank, int(previous_energy_point["rank"]), 1)
            )
            energy_residual_drift = (
                None
                if previous_energy_point is None
                else abs(
                    energy_residual
                    - float(previous_energy_point["heldout_reconstruction_residual"])
                )
            )
            convergence = None
            if previous_basis is not None:
                comparison_key = (
                    min(int(previous_basis.shape[1]), int(subspace_rank_cap)),
                    min(int(basis.shape[1]), int(subspace_rank_cap)),
                )
                convergence = subspace_cache.get(comparison_key)
                if convergence is None:
                    convergence = subspace_convergence_diagnostics(
                        previous_basis,
                        basis,
                        rank_cap=int(subspace_rank_cap),
                    )
                    subspace_cache[comparison_key] = convergence
            energy_curve.append(
                {
                    "retained_energy_target": retained_energy,
                    "epsilon": energy_epsilon,
                    "rank": energy_rank,
                    **energy_capacity,
                    "relative_rank_drift_from_previous": energy_rank_drift,
                    "training_reconstruction_residual": training_residual,
                    "heldout_reconstruction_residual": energy_residual,
                    "heldout_residual_drift_from_previous": energy_residual_drift,
                    "heldout_generalization_gap": energy_residual - training_residual,
                    "leading_subspace_from_previous": convergence,
                }
            )

        primary_energy = float(1.0 - epsilon)
        primary = next(
            point
            for point in energy_curve
            if math.isclose(
                float(point["retained_energy_target"]),
                primary_energy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        rank = int(primary["rank"])
        capacity = {
            key: primary[key]
            for key in (
                "centered_sample_ceiling",
                "capacity_fraction",
                "feature_capacity_fraction",
                "sample_limited",
            )
        }
        residual = float(primary["heldout_reconstruction_residual"])

        empirical_rank = _numerical_spectrum_rank(
            fit.eigenvalues,
            sample_size=int(size),
            feature_dimension=int(matrix.shape[1]),
        )
        empirical_basis = fit.eigenvectors[:, :empirical_rank].detach()
        full_span_residual = _heldout_reconstruction_residual(
            heldout_rows_transformed,
            fit.eigenvectors,
            empirical_rank,
        )
        previous_empirical_basis = previous_bases.get("full_empirical_span")
        empirical_convergence = (
            None
            if previous_empirical_basis is None
            else subspace_convergence_diagnostics(
                previous_empirical_basis,
                empirical_basis,
                rank_cap=int(subspace_rank_cap),
            )
        )
        current_bases["full_empirical_span"] = empirical_basis
        rank_drift = (
            None
            if previous is None
            else abs(rank - int(previous["rank"])) / max(rank, int(previous["rank"]), 1)
        )
        residual_drift = (
            None
            if previous is None
            else abs(residual - float(previous["heldout_reconstruction_residual"]))
        )
        trajectory.append(
            {
                "sample_size": int(size),
                "rank": rank,
                "spectral_energy_target": float(1.0 - epsilon),
                **capacity,
                "relative_rank_drift_from_previous": rank_drift,
                "heldout_reconstruction_residual": residual,
                "heldout_residual_drift_from_previous": residual_drift,
                "multi_energy_rank_curve": energy_curve,
                "full_empirical_span": {
                    "numerical_rank": empirical_rank,
                    "heldout_reconstruction_residual": full_span_residual,
                    "primary_residual_unavoidable_fraction": (
                        full_span_residual / residual if residual > 0.0 else 0.0
                    ),
                    "leading_subspace_from_previous": empirical_convergence,
                },
                "spectrum_total_energy": float(fit.eigenvalues.sum().detach().cpu()),
                "train_indices_sha256": _index_sha256(indices),
                # Sign canonicalization removes the arbitrary sign of each
                # fitted eigenvector. It intentionally does not claim
                # invariance to rotations inside a degenerate eigenspace.
                "sign_canonical_basis_sha256": _sign_canonical_basis_sha256(
                    fit.eigenvectors, rank
                ),
            }
        )
        maximum_stored_rank = max(
            int(basis.shape[1]) for basis in current_bases.values()
        )
        stored_basis = fit.eigenvectors[:, :maximum_stored_rank].detach().clone()
        previous_bases = {
            key: stored_basis[:, : int(basis.shape[1])]
            for key, basis in current_bases.items()
        }

    tail = trajectory[-int(tail_points) :]
    capacity_observed = max(float(point["capacity_fraction"]) for point in tail)
    rank_drift_observed = max(
        float(point["relative_rank_drift_from_previous"]) for point in tail[1:]
    )
    final_residual = float(trajectory[-1]["heldout_reconstruction_residual"])
    residual_drift_observed = max(
        float(point["heldout_residual_drift_from_previous"]) for point in tail[1:]
    )

    multi_energy_summary: list[dict[str, Any]] = []
    for retained_energy in energy_targets:
        curve = [
            next(
                energy_point
                for energy_point in trajectory_point["multi_energy_rank_curve"]
                if math.isclose(
                    float(energy_point["retained_energy_target"]),
                    retained_energy,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            for trajectory_point in trajectory
        ]
        curve_tail = curve[-int(tail_points) :]
        energy_capacity_observed = max(
            float(point["capacity_fraction"]) for point in curve_tail
        )
        energy_rank_drift_observed = max(
            float(point["relative_rank_drift_from_previous"])
            for point in curve_tail[1:]
        )
        energy_residual_drift_observed = max(
            float(point["heldout_residual_drift_from_previous"])
            for point in curve_tail[1:]
        )
        relative_rank_drops = [
            max(0.0, float(previous["rank"] - current["rank"]))
            / max(float(previous["rank"]), 1.0)
            for previous, current in pairwise(curve)
        ]
        multi_energy_summary.append(
            {
                "retained_energy_target": retained_energy,
                "ranks": [int(point["rank"]) for point in curve],
                "capacity_fractions": [
                    float(point["capacity_fraction"]) for point in curve
                ],
                "heldout_reconstruction_residuals": [
                    float(point["heldout_reconstruction_residual"]) for point in curve
                ],
                "rank_growth_loglog_slope": _trajectory_power_slope(
                    sizes, [float(point["rank"]) for point in curve]
                ),
                "heldout_residual_loglog_slope": _trajectory_power_slope(
                    sizes,
                    [
                        float(point["heldout_reconstruction_residual"])
                        for point in curve
                    ],
                ),
                "rank_trajectory_monotone_nondecreasing": all(
                    float(current["rank"]) >= float(previous["rank"])
                    for previous, current in pairwise(curve)
                ),
                "max_relative_rank_drop": max(relative_rank_drops, default=0.0),
                "tail_max_capacity_fraction": energy_capacity_observed,
                "tail_max_relative_rank_drift": energy_rank_drift_observed,
                "tail_max_heldout_residual_drift": (energy_residual_drift_observed),
                "capacity_below_primary_gate": math.isfinite(energy_capacity_observed)
                and energy_capacity_observed < float(max_capacity_fraction),
                "rank_converged_by_primary_gate": math.isfinite(
                    energy_rank_drift_observed
                )
                and energy_rank_drift_observed <= float(max_relative_rank_drift),
            }
        )

    full_span_curve = [point["full_empirical_span"] for point in trajectory]
    final_full_span_residual = float(
        full_span_curve[-1]["heldout_reconstruction_residual"]
    )
    capture_span_sufficient = math.isfinite(final_full_span_residual) and (
        final_full_span_residual <= float(max_heldout_residual)
    )
    final_sample_limited = bool(trajectory[-1]["sample_limited"])
    primary_summary = next(
        point
        for point in multi_energy_summary
        if math.isclose(
            float(point["retained_energy_target"]),
            float(1.0 - epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    lower_energy_summaries = [
        point
        for point in multi_energy_summary
        if float(point["retained_energy_target"]) < float(1.0 - epsilon)
    ]
    stable_lower_energy_core = any(
        bool(point["capacity_below_primary_gate"])
        and bool(point["rank_converged_by_primary_gate"])
        for point in lower_energy_summaries
    )
    estimator_or_distribution_instability = any(
        float(point["max_relative_rank_drop"]) > float(max_relative_rank_drift)
        for point in multi_energy_summary
    )
    lowest_energy = multi_energy_summary[0]
    if estimator_or_distribution_instability:
        intrinsic_dimension_status = (
            "estimator_or_distribution_instability_prevents_dimension_inference"
        )
    elif capture_span_sufficient:
        intrinsic_dimension_status = "capture_span_sufficient_for_configured_residual"
    elif final_sample_limited and stable_lower_energy_core:
        intrinsic_dimension_status = "stable_core_with_sample_unresolved_tail"
    elif final_sample_limited:
        intrinsic_dimension_status = (
            "broad_spectrum_consistent_but_intrinsic_dimension_not_identifiable"
        )
    else:
        intrinsic_dimension_status = "broad_feature_spectrum_evidence"
    identifiability = {
        "diagnostic_only_never_overrides_validation_gates": True,
        "status": intrinsic_dimension_status,
        "current_capture_span_sufficient": capture_span_sufficient,
        "final_fit_is_sample_limited": final_sample_limited,
        "final_full_empirical_span_rank": int(full_span_curve[-1]["numerical_rank"]),
        "final_full_empirical_span_holdout_residual": final_full_span_residual,
        "primary_rank_growth_loglog_slope": primary_summary["rank_growth_loglog_slope"],
        "lowest_energy_target": lowest_energy["retained_energy_target"],
        "lowest_energy_rank_growth_loglog_slope": lowest_energy[
            "rank_growth_loglog_slope"
        ],
        "stable_lower_energy_core": stable_lower_energy_core,
        "estimator_or_distribution_instability": (
            estimator_or_distribution_instability
        ),
        "interpretation": (
            "A failed full empirical span proves that the current training "
            "capture cannot represent the configured holdout residual, but "
            "does not by itself distinguish finite high intrinsic dimension, "
            "a long spectral tail, distribution heterogeneity, or noise. "
            "That distinction requires the preregistered curves to be rerun "
            "on a larger independent capture until ranks and held-out "
            "residuals plateau or the sample count exceeds feature dimension."
        ),
    }

    failures: list[dict[str, Any]] = []
    if not math.isfinite(capacity_observed) or capacity_observed >= float(
        max_capacity_fraction
    ):
        failures.append(
            _failure(
                "sample_ceiling_saturation",
                "candidate rank remains too close to the centered empirical "
                "sample ceiling",
                observed=capacity_observed,
                limit={"operator": "<", "value": float(max_capacity_fraction)},
            )
        )
    if not math.isfinite(rank_drift_observed) or rank_drift_observed > float(
        max_relative_rank_drift
    ):
        failures.append(
            _failure(
                "rank_not_converged",
                "candidate rank changes too much over the validation tail",
                observed=rank_drift_observed,
                limit={"operator": "<=", "value": float(max_relative_rank_drift)},
            )
        )
    if not math.isfinite(final_residual) or final_residual > float(
        max_heldout_residual
    ):
        failures.append(
            _failure(
                "heldout_reconstruction_failed",
                "fitted rank does not reconstruct the disjoint holdout within "
                "the configured residual",
                observed=final_residual,
                limit={"operator": "<=", "value": float(max_heldout_residual)},
            )
        )
    if not math.isfinite(residual_drift_observed) or residual_drift_observed > float(
        max_heldout_residual_drift
    ):
        failures.append(
            _failure(
                "heldout_residual_not_converged",
                "held-out reconstruction residual changes too much over the "
                "validation tail",
                observed=residual_drift_observed,
                limit={
                    "operator": "<=",
                    "value": float(max_heldout_residual_drift),
                },
            )
        )

    config = {
        "sample_sizes": sizes,
        "holdout_rows": int(holdout_rows),
        "epsilon": float(epsilon),
        "estimator": name,
        "winsorize_quantile": float(winsorize_quantile),
        "huber_c": float(huber_c),
        "huber_max_iters": int(huber_max_iters),
        "median_of_means_blocks": int(mom_blocks),
        "seed": int(seed),
        "center": bool(center),
        "estimator_fits_location": bool(center or name != "ordinary"),
        "tail_points": int(tail_points),
        "diagnostic_energy_targets": energy_targets,
        "subspace_rank_cap": int(subspace_rank_cap),
        "gates": {
            "interpretation": (
                "campaign-configured operational thresholds requiring "
                "prospective justification"
            ),
            "max_capacity_fraction": float(max_capacity_fraction),
            "max_relative_rank_drift": float(max_relative_rank_drift),
            "max_heldout_residual": float(max_heldout_residual),
            "max_heldout_residual_drift": float(max_heldout_residual_drift),
        },
    }
    largest_train_indices = train_subsets[sizes[-1]]
    if canonical_group_ids is None:
        training_pool_rows = int(matrix.shape[0] - holdout_rows)
    else:
        selected_holdout_groups = torch.unique(
            canonical_group_ids[holdout_indices], sorted=True
        )
        training_pool_rows = int(
            (~torch.isin(canonical_group_ids, selected_holdout_groups)).sum()
        )
    split = {
        "status": (
            "source_group_disjoint"
            if canonical_group_ids is not None
            else "row_disjoint_only"
        ),
        "algorithm": (
            "source_group_permutation_then_nested_row_prefix_v1"
            if canonical_group_ids is not None
            else "torch_randperm_cpu_single_permutation_v1"
        ),
        "split_unit": "source_group" if canonical_group_ids is not None else "row",
        "without_replacement": True,
        "nested_train_prefixes": True,
        "train_holdout_disjoint": True,
        "group_independence_verified": canonical_group_ids is not None,
        "group_disjoint_was_required": bool(require_group_disjoint),
        "total_rows": int(matrix.shape[0]),
        "training_pool_rows": training_pool_rows,
        "holdout_rows": int(holdout_rows),
        "holdout_indices_sha256": _index_sha256(holdout_indices),
        "largest_train_indices_sha256": _index_sha256(largest_train_indices),
    }
    if canonical_group_ids is not None:
        holdout_groups = selected_holdout_groups
        train_groups = torch.unique(
            canonical_group_ids[largest_train_indices], sorted=True
        )
        if bool(torch.isin(holdout_groups, train_groups).any()):
            raise RuntimeError("internal error: source groups cross validation split")
        all_groups = torch.unique(canonical_group_ids, sorted=True)
        split.update(
            {
                "total_group_count": int(all_groups.numel()),
                "holdout_group_count": int(holdout_groups.numel()),
                "largest_train_group_count": int(train_groups.numel()),
                "all_group_ids_sha256": _index_sha256(all_groups),
                "holdout_group_ids_sha256": _index_sha256(holdout_groups),
                "largest_train_group_ids_sha256": _index_sha256(train_groups),
                "unused_rows_due_to_group_isolation": int(
                    matrix.shape[0] - split["training_pool_rows"] - holdout_rows
                ),
            }
        )
    provenance = {
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "input_device": str(matrix.device),
        "torch_version": str(torch.__version__),
        "matrix_sha256": _tensor_sha256(matrix),
        "weight_sha256": None if weight is None else _tensor_sha256(weight),
        "source": dict(source_provenance or {}),
    }
    candidate_rank = int(trajectory[-1]["rank"])
    report: dict[str, Any] = {
        "schema": "dendritic_fmi_rank_validation/v2",
        "passed": not failures,
        "status": "validated" if not failures else "unresolved",
        "validated_rank": candidate_rank if not failures else None,
        "candidate_rank": candidate_rank,
        "failure_reasons": failures,
        "checks": {
            "sample_ceiling": {
                "passed": math.isfinite(capacity_observed)
                and capacity_observed < float(max_capacity_fraction),
                "observed": capacity_observed,
            },
            "rank_convergence": {
                "passed": math.isfinite(rank_drift_observed)
                and rank_drift_observed <= float(max_relative_rank_drift),
                "observed": rank_drift_observed,
            },
            "heldout_reconstruction": {
                "passed": math.isfinite(final_residual)
                and final_residual <= float(max_heldout_residual),
                "observed": final_residual,
            },
            "heldout_residual_convergence": {
                "passed": math.isfinite(residual_drift_observed)
                and residual_drift_observed <= float(max_heldout_residual_drift),
                "observed": residual_drift_observed,
            },
        },
        "resolution_diagnostics": {
            "multi_energy_summary": multi_energy_summary,
            "full_empirical_span": {
                "numerical_ranks": [
                    int(point["numerical_rank"]) for point in full_span_curve
                ],
                "heldout_reconstruction_residuals": [
                    float(point["heldout_reconstruction_residual"])
                    for point in full_span_curve
                ],
                "rank_growth_loglog_slope": _trajectory_power_slope(
                    sizes,
                    [float(point["numerical_rank"]) for point in full_span_curve],
                ),
                "heldout_residual_loglog_slope": _trajectory_power_slope(
                    sizes,
                    [
                        float(point["heldout_reconstruction_residual"])
                        for point in full_span_curve
                    ],
                ),
            },
            "identifiability": identifiability,
        },
        "trajectory": trajectory,
        "validation_tail_sample_sizes": [point["sample_size"] for point in tail],
        "split": split,
        "config": config,
        "provenance": provenance,
    }
    report = _json_safe(report)
    report["validation_sha256"] = _canonical_hash(
        {
            "schema": report["schema"],
            "config": config,
            "split": split,
            "provenance": provenance,
            "trajectory": trajectory,
        }
    )
    return report


def _load_capture_fields(
    path: Path,
    tensor_key: str | None,
    group_key: str | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    group_ids = None
    if isinstance(payload, dict):
        key = tensor_key or "outputs"
        if key not in payload:
            raise KeyError(
                f"tensor key {key!r} not in artifact keys {sorted(payload)}; "
                "pass --tensor-key"
            )
        matrix = payload[key]
        if group_key is not None:
            if group_key not in payload:
                raise KeyError(
                    f"group key {group_key!r} not in artifact keys {sorted(payload)}"
                )
            group_ids = payload[group_key]
    else:
        matrix = payload
        if group_key is not None:
            raise KeyError("a bare tensor capture cannot contain source group IDs")
    if not torch.is_tensor(matrix):
        raise TypeError("captured artifact must be a tensor or a dict holding one")
    matrix = matrix.reshape(-1, matrix.shape[-1])
    if group_ids is not None:
        if not torch.is_tensor(group_ids):
            raise TypeError("captured group IDs must be a tensor")
        group_ids = group_ids.reshape(-1)
        if int(group_ids.numel()) != int(matrix.shape[0]):
            raise ValueError("captured group IDs do not align with captured rows")
    return matrix, group_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--tensor-key", default=None)
    parser.add_argument(
        "--group-key",
        default=None,
        help="Optional artifact key containing one integer source-group ID per "
        "row (new teacher captures use 'row_group_ids').",
    )
    parser.add_argument(
        "--require-group-disjoint",
        action="store_true",
        help="Fail closed unless --group-key is supplied and train/holdout use "
        "disjoint source groups.",
    )
    parser.add_argument("--sample-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-rows", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--estimator", choices=SPECTRUM_ESTIMATORS, default="ordinary")
    parser.add_argument("--winsorize-quantile", type=float, default=0.995)
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument("--huber-max-iters", type=int, default=5)
    parser.add_argument("--mom-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-capacity-fraction", type=float, default=0.8)
    parser.add_argument("--max-relative-rank-drift", type=float, default=0.1)
    parser.add_argument("--max-heldout-residual", type=float, default=0.02)
    parser.add_argument("--max-heldout-residual-drift", type=float, default=0.01)
    parser.add_argument("--tail-points", type=int, default=2)
    parser.add_argument(
        "--diagnostic-energy-targets",
        type=float,
        nargs="+",
        default=list(DEFAULT_DIAGNOSTIC_ENERGY_TARGETS),
        help="Prospectively fixed retained-energy levels for diagnostic rank "
        "and holdout curves. The primary 1-epsilon target is always added; "
        "these diagnostics never relax the primary validation gates.",
    )
    parser.add_argument(
        "--subspace-rank-cap",
        type=int,
        default=64,
        help="Maximum leading rank used for adjacent-fit principal-angle "
        "diagnostics; fitted ranks and any truncation are recorded.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    matrix, group_ids = _load_capture_fields(
        args.captured, args.tensor_key, args.group_key
    )
    matrix = matrix.to(args.device)
    report = validate_rank_estimate(
        matrix,
        sample_sizes=args.sample_sizes,
        holdout_rows=args.holdout_rows,
        epsilon=args.epsilon,
        estimator=args.estimator,
        winsorize_quantile=args.winsorize_quantile,
        huber_c=args.huber_c,
        huber_max_iters=args.huber_max_iters,
        mom_blocks=args.mom_blocks,
        seed=args.seed,
        max_capacity_fraction=args.max_capacity_fraction,
        max_relative_rank_drift=args.max_relative_rank_drift,
        max_heldout_residual=args.max_heldout_residual,
        max_heldout_residual_drift=args.max_heldout_residual_drift,
        tail_points=args.tail_points,
        diagnostic_energy_targets=args.diagnostic_energy_targets,
        subspace_rank_cap=args.subspace_rank_cap,
        group_ids=group_ids,
        require_group_disjoint=args.require_group_disjoint,
        source_provenance={
            "captured_path": str(args.captured.resolve()),
            "tensor_key": args.tensor_key or "outputs",
            "group_key": args.group_key,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{report['status']}: candidate_rank={report['candidate_rank']}, "
        f"validated_rank={report['validated_rank']}"
    )
    for failure in report["failure_reasons"]:
        print(f"FAIL {failure['code']}: {failure['message']}")
    print(f"wrote {args.output}")
    return report


if __name__ == "__main__":
    main()
