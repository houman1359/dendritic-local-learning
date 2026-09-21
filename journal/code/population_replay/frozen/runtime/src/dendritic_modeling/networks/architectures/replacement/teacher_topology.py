"""Teacher-conditioned initialization for compiled PopulationNetwork cells.

The older exact TopK initializer copies a dense SwiGLU teacher into a matching
flat signed cell.  A general PopulationNetwork may have a different width,
explicit E/I populations, multiple dendritic levels, and a sparse readout, so
there is no row-wise weight-copy correspondence.  This module instead uses a
calibration set and the teacher target to initialize that richer graph:

1. fit the replacement readout to its current core features;
2. backpropagate teacher-reconstruction loss through the complete cell;
3. rank every indexed synapse's source features either with the historical
   module-global attribution or with an output-row-conditioned first-order
   score;
4. reselect its legal support, including internal E/I and dendritic levels;
5. refit the sparse (or positive E-minus-I) readout.

This is an initializer, not an assertion that the selected support is optimal.
Adaptive rewiring or developmental pruning remains responsible for learning
the final deployable topology.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.replacement.projections import (
    PositiveEIOutputProjection,
    TransformedLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import inverse_weight_transform

LEGACY_GLOBAL_TEACHER_TOPOLOGY_METRICS = (
    "gradient",
    "activation_weighted",
    "activation_weighted_robust",
    "activation_weighted_structured",
)
ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS = (
    "row_taylor",
    "row_empirical_fisher",
    "row_signed_covariance",
)
TEACHER_TOPOLOGY_METRICS = (
    *LEGACY_GLOBAL_TEACHER_TOPOLOGY_METRICS,
    *ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS,
)

ROBUST_SALIENCY_QUANTILE = 0.95
DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE = 128

_SCORE_DEFINITIONS = {
    "gradient": "mean absolute gradient with respect to the module input",
    "activation_weighted": (
        "mean absolute input times gradient with respect to the module input"
    ),
    "activation_weighted_robust": (
        "winsorized mean absolute input times gradient with respect to the module input"
    ),
    "activation_weighted_structured": (
        "mean absolute input times gradient with respect to the module input; "
        "legal row groups and column blocks are selected jointly"
    ),
    "row_taylor": (
        "per-output-row mean absolute first-order edge signal E[|dL/dy_o * x_i|]"
    ),
    "row_empirical_fisher": (
        "per-output-row mean squared reconstruction-loss edge gradient "
        "E[(dL_recon/dy_o * x_i)^2] (empirical-Fisher-style diagonal)"
    ),
    "row_signed_covariance": (
        "absolute per-output-row sample covariance |Cov(dL/dy_o, x_i)|"
    ),
}


def _activation_weighted_saliency(
    values: torch.Tensor,
    gradients: torch.Tensor,
    *,
    robust: bool,
) -> torch.Tensor:
    """Aggregate per-example attribution with optional winsorization.

    Standard activation-weighted saliency estimates ``E[|x * grad|]``.  Its
    empirical mean can be dominated by a very small number of extreme
    calibration examples.  The robust control caps each feature at its own
    registered 95th percentile before averaging; it therefore changes only
    the support-selection statistic, not the teacher targets or learned
    synapse values.
    """

    contributions = (values * gradients).abs()
    if robust and contributions.shape[0] > 1:
        cap = torch.quantile(
            contributions,
            q=ROBUST_SALIENCY_QUANTILE,
            dim=0,
        )
        contributions = torch.minimum(contributions, cap.unsqueeze(0))
    return contributions.mean(dim=0)


def _row_conditioned_saliency(
    values: torch.Tensor,
    output_gradients: torch.Tensor,
    *,
    metric: str,
    row_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Score every prospective ``output <- input`` edge in bounded FP32 chunks.

    The output-gradient factor is the derivative with respect to this sparse
    module's own output, not the derivative with respect to its shared input.
    Consequently two somas that consume the same population can receive
    different supports.  Only one output-row chunk of the dense score matrix
    resides on the accelerator at a time; completed chunks are copied to CPU.

    ``row_taylor`` is the mean absolute first-order edge signal.
    ``row_empirical_fisher`` is its mean square: an empirical-Fisher-style
    diagonal score for the explicitly stated reconstruction loss, not a claim
    about the model likelihood's Fisher information. ``row_signed_covariance``
    centers both factors, selects by covariance magnitude, and records the
    unrectified covariance sign in diagnostics.
    """

    if metric not in ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS:
        raise ValueError(
            "row-conditioned metric must be one of "
            f"{ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS}, got {metric!r}"
        )
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, Integral)
        or int(row_chunk_size) < 1
    ):
        raise ValueError("row_chunk_size must be a positive integer")
    x = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float32)
    grad = output_gradients.detach().reshape(-1, output_gradients.shape[-1])
    if x.shape[0] != grad.shape[0]:
        raise ValueError(
            "module inputs and output gradients must have the same number of "
            f"examples, got {x.shape[0]} and {grad.shape[0]}"
        )
    if x.shape[0] < 1:
        raise ValueError("row-conditioned saliency needs at least one example")

    examples = int(x.shape[0])
    out_features = int(grad.shape[1])
    in_features = int(x.shape[1])
    scores = torch.empty(out_features, in_features, dtype=torch.float32)
    signed_positive = 0
    signed_negative = 0
    signed_zero = 0

    if metric == "row_taylor":
        right = x.abs()
    elif metric == "row_empirical_fisher":
        right = x.square()
    else:
        right = x - x.mean(dim=0, keepdim=True)

    for start in range(0, out_features, int(row_chunk_size)):
        end = min(start + int(row_chunk_size), out_features)
        # Convert only this output-row slice to FP32. A full [examples, out]
        # FP32 copy would defeat chunking for wide transformer populations.
        grad_chunk = grad[:, start:end].to(dtype=torch.float32)
        if metric == "row_taylor":
            chunk = grad_chunk.abs().T @ right / float(examples)
        elif metric == "row_empirical_fisher":
            chunk = grad_chunk.square().T @ right / float(examples)
        else:
            centered_grad = grad_chunk - grad_chunk.mean(dim=0, keepdim=True)
            signed = centered_grad.T @ right / float(max(1, examples - 1))
            signed_positive += int((signed > 0).sum())
            signed_negative += int((signed < 0).sum())
            signed_zero += int((signed == 0).sum())
            chunk = signed.abs()
        scores[start:end].copy_(chunk.detach().cpu())

    diagnostics: dict[str, Any] = {
        "score_scope": "output_row_conditioned",
        "score_definition": _SCORE_DEFINITIONS[metric],
        "score_shape": [out_features, in_features],
        "score_compute_dtype": "float32",
        "score_storage_device": "cpu",
        "output_gradient_fp32_chunked": True,
        "row_chunk_size": int(row_chunk_size),
        "examples": examples,
    }
    if metric == "row_signed_covariance":
        total = max(1, signed_positive + signed_negative + signed_zero)
        diagnostics["signed_covariance"] = {
            "selection_transform": "absolute_value",
            "positive_fraction": signed_positive / total,
            "negative_fraction": signed_negative / total,
            "zero_fraction": signed_zero / total,
        }
    return scores, diagnostics


def _flatten_examples(inputs: torch.Tensor, *, max_rows: int) -> torch.Tensor:
    flat = inputs.detach().reshape(-1, inputs.shape[-1])
    if flat.shape[0] > int(max_rows):
        flat = flat[: int(max_rows)]
    return flat


def _replacement_features(replacement: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return the tensor consumed by a compiled cell's output projection."""

    if hasattr(replacement, "gate_core") and hasattr(replacement, "value_core"):
        normalized = replacement.pre_norm(inputs)
        adapted = replacement.input_adapter(normalized)
        gate = replacement._run_population_core(replacement.gate_core, adapted)
        value = replacement._run_population_core(replacement.value_core, adapted)
        return replacement._activate_gate(gate) * value
    encoded = replacement.encode_input(inputs)
    return replacement.run_core(encoded)


def _ridge_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    """Fit ``targets ~= features @ coefficient.T`` with an exact ridge solve.

    The smaller of the primal and sample-space systems is used on the input
    device.  This avoids the historical unconditional CPU ``d x d`` solve and
    never silently changes the estimator to unregularized least squares.
    """

    x = features.detach().reshape(-1, features.shape[-1]).to(dtype=torch.float32)
    y = (
        targets.detach()
        .reshape(-1, targets.shape[-1])
        .to(device=x.device, dtype=torch.float32)
    )
    if x.shape[0] != y.shape[0]:
        raise ValueError("ridge features and targets must have the same row count")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("ridge features and targets must be finite")
    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("ridge must be finite and strictly positive")
    rows, dimension = int(x.shape[0]), int(x.shape[1])
    scale = x.square().sum() / max(1, dimension)
    if float(scale) == 0.0:
        return torch.zeros(
            y.shape[1],
            dimension,
            device=x.device,
            dtype=torch.float32,
        )
    absolute_ridge = float(ridge) * scale
    if rows <= dimension:
        system = x @ x.T
        system.diagonal().add_(absolute_ridge)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if bool((info != 0).any()):
            raise torch.linalg.LinAlgError(
                "sample-space ridge Cholesky failed; refusing an estimator fallback"
            )
        solution = x.T @ torch.cholesky_solve(y, factor)
    else:
        system = x.T @ x
        system.diagonal().add_(absolute_ridge)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if bool((info != 0).any()):
            raise torch.linalg.LinAlgError(
                "primal ridge Cholesky failed; refusing an estimator fallback"
            )
        solution = torch.cholesky_solve(x.T @ y, factor)
    return solution.T.contiguous()


def _legal_support(target: IndexedSparseLinear, scores: torch.Tensor) -> torch.Tensor:
    """Select legal supports from global or output-row-conditioned scores.

    ``scores`` may be ``[in_features]`` (the historical module-global modes)
    or ``[out_features, in_features]``. Consecutive output rows configured to
    share support are optimized jointly by their mean score. Column-block
    structure is selected as whole contiguous blocks. Any incompatible shape,
    mask, or divisibility condition raises instead of silently degrading to an
    unstructured topology.
    """

    base = scores.detach().float().cpu()
    if base.ndim == 1:
        if base.numel() != int(target.in_features):
            raise ValueError(
                f"saliency has {base.numel()} inputs, expected {target.in_features}"
            )
        score_rows = base.unsqueeze(0).expand(target.out_features, -1)
        score_scope = "module_global"
    elif base.ndim == 2:
        expected = (int(target.out_features), int(target.in_features))
        if tuple(base.shape) != expected:
            raise ValueError(
                f"row-conditioned saliency has shape {tuple(base.shape)}, "
                f"expected {expected}"
            )
        score_rows = base
        score_scope = "output_row_conditioned"
    else:
        raise ValueError(
            "saliency must have shape [in_features] or [out_features, in_features]"
        )
    if not bool(torch.isfinite(score_rows).all()):
        raise ValueError(f"{score_scope} saliency contains non-finite values")

    allowed = getattr(target, "_allowed_connection_mask", None)
    if allowed is None or allowed.numel() == 0:
        allowed = torch.ones(
            target.out_features,
            target.in_features,
            dtype=torch.bool,
        )
    else:
        allowed = allowed.detach().bool().cpu().clone()
    forbidden = getattr(target, "_forbidden_input_index_per_output", None)
    if forbidden is not None and forbidden.numel():
        forbidden = forbidden.detach().long().cpu()
        rows = torch.arange(target.out_features)
        valid = forbidden >= 0
        allowed[rows[valid], forbidden[valid]] = False

    group_rows = int(getattr(target, "support_group_rows", 1))
    col_block = int(getattr(target, "support_col_block", 1))
    if group_rows < 1 or col_block < 1:
        raise ValueError(
            "support_group_rows and support_col_block must be >= 1, got "
            f"{group_rows} and {col_block}"
        )
    if int(target.K) % col_block:
        raise ValueError(
            f"K={target.K} must be divisible by support_col_block={col_block}"
        )
    if int(target.in_features) % col_block:
        raise ValueError(
            f"in_features={target.in_features} must be divisible by "
            f"support_col_block={col_block}"
        )

    def stable_topk(values: torch.Tensor, *, k: int) -> torch.Tensor:
        # Stable descending sort makes tie behavior deterministic and favors
        # the lower feature/block index, which keeps initialization auditable.
        return torch.argsort(values, descending=True, stable=True)[:k]

    selected = torch.empty(target.out_features, target.K, dtype=torch.long)
    for start in range(0, target.out_features, group_rows):
        end = min(start + group_rows, target.out_features)
        group_allowed = allowed[start:end].all(dim=0)
        group_scores = score_rows[start:end].mean(dim=0)
        if col_block == 1:
            if int(group_allowed.sum()) < int(target.K):
                raise ValueError(
                    f"output-row group [{start}, {end}) has only "
                    f"{int(group_allowed.sum())} jointly legal inputs for "
                    f"K={target.K}; refusing to violate support_group_rows"
                )
            masked = group_scores.masked_fill(~group_allowed, -torch.inf)
            indices = stable_topk(masked, k=int(target.K)).sort().values
        else:
            blocks = int(target.in_features) // col_block
            block_allowed = group_allowed.reshape(blocks, col_block).all(dim=1)
            blocks_needed = int(target.K) // col_block
            if int(block_allowed.sum()) < blocks_needed:
                raise ValueError(
                    f"output-row group [{start}, {end}) has only "
                    f"{int(block_allowed.sum())} jointly legal column blocks "
                    f"but needs {blocks_needed}; refusing to violate "
                    "support_col_block"
                )
            block_scores = group_scores.reshape(blocks, col_block).mean(dim=1)
            block_scores = block_scores.masked_fill(~block_allowed, -torch.inf)
            chosen_blocks = stable_topk(block_scores, k=blocks_needed)
            offsets = torch.arange(col_block, dtype=torch.long)
            indices = (
                (chosen_blocks[:, None] * col_block + offsets[None, :])
                .reshape(-1)
                .sort()
                .values
            )
        selected[start:end] = indices
    return selected


def _set_support_(target: IndexedSparseLinear, scores: torch.Tensor) -> torch.Tensor:
    indices = _legal_support(target, scores)
    with torch.no_grad():
        target.connection_indices.copy_(
            indices.to(
                device=target.connection_indices.device,
                dtype=target.connection_indices.dtype,
            )
        )
    return indices


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU-normalized tensor payload for initialization provenance."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _copy_effective_indexed_(
    target: IndexedSparseLinear,
    effective_weight: torch.Tensor,
) -> float:
    """Copy a dense effective readout into signed or positive indexed storage."""

    dense = effective_weight.detach().float().cpu()
    if tuple(dense.shape) != (target.out_features, target.in_features):
        raise ValueError("readout coefficient shape does not match indexed projection")
    indices = torch.topk(dense.abs(), k=target.K, dim=1).indices.sort(dim=1).values
    values = dense.gather(1, indices)
    transform = str(target.weight_transform).lower()
    if transform == "identity":
        raw = values
    else:
        if bool((values < 0).any()):
            raise ValueError("positive indexed readout received negative weights")
        raw = inverse_weight_transform(values.clamp_min(1e-8), transform)
    with torch.no_grad():
        target.connection_indices.copy_(
            indices.to(
                device=target.connection_indices.device,
                dtype=target.connection_indices.dtype,
            )
        )
        target.pre_w.copy_(raw.to(device=target.pre_w.device, dtype=target.pre_w.dtype))
    retained = values.square().sum() / dense.square().sum().clamp_min(1e-30)
    return float(retained)


def _low_rank_layers(projection: nn.Sequential) -> tuple[nn.Module, nn.Module]:
    """Validate the canonical two-factor projection and fail closed otherwise."""

    if len(projection) != 2:
        raise ValueError(
            "teacher-conditioned low-rank initialization requires exactly two "
            "linear factors"
        )
    first, second = projection[0], projection[1]
    supported = (nn.Linear, TransformedLinear)
    if not isinstance(first, supported) or not isinstance(second, supported):
        raise ValueError(
            "teacher-conditioned low-rank initialization supports only canonical "
            "Linear or TransformedLinear factors"
        )
    first_weight = first.weight
    second_weight = second.weight
    if (
        first_weight.ndim != 2
        or second_weight.ndim != 2
        or first_weight.shape[0] != second_weight.shape[1]
    ):
        raise ValueError("low-rank factor shapes are incompatible")
    if first.bias is not None:
        raise ValueError("the first low-rank factor must not have a bias")
    return first, second


def _direct_reduced_rank_ridge_factors(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    rank: int,
    ridge: float,
    oversample: int = 8,
    max_iterations: int = 16,
    residual_tolerance: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Solve rank-constrained ridge without materializing a dense coefficient.

    For ``A = X.T X + lambda I``, reduced-rank ridge uses the leading output
    eigenspace of ``Y.T X A^-1 X.T Y``. A content-seeded block subspace
    iteration applies that operator through the smaller Cholesky system in
    sample or feature space.  The returned factors satisfy
    ``prediction = (X @ right.T) @ left.T``.

    This directly optimizes the rank-constrained ridge objective.  It is not
    the generally different operation of fitting a full-rank coefficient and
    truncating it by coefficient Frobenius norm.
    """

    x = features.detach().reshape(-1, features.shape[-1]).to(dtype=torch.float32)
    y = (
        targets.detach()
        .reshape(-1, targets.shape[-1])
        .to(device=x.device, dtype=torch.float32)
    )
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError(
            "reduced-rank ridge requires at least two aligned feature/target rows"
        )
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("reduced-rank ridge features and targets must be finite")
    requested_rank = int(rank)
    rows, dimension = int(x.shape[0]), int(x.shape[1])
    output_dimension = int(y.shape[1])
    identifiable_rank = min(rows, dimension, output_dimension)
    if requested_rank < 1 or requested_rank > min(dimension, output_dimension):
        raise ValueError("reduced-rank ridge rank exceeds coefficient dimensions")
    if requested_rank > identifiable_rank:
        raise ValueError(
            f"requested rank {requested_rank} exceeds calibration-identifiable "
            f"rank {identifiable_rank}"
        )
    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("ridge must be finite and strictly positive")
    if int(oversample) < 0 or int(max_iterations) < 2:
        raise ValueError("oversample must be nonnegative and max_iterations >= 2")
    if not 0.0 < float(residual_tolerance) < 1.0:
        raise ValueError("residual_tolerance must lie in (0, 1)")

    feature_sha256 = _tensor_sha256(x)
    target_sha256 = _tensor_sha256(y)
    scale = float(x.square().sum() / max(1, dimension))
    absolute_ridge = float(ridge) * scale
    factor_shape = {
        "left": [output_dimension, requested_rank],
        "right": [requested_rank, dimension],
    }
    base_diagnostics: dict[str, Any] = {
        "fit_schema": "direct_reduced_rank_ridge/v1",
        "algorithm": "content_seeded_block_subspace_reduced_rank_ridge",
        "objective": ("minimize ||Y-XB||_F^2 + lambda||B||_F^2 subject to rank(B)<=r"),
        "rows": rows,
        "input_dimension": dimension,
        "output_dimension": output_dimension,
        "rank": requested_rank,
        "factor_shapes": factor_shape,
        "relative_ridge": float(ridge),
        "absolute_ridge": absolute_ridge,
        "ridge_scale": "mean_diagonal_XtX_equals_squared_frobenius_X_over_d",
        "compute_dtype": "float32",
        "compute_device": str(x.device),
        "feature_sha256": feature_sha256,
        "target_sha256": target_sha256,
        "dense_coefficient_materialized": False,
        "oversample": int(oversample),
        "max_subspace_iterations": int(max_iterations),
        "subspace_residual_tolerance": float(residual_tolerance),
        "determinism_scope": (
            "content-derived initialization seed with fixed operation order; "
            "factor hashes certify the realized software/device result"
        ),
    }
    if scale == 0.0:
        left = torch.zeros(
            output_dimension,
            requested_rank,
            device=x.device,
            dtype=torch.float32,
        )
        right = torch.zeros(
            requested_rank,
            dimension,
            device=x.device,
            dtype=torch.float32,
        )
        return (
            left,
            right,
            {
                **base_diagnostics,
                "solver_space": "degenerate_zero_features",
                "largest_square_system_dimension": 0,
                "subspace_iterations": 0,
                "operator_calls": 0,
                "converged": True,
                "relative_eigenspace_residual": 0.0,
                "calibration_mse": float(y.square().mean()),
                "calibration_relative_mse": (
                    1.0 if float(y.square().mean()) > 0 else 0.0
                ),
                "regularized_objective": float(y.square().sum()),
            },
        )

    if rows <= dimension:
        solver_space = "sample"
        system = x @ x.T
        system.diagonal().add_(absolute_ridge)
        cholesky, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if bool((info != 0).any()):
            raise torch.linalg.LinAlgError(
                "sample-space reduced-rank ridge Cholesky failed"
            )

        def projected(values: torch.Tensor) -> torch.Tensor:
            return values - absolute_ridge * torch.cholesky_solve(values, cholesky)

        def latent_coefficient(values: torch.Tensor) -> torch.Tensor:
            return x.T @ torch.cholesky_solve(values, cholesky)

        system_dimension = rows
    else:
        solver_space = "primal"
        system = x.T @ x
        system.diagonal().add_(absolute_ridge)
        cholesky, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if bool((info != 0).any()):
            raise torch.linalg.LinAlgError("primal reduced-rank ridge Cholesky failed")

        def projected(values: torch.Tensor) -> torch.Tensor:
            latent = torch.cholesky_solve(x.T @ values, cholesky)
            return x @ latent

        def latent_coefficient(values: torch.Tensor) -> torch.Tensor:
            return torch.cholesky_solve(x.T @ values, cholesky)

        system_dimension = dimension

    def output_operator(basis: torch.Tensor) -> torch.Tensor:
        return y.T @ projected(y @ basis)

    subspace_dimension = min(
        identifiable_rank,
        requested_rank + int(oversample),
    )
    seed_material = hashlib.sha256(
        (
            f"{feature_sha256}:{target_sha256}:{requested_rank}:{subspace_dimension}"
        ).encode("ascii")
    ).digest()
    subspace_seed = int.from_bytes(seed_material[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(subspace_seed)
    basis = torch.randn(
        output_dimension,
        subspace_dimension,
        generator=generator,
        dtype=torch.float32,
    ).to(x.device)
    basis = torch.linalg.qr(basis, mode="reduced").Q
    previous_selected: torch.Tensor | None = None
    relative_residual = math.inf
    principal_sine = None
    selected_values = torch.zeros(requested_rank, device=x.device)
    selected = basis[:, :requested_rank]
    operator_calls = 0
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        image = output_operator(basis)
        operator_calls += 1
        if float(image.square().sum()) <= torch.finfo(torch.float32).tiny:
            selected = torch.zeros_like(selected)
            selected_values.zero_()
            relative_residual = 0.0
            converged = True
            break
        basis = torch.linalg.qr(image, mode="reduced").Q
        image = output_operator(basis)
        operator_calls += 1
        rayleigh = (basis.T @ image + image.T @ basis) * 0.5
        eigenvalues, rotation = torch.linalg.eigh(rayleigh)
        order = torch.argsort(eigenvalues, descending=True, stable=True)
        rotation = rotation[:, order]
        basis = basis @ rotation
        image = image @ rotation
        selected = basis[:, :requested_rank]
        selected_image = image[:, :requested_rank]
        selected_values = eigenvalues[order[:requested_rank]].clamp_min(0.0)
        residual = selected_image - selected * selected_values.unsqueeze(0)
        denominator = torch.linalg.vector_norm(selected_image).clamp_min(
            torch.finfo(torch.float32).eps
        )
        relative_residual = float(torch.linalg.vector_norm(residual) / denominator)
        if previous_selected is not None:
            singular_values = torch.linalg.svdvals(
                previous_selected.T @ selected
            ).clamp(0.0, 1.0)
            principal_sine = float(
                (1.0 - singular_values.square()).clamp_min(0.0).mean().sqrt()
            )
        previous_selected = selected.detach().clone()
        if iterations >= 2 and relative_residual <= float(residual_tolerance):
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "direct reduced-rank ridge output subspace did not converge: "
            f"relative residual {relative_residual:.6g} exceeds "
            f"{float(residual_tolerance):.6g} after {iterations} iterations"
        )

    latent = latent_coefficient(y @ selected)
    left = selected.contiguous()
    right = latent.T.contiguous()
    prediction = (x @ latent) @ left.T
    residual_sum = (y - prediction).square().sum()
    target_energy = y.square().sum()
    coefficient_penalty = latent.square().sum()
    calibration_mse = float((y - prediction).square().mean())
    calibration_relative_mse = float(residual_sum / target_energy.clamp_min(1e-30))
    regularized_objective = float(residual_sum + absolute_ridge * coefficient_penalty)
    diagnostics = {
        **base_diagnostics,
        "solver_space": solver_space,
        "largest_square_system_dimension": int(system_dimension),
        "cholesky_info_max": int(info.max()),
        "subspace_dimension": subspace_dimension,
        "subspace_seed": subspace_seed,
        "subspace_iterations": iterations,
        "operator_calls": operator_calls,
        "converged": converged,
        "relative_eigenspace_residual": relative_residual,
        "adjacent_leading_subspace_rms_sine": principal_sine,
        "selected_eigenvalues": [float(value) for value in selected_values],
        "calibration_mse": calibration_mse,
        "calibration_relative_mse": calibration_relative_mse,
        "regularized_objective": regularized_objective,
        "left_factor_sha256": _tensor_sha256(left),
        "right_factor_sha256": _tensor_sha256(right),
    }
    return left, right, diagnostics


def _deterministic_low_rank_factors(
    coefficient: torch.Tensor,
    *,
    rank: int,
    subspace_iterations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute a deterministic truncated factorization without randomized SVD."""

    matrix = coefficient.detach().to(dtype=torch.float32)
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("low-rank coefficient must be a finite matrix")
    if not 1 <= int(rank) <= min(matrix.shape):
        raise ValueError("low-rank factor rank exceeds the coefficient dimensions")
    column_energy = matrix.square().sum(dim=0)
    columns = torch.argsort(
        column_energy,
        descending=True,
        stable=True,
    )[: int(rank)]
    basis, _ = torch.linalg.qr(matrix.index_select(1, columns), mode="reduced")
    for _ in range(int(subspace_iterations)):
        right_subspace = matrix.T @ basis
        basis, _ = torch.linalg.qr(matrix @ right_subspace, mode="reduced")
    right = basis.T @ matrix
    total_energy = matrix.square().sum().clamp_min(1e-30)
    retained = (right.square().sum() / total_energy).clamp(0.0, 1.0)
    relative_squared_error = float((1.0 - retained).clamp_min(0.0))
    return basis, right, relative_squared_error


def _copy_signed_low_rank_(
    projection: nn.Sequential,
    coefficient: torch.Tensor,
) -> dict[str, Any]:
    first, second = _low_rank_layers(projection)
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise ValueError(
            "an unrestricted low-rank readout must use signed Linear factors"
        )
    expected = (second.out_features, first.in_features)
    if tuple(coefficient.shape) != expected:
        raise ValueError(
            f"low-rank coefficient shape {tuple(coefficient.shape)} != {expected}"
        )
    rank = int(first.out_features)
    left, right, relative_squared_error = _deterministic_low_rank_factors(
        coefficient,
        rank=rank,
    )
    with torch.no_grad():
        first.weight.copy_(
            right.to(device=first.weight.device, dtype=first.weight.dtype)
        )
        second.weight.copy_(
            left.to(device=second.weight.device, dtype=second.weight.dtype)
        )
        if second.bias is not None:
            second.bias.zero_()
    return {
        "readout_initialized": True,
        "readout_mode": "signed_low_rank_coefficient_factorization",
        "factorization_schema": "low_rank_coefficient_factorization/v1",
        "factorization_algorithm": "top_energy_subspace_iteration_qr",
        "factorization_rank": rank,
        "subspace_iterations": 2,
        "factorization_compute_dtype": "float32",
        "factorization_compute_device": str(coefficient.device),
        "coefficient_sha256": _tensor_sha256(coefficient),
        "coefficient_fit_error_scope": (
            "supplied coefficient versus installed two-factor effective weight"
        ),
        "coefficient_relative_squared_frobenius_error": relative_squared_error,
        "claim_boundary": (
            "This utility approximately factorizes a supplied coefficient; it is "
            "not the canonical direct reduced-rank ridge example fit."
        ),
    }


def _copy_signed_low_rank_factors_(
    projection: nn.Sequential,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    fit_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Install direct reduced-rank factors into an unrestricted projection."""

    first, second = _low_rank_layers(projection)
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise ValueError(
            "an unrestricted direct reduced-rank readout needs signed Linear factors"
        )
    expected_left = (second.out_features, first.out_features)
    expected_right = (first.out_features, first.in_features)
    if tuple(left.shape) != expected_left or tuple(right.shape) != expected_right:
        raise ValueError(
            "direct reduced-rank factor shapes do not match the projection: "
            f"left={tuple(left.shape)} versus {expected_left}, "
            f"right={tuple(right.shape)} versus {expected_right}"
        )
    with torch.no_grad():
        first.weight.copy_(
            right.to(device=first.weight.device, dtype=first.weight.dtype)
        )
        second.weight.copy_(
            left.to(device=second.weight.device, dtype=second.weight.dtype)
        )
        if second.bias is not None:
            second.bias.zero_()
    return {
        "readout_initialized": True,
        "readout_mode": "signed_direct_reduced_rank_ridge",
        "factorization_rank": int(first.out_features),
        "fit": dict(fit_diagnostics),
    }


def _copy_positive_path_factors_(
    projection: nn.Sequential,
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    """Install one explicitly nonnegative low-rank pathway."""

    first, second = _low_rank_layers(projection)
    if not isinstance(first, TransformedLinear) or not isinstance(
        second, TransformedLinear
    ):
        raise ValueError(
            "a biological direct low-rank readout needs TransformedLinear factors"
        )
    expected_left = (second.out_features, first.out_features)
    expected_right = (first.out_features, first.in_features)
    if tuple(left.shape) != expected_left or tuple(right.shape) != expected_right:
        raise ValueError(
            "positive pathway factor shapes do not match the projection: "
            f"left={tuple(left.shape)} versus {expected_left}, "
            f"right={tuple(right.shape)} versus {expected_right}"
        )
    if bool((left < 0).any()) or bool((right < 0).any()):
        raise ValueError("positive pathway factors must be nonnegative")
    floor = 1e-8
    with torch.no_grad():
        first.pre_w.copy_(
            inverse_weight_transform(right.clamp_min(floor), first.weight_transform).to(
                device=first.pre_w.device, dtype=first.pre_w.dtype
            )
        )
        second.pre_w.copy_(
            inverse_weight_transform(left.clamp_min(floor), second.weight_transform).to(
                device=second.pre_w.device, dtype=second.pre_w.dtype
            )
        )
        if second.pre_bias is not None:
            second.pre_bias.copy_(
                inverse_weight_transform(
                    torch.full_like(second.pre_bias, floor),
                    second.weight_transform,
                )
            )


def _copy_biological_low_rank_seed_(
    projection: PositiveEIOutputProjection,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    fit_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Algebraically split a signed low-rank seed into positive E/I pathways.

    One signed outer product needs two nonnegative components in each pathway.
    Therefore a configured pathway rank ``r`` receives a directly fitted
    signed seed of rank ``floor(r/2)``.  Recovery may subsequently use all
    ``r`` components in each path, whose signed effective map can have
    algebraic rank up to ``2r``.
    """

    if not isinstance(projection.excitatory, nn.Sequential) or not isinstance(
        projection.inhibitory, nn.Sequential
    ):
        raise ValueError(
            "biological direct reduced-rank initialization requires matched "
            "Sequential E/I pathways"
        )
    e_first, _ = _low_rank_layers(projection.excitatory)
    i_first, _ = _low_rank_layers(projection.inhibitory)
    pathway_rank = int(e_first.weight.shape[0])
    if int(i_first.weight.shape[0]) != pathway_rank:
        raise ValueError("biological E/I low-rank pathways must have equal rank")
    signed_seed_rank = int(left.shape[1])
    if signed_seed_rank < 1 or 2 * signed_seed_rank > pathway_rank:
        raise ValueError(
            "signed biological seed needs two nonnegative components per pathway "
            f"component, got signed rank {signed_seed_rank} and pathway rank "
            f"{pathway_rank}"
        )
    left_positive = left.clamp_min(0.0)
    left_negative = (-left).clamp_min(0.0)
    right_positive = right.clamp_min(0.0)
    right_negative = (-right).clamp_min(0.0)
    excitatory_left = torch.cat([left_positive, left_negative], dim=1)
    excitatory_right = torch.cat([right_positive, right_negative], dim=0)
    inhibitory_left = torch.cat([left_positive, left_negative], dim=1)
    inhibitory_right = torch.cat([right_negative, right_positive], dim=0)
    padding = pathway_rank - 2 * signed_seed_rank
    if padding:
        excitatory_left = F.pad(excitatory_left, (0, padding))
        inhibitory_left = F.pad(inhibitory_left, (0, padding))
        excitatory_right = F.pad(excitatory_right, (0, 0, 0, padding))
        inhibitory_right = F.pad(inhibitory_right, (0, 0, 0, padding))
    _copy_positive_path_factors_(
        projection.excitatory,
        excitatory_left,
        excitatory_right,
    )
    _copy_positive_path_factors_(
        projection.inhibitory,
        inhibitory_left,
        inhibitory_right,
    )
    return {
        "readout_initialized": True,
        "readout_mode": "positive_e_minus_i_direct_reduced_rank_ridge_seed",
        "rank_semantics": "rank_per_e_and_i_pathway",
        "pathway_factor_rank": pathway_rank,
        "signed_seed_rank": signed_seed_rank,
        "effective_signed_rank_upper_bound_after_training": 2 * pathway_rank,
        "seed_decomposition": (
            "before the positive-storage numerical floor, each signed outer "
            "product is exactly decomposed into two nonnegative excitatory and "
            "two nonnegative inhibitory components"
        ),
        "positive_storage_floor": 1e-8,
        "fit": dict(fit_diagnostics),
    }


def _copy_positive_low_rank_(
    projection: nn.Sequential,
    coefficient: torch.Tensor,
    *,
    nmf_iterations: int = 12,
) -> dict[str, Any]:
    """Fit nonnegative factors for one biological E or I pathway."""

    first, second = _low_rank_layers(projection)
    if not isinstance(first, TransformedLinear) or not isinstance(
        second, TransformedLinear
    ):
        raise ValueError(
            "a biological low-rank readout must use TransformedLinear factors"
        )
    expected = (second.out_features, first.in_features)
    if tuple(coefficient.shape) != expected:
        raise ValueError(
            f"positive low-rank coefficient shape {tuple(coefficient.shape)} "
            f"!= {expected}"
        )
    target = coefficient.detach().to(dtype=torch.float32)
    if bool((target < 0).any()) or not bool(torch.isfinite(target).all()):
        raise ValueError("positive low-rank coefficient must be finite and nonnegative")
    rank = int(first.out_features)
    left_seed, right_seed, _ = _deterministic_low_rank_factors(target, rank=rank)
    epsilon = torch.finfo(torch.float32).eps
    if float(target.square().sum()) == 0.0:
        left = torch.zeros_like(left_seed)
        right = torch.zeros_like(right_seed)
    else:
        left = left_seed.abs().clamp_min(epsilon)
        right = right_seed.abs().clamp_min(epsilon)
        for _ in range(int(nmf_iterations)):
            right.mul_((left.T @ target) / ((left.T @ left) @ right).clamp_min(epsilon))
            left.mul_(
                (target @ right.T) / (left @ (right @ right.T)).clamp_min(epsilon)
            )
            scales = left.square().sum(dim=0).sqrt().clamp_min(epsilon)
            left.div_(scales)
            right.mul_(scales.unsqueeze(1))
    target_energy = target.square().sum().clamp_min(1e-30)
    cross = (left * (target @ right.T)).sum()
    approximation_energy = ((left.T @ left) * (right @ right.T)).sum()
    residual = (target_energy - 2.0 * cross + approximation_energy).clamp_min(0.0)
    relative_squared_error = float((residual / target_energy).clamp_min(0.0))
    with torch.no_grad():
        first.pre_w.copy_(
            inverse_weight_transform(right.clamp_min(1e-8), first.weight_transform).to(
                device=first.pre_w.device, dtype=first.pre_w.dtype
            )
        )
        second.pre_w.copy_(
            inverse_weight_transform(left.clamp_min(1e-8), second.weight_transform).to(
                device=second.pre_w.device, dtype=second.pre_w.dtype
            )
        )
        if second.pre_bias is not None:
            second.pre_bias.copy_(
                inverse_weight_transform(
                    torch.full_like(second.pre_bias, 1e-8),
                    second.weight_transform,
                )
            )
    return {
        "readout_initialized": True,
        "readout_mode": "nonnegative_low_rank_coefficient_factorization",
        "factorization_schema": "low_rank_coefficient_factorization/v1",
        "factorization_algorithm": "absolute_subspace_seed_multiplicative_nmf",
        "factorization_rank": rank,
        "nmf_iterations": int(nmf_iterations),
        "factorization_compute_dtype": "float32",
        "factorization_compute_device": str(coefficient.device),
        "coefficient_sha256": _tensor_sha256(coefficient),
        "coefficient_fit_error_scope": (
            "supplied nonnegative pathway coefficient versus installed two-factor "
            "effective weight"
        ),
        "coefficient_relative_squared_frobenius_error": relative_squared_error,
        "claim_boundary": (
            "This compatibility utility approximately factorizes a supplied "
            "coefficient; canonical example fitting uses direct reduced-rank ridge."
        ),
    }


def _set_readout_(projection: nn.Module, coefficient: torch.Tensor) -> dict[str, Any]:
    """Install a regression readout without relaxing a biological contract."""

    if isinstance(projection, PositiveEIOutputProjection):
        positive = coefficient.clamp_min(0.0)
        negative = (-coefficient).clamp_min(0.0)
        if isinstance(projection.excitatory, nn.Sequential) or isinstance(
            projection.inhibitory, nn.Sequential
        ):
            if not isinstance(projection.excitatory, nn.Sequential) or not isinstance(
                projection.inhibitory, nn.Sequential
            ):
                raise ValueError(
                    "positive E/I low-rank readout pathways must use matched "
                    "Sequential factor structures"
                )
            excitatory = _copy_positive_low_rank_(projection.excitatory, positive)
            inhibitory = _copy_positive_low_rank_(projection.inhibitory, negative)
            return {
                "readout_initialized": True,
                "readout_mode": "positive_e_minus_i_coefficient_factorization",
                "factorization_schema": "low_rank_coefficient_factorization/v1",
                "excitatory": excitatory,
                "inhibitory": inhibitory,
            }
        if not isinstance(projection.excitatory, IndexedSparseLinear) or not isinstance(
            projection.inhibitory, IndexedSparseLinear
        ):
            raise ValueError(
                "unsupported positive E/I teacher-conditioned readout structure"
            )
        return {
            "readout_initialized": True,
            "readout_mode": "positive_e_minus_i_ridge",
            "readout_excitatory_energy_retained": _copy_effective_indexed_(
                projection.excitatory, positive
            ),
            "readout_inhibitory_energy_retained": _copy_effective_indexed_(
                projection.inhibitory, negative
            ),
        }
    if isinstance(projection, nn.Sequential):
        return _copy_signed_low_rank_(projection, coefficient)
    if isinstance(projection, IndexedSparseLinear):
        return {
            "readout_initialized": True,
            "readout_mode": "signed_ridge_topk",
            "readout_energy_retained": _copy_effective_indexed_(
                projection, coefficient
            ),
        }
    if isinstance(projection, nn.Linear):
        if tuple(projection.weight.shape) != tuple(coefficient.shape):
            return {"readout_initialized": False, "reason": "readout_shape_mismatch"}
        with torch.no_grad():
            projection.weight.copy_(
                coefficient.to(
                    device=projection.weight.device,
                    dtype=projection.weight.dtype,
                )
            )
            if projection.bias is not None:
                projection.bias.zero_()
        return {"readout_initialized": True, "readout_mode": "dense_ridge"}
    return {"readout_initialized": False, "reason": "unsupported_readout_type"}


def _fit_projection_from_examples_(
    projection: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Fit one readout through its executable representation.

    Low-rank paths use direct reduced-rank ridge and never form a dense
    coefficient. Other readouts retain the exact full-rank ridge path before
    their declared dense or sparse storage projection is applied.
    """

    if isinstance(projection, nn.Sequential):
        first, _ = _low_rank_layers(projection)
        rank = int(first.weight.shape[0])
        left, right, fit = _direct_reduced_rank_ridge_factors(
            features,
            targets,
            rank=rank,
            ridge=ridge,
        )
        diagnostics = _copy_signed_low_rank_factors_(
            projection,
            left,
            right,
            fit_diagnostics=fit,
        )
    elif isinstance(projection, PositiveEIOutputProjection) and (
        isinstance(projection.excitatory, nn.Sequential)
        or isinstance(projection.inhibitory, nn.Sequential)
    ):
        if not isinstance(projection.excitatory, nn.Sequential) or not isinstance(
            projection.inhibitory, nn.Sequential
        ):
            raise ValueError(
                "biological low-rank E/I paths must use matched factor structures"
            )
        e_first, _ = _low_rank_layers(projection.excitatory)
        i_first, _ = _low_rank_layers(projection.inhibitory)
        pathway_rank = int(e_first.weight.shape[0])
        if int(i_first.weight.shape[0]) != pathway_rank:
            raise ValueError("biological low-rank E/I pathway ranks differ")
        signed_seed_rank = pathway_rank // 2
        if signed_seed_rank < 1:
            raise ValueError(
                "teacher-conditioned biological low-rank initialization needs "
                "pathway rank >= 2"
            )
        left, right, fit = _direct_reduced_rank_ridge_factors(
            features,
            targets,
            rank=signed_seed_rank,
            ridge=ridge,
        )
        diagnostics = _copy_biological_low_rank_seed_(
            projection,
            left,
            right,
            fit_diagnostics=fit,
        )
    else:
        coefficient = _ridge_readout(features, targets, ridge=ridge)
        diagnostics = _set_readout_(projection, coefficient)
        diagnostics["full_coefficient_fit"] = {
            "estimator": "exact_relative_ridge",
            "solver": "smaller_of_sample_or_primal_cholesky",
            "estimator_fallback_used": False,
            "coefficient_shape": list(coefficient.shape),
            "coefficient_sha256": _tensor_sha256(coefficient),
        }

    with torch.no_grad():
        projection_parameter = next(projection.parameters())
        canonical_features = (
            features.detach()
            .reshape(-1, features.shape[-1])
            .to(
                device=projection_parameter.device,
                dtype=projection_parameter.dtype,
            )
        )
        canonical_targets = (
            targets.detach()
            .reshape(-1, targets.shape[-1])
            .to(
                device=canonical_features.device,
                dtype=torch.float32,
            )
        )
        prediction = projection(canonical_features).float()
        residual = (prediction - canonical_targets).square().mean()
        baseline = canonical_targets.square().mean().clamp_min(1e-30)
    diagnostics["calibration_prediction_mse"] = float(residual)
    diagnostics["calibration_prediction_relative_mse"] = float(residual / baseline)
    diagnostics["calibration_prediction_scope"] = (
        "installed executable readout versus supplied calibration targets"
    )
    return diagnostics


def _fit_and_set_readout_(
    replacement: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, Any]:
    scale = float(getattr(replacement, "output_scale", 1.0))
    scaled_targets = targets / scale
    residual_targets = scaled_targets
    bypass_diagnostics: dict[str, Any] = {"readout_initialized": False}
    affine_bypass = getattr(replacement, "affine_bypass", None)
    if affine_bypass is not None:
        with torch.no_grad():
            normalized = replacement.pre_norm(inputs)
            bypass_features = replacement.input_adapter(normalized)
        bypass_diagnostics = _fit_projection_from_examples_(
            affine_bypass,
            bypass_features,
            scaled_targets,
            ridge=ridge,
        )
        with torch.no_grad():
            residual_targets = scaled_targets - affine_bypass(bypass_features)
    with torch.no_grad():
        features = _replacement_features(replacement, inputs)
    diagnostics = _fit_projection_from_examples_(
        replacement.output_projection,
        features,
        residual_targets,
        ridge=ridge,
    )
    diagnostics["affine_bypass"] = bypass_diagnostics
    with torch.no_grad():
        prediction = replacement(inputs)
        mse = F.mse_loss(prediction.float(), targets.float())
        baseline = targets.float().square().mean().clamp_min(1e-12)
    end_to_end_relative_mse = float(mse / baseline)
    diagnostics["relative_mse"] = end_to_end_relative_mse
    diagnostics["end_to_end_relative_mse"] = end_to_end_relative_mse
    diagnostics["end_to_end_error_scope"] = (
        "full initialized replacement output versus calibration targets"
    )
    return diagnostics


def initialize_population_topology_from_targets_(
    replacement: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    metric: str = "activation_weighted",
    max_rows: int = 4096,
    ridge: float = 1e-4,
    row_chunk_size: int = DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE,
) -> dict[str, Any]:
    """Initialize a full compiled PopulationNetwork from boundary examples.

    ``inputs`` and ``targets`` must describe the same vector-valued replacement
    boundary. It works for both single-path and gated PopulationNetwork cells.
    """

    from dendritic_modeling.networks.architectures.transformer.modules import (
        unwrap_collapsed_population_replacement,
        unwrap_shared_population_replacement,
    )

    replacement = unwrap_collapsed_population_replacement(
        unwrap_shared_population_replacement(replacement)
    )
    normalized_metric = str(metric).strip().lower()
    if normalized_metric not in TEACHER_TOPOLOGY_METRICS:
        raise ValueError(
            f"metric must be one of {TEACHER_TOPOLOGY_METRICS}, got {metric!r}"
        )
    if not hasattr(replacement, "output_projection"):
        raise TypeError("replacement must expose output_projection")
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, Integral)
        or int(row_chunk_size) < 1
    ):
        raise ValueError("row_chunk_size must be a positive integer")
    flat = _flatten_examples(inputs, max_rows=max_rows)
    flat_targets = targets.detach().reshape(-1, targets.shape[-1])[: flat.shape[0]]
    if flat_targets.shape[0] != flat.shape[0]:
        raise ValueError("teacher targets do not align with calibration inputs")
    parameter = next(replacement.parameters())
    flat = flat.to(device=parameter.device, dtype=parameter.dtype)
    targets = flat_targets.to(device=parameter.device, dtype=parameter.dtype)

    was_training = replacement.training
    replacement.eval()
    initial_readout = _fit_and_set_readout_(
        replacement,
        flat,
        targets,
        ridge=ridge,
    )

    sparse_modules: dict[str, IndexedSparseLinear] = {}
    for core_name in ("core", "gate_core", "value_core"):
        core = getattr(replacement, core_name, None)
        if core is None:
            continue
        for name, module in core.named_modules():
            if isinstance(module, IndexedSparseLinear):
                sparse_modules[f"{core_name}.{name}"] = module

    row_conditioned = normalized_metric in ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS
    global_captures: dict[str, list[torch.Tensor]] = {
        name: [] for name in sparse_modules
    }
    row_captures: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        name: [] for name in sparse_modules
    }
    handles = []
    for name, module in sparse_modules.items():
        if row_conditioned:

            def capture_input_and_output(_module, args, output, *, key=name):
                value = args[0]
                if not torch.is_tensor(value) or not torch.is_tensor(output):
                    raise TypeError(
                        "row-conditioned teacher topology requires tensor "
                        "inputs and outputs"
                    )
                if not output.requires_grad:
                    row_captures[key].append((value.detach(), output))
                    return None
                # Route downstream consumers through a private view so the
                # retained gradient is dL/d(this module's output), including
                # when several sparse pathways share the same input tensor.
                output_alias = output.view_as(output)
                output_alias.retain_grad()
                row_captures[key].append((value.detach(), output_alias))
                return output_alias

            handles.append(module.register_forward_hook(capture_input_and_output))
        else:

            def capture_input(_module, args, *, key=name):
                value = args[0]
                if not torch.is_tensor(value) or not value.requires_grad:
                    global_captures[key].append(value)
                    return None
                # Several synapse layers can consume the SAME tensor object
                # (an E and an I pathway reading one population's output, or
                # the two cores of a gated cell reading the adapter).
                # ``retain_grad`` on that shared tensor accumulates gradient
                # over ALL consumers, so route each consumer through a private
                # view to preserve the historical pathway-specific statistic.
                alias = value.view_as(value)
                alias.retain_grad()
                global_captures[key].append(alias)
                return (alias, *args[1:])

            handles.append(module.register_forward_pre_hook(capture_input))

    try:
        saliency_inputs = flat.detach().requires_grad_(True)
        prediction = replacement(saliency_inputs)
        F.mse_loss(prediction.float(), targets.float()).backward()
    finally:
        for handle in handles:
            handle.remove()

    per_layer: dict[str, Any] = {}
    initialized = 0
    skipped = 0
    for name, module in sparse_modules.items():
        score = None
        score_diagnostics: dict[str, Any]
        if row_conditioned:
            values = []
            output_gradients = []
            for value, output in row_captures[name]:
                if output.grad is None:
                    continue
                flattened_value = value.reshape(-1, value.shape[-1])
                flattened_gradient = output.grad.detach().reshape(-1, output.shape[-1])
                if flattened_value.shape[0] != flattened_gradient.shape[0]:
                    raise ValueError(
                        f"captured leading dimensions do not align for {name}"
                    )
                values.append(flattened_value)
                output_gradients.append(flattened_gradient)
            if values:
                all_values = values[0] if len(values) == 1 else torch.cat(values, dim=0)
                all_output_gradients = (
                    output_gradients[0]
                    if len(output_gradients) == 1
                    else torch.cat(output_gradients, dim=0)
                )
                score, score_diagnostics = _row_conditioned_saliency(
                    all_values,
                    all_output_gradients,
                    metric=normalized_metric,
                    row_chunk_size=int(row_chunk_size),
                )
            else:
                score_diagnostics = {
                    "score_scope": "output_row_conditioned",
                    "score_definition": _SCORE_DEFINITIONS[normalized_metric],
                }
        else:
            for value in global_captures[name]:
                if not torch.is_tensor(value) or value.grad is None:
                    continue
                flattened_value = value.detach().reshape(-1, value.shape[-1]).float()
                flattened_grad = (
                    value.grad.detach().reshape(-1, value.shape[-1]).float()
                )
                if normalized_metric == "gradient":
                    contribution = flattened_grad.abs().mean(dim=0)
                else:
                    contribution = _activation_weighted_saliency(
                        flattened_value,
                        flattened_grad,
                        robust=(normalized_metric == "activation_weighted_robust"),
                    )
                score = contribution if score is None else score + contribution
            score_diagnostics = {
                "score_scope": "module_global_legacy",
                "score_definition": _SCORE_DEFINITIONS[normalized_metric],
                "score_shape": [int(module.in_features)],
                "score_compute_dtype": "float32",
            }
        score_total = float(score.double().sum()) if score is not None else float("nan")
        if (
            score is None
            or not bool(torch.isfinite(score).all())
            or not math.isfinite(score_total)
            or score_total <= 0
        ):
            per_layer[name] = {"initialized": False, "reason": "no_finite_saliency"}
            skipped += 1
            continue
        indices = _set_support_(module, score)
        unique_supports = int(torch.unique(indices, dim=0).shape[0])
        per_layer[name] = {
            "initialized": True,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "k": int(module.K),
            "saliency_sum": score_total,
            "support_group_rows": int(getattr(module, "support_group_rows", 1)),
            "support_col_block": int(getattr(module, "support_col_block", 1)),
            "unique_support_rows": unique_supports,
            "support_union_features": int(torch.unique(indices).numel()),
            "support_indices_sha256": _tensor_sha256(indices),
            **score_diagnostics,
        }
        initialized += 1

    replacement.zero_grad(set_to_none=True)
    final_readout = _fit_and_set_readout_(
        replacement,
        flat,
        targets,
        ridge=ridge,
    )
    replacement.train(was_training)
    diagnostics = {
        "schema": "population_teacher_topology_init/v2",
        "metric": normalized_metric,
        "metric_scope": (
            "output_row_conditioned" if row_conditioned else "module_global_legacy"
        ),
        "score_definition": _SCORE_DEFINITIONS[normalized_metric],
        "saliency_objective": "mean_squared_teacher_reconstruction",
        "saliency_loss_reduction": "mean",
        "score_compute_dtype": "float32",
        "row_chunk_size": int(row_chunk_size) if row_conditioned else None,
        "robust_saliency_quantile": (
            ROBUST_SALIENCY_QUANTILE
            if normalized_metric == "activation_weighted_robust"
            else None
        ),
        "calibration_rows": int(flat.shape[0]),
        "calibration_inputs_sha256": _tensor_sha256(flat),
        "calibration_targets_sha256": _tensor_sha256(targets),
        "topology_layers_considered": len(sparse_modules),
        "topology_layers_initialized": initialized,
        "topology_layers_skipped": skipped,
        "initial_readout": initial_readout,
        "final_readout": final_readout,
        "layers": per_layer,
        "status": "teacher_conditioned_initializer_not_optimality_claim",
    }
    replacement.teacher_topology_diagnostics = diagnostics
    # Keep the historical checkpoint/report field populated as well.
    replacement.teacher_topk_diagnostics = diagnostics
    manifest = getattr(replacement, "selection_manifest", None)
    if isinstance(manifest, dict):
        execution = manifest.setdefault("execution", {})
        execution["teacher_support_metric"] = {
            "status": "executed",
            "initializer_schema": diagnostics["schema"],
            "metric": normalized_metric,
            "metric_scope": diagnostics["metric_scope"],
            "score_definition": diagnostics["score_definition"],
            "saliency_objective": diagnostics["saliency_objective"],
            "saliency_loss_reduction": diagnostics["saliency_loss_reduction"],
            "score_compute_dtype": diagnostics["score_compute_dtype"],
            "row_chunk_size": diagnostics["row_chunk_size"],
            "calibration_rows": int(flat.shape[0]),
            "calibration_inputs_sha256": diagnostics["calibration_inputs_sha256"],
            "calibration_targets_sha256": diagnostics["calibration_targets_sha256"],
            "core_topology_layers_initialized": initialized,
            "core_topology_layers_considered": len(sparse_modules),
            "readout_initialized": bool(
                final_readout.get("readout_initialized", False)
            ),
            "affine_bypass_readout": final_readout.get("affine_bypass", {}),
        }
    return diagnostics


def initialize_population_topology_from_teacher_(
    replacement: nn.Module,
    teacher: nn.Module,
    inputs: torch.Tensor,
    **kwargs: Any,
) -> dict[str, Any]:
    """Profile a teacher at a shared vector boundary and initialize the cell."""

    max_rows = int(kwargs.get("max_rows", 4096))
    flat = _flatten_examples(inputs, max_rows=max_rows)
    teacher_parameter = next(teacher.parameters(), None)
    replacement_parameter = next(replacement.parameters())
    teacher_device = (
        replacement_parameter.device
        if teacher_parameter is None
        else teacher_parameter.device
    )
    teacher_dtype = (
        replacement_parameter.dtype
        if teacher_parameter is None
        else teacher_parameter.dtype
    )
    with torch.no_grad():
        targets = teacher(flat.to(device=teacher_device, dtype=teacher_dtype)).detach()
    return initialize_population_topology_from_targets_(
        replacement,
        flat,
        targets,
        **kwargs,
    )


def supports_teacher_topology_initialization(module: nn.Module) -> bool:
    from dendritic_modeling.networks.architectures.transformer.modules import (
        unwrap_collapsed_population_replacement,
        unwrap_shared_population_replacement,
    )

    module = unwrap_collapsed_population_replacement(
        unwrap_shared_population_replacement(module)
    )
    return (
        hasattr(module, "output_projection")
        and (hasattr(module, "core") or hasattr(module, "gate_core"))
        and bool(getattr(module, "teacher_support_metric", ""))
    )


__all__ = [
    "DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE",
    "LEGACY_GLOBAL_TEACHER_TOPOLOGY_METRICS",
    "ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS",
    "TEACHER_TOPOLOGY_METRICS",
    "initialize_population_topology_from_targets_",
    "initialize_population_topology_from_teacher_",
    "supports_teacher_topology_initialization",
]
