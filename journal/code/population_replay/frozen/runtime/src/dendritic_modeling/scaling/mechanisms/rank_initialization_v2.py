"""TRAIN-only subspaces from a boundary-weighted second-order ball identity.

For uniform X in a d-ball of radius R, put phi=(R^2-||X||^2)^2.
Both phi and its first derivatives vanish on the boundary. Two integrations
by parts give E[f Hess(phi)] = E[phi Hess(f)], where
Hess(phi)=8XX^T-4(R^2-||X||^2)I. For additive independent blocks and ridge
components with nonzero, common-sign curvature, this matrix has exactly the
active subspace as its range. This is not a guarantee for generic nonlinear
responses, cancellation between curvatures, or finite samples.

All operations here use supplied TRAIN inputs/scalar labels and the declared
input law. Same-sample affine residualization does not claim unbiasedness.
Saved eigengaps/half-split disagreement diagnose stability, never select a fit.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .rank_learning import RankBlockModel


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ball_stein_kernel(x, radius=0.5):
    """Return normalized Hess(phi); E(phi)=8R^4/((d+2)(d+4))."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or not np.isfinite(x).all():
        raise ValueError("Finite coordinate arrays required")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite ball radius required")
    squared = np.sum(x * x, axis=-1)
    if np.any(squared > radius**2 + 1e-12):
        raise ValueError("Input outside declared ball")
    d = x.shape[-1]
    mean_phi = 8 * radius**4 / ((d + 2) * (d + 4))
    return (
        8 * x[..., :, None] * x[..., None, :]
        - 4 * (radius**2 - squared)[..., None, None] * np.eye(d)
    ) / mean_phi


def _affine_residual(x, y):
    matrix = np.column_stack((np.ones(len(x)), x.reshape(len(x), -1)))
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, y, rcond=1e-12)
    return (
        y - matrix @ coefficients,
        coefficients[1:].reshape(x.shape[1:]),
        {
            "rank": int(rank),
            "columns": matrix.shape[1],
            "rcond": 1e-12,
            "singular_values": singular.tolist(),
            "coefficients": coefficients.tolist(),
        },
    )


def _canonicalize(vectors):
    vectors = vectors.copy()
    for column in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return vectors


def _moment_basis(x, response, radius, rank):
    moment = np.einsum("n,nkij->kij", response, ball_stein_kernel(x, radius)) / len(x)
    values, vectors = np.linalg.eigh(moment)
    ordered_values, bases, gaps = [], [], []
    for eigenvalues, eigenvectors in zip(values, vectors):
        order = np.argsort(-np.abs(eigenvalues), kind="stable")
        sorted_values = eigenvalues[order]
        ordered_values.append(sorted_values)
        bases.append(_canonicalize(eigenvectors[:, order[:rank]]))
        gaps.append(
            float(abs(sorted_values[rank - 1]) - abs(sorted_values[rank]))
            if rank < x.shape[-1]
            else None
        )
    return np.array(bases), moment, np.array(ordered_values), gaps


def _complete(first, rank):
    vectors = [first / np.linalg.norm(first)]
    while len(vectors) < rank:
        projector = (
            np.eye(len(first)) - np.column_stack(vectors) @ np.column_stack(vectors).T
        )
        axis = int(np.argmax(np.sum(projector**2, axis=0)))
        vector = projector[:, axis]
        vectors.append(vector / np.linalg.norm(vector))
    return np.column_stack(vectors)


def estimate_subspaces(x, y, rank=2, radius=0.5, method="stein", residualize=True):
    """Return orthonormal ``[K,d,rank]`` bank and JSON-serializable receipt.

    The spectral span is oriented toward the projected TRAIN OLS slope, with
    deterministic completion inside that span. A zero projected slope retains
    the canonical eigensolver bank. No eigengap threshold changes
    the selected rank or method. OLS uses a fixed coordinate fallback if its
    finite-sample slope is zero, including exactly symmetric quadratic data.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or y.shape != (len(x),) or len(x) < 4:
        raise ValueError("At least four raw block inputs and scalar labels required")
    if not np.isfinite(y).all() or not 1 <= rank <= x.shape[-1]:
        raise ValueError("Finite labels and supported rank required")
    if method not in ("stein", "ols"):
        raise ValueError("Initializer must be stein or ols")
    ball_stein_kernel(x, radius)
    residual, slopes, affine = _affine_residual(x, y)
    selected_response = residual if residualize else y - y.mean()
    spectral, moment, eigenvalues, gaps = _moment_basis(
        x, selected_response, radius, rank
    )
    _, raw_moment, raw_eigenvalues, _ = _moment_basis(x, y - y.mean(), radius, rank)
    bank, fallbacks, spectral_orientation_fallbacks = [], [], []
    for block in range(x.shape[1]):
        if method == "stein":
            basis = spectral[block]
            projected = basis @ (basis.T @ slopes[block])
            if np.linalg.norm(projected) > 1e-12 * max(
                1.0, np.linalg.norm(slopes[block])
            ):
                oriented = [projected / np.linalg.norm(projected)]
                while len(oriented) < rank:
                    current = np.column_stack(oriented)
                    candidates = basis - current @ (current.T @ basis)
                    pivot = int(np.argmax(np.linalg.norm(candidates, axis=0)))
                    direction = candidates[:, pivot]
                    oriented.append(direction / np.linalg.norm(direction))
                basis = np.column_stack(oriented)
            else:
                spectral_orientation_fallbacks.append(block)
        else:
            first = slopes[block].copy()
            if np.linalg.norm(first) < 1e-12:
                first = np.eye(x.shape[-1])[0]
                fallbacks.append(block)
            basis = _complete(first, rank)
        bank.append(basis)
    halves = []
    for index in (np.arange(0, len(x), 2), np.arange(1, len(x), 2)):
        half_residual, _, _ = _affine_residual(x[index], y[index])
        response = half_residual if residualize else y[index] - y[index].mean()
        halves.append(_moment_basis(x[index], response, radius, rank)[0])
    disagreement = [
        float(np.linalg.norm(left @ left.T - right @ right.T) / math.sqrt(2))
        for left, right in zip(*halves)
    ]
    bank = np.array(bank)
    return bank, {
        "method": method,
        "rank": rank,
        "radius": radius,
        "rows": len(x),
        "shape": list(x.shape),
        "train_x_sha256": _hash(x),
        "train_y_sha256": _hash(y),
        "direction_bank_sha256": _hash(bank),
        "residualized": bool(residualize),
        "affine_regression": affine,
        "moment": moment.tolist(),
        "raw_centered_label_moment": raw_moment.tolist(),
        "eigenvalues_absolute_order": eigenvalues.tolist(),
        "raw_eigenvalues_absolute_order": raw_eigenvalues.tolist(),
        "absolute_eigengap_at_rank": gaps,
        "within_selected_absolute_eigenvalue_gaps": np.diff(
            -np.abs(eigenvalues[:, :rank]), axis=1
        ).tolist(),
        "half_split_projector_distance": disagreement,
        "half_split_rule": "Fixed even/odd rows, refit affine residualizer within each half; diagnostics only, never selection.",
        "ols_zero_fallback_blocks": fallbacks,
        "spectral_orientation_zero_projected_ols_blocks": spectral_orientation_fallbacks,
        "orientation": "First spectral-bank direction is normalized projection of TRAIN joint OLS into the estimated span; remaining directions complete that span deterministically. If projected OLS is zero, retain canonical eigensolver basis.",
        "scope": "TRAIN-only observed scalar-label estimator. Input law/radius supplied, no teacher directions/curvatures. Same-sample residualization is not finite-sample unbiased. Top-absolute-eigenvector guarantee requires noncanceling sign-definite ridge curvature in the population. Equal-response orthogonal mixture axes have repeated active eigenvalues: this identifies their span, not the individual additive axes.",
    }


def node_directions(bank, nodes_per_block=1, pattern="equally_spaced", spread=0.1):
    """Fixed node directions from a common TRAIN bank, in block-major order.

    Equally spaced L nodes use angles j*pi/L in the supplied plane. L=1 uses
    only the first direction; L=2 uses both axes; L=4 adds their diagonals.
    Spread is a separate explicit matching control, never outcome-selected.
    """
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 3 or not np.isfinite(bank).all() or nodes_per_block < 1:
        raise ValueError("Finite direction bank and positive node count required")
    if int(nodes_per_block) != nodes_per_block:
        raise ValueError("Whole node count required")
    if pattern not in ("equally_spaced", "spread"):
        raise ValueError("Unknown node direction pattern")
    if bank.shape[-1] < 2 and nodes_per_block > 1:
        raise ValueError("Multiple nodes require a supplied plane")
    if not math.isfinite(spread) or spread < 0:
        raise ValueError("Nonnegative finite spread required")
    result = []
    for block in bank:
        for node in range(nodes_per_block):
            if nodes_per_block == 1:
                direction = block[:, 0]
            elif pattern == "equally_spaced":
                angle = node * math.pi / nodes_per_block
                direction = (
                    math.cos(angle) * block[:, 0] + math.sin(angle) * block[:, 1]
                )
            else:
                direction = block[:, 0] + (-1) ** node * spread * block[:, 1]
            result.append(direction / np.linalg.norm(direction))
    return np.array(result)


def node_banks(bank, nodes_per_block=1, pattern="equally_spaced", spread=0.1):
    """Rotate each observed plane to the requested node direction and its perpendicular."""
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[-1] != 2:
        raise ValueError("A two-column direction bank required")
    if not np.allclose(
        bank.transpose(0, 2, 1) @ bank, np.eye(2), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Node banks require an orthonormal input bank")
    first = node_directions(bank, nodes_per_block, pattern, spread)
    result = []
    for index, direction in enumerate(first):
        plane = bank[index // nodes_per_block]
        c, s = direction @ plane
        perpendicular = -s * plane[:, 0] + c * plane[:, 1]
        result.append(np.column_stack((direction, perpendicular)))
    return np.array(result)


def make_initialized_model(
    capacities,
    family,
    architecture,
    x,
    y,
    bank,
    *,
    seed=0,
    initialization_spread=0.1,
    threshold_mode="quantile",
    initializer_receipt=None,
    coverage="legacy",
):
    """Use the same bank, scales, knots and starts for full/rank2 at equal M.

    Rank1 uses the bank's first axis only. Width callers can instead pass
    ``node_directions`` to their counted wrapper's explicit-direction API.
    """
    bank = np.asarray(bank, dtype=np.float64)
    if coverage not in ("legacy", "axial"):
        raise ValueError("Coverage must be legacy or axial")
    if (
        bank.shape[:2] != (len(capacities), np.asarray(x).shape[-1])
        or bank.shape[-1] < 2
    ):
        raise ValueError("A two-direction bank for every original block required")
    gram = np.swapaxes(bank, 1, 2) @ bank
    if not np.allclose(gram, np.eye(bank.shape[-1]), atol=1e-10, rtol=1e-10):
        raise ValueError("Direction bank must be orthonormal")
    model = RankBlockModel(
        capacities,
        family,
        architecture,
        inputs_per_block=bank.shape[1],
        seed=seed,
        train_data=(x, y),
        effective_directions=bank[:, :, 0],
        initialization_spread=initialization_spread,
        threshold_mode=threshold_mode,
    )
    with torch.no_grad():
        basis = torch.as_tensor(bank[:, :, :2], dtype=torch.float64)
        if coverage == "axial" and architecture != "rank1":
            projections, coefficients, thresholds = [], [], []
            for block, capacity in enumerate(capacities):
                for axis, local_capacity in enumerate(
                    ((capacity + 1) // 2, capacity // 2)
                ):
                    if local_capacity == 0:
                        continue
                    local = RankBlockModel(
                        [local_capacity],
                        family,
                        "rank1",
                        inputs_per_block=bank.shape[1],
                        seed=seed + 1009 * block + 9176 * axis,
                        train_data=(np.asarray(x)[:, block : block + 1], y),
                        effective_directions=bank[block : block + 1, :, axis],
                        threshold_mode=threshold_mode,
                    )
                    projections.append(local.effective_projection())
                    gain = torch.zeros(local_capacity, 2, dtype=torch.float64)
                    gain[:, axis] = local.branch_coefficients[:, 0]
                    coefficients.append(gain)
                    thresholds.append(local.threshold)
            model.threshold.copy_(torch.cat(thresholds))
            if architecture == "full":
                model.projection.copy_(torch.cat(projections))
            else:
                model.basis.copy_(basis)
                model.branch_coefficients.copy_(torch.cat(coefficients))
        elif architecture == "rank2":
            if initialization_spread == 0:
                model.basis.copy_(
                    torch.stack(
                        (basis[:, :, 0] - 0.1 * basis[:, :, 1], basis[:, :, 1]), dim=2
                    )
                )
            else:
                model.basis.copy_(basis)
        elif architecture == "full":
            magnitude = model.projection.norm(dim=1)
            signs = torch.cat([1 - 2 * (torch.arange(c) % 2) for c in capacities]).to(
                torch.float64
            )
            direction = (
                basis[model.group, :, 0]
                + initialization_spread * signs[:, None] * basis[model.group, :, 1]
            )
            direction /= math.sqrt(1 + initialization_spread**2)
            model.projection.copy_(magnitude[:, None] * direction)
    model.initialization_receipt["direction_bank"] = {
        "sha256": _hash(bank),
        "basis": bank.tolist(),
        "estimator": initializer_receipt,
        "coverage": coverage,
        "coverage_rule": "Legacy uses alternating narrow +/-spread directions; axial assigns ceil(capacity/2) branches to bank0 and floor(capacity/2) to bank1, with TRAIN projection-specific scales/knots. Rank1 always uses bank0. Axial initializer-only local seed=seed+1009*block+9176*axis; initial readouts retain common original seed.",
        "matching": "Full/rank2 exactly share effective features at equal capacities/seed/spread. Rank1 uses first bank axis only.",
    }
    return model
