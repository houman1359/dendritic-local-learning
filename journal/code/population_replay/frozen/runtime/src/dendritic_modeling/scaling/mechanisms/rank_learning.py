"""Stored-slot-matched block models and stable positive-ridge variable projection.

Soma multipliers are stored and counted but fixed at one in BOTH optimizers.
Ridge penalizes coefficients of TRAIN-centered, RMS-standardized features,
never redundant internal readout coordinates. Normalization is differentiated
and folded into existing deployed slots. No normal equations or singular-value
cutoff is used. The fixed RMS floor slightly breaks exact scale invariance only
for near-constant features.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

FAMILIES = ("shunt", "relu", "tanh")
ARCHITECTURES = ("full", "rank1", "rank2")
RMS_EPSILON = 1e-8


def _tensor(x, reference=None):
    return torch.as_tensor(
        x, dtype=torch.float64, device=None if reference is None else reference.device
    )


def _first_directions(x, y):
    matrix = (x - x.mean(0)).reshape(len(x), -1)
    rhs = y - y.mean()
    # Initializer only. This regression is separate from the positive-ridge
    # inner optimization; its numerical rank is explicitly recorded.
    solution = torch.linalg.lstsq(matrix, rhs, driver="gelsd", rcond=1e-12)
    coefficients = solution.solution.reshape(x.shape[1:])
    norm = coefficients.norm(dim=1, keepdim=True)
    if bool((norm < 1e-14).any()):
        raise ValueError(
            "Zero observational direction; use the explicit random initializer"
        )
    return coefficients / norm, {
        "method": "Scalar TRAIN-only centered joint OLS; extract one direction per supplied block.",
        "rows": len(x),
        "rank": int(solution.rank),
        "rcond": 1e-12,
    }


def _orthogonal(first):
    axis = torch.argmin(first.abs(), dim=1)
    second = torch.zeros_like(first)
    second[torch.arange(len(first)), axis] = 1
    second -= (second * first).sum(dim=1, keepdim=True) * first
    return second / second.norm(dim=1, keepdim=True)


class RankBlockModel(nn.Module):
    """Full or factorized branch input weights, with identical nonlinear units.

    With positive ``initialization_spread``, full/rank2 have the same genuinely
    non-collinear effective inputs; rank1 uses the first direction alone. With
    spread zero, all three architectures match algebraically, and rank2 uses a
    compensated nonzero second coefficient. That control can still be trapped
    at a symmetric rank-one stationary point; nonzero coordinates are no cure.
    """

    def __init__(
        self,
        capacities,
        family,
        architecture="full",
        inputs_per_block=3,
        seed=0,
        train_data=None,
        initialization="observational",
        effective_directions=None,
        initialization_spread=0.1,
        threshold_mode="quantile",
    ):
        super().__init__()
        if family not in FAMILIES or architecture not in ARCHITECTURES:
            raise ValueError("Unknown family or architecture")
        if not capacities or any(int(c) != c or c < 1 for c in capacities):
            raise ValueError("Positive integer branch capacities required")
        if inputs_per_block < 1 or (architecture == "rank2" and inputs_per_block < 2):
            raise ValueError("Input dimension cannot support requested rank")
        if initialization not in ("observational", "random"):
            raise ValueError("Unknown initializer")
        if not math.isfinite(initialization_spread) or initialization_spread < 0:
            raise ValueError("Nonnegative finite initialization spread required")
        if threshold_mode not in ("quantile", "midpoint"):
            raise ValueError("Unknown threshold mode")
        self.capacities = tuple(int(c) for c in capacities)
        self.family, self.architecture = family, architecture
        self.inputs_per_block = int(inputs_per_block)
        self.rank = 0 if architecture == "full" else int(architecture[-1])
        self.threshold_mode = threshold_mode
        self.initialization_spread = float(initialization_spread)
        blocks, branches = len(capacities), sum(capacities)
        self.register_buffer(
            "group",
            torch.repeat_interleave(torch.arange(blocks), torch.tensor(capacities)),
        )
        generator = torch.Generator().manual_seed(seed)
        x = None if train_data is None else _tensor(train_data[0])
        if x is not None and (x.ndim != 3 or x.shape[1:] != (blocks, inputs_per_block)):
            raise ValueError("Training inputs do not match supplied block geometry")
        if effective_directions is not None:
            first = _tensor(effective_directions)
            if first.shape != (blocks, inputs_per_block) or not bool(
                torch.isfinite(first).all()
            ):
                raise ValueError("Explicit effective directions must have shape [K,d]")
            if bool((first.norm(dim=1) < 1e-14).any()):
                raise ValueError("Explicit directions must be nonzero")
            first = first / first.norm(dim=1, keepdim=True)
            receipt = {
                "method": "Explicit common first directions supplied by caller; their provenance belongs in the campaign protocol."
            }
        elif initialization == "observational":
            if train_data is None:
                raise ValueError(
                    "Observational initializer requires scalar training data"
                )
            first, receipt = _first_directions(x, _tensor(train_data[1]))
        else:
            first = torch.randn(
                blocks, inputs_per_block, generator=generator, dtype=torch.float64
            )
            first /= first.norm(dim=1, keepdim=True)
            receipt = {"method": "Seeded random first direction per block."}
        second = _orthogonal(first) if inputs_per_block > 1 else torch.zeros_like(first)
        scales, offsets, second_coefficients = [], [], []
        for block, capacity in enumerate(capacities):
            projected = None if x is None else x[:, block] @ first[block]
            low, high = (
                (-0.5, 0.5)
                if projected is None
                else (float(projected.min()), float(projected.max()))
            )
            span = max(high - low, 1e-6)
            quantiles = (
                torch.linspace(low, high, capacity + 1, dtype=torch.float64)[:-1]
                if projected is None
                else torch.quantile(
                    projected,
                    (torch.arange(capacity, dtype=torch.float64) + 0.25) / capacity,
                )
            )
            if family == "relu":
                scale = torch.ones(capacity, dtype=torch.float64) / span
                knots = quantiles.clone()
                knots[0] = low - 0.01 * span
            else:
                jitter = (
                    torch.rand(capacity, generator=generator, dtype=torch.float64) - 0.5
                ) * 0.15
                scale = (
                    torch.exp(
                        torch.linspace(
                            math.log(0.1), math.log(30), capacity, dtype=torch.float64
                        )
                        + jitter
                    )
                    / span
                )
                if family == "shunt":
                    knots = torch.full(
                        (capacity,), low - 0.01 * span, dtype=torch.float64
                    )
                elif threshold_mode == "midpoint":
                    knots = torch.full(
                        (capacity,), (low + high) / 2, dtype=torch.float64
                    )
                else:
                    knots = quantiles.clone()
            spread = initialization_spread * (1 - 2 * (torch.arange(capacity) % 2)).to(
                torch.float64
            )
            scales.append(scale)
            offsets.append(-scale * knots)
            second_coefficients.append(spread)
        scale, spread = torch.cat(scales), torch.cat(second_coefficients)
        self.threshold = nn.Parameter(torch.cat(offsets))
        if architecture == "full":
            direction = first[self.group]
            if inputs_per_block > 1 and initialization_spread > 0:
                direction = (
                    direction + spread[:, None] * second[self.group]
                ) / torch.sqrt(1 + spread**2)[:, None]
            self.projection = nn.Parameter(scale[:, None] * direction)
            self.register_parameter("basis", None)
            self.register_parameter("branch_coefficients", None)
        else:
            self.register_parameter("projection", None)
            if self.rank == 1:
                basis, coefficients = first[:, :, None], scale[:, None]
            elif initialization_spread > 0:
                basis = torch.stack((first, second), dim=2)
                coefficients = (
                    scale[:, None]
                    * torch.stack((torch.ones_like(spread), spread), dim=1)
                    / torch.sqrt(1 + spread**2)[:, None]
                )
            else:
                epsilon = 0.1
                basis = torch.stack((first - epsilon * second, second), dim=2)
                coefficients = torch.stack((scale, epsilon * scale), dim=1)
            self.basis = nn.Parameter(basis.clone())
            self.branch_coefficients = nn.Parameter(coefficients.clone())
        self.internal_readout = nn.Parameter(
            torch.randn(branches, generator=generator, dtype=torch.float64)
            / math.sqrt(branches)
        )
        self.soma_readout = nn.Parameter(
            torch.ones(blocks, dtype=torch.float64), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.initialization_receipt = {
            "seed": seed,
            "initializer": initialization,
            "spread": initialization_spread,
            "threshold_mode": threshold_mode,
            **receipt,
            "matching": "Positive spread: full/rank2 match effective features; rank1 uses first direction only. Zero spread: all match algebraically, not a guarantee against rank-one stationary saddles.",
            "soma_gauge": "Stored soma=1, fixed in both reduced and joint optimization; every stored scalar still counted.",
        }
        expected = (
            (inputs_per_block + 2) * branches + blocks + 1
            if self.rank == 0
            else (self.rank + 2) * branches
            + inputs_per_block * self.rank * blocks
            + blocks
            + 1
        )
        assert self.parameter_count == expected

    def effective_projection(self):
        return (
            self.projection
            if self.rank == 0
            else torch.einsum(
                "mdr,mr->md", self.basis[self.group], self.branch_coefficients
            )
        )

    def features(self, x):
        pre = (x[:, self.group, :] * self.effective_projection()).sum(
            dim=2
        ) + self.threshold
        if self.family == "tanh":
            return torch.tanh(pre)
        excitation = torch.relu(pre)
        return excitation if self.family == "relu" else excitation / (1 + excitation)

    def effective_readout(self):
        return (
            self.internal_readout
            * self.soma_readout[self.group]
            / math.sqrt(len(self.capacities))
        )

    def components(self, x):
        weighted = self.features(x) * self.effective_readout()
        result = x.new_zeros((len(x), len(self.capacities)))
        result.scatter_add_(1, self.group.expand(len(x), -1), weighted)
        return result

    def forward(self, x):
        return self.features(x) @ self.effective_readout() + self.bias

    def body_parameters(self):
        return (
            [self.projection]
            if self.rank == 0
            else [self.basis, self.branch_coefficients]
        ) + [self.threshold]

    @property
    def parameter_count(self):
        """All stored parameter slots, including the disclosed fixed soma gauge."""
        return sum(value.numel() for value in self.parameters())

    @property
    def gradient_parameter_count(self):
        return sum(value.numel() for value in self.parameters() if value.requires_grad)


def solve_effective_readout(features, y, ridge):
    """Augmented SVD ridge solve, with unpenalized intercept and no truncation."""
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Strictly positive finite ridge required")
    with torch.no_grad():
        matrix, target = features.detach(), y.detach()
        mean_x, mean_y = matrix.mean(dim=0), target.mean()
        centered, centered_y = matrix - mean_x, target - mean_y
        n, width = matrix.shape
        augmented = torch.cat(
            (
                centered / math.sqrt(n),
                math.sqrt(ridge)
                * torch.eye(width, dtype=matrix.dtype, device=matrix.device),
            )
        )
        rhs = torch.cat((centered_y / math.sqrt(n), target.new_zeros(width)))
        left, singular, right = torch.linalg.svd(augmented, full_matrices=False)
        # Positive ridge makes all singular values nonzero; none are discarded.
        if not bool(torch.isfinite(singular).all()) or float(singular.min()) <= 0:
            raise FloatingPointError(
                "Augmented positive-ridge matrix is numerically singular"
            )
        beta = right.T @ ((left.T @ rhs) / singular)
        intercept = mean_y - mean_x @ beta
        residual = matrix @ beta + intercept - target
        stationarity = centered.T @ residual / n + ridge * beta
        details = {
            "ridge": ridge,
            "rows": n,
            "columns": width,
            "augmented_rows": n + width,
            "solve_rank": width,
            "singular_values": singular.cpu().tolist(),
            "minimum_singular_value": float(singular.min()),
            "maximum_singular_value": float(singular.max()),
            "condition_number": float(singular.max() / singular.min()),
            "coefficient_l2": float(beta.norm()),
            "intercept": float(intercept),
            "stationarity_l2": float(stationarity.norm()),
            "residual_mean": float(residual.mean()),
            "solver": "Augmented SVD, no normal equations, no rank cutoff; intercept unpenalized.",
        }
    return beta, intercept, details


def _commit(model, beta, intercept):
    with torch.no_grad():
        model.soma_readout.fill_(1)
        model.internal_readout.copy_(beta * math.sqrt(len(model.capacities)))
        model.bias.copy_(intercept)


def standardized_features(features, *, projection=None, projection_rms_floor=0.0):
    """TRAIN centering with an optional effective-input slope penalty.

    The extra squared scale is tau**2 * ||w||**2, where w is the effective
    branch input projection. It constrains rare/near-inactive TRAIN features
    without changing inference or the model class. The existing epsilon keeps
    zero projections finite; exact positive-scale invariance is consequently
    only approximate near that absolute epsilon, as in the original recipe.
    """
    if (
        isinstance(projection_rms_floor, bool)
        or not math.isfinite(projection_rms_floor)
        or projection_rms_floor < 0
    ):
        raise ValueError("Nonnegative finite projection RMS floor required")
    mean = features.mean(dim=0)
    centered = features - mean
    scale_squared = centered.square().mean(dim=0) + RMS_EPSILON**2
    if projection_rms_floor:
        if (
            projection is None
            or projection.ndim != 2
            or projection.shape[0] != features.shape[1]
            or not bool(torch.isfinite(projection).all())
        ):
            raise ValueError(
                "One finite effective input projection per feature required"
            )
        scale_squared = (
            scale_squared + projection_rms_floor** 2 * projection.square().sum(dim=1)
        )
    scale = torch.sqrt(scale_squared)
    return centered / scale, mean, scale


def _model_standardized_features(model, x, projection_rms_floor=0.0):
    return standardized_features(
        model.features(x),
        projection=model.effective_projection() if projection_rms_floor else None,
        projection_rms_floor=projection_rms_floor,
    )


def reduced_objective(model, x, y, ridge, *, commit=False, projection_rms_floor=0.0):
    """Return complete penalized envelope objective and detached solve details."""
    x, y = _tensor(x, model.bias), _tensor(y, model.bias)
    standardized, mean, scale = _model_standardized_features(
        model, x, projection_rms_floor
    )
    beta, intercept, details = solve_effective_readout(standardized, y, ridge)
    mse = (standardized @ beta + intercept - y).square().mean()
    penalty = ridge * beta.square().sum()
    objective = mse + penalty
    if commit:
        raw_beta = beta / scale.detach()
        _commit(model, raw_beta, intercept - mean.detach() @ raw_beta)
    return objective, {
        **details,
        "mse": float(mse.detach()),
        "penalty": float(penalty),
        "objective": float(objective.detach()),
        "rms_epsilon": RMS_EPSILON,
        "training_feature_means": mean.detach().cpu().tolist(),
        "training_feature_scales": scale.detach().cpu().tolist(),
        "coefficient_coordinate": "TRAIN-centered, RMS-standardized feature coefficients; mean and scale differentiated.",
        **(
            {
                "projection_rms_floor": projection_rms_floor,
                "coefficient_coordinate": "TRAIN-centered coefficients scaled by sqrt(TRAIN variance + epsilon squared + tau squared times effective projection norm squared); the complete scale is differentiated.",
            }
            if projection_rms_floor
            else {}
        ),
    }


def joint_objective(model, x, y, ridge, *, projection_rms_floor=0.0):
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Strictly positive finite ridge required")
    x, y = _tensor(x, model.bias), _tensor(y, model.bias)
    mse = (model(x) - y).square().mean()
    _, _, scale = _model_standardized_features(model, x, projection_rms_floor)
    penalty = ridge * (model.effective_readout() * scale).square().sum()
    return mse + penalty, {
        "mse": float(mse.detach()),
        "penalty": float(penalty.detach()),
        "objective": float((mse + penalty).detach()),
    }


@dataclass(frozen=True)
class FitConfig:
    ridge: float = 1e-4
    steps: int = 100
    learning_rate: float = 0.5
    objective_scale: float = 1.0
    tolerance_grad: float = 1e-10
    tolerance_change: float = 1e-14
    history_size: int = 30
    projection_rms_floor: float = 0.0

    def __post_init__(self):
        if (
            isinstance(self.projection_rms_floor, bool)
            or not math.isfinite(self.projection_rms_floor)
            or self.projection_rms_floor < 0
        ):
            raise ValueError("Nonnegative finite projection RMS floor required")
        if min(self.steps, self.history_size) < 1:
            raise ValueError("Positive optimizer counts required")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.ridge,
                self.learning_rate,
                self.objective_scale,
                self.tolerance_grad,
                self.tolerance_change,
            )
        ):
            raise ValueError("Positive finite numerical settings required")


def _json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_state(model, path):
    metadata = {
        "capacities": model.capacities,
        "family": model.family,
        "architecture": model.architecture,
        "inputs_per_block": model.inputs_per_block,
        "threshold_mode": model.threshold_mode,
        "initialization_spread": model.initialization_spread,
        "initialization_receipt": model.initialization_receipt,
    }
    path = Path(path)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: value.detach().cpu().numpy()
                for name, value in model.state_dict().items()
            },
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stored_parameters": model.parameter_count,
    }


def load_state(path, device="cpu"):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata_json"]))
        model = RankBlockModel(
            metadata["capacities"],
            metadata["family"],
            metadata["architecture"],
            inputs_per_block=metadata["inputs_per_block"],
            initialization="random",
            initialization_spread=metadata["initialization_spread"],
            threshold_mode=metadata["threshold_mode"],
        )
        model.load_state_dict(
            {name: torch.from_numpy(arrays[name].copy()) for name in model.state_dict()}
        )
    model.initialization_receipt = metadata["initialization_receipt"]
    return model.to(device=device, dtype=torch.float64)


def _gradient_norm(parameters, scale=1.0):
    gradients = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError(
            "Nonfinite body/readout gradient; partial model is preserved"
        )
    result = (
        0.0
        if not gradients
        else float(torch.linalg.vector_norm(torch.cat(gradients))) / scale
    )
    if not math.isfinite(result):
        raise FloatingPointError(
            "Gradient norm overflow despite finite gradient entries; partial model is preserved"
        )
    return result


def geometry_diagnostics(model, x, *, projection_rms_floor=0.0):
    """Diagnostic ranks only; these thresholds never enter any fitted solve."""
    with torch.no_grad():
        features, _, _ = _model_standardized_features(model, x, projection_rms_floor)
        singular = torch.linalg.svdvals(features / math.sqrt(len(x)))
        cutoff = (
            torch.finfo(features.dtype).eps
            * max(features.shape)
            * float(singular.max())
        )
        projection = model.effective_projection()
        block_singular = [
            torch.linalg.svdvals(projection[model.group == block]).cpu().tolist()
            for block in range(len(model.capacities))
        ]
    return {
        "standardized_design_singular_values": singular.cpu().tolist(),
        "standardized_design_rank": int((singular > cutoff).sum()),
        "diagnostic_rank_cutoff": cutoff,
        "rank_cutoff_used_in_optimization": False,
        "effective_projection_block_singular_values": block_singular,
    }


def fit(model, x, y, config=None, *, optimizer="reduced", output_dir=None):
    """Train only on supplied scalar data; joint and reduced share ridge warm start.

    Saving/loading resumes parameters with fresh LBFGS history. This does not
    promise trajectory identity with an uninterrupted optimizer run.
    """
    config = config or FitConfig()
    objective_options = {"projection_rms_floor": config.projection_rms_floor}
    if optimizer not in ("reduced", "joint"):
        raise ValueError("Optimizer must be reduced or joint")
    x, y = _tensor(x, model.bias), _tensor(y, model.bias)
    if (
        x.ndim != 3
        or y.shape != (len(x),)
        or not bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    ):
        raise ValueError("Finite raw block inputs and scalar labels required")
    output = None if output_dir is None else Path(output_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial_effective = model.effective_projection().detach().clone()
    records, stages = [], []
    receipt = {
        "optimizer": optimizer,
        "config": asdict(config),
        "stored_parameters": model.parameter_count,
        "gradient_parameters": model.gradient_parameter_count,
        "initialization": model.initialization_receipt,
        "soma_gauge": "Soma fixed1 in both optimizers and still counted. Ridge is on standardized beta=(internal*soma/sqrt(K))*TRAIN_RMS; RMS floor1e-8. Centering and scale differentiate with the body.",
    }
    if config.projection_rms_floor:
        receipt["soma_gauge"] = (
            "Soma fixed1 and counted. Ridge is on raw effective readout times "
            "sqrt(TRAIN feature variance + epsilon squared + tau squared times "
            "effective projection norm squared). The full scale is differentiated; "
            "no new model slots or inference operation."
        )
    if output is not None:
        receipt["initial_state"] = save_state(model, output / "initial.npz")
    try:
        warm_objective, warm = reduced_objective(
            model, x, y, config.ridge, commit=True, **objective_options
        )
        stages.append({"stage": "common_ridge_warm_start", **warm})
        receipt["initial_geometry"] = geometry_diagnostics(
            model, x, **objective_options
        )
        if output is not None:
            receipt["warm_start_state"] = save_state(model, output / "warm_start.npz")
        parameters = (
            model.body_parameters()
            if optimizer == "reduced"
            else [p for p in model.parameters() if p.requires_grad]
        )
        opt = torch.optim.LBFGS(
            parameters,
            lr=config.learning_rate,
            max_iter=config.steps,
            max_eval=3 * config.steps,
            history_size=config.history_size,
            tolerance_grad=config.tolerance_grad,
            tolerance_change=config.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            model.zero_grad(set_to_none=True)
            objective, details = (
                reduced_objective(
                    model, x, y, config.ridge, commit=True, **objective_options
                )
                if optimizer == "reduced"
                else joint_objective(model, x, y, config.ridge, **objective_options)
            )
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Nonfinite full penalized training objective")
            (objective * config.objective_scale).backward()
            gradient = _gradient_norm(parameters, config.objective_scale)
            records.append(
                {
                    "evaluation": len(records),
                    "objective": details["objective"],
                    "mse": details["mse"],
                    "penalty": details["penalty"],
                    "unscaled_gradient_l2": gradient,
                    "coefficient_l2": float(model.effective_readout().detach().norm()),
                    "minimum_singular_value": details.get("minimum_singular_value"),
                }
            )
            return objective * config.objective_scale

        opt.step(closure)
        model.zero_grad(set_to_none=True)
        objective, terminal = (
            reduced_objective(
                model, x, y, config.ridge, commit=True, **objective_options
            )
            if optimizer == "reduced"
            else joint_objective(model, x, y, config.ridge, **objective_options)
        )
        objective.backward()
        terminal_gradient = _gradient_norm(parameters)
        model.zero_grad(set_to_none=True)
        stages.append({"stage": "terminal", **terminal})
        # Diagnostic optimal readout at the final body. Joint endpoints are never
        # replaced by this solve; the diagnostic cannot silently improve them.
        final_standardized, _, _ = _model_standardized_features(
            model, x, config.projection_rms_floor
        )
        _, _, terminal_design = solve_effective_readout(
            final_standardized, y, config.ridge
        )
        receipt.update(
            status="complete",
            stages=stages,
            closure_history=records,
            closure_calls=len(records),
            iterations=int(opt.state[parameters[0]].get("n_iter", 0)),
            terminal_unscaled_gradient_l2=terminal_gradient,
            effective_projection_movement_l2=float(
                (model.effective_projection().detach() - initial_effective).norm()
            ),
            parameter_movement_l2={
                name: float((p.detach() - initial[name]).norm())
                for name, p in model.named_parameters()
            },
            terminal_design_diagnostics=terminal_design,
            terminal_geometry=geometry_diagnostics(model, x, **objective_options),
            elapsed_seconds=time.monotonic() - start,
            warm_start_objective=float(warm_objective.detach()),
            terminal_objective=float(objective.detach()),
        )
        if output is not None:
            receipt["final_state"] = save_state(model, output / "final.npz")
            _json(output / "fit.json", receipt)
        return receipt
    except Exception as error:
        if output is not None:
            _json(
                output / "failure.json",
                {
                    "status": "failed",
                    "error": repr(error),
                    "stages": stages,
                    "closure_history": records,
                    "partial_state": save_state(model, output / "partial.npz"),
                },
            )
        raise


def fit_reduced(model, x, y, config=None, output_dir=None):
    return fit(model, x, y, config, optimizer="reduced", output_dir=output_dir)


def fit_joint(model, x, y, config=None, output_dir=None):
    return fit(model, x, y, config, optimizer="joint", output_dir=output_dir)


def assemble(local_models):
    """Exact sum of fitted one-block scalar functions with collapsed biases."""
    if not local_models or any(len(model.capacities) != 1 for model in local_models):
        raise ValueError("One-block models required")
    first = local_models[0]
    if any(
        (m.family, m.architecture, m.inputs_per_block)
        != (first.family, first.architecture, first.inputs_per_block)
        for m in local_models
    ):
        raise ValueError("Compatible architectures, families and dimensions required")
    result = RankBlockModel(
        [m.capacities[0] for m in local_models],
        first.family,
        first.architecture,
        first.inputs_per_block,
        initialization="random",
        initialization_spread=first.initialization_spread,
        threshold_mode=first.threshold_mode,
    )
    result.to(first.bias.device)
    with torch.no_grad():
        if first.rank == 0:
            result.projection.copy_(torch.cat([m.projection for m in local_models]))
        else:
            result.basis.copy_(torch.cat([m.basis for m in local_models]))
            result.branch_coefficients.copy_(
                torch.cat([m.branch_coefficients for m in local_models])
            )
        result.threshold.copy_(torch.cat([m.threshold for m in local_models]))
        result.internal_readout.copy_(
            torch.cat([m.effective_readout() for m in local_models])
            * math.sqrt(len(local_models))
        )
        result.soma_readout.fill_(1)
        result.bias.copy_(torch.stack([m.bias for m in local_models]).sum())
    result.initialization_receipt = {
        "method": "Exact sum-preserving assembly; local biases collapse into one global stored bias.",
        "local_initializations": [m.initialization_receipt for m in local_models],
    }
    return result
