"""Smooth block-mixture tasks and observation-only, train-only fitting primitives.

This module does not select allocations or inspect validation/test data. Teacher
components, intervals and directions are private generation/audit information.
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
CONDITIONS = ("equal_range", "mixed_range")
RECIPES = ("joint", "readout_ls_then_joint", "alternating_ls_then_joint")


@dataclass(frozen=True)
class TaskConfig:
    condition: str = "mixed_range"
    teacher_seed: int = 2026091301
    blocks: int = 4
    inputs_per_block: int = 3
    input_center: float = 0.5
    perpendicular_radius: float = 1 / math.sqrt(3)
    quadrature_order: int = 512

    def __post_init__(self):
        if (
            self.condition not in CONDITIONS
            or self.blocks != 4
            or self.inputs_per_block != 3
        ):
            raise ValueError(
                "The fixed task has four three-input blocks and two conditions"
            )
        if not 0 <= self.input_center <= 1 or self.perpendicular_radius <= 0:
            raise ValueError("Invalid input distribution")
        if self.quadrature_order < 64:
            raise ValueError("At least 64 deterministic normalization nodes required")


def _as_tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


class SmoothBlockTask:
    """Only ``sample`` and scalar ``evaluate`` are learner-facing interfaces."""

    def __init__(self, config: TaskConfig | None = None):
        self.config = config or TaskConfig()
        rng = np.random.default_rng(self.config.teacher_seed)
        directions = rng.normal(size=(4, 3))
        self._directions = directions / np.linalg.norm(
            directions, axis=1, keepdims=True
        )
        self._intervals = np.array(
            [[0.25, 4.0]] * 4
            if self.config.condition == "equal_range"
            else [[1.0, 2.0], [0.25, 4.0], [0.05, 4.0], [0.01, 4.0]]
        )
        nodes, weights = np.polynomial.legendre.leggauss(self.config.quadrature_order)
        values = self._raw_mixtures((nodes[:, None] + 1) / 2)
        self._means = weights @ values / 2
        self._scales = np.sqrt(weights @ ((values - self._means) ** 2) / 2)

    def _raw_mixtures(self, t):
        low, high = self._intervals.T
        return t * np.log1p((high - low) / (low + t)) / (high - low)

    def _coordinates(self, x):
        values = np.asarray(x, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1:] != (4, 3)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Expected finite inputs with shape [n,4,3]")
        longitudinal = np.einsum("nkd,kd->nk", values, self._directions)
        t = longitudinal + self.config.input_center
        perpendicular = values - longitudinal[..., None] * self._directions
        if (
            np.any(t < -1e-10)
            or np.any(t > 1 + 1e-10)
            or np.any(
                np.linalg.norm(perpendicular, axis=2)
                > self.config.perpendicular_radius + 1e-10
            )
        ):
            raise ValueError(
                "Scalar query lies outside the admissible block product domain"
            )
        return t

    def components(self, x):
        """PRIVATE audit/generation values; never pass these to the selector."""
        t = self._coordinates(x)
        return _as_tensor((self._raw_mixtures(t) - self._means) / self._scales / 2)

    def evaluate(self, x):
        """Scalar labels for admissible ordinary or paired-intervention queries."""
        return self.components(x).sum(dim=1)

    def sample(self, seed, n):
        x = self.sample_inputs(seed, n)
        return x, self.evaluate(x)

    def sample_inputs(self, seed, n):
        """Generate raw inputs without requesting or computing scalar labels."""
        if n < 1:
            raise ValueError("Positive sample count required")
        # Each observation has its own fixed random packet. The same seed gives
        # identical prefixes when sample counts or parameter budgets change.
        rng = np.random.default_rng(seed)
        uniform = rng.uniform(size=(n, 4, 5))
        # Box-Muller supplies an isotropic 3D normal before orthogonal projection.
        normal = np.stack(
            (
                np.sqrt(-2 * np.log(np.maximum(uniform[..., 1], 1e-300)))
                * np.cos(2 * np.pi * uniform[..., 2]),
                np.sqrt(-2 * np.log(np.maximum(uniform[..., 1], 1e-300)))
                * np.sin(2 * np.pi * uniform[..., 2]),
                np.sqrt(-2 * np.log(np.maximum(uniform[..., 3], 1e-300)))
                * np.cos(2 * np.pi * uniform[..., 4]),
            ),
            axis=-1,
        )
        perpendicular = (
            normal
            - np.einsum("nkd,kd->nk", normal, self._directions)[..., None]
            * self._directions
        )
        perpendicular /= np.maximum(
            np.linalg.norm(perpendicular, axis=2, keepdims=True), 1e-300
        )
        # Independent radial random packets preserve T/perpendicular independence.
        radial = np.random.default_rng(
            np.random.SeedSequence([int(seed), 817])
        ).uniform(size=(n, 4, 1))
        perpendicular *= self.config.perpendicular_radius * np.sqrt(radial)
        return _as_tensor(
            (uniform[..., 0] - self.config.input_center)[..., None] * self._directions
            + perpendicular
        )

    def specification(self):
        return {
            "config": asdict(self.config),
            "directions": self._directions.tolist(),
            "intervals": self._intervals.tolist(),
            "component_means": self._means.tolist(),
            "component_scales": self._scales.tolist(),
            "target": "Sum of centered unit-variance positive uniform conductance mixtures divided by sqrt(4).",
            "distribution": "X_i=(T_i-input_center)u_i+V_i; T_i uniform[0,1]; V_i independent uniform perpendicular disk.",
            "scope": "Private generator/audit information; only raw block inputs and summed scalar labels reach training/selection.",
        }


def observational_directions(x, y, rcond=1e-12):
    """Train-only scalar-label OLS; supports both one-block and global fitting."""
    x, y = _as_tensor(x), _as_tensor(y)
    centered_x = x - x.mean(dim=0)
    centered_y = y - y.mean()
    matrix = centered_x.reshape(len(x), -1)
    solution = torch.linalg.lstsq(matrix, centered_y, rcond=rcond, driver="gelsd")
    coefficients = solution.solution.reshape(x.shape[1:])
    norms = coefficients.norm(dim=1, keepdim=True)
    if bool((norms < 1e-14).any()):
        raise ValueError(
            "Scalar training regression has a zero direction; choose explicit random initialization"
        )
    return coefficients / norms, {
        "method": "Joint OLS of centered scalar training labels on all supplied raw blocks; extract block coefficients.",
        "train_rows": len(x),
        "rank": int(solution.rank),
        "rcond": rcond,
        "coefficient_norms": norms[:, 0].tolist(),
    }


class SmoothBlockModel(nn.Module):
    """All affine, internal readout, soma and global bias slots remain free."""

    def __init__(
        self,
        capacities,
        family,
        seed=0,
        train_data=None,
        initialization="observational",
        inputs_per_block=3,
        threshold_mode="quantile",
    ):
        super().__init__()
        if (
            family not in FAMILIES
            or not capacities
            or any(int(c) != c or c < 1 for c in capacities)
        ):
            raise ValueError("Valid family and positive integer capacities required")
        if initialization not in ("observational", "random"):
            raise ValueError("Unknown initialization")
        if threshold_mode not in ("quantile", "midpoint") or (
            family != "tanh" and threshold_mode != "quantile"
        ):
            raise ValueError(
                "Midpoint versus quantile threshold search is a tanh-only extra control"
            )
        if initialization == "observational" and train_data is None:
            raise ValueError("Observational initialization requires only training x,y")
        self.capacities = tuple(int(c) for c in capacities)
        self.family, self.inputs_per_block = family, inputs_per_block
        self.threshold_mode = threshold_mode
        generator = torch.Generator().manual_seed(seed)
        count, blocks = sum(capacities), len(capacities)
        self.register_buffer(
            "group",
            torch.repeat_interleave(torch.arange(blocks), torch.tensor(capacities)),
        )
        if initialization == "observational":
            x, y = map(_as_tensor, train_data)
            if x.shape[1:] != (blocks, inputs_per_block):
                raise ValueError("Training shape and architecture do not match")
            directions, receipt = observational_directions(x, y)
        else:
            directions = torch.randn(
                blocks, inputs_per_block, generator=generator, dtype=torch.float64
            )
            directions /= directions.norm(dim=1, keepdim=True)
            x = None if train_data is None else _as_tensor(train_data[0])
            receipt = {
                "method": "One random unit direction per block, freely trainable branch copies.",
                "train_rows": 0 if x is None else len(x),
            }
        projection, threshold = [], []
        for block, capacity in enumerate(capacities):
            if x is None:
                low, high = -0.5, 0.5
                quantiles = torch.linspace(
                    low, high, capacity + 1, dtype=torch.float64
                )[:-1]
            else:
                projected = x[:, block] @ directions[block]
                low, high = float(projected.min()), float(projected.max())
                quantiles = torch.quantile(
                    projected,
                    (torch.arange(capacity, dtype=torch.float64) + 0.25) / capacity,
                )
            span = max(high - low, 1e-6)
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
            projection.append(scale[:, None] * directions[block])
            threshold.append(-scale * knots)
        self.projection = nn.Parameter(torch.cat(projection))
        self.threshold = nn.Parameter(torch.cat(threshold))
        self.internal_readout = nn.Parameter(
            torch.randn(count, generator=generator, dtype=torch.float64)
            / math.sqrt(count)
        )
        self.soma_readout = nn.Parameter(torch.ones(blocks, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.initialization_receipt = {
            "initialization": initialization,
            "seed": seed,
            "threshold_mode": threshold_mode,
            **receipt,
            "feature_initialization": "Training projection range/quantiles; shunt anchor below observed minimum, ReLU observed quantiles; tanh quantile or midpoint as explicitly configured. No private teacher constants.",
        }
        assert self.parameter_count == (inputs_per_block + 2) * count + blocks + 1

    def features(self, x):
        pre = (x[:, self.group] * self.projection).sum(dim=2) + self.threshold
        if self.family == "tanh":
            return torch.tanh(pre)
        excitation = torch.relu(pre)
        return excitation if self.family == "relu" else excitation / (1 + excitation)

    def components(self, x):
        weighted = self.features(x) * self.internal_readout
        result = x.new_zeros((len(x), len(self.capacities)))
        result.scatter_add_(1, self.group.expand(len(x), -1), weighted)
        return result * self.soma_readout / math.sqrt(len(self.capacities))

    def forward(self, x):
        return self.components(x).sum(dim=1) + self.bias

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class FitConfig:
    recipe: str = "readout_ls_then_joint"
    lbfgs_steps: int = 150
    learning_rate: float = 0.5
    objective_scale: float = 1e6
    tolerance_grad: float = 1e-10
    tolerance_change: float = 1e-18
    history_size: int = 30
    ls_rcond: float = 1e-10
    alternating_rounds: int = 2
    alternating_body_steps: int = 30

    def __post_init__(self):
        if (
            self.recipe not in RECIPES
            or min(
                self.lbfgs_steps,
                self.history_size,
                self.alternating_rounds,
                self.alternating_body_steps,
            )
            < 1
        ):
            raise ValueError("Unknown recipe or nonpositive fitting exposure")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.learning_rate,
                self.objective_scale,
                self.tolerance_grad,
                self.tolerance_change,
                self.ls_rcond,
            )
        ):
            raise ValueError("Positive finite numerical settings required")


def evaluate(model, x, y):
    x, y = _as_tensor(x), _as_tensor(y)
    with torch.no_grad():
        residual = model(x) - y
        return {
            "mse": float(residual.square().mean()),
            "max_abs_error": float(residual.abs().max()),
        }


def readout_least_squares(model, x, y, rcond=1e-10):
    """Solve branch effective coefficients/intercept using training data only."""
    x, y = _as_tensor(x), _as_tensor(y)
    with torch.no_grad():
        matrix = torch.cat(
            (model.features(x), torch.ones((len(x), 1), dtype=x.dtype)), dim=1
        )
        solution = torch.linalg.lstsq(matrix, y, rcond=rcond, driver="gelsd")
        # Normalize only the redundant soma/internal factorization. Every slot
        # remains an independent free parameter in subsequent joint fitting.
        model.internal_readout.copy_(
            solution.solution[:-1] * math.sqrt(len(model.capacities))
        )
        model.soma_readout.fill_(1)
        model.bias.copy_(solution.solution[-1])
        return {
            "stage": "readout_ls",
            "rank": int(solution.rank),
            "columns": matrix.shape[1],
            "rcond": rcond,
            "coefficient_l2": float(solution.solution.norm()),
            **evaluate(model, x, y),
        }


def fit(model, x, y, config: FitConfig | None = None):
    """Fit to these scalar TRAIN observations only, retaining all stage endpoints."""
    config = config or FitConfig()
    x, y = _as_tensor(x), _as_tensor(y)
    if (
        x.ndim != 3
        or y.shape != (len(x),)
        or not bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    ):
        raise ValueError("Finite scalar training observations required")
    start = time.monotonic()
    reference = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    stages = [{"stage": "initial", **evaluate(model, x, y)}]
    closures = []
    model.zero_grad(set_to_none=True)
    (model(x) - y).square().mean().backward()
    initial_gradient = math.sqrt(
        sum(
            float(value.grad.square().sum())
            for value in model.parameters()
            if value.grad is not None
        )
    )
    model.zero_grad(set_to_none=True)

    def optimize(stage, parameters, steps):
        body_start = (
            model.projection.detach().clone(),
            model.threshold.detach().clone(),
        )
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=config.learning_rate,
            max_iter=steps,
            max_eval=steps * 3,
            history_size=config.history_size,
            tolerance_grad=config.tolerance_grad,
            tolerance_change=config.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            model.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean()
            scaled = loss * config.objective_scale
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError(
                    "Nonfinite training objective; preserve this failed outcome"
                )
            scaled.backward()
            closures.append({"stage": stage, "train_mse": float(loss.detach())})
            return scaled

        optimizer.step(closure)
        first_parameter = next(iter(optimizer.param_groups[0]["params"]))
        state = optimizer.state[first_parameter]
        stages.append(
            {
                "stage": stage,
                "iterations": int(state.get("n_iter", 0)),
                "closure_calls": int(state.get("func_evals", 0)),
                "stage_body_movement_l2": math.sqrt(
                    float(
                        (model.projection.detach() - body_start[0]).square().sum()
                        + (model.threshold.detach() - body_start[1]).square().sum()
                    )
                ),
                **evaluate(model, x, y),
            }
        )

    if config.recipe != "joint":
        stages.append(readout_least_squares(model, x, y, config.ls_rcond))
    if config.recipe == "alternating_ls_then_joint":
        for index in range(config.alternating_rounds):
            optimize(
                f"body_lbfgs_{index}",
                [model.projection, model.threshold],
                config.alternating_body_steps,
            )
            stages.append(readout_least_squares(model, x, y, config.ls_rcond))
    optimize("joint_lbfgs", list(model.parameters()), config.lbfgs_steps)
    model.zero_grad(set_to_none=True)
    terminal_loss = (model(x) - y).square().mean()
    terminal_loss.backward()
    movement = {
        name: float((value.detach() - reference[name]).norm())
        for name, value in model.named_parameters()
    }
    initial_norms = {name: float(value.norm()) for name, value in reference.items()}
    gradient = math.sqrt(
        sum(
            float(value.grad.square().sum())
            for value in model.parameters()
            if value.grad is not None
        )
    )
    model.zero_grad(set_to_none=True)
    return {
        "config": asdict(config),
        "initialization": model.initialization_receipt,
        "parameters": model.parameter_count,
        "train_rows": len(x),
        "stages": stages,
        "closure_history": closures,
        "optimizer_closure_calls": len(closures),
        "initial_train_mse": stages[0]["mse"],
        "terminal_train_mse": float(terminal_loss.detach()),
        "parameter_movement_l2": movement,
        "parameter_initial_norm_l2": initial_norms,
        "body_movement_l2": math.sqrt(
            movement["projection"] ** 2 + movement["threshold"] ** 2
        ),
        "initial_unscaled_gradient_l2": initial_gradient,
        "terminal_unscaled_gradient_l2": gradient,
        "elapsed_seconds": time.monotonic() - start,
        "scope": "Training-only optimizer; fixed objective multiplier, no output clipping or validation/test access. LS rank truncation is numerical and all learned slots remain counted/free.",
    }


def assemble_blocks(local_models):
    """Preserve the exact sum of scalar K=1 model outputs, including intercepts."""
    if not local_models or any(len(model.capacities) != 1 for model in local_models):
        raise ValueError("One-block fitted models required")
    family, dimension = local_models[0].family, local_models[0].inputs_per_block
    if any(
        model.family != family or model.inputs_per_block != dimension
        for model in local_models
    ):
        raise ValueError("All local models must use the same family/input dimension")
    result = SmoothBlockModel(
        [model.capacities[0] for model in local_models],
        family,
        initialization="random",
        inputs_per_block=dimension,
    )
    with torch.no_grad():
        result.projection.copy_(torch.cat([model.projection for model in local_models]))
        result.threshold.copy_(torch.cat([model.threshold for model in local_models]))
        result.internal_readout.copy_(
            torch.cat([model.internal_readout for model in local_models])
            * math.sqrt(len(local_models))
        )
        result.soma_readout.copy_(
            torch.cat([model.soma_readout for model in local_models])
        )
        result.bias.copy_(torch.stack([model.bias for model in local_models]).sum())
    result.initialization_receipt = {
        "method": "Exact sum-preserving assembly of separately fitted scalar one-block models.",
        "local_initializations": [
            model.initialization_receipt for model in local_models
        ],
    }
    result.threshold_mode = "assembled"
    return result


def fit_intercept(model, x, y):
    """Fit only the global offset to scalar training observations."""
    with torch.no_grad():
        model.bias.add_((_as_tensor(y) - model(_as_tensor(x))).mean())
    return evaluate(model, x, y)


def save_state(model, path):
    path = Path(path)
    metadata = {
        "family": model.family,
        "capacities": model.capacities,
        "inputs_per_block": model.inputs_per_block,
        "threshold_mode": model.threshold_mode,
        "initialization": model.initialization_receipt,
    }
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: tensor.detach().cpu().numpy()
                for name, tensor in model.state_dict().items()
            },
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": model.parameter_count,
    }


def load_state(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        mode = metadata.get("threshold_mode", "quantile")
        model = SmoothBlockModel(
            metadata["capacities"],
            metadata["family"],
            initialization="random",
            inputs_per_block=metadata["inputs_per_block"],
            threshold_mode="quantile" if mode == "assembled" else mode,
        )
        model.load_state_dict(
            {name: torch.as_tensor(data[name].copy()) for name in model.state_dict()}
        )
        model.initialization_receipt = metadata["initialization"]
        model.threshold_mode = mode
    return model
