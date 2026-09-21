"""Learned, parameter-matched radial-response mechanism experiments.

This small CPU experiment concerns a scalar target on a fixed interval. It does
not establish a language-model law or a general dendritic advantage. In
particular the generic divisive control is functionally identical to the shunt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

FAMILIES = (
    "homogeneous_relu",
    "biased_relu",
    "positive_shunt",
    "divisive_control",
)


@dataclass(frozen=True)
class RadialConfig:
    """A complete campaign prescription, frozen before development starts."""

    budgets: tuple[int, ...] = (7, 13, 25, 49, 97, 193)
    development_seeds: tuple[int, ...] = (1103, 1109)
    confirmation_seeds: tuple[int, ...] = (2203, 2207, 2213)
    learning_rates: tuple[float, ...] = (0.3, 1.0)
    steps: int = 180
    train_points: int = 257
    validation_points: int = 263
    test_points: int = 4096

    def validate(self) -> None:
        if not self.budgets or any(p < 7 or (p - 1) % 6 for p in self.budgets):
            raise ValueError("Every budget must have the form P=6k+1, k>=1.")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError("Budgets must be unique.")
        if not self.development_seeds or not self.confirmation_seeds:
            raise ValueError("Development and confirmation seeds are required.")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("Development and confirmation seeds must be disjoint.")
        if any(
            len(set(s)) != len(s)
            for s in (self.development_seeds, self.confirmation_seeds)
        ):
            raise ValueError("Seeds must be unique within each split.")
        if not self.learning_rates or any(lr <= 0 for lr in self.learning_rates):
            raise ValueError("Learning rates must be positive.")
        if (
            self.steps < 1
            or min(self.train_points, self.validation_points, self.test_points) < 2
        ):
            raise ValueError(
                "Positive steps and at least two points per split are required."
            )


def target(t: torch.Tensor) -> torch.Tensor:
    return t / (1.0 + t)


def make_grids(config: RadialConfig) -> dict[str, torch.Tensor]:
    """Disjoint deterministic grids; test also contains the two endpoints."""
    config.validate()
    grids = {
        "train": (torch.arange(config.train_points, dtype=torch.float64) + 1 / 3)
        / config.train_points,
        "validation": (
            torch.arange(config.validation_points, dtype=torch.float64) + 2 / 3
        )
        / config.validation_points,
        "test": torch.cat(
            (
                torch.tensor([0.0, 1.0], dtype=torch.float64),
                (torch.arange(config.test_points, dtype=torch.float64) + 0.5)
                / config.test_points,
            )
        ),
    }
    combined = torch.cat(list(grids.values()))
    if torch.unique(combined).numel() != combined.numel():
        raise ValueError("These grid sizes cause coincident samples across splits.")
    return grids


class RadialModel(nn.Module):
    """All free coefficients are registered parameters; there is no padding.

    ReLU: sum_j a_j ReLU(w_j t + b_j) + c, with b_j absent in
    the homogeneous family. Shunt/divisive: sum_j a_j e_j/(1+e_j)+c,
    e_j=exp(w_j)t. Every family has exactly the requested P=6k+1.
    """

    def __init__(self, family: str, budget: int, seed: int):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        if budget < 7 or (budget - 1) % 6:
            raise ValueError("Budget must have the form P=6k+1, k>=1.")
        self.family = family
        self.width = (budget - 1) // (3 if family == "biased_relu" else 2)
        generator = torch.Generator().manual_seed(seed)

        def rand() -> torch.Tensor:
            return torch.rand(self.width, generator=generator, dtype=torch.float64)

        if family in ("positive_shunt", "divisive_control"):
            weights = (2.0 * rand() - 1.0) * math.log(4.0)
        else:
            weights = 0.8 + 0.4 * rand()
        self.input_weight = nn.Parameter(weights)
        if family == "biased_relu":
            # Target-independent stratification covers the interval. One unit
            # starts active everywhere so an affine trend is not excluded.
            knots = (
                torch.arange(self.width, dtype=torch.float64) + rand()
            ) / self.width
            knots[0] = -0.1
            self.input_bias = nn.Parameter(-weights * knots)
        else:
            self.register_parameter("input_bias", None)
        self.output_weight = nn.Parameter(
            torch.randn(self.width, generator=generator, dtype=torch.float64)
            / math.sqrt(self.width)
        )
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        if parameter_count(self) != budget:
            raise AssertionError("Parameter counting contract violated.")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.family in ("positive_shunt", "divisive_control"):
            excitation = t[..., None] * self.input_weight.exp()
            features = excitation / (1.0 + excitation)
        else:
            preactivation = t[..., None] * self.input_weight
            if self.input_bias is not None:
                preactivation = preactivation + self.input_bias
            features = torch.relu(preactivation)
        return features @ self.output_weight + self.output_bias


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(model: nn.Module, grid: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        error = model(grid) - target(grid)
        return {
            "mse": float(error.square().mean()),
            "max_abs_error": float(error.abs().max()),
        }


def fit_model(
    family: str,
    budget: int,
    seed: int,
    learning_rate: float,
    config: RadialConfig,
    grids: dict[str, torch.Tensor],
) -> tuple[RadialModel, list[dict], dict]:
    """Optimize all coefficients; this function never reads the test grid."""
    model = RadialModel(family, budget, seed)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        max_iter=1,
        max_eval=20,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    trace = []
    closure_calls = 0

    def record(step: int) -> None:
        trace.append(
            {
                "step": step,
                "closure_calls": closure_calls,
                "train_mse": _metrics(model, grids["train"])["mse"],
                "validation_mse": _metrics(model, grids["validation"])["mse"],
            }
        )

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = (model(grids["train"]) - target(grids["train"])).square().mean()
        loss.backward()
        return loss

    record(0)
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        optimizer.step(closure)
        record(step)
    summary = {
        "family": family,
        "budget": budget,
        "actual_parameters": parameter_count(model),
        "seed": seed,
        "learning_rate": learning_rate,
        "steps": config.steps,
        "closure_calls": closure_calls,
        "initial_train_mse": trace[0]["train_mse"],
        "initial_validation_mse": trace[0]["validation_mse"],
        "terminal_train_mse": trace[-1]["train_mse"],
        "terminal_validation_mse": trace[-1]["validation_mse"],
        "parameter_change_l2": math.sqrt(
            sum(
                float((p.detach() - initial[name]).square().sum())
                for name, p in model.named_parameters()
            )
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    if not all(
        math.isfinite(row["train_mse"]) and math.isfinite(row["validation_mse"])
        for row in trace
    ):
        raise FloatingPointError("Nonfinite optimization trajectory.")
    return model, trace, summary


def interpolation_oracle(budget: int, grid: torch.Tensor) -> dict[str, float]:
    """A constructive H-piece spline using H ReLUs, all 3H+1 slots counted.

    This is an explicit target-informed construction, never a trained result.
    The first ReLU realizes t on [0,1]; the rest realize interior hinges.
    """
    width = (budget - 1) // 3
    if 3 * width + 1 != budget or width < 1:
        raise ValueError("Spline budget must equal 3H+1.")
    knots = torch.linspace(0, 1, width + 1, dtype=torch.float64)
    values = target(knots)
    slopes = (values[1:] - values[:-1]) * width
    coefficients = torch.cat((slopes[:1], slopes[1:] - slopes[:-1]))
    prediction = torch.relu(grid[..., None] - knots[:-1]) @ coefficients
    error = prediction - target(grid)
    return {
        "budget": budget,
        "width": width,
        "registered_parameter_slots": 3 * width + 1,
        "mse": float(error.square().mean()),
        "max_abs_error": float(error.abs().max()),
        "uniform_error_lower_bound_any_H_piece_spline": 1 / (64 * width**2),
        "uniform_error_interpolation_upper_bound": 1 / (4 * width**2),
    }


def run_campaign(output_dir: str | Path, config: RadialConfig | None = None) -> dict:
    """Run development, freeze validation selection, then score confirmation."""
    config = config or RadialConfig()
    config.validate()
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prespecification.json").exists():
        raise FileExistsError(
            "Use a new output directory; campaign evidence is immutable."
        )
    grids = make_grids(config)
    started = time.perf_counter()
    source = Path(__file__).resolve()
    _write_json(
        output / "prespecification.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "source_path": str(source),
            "source_sha256": _sha(source),
            "target": "t/(1+t), t uniform on [0,1]",
            "dtype": "torch.float64",
            "device": "cpu",
            "torch_threads": 1,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "families": FAMILIES,
            "parameter_matching": "P=6k+1; biased ReLU H=2k; other families S=3k. Every registered learnable coefficient counted; no padding.",
            "selection": "Per family and budget, select LR by mean terminal validation MSE across development seeds. Tie break smaller LR. Confirmation uses terminal step, never test-selected checkpoint.",
            "prediction": {
                "homogeneous_relu": "Approximation MSE stays above the exact affine population floor at every P.",
                "biased_relu": "The constructive interpolation oracle has MSE O(P^-4); learned optimization can fail to attain this rate.",
                "positive_shunt": "Exact representation exists at 3 parameters; larger matched models need optimization, with no asserted learned exponent.",
                "divisive_control": "Exact formula and initialization equivalence predicts bitwise-identical learned traces to the shunt.",
            },
            "interpretation": "Restricted radial representation experiment, not a dendrite-specific advantage over matched arithmetic or a full-network scaling result.",
            "confirmation_status": "Not started; development selection has not yet occurred.",
        },
    )
    np.savez(
        output / "grids.npz", **{name: value.numpy() for name, value in grids.items()}
    )
    development = []
    confirmation = []
    selection: dict[str, dict[str, float]] = {}
    for split, seeds in (
        ("development", config.development_seeds),
        ("confirmation", config.confirmation_seeds),
    ):
        for family in FAMILIES:
            for budget in config.budgets:
                rates = (
                    config.learning_rates
                    if split == "development"
                    else (selection[family][str(budget)],)
                )
                for rate in rates:
                    for seed in seeds:
                        model, trace, row = fit_model(
                            family, budget, seed, rate, config, grids
                        )
                        row["split"] = split
                        run_dir = (
                            output / split / family / f"p{budget}_seed{seed}_lr{rate:g}"
                        )
                        run_dir.mkdir(parents=True)
                        _write_csv(run_dir / "trace.csv", trace)
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "family": family,
                                "budget": budget,
                                "seed": seed,
                            },
                            run_dir / "terminal.pt",
                        )
                        if split == "confirmation":
                            row.update(
                                {
                                    f"test_{key}": value
                                    for key, value in _metrics(
                                        model, grids["test"]
                                    ).items()
                                }
                            )
                            confirmation.append(row)
                        else:
                            development.append(row)
                        _write_json(run_dir / "metrics.json", row)
                print(f"radial {split}: {family} P={budget} finished", flush=True)
        if split == "development":
            _write_csv(output / "development_summary.csv", development)
            selection = {}
            candidates = []
            for family in FAMILIES:
                selection[family] = {}
                for budget in config.budgets:
                    for rate in config.learning_rates:
                        losses = [
                            row["terminal_validation_mse"]
                            for row in development
                            if row["family"] == family
                            and row["budget"] == budget
                            and row["learning_rate"] == rate
                        ]
                        candidates.append(
                            {
                                "family": family,
                                "budget": budget,
                                "learning_rate": rate,
                                "mean_terminal_validation_mse": float(np.mean(losses)),
                            }
                        )
                    selected = min(
                        (
                            row
                            for row in candidates
                            if row["family"] == family and row["budget"] == budget
                        ),
                        key=lambda row: (
                            row["mean_terminal_validation_mse"],
                            row["learning_rate"],
                        ),
                    )
                    selection[family][str(budget)] = selected["learning_rate"]
            _write_json(
                output / "frozen_selection.json",
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "prespecification_sha256": _sha(output / "prespecification.json"),
                    "development_summary_sha256": _sha(
                        output / "development_summary.csv"
                    ),
                    "confirmation_status": "Not started; no test outcomes evaluated.",
                    "selected_learning_rates": selection,
                    "candidates": candidates,
                },
            )
    _write_csv(output / "confirmation_summary.csv", confirmation)
    oracles = [interpolation_oracle(p, grids["test"]) for p in config.budgets]
    _write_csv(output / "constructive_interpolation_oracle.csv", oracles)
    analytic = {
        "homogeneous_best_affine_uniform_error": (3 - 2 * math.sqrt(2)) / 4,
        "homogeneous_best_affine_population_mse": 36 * math.log(2)
        - 28 * math.log(2) ** 2
        - 11.5,
        "homogeneous_best_affine_slope": 18 * math.log(2) - 12,
        "homogeneous_best_affine_bias": 7 - 10 * math.log(2),
        "shunt_minimal_exact_real_parameter_count": 3,
        "shunt_exact_parameters": {
            "raw_input_weight": 0.0,
            "output_weight": 1.0,
            "output_bias": 0.0,
        },
        "theoretical_interpolation_mse_upper_rate": "O(P^-4); an upper construction, not a fitted learned exponent",
    }
    _write_json(output / "analytic_reference.json", analytic)
    equivalence = []
    for budget in config.budgets:
        for seed in config.confirmation_seeds:
            left = next(
                row
                for row in confirmation
                if row["family"] == "positive_shunt"
                and row["budget"] == budget
                and row["seed"] == seed
            )
            right = next(
                row
                for row in confirmation
                if row["family"] == "divisive_control"
                and row["budget"] == budget
                and row["seed"] == seed
            )
            left_dir = (
                output
                / "confirmation"
                / "positive_shunt"
                / f"p{budget}_seed{seed}_lr{left['learning_rate']:g}"
            )
            right_dir = (
                output
                / "confirmation"
                / "divisive_control"
                / f"p{budget}_seed{seed}_lr{right['learning_rate']:g}"
            )
            equivalence.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "same_selected_lr": left["learning_rate"] == right["learning_rate"],
                    "bitwise_identical_trace": (left_dir / "trace.csv").read_bytes()
                    == (right_dir / "trace.csv").read_bytes(),
                    "same_test_metrics": all(
                        left[key] == right[key]
                        for key in ("test_mse", "test_max_abs_error")
                    ),
                }
            )
    receipt = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "development_runs": len(development),
        "confirmation_runs": len(confirmation),
        "source_unchanged": _sha(source)
        == json.loads((output / "prespecification.json").read_text())["source_sha256"],
        "all_counts_match": all(
            row["budget"] == row["actual_parameters"]
            for row in development + confirmation
        ),
        "all_models_optimized": all(
            row["parameter_change_l2"] > 0 for row in development + confirmation
        ),
        "all_terminal_training_losses_improved": all(
            row["terminal_train_mse"] < row["initial_train_mse"]
            for row in development + confirmation
        ),
        "generic_control_equivalence": equivalence,
        "test_mse_definition": "Mean squared error on 4096 midpoint samples plus both endpoints (or configured point count). Deterministic quadrature, not IID statistical replicates.",
        "artifact_sha256": {
            str(path.relative_to(output)): _sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, help="JSON object containing RadialConfig fields."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Two budgets and reduced steps for a smoke experiment.",
    )
    args = parser.parse_args()
    if args.quick and args.config:
        parser.error("--quick and --config are mutually exclusive")
    config = (
        RadialConfig(
            budgets=(7, 13),
            steps=30,
            development_seeds=(1103,),
            confirmation_seeds=(2203,),
        )
        if args.quick
        else RadialConfig()
    )
    if args.config:
        payload = json.loads(args.config.read_text())
        for name in (
            "budgets",
            "development_seeds",
            "confirmation_seeds",
            "learning_rates",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        config = RadialConfig(**payload)
    run_campaign(args.output_dir, config)


if __name__ == "__main__":
    main()
