"""Learn three fixed conductance-range targets with equally calibrated controls.

All features in the learned arms start without teacher quadrature information.
The deep ReLU comparator uses its actual parameter count under each ceiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
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
    "deep_relu",
    "biased_tanh",
)
RECIPES = ("joint", "readout_ls_then_joint")
TARGETS = {"narrow": (1.0, 2.0), "medium": (0.25, 4.0), "broad": (0.01, 4.0)}
GAIN_SPANS = {"moderate": (0.25, 4.0), "wide": (0.25, 100.0)}


@dataclass(frozen=True)
class Config:
    budgets: tuple[int, ...] = (7, 13, 19, 25, 37, 49, 73, 97, 145, 193)
    targets: tuple[str, ...] = tuple(TARGETS)
    gain_spans: tuple[str, ...] = tuple(GAIN_SPANS)
    heldout_budget: int = 385
    development_seeds: tuple[int, ...] = (5101, 5107)
    confirmation_seeds: tuple[int, ...] = (6101, 6107, 6113)
    learning_rates: tuple[float, ...] = (0.3, 1.0)
    recipes: tuple[str, ...] = RECIPES
    steps: int = 400
    train_points: int = 513
    validation_points: int = 521
    test_points: int = 8192
    objective_scale: float = 1e6
    tolerance_grad: float = 1e-12
    tolerance_change: float = 1e-20
    ls_rcond: float = 1e-12
    selection_floor: float = 1e-24

    def validate(self):
        for values, supported, label in (
            (self.targets, TARGETS, "targets"),
            (self.gain_spans, GAIN_SPANS, "gain spans"),
        ):
            if (
                not values
                or len(values) != len(set(values))
                or not set(values) <= set(supported)
            ):
                raise ValueError(f"Invalid {label}")
        if not self.budgets or sorted(set(self.budgets)) != list(self.budgets):
            raise ValueError("Development budgets must be increasing and unique")
        if any(p < 7 or (p - 1) % 6 for p in (*self.budgets, self.heldout_budget)):
            raise ValueError("All ceilings must equal6k+1 with k>=1")
        if self.heldout_budget <= max(self.budgets):
            raise ValueError("Held-out budget must exceed every development budget")
        if not self.development_seeds or not self.confirmation_seeds:
            raise ValueError("Both seed splits are required")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("Development and confirmation seeds must be disjoint")
        if any(
            len(set(s)) != len(s)
            for s in (self.development_seeds, self.confirmation_seeds)
        ):
            raise ValueError("Seeds must be unique within each split")
        if set(self.recipes) != set(RECIPES) or len(self.recipes) != 2:
            raise ValueError("Both prescribed recipes must be retained")
        if not self.learning_rates or any(
            lr <= 0 or not math.isfinite(lr) for lr in self.learning_rates
        ):
            raise ValueError("Positive finite learning rates required")
        if (
            self.steps < 1
            or min(self.train_points, self.validation_points, self.test_points) < 2
        ):
            raise ValueError("Positive steps and grid sizes required")
        if (
            min(
                self.objective_scale,
                self.tolerance_grad,
                self.tolerance_change,
                self.ls_rcond,
                self.selection_floor,
            )
            <= 0
        ):
            raise ValueError("Numerical settings must be positive")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def csv_write(path, records):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def target(x, target_id):
    a, b = TARGETS[target_id]
    return x * torch.log1p((b - a) / (x + a)) / (b - a)


def grids(config):
    result = {
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
    values = torch.cat(list(result.values()))
    if len(values) != torch.unique(values).numel():
        raise ValueError("Configured grids overlap")
    return result


def geometry(family, ceiling):
    if family not in FAMILIES or ceiling < 7 or (ceiling - 1) % 6:
        raise ValueError("Unsupported family or budget")
    if family == "deep_relu":
        width = math.isqrt(ceiling + 3) - 2
        actual = width**2 + 4 * width + 1
    else:
        width = (ceiling - 1) // (3 if family in ("biased_relu", "biased_tanh") else 2)
        actual = ceiling
    return width, actual


class Model(nn.Module):
    def __init__(self, family, ceiling, seed, gain_span):
        super().__init__()
        self.family = family
        self.width, self.actual_parameters = geometry(family, ceiling)
        generator = torch.Generator().manual_seed(seed)

        def uniform():
            return torch.rand(self.width, generator=generator, dtype=torch.float64)

        low, high = GAIN_SPANS[gain_span]
        log_gain = math.log(low) + uniform() * math.log(high / low)
        w = (
            log_gain
            if family in ("positive_shunt", "divisive_control")
            else log_gain.exp()
        )
        self.input_weight = nn.Parameter(w)
        if family in ("biased_relu", "biased_tanh", "deep_relu"):
            knot = (
                torch.arange(self.width, dtype=torch.float64) + uniform()
            ) / self.width
            knot[0] = 0.0
            self.input_bias = nn.Parameter(-w * knot)
        else:
            self.register_parameter("input_bias", None)
        if family == "deep_relu":
            hidden = torch.randn(
                self.width, self.width, generator=generator, dtype=torch.float64
            ) / math.sqrt(self.width)
            self.hidden_weight = nn.Parameter(hidden)
            # Target-free domain centering prevents every second-layer feature
            # from starting dead. The offsets remain freely learned parameters.
            domain = torch.linspace(0, 1, 65, dtype=torch.float64)
            pre = torch.relu(domain[:, None] * w + self.input_bias.detach()) @ hidden.T
            self.hidden_bias = nn.Parameter(-0.5 * (pre.amin(0) + pre.amax(0)))
        else:
            self.register_parameter("hidden_weight", None)
            self.register_parameter("hidden_bias", None)
        self.output_weight = nn.Parameter(
            torch.randn(self.width, generator=generator, dtype=torch.float64)
            / math.sqrt(self.width)
        )
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        if sum(p.numel() for p in self.parameters()) != self.actual_parameters:
            raise AssertionError("Whole parameter count mismatch")

    def features(self, x):
        if self.family in ("positive_shunt", "divisive_control"):
            # Identical to g*x/(1+g*x), evaluated without gain overflow.
            # At x=0, log(x)=-inf and sigmoid is exactly zero.
            return torch.sigmoid(x[:, None].log() + self.input_weight)
        z = x[:, None] * self.input_weight
        if self.input_bias is not None:
            z = z + self.input_bias
        z = torch.tanh(z) if self.family == "biased_tanh" else torch.relu(z)
        if self.hidden_weight is not None:
            z = torch.relu(z @ self.hidden_weight.T + self.hidden_bias)
        return z

    def forward(self, x):
        return self.features(x) @ self.output_weight + self.output_bias


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def movement(model, reference):
    return math.sqrt(
        sum(
            float((p.detach() - reference[name]).square().sum())
            for name, p in model.named_parameters()
        )
    )


def metrics(model, x, target_id):
    with torch.no_grad():
        error = model(x) - target(x, target_id)
        return {
            "mse": float(error.square().mean()),
            "max_abs_error": float(error.abs().max()),
        }


def gradient_metrics(model, x, target_id):
    loss = (model(x) - target(x, target_id)).square().mean()
    grad = torch.autograd.grad(loss, tuple(model.parameters()))
    return (
        float(loss.detach()),
        math.sqrt(sum(float(g.square().sum()) for g in grad)),
        max(float(g.abs().max()) for g in grad),
    )


def least_squares_readout(model, x, rcond, target_id, apply=True):
    with torch.no_grad():
        design = torch.cat((model.features(x), torch.ones(len(x), 1, dtype=x.dtype)), 1)
        solved = torch.linalg.lstsq(
            design, target(x, target_id)[:, None], rcond=rcond, driver="gelsd"
        )
        coefficient = solved.solution[:, 0]
        predicted = design @ coefficient
        singular = solved.singular_values
        information = {
            "rcond": rcond,
            "rank": int(solved.rank),
            "columns": design.shape[1],
            "singular_values": [float(s) for s in singular],
            "readout_l2": float(coefficient.norm()),
            "readout_max_abs": float(coefficient.abs().max()),
            "train_mse": float((predicted - target(x, target_id)).square().mean()),
        }
        if apply:
            model.output_weight.copy_(coefficient[:-1])
            model.output_bias.copy_(coefficient[-1])
    return information


class FitError(RuntimeError):
    """Retain the scientific state when a fit cannot complete."""

    def __init__(self, reason, model, initial, prepared, trace):
        super().__init__(reason)
        self.model = model
        self.initial = initial
        self.prepared = prepared
        self.trace = trace


def fit(target_id, family, ceiling, seed, gain_span, recipe, lr, config, data):
    """No access to the test grid; every trainable coefficient stays free."""
    if recipe not in RECIPES:
        raise ValueError("Unknown recipe")
    if target_id not in config.targets or gain_span not in config.gain_spans:
        raise ValueError("Unsupported target/span")
    model = Model(family, ceiling, seed, gain_span)
    initial = clone_state(model)
    initial_loss, initial_grad, initial_max_grad = gradient_metrics(
        model, data["train"], target_id
    )
    initial_validation = metrics(model, data["validation"], target_id)["mse"]
    ls = None
    if recipe == "readout_ls_then_joint":
        try:
            ls = least_squares_readout(model, data["train"], config.ls_rcond, target_id)
        except Exception as error:
            raise FitError(
                str(error), model, initial, clone_state(model), []
            ) from error
    prepared = clone_state(model)
    stage_movement = movement(model, initial)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=1,
        max_eval=20,
        history_size=50,
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        line_search_fn="strong_wolfe",
    )
    calls = 0
    trace = []

    def record(step):
        loss, grad, max_grad = gradient_metrics(model, data["train"], target_id)
        trace.append(
            {
                "step": step,
                "optimizer_closure_calls": calls,
                "train_mse": loss,
                "validation_mse": metrics(model, data["validation"], target_id)["mse"],
                "unscaled_gradient_l2": grad,
                "unscaled_gradient_max_abs": max_grad,
            }
        )

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = (
            config.objective_scale
            * (model(data["train"]) - target(data["train"], target_id)).square().mean()
        )
        loss.backward()
        return loss

    started = time.perf_counter()
    try:
        record(0)
        for step in range(1, config.steps + 1):
            optimizer.step(closure)
            record(step)
        if not all(
            math.isfinite(row[key])
            for row in trace
            for key in ("train_mse", "validation_mse", "unscaled_gradient_l2")
        ):
            raise FloatingPointError("Nonfinite optimization trajectory")
    except Exception as error:
        raise FitError(str(error), model, initial, prepared, trace) from error
    summary = {
        "target_id": target_id,
        "gain_span": gain_span,
        "family": family,
        "parameter_ceiling": ceiling,
        "actual_parameters": model.actual_parameters,
        "width": model.width,
        "exact_ceiling_match": model.actual_parameters == ceiling,
        "seed": seed,
        "recipe": recipe,
        "learning_rate": lr,
        "steps": config.steps,
        "initial_train_mse": initial_loss,
        "initial_validation_mse": initial_validation,
        "initial_gradient_l2": initial_grad,
        "initial_gradient_max_abs": initial_max_grad,
        "post_readout_train_mse": trace[0]["train_mse"],
        "post_readout_validation_mse": trace[0]["validation_mse"],
        "post_readout_gradient_l2": trace[0]["unscaled_gradient_l2"],
        "terminal_train_mse": trace[-1]["train_mse"],
        "terminal_validation_mse": trace[-1]["validation_mse"],
        "terminal_gradient_l2": trace[-1]["unscaled_gradient_l2"],
        "readout_stage_parameter_change_l2": stage_movement,
        "joint_stage_parameter_change_l2": movement(model, prepared),
        "total_parameter_change_l2": movement(model, initial),
        "optimizer_closure_calls": calls,
        "monitoring_gradient_evaluations": len(trace) + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "ls_rank": None if ls is None else ls["rank"],
        "ls_rcond": None if ls is None else ls["rcond"],
        "ls_readout_l2": None if ls is None else ls["readout_l2"],
        "ls_readout_max_abs": None if ls is None else ls["readout_max_abs"],
        "terminal_readout_l2": float(
            torch.cat(
                (model.output_weight.detach(), model.output_bias.detach()[None])
            ).norm()
        ),
        "terminal_readout_max_abs": max(
            float(model.output_weight.detach().abs().max()),
            abs(float(model.output_bias.detach())),
        ),
    }
    return model, initial, prepared, trace, summary, ls


def config_from_file(path):
    values = json.loads(Path(path).read_text())
    config = Config(**values)
    config.validate()
    return config


def initialize(output, config_path):
    config = config_from_file(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("Campaign protocol is immutable")
    data = grids(config)
    np.savez(
        output / "grids.npz", **{key: value.numpy() for key, value in data.items()}
    )
    dump(
        output / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha(__file__),
            "config_sha256": sha(config_path),
            "grids_sha256": sha(output / "grids.npz"),
            "target_formula": "t*log1p((b-a)/(t+a))/(b-a), fixed per target across every budget",
            "target_intervals": TARGETS,
            "gain_spans": GAIN_SPANS,
            "families": FAMILIES,
            "no_teacher_quadrature_information": True,
            "dtype": "float64",
            "device": "cpu",
            "threads": 1,
            "initializer": "Common log-uniform gain spans for every family, independent of target ID. Biased families use target-free stratified thresholds with first threshold0. Deep second-layer biases center feature ranges on65 domain points, without labels.",
            "stable_shunt_evaluation": "sigmoid(log(g)+log(t)), mathematically g*t/(1+g*t); identical generic-control operations, exactly0 at t=0. All raw gain parameters remain free.",
            "recipes": "Joint LBFGS, or training-only rank-revealing least-squares readout initialization then free joint LBFGS. Every learned slot counted in both stages.",
            "selection": "Equal2spans x2recipes x2LRs x2developmentseeds at every development budget/family. Minimize mean terminal validation MSE floored at1e-24; ties follow span order then recipe order then smaller LR. No test or heldout-budget development.",
            "prediction": "Heldout385 inherits P193 span/recipe/LR. Before confirmation, fit the last3development points: constant for homogeneous; log-log for ReLU; log-linear in actualP for shunt/divisive/tanh. Clip predicted MSE below at the declared numerical interpretation floor. Predictions are heuristics, not asserted theorem exponents.",
            "deep_comparator": "Two equal-width hidden layers, actualP=H^2+4H+1 under each ceiling; use actualP in plots and ratios. The shallow ReLU lower bound does not apply.",
            "generic_control": "Divisive and shunt expressions and initialization are identical; duplicated runs check arithmetic equivalence and are not independent evidence.",
            "readout_sensitivity": "At heldout385 only, record training-only LS fits for rcond1e-10,1e-12,1e-14 on initial features, with no span, recipe, or rcond reselection.",
            "failure_policy": "Any failed fit retains initial/prepared/failed-terminal checkpoints and available trace; family fails and global selection is blocked. No silent retries or cell replacement.",
            "confirmation": "Fresh fixed seeds; terminal checkpoint only after frozen selection. All selected-family/budget outcomes retained; no guarantee every stage improves.",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    )


def load_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_sha256"] != sha(__file__):
        raise ValueError("Campaign source differs from frozen protocol")
    if protocol["grids_sha256"] != sha(output / "grids.npz"):
        raise ValueError("Campaign grids differ from frozen protocol")
    config = Config(**protocol["config"])
    config.validate()
    data = {
        key: torch.from_numpy(value)
        for key, value in np.load(output / "grids.npz").items()
    }
    return protocol, config, data


def run_key(ceiling, seed, gain_span, recipe, lr):
    return f"p{ceiling}_seed{seed}_{gain_span}_{recipe}_lr{lr:g}"


def run_family(output, target_id, family, stage):
    torch.set_num_threads(1)
    output = Path(output)
    _, config, data = load_protocol(output)
    if (
        target_id not in config.targets
        or family not in FAMILIES
        or stage not in ("development", "confirmation")
    ):
        raise ValueError("Unsupported target/family/stage")
    stage_dir = output / stage / target_id / family
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite family evidence")
    selection = None
    if stage == "confirmation":
        selection = json.loads((output / "frozen_selection.json").read_text())
        if selection["protocol_sha256"] != sha(output / "protocol.json"):
            raise ValueError("Selection/protocol binding differs")
    stage_dir.mkdir(parents=True, exist_ok=True)
    ceilings = (
        (*config.budgets, config.heldout_budget)
        if stage == "confirmation"
        else config.budgets
    )
    seeds = (
        config.confirmation_seeds
        if stage == "confirmation"
        else config.development_seeds
    )
    records = []
    started = time.perf_counter()
    for ceiling in ceilings:
        if stage == "development":
            choices = [
                (span, recipe, lr)
                for span in config.gain_spans
                for recipe in config.recipes
                for lr in config.learning_rates
            ]
        else:
            choice = selection["selected"][target_id][family][str(ceiling)]
            choices = [(choice["gain_span"], choice["recipe"], choice["learning_rate"])]
        for span, recipe, lr in choices:
            for seed in seeds:
                run = stage_dir / run_key(ceiling, seed, span, recipe, lr)
                run.mkdir()
                try:
                    model, initial, prepared, trace, summary, ls = fit(
                        target_id, family, ceiling, seed, span, recipe, lr, config, data
                    )
                except FitError as error:
                    for name, state in (
                        ("initial", error.initial),
                        ("post_readout", error.prepared),
                        ("failed_terminal", clone_state(error.model)),
                    ):
                        torch.save(
                            {
                                "state_dict": state,
                                "target_id": target_id,
                                "gain_span": span,
                                "family": family,
                                "parameter_ceiling": ceiling,
                                "actual_parameters": error.model.actual_parameters,
                            },
                            run / f"{name}.pt",
                        )
                    if error.trace:
                        csv_write(run / "failed_trace.csv", error.trace)
                    dump(
                        run / "failure.json",
                        {
                            "status": "incomplete_fit",
                            "reason": str(error),
                            "source_sha256": sha(__file__),
                            "protocol_sha256": sha(output / "protocol.json"),
                            "policy": "Preserve evidence; stop this family and block selection. No silent cell replacement.",
                        },
                    )
                    raise
                summary["stage"] = stage
                summary["heldout_budget"] = ceiling == config.heldout_budget
                for name, state in (
                    ("initial", initial),
                    ("post_readout", prepared),
                    ("terminal", clone_state(model)),
                ):
                    torch.save(
                        {
                            "state_dict": state,
                            "target_id": target_id,
                            "gain_span": span,
                            "family": family,
                            "parameter_ceiling": ceiling,
                            "actual_parameters": model.actual_parameters,
                        },
                        run / f"{name}.pt",
                    )
                csv_write(run / "trace.csv", trace)
                if ls is not None:
                    dump(run / "least_squares.json", ls)
                if stage == "confirmation":
                    summary.update(
                        {
                            "test_" + key: value
                            for key, value in metrics(
                                model, data["test"], target_id
                            ).items()
                        }
                    )
                    if ceiling == config.heldout_budget:
                        initial_model = Model(family, ceiling, seed, span)
                        initial_model.load_state_dict(initial)
                        sensitivity = [
                            least_squares_readout(
                                initial_model,
                                data["train"],
                                rcond,
                                target_id,
                                apply=False,
                            )
                            for rcond in (1e-10, 1e-12, 1e-14)
                        ]
                        dump(run / "heldout_ls_sensitivity.json", sensitivity)
                dump(run / "metrics.json", summary)
                records.append(summary)
        print(
            f"range {stage}: {target_id}/{family} ceiling={ceiling} finished",
            flush=True,
        )
    csv_write(stage_dir / "summary.csv", records)
    dump(
        stage_dir / "receipt.json",
        {
            "target_id": target_id,
            "family": family,
            "stage": stage,
            "runs": len(records),
            "elapsed_seconds": time.perf_counter() - started,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_sha256": sha(__file__),
            "protocol_sha256": sha(output / "protocol.json"),
            "selection_sha256": (
                None if selection is None else sha(output / "frozen_selection.json")
            ),
            "artifacts_sha256": {
                str(path.relative_to(stage_dir)): sha(path)
                for path in sorted(stage_dir.rglob("*"))
                if path.is_file()
            },
        },
    )


def select_target(output, target_id, config):
    selected, candidates, predictions, bindings = {}, [], {}, {}
    for family in FAMILIES:
        directory = output / "development" / target_id / family
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["protocol_sha256"] != sha(output / "protocol.json"):
            raise ValueError("Development protocol binding mismatch")
        with (directory / "summary.csv").open() as handle:
            records = list(csv.DictReader(handle))
        if sha(directory / "summary.csv") != receipt["artifacts_sha256"]["summary.csv"]:
            raise ValueError("Development summary hash mismatch")
        bindings[family] = sha(directory / "receipt.json")
        selected[family] = {}
        means = []
        for ceiling in config.budgets:
            current = []
            for span in config.gain_spans:
                for recipe in config.recipes:
                    for lr in config.learning_rates:
                        found = [
                            r
                            for r in records
                            if int(r["parameter_ceiling"]) == ceiling
                            and r["gain_span"] == span
                            and r["recipe"] == recipe
                            and float(r["learning_rate"]) == lr
                        ]
                        if sorted(int(r["seed"]) for r in found) != sorted(
                            config.development_seeds
                        ):
                            raise ValueError("Incomplete development cell")
                        mean = float(
                            np.mean(
                                [float(r["terminal_validation_mse"]) for r in found]
                            )
                        )
                        row = {
                            "target_id": target_id,
                            "family": family,
                            "parameter_ceiling": ceiling,
                            "gain_span": span,
                            "recipe": recipe,
                            "learning_rate": lr,
                            "mean_validation_mse": mean,
                            "selection_score": max(mean, config.selection_floor),
                        }
                        current.append(row)
                        candidates.append(row)
            chosen = min(
                current,
                key=lambda r: (
                    r["selection_score"],
                    config.gain_spans.index(r["gain_span"]),
                    config.recipes.index(r["recipe"]),
                    r["learning_rate"],
                ),
            )
            selected[family][str(ceiling)] = chosen
            means.append((geometry(family, ceiling)[1], chosen["selection_score"]))
        selected[family][str(config.heldout_budget)] = {
            **selected[family][str(max(config.budgets))],
            "parameter_ceiling": config.heldout_budget,
            "inherited_from_ceiling": max(config.budgets),
        }
        actual = geometry(family, config.heldout_budget)[1]
        last = means[-min(3, len(means)) :]
        x = np.array([r[0] for r in last], dtype=float)
        y = np.log([r[1] for r in last])
        if family == "homogeneous_relu":
            predicted, method, slope = (
                last[-1][1],
                "constant last development budget",
                0.0,
            )
        else:
            logarithmic = family in ("biased_relu", "deep_relu")
            slope, intercept = np.polyfit(np.log(x) if logarithmic else x, y, 1)
            log_prediction = (
                slope * (math.log(actual) if logarithmic else actual) + intercept
            )
            predicted = max(config.selection_floor, math.exp(min(log_prediction, 700)))
            method = "log-log last3" if logarithmic else "log-linear last3"
        predictions[family] = {
            "parameter_ceiling": config.heldout_budget,
            "actual_parameters": actual,
            "predicted_mse": predicted,
            "method": method,
            "descriptive_fitted_slope": float(slope),
            "fitted_development_points": last,
            "floor": config.selection_floor,
            "interpretation": "Heuristic frozen before any confirmation score, not a claimed learned asymptotic law.",
        }
    return selected, candidates, predictions, bindings


def select(output):
    output = Path(output)
    _, config, _ = load_protocol(output)
    if (output / "frozen_selection.json").exists():
        raise FileExistsError("Selection is immutable")
    selected, candidates, predictions, bindings = {}, [], {}, {}
    for target_id in config.targets:
        selected[target_id], found, predictions[target_id], bindings[target_id] = (
            select_target(output, target_id, config)
        )
        candidates.extend(found)
    dump(
        output / "frozen_selection.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha(output / "protocol.json"),
            "development_receipt_sha256": bindings,
            "confirmation_status": "Not started; no test scores consulted.",
            "selected": selected,
            "candidates": candidates,
            "heldout_predictions": predictions,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("initialize", "development", "select", "confirmation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    if args.stage == "initialize":
        if args.config is None:
            parser.error("initialize requires --config")
        initialize(args.output_dir, args.config)
    elif args.stage == "select":
        select(args.output_dir)
    else:
        if args.target is None or args.family is None:
            parser.error("family stages require --target and --family")
        run_family(args.output_dir, args.target, args.family, args.stage)


if __name__ == "__main__":
    main()
