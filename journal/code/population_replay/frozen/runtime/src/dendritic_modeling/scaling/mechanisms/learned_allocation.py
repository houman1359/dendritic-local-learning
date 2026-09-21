"""Learn internal features and allocate branches using scalar pilot observations.

Raw block supports are supplied equally. No Fourier dictionary, target feature,
target coefficient, or teacher complexity is an input to the allocation rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

CONDITIONS = ("rational_unequal", "rational_equal", "spline_unequal", "spline_equal")
FAMILIES = ("relu", "shunt")
POLICIES = ("uniform", "pilot", "shuffled_pilot")


@dataclass(frozen=True)
class LearnedAllocationConfig:
    condition: str = "rational_unequal"
    blocks: int = 8
    inputs_per_block: int = 4
    development_uniform_capacities: tuple[int, ...] = (1, 2, 4, 8, 12)
    confirmation_uniform_capacity: int = 16
    development_seeds: tuple[int, ...] = (4101, 4103)
    confirmation_seeds: tuple[int, ...] = (4201, 4203, 4207)
    pilot_capacities: tuple[int, int] = (1, 4)
    pilot_train_samples: int = 512
    pilot_validation_samples: int = 512
    train_samples: int = 2048
    validation_samples: int = 1024
    test_samples: int = 4096
    training_steps: int = 600
    pilot_steps: int = 600
    batch_size: int = 256
    learning_rate: float = 0.01
    final_learning_rate_fraction: float = 0.1
    initial_output_rms: float = 0.1
    allocation_floor_fraction: float = 0.05
    teacher_seed: int = 2026091900
    shuffle_seed: int = 2026091901

    def __post_init__(self):
        if (
            self.condition not in CONDITIONS
            or self.blocks != 8
            or self.inputs_per_block != 4
        ):
            raise ValueError(
                "This initial protocol fixes eight four-input blocks and four conditions"
            )
        if (
            not self.development_uniform_capacities
            or min(self.development_uniform_capacities) < 1
        ):
            raise ValueError("Positive development capacities required")
        if self.confirmation_uniform_capacity <= max(
            self.development_uniform_capacities
        ):
            raise ValueError("Larger confirmation capacity must be held out")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("Development and confirmation seeds must be distinct")
        if (
            len(self.pilot_capacities) != 2
            or self.pilot_capacities[0] >= self.pilot_capacities[1]
        ):
            raise ValueError("Two increasing pilot capacities required")
        if min(self.training_steps, self.pilot_steps, self.batch_size) < 1:
            raise ValueError("Positive fitting exposure required")

    def parameters(self, uniform_capacity):
        return (
            self.blocks * uniform_capacity * (self.inputs_per_block + 2)
            + self.blocks
            + 1
        )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class BlockTeacher:
    """Fixed scalar additive target; private components are not training labels."""

    def __init__(self, config):
        self.config = config
        self.kind = config.condition.split("_")[0]
        self.complexities = (
            [1, 1, 2, 2, 4, 8, 16, 32]
            if config.condition.endswith("unequal")
            else [8] * config.blocks
        )
        rng = np.random.default_rng(config.teacher_seed)
        projection = rng.normal(size=(config.blocks, config.inputs_per_block))
        self.projection = projection / np.abs(projection).sum(axis=1, keepdims=True)
        self.shapes = []
        for complexity in self.complexities:
            if self.kind == "rational":
                knots = (
                    np.linspace(-0.6, 0.6, complexity)
                    if complexity > 1
                    else np.array([0.0])
                )
                gains = np.full(complexity, 1.5 / (1.2 / max(complexity - 1, 1)))
                coefficients = (2 * (np.arange(complexity) % 2) - 1) * rng.uniform(
                    0.8, 1.2, complexity
                )
                coefficients *= rng.choice([-1, 1])
                self.shapes.append(
                    {"knots": knots, "gains": gains, "coefficients": coefficients}
                )
            else:
                knots = np.linspace(-0.65, 0.65, complexity + 2)
                values = (2 * (np.arange(complexity + 2) % 2) - 1) * rng.uniform(
                    0.8, 1.2, complexity + 2
                )
                values *= rng.choice([-1, 1])
                self.shapes.append({"knots": knots, "values": values})
        # Teacher definition only: fixed independent design sample defines
        # centering/scales. These component values are never learner inputs.
        x = np.random.default_rng(config.teacher_seed + 101).uniform(
            -1, 1, (16384, config.blocks, config.inputs_per_block)
        )
        values = self.raw_components(x)
        self.means = values.mean(axis=0)
        self.scales = values.std(axis=0)
        if np.any(self.scales < 1e-8):
            raise AssertionError("Degenerate teacher component")

    def raw_components(self, x):
        z = np.einsum("nbd,bd->nb", x, self.projection)
        components = []
        for block, shape in enumerate(self.shapes):
            if self.kind == "rational":
                excitation = np.maximum(
                    0.0, (z[:, block, None] - shape["knots"]) * shape["gains"]
                )
                values = (excitation / (1 + excitation)) @ shape["coefficients"]
            else:
                knots, values_at_knots = shape["knots"], shape["values"]
                slopes = np.diff(values_at_knots) / np.diff(knots)
                values = values_at_knots[0] + slopes[0] * (z[:, block] - knots[0])
                if len(slopes) > 1:
                    values = values + np.maximum(
                        0.0, z[:, block, None] - knots[1:-1]
                    ) @ np.diff(slopes)
            components.append(values)
        return np.column_stack(components)

    def evaluate(self, x):
        return np.sum(
            (self.raw_components(x) - self.means) / self.scales, axis=1
        ) / np.sqrt(self.config.blocks)

    def specification(self):
        return {
            "condition": self.config.condition,
            "projection": self.projection.tolist(),
            "complexities": self.complexities,
            "shapes": [
                {k: v.tolist() for k, v in shape.items()} for shape in self.shapes
            ],
            "component_means": self.means.tolist(),
            "component_scales": self.scales.tolist(),
            "scope": "Private target generator; only scalar summed labels and raw block inputs are supplied to learners.",
        }


class LearnedBlockNetwork(nn.Module):
    """Learned affine inputs, nonlinear branches, internal and soma readouts."""

    def __init__(self, capacities, family, inputs_per_block=4, seed=0):
        super().__init__()
        if family not in FAMILIES or min(capacities) < 1:
            raise ValueError("Valid family and at least one branch per block required")
        self.capacities = tuple(int(c) for c in capacities)
        self.family = family
        self.inputs_per_block = inputs_per_block
        group = torch.repeat_interleave(
            torch.arange(len(capacities)), torch.tensor(capacities)
        )
        self.register_buffer("group", group)
        generator = torch.Generator().manual_seed(seed)
        count = sum(capacities)
        self.projection = nn.Parameter(
            torch.randn(
                count, inputs_per_block, generator=generator, dtype=torch.float64
            )
            / np.sqrt(inputs_per_block)
        )
        self.threshold = nn.Parameter(
            torch.rand(count, generator=generator, dtype=torch.float64) * 0.8 - 0.4
        )
        scale = torch.tensor(capacities, dtype=torch.float64)[group].sqrt()
        self.internal_readout = nn.Parameter(
            torch.randn(count, generator=generator, dtype=torch.float64) * 0.2 / scale
        )
        self.soma_readout = nn.Parameter(
            torch.ones(len(capacities), dtype=torch.float64)
        )
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def components(self, x):
        pre = torch.sum(x[:, self.group, :] * self.projection, dim=-1) + self.threshold
        excitation = torch.relu(pre)
        nonlinear = (
            excitation if self.family == "relu" else excitation / (1 + excitation)
        )
        weighted = nonlinear * self.internal_readout
        blocks = torch.zeros(
            (len(x), len(self.capacities)), dtype=x.dtype, device=x.device
        )
        blocks.scatter_add_(1, self.group.expand(len(x), -1), weighted)
        return blocks * self.soma_readout / np.sqrt(len(self.capacities))

    def forward(self, x):
        return self.components(x).sum(dim=1) + self.bias

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def centered_pilot_gains(
    low_train,
    high_train,
    low_validation,
    high_validation,
    train_target,
    validation_target,
    floor_fraction=0.05,
):
    """Pure observation-only selector inputs; component origins cancel exactly."""
    low_mean, high_mean = low_train.mean(axis=0), high_train.mean(axis=0)
    # Fitted global offset is determined solely by pilot training observations.
    intercept = float(train_target.mean())
    baseline = (low_validation - low_mean).sum(axis=1) + intercept
    deltas = (high_validation - high_mean) - (low_validation - low_mean)
    residual = validation_target - baseline
    gains = np.mean(residual[:, None] ** 2 - (residual[:, None] - deltas) ** 2, axis=0)
    clipped = np.maximum(gains, 0.0)
    floor = floor_fraction * float(clipped.mean())
    weights = np.sqrt(clipped + floor) if clipped.sum() else np.ones_like(clipped)
    return {
        "low_train_component_means": low_mean.tolist(),
        "high_train_component_means": high_mean.tolist(),
        "training_intercept": intercept,
        "raw_validation_gains": gains.tolist(),
        "clipped_validation_gains": clipped.tolist(),
        "floor": floor,
        "weights": weights.tolist(),
        "all_nonpositive_fallback": bool(not clipped.sum()),
        "rule": "Prespecified floored square-root marginal-gain heuristic; no optimal-allocation theorem is asserted.",
    }


def allocate_capacities(total_branches, weights):
    weights = np.asarray(weights, dtype=float)
    if (
        total_branches < len(weights)
        or np.any(weights < 0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("Invalid branch budget/weights")
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    extra = (total_branches - len(weights)) * weights / weights.sum()
    integer = np.floor(extra).astype(int)
    remaining = total_branches - len(weights) - int(integer.sum())
    priority = np.lexsort((np.arange(len(weights)), -(extra - integer)))
    integer[priority[:remaining]] += 1
    capacities = integer + 1
    assert capacities.sum() == total_branches and np.all(capacities >= 1)
    return tuple(int(c) for c in capacities)


def fit_network(config, capacities, family, train_x, train_y, seed, steps):
    model = LearnedBlockNetwork(capacities, family, config.inputs_per_block, seed)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    with torch.no_grad():
        rms = model(x[: min(256, len(x))]).square().mean().sqrt()
        model.internal_readout.mul_(config.initial_output_rms / max(float(rms), 1e-12))
        initial_loss = float(torch.mean((model(x) - y) ** 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 887)
    traces = []
    clipped_steps = 0
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(len(x), (config.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[index])
        loss = torch.mean((prediction - y[index]) ** 2)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        clipped_steps += int(float(norm) > 10)
        factor = config.final_learning_rate_fraction + (
            1 - config.final_learning_rate_fraction
        ) * 0.5 * (1 + np.cos(np.pi * step / max(steps - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * factor
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            traces.append(
                {
                    "step": step + 1,
                    "batch_mse": float(loss.detach()),
                    "gradient_norm": float(norm),
                }
            )
    with torch.no_grad():
        final_loss = float(torch.mean((model(x) - y) ** 2))
    receipt = {
        "initial_train_mse": initial_loss,
        "terminal_train_mse": final_loss,
        "steps": steps,
        "sample_exposure": steps * config.batch_size,
        "clipped_steps": clipped_steps,
        "fit_seconds": time.perf_counter() - started,
        "trace": traces,
        "optimizer": "Adam, cosine learning-rate schedule, global clip10; fixed recipe, no endpoint selection",
    }
    return model, receipt


def _state_npz(path, model):
    np.savez(
        path,
        capacities=np.array(model.capacities),
        **{
            name: tensor.detach().numpy() for name, tensor in model.state_dict().items()
        },
    )


def _make_data(config, teacher, seed):
    names = ("pilot_train", "pilot_validation", "train", "validation", "test")
    sizes = (
        config.pilot_train_samples,
        config.pilot_validation_samples,
        config.train_samples,
        config.validation_samples,
        config.test_samples,
    )
    result = {}
    for name, size, child in zip(
        names, sizes, np.random.SeedSequence(seed).spawn(len(names))
    ):
        x = np.random.default_rng(child).uniform(
            -1, 1, (size, config.blocks, config.inputs_per_block)
        )
        result[name] = (x, teacher.evaluate(x))
    return result


def run_campaign(output_dir, config=None):
    config = config or LearnedAllocationConfig()
    if isinstance(config, dict):
        config = LearnedAllocationConfig(**config)
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen_plan.json").exists():
        raise FileExistsError("Campaign outputs are immutable; use a new path")
    started = time.perf_counter()
    teacher = BlockTeacher(config)
    write_json(output / "teacher_definition.json", teacher.specification())
    plan = {
        "schema": "learned_internal_allocation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "source_sha256": sha(__file__),
        "teacher_definition_sha256": sha(output / "teacher_definition.json"),
        "architecture": "Eight supplied four-coordinate blocks; branch learns affine input and threshold, ReLU excitation, optionally excitation/(1+excitation), internal coefficient, soma coefficient; global bias.",
        "whole_parameters": "P=(d+2)M+B+1=6M+9; every branch and soma remains connected; all compared policies/families share exact P.",
        "feature_scope": "Learned raw-input features with counted biased affine front end. This is not the earlier bias-free factory input recipe. No target dictionary supplied.",
        "selection_scope": "Pilot uses scalar labels only; training means remove block offsets; validation marginal gains set frozen heuristic weights. Teacher metadata and main train/validation/test are unavailable to selector.",
        "pilot_resource": "Each policy is charged both pilot fits and the same pilot observations. Pilots are physically computed once per family/sample seed and shared across policy comparisons; uniform does not use the resulting weights.",
        "comparison_scope": "Same observations, exact P, main updates, pilot opportunity and initialization rule. Initialization uses common output RMS but does not guarantee equal relative parameter updates.",
        "forecast_scope": "Fit descriptive log-error/log-P on development-seed small-budget validation errors, freeze predictions, then evaluate new confirmation seeds including larger held-out P.",
        "allocation_rule": "Minimum one branch per block, floor0.05 times mean positive pilot gain, square-root weights, largest-remainder integer budget; fixed shuffle for control. Heuristic, not proved optimum.",
        "no_test_selection": True,
        "families": FAMILIES,
        "policies": POLICIES,
    }
    write_json(output / "frozen_plan.json", plan)
    plan_hash = sha(output / "frozen_plan.json")
    datasets, selection, pilot_receipts, observation_receipts = {}, {}, [], []
    for seed in (*config.development_seeds, *config.confirmation_seeds):
        data = _make_data(config, teacher, seed)
        datasets[seed] = data
        np.savez(
            output / f"observations_s{seed}.npz",
            **{
                f"{name}_{key}": value
                for name, pair in data.items()
                for key, value in zip(("x", "y"), pair)
            },
        )
        observation_receipts.append(
            {
                "sample_seed": seed,
                "archive_sha256": sha(output / f"observations_s{seed}.npz"),
                "splits": {
                    name: {
                        "samples": len(pair[0]),
                        "inputs_sha256": hashlib.sha256(pair[0].tobytes()).hexdigest(),
                        "scalar_labels_sha256": hashlib.sha256(
                            pair[1].tobytes()
                        ).hexdigest(),
                    }
                    for name, pair in data.items()
                },
            }
        )
        for family in FAMILIES:
            models = []
            for capacity in config.pilot_capacities:
                model, receipt = fit_network(
                    config,
                    (capacity,) * config.blocks,
                    family,
                    *data["pilot_train"],
                    seed + 100000,
                    config.pilot_steps,
                )
                receipt.update(
                    {
                        "family": family,
                        "sample_seed": seed,
                        "capacity": capacity,
                        "parameters": model.parameter_count,
                    }
                )
                pilot_receipts.append(receipt)
                _state_npz(output / f"pilot_{family}_c{capacity}_s{seed}.npz", model)
                models.append(model)
            with torch.no_grad():
                low_train, high_train = [
                    m.components(torch.from_numpy(data["pilot_train"][0])).numpy()
                    for m in models
                ]
                low_val, high_val = [
                    m.components(torch.from_numpy(data["pilot_validation"][0])).numpy()
                    for m in models
                ]
            gains = centered_pilot_gains(
                low_train,
                high_train,
                low_val,
                high_val,
                data["pilot_train"][1],
                data["pilot_validation"][1],
                config.allocation_floor_fraction,
            )
            permutation = np.random.default_rng(config.shuffle_seed).permutation(
                config.blocks
            )
            gains.update(
                {
                    "family": family,
                    "sample_seed": seed,
                    "permutation": permutation.tolist(),
                    "capacities_by_uniform_capacity": {
                        str(c): {
                            "uniform": [c] * config.blocks,
                            "pilot": list(
                                allocate_capacities(c * config.blocks, gains["weights"])
                            ),
                            "shuffled_pilot": np.asarray(
                                allocate_capacities(c * config.blocks, gains["weights"])
                            )[permutation].tolist(),
                        }
                        for c in (
                            *config.development_uniform_capacities,
                            config.confirmation_uniform_capacity,
                        )
                    },
                }
            )
            selection[(seed, family)] = gains
        print(
            json.dumps(
                {
                    "condition": config.condition,
                    "pilots_finished_seed": seed,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    write_json(output / "pilot_fit_receipts.json", pilot_receipts)
    write_json(output / "observation_receipts.json", observation_receipts)
    write_json(output / "frozen_allocations.json", list(selection.values()))
    allocation_hash = sha(output / "frozen_allocations.json")
    rows = []

    def evaluate(phase, seeds, capacities):
        for capacity in capacities:
            for seed in seeds:
                data = datasets[seed]
                for family in FAMILIES:
                    for policy in POLICIES:
                        assigned = selection[(seed, family)][
                            "capacities_by_uniform_capacity"
                        ][str(capacity)][policy]
                        model, receipt = fit_network(
                            config,
                            assigned,
                            family,
                            *data["train"],
                            seed + 200000,
                            config.training_steps,
                        )
                        assert model.parameter_count == config.parameters(capacity)
                        row = {
                            "condition": config.condition,
                            "phase": phase,
                            "sample_seed": seed,
                            "family": family,
                            "policy": policy,
                            "uniform_capacity_reference": capacity,
                            "actual_parameters": model.parameter_count,
                            "total_branches": sum(assigned),
                            "capacities": assigned,
                            "supplied_block_support_indices": config.blocks
                            * config.inputs_per_block,
                            "pilot_fits_charged": 2,
                            "pilot_samples_charged": config.pilot_train_samples
                            + config.pilot_validation_samples,
                            "pilot_update_exposure_charged": 2
                            * config.pilot_steps
                            * config.batch_size,
                            **receipt,
                        }
                        with torch.no_grad():
                            for name in (
                                ("validation",)
                                if phase == "development"
                                else ("validation", "test")
                            ):
                                xx, yy = data[name]
                                prediction = model(torch.from_numpy(xx)).numpy()
                                row[name + "_mse"] = float(
                                    np.mean((prediction - yy) ** 2)
                                )
                                row[name + "_target_variance"] = float(np.var(yy))
                        filename = f"fit_{family}_{policy}_c{capacity}_s{seed}.npz"
                        _state_npz(output / filename, model)
                        row["state_file"] = filename
                        rows.append(row)
                        with (output / "results.jsonl").open("a") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "condition": config.condition,
                        "phase": phase,
                        "finished_capacity": capacity,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    evaluate(
        "development", config.development_seeds, config.development_uniform_capacities
    )
    predictions = []
    for family in FAMILIES:
        for policy in POLICIES:
            means = [
                np.mean(
                    [
                        r["validation_mse"]
                        for r in rows
                        if r["family"] == family
                        and r["policy"] == policy
                        and r["uniform_capacity_reference"] == c
                    ]
                )
                for c in config.development_uniform_capacities
            ]
            parameters = [
                config.parameters(c) for c in config.development_uniform_capacities
            ]
            slope, intercept = np.polyfit(np.log(parameters), np.log(means), 1)
            predictions.append(
                {
                    "family": family,
                    "policy": policy,
                    "development_validation_power": float(-slope),
                    "predicted_larger_budget_mse": float(
                        np.exp(intercept)
                        * config.parameters(config.confirmation_uniform_capacity)
                        ** slope
                    ),
                    "scope": "Descriptive extrapolation of validation endpoints, not an approximation exponent; forecast may fail.",
                }
            )
    write_json(
        output / "predictions_before_confirmation.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "confirmation_rows_evaluated": 0,
            "development_rows": len(rows),
            "frozen_allocations_sha256": allocation_hash,
            "predictions": predictions,
        },
    )
    evaluate(
        "confirmation",
        config.confirmation_seeds,
        (*config.development_uniform_capacities, config.confirmation_uniform_capacity),
    )
    summary = {
        "status": "completed",
        "condition": config.condition,
        "config": asdict(config),
        "main_fits": len(rows),
        "pilot_fits": len(pilot_receipts),
        "frozen_plan_sha256": plan_hash,
        "frozen_allocations_sha256": allocation_hash,
        "predictions_sha256": sha(output / "predictions_before_confirmation.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_jobs": 0,
        "scope": plan["feature_scope"],
        "selection_scope": plan["selection_scope"],
        "confirmation_larger_budget": [
            {
                "family": family,
                "policy": policy,
                "test_mse_mean": float(
                    np.mean(
                        [
                            r["test_mse"]
                            for r in rows
                            if r["phase"] == "confirmation"
                            and r["family"] == family
                            and r["policy"] == policy
                            and r["uniform_capacity_reference"]
                            == config.confirmation_uniform_capacity
                        ]
                    )
                ),
                "test_mse_each_seed": [
                    r["test_mse"]
                    for r in rows
                    if r["phase"] == "confirmation"
                    and r["family"] == family
                    and r["policy"] == policy
                    and r["uniform_capacity_reference"]
                    == config.confirmation_uniform_capacity
                ],
            }
            for family in FAMILIES
            for policy in POLICIES
        ],
    }
    assert sha(output / "frozen_plan.json") == plan_hash
    assert sha(output / "frozen_allocations.json") == allocation_hash
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Brief fitting timing canary; does not run a campaign",
    )
    args = parser.parse_args()
    config_dict = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.condition:
        config_dict["condition"] = args.condition
    config = LearnedAllocationConfig(**config_dict)
    torch.set_num_threads(1)
    if args.profile:
        teacher = BlockTeacher(config)
        data = _make_data(config, teacher, 4999)
        results = []
        for family in FAMILIES:
            _, receipt = fit_network(
                config, [121, 1, 1, 1, 1, 1, 1, 1], family, *data["train"], 4999, 100
            )
            results.append(
                {"family": family, "steps": 100, "fit_seconds": receipt["fit_seconds"]}
            )
        print(
            json.dumps({"profile": results, "source_sha256": sha(__file__)}, indent=2)
        )
        return
    if not args.output_dir:
        parser.error("--output-dir required unless --profile")
    print(json.dumps(run_campaign(args.output_dir, config), indent=2))


if __name__ == "__main__":
    main()
