"""Frozen structured-query allocation campaign; no private data in the selector."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .smooth_block import (
        FitConfig,
        SmoothBlockModel,
        SmoothBlockTask,
        TaskConfig,
        assemble_blocks,
        evaluate,
        fit,
        fit_intercept,
        load_state,
        save_state,
    )
else:
    from smooth_block import (
        FitConfig,
        SmoothBlockModel,
        SmoothBlockTask,
        TaskConfig,
        assemble_blocks,
        evaluate,
        fit,
        fit_intercept,
        load_state,
        save_state,
    )


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def default_config():
    return {
        "teacher_seed": 2026092301,
        "conditions": ["equal_range", "mixed_range"],
        "families": ["shunt", "relu", "tanh"],
        "development_seeds": [5101, 5107],
        "confirmation_seeds": [6101, 6103, 6107],
        "local_train_n": 512,
        "selector_n": 1024,
        "report_n": 1024,
        "test_n": 4096,
        "capacity_grid": [1, 2, 3, 4, 6, 8, 12, 16, 24],
        "development_budgets": [4, 8, 12, 16, 24, 32],
        "withheld_budget": 48,
        "diagnostic_budgets": [16, 48],
        "shuffle": [2, 0, 3, 1],
        "recipes": ["readout_ls_then_joint", "alternating_ls_then_joint"],
        "steps": [150, 600],
        "learning_rate": 0.5,
        "tanh_modes": ["quantile", "midpoint"],
        "descriptive_error_threshold": 1e-18,
        "policies": [
            "uniform",
            "validation_selected",
            "shuffled_selected",
            "small_pilot",
            "certificate_oracle",
        ],
        "scope": "Structured scalar-query training; unknown projections; IID selector and reporting/test. No main endpoint refit after selection.",
    }


def choices(config, family):
    result = []
    for recipe, steps, mode in itertools.product(
        config["recipes"],
        config["steps"],
        config["tanh_modes"] if family == "tanh" else ["quantile"],
    ):
        result.append(
            {
                "id": f"{recipe}_s{steps}_{mode}",
                "threshold_mode": mode,
                "fit": asdict(
                    FitConfig(
                        recipe=recipe,
                        lbfgs_steps=steps,
                        learning_rate=config["learning_rate"],
                    )
                ),
            }
        )
    return result


def candidates(total, grid):
    """Uniform first, then lexicographic; floor is half the uniform share."""
    if total % 4:
        raise ValueError("Whole budgets must be divisible by four")
    uniform = (total // 4,) * 4
    feasible = [
        s
        for s in itertools.product(grid, repeat=4)
        if sum(s) == total and min(s) >= math.ceil(total / 8)
    ]
    if uniform not in feasible:
        raise ValueError("Uniform must be present")
    return [uniform, *sorted(s for s in feasible if s != uniform)]


def choose_observational(predictions, y, anchor_y, allocations):
    """TRAIN-fixed local predictions and IID scalar validation labels only.

    Direct residuals avoid catastrophic cancellation of Gram-expanded risks.
    Equal losses use the declared allocation order (uniform first).
    """
    y = np.asarray(y)
    risks = []
    for allocation in allocations:
        prediction = np.full_like(y, float(anchor_y))
        for block, capacity in enumerate(allocation):
            prediction += predictions[(block, capacity)]
        risks.append(float(np.mean((prediction - y) ** 2)))
    if not np.isfinite(risks).all():
        raise FloatingPointError("Nonfinite candidate validation risk")
    selected = int(np.argmin(risks))
    return allocations[selected], {
        "candidate_allocations": allocations,
        "candidate_mse": risks,
        "selected_index": selected,
        "candidate_count": len(allocations),
    }


def pilot_weights(predictions, train_means, y, anchor_y):
    low = np.column_stack([predictions[(i, 1)] for i in range(4)])
    high = np.column_stack([predictions[(i, 4)] for i in range(4)])
    lo_mean = np.array([train_means[(i, 1)] for i in range(4)])
    hi_mean = np.array([train_means[(i, 4)] for i in range(4)])
    # Means are estimated on each block's own TRAIN marginal sample.
    baseline = (low - lo_mean).sum(axis=1) + float(anchor_y) + lo_mean.sum()
    delta = high - hi_mean - (low - lo_mean)
    residual = np.asarray(y) - baseline
    gains = np.mean(residual[:, None] ** 2 - (residual[:, None] - delta) ** 2, axis=0)
    positive = np.maximum(gains, 0)
    weights = (
        np.sqrt(positive + 0.05 * positive.mean()) if positive.sum() else np.ones(4)
    )
    return weights, {
        "raw_validation_gains": gains.tolist(),
        "weights": weights.tolist(),
        "low_train_means": lo_mean.tolist(),
        "high_train_means": hi_mean.tolist(),
        "rule": "Centered capacity1/4 marginal gains, floored square-root weights; closest common-grid allocation to continuous minimum-one allocation. Adapted heuristic control.",
    }


def pilot_allocation(total, weights, allocations):
    desired = 1 + (total - 4) * np.asarray(weights) / np.sum(weights)
    return min(allocations, key=lambda a: float(np.sum((np.array(a) - desired) ** 2)))


def certificate_allocations(private_specification, budgets, grid):
    """Explicit private oracle. Never call from the observation-only selector."""
    intervals = np.array(private_specification["intervals"])
    scales = np.array(private_specification["component_scales"])
    low, high = intervals.T
    center = (high + low) / 2
    q = (high - low) / (high + low)
    amplitude = 2 / ((1 + center) * (1 - q))
    b = (amplitude / (2 * scales)) ** 2
    rate = 4 * np.log(1 / q)
    return {
        str(total): list(
            min(
                candidates(total, grid),
                key=lambda a: float(np.sum(b * np.exp(-rate * np.array(a)))),
            )
        )
        for total in budgets
    }


def generate_data(task, seed, config, confirmation=False):
    """Exactly one anchor plus 4*n structured TRAIN label queries."""
    n = config["local_train_n"]
    anchor = task.sample_inputs(seed * 100 + 1, 1)
    anchor_y = task.evaluate(anchor)[0]
    raw = task.sample_inputs(seed * 100 + 2, n)
    query_x = anchor.repeat(4 * n, 1, 1)
    for block in range(4):
        query_x[block * n : (block + 1) * n, block] = raw[:, block]
    query_y = task.evaluate(query_x)
    data = {
        "anchor_x": anchor.numpy(),
        "anchor_y": anchor_y.numpy(),
        "query_x": query_x.numpy(),
        "query_y": query_y.numpy(),
        "query_block": np.repeat(np.arange(4), n),
        "query_id": np.arange(1, 4 * n + 1),
        "anchor_query_id": np.array(0),
        "local_x": raw.numpy(),
        "local_y": (query_y.reshape(4, n).T - anchor_y).numpy(),
    }
    for name, count, offset in [
        ("selector", config["selector_n"], 3),
        (
            "test" if confirmation else "report",
            config["test_n"] if confirmation else config["report_n"],
            4,
        ),
    ]:
        x, y = task.sample(seed * 100 + offset, count)
        data[f"{name}_x"], data[f"{name}_y"] = x.numpy(), y.numpy()
    if confirmation:
        x, y = task.sample(seed * 100 + 5, 4 * n + 1)
        data["iid_train_x"], data["iid_train_y"] = x.numpy(), y.numpy()
    return data


def data_path(root, stage, condition, seed):
    return Path(root) / "data" / f"{stage}_{condition}_{seed}.npz"


def save_data(root, config, stage):
    confirmation = stage == "confirmation"
    for condition in config["conditions"]:
        task = SmoothBlockTask(
            TaskConfig(condition=condition, teacher_seed=config["teacher_seed"])
        )
        for seed in config[f"{stage}_seeds"]:
            data = generate_data(task, seed, config, confirmation)
            path = data_path(root, stage, condition, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                np.savez_compressed(handle, **data)
            write_json(
                path.with_suffix(".json"),
                {
                    "seed": seed,
                    "stage": stage,
                    "condition": condition,
                    "file": str(path),
                    "sha256": sha(path),
                    "created_utc": now(),
                    "structured_train_scalar_queries": 1 + 4 * config["local_train_n"],
                    "iid_diagnostic_train_scalar_queries": (
                        1 + 4 * config["local_train_n"] if confirmation else 0
                    ),
                    "selector_scalar_labels": config["selector_n"],
                    "report_scalar_labels": 0 if confirmation else config["report_n"],
                    "test_scalar_labels": config["test_n"] if confirmation else 0,
                    "note": "Raw input generation calls sample_inputs and evaluates no discarded labels. Arrays retain the complete scalar query transcript. Families/policies reuse identical files.",
                },
            )


def initialize(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    private = {}
    budgets = config["development_budgets"] + [config["withheld_budget"]]
    for condition in config["conditions"]:
        task = SmoothBlockTask(
            TaskConfig(condition=condition, teacher_seed=config["teacher_seed"])
        )
        private[condition] = task.specification()
    write_json(root / "private_teacher_audit.json", private)
    write_json(
        root / "private_certificate_oracle.json",
        {
            condition: certificate_allocations(spec, budgets, config["capacity_grid"])
            for condition, spec in private.items()
        },
    )
    save_data(root, config, "development")
    write_json(
        root / "initialized.json",
        {
            "utc": now(),
            "config_sha256": sha(root / "config.json"),
            "candidate_counts": {
                str(m): len(candidates(m, config["capacity_grid"])) for m in budgets
            },
        },
    )


def read_data(root, stage, condition, seed):
    path = data_path(root, stage, condition, seed)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != receipt["sha256"]:
        raise ValueError("Dataset hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: torch.from_numpy(arrays[name].copy()) for name in arrays.files}


def fit_saved(path, capacities, family, init_seed, x, y, choice):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model = SmoothBlockModel(
        capacities,
        family,
        seed=init_seed,
        train_data=(x, y),
        threshold_mode=choice["threshold_mode"],
    )
    initial = save_state(model, path / "initial.npz")
    try:
        receipt = fit(model, x, y, FitConfig(**choice["fit"]))
        before = float(model.bias.detach())
        deployed = fit_intercept(model, x, y)
        receipt.update(
            {
                "status": "complete",
                "intercept_refit_train_only_delta": float(model.bias.detach()) - before,
                "deployed_train_metrics": deployed,
                "initial_state": initial,
                "state": save_state(model, path / "state.npz"),
                "completed_utc": now(),
            }
        )
        write_json(path / "fit.json", receipt)
    except Exception as error:
        write_json(
            path / "failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "utc": now(),
                "initial_state": initial,
                "partial_state": save_state(model, path / "partial_state.npz"),
                "note": "Failure blocks stage completion and global selection; no failed fit is discarded. In-function closure history is unavailable after an exception.",
            },
        )
        raise
    return model


def library_dir(root, stage, condition, family, choice, seed):
    return Path(root) / stage / condition / family / choice["id"] / f"s{seed}"


def fit_library(root, stage, condition, family, choice, seed, data, capacities):
    base = library_dir(root, stage, condition, family, choice, seed)
    models = {}
    for block, capacity in itertools.product(range(4), capacities):
        path = base / f"b{block}_c{capacity}"
        models[(block, capacity)] = fit_saved(
            path,
            [capacity],
            family,
            seed * 10000 + block * 100 + capacity,
            data["local_x"][:, block : block + 1],
            data["local_y"][:, block],
            choice,
        )
    return models


def load_library(root, stage, condition, family, choice, seed, capacities):
    base = library_dir(root, stage, condition, family, choice, seed)
    result = {}
    for block, capacity in itertools.product(range(4), capacities):
        path = base / f"b{block}_c{capacity}"
        receipt = json.loads((path / "fit.json").read_text())
        if (
            receipt["status"] != "complete"
            or sha(path / "state.npz") != receipt["state"]["sha256"]
        ):
            raise ValueError("Incomplete or changed library fit")
        result[(block, capacity)] = load_state(path / "state.npz")
    return result


def assemble(library, allocation, anchor_y):
    model = assemble_blocks([library[(i, c)] for i, c in enumerate(allocation)])
    with torch.no_grad():
        model.bias.add_(anchor_y)
    assert model.parameter_count == 5 * sum(allocation) + 5
    return model


def evaluate_policies(
    root, stage, condition, family, choice, seed, data, library, budgets, config
):
    with torch.no_grad():
        predictions = {
            (i, c): model(data["selector_x"][:, i : i + 1]).numpy()
            for (i, c), model in library.items()
        }
        means = {
            (i, c): float(model(data["local_x"][:, i : i + 1]).mean())
            for (i, c), model in library.items()
        }
    weights, pilot_receipt = pilot_weights(
        predictions, means, data["selector_y"].numpy(), data["anchor_y"]
    )
    # The oracle receipt crosses only into the separately labeled oracle policy.
    oracle = json.loads((Path(root) / "private_certificate_oracle.json").read_text())[
        condition
    ]
    rows = []
    for total in budgets:
        feasible = candidates(total, config["capacity_grid"])
        selected, selection = choose_observational(
            predictions, data["selector_y"].numpy(), data["anchor_y"], feasible
        )
        allocations = {
            "uniform": feasible[0],
            "validation_selected": selected,
            "shuffled_selected": tuple(selected[j] for j in config["shuffle"]),
            "small_pilot": pilot_allocation(total, weights, feasible),
            "certificate_oracle": tuple(oracle[str(total)]),
        }
        dest = (
            Path(root)
            / "assembled"
            / stage
            / condition
            / family
            / f"s{seed}"
            / f"m{total}"
        )
        write_json(
            dest / "selection.json",
            {
                **selection,
                "pilot": pilot_receipt,
                "choice_id": choice["id"],
                "data_sha256": sha(data_path(root, stage, condition, seed)),
                "oracle_is_private_control": True,
                "selected_utc": now(),
            },
        )
        for policy, allocation in allocations.items():
            model = assemble(library, allocation, data["anchor_y"])
            row = {
                "stage": stage,
                "condition": condition,
                "family": family,
                "seed": seed,
                "budget": total,
                "parameters": model.parameter_count,
                "allocation": allocation,
                "policy": policy,
                "choice_id": choice["id"],
                "candidate_count": len(feasible),
                "selector": evaluate(model, data["selector_x"], data["selector_y"]),
                "state": save_state(model, dest / f"{policy}.npz"),
            }
            endpoint = "test" if stage == "confirmation" else "report"
            row[endpoint] = evaluate(
                model, data[f"{endpoint}_x"], data[f"{endpoint}_y"]
            )
            # Endpoint evaluation never changes weights or allocation.
            write_json(dest / f"{policy}.json", row)
            rows.append(row)
    return rows


def development(root, config, condition, family):
    capacities = [c for c in config["capacity_grid"] if c < 24]
    rows = []
    for choice in choices(config, family):
        for seed in config["development_seeds"]:
            data = read_data(root, "development", condition, seed)
            library = fit_library(
                root, "development", condition, family, choice, seed, data, capacities
            )
            for total in config["development_budgets"]:
                model = assemble(library, (total // 4,) * 4, data["anchor_y"])
                rows.append(
                    {
                        "condition": condition,
                        "family": family,
                        "seed": seed,
                        "choice_id": choice["id"],
                        "budget": total,
                        "parameters": model.parameter_count,
                        "report": evaluate(model, data["report_x"], data["report_y"]),
                    }
                )
            print(
                json.dumps(
                    {
                        "stage": "development",
                        "condition": condition,
                        "family": family,
                        "choice": choice["id"],
                        "seed": seed,
                        "completed_utc": now(),
                    }
                ),
                flush=True,
            )
    write_json(
        Path(root) / "development" / f"{condition}_{family}_complete.json",
        {"rows": rows, "utc": now()},
    )


def global_select(root, config):
    root = Path(root)
    selected, all_rows, forecasts = {}, [], []
    threshold = config["descriptive_error_threshold"]
    for condition, family in itertools.product(
        config["conditions"], config["families"]
    ):
        receipt = json.loads(
            (root / "development" / f"{condition}_{family}_complete.json").read_text()
        )
        family_choices = choices(config, family)
        scores = {
            choice["id"]: float(
                np.mean(
                    [
                        np.log(max(row["report"]["mse"], threshold))
                        for row in receipt["rows"]
                        if row["choice_id"] == choice["id"]
                    ]
                )
            )
            for choice in family_choices
        }
        choice = min(family_choices, key=lambda item: scores[item["id"]])
        selected[f"{condition}/{family}"] = {
            "choice": choice,
            "scores": scores,
            "rule": "Minimum mean log thresholded uniform-report risk across every development budget/seed. Selector validation is not used for recipe calibration.",
        }
        rows = []
        for seed in config["development_seeds"]:
            data = read_data(root, "development", condition, seed)
            library = load_library(
                root,
                "development",
                condition,
                family,
                choice,
                seed,
                [c for c in config["capacity_grid"] if c < 24],
            )
            rows.extend(
                evaluate_policies(
                    root,
                    "development",
                    condition,
                    family,
                    choice,
                    seed,
                    data,
                    library,
                    config["development_budgets"],
                    config,
                )
            )
        all_rows.extend(rows)
        last = config["development_budgets"][-3:]
        for policy in config["policies"]:
            risks = [
                float(
                    np.mean(
                        [
                            row["report"]["mse"]
                            for row in rows
                            if row["policy"] == policy and row["budget"] == m
                        ]
                    )
                )
                for m in last
            ]
            p = np.array([5 * m + 5 for m in last])
            log_risk = np.log(np.maximum(risks, threshold))
            for form, transformed in [
                ("power", np.log(p)),
                ("exponential_in_parameters", p),
            ]:
                slope, intercept = np.polyfit(transformed, log_risk, 1)
                next_p = 5 * config["withheld_budget"] + 5
                next_x = np.log(next_p) if form == "power" else next_p
                forecasts.append(
                    {
                        "condition": condition,
                        "family": family,
                        "policy": policy,
                        "form": form,
                        "fit_budgets": last,
                        "fit_mean_mse": risks,
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "withheld_parameters": next_p,
                        "predicted_mse": float(np.exp(slope * next_x + intercept)),
                        "above_descriptive_threshold_points": int(
                            np.sum(np.array(risks) > threshold)
                        ),
                        "scope": "Three-point descriptive extrapolation, not asymptotic exponent evidence.",
                    }
                )
    write_json(root / "selected_recipes.json", {"selected": selected, "utc": now()})
    write_json(root / "selected_development_rows.json", all_rows)
    write_json(
        root / "frozen_forecasts.json",
        {
            "forecasts": forecasts,
            "utc": now(),
            "selected_sha256": sha(root / "selected_recipes.json"),
            "withheld_whole_budget": config["withheld_budget"],
            "withheld_new_local_capacity": 24,
        },
    )
    # Confirmation observations do not exist until recipes and forecasts are frozen.
    save_data(root, config, "confirmation")
    write_json(
        root / "selection_complete.json",
        {"utc": now(), "forecast_sha256": sha(root / "frozen_forecasts.json")},
    )


def confirmation(root, config, condition, family):
    root = Path(root)
    completion = json.loads((root / "selection_complete.json").read_text())
    if sha(root / "frozen_forecasts.json") != completion["forecast_sha256"]:
        raise ValueError("Forecast changed after freeze")
    forecasts = json.loads((root / "frozen_forecasts.json").read_text())
    if sha(root / "selected_recipes.json") != forecasts["selected_sha256"]:
        raise ValueError("Selected recipe changed after forecast freeze")
    selection = json.loads((root / "selected_recipes.json").read_text())
    choice = selection["selected"][f"{condition}/{family}"]["choice"]
    rows, diagnostics = [], []
    for seed in config["confirmation_seeds"]:
        data = read_data(root, "confirmation", condition, seed)
        library = fit_library(
            root,
            "confirmation",
            condition,
            family,
            choice,
            seed,
            data,
            config["capacity_grid"],
        )
        rows.extend(
            evaluate_policies(
                root,
                "confirmation",
                condition,
                family,
                choice,
                seed,
                data,
                library,
                config["development_budgets"] + [config["withheld_budget"]],
                config,
            )
        )
        for total, design in itertools.product(
            config["diagnostic_budgets"], ["paired_global", "iid_global"]
        ):
            if design == "paired_global":
                x = torch.cat([data["anchor_x"], data["query_x"]])
                y = torch.cat([data["anchor_y"].reshape(1), data["query_y"]])
            else:
                x, y = data["iid_train_x"], data["iid_train_y"]
            path = (
                root
                / "diagnostics"
                / condition
                / family
                / f"s{seed}"
                / f"m{total}_{design}"
            )
            model = fit_saved(
                path, [total // 4] * 4, family, seed * 10000 + total, x, y, choice
            )
            row = {
                "condition": condition,
                "family": family,
                "seed": seed,
                "budget": total,
                "parameters": model.parameter_count,
                "design": design,
                "train_labels": len(y),
                "choice_id": choice["id"],
                "test": evaluate(model, data["test_x"], data["test_y"]),
                "scope": "Uniform global-fit diagnostic with inherited local-calibrated recipe; not a separately optimized IID frontier.",
            }
            write_json(path / "evaluation.json", row)
            diagnostics.append(row)
        print(
            json.dumps(
                {
                    "stage": "confirmation",
                    "condition": condition,
                    "family": family,
                    "seed": seed,
                    "completed_utc": now(),
                }
            ),
            flush=True,
        )
    write_json(
        root / "confirmation" / f"{condition}_{family}_complete.json",
        {"rows": rows, "diagnostics": diagnostics, "utc": now()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["initialize", "development", "select", "confirmation"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--family")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "initialize":
        initialize(args.root, default_config())
        return
    config = json.loads((args.root / "config.json").read_text())
    initialized = json.loads((args.root / "initialized.json").read_text())
    if sha(args.root / "config.json") != initialized["config_sha256"]:
        raise ValueError("Configuration changed after initialization")
    bindings = json.loads((args.root / "frozen_sources.json").read_text())
    for name, digest in bindings.items():
        if sha(Path(__file__).resolve().parent / name) != digest:
            raise ValueError(f"Frozen source mismatch: {name}")
    if args.stage == "select":
        global_select(args.root, config)
    else:
        if (
            args.condition not in config["conditions"]
            or args.family not in config["families"]
        ):
            parser.error("Declared condition and family required")
        (development if args.stage == "development" else confirmation)(
            args.root, config, args.condition, args.family
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
