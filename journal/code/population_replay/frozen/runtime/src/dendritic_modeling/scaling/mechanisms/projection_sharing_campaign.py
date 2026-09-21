"""Frozen sample-size and projection-sharing experiment with a two-index control."""

from __future__ import annotations

import argparse
import itertools
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import projection_sharing as model_code
    from .smooth_allocation_campaign import now, sha, write_json
else:
    import projection_sharing as model_code
    from smooth_allocation_campaign import now, sha, write_json


def configuration():
    return {
        "development_p": [65, 125, 185, 245, 365],
        "withheld_p": 485,
        "architectures": ["independent", "shared"],
        "families": ["shunt", "relu", "tanh"],
        "development_seeds": [2026102101, 2026102103],
        "confirmation_seeds": [2026102201, 2026102203, 2026102207],
        "counter_seeds": [2026102301, 2026102303, 2026102307],
        "train_ns": [128, 512, 2048],
        "development_n": 512,
        "validation_n": 1024,
        "test_n": 8192,
        "counter_p": [125, 245, 485],
        "single_teacher": 2026102001,
        "counter_teacher": 2026102002,
        "recipes": ["readout_ls_then_joint", "alternating_ls_then_joint"],
        "steps": [600, 1200],
        "threshold": 1e-18,
        "optimizer_fits": 3012,
        "scope": "Signed shared branch gains versus independent affine vectors for every family; exact common P and same-M controls. Nested sample sizes. Separate two-index counterexample with inherited recipe.",
    }


def capacity(p, architecture):
    numerator, denominator = (
        (p - 5, 20) if architecture == "independent" else (p - 17, 12)
    )
    if numerator <= 0 or numerator % denominator:
        raise ValueError("Budget does not give a positive integer uniform capacity")
    return numerator // denominator


def choices(config):
    return [
        {
            "id": f"{recipe}_{steps}",
            "fit": asdict(
                model_code.FitConfig(
                    recipe=recipe, lbfgs_steps=steps, learning_rate=0.5
                )
            ),
        }
        for recipe, steps in itertools.product(config["recipes"], config["steps"])
    ]


def data_path(root, stage, seed):
    return root / "data" / f"{stage}_{seed}.npz"


def generate_data(task, seed, n, endpoint, endpoint_n):
    anchor = task.sample_inputs(seed * 100 + 1, 1)
    y0 = task.evaluate(anchor)[0]
    raw = task.sample_inputs(seed * 100 + 2, n)
    queries = anchor.repeat(4 * n, 1, 1)
    for block in range(4):
        queries[block * n : (block + 1) * n, block] = raw[:, block]
    labels = task.evaluate(queries)
    ex, ey = task.sample(seed * 100 + 3, endpoint_n)
    return {
        "anchor_x": anchor.numpy(),
        "anchor_y": y0.numpy(),
        "local_x": raw.numpy(),
        "local_y": (labels.reshape(4, n).T - y0).numpy(),
        "query_x": queries.numpy(),
        "query_y": labels.numpy(),
        "query_block": np.repeat(np.arange(4), n),
        "query_id": np.arange(1, 4 * n + 1),
        "anchor_query_id": np.array(0),
        f"{endpoint}_x": ex.numpy(),
        f"{endpoint}_y": ey.numpy(),
    }


def save_data(root, config, stage):
    counter = stage == "counter"
    task = model_code.ProjectionTask(
        "two_index" if counter else "single_index",
        config["counter_teacher" if counter else "single_teacher"],
    )
    write_json(root / f"private_{stage}_teacher.json", task.specification())
    n = max(config["train_ns"]) if stage == "confirmation" else config["development_n"]
    endpoint = "validation" if stage == "development" else "test"
    for seed in config[f"{stage}_seeds"]:
        data = generate_data(task, seed, n, endpoint, config[f"{endpoint}_n"])
        path = data_path(root, stage, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **data)
        write_json(
            path.with_suffix(".json"),
            {
                "sha256": sha(path),
                "stage": stage,
                "seed": seed,
                "utc": now(),
                "unique_training_labels": 1 + 4 * n,
                "endpoint_labels": config[f"{endpoint}_n"],
                "sample_dose_scope": "Smaller-N learners receive only prefixes of each block's TRAIN pairs; physical collection counted at largest N once.",
            },
        )


def read_data(root, stage, seed):
    path = data_path(root, stage, seed)
    if sha(path) != json.loads(path.with_suffix(".json").read_text())["sha256"]:
        raise ValueError("Changed data")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: torch.from_numpy(arrays[name].copy()) for name in arrays.files}


def initialize(root):
    config = configuration()
    write_json(root / "config.json", config)
    save_data(root, config, "development")
    write_json(
        root / "initialized.json",
        {
            "config_sha256": sha(root / "config.json"),
            "utc": now(),
            "expected_fits": 3012,
        },
    )


def location(root, stage, architecture, family, choice, seed, n):
    return root / stage / architecture / family / choice["id"] / f"s{seed}" / f"n{n}"


def fit_library(root, stage, architecture, family, choice, seed, n, capacities, data):
    base = location(root, stage, architecture, family, choice, seed, n)
    library = {}
    for block, c in itertools.product(range(4), capacities):
        path = base / f"b{block}_c{c}"
        path.mkdir(parents=True, exist_ok=False)
        x, y = data["local_x"][:n, block : block + 1], data["local_y"][:n, block]
        model = None
        try:
            model = model_code.ProjectionModel(
                [c],
                family,
                architecture,
                seed=seed * 100000 + block * 1000 + c,
                train_data=(x, y),
            )
            initial = model_code.save_state(model, path / "initial.npz")
            receipt = model_code.fit(model, x, y, model_code.FitConfig(**choice["fit"]))
            receipt.update(
                status="complete",
                initial_state=initial,
                state=model_code.save_state(model, path / "state.npz"),
                completed_utc=now(),
            )
            write_json(path / "fit.json", receipt)
        except Exception as error:
            write_json(
                path / "failure.json",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "utc": now(),
                    "partial_state": (
                        None
                        if model is None
                        else model_code.save_state(model, path / "partial.npz")
                    ),
                },
            )
            raise
        library[(block, c)] = model
    return library


def load_library(root, stage, architecture, family, choice, seed, n, capacities):
    result = {}
    base = location(root, stage, architecture, family, choice, seed, n)
    for block, c in itertools.product(range(4), capacities):
        path = base / f"b{block}_c{c}"
        receipt = json.loads((path / "fit.json").read_text())
        if sha(path / "state.npz") != receipt["state"]["sha256"]:
            raise ValueError("Changed model")
        result[(block, c)] = model_code.load_state(path / "state.npz")
    return result


def endpoints(
    root,
    stage,
    architecture,
    family,
    choice,
    seed,
    n,
    library,
    capacities,
    data,
    save=True,
):
    endpoint = "validation" if stage == "development" else "test"
    rows = []
    for c in capacities:
        model = model_code.assemble(
            [library[(block, c)] for block in range(4)], data["anchor_y"]
        )
        row = {
            "stage": stage,
            "architecture": architecture,
            "family": family,
            "choice_id": choice["id"],
            "seed": seed,
            "train_n_per_block": n,
            "training_labels_available": 4 * n + 1,
            "capacity": c,
            "branches": 4 * c,
            "parameters": model.parameter_count,
            endpoint: model_code.evaluate(
                model, data[f"{endpoint}_x"], data[f"{endpoint}_y"]
            ),
        }
        if save:
            path = (
                root
                / "assembled"
                / stage
                / architecture
                / family
                / f"s{seed}"
                / f"n{n}"
                / f"c{c}"
            )
            path.mkdir(parents=True, exist_ok=False)
            row["state"] = model_code.save_state(model, path / "state.npz")
            write_json(path / "evaluation.json", row)
        rows.append(row)
    return rows


def development(root, config, architecture, family):
    caps = [capacity(p, architecture) for p in config["development_p"]]
    rows = []
    for choice in choices(config):
        for seed in config["development_seeds"]:
            data = read_data(root, "development", seed)
            library = fit_library(
                root, "development", architecture, family, choice, seed, 512, caps, data
            )
            rows.extend(
                endpoints(
                    root,
                    "development",
                    architecture,
                    family,
                    choice,
                    seed,
                    512,
                    library,
                    caps,
                    data,
                    save=False,
                )
            )
            print(
                json.dumps(
                    {
                        "stage": "development",
                        "architecture": architecture,
                        "family": family,
                        "choice": choice["id"],
                        "seed": seed,
                        "utc": now(),
                    }
                ),
                flush=True,
            )
    write_json(
        root / "development" / f"{architecture}_{family}_complete.json",
        {"rows": rows, "utc": now()},
    )


def select(root, config):
    selections, forecasts = {}, []
    for architecture, family in itertools.product(
        config["architectures"], config["families"]
    ):
        rows = json.loads(
            (
                root / "development" / f"{architecture}_{family}_complete.json"
            ).read_text()
        )["rows"]
        scores = {
            choice["id"]: float(
                np.mean(
                    [
                        np.log(max(row["validation"]["mse"], config["threshold"]))
                        for row in rows
                        if row["choice_id"] == choice["id"]
                    ]
                )
            )
            for choice in choices(config)
        }
        chosen = min(choices(config), key=lambda choice: scores[choice["id"]])
        selections[f"{architecture}/{family}"] = {"choice": chosen, "scores": scores}
        caps = [capacity(p, architecture) for p in config["development_p"]]
        for seed in config["development_seeds"]:
            data = read_data(root, "development", seed)
            library = load_library(
                root, "development", architecture, family, chosen, seed, 512, caps
            )
            endpoints(
                root,
                "development",
                architecture,
                family,
                chosen,
                seed,
                512,
                library,
                caps,
                data,
            )
        last = config["development_p"][-3:]
        risks = [
            np.mean(
                [
                    row["validation"]["mse"]
                    for row in rows
                    if row["choice_id"] == chosen["id"] and row["parameters"] == p
                ]
            )
            for p in last
        ]
        for form in ["power", "exponential"]:
            x = np.log(last) if form == "power" else np.array(last)
            slope, intercept = np.polyfit(
                x, np.log(np.maximum(risks, config["threshold"])), 1
            )
            target = (
                np.log(config["withheld_p"])
                if form == "power"
                else config["withheld_p"]
            )
            forecasts.append(
                {
                    "architecture": architecture,
                    "family": family,
                    "form": form,
                    "development_n": 512,
                    "fit_p": last,
                    "fit_mean_mse": [float(v) for v in risks],
                    "predicted_mse": float(np.exp(intercept + slope * target)),
                    "slope": float(slope),
                    "scope": "Calibrated three-point descriptive forecast; not an empirical asymptotic exponent.",
                }
            )
    write_json(root / "selected_recipes.json", {"selections": selections, "utc": now()})
    write_json(
        root / "frozen_forecasts.json",
        {
            "forecasts": forecasts,
            "selected_sha256": sha(root / "selected_recipes.json"),
            "utc": now(),
        },
    )
    save_data(root, config, "confirmation")
    save_data(root, config, "counter")
    write_json(
        root / "selection_complete.json",
        {"forecast_sha256": sha(root / "frozen_forecasts.json"), "utc": now()},
    )


def confirmation(root, config, architecture, family):
    completion = json.loads((root / "selection_complete.json").read_text())
    if sha(root / "frozen_forecasts.json") != completion["forecast_sha256"]:
        raise ValueError("Changed forecast")
    forecasts = json.loads((root / "frozen_forecasts.json").read_text())
    if sha(root / "selected_recipes.json") != forecasts["selected_sha256"]:
        raise ValueError("Changed recipe")
    choice = json.loads((root / "selected_recipes.json").read_text())["selections"][
        f"{architecture}/{family}"
    ]["choice"]
    primary = [
        capacity(p, architecture)
        for p in config["development_p"] + [config["withheld_p"]]
    ]
    caps = sorted(
        set(
            primary
            + (
                [
                    capacity(p, "independent")
                    for p in config["development_p"] + [config["withheld_p"]]
                ]
                if architecture == "shared"
                else []
            )
        )
    )
    rows = []
    for seed, n in itertools.product(config["confirmation_seeds"], config["train_ns"]):
        data = read_data(root, "confirmation", seed)
        library = fit_library(
            root, "confirmation", architecture, family, choice, seed, n, caps, data
        )
        rows.extend(
            endpoints(
                root,
                "confirmation",
                architecture,
                family,
                choice,
                seed,
                n,
                library,
                caps,
                data,
            )
        )
        print(
            json.dumps(
                {
                    "stage": "confirmation",
                    "architecture": architecture,
                    "family": family,
                    "seed": seed,
                    "n": n,
                    "utc": now(),
                }
            ),
            flush=True,
        )
    counter_caps = [capacity(p, architecture) for p in config["counter_p"]]
    counter_rows = []
    for seed in config["counter_seeds"]:
        data = read_data(root, "counter", seed)
        library = fit_library(
            root, "counter", architecture, family, choice, seed, 512, counter_caps, data
        )
        counter_rows.extend(
            endpoints(
                root,
                "counter",
                architecture,
                family,
                choice,
                seed,
                512,
                library,
                counter_caps,
                data,
            )
        )
    write_json(
        root / "confirmation" / f"{architecture}_{family}_complete.json",
        {"rows": rows, "counter_rows": counter_rows, "utc": now()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["initialize", "development", "select", "confirmation"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--architecture", choices=["independent", "shared"])
    parser.add_argument("--family", choices=["shunt", "relu", "tanh"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "initialize":
        initialize(args.root)
        return
    for name, digest in json.loads(
        (args.root / "frozen_sources.json").read_text()
    ).items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Changed source {name}")
    if (
        sha(args.root / "config.json")
        != json.loads((args.root / "initialized.json").read_text())["config_sha256"]
    ):
        raise ValueError("Changed configuration")
    config = json.loads((args.root / "config.json").read_text())
    if args.stage == "select":
        select(args.root, config)
    else:
        if args.architecture is None or args.family is None:
            parser.error("Architecture and family required")
        (development if args.stage == "development" else confirmation)(
            args.root, config, args.architecture, args.family
        )


if __name__ == "__main__":
    main()
