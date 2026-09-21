"""Prospective IID rank/optimizer crossover, with frozen larger-size forecasts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import rank_learning as model_code, rank_study_tasks as task_code
else:
    import rank_learning as model_code
    import rank_study_tasks as task_code


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_initialized(root):
    receipt = read(root / "initialized.json")
    for filename, key in [
        ("config.json", "config_sha256"),
        ("choices.json", "choices_sha256"),
        ("protocol.md", "protocol_sha256"),
    ]:
        if sha(root / filename) != receipt[key]:
            raise ValueError(f"Changed frozen {filename}")
    for source in receipt["sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError(f"Changed source {source['path']}")
    return receipt


def verify_selection(root):
    receipt = read(root / "selection_complete.json")
    if sha(root / "selected_recipes.json") != receipt["selected_sha256"]:
        raise ValueError("Changed selected recipes")


def verify_forecast(root):
    verify_selection(root)
    receipt = read(root / "forecast_complete.json")
    if sha(root / "frozen_forecasts.json") != receipt["forecast_sha256"]:
        raise ValueError("Changed forecast")
    if (
        sha(root / "selected_recipes.json")
        != read(root / "frozen_forecasts.json")["selected_sha256"]
    ):
        raise ValueError("Forecast selection mismatch")


def configuration():
    return {
        "families": ["shunt", "relu", "tanh"],
        "architectures": ["full", "rank1", "rank2"],
        "development_teachers": [0, 1],
        "confirmation_teachers": [10, 11, 12, 13],
        "sine_teacher": 20,
        "teacher_seed_base": 2026105001,
        "observation_seed_base": 2026106001,
        "calibration_p": [65, 245],
        "bridge_p": 125,
        "withheld_p": 485,
        "primary_p": [65, 125, 245, 485],
        "sensitivity_p": [125, 485],
        "same_m": [24, 48],
        "intrinsic_ranks": [1, 2],
        "main_n": 2048,
        "small_n": 512,
        "validation_n": 2048,
        "test_n": 8192,
        "noise_sigma": 0.03,
        "initialization_spread": 0.1,
        "log_risk_threshold": 1e-18,
        "expected_fits": 1980,
        "fit_counts": {
            "calibration": 288,
            "bridge": 36,
            "primary": 576,
            "small_sample": 288,
            "noise": 288,
            "random": 288,
            "same_m_additional": 144,
            "sine": 72,
        },
        "scope": "Four fresh mixture teachers are the inference units; two observation replicates each. Primary shuntq/rank interaction atP485,N2048. Allrank/primitive combinations retained. No assumption that every primitive wins; no empirical exponent claimed.",
    }


def choices():
    return [
        {
            "id": f"{method}_ridge{ridge:.0e}",
            "optimizer": method,
            "fit": asdict(
                model_code.FitConfig(
                    ridge=ridge,
                    steps=150 if method == "reduced" else 1200,
                    learning_rate=0.5,
                    objective_scale=1e4,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-14,
                )
            ),
        }
        for method, ridge in itertools.product(["reduced", "joint"], [1e-8, 1e-5])
    ]


def count_parameters(m, architecture):
    return (
        5 * m + 5
        if architecture == "full"
        else (3 * m + 17 if architecture == "rank1" else 4 * m + 29)
    )


def capacities_for(p, architecture, teacher_index):
    mult, overhead = {"full": (5, 5), "rank1": (3, 17), "rank2": (4, 29)}[architecture]
    if (p - overhead) % mult or (p - overhead) // mult < 4:
        raise ValueError("Not an exact positive common budget")
    return capacities_for_m((p - overhead) // mult, teacher_index)


def capacities_for_m(m, teacher_index):
    if m < 4:
        raise ValueError("Every block requires a branch")
    caps = np.full(4, m // 4, dtype=int)
    permutation = np.random.default_rng(
        np.random.SeedSequence([teacher_index, 971])
    ).permutation(4)
    caps[permutation[: m % 4]] += 1
    return caps.tolist()


def observation_seed(config, index, replicate):
    return config["observation_seed_base"] + 100 * index + 2 * replicate


def data_path(root, stage, index, q, seed):
    return Path(root) / "data" / stage / f"t{index}_q{q}_s{seed}.npz"


def save_datasets(root, config, stage):
    if stage != "development":
        verify_forecast(root)
    indices = (
        config["development_teachers"]
        if stage == "development"
        else (
            config["confirmation_teachers"]
            if stage == "confirmation"
            else [config["sine_teacher"]]
        )
    )
    for index in indices:
        teacher = task_code.RankTeacher(
            task_code.TeacherConfig(
                index,
                config["teacher_seed_base"] + index,
                "sine" if stage == "sine" else "mixture",
            )
        )
        write(
            root / "private_teachers" / f"{stage}_t{index}.json",
            teacher.specification(),
        )
        for replicate, q in itertools.product(
            range(1 if stage == "development" else 2), config["intrinsic_ranks"]
        ):
            seed = observation_seed(config, index, replicate)
            endpoint = "validation" if stage == "development" else "test"
            data = task_code.generate_data(
                teacher, seed, q, config["main_n"], endpoint, config[endpoint + "_n"]
            )
            path = data_path(root, stage, index, q, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                np.savez_compressed(f, **data)
            write(
                path.with_suffix(".json"),
                {
                    "stage": stage,
                    "teacher": index,
                    "intrinsic_rank": q,
                    "seed": seed,
                    "utc": now(),
                    "sha256": sha(path),
                    "train_labels": config["main_n"],
                    "endpoint_labels": config[endpoint + "_n"],
                    "noise_scope": "Independentstandardnormal TRAIN and endpointstreams shared acrosspairedq targets; noisevariantusesclean+sigma*epsilon atbothTRAIN/test. Noisevariantinheritsnoiselesscalibration; clean testMSEreportedonlyasseparateideal-teacherdiagnostic.",
                },
            )


def load_data(root, stage, index, q, seed):
    path = data_path(root, stage, index, q, seed)
    if sha(path) != read(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Changed dataset")
    with np.load(path, allow_pickle=False) as a:
        return {k: torch.from_numpy(a[k].copy()) for k in a.files}


def initialize(root):
    if not (root / "protocol.md").is_file():
        raise ValueError("Protocol must exist before initialization")
    config = configuration()
    write(root / "config.json", config)
    write(root / "choices.json", choices())
    save_datasets(root, config, "development")
    sources = [
        Path(__file__).resolve(),
        Path(model_code.__file__).resolve(),
        Path(task_code.__file__).resolve(),
    ]
    write(
        root / "initialized.json",
        {
            "utc": now(),
            "config_sha256": sha(root / "config.json"),
            "choices_sha256": sha(root / "choices.json"),
            "protocol_sha256": sha(root / "protocol.md"),
            "sources": [{"path": str(p), "sha256": sha(p)} for p in sources],
        },
    )


def evaluate(model, x, clean_y, epsilon, sigma):
    with torch.no_grad():
        prediction = model(x)
    observed_y = clean_y + sigma * epsilon
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(observed_y).all()):
        raise FloatingPointError("Nonfinite held-out prediction or label")
    return {
        "observed_mse": float((prediction - observed_y).square().mean()),
        "clean_teacher_mse": float((prediction - clean_y).square().mean()),
        "expected_observed_mse_from_clean": float(
            (prediction - clean_y).square().mean()
        )
        + sigma * sigma,
        "max_observed_error": float((prediction - observed_y).abs().max()),
        "sigma": sigma,
    }


def fit_one(
    root,
    config,
    stage,
    index,
    q,
    seed,
    family,
    architecture,
    choice,
    p,
    variant="primary",
    m=None,
):
    data_stage = (
        "development"
        if stage in ["development", "bridge"]
        else ("sine" if variant == "sine" else "confirmation")
    )
    data = load_data(root, data_stage, index, q, seed)
    n = config["small_n"] if variant == "small_sample" else config["main_n"]
    sigma = config["noise_sigma"] if variant == "noise" else 0.0
    initialization = "random" if variant == "random" else "observational"
    caps = (
        capacities_for(p, architecture, index)
        if m is None
        else capacities_for_m(m, index)
    )
    actual_p = count_parameters(sum(caps), architecture)
    path = (
        root
        / "fits"
        / stage
        / variant
        / architecture
        / family
        / f"q{q}"
        / f"t{index}_s{seed}"
        / choice["id"]
        / f"p{actual_p}"
    )
    model = None
    try:
        x = data["train_x"][:n]
        y = data["train_y_clean"][:n] + sigma * data["train_epsilon"][:n]
        model = model_code.RankBlockModel(
            caps,
            family,
            architecture,
            seed=seed,
            train_data=(x, y),
            initialization=initialization,
            initialization_spread=config["initialization_spread"],
        )
        if model.parameter_count != actual_p:
            raise ValueError("Runtime parameter inventory mismatch")
        fit = model_code.fit(
            model,
            x,
            y,
            model_code.FitConfig(**choice["fit"]),
            optimizer=choice["optimizer"],
            output_dir=path,
        )
        endpoint = "validation" if data_stage == "development" else "test"
        row = {
            "stage": stage,
            "variant": variant,
            "teacher": index,
            "intrinsic_rank": q,
            "seed": seed,
            "family": family,
            "architecture": architecture,
            "parameters": actual_p,
            "branches": sum(caps),
            "capacities": caps,
            "train_n": n,
            "sigma": sigma,
            "initialization": initialization,
            "model_seed": seed,
            "choice_id": choice["id"],
            "optimizer": choice["optimizer"],
            "dataset": str(data_path(root, data_stage, index, q, seed)),
            "state": fit["final_state"],
            "fit_file": str(path / "fit.json"),
            "utc": now(),
            endpoint: evaluate(
                model,
                data[endpoint + "_x"],
                data[endpoint + "_y_clean"],
                data[endpoint + "_epsilon"],
                sigma,
            ),
        }
        write(path / "evaluation.json", row)
        return row
    except Exception as error:
        if not path.exists():
            path.mkdir(parents=True)
        if not (path / "campaign_failure.json").exists():
            write(
                path / "campaign_failure.json",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "utc": now(),
                },
            )
        raise


def development(root, config, architecture, family):
    rows = []
    for q, index, choice, p in itertools.product(
        config["intrinsic_ranks"],
        config["development_teachers"],
        read(root / "choices.json"),
        config["calibration_p"],
    ):
        rows.append(
            fit_one(
                root,
                config,
                "development",
                index,
                q,
                observation_seed(config, index, 0),
                family,
                architecture,
                choice,
                p,
            )
        )
    write(
        root / "completed/development" / f"{architecture}_{family}.json",
        {"utc": now(), "rows": rows},
    )


def select(root, config):
    selections = {}
    for architecture, family in itertools.product(
        config["architectures"], config["families"]
    ):
        rows = read(root / "completed/development" / f"{architecture}_{family}.json")[
            "rows"
        ]
        for q in config["intrinsic_ranks"]:
            scores = {
                c["id"]: float(
                    np.mean(
                        [
                            np.log(
                                max(
                                    r["validation"]["observed_mse"],
                                    config["log_risk_threshold"],
                                )
                            )
                            for r in rows
                            if r["intrinsic_rank"] == q and r["choice_id"] == c["id"]
                        ]
                    )
                )
                for c in read(root / "choices.json")
            }
            selected = min(scores, key=scores.get)
            selections[f"{architecture}/{family}/q{q}"] = {
                "scores": scores,
                "choice": next(
                    c for c in read(root / "choices.json") if c["id"] == selected
                ),
            }
    write(root / "selected_recipes.json", {"utc": now(), "selections": selections})
    write(
        root / "selection_complete.json",
        {"utc": now(), "selected_sha256": sha(root / "selected_recipes.json")},
    )


def selected_choice(root, architecture, family, q):
    verify_selection(root)
    return read(root / "selected_recipes.json")["selections"][
        f"{architecture}/{family}/q{q}"
    ]["choice"]


def bridge(root, config, architecture, family):
    rows = []
    for q, index in itertools.product(
        config["intrinsic_ranks"], config["development_teachers"]
    ):
        rows.append(
            fit_one(
                root,
                config,
                "bridge",
                index,
                q,
                observation_seed(config, index, 0),
                family,
                architecture,
                selected_choice(root, architecture, family, q),
                config["bridge_p"],
            )
        )
    write(
        root / "completed/bridge" / f"{architecture}_{family}.json",
        {"utc": now(), "rows": rows},
    )


def freeze_forecasts(root, config):
    forecasts = []
    risks = {}
    for architecture, family in itertools.product(
        config["architectures"], config["families"]
    ):
        rows = (
            read(root / "completed/development" / f"{architecture}_{family}.json")[
                "rows"
            ]
            + read(root / "completed/bridge" / f"{architecture}_{family}.json")["rows"]
        )
        for q in config["intrinsic_ranks"]:
            choice = selected_choice(root, architecture, family, q)
            selected = [
                r
                for r in rows
                if r["intrinsic_rank"] == q and r["choice_id"] == choice["id"]
            ]
            ps = sorted(config["calibration_p"] + [config["bridge_p"]])
            means = np.array(
                [
                    np.mean(
                        [
                            r["validation"]["observed_mse"]
                            for r in selected
                            if r["parameters"] == p
                        ]
                    )
                    for p in ps
                ]
            )
            risks[(architecture, family, q)] = float(means[-1])
            for form in ["power", "exponential", "constant_last"]:
                slope = 0.0
                prediction = means[-1]
                if form != "constant_last":
                    x = np.log(ps) if form == "power" else np.array(ps)
                    xt = (
                        np.log(config["withheld_p"])
                        if form == "power"
                        else config["withheld_p"]
                    )
                    slope, intercept = np.polyfit(
                        x, np.log(np.maximum(means, config["log_risk_threshold"])), 1
                    )
                    prediction = np.exp(intercept + slope * xt)
                forecasts.append(
                    {
                        "architecture": architecture,
                        "family": family,
                        "intrinsic_rank": q,
                        "form": form,
                        "fit_p": ps,
                        "fit_mean_mse": means.tolist(),
                        "slope": float(slope),
                        "predicted_mse": float(prediction),
                        "withheld_p": config["withheld_p"],
                        "scope": "Descriptive3-pointforecast on developmentteachers. Noasymptoticexponentclaim.",
                    }
                )
    architecture_predictions = [
        {
            "family": family,
            "intrinsic_rank": q,
            "predicted_architecture": min(
                config["architectures"], key=lambda a: risks[(a, family, q)]
            ),
            "rule": "Transfer best mean-risk architecture at largest fittedP245 toP485 withouttestselection.",
        }
        for family, q in itertools.product(
            config["families"], config["intrinsic_ranks"]
        )
    ]
    write(
        root / "frozen_forecasts.json",
        {
            "utc": now(),
            "selected_sha256": sha(root / "selected_recipes.json"),
            "forecasts": forecasts,
            "architecture_predictions": architecture_predictions,
            "primary_hypothesis": "Shunt teacher-level interaction D=log(Rrank1/Rrank2)q2-log(Rrank1/Rrank2)q1>0 atP485,N2048. Also reportdirectq2rank2/rank1 andq1rank1/full contrasts; noallprimitivewinrequirement.",
        },
    )
    write(
        root / "forecast_complete.json",
        {"utc": now(), "forecast_sha256": sha(root / "frozen_forecasts.json")},
    )
    save_datasets(root, config, "confirmation")
    save_datasets(root, config, "sine")
    write(
        root / "confirmation_initialized.json",
        {"utc": now(), "forecast_sha256": sha(root / "frozen_forecasts.json")},
    )


def confirmation(root, config, architecture, family):
    verify_forecast(root)
    frozen = read(root / "confirmation_initialized.json")
    if frozen["forecast_sha256"] != sha(root / "frozen_forecasts.json"):
        raise ValueError("Forecastchanged")
    rows = []
    for index, q, replicate in itertools.product(
        config["confirmation_teachers"], config["intrinsic_ranks"], range(2)
    ):
        seed = observation_seed(config, index, replicate)
        choice = selected_choice(root, architecture, family, q)
        for p in config["primary_p"]:
            rows.append(
                fit_one(
                    root,
                    config,
                    "confirmation",
                    index,
                    q,
                    seed,
                    family,
                    architecture,
                    choice,
                    p,
                )
            )
        for variant, p in itertools.product(
            ["small_sample", "noise", "random"], config["sensitivity_p"]
        ):
            rows.append(
                fit_one(
                    root,
                    config,
                    "confirmation",
                    index,
                    q,
                    seed,
                    family,
                    architecture,
                    choice,
                    p,
                    variant,
                )
            )
        for m in config["same_m"]:
            p = count_parameters(m, architecture)
            if p not in config["primary_p"]:
                rows.append(
                    fit_one(
                        root,
                        config,
                        "confirmation",
                        index,
                        q,
                        seed,
                        family,
                        architecture,
                        choice,
                        p,
                        "same_m",
                        m=m,
                    )
                )
    for q, replicate, p in itertools.product(
        config["intrinsic_ranks"], range(2), config["sensitivity_p"]
    ):
        index = config["sine_teacher"]
        seed = observation_seed(config, index, replicate)
        rows.append(
            fit_one(
                root,
                config,
                "confirmation",
                index,
                q,
                seed,
                family,
                architecture,
                selected_choice(root, architecture, family, q),
                p,
                "sine",
            )
        )
    write(
        root / "completed/confirmation" / f"{architecture}_{family}.json",
        {"utc": now(), "rows": rows},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=[
            "initialize",
            "development",
            "select",
            "bridge",
            "freeze",
            "confirmation",
            "complete",
        ],
        required=True,
    )
    p.add_argument("--architecture")
    p.add_argument("--family")
    args = p.parse_args()
    root = args.root
    torch.set_num_threads(1)
    if args.stage == "initialize":
        initialize(root)
        return
    verify_initialized(root)
    config = read(root / "config.json")
    if args.stage == "select":
        select(root, config)
    elif args.stage == "freeze":
        freeze_forecasts(root, config)
    elif args.stage == "complete":
        verify_forecast(root)
        failures = list((root / "fits").rglob("*failure.json"))
        if failures:
            raise ValueError(f"Unresolved failed fits: {len(failures)}")
        for stage, a, f in itertools.product(
            ["development", "bridge", "confirmation"],
            config["architectures"],
            config["families"],
        ):
            read(root / "completed" / stage / f"{a}_{f}.json")
        count = sum(1 for _ in (root / "fits").rglob("fit.json"))
        if count != config["expected_fits"]:
            raise ValueError(f"Fit inventory{count}")
        if sum(1 for _ in (root / "fits").rglob("evaluation.json")) != count:
            raise ValueError("Missing evaluation")
        write(
            root / "campaign_complete.json",
            {"utc": now(), "status": "complete", "fits": count},
        )
    else:
        globals()[args.stage](root, config, args.architecture, args.family)


if __name__ == "__main__":
    main()
