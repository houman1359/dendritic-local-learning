"""Prespecified teacher-cluster analysis of the generic radial-width campaign.

This source selects no fit, initializer, primitive, endpoint or forecast. It
requires complete campaign barriers before loading confirmation metrics. The
old reviewed statistical helpers are byte-bound and reused without changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATISTICS_SHA256 = "466122ae37effede215a7f7cf45242985a6f996e8de598ddbdb525366399de18"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 2026146001
PAIRS = (
    *tuple((f"width{n}", "rank2") for n in (1, 2, 3, 4, 5, 8)),
    ("rank2", "full"),
    ("width8", "full"),
)
FORECAST_PAIRS = (
    ("width2", "rank2"),
    ("width4", "rank2"),
    ("width8", "rank2"),
    ("rank2", "full"),
    ("width8", "full"),
)
THEORY_REFERENCES = {
    "radial_m2": {
        "geometry": "width2",
        "population_mse_lower_reference": 35 / 10656,
        "exact_population_reference": "35/10656",
    },
    "radial_m4": {
        "geometry": "width4",
        "population_mse_lower_reference": 33 / 30645760,
        "exact_population_reference": "33/30645760",
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    if not rows:
        Path(path).touch(exist_ok=False)
        return
    headers = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def statistics_module(source_path=None):
    path = Path(
        source_path or Path(__file__).with_name("rank_width_analysis.py")
    ).resolve()
    if sha(path) != STATISTICS_SHA256:
        raise ValueError(
            "Statistical helper source differs from reviewed immutable version"
        )
    spec = importlib.util.spec_from_file_location(
        "radial_width_reviewed_statistics", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_confirmation(results, config):
    teachers = sorted(config["confirmation_teachers"])
    if len(teachers) != len(set(teachers)) or len(teachers) < 2:
        raise ValueError("Unique paired teacher clusters required")
    groups = {}
    paired_data = {}
    for result in results:
        case = result["case"]
        if case["stage"] != "confirmation":
            continue
        if result["status"] != "complete":
            raise ValueError("Failed confirmation cases cannot be dropped")
        key = tuple(
            case[name] for name in ("task", "family", "geometry", "train_n", "ceiling")
        )
        teacher, observation = case["teacher"], case["observation"]
        if teacher not in teachers or observation not in (0, 1):
            raise ValueError("Unexpected teacher/observation")
        data_key = (case["task"], teacher, observation)
        if (
            paired_data.setdefault(data_key, result["data_sha256"])
            != result["data_sha256"]
        ):
            raise ValueError(
                "Paired arms/sample sizes do not share the same archived observations"
            )
        group = groups.setdefault(
            key, {"train": {}, "endpoint": {}, "parameters": set()}
        )
        for split in ("train", "endpoint"):
            index = (teacher, observation)
            value = result["metrics"][split + "_mse"]
            if index in group[split] or not np.isfinite(value) or value < 0:
                raise ValueError("Repeated or invalid confirmation risk")
            group[split][index] = value
        group["parameters"].add(result["counted_inventory"]["stored_parameters"])
    keys = set(
        itertools.product(
            config["tasks"],
            config["families"],
            config["geometries"],
            config["train_sizes"],
            config["primary_p"],
        )
    )
    if set(groups) != keys:
        raise ValueError("Incomplete or unexpected confirmation grid")
    expected = set(itertools.product(teachers, (0, 1)))
    for group in groups.values():
        if len(group["parameters"]) != 1:
            raise ValueError("Parameter count changed within a paired curve")
        group["actual_parameters"] = group.pop("parameters").pop()
        for split in ("train", "endpoint"):
            if set(group[split]) != expected:
                raise ValueError("Each teacher needs both paired observation draws")
            group[split] = np.asarray(
                [[group[split][t, o] for o in (0, 1)] for t in teachers],
                dtype=np.float64,
            )
    return teachers, groups


def primary_identity(task, family, train_n, ceiling, numerator, denominator):
    return (
        family == "relu"
        and train_n == 8192
        and ceiling == 1925
        and denominator == "rank2"
        and (
            (task == "radial_m2" and numerator == "width2")
            or (task == "radial_m4" and numerator == "width4")
        )
    )


def bonferroni_intervals(numerator, denominator, threshold, indices, statistics):
    na, nl = statistics.teacher_statistics(numerator, threshold)
    da, dl = statistics.teacher_statistics(denominator, threshold)
    delta = nl - dl
    low, high = np.quantile(delta[indices].mean(1), [0.0125, 0.9875], method="linear")
    alo, ahi = np.quantile(
        na[indices].mean(1) / da[indices].mean(1), [0.0125, 0.9875], method="linear"
    )
    return {
        "bonferroni_comparisons": 2,
        "bonferroni_per_comparison_confidence": 0.975,
        "bonferroni_mean_log_ratio_ci_low": float(low),
        "bonferroni_mean_log_ratio_ci_high": float(high),
        "bonferroni_geometric_ratio_ci_low": float(np.exp(low)),
        "bonferroni_geometric_ratio_ci_high": float(np.exp(high)),
        "bonferroni_arithmetic_ratio_ci_low": float(alo),
        "bonferroni_arithmetic_ratio_ci_high": float(ahi),
    }


def summarize(results, config, statistics=None, *, smoke=False):
    s = statistics or statistics_module()
    teachers, groups = group_confirmation(results, config)
    indices = s.bootstrap_indices(
        len(teachers), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    threshold = config["log_risk_threshold"]
    tables = {
        name: []
        for name in (
            "curves",
            "teacher_curves",
            "paired_ratios",
            "teacher_pairs",
            "primary_comparisons",
            "sample_size_ratios",
            "teacher_sample_size_ratios",
            "budget_changes",
        )
    }

    def context(key):
        return dict(zip(("task", "family", "geometry", "train_n", "ceiling"), key)) | {
            "actual_parameters": groups[key]["actual_parameters"]
        }

    for key, group in sorted(groups.items()):
        for split in ("train", "endpoint"):
            tables["curves"].append(
                context(key)
                | {"split": split}
                | s.summarize_risks(group[split], threshold, indices)
            )
            arithmetic, logs = s.teacher_statistics(group[split], threshold)
            for i, teacher in enumerate(teachers):
                row = context(key) | {
                    "split": split,
                    "teacher": teacher,
                    "observation0_mse": float(group[split][i, 0]),
                    "observation1_mse": float(group[split][i, 1]),
                    "arithmetic_mse": float(arithmetic[i]),
                    "mean_log_mse": float(logs[i]),
                    "geometric_mse": float(np.exp(logs[i])),
                    "log_threshold_clipped_observations": int(
                        (group[split][i] < threshold).sum()
                    ),
                }
                tables["teacher_curves"].append(row)
    for task, family, train_n, p in itertools.product(
        config["tasks"], config["families"], config["train_sizes"], config["primary_p"]
    ):
        for numerator, denominator in PAIRS:
            nk = (task, family, numerator, train_n, p)
            dk = (task, family, denominator, train_n, p)
            first, second = groups[nk], groups[dk]
            summary, delta = s.summarize_pair(
                first["endpoint"], second["endpoint"], threshold, indices
            )
            primary = primary_identity(task, family, train_n, p, numerator, denominator)
            ctx = {
                "task": task,
                "family": family,
                "train_n": train_n,
                "ceiling": p,
                "numerator": numerator,
                "denominator": denominator,
                "numerator_actual_parameters": first["actual_parameters"],
                "denominator_actual_parameters": second["actual_parameters"],
                "primary_comparison": primary,
            }
            tables["paired_ratios"].append(ctx | summary)
            for teacher, value in zip(teachers, delta, strict=True):
                tables["teacher_pairs"].append(
                    ctx
                    | {
                        "teacher": teacher,
                        "mean_log_ratio": float(value),
                        "geometric_ratio": float(np.exp(value)),
                    }
                )
            if primary:
                tables["primary_comparisons"].append(
                    ctx
                    | summary
                    | bonferroni_intervals(
                        first["endpoint"], second["endpoint"], threshold, indices, s
                    )
                    | {
                        "primary_estimand": "teacher_mean_log_risk_ratio",
                        "arithmetic_interval_scope": "Descriptive companion only; never substitute its sign or interval for the two primary log-ratio decisions.",
                    }
                )
    sizes = sorted(config["train_sizes"])
    if (sizes != [2048, 8192] and not smoke) or len(sizes) != 2:
        raise ValueError("Prespecified sample-size contrast requires TRAIN2048 and8192")
    for task, family, geometry, p in itertools.product(
        config["tasks"], config["families"], config["geometries"], config["primary_p"]
    ):
        small, large = (
            groups[task, family, geometry, sizes[0], p],
            groups[task, family, geometry, sizes[1], p],
        )
        if small["actual_parameters"] != large["actual_parameters"]:
            raise ValueError("Sample-size contrast changed parameter count")
        summary, delta = s.summarize_pair(
            large["endpoint"], small["endpoint"], threshold, indices
        )
        ctx = {
            "task": task,
            "family": family,
            "geometry": geometry,
            "ceiling": p,
            "actual_parameters": large["actual_parameters"],
            "numerator_train_n": sizes[1],
            "denominator_train_n": sizes[0],
        }
        tables["sample_size_ratios"].append(ctx | summary)
        for teacher, value in zip(teachers, delta, strict=True):
            tables["teacher_sample_size_ratios"].append(
                ctx
                | {
                    "teacher": teacher,
                    "mean_log_ratio": float(value),
                    "geometric_ratio": float(np.exp(value)),
                }
            )
    budgets = sorted(config["primary_p"])
    for task, family, geometry, n in itertools.product(
        config["tasks"], config["families"], config["geometries"], config["train_sizes"]
    ):
        for small, large in itertools.pairwise(budgets):
            a, b = (
                groups[task, family, geometry, n, large],
                groups[task, family, geometry, n, small],
            )
            summary, _ = s.summarize_pair(
                a["endpoint"], b["endpoint"], threshold, indices
            )
            tables["budget_changes"].append(
                {
                    "task": task,
                    "family": family,
                    "geometry": geometry,
                    "train_n": n,
                    "smaller_ceiling": small,
                    "larger_ceiling": large,
                    "smaller_actual_parameters": b["actual_parameters"],
                    "larger_actual_parameters": a["actual_parameters"],
                }
                | summary
            )
    if len(tables["primary_comparisons"]) != 2 and not smoke:
        raise ValueError("Both prespecified primary comparisons are required")
    return tables


def evaluate_forecasts(forecasts, tables, config):
    curves = {
        (r["task"], r["geometry"], r["family"], r["train_n"]): r
        for r in tables["curves"]
        if r["split"] == "endpoint" and r["ceiling"] == config["withheld_p"]
    }
    predictions = forecasts["curves"]
    expected = {"/".join(map(str, key)) for key in curves}
    if set(predictions) != expected:
        raise ValueError("All96 frozen curve forecasts required")
    errors = []
    for key, p in predictions.items():
        task, geometry, family, n = key.split("/")
        n = int(n)
        observed = curves[task, geometry, family, n]
        if observed["actual_parameters"] != p["withheld_actual_parameters"]:
            raise ValueError("Forecast parameter count changed")
        for model in ("constant", "power", "exponential"):
            predicted = p[model + "_log_risk"]
            error = predicted - observed["mean_log_mse"]
            errors.append(
                {
                    "task": task,
                    "geometry": geometry,
                    "family": family,
                    "train_n": n,
                    "actual_parameters": observed["actual_parameters"],
                    "forecast_model": model,
                    "predicted_log_mse": predicted,
                    "observed_mean_log_mse": observed["mean_log_mse"],
                    "observed_arithmetic_mse": observed["arithmetic_mse"],
                    "observed_geometric_mse": observed["geometric_mse"],
                    "signed_log_error": error,
                    "absolute_log_error": abs(error),
                }
            )
    pairs = {
        (r["task"], r["family"], r["train_n"], r["numerator"], r["denominator"]): r
        for r in tables["paired_ratios"]
        if r["ceiling"] == config["withheld_p"]
    }
    expected = {
        (task, fam, n, num, den)
        for task, fam, n, (num, den) in itertools.product(
            config["tasks"], config["families"], config["train_sizes"], FORECAST_PAIRS
        )
    }
    seen = set()
    predicates = []
    for prediction in forecasts["architecture_comparisons"]:
        key = tuple(
            prediction[name]
            for name in ("task", "family", "train_n", "numerator", "denominator")
        )
        if key in seen or key not in expected:
            raise ValueError("Repeated or unexpected architecture forecast")
        seen.add(key)
        observed = pairs[key]
        delta = observed["mean_log_ratio"]
        predicates.append(
            prediction
            | {
                "observed_log_ratio": delta,
                "observed_geometric_ratio": observed["geometric_ratio"],
                "observed_ratio_of_arithmetic_means": observed[
                    "ratio_of_arithmetic_means"
                ],
                "observed_outcome": (
                    "tie"
                    if delta == 0
                    else "numerator_better" if delta < 0 else "numerator_worse"
                ),
                "correct": (
                    None
                    if delta == 0
                    else bool((delta < 0) == prediction["predicted_numerator_better"])
                ),
            }
        )
    if seen != expected:
        raise ValueError("All60 architecture forecasts required")
    return errors, predicates


def load_completed(root):
    """Verify complete/source/stage/selection/data/result bindings before analysis."""
    root = Path(root).resolve()
    # Never open confirmation result files before the global barrier exists.
    complete = read(root / "complete.json")
    if complete.get("status") != "complete":
        raise ValueError("Completed campaign required before analysis")
    config = read(root / "config.json")
    initialized = read(root / "initialized.json")
    bindings = {}
    if (
        config.get("bootstrap_draws", BOOTSTRAP_DRAWS) != BOOTSTRAP_DRAWS
        or config.get("bootstrap_seed", BOOTSTRAP_SEED) != BOOTSTRAP_SEED
    ):
        raise ValueError(
            "Bootstrap configuration differs from the prespecified analysis"
        )

    def verify(path, digest=None):
        path = Path(path)
        path = path if path.is_absolute() else root / path
        key = str(path.resolve())
        if key not in bindings:
            bindings[key] = sha(path)
        if digest is not None and bindings[key] != digest:
            raise ValueError(f"Bound source/data/result changed: {path}")
        return bindings[key]

    for path, digest in initialized["bindings"].items():
        verify(path, digest)
    initialized_hash = verify("initialized.json")
    for path in ("config.json", "complete.json"):
        verify(path)
    stages = {}
    for stage, count in config["fit_counts"].items():
        marker = read(root / f"{stage}_complete.json")
        if (
            marker.get("status") != "complete"
            or marker["fits"] != count
            or len(marker["results"]) != count
        ):
            raise ValueError("Incomplete stage evidence")
        verify(f"{stage}_complete.json")
        stages[stage] = marker
    if (
        complete["fits"] != sum(config["fit_counts"].values())
        or complete["fits"] != config["expected_fits"]
    ):
        raise ValueError("Global fit count differs from frozen plan")
    selection = read(root / "selection_complete.json")
    verify("selected_recipes.json", selection["selected_sha256"])
    verify("development_complete.json", selection["development_barrier_sha256"])
    verify("selection_complete.json")
    forecast = read(root / "forecast_complete.json")
    forecast_hash = verify("frozen_forecasts.json", forecast["forecast_sha256"])
    verify("bridge_complete.json", forecast["bridge_barrier_sha256"])
    verify("forecast_complete.json")
    if read(root / "frozen_forecasts.json")["selected_sha256"] != sha(
        root / "selected_recipes.json"
    ):
        raise ValueError("Forecast selection binding changed")
    recipes = read(root / "selected_recipes.json")
    choices = read(root / "choices.json")
    choice_ids = {c["id"] for c in choices}
    inventory = {
        (entry["ceiling"], g["geometry"]): g
        for entry in read(root / "parameter_inventory.json")
        for g in entry["geometries"]
    }
    expected_case_keys = set()
    for stage in stages:
        teacher_ids = (
            config["confirmation_teachers"]
            if stage == "confirmation"
            else config["development_teachers"]
        )
        observations = (0, 1) if stage == "confirmation" else (0,)
        budgets = (
            config["primary_p"]
            if stage == "confirmation"
            else [config["bridge_p"]] if stage == "bridge" else config["calibration_p"]
        )
        for task, geometry, family, t, o, n in itertools.product(
            config["tasks"],
            config["geometries"],
            config["families"],
            teacher_ids,
            observations,
            config["train_sizes"],
        ):
            key = "/".join(map(str, (task, geometry, family, n)))
            selected = (
                choice_ids if stage == "development" else [recipes[key]["choice"]]
            )
            expected_case_keys.update(
                (stage, task, geometry, family, t, o, n, c, p)
                for c, p in itertools.product(selected, budgets)
            )
    declared = [
        (stage, path, digest)
        for stage, marker in stages.items()
        for path, digest in marker["results"].items()
    ]

    def load(item):
        stage, path, digest = item
        p = root / path
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Declared result hash changed")
        return stage, str(p.resolve()), digest, json.loads(data)

    rows = []
    seen_cases = set()
    data_receipts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for stage, path, digest, result in pool.map(load, declared):
            bindings[path] = digest
            if (
                result["case"]["stage"] != stage
                or result["initialized_sha256"] != initialized_hash
            ):
                raise ValueError("Result stage/source initialization binding differs")
            if result.get("status") != "complete":
                raise ValueError("Failed fits cannot be silently omitted")
            case = result["case"]
            case_key = tuple(
                case[k]
                for k in (
                    "stage",
                    "task",
                    "geometry",
                    "family",
                    "teacher",
                    "observation",
                    "train_n",
                    "choice",
                    "ceiling",
                )
            )
            if case_key not in expected_case_keys or case_key in seen_cases:
                raise ValueError(
                    "Missing, duplicated or unexpected frozen scientific cell"
                )
            seen_cases.add(case_key)
            if (
                case["choice"] not in choice_ids
                or case["coverage"] != "plane8"
                or case["rebalance"] is not True
            ):
                raise ValueError("Unexpected frozen fitting choice/coverage")
            if stage != "development":
                key = "/".join(
                    str(case[k]) for k in ("task", "geometry", "family", "train_n")
                )
                if case["choice"] != recipes[key]["choice"]:
                    raise ValueError("Endpoint differs from frozen selected recipe")
            parameter_record = inventory[case["ceiling"], case["geometry"]]
            if (
                result["counted_inventory"]["stored_parameters"]
                != parameter_record["stored_parameters"]
            ):
                raise ValueError("Result parameter count differs from frozen inventory")
            release = "confirmation" if stage == "confirmation" else "development"
            expected_data = (
                root
                / "data"
                / release
                / f"{case['task']}_t{case['teacher']}_o{case['observation']}.npz"
            )
            if Path(result["data_path"]).resolve() != expected_data.resolve():
                raise ValueError(
                    "Result data path differs from declared teacher/observation split"
                )
            if expected_data not in data_receipts:
                data_receipts[expected_data] = read(expected_data.with_suffix(".json"))
                verify(expected_data.with_suffix(".json"))
            data_receipt = data_receipts[expected_data]
            if data_receipt["sha256"] != result["data_sha256"] or any(
                data_receipt[k] != case[k] for k in ("task", "teacher", "observation")
            ):
                raise ValueError("Dataset receipt differs from paired scientific cell")
            if data_receipt["release"] != release or data_receipt["endpoint"] != (
                "test" if stage == "confirmation" else "validation"
            ):
                raise ValueError("Dataset endpoint release differs")
            if (
                stage == "confirmation"
                and data_receipt["forecast_sha256"] != forecast_hash
            ):
                raise ValueError(
                    "Confirmation dataset is not bound to the frozen forecast release"
                )
            for name in ("data", "fit", "state", "predictions"):
                verify(result[name + "_path"], result[name + "_sha256"])
            rows.append(result | {"result_path": path})
    if seen_cases != expected_case_keys:
        raise ValueError("The full frozen scientific grid is not complete")
    return config, rows, bindings


def plot_tables(tables, config, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    geometries = config["geometries"]
    colors = dict(zip(geometries, plt.get_cmap("tab10").colors, strict=False))
    labels = {
        "radial_m2": "Quartic radial target",
        "radial_m4": "Degree-eight radial target",
    }

    def ticks(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(config["primary_p"]))
        ax.xaxis.set_major_formatter(
            FixedFormatter([str(p) for p in config["primary_p"]])
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(230, 2050)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Stored parameters P (actual)")

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    for n in config["train_sizes"]:
        for metric in ("arithmetic_mse", "geometric_mse"):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8.3), squeeze=False)
            for i, task in enumerate(config["tasks"]):
                for j, family in enumerate(config["families"]):
                    ax = axes[i, j]
                    for g in geometries:
                        rows = sorted(
                            (
                                r
                                for r in tables["curves"]
                                if r["split"] == "endpoint"
                                and (
                                    r["task"],
                                    r["family"],
                                    r["geometry"],
                                    r["train_n"],
                                )
                                == (task, family, g, n)
                            ),
                            key=lambda r: r["actual_parameters"],
                        )
                        x = [r["actual_parameters"] for r in rows]
                        y = [r[metric] for r in rows]
                        ax.plot(
                            x,
                            y,
                            "o-",
                            color=colors[g],
                            label=g,
                            markersize=3,
                            linewidth=1.4,
                        )
                        ax.fill_between(
                            x,
                            [r[metric + "_ci_low"] for r in rows],
                            [r[metric + "_ci_high"] for r in rows],
                            color=colors[g],
                            alpha=0.055,
                        )
                    ref = THEORY_REFERENCES[task]
                    ax.axhline(
                        ref["population_mse_lower_reference"],
                        color=".55",
                        linestyle=":",
                        linewidth=0.9,
                    )
                    ax.text(
                        0.02,
                        0.03,
                        f"Population reference: {ref['geometry']} only",
                        transform=ax.transAxes,
                        fontsize=8,
                        color=".4",
                    )
                    ax.set_yscale("log")
                    ticks(ax)
                    ax.set_title(f"{labels[task]} · {family}")
                    ax.set_ylabel("Test MSE")
            handles, legend = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                legend,
                loc="lower center",
                ncol=8,
                frameon=False,
                bbox_to_anchor=(0.5, -0.015),
            )
            fig.suptitle(
                f"{'Arithmetic' if metric.startswith('arithmetic') else 'Geometric'} test risk · TRAIN N={n}",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0.025, 1, 0.96))
            save(fig, f"risk_{metric}_n{n}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, task, numerator in zip(
        axes, config["tasks"], ("width2", "width4"), strict=True
    ):
        for n, color in zip(config["train_sizes"], ("#285887", "#b04b36"), strict=True):
            rows = sorted(
                (
                    r
                    for r in tables["paired_ratios"]
                    if (
                        r["task"],
                        r["family"],
                        r["train_n"],
                        r["numerator"],
                        r["denominator"],
                    )
                    == (task, "relu", n, numerator, "rank2")
                ),
                key=lambda r: r["ceiling"],
            )
            x = [r["numerator_actual_parameters"] for r in rows]
            y = [r["geometric_ratio"] for r in rows]
            ax.plot(x, y, "o-", color=color, label=f"TRAIN N={n}")
            ax.fill_between(
                x,
                [r["geometric_ratio_ci_low"] for r in rows],
                [r["geometric_ratio_ci_high"] for r in rows],
                color=color,
                alpha=0.12,
            )
        ax.axhline(1, color=".4", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ticks(ax)
        ax.set_title(f"{labels[task]} · ReLU")
        ax.set_ylabel(f"Geometric risk ratio {numerator}/rank2")
        ax.legend(frameon=False)
    for ax, task in zip(axes, config["tasks"], strict=True):
        primary = next(r for r in tables["primary_comparisons"] if r["task"] == task)
        value = primary["geometric_ratio"]
        ax.errorbar(
            [primary["numerator_actual_parameters"]],
            [value],
            yerr=[
                [max(0, value - primary["bonferroni_geometric_ratio_ci_low"])],
                [max(0, primary["bonferroni_geometric_ratio_ci_high"] - value)],
            ],
            color="black",
            fmt="none",
            capsize=4,
            linewidth=1.2,
        )
    fig.suptitle(
        "Shading: pointwise95%; black terminal bars:97.5% Bonferroni intervals",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "primary_directional_comparisons")


def run(root, output_dir, statistics_source=None, *, smoke=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Analysis output must be a new directory")
    output.mkdir(parents=True)
    try:
        config, results, bindings = load_completed(root)
        s = statistics_module(statistics_source)
        tables = summarize(results, config, s, smoke=smoke)
        forecasts, predicates = evaluate_forecasts(
            read(Path(root) / "frozen_forecasts.json"), tables, config
        )
        tables.update(forecast_errors=forecasts, architecture_predictions=predicates)
        tables["all_endpoints"] = [
            r["case"]
            | {
                "actual_parameters": r["counted_inventory"]["stored_parameters"],
                "train_mse": r["metrics"]["train_mse"],
                "endpoint_mse": r["metrics"]["endpoint_mse"],
                "objective": r["objective"],
                "fit_elapsed_seconds": r["fit_elapsed_seconds"],
                "iterations": r["iterations"],
                "closure_calls": r["closure_calls"],
                "status": r["status"],
                "previous_incomplete_attempts": r.get(
                    "previous_incomplete_attempts", 0
                ),
                "result_path": r["result_path"],
            }
            for r in results
        ]
        for name, rows in tables.items():
            write_csv(output / (name + ".csv"), rows)
        if not smoke:
            plot_tables(tables, config, output)
        summary = {
            "primary_comparisons": tables["primary_comparisons"],
            "forecast_summary": {},
            "architecture_prediction_successes": sum(
                r["correct"] is True for r in predicates
            ),
            "architecture_prediction_failures": [
                r for r in predicates if r["correct"] is False
            ],
            "architecture_prediction_ties": sum(
                r["correct"] is None for r in predicates
            ),
            "population_reference_scope": "Approximate population lower-reference values are annotations for their specified deficient width class only, not universal/finitely sampled test bounds or assertions about rank2/full/adequate width.",
            "population_references": THEORY_REFERENCES,
        }
        for model in ("constant", "power", "exponential"):
            errors = [
                r["absolute_log_error"]
                for r in forecasts
                if r["forecast_model"] == model
            ]
            summary["forecast_summary"][model] = {
                "count": len(errors),
                "mean_absolute_log_error": float(np.mean(errors)),
                "median_absolute_log_error": float(np.median(errors)),
            }
        write(output / "summary.json", summary)
        (output / "README.md").write_text(
            """Every frozen arm, TRAIN size, parameter budget, teacher and observation is retained. Arithmetic raw MSE and thresholded geometric MSE are separate. The teacher is the inference unit: two paired observation draws are averaged within each teacher before resampling. The same teacher resample preserves all paired arms, tasks, budgets and sample sizes. Intervals use10,000 fixed-seed teacher-cluster draws and NumPy linear percentile interpolation. Pointwise95% intervals are descriptive and conditional on the selected procedures. Exactly two prespecified ReLU,N8192,P1925 teacher-mean log-risk-ratio estimands receive97.5% Bonferroni intervals, giving nominal95% simultaneous coverage for this two-comparison family only. Percentile-bootstrap coverage is approximate with eight clusters, not a finite-sample guarantee. Arithmetic ratio intervals are descriptive companions and cannot replace an unfavorable primary log-ratio result; no multiplicity-control claim covers selection between arithmetic and geometric summaries or any remaining descriptive contrasts.

The primary comparisons are width2/rank2 on the quartic radial target and width4/rank2 on the degree-eight radial target. All other width/rank2, rank2/full and width8/full contrasts remain visible at all budgets. Width3 and width5 are adequate-direction conventional controls. These data assess conditional finite-budget learning and do not establish a universal dendritic advantage or a learned asymptotic exponent. The fitted models use generic TRAIN-only initialization and do not receive a known radial profile. Constructed/private-profile models belong to separate evidence.

The N8192/N2048 comparison preserves teacher and observation pairing and uses actual parameter counts. Each sample size has its separately selected frozen recipe, so the comparison is the effect of more TRAIN data plus that declared calibration procedure; it does not hold the selected recipe fixed. Both TRAIN sizes are prefixes of the same archived observations; they are not independent datasets.

All three numerical forecast models are evaluated on the same fresh-teacher aggregate; no test-based forecast selection occurs. Architecture direction forecasts are evaluated without ties being relabeled as successes. Adjacent-budget changes are finite-size diagnostics, not exponents. Stored soma gauge slots remain counted although fixed; width3/width2/width5/width8 actual parameter counts may lie below the common ceiling without padding. Every plot uses actual parameter locations. Population reference lines apply only to their labeled deficient width class and are not certified lower bounds on finite IID test-grid scores.

Fit execution/analysis completion is separate from independent numerical audit status. This analysis verifies declared sources, stage/selection/forecast receipts and all linked result/data/final-state/fit/prediction hashes. It does not replace independent model-forward or high-precision auditing. No failed fit is silently dropped or retrained, and any preserved prior incomplete attempt is listed in the analysis receipt.
"""
        )
        if smoke:
            (output / "README.md").write_text(
                "Engineering smoke analysis only. The input configuration has a reduced sample/teacher/arm grid. All source, data, result, selection, forecast and pairing checks remain active. No primary scientific result, inference interval claim, scaling improvement or theory comparison is inferred from this smoke output. Main primary comparisons may be absent by design; all available smoke outcomes and forecasts remain tabulated.\n"
            )
        receipt = {
            "status": "complete",
            "scope": (
                "engineering_smoke_only" if smoke else "prespecified_primary_campaign"
            ),
            "utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(__file__),
            "statistics_source_sha256": sha(s.__file__),
            "input_bindings": bindings,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_quantile_method": "linear",
            "pointwise_confidence": 0.95,
            "primary_bonferroni_per_comparison_confidence": 0.975,
            "primary_comparison_count": len(tables["primary_comparisons"]),
            "prespecified_main_primary_comparison_count": 2,
            "primary_estimand": "teacher_mean_log_risk_ratio",
            "multiplicity_scope": "Two primary log-ratio estimands only; arithmetic companions are descriptive, bootstrap coverage nominal/approximate.",
            "teacher_count": len(config["confirmation_teachers"]),
            "observations_per_teacher": 2,
            "log_risk_threshold": config["log_risk_threshold"],
            "table_rows": {name: len(rows) for name, rows in tables.items()},
            "failed_results": [],
            "preserved_incomplete_attempts": [
                {"path": r["result_path"], "count": r["previous_incomplete_attempts"]}
                for r in results
                if r.get("previous_incomplete_attempts", 0)
            ],
            "numerical_audit_status": "Separate independent receipt required; execution completion does not imply a pass",
            "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        }
        write(output / "analysis_receipt.json", receipt)
        return receipt
    except Exception:
        write(
            output / "analysis_failure.json",
            {
                "status": "failed",
                "source_sha256": sha(__file__),
                "traceback": traceback.format_exc(),
                "scope": "Preserved analysis failure; no data, fit, result or tolerance was changed.",
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statistics-source", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Explicit reduced-grid engineering check, never primary evidence",
    )
    args = parser.parse_args()
    receipt = run(args.root, args.output_dir, args.statistics_source, smoke=args.smoke)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "table_rows": receipt["table_rows"],
                "source_sha256": receipt["source_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
