"""Teacher-cluster summaries for a completed, immutable rank/width study.

This module does not choose, retrain or replace a model. The two observation
draws are averaged within each teacher before bootstrap resampling. Arithmetic
risk and geometric risk remain distinct throughout every comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 2026136001
PAIRS = (
    ("width1", "full"),
    ("width2", "full"),
    ("rank2", "full"),
    ("width2", "rank2"),
    ("width4", "width2"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        Path(path).touch(exist_ok=False)
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_indices(teachers, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if teachers < 2 or draws < 1:
        raise ValueError("At least two teacher clusters and positive draws required")
    return np.random.default_rng(seed).integers(0, teachers, size=(draws, teachers))


def interval(values):
    low, high = np.quantile(np.asarray(values), [0.025, 0.975], method="linear")
    return float(low), float(high)


def teacher_statistics(risks, threshold):
    """One row per teacher, one column per paired observation draw."""
    risks = np.asarray(risks, dtype=np.float64)
    if risks.ndim != 2 or risks.shape[1] != 2 or risks.shape[0] < 2:
        raise ValueError(
            "Expected two observation draws for each of at least two teachers"
        )
    if (
        not np.isfinite(risks).all()
        or np.any(risks < 0)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError(
            "Finite nonnegative risks and a positive log threshold required"
        )
    return risks.mean(axis=1), np.log(np.maximum(risks, threshold)).mean(axis=1)


def summarize_risks(risks, threshold, indices):
    arithmetic, log_risk = teacher_statistics(risks, threshold)
    arithmetic_low, arithmetic_high = interval(arithmetic[indices].mean(axis=1))
    log_low, log_high = interval(log_risk[indices].mean(axis=1))
    return {
        "teachers": len(arithmetic),
        "observations_per_teacher": 2,
        "arithmetic_mse": float(arithmetic.mean()),
        "arithmetic_mse_ci_low": arithmetic_low,
        "arithmetic_mse_ci_high": arithmetic_high,
        "geometric_mse": float(np.exp(log_risk.mean())),
        "geometric_mse_ci_low": float(np.exp(log_low)),
        "geometric_mse_ci_high": float(np.exp(log_high)),
        "mean_log_mse": float(log_risk.mean()),
        "raw_min_mse": float(np.min(risks)),
        "raw_max_mse": float(np.max(risks)),
        "log_threshold_clipped_endpoints": int(np.sum(np.asarray(risks) < threshold)),
    }


def summarize_pair(numerator, denominator, threshold, indices):
    numerator_mean, numerator_log = teacher_statistics(numerator, threshold)
    denominator_mean, denominator_log = teacher_statistics(denominator, threshold)
    if np.any(denominator_mean <= 0):
        raise ValueError(
            "Arithmetic ratio undefined for a zero-risk denominator teacher"
        )
    delta = numerator_log - denominator_log
    low, high = interval(delta[indices].mean(axis=1))
    arithmetic_boot = numerator_mean[indices].mean(axis=1) / denominator_mean[
        indices
    ].mean(axis=1)
    arithmetic_low, arithmetic_high = interval(arithmetic_boot)
    return {
        "teachers": len(delta),
        "mean_log_ratio": float(delta.mean()),
        "mean_log_ratio_ci_low": low,
        "mean_log_ratio_ci_high": high,
        "geometric_ratio": float(np.exp(delta.mean())),
        "geometric_ratio_ci_low": float(np.exp(low)),
        "geometric_ratio_ci_high": float(np.exp(high)),
        "ratio_of_arithmetic_means": float(
            numerator_mean.mean() / denominator_mean.mean()
        ),
        "ratio_of_arithmetic_means_ci_low": arithmetic_low,
        "ratio_of_arithmetic_means_ci_high": arithmetic_high,
        "teacher_geometric_wins": int(np.sum(delta < 0)),
        "teacher_geometric_ties": int(np.sum(delta == 0)),
        "teacher_arithmetic_wins": int(np.sum(numerator_mean < denominator_mean)),
        "teacher_arithmetic_ties": int(np.sum(numerator_mean == denominator_mean)),
        "paired_observation_wins": int(
            np.sum(np.asarray(numerator) < np.asarray(denominator))
        ),
        "paired_observation_ties": int(
            np.sum(np.asarray(numerator) == np.asarray(denominator))
        ),
        "log_threshold_clipped_endpoints": int(
            np.sum(np.asarray(numerator) < threshold)
            + np.sum(np.asarray(denominator) < threshold)
        ),
    }, delta


def confirmation_arrays(rows, config):
    """Reject absent, repeated or unmatched observations instead of dropping them."""
    teachers = sorted(config["confirmation_teachers"])
    teacher_index = {teacher: index for index, teacher in enumerate(teachers)}
    expected = {
        (task, family, geometry, ceiling)
        for task in config["tasks"]
        for family in config["families"]
        for geometry in config["geometries"]
        for ceiling in config["primary_p"]
    }
    groups = {}
    seen = set()
    for row in rows:
        case = row["case"]
        if case["stage"] != "confirmation" or row["status"] != "complete":
            raise ValueError(
                "Only complete confirmation rows may form the primary curves"
            )
        key = tuple(case[name] for name in ("task", "family", "geometry", "ceiling"))
        identity = key + (case["teacher"], case["observation"])
        if (
            key not in expected
            or case["teacher"] not in teacher_index
            or case["observation"] not in (0, 1)
            or identity in seen
        ):
            raise ValueError("Unexpected or duplicate confirmation cell")
        seen.add(identity)
        group = groups.setdefault(
            key,
            {
                "train": np.full((len(teachers), 2), np.nan),
                "endpoint": np.full((len(teachers), 2), np.nan),
                "parameters": set(),
            },
        )
        i, j = teacher_index[case["teacher"]], case["observation"]
        for split in ("train", "endpoint"):
            group[split][i, j] = row["metrics"][split + "_mse"]
        group["parameters"].add(row["counted_inventory"]["stored_parameters"])
    if set(groups) != expected or any(
        len(group["parameters"]) != 1
        or not np.isfinite(group["endpoint"]).all()
        or not np.isfinite(group["train"]).all()
        for group in groups.values()
    ):
        raise ValueError(
            "Incomplete paired confirmation grid or changing parameter counts"
        )
    for group in groups.values():
        group["actual_parameters"] = group.pop("parameters").pop()
    return teachers, groups


def primary_summaries(rows, config):
    teachers, groups = confirmation_arrays(rows, config)
    indices = bootstrap_indices(len(teachers))
    threshold = config["log_risk_threshold"]
    curves, per_teacher, pairs, pair_teachers, interactions, interaction_teachers = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for key, group in sorted(groups.items()):
        task, family, geometry, ceiling = key
        context = {
            "task": task,
            "family": family,
            "geometry": geometry,
            "ceiling": ceiling,
            "actual_parameters": group["actual_parameters"],
        }
        for split in ("train", "endpoint"):
            curves.append(
                {
                    **context,
                    "split": split,
                    **summarize_risks(group[split], threshold, indices),
                }
            )
            arithmetic, logs = teacher_statistics(group[split], threshold)
            for teacher, mean, log_mean, observations in zip(
                teachers, arithmetic, logs, group[split], strict=True
            ):
                per_teacher.append(
                    {
                        **context,
                        "split": split,
                        "teacher": teacher,
                        "observation0_mse": float(observations[0]),
                        "observation1_mse": float(observations[1]),
                        "arithmetic_mse": float(mean),
                        "mean_log_mse": float(log_mean),
                        "geometric_mse": float(np.exp(log_mean)),
                        "log_threshold_clipped_observations": int(
                            np.sum(observations < threshold)
                        ),
                    }
                )
    paired_deltas = {}
    for task in config["tasks"]:
        for family in config["families"]:
            for numerator, denominator in PAIRS:
                for ceiling in config["primary_p"]:
                    first, second = (
                        groups[(task, family, numerator, ceiling)],
                        groups[(task, family, denominator, ceiling)],
                    )
                    context = {
                        "task": task,
                        "family": family,
                        "numerator": numerator,
                        "denominator": denominator,
                        "ceiling": ceiling,
                        "numerator_actual_parameters": first["actual_parameters"],
                        "denominator_actual_parameters": second["actual_parameters"],
                    }
                    summary, delta = summarize_pair(
                        first["endpoint"], second["endpoint"], threshold, indices
                    )
                    pairs.append({**context, **summary})
                    paired_deltas[(task, family, numerator, denominator, ceiling)] = (
                        delta
                    )
                    for teacher, value in zip(teachers, delta, strict=True):
                        pair_teachers.append(
                            {
                                **context,
                                "teacher": teacher,
                                "mean_log_ratio": float(value),
                                "geometric_ratio": float(np.exp(value)),
                            }
                        )
    for family in config["families"]:
        for ceiling in config["primary_p"]:
            delta = (
                paired_deltas[("mixture_q2", family, "width1", "full", ceiling)]
                - paired_deltas[("mixture_q1", family, "width1", "full", ceiling)]
            )
            low, high = interval(delta[indices].mean(axis=1))
            interactions.append(
                {
                    "family": family,
                    "ceiling": ceiling,
                    "actual_parameters": groups[
                        ("mixture_q2", family, "width1", ceiling)
                    ]["actual_parameters"],
                    "primary_primitive": family == "shunt",
                    "teachers": len(teachers),
                    "interaction_mean_log_ratio": float(delta.mean()),
                    "interaction_ci_low": low,
                    "interaction_ci_high": high,
                    "teacher_positive_signs": int(np.sum(delta > 0)),
                    "teacher_zero_signs": int(np.sum(delta == 0)),
                }
            )
            interaction_teachers.extend(
                {
                    "family": family,
                    "ceiling": ceiling,
                    "teacher": teacher,
                    "interaction_log_ratio": float(value),
                }
                for teacher, value in zip(teachers, delta, strict=True)
            )
    budget_changes = []
    budgets = sorted(config["primary_p"])
    for task in config["tasks"]:
        for family in config["families"]:
            for geometry in config["geometries"]:
                for smaller, larger in zip(budgets[:-1], budgets[1:], strict=True):
                    small = groups[(task, family, geometry, smaller)]
                    large = groups[(task, family, geometry, larger)]
                    summary, _ = summarize_pair(
                        large["endpoint"], small["endpoint"], threshold, indices
                    )
                    budget_changes.append(
                        {
                            "task": task,
                            "family": family,
                            "geometry": geometry,
                            "comparison": "larger_budget_vs_smaller_budget",
                            "smaller_ceiling": smaller,
                            "larger_ceiling": larger,
                            "smaller_actual_parameters": small["actual_parameters"],
                            "larger_actual_parameters": large["actual_parameters"],
                            **summary,
                        }
                    )
    return {
        "curves": curves,
        "teacher_curves": per_teacher,
        "paired_ratios": pairs,
        "teacher_pairs": pair_teachers,
        "interactions": interactions,
        "teacher_interactions": interaction_teachers,
        "budget_changes": budget_changes,
    }


def evaluate_forecasts(forecasts, primary, config):
    curves = {
        (row["task"], row["geometry"], row["family"]): row
        for row in primary["curves"]
        if row["split"] == "endpoint" and row["ceiling"] == config["withheld_p"]
    }
    errors = []
    for key, prediction in forecasts["curves"].items():
        observed = curves[tuple(key.split("/"))]
        if observed["actual_parameters"] != prediction["withheld_actual_parameters"]:
            raise ValueError("Forecast evaluated at a different actual parameter count")
        task, geometry, family = key.split("/")
        for model in ("constant", "power", "exponential"):
            predicted = prediction[model + "_log_risk"]
            errors.append(
                {
                    "task": task,
                    "geometry": geometry,
                    "family": family,
                    "actual_parameters": observed["actual_parameters"],
                    "forecast_model": model,
                    "predicted_log_mse": predicted,
                    "observed_mean_log_mse": observed["mean_log_mse"],
                    "observed_arithmetic_mse": observed["arithmetic_mse"],
                    "observed_geometric_mse": observed["geometric_mse"],
                    "signed_log_error": predicted - observed["mean_log_mse"],
                    "absolute_log_error": abs(predicted - observed["mean_log_mse"]),
                }
            )
    pair_lookup = {
        (row["task"], row["family"], row["numerator"], row["denominator"]): row
        for row in primary["paired_ratios"]
        if row["ceiling"] == config["withheld_p"]
    }
    predicates = []
    for prediction in forecasts["architecture_predictions"]:
        observed = pair_lookup[
            tuple(
                prediction[name]
                for name in ("task", "family", "numerator", "denominator")
            )
        ]
        delta = observed["mean_log_ratio"]
        predicates.append(
            {
                **prediction,
                "observed_log_ratio": delta,
                "observed_geometric_ratio": observed["geometric_ratio"],
                "observed_ratio_of_arithmetic_means": observed[
                    "ratio_of_arithmetic_means"
                ],
                "observed_outcome": (
                    "tie"
                    if delta == 0
                    else ("numerator_better" if delta < 0 else "numerator_worse")
                ),
                "correct": (
                    None
                    if delta == 0
                    else bool((delta < 0) == prediction["predicted_numerator_better"])
                ),
            }
        )
    if len(errors) != len(curves) * 3 or len(predicates) != 36:
        raise ValueError(
            "All three forecasts and all36 frozen architecture predicates required"
        )
    return errors, predicates


def diagnostic_summaries(rows):
    raw, contrasts = [], []
    lookup = {}
    for row in rows:
        case = row["case"]
        if case["stage"] != "diagnostic":
            raise ValueError("Diagnostic stage required")
        raw.append(
            {
                **case,
                "status": row["status"],
                "objective": row.get("objective"),
                "actual_parameters": row.get("counted_inventory", {}).get(
                    "stored_parameters"
                ),
                "fit_elapsed_seconds": row.get("fit_elapsed_seconds"),
                "case_elapsed_seconds": row.get("elapsed_seconds"),
                "closure_calls": row.get("closure_calls"),
                "iterations": row.get("iterations"),
                "train_mse": row.get("metrics", {}).get("train_mse"),
                "validation_mse": row.get("metrics", {}).get("endpoint_mse"),
                "traceback": row.get("traceback"),
            }
        )
        key = tuple(
            case[name]
            for name in (
                "task",
                "family",
                "geometry",
                "teacher",
                "choice",
                "coverage",
                "rebalance",
                "ceiling",
            )
        )
        if key in lookup:
            raise ValueError("Repeated diagnostic cell")
        lookup[key] = row
    for key, numerator in lookup.items():
        task, family, geometry, teacher, choice, coverage, rebalance, ceiling = key
        candidates = []
        if geometry == "rank2" and rebalance:
            candidates.append(("QR_vs_restart", key[:6] + (False, ceiling)))
        if choice == "ols_ridge1e-05" and coverage == "axial" and geometry == "rank2":
            candidates.append(
                (
                    "OLS_axial_vs_OLS_legacy",
                    (
                        task,
                        family,
                        geometry,
                        teacher,
                        choice,
                        "legacy",
                        rebalance,
                        ceiling,
                    ),
                )
            )
        if choice == "stein_ridge1e-05" and coverage == "axial":
            candidates.append(
                (
                    "Stein_axial_vs_OLS_axial",
                    (
                        task,
                        family,
                        geometry,
                        teacher,
                        "ols_ridge1e-05",
                        "axial",
                        rebalance,
                        ceiling,
                    ),
                )
            )
        for contrast, denominator_key in candidates:
            if denominator_key not in lookup:
                raise ValueError("Prespecified paired diagnostic denominator absent")
            denominator = lookup[denominator_key]
            complete = numerator["status"] == denominator["status"] == "complete"
            context = {
                "contrast": contrast,
                "task": task,
                "family": family,
                "geometry": geometry,
                "teacher": teacher,
                "choice": choice,
                "coverage": coverage,
                "rebalance": rebalance,
                "ceiling": ceiling,
                "numerator_status": numerator["status"],
                "denominator_status": denominator["status"],
            }
            if complete:
                for metric in (
                    "objective",
                    "fit_elapsed_seconds",
                    "closure_calls",
                    "iterations",
                ):
                    first, second = numerator[metric], denominator[metric]
                    context["numerator_" + metric], context["denominator_" + metric] = (
                        first,
                        second,
                    )
                    context[metric + "_ratio"] = (
                        float(first / second) if second > 0 else None
                    )
                context["actual_parameters"] = numerator.get(
                    "counted_inventory", {}
                ).get("stored_parameters")
                if numerator["ridge"] != denominator["ridge"]:
                    raise ValueError(
                        "Diagnostic comparison changes the objective penalty"
                    )
                context["ridge"] = numerator["ridge"]
            contrasts.append(context)
    return raw, contrasts


def aggregate_diagnostics(paired):
    """Descriptive development-teacher summaries, retaining failed pairs."""
    groups = {}
    names = (
        "contrast",
        "task",
        "family",
        "geometry",
        "choice",
        "coverage",
        "rebalance",
        "ceiling",
    )
    for row in paired:
        groups.setdefault(tuple(row[name] for name in names), []).append(row)
    results = []
    for key, rows in sorted(groups.items()):
        failed = sum(
            row["numerator_status"] != "complete"
            or row["denominator_status"] != "complete"
            for row in rows
        )
        result = {
            **dict(zip(names, key, strict=True)),
            "teacher_pairs": len(rows),
            "failed_pairs": failed,
        }
        if not failed:
            for metric in ("objective", "fit_elapsed_seconds"):
                numerator = np.array([row["numerator_" + metric] for row in rows])
                denominator = np.array([row["denominator_" + metric] for row in rows])
                if np.any(numerator <= 0) or np.any(denominator <= 0):
                    raise ValueError("Positive complete objectives and timing required")
                ratios = numerator / denominator
                result[metric + "_geometric_ratio"] = float(
                    np.exp(np.log(ratios).mean())
                )
                result[metric + "_ratio_of_means"] = float(
                    numerator.mean() / denominator.mean()
                )
                result[metric + "_teacher_wins"] = int(np.sum(ratios < 1))
                result[metric + "_teacher_ties"] = int(np.sum(ratios == 1))
        results.append(result)
    return results


def load_snapshot(root):
    initialized = read(root / "initialized.json")
    for name, digest in initialized["bindings"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    package_name = "width_analysis_snapshot_" + sha(root / "initialized.json")[:16]
    specification = importlib.util.spec_from_file_location(
        package_name,
        root / "source/__init__.py",
        submodule_search_locations=[str(root / "source")],
    )
    package = importlib.util.module_from_spec(specification)
    sys.modules[package_name] = package
    specification.loader.exec_module(package)
    return importlib.import_module(package_name + ".rank_width_campaign")


def failure_inventory(root, campaign, config, stages=None):
    """Use declared result paths; avoid recursively walking checkpoint trees."""
    paths, failures, incomplete, inner_failures = [], [], [], []
    requested_stages = tuple(config["fit_counts"] if stages is None else stages)
    recipes = (
        read(root / "selected_recipes.json")
        if any(stage in ("bridge", "confirmation") for stage in requested_stages)
        and (root / "selected_recipes.json").exists()
        else None
    )
    for stage in requested_stages:
        marker = root / f"{stage}_complete.json"
        if marker.exists():
            paths.extend(root / name for name in read(marker)["results"])
        elif stage not in ("bridge", "confirmation") or recipes is not None:
            paths.extend(
                root / "cases" / campaign.case_key(case) / "result.json"
                for case in campaign.case_grid(config, stage, recipes)
            )
    results, missing = [], []
    for path in sorted(paths):
        if not path.exists():
            missing.append(str(path))
            # Only an unfinished case needs a shallow attempt-directory query.
            attempts = (
                sorted(path.parent.glob("attempt_*")) if path.parent.exists() else []
            )
            incomplete.extend(str(attempt) for attempt in attempts)
        else:
            row = read(path)
            results.append(path)
            if row["status"] != "complete":
                failures.append(
                    {
                        "path": str(path),
                        "sha256": sha(path),
                        "case": row["case"],
                        "status": row["status"],
                        "traceback": row.get("traceback"),
                    }
                )
            attempts = [
                path.parent / f"attempt_{index:03d}"
                for index in range(row["previous_incomplete_attempts"])
            ]
            incomplete.extend(str(attempt) for attempt in attempts)
            attempts.append(Path(row["attempt_path"]))
        for attempt in attempts:
            candidates = [attempt / "fit/failure.json"] + [
                attempt / "fit" / f"stage_{index:03d}" / "failure.json"
                for index in range(config["stages"])
            ]
            for candidate in candidates:
                if candidate.exists():
                    inner_failures.append(
                        {
                            "path": str(candidate),
                            "sha256": sha(candidate),
                            "details": read(candidate),
                        }
                    )
    return {
        "result_files": len(results),
        "result_paths": [str(path) for path in results],
        "missing_declared_results": missing,
        "failed_results": failures,
        "optimizer_failure_receipts": inner_failures,
        "preserved_incomplete_attempts": incomplete,
    }


def diagnostic_figures(output, paired, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrast_names = (
        "QR_vs_restart",
        "OLS_axial_vs_OLS_legacy",
        "Stein_axial_vs_OLS_axial",
    )
    names = ("task", "family", "geometry", "choice", "coverage", "rebalance")
    for metric, label in (
        ("objective", "Complete TRAIN objective ratio"),
        ("fit_elapsed_seconds", "Training wall-time ratio"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.5, 8))
        for axis, contrast in zip(axes, contrast_names, strict=True):
            groups = [row for row in summary if row["contrast"] == contrast]
            for index, row in enumerate(groups):
                matches = [
                    item
                    for item in paired
                    if item["contrast"] == contrast
                    and all(item[name] == row[name] for name in names)
                ]
                if not row["failed_pairs"]:
                    axis.scatter(
                        [item[metric + "_ratio"] for item in matches],
                        [index] * len(matches),
                        color="#0072B2",
                        alpha=0.45,
                        s=15,
                    )
                    axis.scatter(
                        row[metric + "_geometric_ratio"],
                        index,
                        color="black",
                        marker="|",
                        s=100,
                    )
            labels = [
                f"{row['task']} / {row['family']} / {row['geometry']}\n{row['choice'].split('_ridge')[0]} {row['coverage']} / {'QR' if row['rebalance'] else 'restart'}"
                for row in groups
            ]
            axis.set_yticks(range(len(labels)), labels, fontsize=6)
            axis.axvline(1, color="black", linewidth=0.7)
            axis.set(xscale="log", xlabel=label, title=contrast.replace("_", " "))
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            "Development diagnostics · dots are paired teachers; bars are geometric means"
        )
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(output / f"diagnostic_{metric}.{extension}", dpi=180)
        plt.close(figure)


def analyze_diagnostics(root, output):
    """A sealed development-only release; never reads confirmation outcomes."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        campaign = load_snapshot(root)
        campaign.verify(root)
        config = read(root / "config.json")
        failures = failure_inventory(root, campaign, config, stages=("diagnostic",))
        write(output / "failures.json", failures)
        rows = campaign.stage_results(root, "diagnostic", config)
        marker = root / "diagnostic_complete.json"
        if (
            not marker.exists()
            or failures["failed_results"]
            or failures["missing_declared_results"]
        ):
            raise ValueError("Complete sealed diagnostic stage required")
        paths = [root / "initialized.json", root / "config.json", marker] + [
            Path(path) for path in failures["result_paths"]
        ]
        bindings = {str(path): sha(path) for path in paths}
        raw, pairs = diagnostic_summaries(rows)
        summary = aggregate_diagnostics(pairs)
        tables = {
            "diagnostic_cells": raw,
            "diagnostic_pairs": pairs,
            "diagnostic_summary": summary,
        }
        for name, items in tables.items():
            write_csv(output / f"{name}.csv", items)
        write(output / "summaries.json", tables)
        diagnostic_figures(output, pairs, summary)
        (output / "README.md").write_text(
            "# Development-only optimization diagnostics\n\nAll80 prespecified development endpoints are retained. Paired comparisons share teacher, response, family, geometry, parameter count and ridge coefficient. Objective ratios use the complete TRAIN MSE-plus-standardized-readout-ridge objective. Time ratios are observed training wall time; two development teachers provide descriptive comparisons only. QR versus restart and initialization contrasts remain separate. These results do not choose a model, change the frozen main campaign, establish a family lower bound or estimate a learned scaling exponent. Confirmation outcomes and forecasts are not loaded by this release.\n"
        )
        if any(sha(path) != digest for path, digest in bindings.items()):
            raise ValueError("Diagnostic evidence changed during analysis")
        receipt = {
            "status": "complete",
            "scope": "development-only optimization diagnostics",
            "analysis_source_sha256": sha(__file__),
            "input_bindings": bindings,
            "table_rows": {name: len(items) for name, items in tables.items()},
            "outputs": {
                path.name: sha(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
        write(output / "analysis_receipt.json", receipt)
        return receipt
    except Exception as error:
        write(
            output / "analysis_failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "analysis_source_sha256": sha(__file__),
            },
        )
        raise


def make_figures(output, summaries, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "full": "#252525",
        "rank2": "#0072B2",
        "width1": "#D55E00",
        "width2": "#009E73",
        "width4": "#CC79A7",
    }
    for average in ("arithmetic", "geometric"):
        average_label = (
            "Thresholded geometric"
            if average == "geometric"
            and any(
                row["log_threshold_clipped_endpoints"]
                for row in summaries["curves"]
                if row["split"] == "endpoint"
            )
            else average.capitalize()
        )
        figure, axes = plt.subplots(3, 3, figsize=(12.3, 10.1), squeeze=False)
        for i, task in enumerate(config["tasks"]):
            for j, family in enumerate(config["families"]):
                axis = axes[i, j]
                for geometry in config["geometries"]:
                    rows = sorted(
                        [
                            r
                            for r in summaries["curves"]
                            if r["task"] == task
                            and r["family"] == family
                            and r["geometry"] == geometry
                            and r["split"] == "endpoint"
                        ],
                        key=lambda r: r["actual_parameters"],
                    )
                    x = [r["actual_parameters"] for r in rows]
                    y = np.array([r[average + "_mse"] for r in rows])
                    low = np.array([r[average + "_mse_ci_low"] for r in rows])
                    high = np.array([r[average + "_mse_ci_high"] for r in rows])
                    axis.plot(
                        x,
                        y,
                        marker="o",
                        markersize=3,
                        color=colors[geometry],
                        label=geometry,
                    )
                    axis.fill_between(x, low, high, color=colors[geometry], alpha=0.10)
                axis.set(
                    xscale="log",
                    yscale="log",
                    title=f"{task} · {family}",
                    xlabel="Actual stored parameters",
                    ylabel=f"{average_label} test MSE",
                )
                axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(
            f"Learned risk · {average_label.lower()} mean · teacher-cluster 95% intervals"
        )
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(output / f"risk_curves_{average}.{extension}", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for axis, family in zip(axes, config["families"], strict=True):
        rows = sorted(
            [r for r in summaries["interactions"] if r["family"] == family],
            key=lambda r: r["actual_parameters"],
        )
        x, y = [r["actual_parameters"] for r in rows], [
            r["interaction_mean_log_ratio"] for r in rows
        ]
        axis.plot(x, y, marker="o", color="#0072B2")
        axis.fill_between(
            x,
            [r["interaction_ci_low"] for r in rows],
            [r["interaction_ci_high"] for r in rows],
            color="#0072B2",
            alpha=0.18,
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(
            xscale="log",
            xlabel="Actual stored parameters",
            ylabel="q2−q1 paired log-ratio interaction",
            title=family + (" (primary)" if family == "shunt" else ""),
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Interaction: log(width1/full) on mixture q2 minus mixture q1")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"rank_interaction.{extension}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(3, 3, figsize=(12.3, 10.1), squeeze=False)
    for i, task in enumerate(config["tasks"]):
        for j, family in enumerate(config["families"]):
            axis = axes[i, j]
            for numerator, denominator in PAIRS:
                rows = sorted(
                    [
                        row
                        for row in summaries["paired_ratios"]
                        if (
                            row["task"],
                            row["family"],
                            row["numerator"],
                            row["denominator"],
                        )
                        == (task, family, numerator, denominator)
                    ],
                    key=lambda row: row["ceiling"],
                )
                axis.plot(
                    [row["ceiling"] for row in rows],
                    [row["geometric_ratio"] for row in rows],
                    marker="o",
                    markersize=3,
                    label=f"{numerator}/{denominator}",
                )
            axis.axhline(1, color="black", linewidth=0.7)
            axis.set(
                xscale="log",
                yscale="log",
                title=f"{task} · {family}",
                xlabel="Common P ceiling (actual counts in CSV)",
                ylabel="Paired geometric test-risk ratio",
            )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Paired risk ratios · all teachers and both observation draws retained"
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"paired_risk_ratios.{extension}", dpi=180)
    plt.close(figure)


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        campaign = load_snapshot(root)
        config = read(root / "config.json")
        failures = failure_inventory(root, campaign, config)
        write(output / "failures.json", failures)
        campaign.verify_forecast(root)
        complete = read(root / "complete.json")
        if (
            complete["status"] != "complete"
            or complete["fits"] != config["expected_fits"]
        ):
            raise ValueError(
                "A complete frozen campaign is required for primary inference"
            )
        stages = {
            stage: campaign.stage_results(root, stage, config)
            for stage in config["fit_counts"]
        }
        all_rows = [row for rows in stages.values() for row in rows]
        if (
            failures["failed_results"]
            or failures["missing_declared_results"]
            or len(all_rows) != failures["result_files"]
        ):
            raise ValueError(
                "Failed or missing declared endpoints cannot be silently excluded"
            )
        input_paths = [
            root / name
            for name in (
                "initialized.json",
                "config.json",
                "complete.json",
                "selected_recipes.json",
                "selection_complete.json",
                "frozen_forecasts.json",
                "forecast_complete.json",
            )
        ]
        input_paths += [root / f"{stage}_complete.json" for stage in stages]
        input_paths += [Path(path) for path in failures["result_paths"]]
        bindings = {str(path): sha(path) for path in input_paths}
        primary = primary_summaries(stages["confirmation"], config)
        forecast_errors, predicates = evaluate_forecasts(
            read(root / "frozen_forecasts.json"), primary, config
        )
        diagnostic_rows, diagnostic_pairs = diagnostic_summaries(stages["diagnostic"])
        tables = {
            **primary,
            "forecast_errors": forecast_errors,
            "architecture_predictions": predicates,
            "diagnostic_cells": diagnostic_rows,
            "diagnostic_pairs": diagnostic_pairs,
            "diagnostic_summary": aggregate_diagnostics(diagnostic_pairs),
        }
        tables["all_endpoints"] = [
            {
                **row["case"],
                "status": row["status"],
                "actual_parameters": row["counted_inventory"]["stored_parameters"],
                **row["metrics"],
                "objective": row["objective"],
                "fit_elapsed_seconds": row["fit_elapsed_seconds"],
                "closure_calls": row["closure_calls"],
                "iterations": row["iterations"],
            }
            for row in all_rows
        ]
        for name, rows in tables.items():
            write_csv(output / f"{name}.csv", rows)
        write(output / "summaries.json", tables)
        make_figures(output, primary, config)
        diagnostic_figures(output, diagnostic_pairs, tables["diagnostic_summary"])
        boundaries = """# Rank/width evidence boundaries

These summaries retain every frozen endpoint and every architecture/family arm. Arithmetic risk is the mean of each teacher's two-observation arithmetic mean. Geometric risk averages the two log risks within teacher and then averages teachers; the frozen log-risk threshold applies only before logarithms. Raw risks and clipping counts remain in the tables. The two summaries can disagree, and geometric wins do not imply an arithmetic improvement.

All intervals resample whole teachers, with their two observations and all paired architectures/ranks carried together. They are pointwise percentile intervals from10,000 fixed-seed draws, using linear interpolation of empirical quantiles and no multiplicity adjustment. Four or eight teachers give limited uncertainty resolution, conditional on the frozen selected methods. Observation-level wins are descriptive counts, not independent training replications. If any log-risk clipping occurs, the geometric risks, geometric sign counts and log-ratio intervals are thresholded summaries; raw arithmetic risks and arithmetic/observation signs remain separate.

The primary interaction is the within-teacher mixture q2 minus q1 difference in log(width1/full), with shunt the primary primitive. This is a different contrast from the earlier rank2/rank1 interaction and its numerical value is not a direct continuation of that statistic. Multiple separately directed nodes per raw block evade the one-projection information restriction. These finite-data learned curves do not establish a global or learned asymptotic exponent. The mixture task changes private response and orientation; quadratic teacher draws change orientation only. All labels here are noiseless, and results remain conditional on the initializer, standardized-feature ridge and finite optimization exposure.

All three prespecified larger-budget risk forecasts and all36 architecture predictions are evaluated. A forecast slope is a finite-range prediction, not an established scaling law. The P1925 timing-only canary is outside quality-based selection and forecasting. No model, primitive, recipe, endpoint or teacher is selected by this analysis.

Diagnostics compare the complete TRAIN objective with the same ridge coefficient and report paired timing/closure exposure. They describe optimizer and initializer behavior, not an architecture lower bound. Independent numerical model replay is a separate audit; this analysis summarizes source-bound stored metrics. Failed attempts and incomplete preemption attempts are listed in failures.json and never silently replaced by an analysis choice.
"""
        (output / "README.md").write_text(boundaries)
        if any(sha(path) != digest for path, digest in bindings.items()):
            raise ValueError("Campaign evidence changed during analysis")
        receipt = {
            "status": "complete",
            "utc": datetime.now(timezone.utc).isoformat(),
            "analysis_source_sha256": sha(__file__),
            "input_bindings": bindings,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_quantile_method": "linear",
            "bootstrap_interval": "Pointwise95% teacher-cluster percentile; no multiplicity adjustment",
            "log_risk_threshold": config["log_risk_threshold"],
            "inference_unit": "teacher, retaining both paired observation draws",
            "confirmation_teacher_count": len(config["confirmation_teachers"]),
            "table_rows": {name: len(rows) for name, rows in tables.items()},
            "failed_results": len(failures["failed_results"]),
            "optimizer_failure_receipts": len(failures["optimizer_failure_receipts"]),
            "preserved_incomplete_attempts": len(
                failures["preserved_incomplete_attempts"]
            ),
            "outputs": {
                str(path.relative_to(output)): sha(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
        write(output / "analysis_receipt.json", receipt)
        return receipt
    except Exception as error:
        write(
            output / "analysis_failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "analysis_source_sha256": sha(__file__),
                "scope": "Partial diagnostic artifacts retained; no complete primary inference receipt issued.",
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diagnostic-only", action="store_true")
    arguments = parser.parse_args()
    receipt = (analyze_diagnostics if arguments.diagnostic_only else analyze)(
        arguments.root, arguments.output_dir
    )
    print(json.dumps({"status": receipt["status"], "tables": receipt["table_rows"]}))


if __name__ == "__main__":
    main()
