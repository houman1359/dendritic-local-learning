"""Transparent descriptive summaries of the fixed IID rank campaign."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def teacher_interval(values, exponentiate=False):
    values = np.asarray(values)
    samples = np.array(list(itertools.product(range(len(values)), repeat=len(values))))
    interval = np.quantile(values[samples].mean(1), [0.025, 0.975])
    point = values.mean()
    if exponentiate:
        point, interval = np.exp(point), np.exp(interval)
    return {
        "estimate": float(point),
        "descriptive_bootstrap_low": float(interval[0]),
        "descriptive_bootstrap_high": float(interval[1]),
        "teachers": len(values),
        "bootstrap_scope": (
            f"Exact empirical teacher resampling; n={len(values)}, descriptive only."
            if len(values) > 1
            else "One teacher: degenerate resampling interval, no inference."
        ),
    }


def run(root, output):
    complete = read(root / "campaign_complete.json")
    if complete["fits"] != 1980:
        raise ValueError("Unexpected complete inventory")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((root / "fits").rglob("evaluation.json")):
        row = read(path)
        endpoint = "validation" if row["stage"] in ["development", "bridge"] else "test"
        metrics = row.pop(endpoint)
        fit = read(row["fit_file"])
        row.update(
            metrics,
            endpoint=endpoint,
            evaluation_file=str(path),
            ridge=fit["config"]["ridge"],
            elapsed_seconds=fit["elapsed_seconds"],
            gradient_l2=fit["terminal_unscaled_gradient_l2"],
            terminal_objective=fit["terminal_objective"],
            warm_objective=fit["warm_start_objective"],
            projection_movement=fit["effective_projection_movement_l2"],
            iterations=fit["iterations"],
            closures=fit["closure_calls"],
        )
        for key in ["state", "capacities"]:
            row[key] = json.dumps(row[key])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if (
        len(frame) != complete["fits"]
        or not np.isfinite(frame.observed_mse).all()
        or not (frame.observed_mse > 0).all()
    ):
        raise ValueError("Incomplete, nonpositive or nonfinite result table")
    confirmation = frame[frame.stage == "confirmation"]
    primary = confirmation[confirmation.variant == "primary"]
    group = [
        "variant",
        "family",
        "architecture",
        "intrinsic_rank",
        "parameters",
        "branches",
        "train_n",
    ]
    means = (
        confirmation.groupby(group)
        .agg(
            mean_mse=("observed_mse", "mean"),
            median_mse=("observed_mse", "median"),
            min_mse=("observed_mse", "min"),
            max_mse=("observed_mse", "max"),
            mean_clean_mse=("clean_teacher_mse", "mean"),
            median_gradient=("gradient_l2", "median"),
            fits=("observed_mse", "size"),
        )
        .reset_index()
    )
    contrasts = []
    for variant, subset in confirmation[confirmation.variant != "same_m"].groupby(
        "variant"
    ):
        for (family, p), cell in subset.groupby(["family", "parameters"]):
            table = cell.pivot(
                index=["teacher", "seed", "intrinsic_rank"],
                columns="architecture",
                values="observed_mse",
            )
            if (
                set(table.columns) != {"full", "rank1", "rank2"}
                or table.isna().any().any()
            ):
                raise ValueError("Missing paired architecture endpoint")
            if set(table.index.get_level_values("intrinsic_rank")) != {1, 2}:
                raise ValueError("Missing paired target rank")
            if not (
                table.groupby(level=["teacher", "intrinsic_rank"]).size() == 2
            ).all():
                raise ValueError("Missing paired observation replicate")
            for (teacher, seed, q), values in table.iterrows():
                for numerator, denominator in [
                    ("rank1", "full"),
                    ("rank2", "full"),
                    ("rank2", "rank1"),
                ]:
                    contrasts.append(
                        {
                            "variant": variant,
                            "family": family,
                            "parameters": p,
                            "teacher": teacher,
                            "seed": seed,
                            "intrinsic_rank": q,
                            "numerator": numerator,
                            "denominator": denominator,
                            "ratio": values[numerator] / values[denominator],
                            "log_ratio": np.log(
                                values[numerator] / values[denominator]
                            ),
                        }
                    )
    contrasts = pd.DataFrame(contrasts)
    teacher_contrasts = (
        contrasts.groupby(
            [
                "variant",
                "family",
                "parameters",
                "teacher",
                "intrinsic_rank",
                "numerator",
                "denominator",
            ]
        )
        .agg(
            mean_log_ratio=("log_ratio", "mean"),
            wins=("ratio", lambda x: int((x < 1).sum())),
            replicates=("ratio", "size"),
        )
        .reset_index()
    )
    teacher_contrasts["geometric_ratio"] = np.exp(teacher_contrasts.mean_log_ratio)
    interactions = []
    target = contrasts[
        (contrasts.numerator == "rank2") & (contrasts.denominator == "rank1")
    ]
    for (variant, family, p, teacher), cell in target.groupby(
        ["variant", "family", "parameters", "teacher"]
    ):
        qs = cell.groupby("intrinsic_rank").log_ratio.mean()
        interactions.append(
            {
                "variant": variant,
                "family": family,
                "parameters": p,
                "teacher": teacher,
                "D": float(qs.loc[1] - qs.loc[2]),
            }
        )
    interactions = pd.DataFrame(interactions)
    interaction_summary = []
    for (variant, family, p), cell in interactions.groupby(
        ["variant", "family", "parameters"]
    ):
        interaction_summary.append(
            {
                "variant": variant,
                "family": family,
                "parameters": p,
                "positive_teachers": int((cell.D > 0).sum()),
                **teacher_interval(cell.D),
            }
        )
    interaction_summary = pd.DataFrame(interaction_summary)
    same_branch = []
    same_candidates = confirmation[confirmation.variant.isin(["primary", "same_m"])]
    for (family, q, teacher, seed, m), cell in same_candidates[
        same_candidates.branches.isin([24, 48])
    ].groupby(["family", "intrinsic_rank", "teacher", "seed", "branches"]):
        by = cell.set_index("architecture")
        for architecture in ["rank1", "rank2"]:
            same_branch.append(
                {
                    "family": family,
                    "intrinsic_rank": q,
                    "teacher": teacher,
                    "seed": seed,
                    "branches": m,
                    "architecture": architecture,
                    "parameters": int(by.loc[architecture, "parameters"]),
                    "full_parameters": int(by.loc["full", "parameters"]),
                    "ratio_to_full": float(
                        by.loc[architecture, "observed_mse"]
                        / by.loc["full", "observed_mse"]
                    ),
                }
            )
    same_branch = pd.DataFrame(same_branch)
    predictions = []
    frozen = read(root / "frozen_forecasts.json")
    for forecast in frozen["forecasts"]:
        cell = primary[
            (primary.architecture == forecast["architecture"])
            & (primary.family == forecast["family"])
            & (primary.intrinsic_rank == forecast["intrinsic_rank"])
            & (primary.parameters == forecast["withheld_p"])
        ]
        actual = float(cell.observed_mse.mean())
        predictions.append(
            {
                **forecast,
                "actual_fresh_teacher_mean_mse": actual,
                "predicted_over_actual": forecast["predicted_mse"] / actual,
                "absolute_log_error": abs(
                    np.log(max(forecast["predicted_mse"], 1e-300) / actual)
                ),
            }
        )
    forecasts = pd.DataFrame(predictions)
    architecture_predictions = []
    for prediction in frozen["architecture_predictions"]:
        cell = primary[
            (primary.family == prediction["family"])
            & (primary.intrinsic_rank == prediction["intrinsic_rank"])
            & (primary.parameters == 485)
        ]
        risks = cell.groupby("architecture").observed_mse.mean()
        architecture_predictions.append(
            {
                **prediction,
                "observed_best_architecture": risks.idxmin(),
                "correct": prediction["predicted_architecture"] == risks.idxmin(),
                "selected_over_best": risks.loc[prediction["predicted_architecture"]]
                / risks.min(),
            }
        )
    architecture_predictions = pd.DataFrame(architecture_predictions)
    development = frame[frame.stage == "development"]
    optimizer_pairs = development.pivot(
        index=[
            "teacher",
            "seed",
            "intrinsic_rank",
            "family",
            "architecture",
            "parameters",
            "ridge",
        ],
        columns="optimizer",
        values=["observed_mse", "terminal_objective", "gradient_l2", "elapsed_seconds"],
    )
    optimizer_pairs.columns = ["_".join(c) for c in optimizer_pairs.columns]
    optimizer_pairs = optimizer_pairs.reset_index()
    optimizer_pairs["reduced_over_joint_validation_mse"] = (
        optimizer_pairs.observed_mse_reduced / optimizer_pairs.observed_mse_joint
    )
    selections = pd.DataFrame(
        [
            {
                "condition": key,
                "optimizer": value["choice"]["optimizer"],
                "choice_id": value["choice"]["id"],
                "ridge": value["choice"]["fit"]["ridge"],
                "scores": json.dumps(value["scores"]),
            }
            for key, value in read(root / "selected_recipes.json")["selections"].items()
        ]
    )
    tables = {
        "all_fits": frame,
        "mean_curves": means,
        "seed_contrasts": contrasts,
        "teacher_contrasts": teacher_contrasts,
        "rank_interactions": interactions,
        "interaction_summary": interaction_summary,
        "same_branch_controls": same_branch,
        "size_forecasts": forecasts,
        "architecture_forecasts": architecture_predictions,
        "optimizer_calibration": optimizer_pairs,
        "selected_recipes": selections,
    }
    with pd.ExcelWriter(output / "rank_iid_results.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_csv(output / (name + ".csv"), index=False)
            excel = table.copy()
            for col in excel.columns:
                if excel[col].dtype == object:
                    excel[col] = excel[col].map(
                        lambda value: (
                            json.dumps(value)
                            if isinstance(value, (list, dict))
                            else value
                        )
                    )
            excel.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    colors = {"full": "#444444", "rank1": "#1976a3", "rank2": "#c05032"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, q in enumerate([1, 2]):
        for col, family in enumerate(["shunt", "relu", "tanh"]):
            ax = axes[row, col]
            for architecture in ["full", "rank1", "rank2"]:
                cell = primary[
                    (primary.family == family)
                    & (primary.intrinsic_rank == q)
                    & (primary.architecture == architecture)
                ]
                for _, teacher in cell.groupby("teacher"):
                    ys = teacher.groupby("parameters").observed_mse.mean()
                    ax.loglog(
                        ys.index,
                        ys.values,
                        color=colors[architecture],
                        alpha=0.2,
                        lw=0.8,
                    )
                ys = cell.groupby("parameters").observed_mse.mean()
                ax.loglog(
                    ys.index,
                    ys.values,
                    "o-",
                    color=colors[architecture],
                    label=architecture,
                )
            ax.axvline(485, color="#aaaaaa", ls=":", lw=1)
            ax.set(
                title=f"{family}; target rank {q}",
                xlabel="Stored parameters",
                ylabel="IID test MSE",
            )
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Fresh-teacher IID scaling: means and four teacher curves\nNo monotone envelope; P485 forecasts frozen before confirmation"
    )
    fig.savefig(output / "primary_curves.pdf")
    fig.savefig(output / "primary_curves.png", dpi=180)
    plt.close(fig)
    summary = {
        "fits": len(frame),
        "primary_interaction": interaction_summary[
            (interaction_summary.variant == "primary")
            & (interaction_summary.family == "shunt")
            & (interaction_summary.parameters == 485)
        ].to_dict("records"),
        "architecture_predictions": architecture_predictions.to_dict("records"),
        "selected_method_counts": selections.optimizer.value_counts().to_dict(),
        "summed_fit_elapsed_hours": float(frame.elapsed_seconds.sum() / 3600),
        "scope": "Four fresh mixture teachers, two observation replicates. Descriptive uncertainty, no asymptotic exponent or global convergence claim.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "campaign_completion_sha256": hashlib.sha256(
            (root / "campaign_complete.json").read_bytes()
        ).hexdigest(),
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))
