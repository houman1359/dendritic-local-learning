"""Collect every planned run and fit guarded fixed-recipe parameter curves.

Validation is the default. This tool never selects architectures or interprets
descriptive fitted exponents as evidence for different scaling laws. CSV files
contain all seed observations underlying the optional plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HELD_OUT_PHASES = {"held_out", "heldout", "held-out"}


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _positive(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def collect_campaign(
    campaign: str | Path, metric_split: str = "validation"
) -> list[dict]:
    """Return one row per manifest task, including absent or invalid receipts."""
    if metric_split not in {"validation", "test"}:
        raise ValueError("metric_split must be validation or test")
    directory = Path(campaign).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    tasks = manifest["tasks"]
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Manifest contains duplicate task IDs")
    rows = []
    for task in tasks:
        row = {
            "campaign": str(directory),
            "id": task["id"],
            "status": "missing",
            "reason": "receipt.json is absent",
            "metric_split": metric_split,
        }
        rows.append(row)
        path = directory / task["config"]
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != task["config_file_sha256"]:
                raise ValueError("Configuration differs from manifest hash")
            config = json.loads(raw)
            row.update(
                architecture_id=config["architecture_id"],
                seed=int(config["seed"]),
                phase=config["phase"],
                target_parameters=config["target_parameters"],
                recipe_identity=_identity(
                    {
                        key: value
                        for key, value in config["model"].items()
                        if key not in {"width", "seed", "topology_seed"}
                    }
                ),
            )
            receipt_path = directory / task["output"] / "receipt.json"
            if not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text())
            if "status" not in receipt:
                raise ValueError("Receipt has no status; completion must be explicit")
            row.update(
                status=receipt["status"],
                reason=receipt.get("error", receipt.get("failure", "")),
            )
            if receipt["status"] != "completed":
                continue
            if receipt.get("config") != config:
                raise ValueError("Receipt config differs from frozen campaign config")
            if receipt.get("config_sha256") != _identity(config):
                raise ValueError("Receipt config_sha256 does not verify")
            for key in ("id", "phase", "seed"):
                if receipt.get(key) != config[key]:
                    raise ValueError(f"Receipt {key} differs from frozen config")
            parameters = int(
                _positive(receipt["model_report"]["total_parameters"], "P")
            )
            if parameters != int(task["total_parameters"]):
                raise ValueError(
                    "Receipt parameter count differs from instantiated manifest"
                )
            data = receipt["data"]
            unique = data.get("unique_train_examples", data.get("train_size"))
            unique = int(_positive(unique, "Unique training examples"))
            if unique != int(config["data"]["train_size"]):
                raise ValueError(
                    "Receipt unique-data count differs from planned train_size"
                )
            training = receipt["training"]
            steps = int(_positive(training["steps"], "Training steps"))
            processed = int(
                _positive(training["processed_examples"], "Processed examples")
            )
            if steps != int(config["training"].get("steps", 100)):
                raise ValueError(
                    "Completed receipt did not reach the planned training horizon"
                )
            if processed != steps * int(config["training"].get("batch_size", 64)):
                raise ValueError(
                    "Processed examples do not match the fixed-horizon batch protocol"
                )
            elapsed = _positive(training["elapsed_seconds"], "Elapsed seconds")
            metric = receipt["metrics"].get(f"{metric_split}_loss")
            if metric is None:
                raise ValueError(f"Receipt does not contain {metric_split}_loss")
            loss = float(metric)
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Loss must be finite and nonnegative")
            # U varies in the experiment; the target function/data source must not.
            identity = data["identity"]
            unhashed_identity = {
                key: value for key, value in identity.items() if key != "sha256"
            }
            if data["identity_sha256"] != _identity(unhashed_identity):
                raise ValueError("Receipt data identity does not verify")
            identity_data = {
                key: value
                for key, value in identity.items()
                if key
                not in {"sha256", "declared_sizes", "splits", "test_materialized"}
            }
            identity_data["evaluation_split"] = identity["splits"][metric_split]
            train_identity = _identity(identity["splits"]["train"])
            protocol = {
                key: value
                for key, value in config["training"].items()
                if key not in {"device", "num_workers", "num_threads", "log_every"}
            }
            gpu_count = training.get("gpu_count")
            if gpu_count is not None and (
                float(gpu_count) < 0 or not math.isfinite(float(gpu_count))
            ):
                raise ValueError("gpu_count must be finite and nonnegative")
            row.update(
                status="completed",
                reason="",
                parameters=parameters,
                unique_examples=unique,
                task_identity=_identity(identity_data),
                train_identity=train_identity,
                protocol_identity=_identity(protocol),
                steps=steps,
                processed_examples=processed,
                elapsed_seconds=elapsed,
                examples_per_second=processed / elapsed,
                gpu_count=gpu_count,
                recorded_run_gpu_hours=(
                    elapsed * float(gpu_count) / 3600 if gpu_count is not None else None
                ),
                loss=loss,
                accuracy=receipt["metrics"].get(f"{metric_split}_accuracy"),
                config_sha256=receipt["config_sha256"],
            )
        except (KeyError, ValueError, TypeError, OSError) as error:
            row.update(status="invalid", reason=str(error))
    return rows


GROUP_FIELDS = (
    "architecture_id",
    "recipe_identity",
    "unique_examples",
    "task_identity",
    "train_identity",
    "protocol_identity",
    "steps",
    "processed_examples",
    "metric_split",
)


def aggregate_runs(rows: list[dict]) -> list[dict]:
    """Keep phase and actual count separate; SD exists only for >=2 runs."""
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            groups[
                tuple(row[key] for key in (*GROUP_FIELDS, "phase", "parameters"))
            ].append(row)
    aggregated = []
    for key, records in sorted(groups.items()):
        seeds = [record["seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                "Duplicate training seeds within a curve cell; cannot treat reruns as replicates"
            )
        losses = [record["loss"] for record in records]
        item = dict(zip((*GROUP_FIELDS, "phase", "parameters"), key))
        item.update(
            n_seeds=len(records),
            seeds=json.dumps(sorted(seeds)),
            loss_mean=statistics.mean(losses),
            loss_sd=statistics.stdev(losses) if len(losses) > 1 else None,
            elapsed_seconds_total=sum(record["elapsed_seconds"] for record in records),
        )
        aggregated.append(item)
    return aggregated


def fit_parameter_curve(rows: list[dict]) -> dict:
    """Fit L=floor+A(P/P_ref)^(-alpha); held-out rows never enter optimization.

    At least four actual parameter sizes are required. This is a descriptive
    per-U curve; no joint data exponent or confidence interval is inferred.
    """
    import numpy as np
    from scipy.optimize import least_squares

    if not rows or any(row.get("status") != "completed" for row in rows):
        raise ValueError("Fitting requires explicitly completed runs only")
    for key in GROUP_FIELDS:
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Refusing a mixed curve: {key} differs")
    fitted_rows = [row for row in rows if row["phase"] not in HELD_OUT_PHASES]
    if len({row["phase"] for row in fitted_rows}) > 1:
        raise ValueError("Refusing to mix discovery and confirmation phases in one fit")
    if len({row["parameters"] for row in fitted_rows}) < 4:
        raise ValueError("At least four distinct fitting parameter sizes are required")
    aggregate = aggregate_runs(fitted_rows)
    if any(item["n_seeds"] < 2 for item in aggregate):
        raise ValueError(
            "At least two independent training seeds per fitting size are required"
        )
    parameters = np.array([item["parameters"] for item in aggregate], dtype=float)
    losses = np.array([item["loss_mean"] for item in aggregate], dtype=float)
    if np.any(parameters <= 0) or np.any(losses <= 0) or not np.isfinite(losses).all():
        raise ValueError("Power-law fitting requires finite positive P and loss")
    reference = float(parameters.min())
    x = parameters / reference
    scale = max(float(losses.max()), 1e-6)

    def predict(z, query):
        floor, amplitude, exponent = np.exp(z)
        return floor + amplitude * query ** (-exponent)

    candidates = []
    for fraction in (0.05, 0.5, 0.9):
        floor = max(float(losses.min()) * fraction, 1e-12)
        initial = np.log([floor, max(float(losses.max()) - floor, 1e-8), 0.5])
        result = least_squares(
            lambda z: (predict(z, x) - losses) / scale,
            initial,
            bounds=(
                [-30, -30, math.log(0.01)],
                [math.log(scale * 100), math.log(scale * 100), math.log(10)],
            ),
            max_nfev=10000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if result.success:
            candidates.append(result)
    if not candidates:
        raise ValueError("Power-law optimizer did not converge")
    optimum = min(candidates, key=lambda result: float(np.sum(result.fun**2)))
    floor, amplitude, exponent = map(float, np.exp(optimum.x))
    predicted = predict(optimum.x, x)
    held_out = []
    for row in rows:
        if row["phase"] in HELD_OUT_PHASES:
            estimate = float(predict(optimum.x, row["parameters"] / reference))
            held_out.append(
                {
                    "id": row["id"],
                    "seed": row["seed"],
                    "parameters": row["parameters"],
                    "observed_loss": row["loss"],
                    "predicted_loss": estimate,
                    "residual": row["loss"] - estimate,
                }
            )
    return {
        "model": "floor + amplitude * (P / reference_parameters) ** (-alpha)",
        "floor": floor,
        "amplitude": amplitude,
        "alpha": exponent,
        "reference_parameters": reference,
        "fitting_sizes": sorted(set(parameters.tolist())),
        "fitting_run_ids": [row["id"] for row in fitted_rows],
        "rmse_on_size_means": float(np.sqrt(np.mean((predicted - losses) ** 2))),
        "held_out_predictions": held_out,
        "interpretation": "Descriptive positive-floor fit; no exponent-difference claim or confidence interval.",
        "cautions": [
            "Four sizes give weak floor/exponent identification; assess more sizes and floor sensitivity.",
            "Held-out observations are excluded from fitting. No architecture is automatically selected.",
        ],
    }


def _write_csv(path: Path, records: list[dict]) -> None:
    columns = sorted({key for record in records for key in record})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status"])
        writer.writeheader()
        writer.writerows(records)


def analyze_campaigns(
    campaigns: list[str | Path],
    output_dir: str | Path,
    *,
    metric_split: str = "validation",
    plots: bool = True,
) -> dict:
    rows = [
        row
        for campaign in campaigns
        for row in collect_campaign(campaign, metric_split)
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "runs.csv", rows)
    aggregated = aggregate_runs(rows)
    _write_csv(destination / "aggregate.csv", aggregated)
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            groups[tuple(row[key] for key in GROUP_FIELDS)].append(row)
    fits = []
    for key, records in sorted(groups.items()):
        entry = dict(zip(GROUP_FIELDS, key))
        try:
            entry.update(status="fitted", fit=fit_parameter_curve(records))
        except ValueError as error:
            entry.update(status="not_fitted", reason=str(error))
        fits.append(entry)
    complete = [row for row in rows if row["status"] == "completed"]
    timing_known = [
        row for row in complete if row["recorded_run_gpu_hours"] is not None
    ]
    report = {
        "schema_version": "dendritic_parameter_scaling_analysis_v1",
        "metric_split": metric_split,
        "selection": "No architecture selection is performed; all candidates remain visible.",
        "campaigns": [str(Path(path).resolve()) for path in campaigns],
        "grouping": "Fixed recipe, unique data, task and evaluation identity, training-set identity, optimizer and horizon; incompatible groups are never pooled.",
        "task_status_counts": dict(Counter(row["status"] for row in rows)),
        "incomplete_tasks": [
            {key: row.get(key) for key in ("campaign", "id", "status", "reason")}
            for row in rows
            if row["status"] != "completed"
        ],
        "measured_cost": {
            "completed_elapsed_seconds": sum(
                row["elapsed_seconds"] for row in complete
            ),
            "recorded_run_gpu_hours": sum(
                row["recorded_run_gpu_hours"] for row in timing_known
            ),
            "completed_runs_without_gpu_count": len(complete) - len(timing_known),
            "note": "Completed receipt elapsed time times device count only; this is not allocated Slurm GPU-hours. Imports, source verification, job setup, and failed/missing runs may incur additional cost. Use scheduler allocation records for total cluster expenditure.",
        },
        "curves": fits,
        "uncertainty": "Seed SD only for n>=2; no small-sample confidence intervals or automatic exponent claims.",
    }
    if plots and complete:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Separate task and training regimes; never draw a mixed-domain curve.
        panels = defaultdict(list)
        for key, records in groups.items():
            panels[key[2:]].append((key[0], records))
        for index, entries in enumerate(panels.values()):
            fig, axis = plt.subplots(figsize=(7, 4.5))
            for architecture, records in entries:
                points = aggregate_runs(records)
                for phase in sorted({point["phase"] for point in points}):
                    part = sorted(
                        [point for point in points if point["phase"] == phase],
                        key=lambda point: point["parameters"],
                    )
                    (line,) = axis.plot(
                        [point["parameters"] for point in part],
                        [point["loss_mean"] for point in part],
                        marker="o",
                        linestyle="--" if phase in HELD_OUT_PHASES else "-",
                        label=f"{architecture} ({phase})",
                    )
                    individual = [row for row in records if row["phase"] == phase]
                    axis.scatter(
                        [row["parameters"] for row in individual],
                        [row["loss"] for row in individual],
                        color=line.get_color(),
                        alpha=0.45,
                        s=14,
                    )
                    with_sd = [point for point in part if point["loss_sd"] is not None]
                    axis.errorbar(
                        [point["parameters"] for point in with_sd],
                        [point["loss_mean"] for point in with_sd],
                        yerr=[point["loss_sd"] for point in with_sd],
                        fmt="none",
                        color=line.get_color(),
                    )
            sample = entries[0][1][0]
            axis.set(
                xscale="log",
                xlabel="Actual whole-model learned parameters",
                ylabel=f"{metric_split} loss",
                title=f"U={sample['unique_examples']}; steps={sample['steps']} (points: seeds; bars: SD)",
            )
            axis.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(destination / f"curves_{index:03d}.png", dpi=150)
            plt.close(fig)
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        action="append",
        required=True,
        help="Repeat to combine compatible fitting and held-out campaigns",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_campaigns(
        args.campaign,
        args.output_dir,
        metric_split=args.metric_split,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "task_status_counts": report["task_status_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
