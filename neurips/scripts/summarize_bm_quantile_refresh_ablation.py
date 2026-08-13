#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DRAFT_DIR = REPO_ROOT / "drafts" / "dendritic-local-learning"
ANALYSIS_DIR = DRAFT_DIR / "analysis" / "bm_quantile_refresh_summary_20260415"

SWEEPS = {
    "baseline_mnist": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_training_strategies_mnist_classification_20260415_001428"
    ),
    "baseline_cifar10": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_training_strategies_cifar10_classification_20260415_001428"
    ),
    "ablation_mnist": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_quantile_refresh_ablation_mnist_classification_20260415_101433"
    ),
    "ablation_cifar10": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_quantile_refresh_ablation_cifar10_classification_20260415_101433"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _branch_label(branch_factors: list[int]) -> str:
    return "bf" + "".join(str(v) for v in branch_factors)


def _core_label(core_type: str) -> str:
    return "shunting" if "shunting" in core_type else "additive"


def _init_label(init_policy: str) -> str:
    return "previnit" if init_policy == "analytical" else "quantinit"


def _bm_scheme(strategy: str, reactivation_update_mode: str) -> str:
    if strategy == "standard":
        return "quantile_bm" if reactivation_update_mode == "quantile" else "backprop_bm"
    return "quantile_bm" if reactivation_update_mode == "quantile" else "learned_local_bm"


def _ema_label(alpha: float | None) -> str:
    if alpha is None or abs(alpha - 1.0) < 1e-9:
        return "hard"
    return f"ema{int(round(alpha * 100)):03d}"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _metric_triplet(result_dir: Path) -> tuple[float | None, float | None, float | None]:
    final_path = result_dir / "performance" / "final.json"
    if not final_path.exists():
        return None, None, None
    payload = _load_json(final_path)
    acc = payload.get("accuracy", {})
    return acc.get("train"), acc.get("valid"), acc.get("test")


def _iter_rows(sweep_name: str, sweep_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted((sweep_dir / "results").glob("config_*/config.json")):
        cfg = _load_json(config_path)
        result_dir = config_path.parent
        train_acc, valid_acc, test_acc = _metric_triplet(result_dir)
        status = "completed" if test_acc is not None else "missing"
        common_cfg = cfg["training"]["main"]["common"]
        local_cfg = cfg["training"]["main"].get("learning_strategy_config") or {}
        strategy = cfg["training"]["main"]["strategy"]
        reactivation_update_mode = common_cfg.get("reactivation_update_mode", "backprop")
        bm_scheme = _bm_scheme(strategy=strategy, reactivation_update_mode=reactivation_update_mode)
        row = {
            "sweep_name": sweep_name,
            "sweep_family": "ablation" if sweep_name.startswith("ablation") else "baseline",
            "dataset": cfg["data"]["dataset_name"],
            "strategy": strategy,
            "core": _core_label(cfg["model"]["core"]["type"]),
            "morphology": _branch_label(cfg["model"]["core"]["architecture"]["excitatory_branch_factors"]),
            "init_label": _init_label(cfg["model"]["core"]["reactivation"]["init_policy"]),
            "bm_scheme": bm_scheme,
            "status": status,
            "train_accuracy": train_acc,
            "valid_accuracy": valid_acc,
            "test_accuracy": test_acc,
            "reactivation_update_mode": reactivation_update_mode,
            "recalibrate_every": common_cfg.get("recalibrate_reactivation_every"),
            "recalibration_start_epoch": common_cfg.get("reactivation_recalibration_start_epoch"),
            "recalibration_num_batches": common_cfg.get("reactivation_recalibration_num_batches"),
            "ema_alpha": common_cfg.get("reactivation_recalibration_ema_alpha"),
            "ema_label": _ema_label(common_cfg.get("reactivation_recalibration_ema_alpha")),
            "update_reactivation": local_cfg.get("update_reactivation"),
            "run_name": cfg["outputs"]["run_name"],
            "result_dir": str(result_dir),
            "config_path": str(config_path),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return mean(values) if values else None


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = _completed(rows)
    if not completed:
        return None
    return max(completed, key=lambda row: float(row["test_accuracy"]))


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, bucket in sorted(buckets.items()):
        record = {key: value for key, value in zip(keys, key_values)}
        completed_bucket = _completed(bucket)
        record["count"] = len(bucket)
        record["completed_count"] = len(completed_bucket)
        record["mean_test_accuracy"] = _mean(completed_bucket, "test_accuracy")
        best = _best(bucket)
        record["best_run_name"] = best["run_name"] if best else None
        record["best_test_accuracy"] = best["test_accuracy"] if best else None
        out.append(record)
    return out


def _status_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    grouped = _group(rows, ("dataset", "sweep_family", "sweep_name"))
    return [
        [
            record["dataset"],
            record["sweep_family"],
            record["sweep_name"],
            str(record["completed_count"]),
            str(record["count"] - record["completed_count"]),
        ]
        for record in grouped
    ]


def _quantile_comparison_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for strategy in sorted({row["strategy"] for row in dataset_rows}):
            for core in sorted({row["core"] for row in dataset_rows}):
                slice_rows = [
                    row
                    for row in dataset_rows
                    if row["strategy"] == strategy and row["core"] == core
                ]
                baseline_quant = [
                    row for row in slice_rows
                    if row["sweep_family"] == "baseline" and row["bm_scheme"] == "quantile_bm"
                ]
                ablation_quant = [
                    row for row in slice_rows
                    if row["sweep_family"] == "ablation"
                ]
                learned_ref = [
                    row
                    for row in slice_rows
                    if row["sweep_family"] == "baseline" and row["bm_scheme"] != "quantile_bm"
                ]
                baseline_mean = _mean(_completed(baseline_quant), "test_accuracy")
                ablation_mean = _mean(_completed(ablation_quant), "test_accuracy")
                learned_mean = _mean(_completed(learned_ref), "test_accuracy")
                baseline_best = _best(baseline_quant)
                ablation_best = _best(ablation_quant)
                learned_best = _best(learned_ref)
                table_rows.append(
                    [
                        dataset,
                        strategy,
                        core,
                        _fmt(learned_mean),
                        _fmt(baseline_mean),
                        _fmt(ablation_mean),
                        _fmt(None if ablation_mean is None or baseline_mean is None else ablation_mean - baseline_mean),
                        _fmt(None if ablation_mean is None or learned_mean is None else ablation_mean - learned_mean),
                        _fmt(learned_best["test_accuracy"] if learned_best else None),
                        _fmt(baseline_best["test_accuracy"] if baseline_best else None),
                        _fmt(ablation_best["test_accuracy"] if ablation_best else None),
                    ]
                )
    return table_rows


def _ablation_ema_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    ablation_rows = [row for row in rows if row["sweep_family"] == "ablation"]
    grouped = _group(
        ablation_rows,
        ("dataset", "strategy", "core", "ema_label"),
    )
    return [
        [
            record["dataset"],
            record["strategy"],
            record["core"],
            record["ema_label"],
            str(record["completed_count"]),
            _fmt(record["mean_test_accuracy"]),
            _fmt(record["best_test_accuracy"]),
            record["best_run_name"] or "-",
        ]
        for record in grouped
    ]


def _winner_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for strategy in sorted({row["strategy"] for row in dataset_rows}):
            baseline_rows = [
                row for row in dataset_rows
                if row["strategy"] == strategy and row["sweep_family"] == "baseline"
            ]
            ablation_rows = [
                row for row in dataset_rows
                if row["strategy"] == strategy and row["sweep_family"] == "ablation"
            ]
            baseline_best = _best(baseline_rows)
            ablation_best = _best(ablation_rows)
            table_rows.append(
                [
                    dataset,
                    strategy,
                    baseline_best["run_name"] if baseline_best else "-",
                    _fmt(baseline_best["test_accuracy"] if baseline_best else None),
                    ablation_best["run_name"] if ablation_best else "-",
                    _fmt(ablation_best["test_accuracy"] if ablation_best else None),
                ]
            )
    return table_rows


def _top_findings(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for strategy in sorted({row["strategy"] for row in rows if row["dataset"] == dataset}):
            slice_rows = [row for row in rows if row["dataset"] == dataset and row["strategy"] == strategy]
            baseline_quant = [
                row for row in slice_rows
                if row["sweep_family"] == "baseline" and row["bm_scheme"] == "quantile_bm"
            ]
            ablation_quant = [
                row for row in slice_rows
                if row["sweep_family"] == "ablation"
            ]
            learned_ref = [
                row
                for row in slice_rows
                if row["sweep_family"] == "baseline" and row["bm_scheme"] != "quantile_bm"
            ]
            baseline_best = _best(baseline_quant)
            ablation_best = _best(ablation_quant)
            learned_best = _best(learned_ref)
            if ablation_best is None:
                continue
            delta_vs_baseline = None
            if baseline_best is not None:
                delta_vs_baseline = float(ablation_best["test_accuracy"]) - float(baseline_best["test_accuracy"])
            delta_vs_learned = None
            if learned_best is not None:
                delta_vs_learned = float(ablation_best["test_accuracy"]) - float(learned_best["test_accuracy"])
            findings.append(
                (
                    f"- `{dataset}` / `{strategy}`: best new quantile-refresh run is "
                    f"`{ablation_best['run_name']}` with test `{ablation_best['test_accuracy']:.4f}`; "
                    f"vs old quantile best `{_fmt(baseline_best['test_accuracy'] if baseline_best else None)}` "
                    f"and learned/BP best `{_fmt(learned_best['test_accuracy'] if learned_best else None)}` "
                    f"(delta vs old quantile `{_fmt(delta_vs_baseline)}`, "
                    f"delta vs learned/BP `{_fmt(delta_vs_learned)}`)."
                )
            )
    return findings


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sweep_name, sweep_dir in SWEEPS.items():
        rows.extend(_iter_rows(sweep_name=sweep_name, sweep_dir=sweep_dir))

    detailed_csv = ANALYSIS_DIR / "bm_quantile_refresh_detailed_results.csv"
    _write_csv(detailed_csv, rows)

    grouped_csv = ANALYSIS_DIR / "bm_quantile_refresh_grouped_results.csv"
    _write_csv(
        grouped_csv,
        _group(
            rows,
            ("dataset", "sweep_family", "strategy", "core", "bm_scheme", "init_label", "ema_label"),
        ),
    )

    report_path = ANALYSIS_DIR / "bm_quantile_refresh_summary.md"
    report = [
        "# b,m Quantile Refresh Summary (2026-04-15)",
        "",
        "This report compares the original `b,m` classification sweeps against the new quantile-refresh ablation.",
        "",
        "Refresh schedule comparison:",
        "- Baseline quantile runs: `every=5`, `start_epoch=5`, default calibration batch count, hard overwrite.",
        "- New ablation: `every=1`, `start_epoch=2`, `num_batches=8`, and EMA variants `hard / 0.50 / 0.25`.",
        "",
        "Status:",
        _table(
            ["Dataset", "Sweep Family", "Sweep Name", "Completed", "Missing"],
            _status_rows(rows),
        ),
        "",
        "Top findings:",
        *_top_findings(rows),
        "",
        "Quantile refresh vs prior baselines:",
        _table(
            [
                "Dataset",
                "Strategy",
                "Core",
                "Mean Learned/BP",
                "Mean Old Quantile",
                "Mean New Quantile",
                "New-Old Quantile",
                "New-Learned/BP",
                "Best Learned/BP",
                "Best Old Quantile",
                "Best New Quantile",
            ],
            _quantile_comparison_rows(rows),
        ),
        "",
        "Ablation by EMA variant:",
        _table(
            [
                "Dataset",
                "Strategy",
                "Core",
                "EMA",
                "Completed",
                "Mean Test",
                "Best Test",
                "Best Run",
            ],
            _ablation_ema_rows(rows),
        ),
        "",
        "Overall winners by strategy:",
        _table(
            [
                "Dataset",
                "Strategy",
                "Best Baseline Run",
                "Best Baseline Test",
                "Best New Quantile Run",
                "Best New Quantile Test",
            ],
            _winner_rows(rows),
        ),
        "",
        f"Detailed CSV: `{detailed_csv}`",
        f"Grouped CSV: `{grouped_csv}`",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)
    print(detailed_csv)
    print(grouped_csv)


if __name__ == "__main__":
    main()
