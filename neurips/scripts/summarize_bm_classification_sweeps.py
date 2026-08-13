#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DRAFT_DIR = REPO_ROOT / "drafts" / "dendritic-local-learning"
ANALYSIS_DIR = DRAFT_DIR / "analysis" / "bm_classification_summary_20260415"

SWEEP_DIRS = {
    "mnist": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_training_strategies_mnist_classification_20260415_001428"
    ),
    "cifar10": Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_bm_training_strategies_cifar10_classification_20260415_001428"
    ),
}


def _parse_config_paths(array_script: Path) -> list[Path]:
    text = array_script.read_text(encoding="utf-8")
    match = re.search(r"CONFIG_FILES=\((.*?)\)\nCONFIG=", text, re.S)
    if match is None:
        raise RuntimeError(f"Could not parse CONFIG_FILES from {array_script}")
    paths: list[Path] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith('"'):
            paths.append(Path(line.strip('"')))
    return paths


def _branch_label(branch_factors: list[int]) -> str:
    return "bf" + "".join(str(v) for v in branch_factors)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _status_for_result_dir(result_dir: Path) -> str:
    final_path = result_dir / "performance" / "final.json"
    if final_path.exists():
        return "completed"
    if result_dir.exists():
        return "started"
    return "pending"


def _metric_triplet(result_dir: Path) -> tuple[float | None, float | None, float | None]:
    final_path = result_dir / "performance" / "final.json"
    if not final_path.exists():
        return None, None, None
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    acc = payload.get("accuracy", {})
    return acc.get("train"), acc.get("valid"), acc.get("test")


def _bm_scheme(
    strategy: str,
    reactivation_update_mode: str,
    update_reactivation: bool,
) -> str:
    if strategy == "standard":
        return "quantile_bm" if reactivation_update_mode == "quantile" else "backprop_bm"
    return "quantile_bm" if reactivation_update_mode == "quantile" else "learned_local_bm"


def _bp_path_active(strategy: str, encoder_update_mode: str, decoder_update_mode: str) -> bool:
    if strategy == "standard":
        return True
    return encoder_update_mode == "backprop" or decoder_update_mode == "backprop"


def _derive_row(dataset: str, idx: int, config_path: Path, sweep_dir: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    strategy = cfg["training"]["main"]["strategy"]
    core_type = cfg["model"]["core"]["type"]
    core = "shunting" if "shunting" in core_type else "additive"
    branch_factors = cfg["model"]["core"]["architecture"]["excitatory_branch_factors"]
    morphology = _branch_label(branch_factors)
    init_policy = cfg["model"]["core"]["reactivation"]["init_policy"]
    common_cfg = cfg["training"]["main"]["common"]
    local_cfg = cfg["training"]["main"].get("learning_strategy_config", {})
    reactivation_update_mode = common_cfg.get("reactivation_update_mode", "backprop")
    update_reactivation = bool(local_cfg.get("update_reactivation", False))
    encoder_update_mode = str(local_cfg.get("encoder_update_mode", "none"))
    decoder_update_mode = str(local_cfg.get("decoder_update_mode", "backprop"))
    bm_update_scheme = _bm_scheme(
        strategy=strategy,
        reactivation_update_mode=reactivation_update_mode,
        update_reactivation=update_reactivation,
    )
    bp_path_active = _bp_path_active(
        strategy=strategy,
        encoder_update_mode=encoder_update_mode,
        decoder_update_mode=decoder_update_mode,
    )
    result_dir = sweep_dir / "results" / f"config_{idx}"
    status = _status_for_result_dir(result_dir)
    train_acc, valid_acc, test_acc = _metric_triplet(result_dir)
    return {
        "dataset": dataset,
        "config_index": idx,
        "config_name": config_path.stem,
        "training_strategy": strategy,
        "core": core,
        "morphology": morphology,
        "init_policy": init_policy,
        "bm_update_scheme": bm_update_scheme,
        "reactivation_update_mode": reactivation_update_mode,
        "update_reactivation": update_reactivation,
        "encoder_update_mode": encoder_update_mode,
        "decoder_update_mode": decoder_update_mode,
        "bp_path_active": bp_path_active,
        "status": status,
        "train_accuracy": train_acc,
        "valid_accuracy": valid_acc,
        "test_accuracy": test_acc,
        "config_path": str(config_path),
        "result_dir": str(result_dir),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return mean(values) if values else float("nan")


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    grouped: list[dict[str, Any]] = []
    for key_values, bucket in sorted(buckets.items()):
        record = {key: value for key, value in zip(keys, key_values)}
        record["completed_count"] = len(bucket)
        record["mean_train_accuracy"] = round(_mean_metric(bucket, "train_accuracy"), 6)
        record["mean_valid_accuracy"] = round(_mean_metric(bucket, "valid_accuracy"), 6)
        record["mean_test_accuracy"] = round(_mean_metric(bucket, "test_accuracy"), 6)
        best = max(bucket, key=lambda row: float(row["test_accuracy"]))
        record["best_config_name"] = best["config_name"]
        record["best_test_accuracy"] = round(float(best["test_accuracy"]), 6)
        grouped.append(record)
    return grouped


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _display_name(row: dict[str, Any]) -> str:
    name = row["config_name"]
    if row["training_strategy"] == "local_ca" and row["bm_update_scheme"] == "learned_local_bm":
        return name.replace("_bp_", "_learnedbm_")
    if row["training_strategy"] == "standard" and row["bm_update_scheme"] == "backprop_bm":
        return name.replace("_bp_", "_bpbm_")
    return name


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _status_summary(rows: list[dict[str, Any]]) -> list[list[str]]:
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_dataset[row["dataset"]][row["status"]] += 1
    table_rows: list[list[str]] = []
    for dataset in sorted(by_dataset):
        counts = by_dataset[dataset]
        table_rows.append(
            [
                dataset,
                str(counts.get("completed", 0)),
                str(counts.get("started", 0)),
                str(counts.get("pending", 0)),
            ]
        )
    return table_rows


def _rollup_rows(rows: list[dict[str, Any]], dataset: str) -> list[list[str]]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    grouped = _group_rows(
        dataset_rows,
        ("training_strategy", "core", "bm_update_scheme"),
    )
    return [
        [
            record["training_strategy"],
            record["core"],
            record["bm_update_scheme"],
            str(record["completed_count"]),
            f"{record['mean_train_accuracy']:.4f}",
            f"{record['mean_valid_accuracy']:.4f}",
            f"{record['mean_test_accuracy']:.4f}",
            _display_name({"config_name": record["best_config_name"], "training_strategy": record["training_strategy"], "bm_update_scheme": record["bm_update_scheme"]}),
        ]
        for record in grouped
    ]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row["test_accuracy"]))


def _winner_rows(rows: list[dict[str, Any]], dataset: str) -> list[list[str]]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    table_rows: list[list[str]] = []
    for training_strategy in sorted({row["training_strategy"] for row in dataset_rows}):
        strategy_rows = [row for row in dataset_rows if row["training_strategy"] == training_strategy]
        additive_rows = [row for row in strategy_rows if row["core"] == "additive"]
        shunting_rows = [row for row in strategy_rows if row["core"] == "shunting"]
        if not additive_rows or not shunting_rows:
            continue
        best_additive = _best_row(additive_rows)
        best_shunting = _best_row(shunting_rows)
        overall = _best_row([best_additive, best_shunting])
        table_rows.append(
            [
                training_strategy,
                _display_name(best_additive),
                f"{float(best_additive['test_accuracy']):.4f}",
                _display_name(best_shunting),
                f"{float(best_shunting['test_accuracy']):.4f}",
                overall["core"],
                _display_name(overall),
                f"{float(overall['test_accuracy']):.4f}",
            ]
        )
    return table_rows


def _detail_rows(rows: list[dict[str, Any]], dataset: str, limit: int | None = None) -> list[list[str]]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    dataset_rows.sort(
        key=lambda row: (row["status"] != "completed", -(row["test_accuracy"] or -1.0)),
    )
    if limit is not None:
        dataset_rows = dataset_rows[:limit]
    return [
        [
            _display_name(row),
            row["training_strategy"],
            row["core"],
            row["morphology"],
            row["bm_update_scheme"],
            row["init_policy"],
            row["status"],
            _fmt(row["valid_accuracy"]),
            _fmt(row["test_accuracy"]),
        ]
        for row in dataset_rows
    ]


def _local_ca_gap_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    completed_rows = [row for row in rows if row["dataset"] == "mnist" and row["training_strategy"] == "local_ca"]
    grouped = _group_rows(
        completed_rows,
        ("core", "morphology", "bm_update_scheme"),
    )
    grouped_map = {
        (row["core"], row["morphology"], row["bm_update_scheme"]): row for row in grouped
    }
    table_rows: list[list[str]] = []
    combos = sorted({(row["core"], row["morphology"]) for row in completed_rows})
    for core, morphology in combos:
        learned = [
            row
            for row in completed_rows
            if row["core"] == core
            and row["morphology"] == morphology
            and row["bm_update_scheme"] == "learned_local_bm"
        ]
        quantile = [
            row
            for row in completed_rows
            if row["core"] == core
            and row["morphology"] == morphology
            and row["bm_update_scheme"] == "quantile_bm"
        ]
        table_rows.append(
            [
                core,
                morphology,
                f"{_mean_metric(learned, 'train_accuracy'):.4f}",
                f"{_mean_metric(quantile, 'train_accuracy'):.4f}",
                f"{_mean_metric(learned, 'valid_accuracy'):.4f}",
                f"{_mean_metric(quantile, 'valid_accuracy'):.4f}",
                f"{_mean_metric(learned, 'test_accuracy'):.4f}",
                f"{_mean_metric(quantile, 'test_accuracy'):.4f}",
            ]
        )
    return table_rows


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    completed_rows = _completed(rows)
    md: list[str] = []
    md.append("# b,m classification sweep summary")
    md.append("")
    md.append("This report uses corrected naming for the local CA runs.")
    md.append("")
    md.append("- `standard + backprop_bm`: optimizer/BP updates `b,m`.")
    md.append("- `standard + quantile_bm`: optimizer/BP updates the rest; `b,m` are refreshed by the quantile rule.")
    md.append("- `local_ca + learned_local_bm`: local CA updates the network and also updates `b,m` with the local rule.")
    md.append("- `local_ca + quantile_bm`: local CA updates the network, but `b,m` are frozen from local updates and refreshed by the quantile rule.")
    md.append("")
    md.append("Historical local-CA config filenames on disk still contain `_bp_`; in this report those runs are renamed to `learnedbm` in the displayed tables.")
    md.append("")
    md.append("For the current MNIST/CIFAR local CA sweeps, there is no active BP path in the local CA branch:")
    md.append("")
    md.append("- `encoder_update_mode = none`")
    md.append("- `decoder_update_mode = local`")
    md.append("- `update_reactivation = true` only for `learned_local_bm`")
    md.append("")
    md.append("## Status")
    md.append("")
    md.append(_table(["dataset", "completed", "started", "pending"], _status_summary(rows)))
    md.append("")
    md.append("## Morphologies tested")
    md.append("")
    md.append("- `bf22` = `[2, 2]`")
    md.append("- `bf333` = `[3, 3, 3]`")
    md.append("")
    md.append("## MNIST rollup")
    md.append("")
    md.append(
        _table(
            [
                "strategy",
                "core",
                "bm_update",
                "n",
                "mean_train",
                "mean_valid",
                "mean_test",
                "best_config",
            ],
            _rollup_rows(completed_rows, "mnist"),
        )
    )
    md.append("")
    md.append("## MNIST additive vs shunting winners")
    md.append("")
    md.append(
        _table(
            [
                "strategy",
                "best_additive",
                "additive_test",
                "best_shunting",
                "shunting_test",
                "winner_core",
                "winner_config",
                "winner_test",
            ],
            _winner_rows(completed_rows, "mnist"),
        )
    )
    md.append("")
    md.append("## CIFAR10 rollup")
    md.append("")
    md.append(
        _table(
            [
                "strategy",
                "core",
                "bm_update",
                "n",
                "mean_train",
                "mean_valid",
                "mean_test",
                "best_config",
            ],
            _rollup_rows(completed_rows, "cifar10"),
        )
    )
    md.append("")
    md.append("## CIFAR10 additive vs shunting winners")
    md.append("")
    md.append(
        _table(
            [
                "strategy",
                "best_additive",
                "additive_test",
                "best_shunting",
                "shunting_test",
                "winner_core",
                "winner_config",
                "winner_test",
            ],
            _winner_rows(completed_rows, "cifar10"),
        )
    )
    md.append("")
    md.append("## Why `quantile_bm` currently trails `learned_local_bm` under local CA")
    md.append("")
    md.append(
        "Across every completed MNIST local-CA subgroup, the `quantile_bm` runs are lower on "
        "train, validation, and test accuracy than the matched `learned_local_bm` runs. "
        "That pattern points to optimization mismatch or underfitting, not to quantile refresh "
        "acting as a useful regularizer."
    )
    md.append("")
    md.append(
        "Inference: under local CA, the branch weights, decoder, and gate parameters co-adapt "
        "through local updates. Freezing local `b,m` updates and then resetting `b,m` every 5 epochs "
        "with the quantile rule likely disrupts that co-adaptation. This is an inference from the "
        "metric pattern, not yet a direct mechanistic measurement."
    )
    md.append("")
    md.append(
        _table(
            [
                "core",
                "morphology",
                "learned_train",
                "quantile_train",
                "learned_valid",
                "quantile_valid",
                "learned_test",
                "quantile_test",
            ],
            _local_ca_gap_rows(rows),
        )
    )
    md.append("")
    md.append("## Detailed results")
    md.append("")
    md.append("### MNIST")
    md.append("")
    md.append(
        _table(
            [
                "config_name",
                "strategy",
                "core",
                "morphology",
                "bm_update",
                "init",
                "status",
                "valid",
                "test",
            ],
            _detail_rows(rows, "mnist"),
        )
    )
    md.append("")
    md.append("### CIFAR10")
    md.append("")
    md.append(
        _table(
            [
                "config_name",
                "strategy",
                "core",
                "morphology",
                "bm_update",
                "init",
                "status",
                "valid",
                "test",
            ],
            _detail_rows(rows, "cifar10"),
        )
    )
    md.append("")
    md.append("Row-level CSV: `bm_classification_detailed_results.csv`")
    md.append("")
    md.append("Grouped CSV: `bm_classification_grouped_summary.csv`")
    md.append("")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset, sweep_dir in SWEEP_DIRS.items():
        config_paths = _parse_config_paths(sweep_dir / "jobs" / "run_array_sweep.sh")
        for idx, config_path in enumerate(config_paths):
            rows.append(_derive_row(dataset, idx, config_path, sweep_dir))

    detailed_csv = ANALYSIS_DIR / "bm_classification_detailed_results.csv"
    grouped_csv = ANALYSIS_DIR / "bm_classification_grouped_summary.csv"
    summary_md = ANALYSIS_DIR / "bm_classification_summary.md"

    _write_csv(detailed_csv, rows)
    _write_csv(
        grouped_csv,
        _group_rows(
            _completed(rows),
            (
                "dataset",
                "training_strategy",
                "core",
                "morphology",
                "bm_update_scheme",
                "init_policy",
            ),
        ),
    )
    _write_markdown(summary_md, rows)

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {grouped_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
