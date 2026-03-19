#!/usr/bin/env python3
"""Aggregate seed-averaged cue-routing results for publication figures."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = DRAFT_DIR / "analysis"

DEFAULT_SWEEP_DIRS = [
    Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_neurips_cue_routing_seed_repeats_20260306232402/results"
    ),
    Path(
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "sweep_runs/sweep_neurips_cue_routing_tuning_20260307000918/results"
    ),
]

RAW_CSV = ANALYSIS_DIR / "cue_routing_runs.csv"
SUMMARY_CSV = ANALYSIS_DIR / "cue_routing_summary.csv"

CONDITION_INFO: dict[str, dict[str, Any]] = {
    "cue_hard_fixed_additive_localca": {
        "strategy": "local_ca",
        "network_type": "dendritic_additive",
        "router_mode": "fixed",
        "variant": "baseline",
        "display_label": "Fixed\nAdditive",
    },
    "cue_hard_fixed_shunting_localca": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "fixed",
        "variant": "baseline",
        "display_label": "Fixed\nShunting",
    },
    "cue_hard_learned_additive_localca": {
        "strategy": "local_ca",
        "network_type": "dendritic_additive",
        "router_mode": "learned",
        "variant": "baseline",
        "display_label": "Learned\nAdditive",
    },
    "cue_hard_learned_shunting_localca": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "baseline",
        "display_label": "Learned\nShunting\nper-soma",
    },
    "cue_hard_learned_shunting_localca_temp03": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "temp03",
        "display_label": "Learned\nShunting\nlow-temp",
    },
    "cue_hard_learned_shunting_localca_freeze20": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "freeze20",
        "display_label": "Learned\nShunting\nfreeze-20",
    },
    "cue_hard_learned_shunting_localca_pathway_vector": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "pathway_vector_hierarchical",
        "display_label": "Learned\nShunting\npathway-vector",
    },
    "cue_hard_learned_additive_standard": {
        "strategy": "standard",
        "network_type": "dendritic_additive",
        "router_mode": "learned",
        "variant": "baseline",
        "display_label": "Learned\nAdditive\nBP",
    },
    "cue_hard_learned_shunting_standard": {
        "strategy": "standard",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "baseline",
        "display_label": "Learned\nShunting\nBP",
    },
    "cue_hard_learned_shunting_localca_e140": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "baseline_e140",
        "display_label": "Learned\nShunting\nper-soma\n140 ep",
    },
    "cue_hard_learned_shunting_localca_lr10_e140": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "baseline_tuned",
        "display_label": "Learned\nShunting\nper-soma\n+ tune",
    },
    "cue_hard_learned_shunting_localca_pathway_vector_e140": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "pathway_vector_e140",
        "display_label": "Learned\nShunting\npathway-vector\n140 ep",
    },
    "cue_hard_learned_shunting_localca_pathway_vector_lr10_e140": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "pathway_vector_tuned",
        "display_label": "Learned\nShunting\npathway-vector\n+ tune",
    },
    "cue_hard_learned_shunting_localca_pathway_vector_lr8_e140": {
        "strategy": "local_ca",
        "network_type": "dendritic_shunting",
        "router_mode": "learned",
        "variant": "pathway_vector_tuned_lr8",
        "display_label": "Learned\nShunting\npathway-vector\nlr 8e-4",
    },
}


def _run_group(run_name: str) -> str:
    return re.sub(r"_s\d+$", "", run_name)


def _router_metrics(cfg: dict[str, Any], run_dir: Path) -> dict[str, float]:
    model_path = run_dir / "final_model.pt"
    if (
        not model_path.exists()
        or cfg.get("model", {}).get("encoder", {}).get("type") != "pathway_router"
        or cfg.get("model", {}).get("encoder", {}).get("params", {}).get("router_mode")
        != "learned"
    ):
        return {}

    temperature = float(
        cfg.get("model", {})
        .get("encoder", {})
        .get("params", {})
        .get("learned_router_temperature", 1.0)
        or 1.0
    )
    state = torch.load(model_path, map_location="cpu")
    logits_key = next((key for key in state if key.endswith("assignment_logits")), None)
    if logits_key is None:
        return {}

    logits = state[logits_key].float()
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    entropy = -(
        probs * torch.clamp(probs, min=1e-12).log()
    ).sum(dim=-1).mean().item()
    return {
        "router_mean_max_assignment": probs.max(dim=-1).values.mean().item(),
        "router_assignment_entropy": entropy,
    }


def _collect_runs(sweep_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        for cfg_path in sorted(sweep_dir.glob("config_*/config.json")):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            run_dir = cfg_path.parent
            perf_path = run_dir / "performance" / "final.json"
            if not perf_path.exists():
                continue

            run_name = str(cfg["outputs"]["run_name"])
            group = _run_group(run_name)
            info = CONDITION_INFO.get(group)
            if info is None:
                continue

            perf = json.loads(perf_path.read_text(encoding="utf-8"))["accuracy"]
            row: dict[str, Any] = {
                "group": group,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "seed": int(cfg["experiment"]["seed"]),
                "strategy": info["strategy"],
                "network_type": info["network_type"],
                "router_mode": info["router_mode"],
                "variant": info["variant"],
                "display_label": info["display_label"],
                "train_accuracy": float(perf["train"]),
                "valid_accuracy": float(perf["valid"]),
                "test_accuracy": float(perf["test"]),
            }
            row.update(_router_metrics(cfg, run_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group, group_df in runs.groupby("group", sort=True):
        tests = group_df["test_accuracy"].tolist()
        mean_test = _mean(tests)
        rep_row = min(
            group_df.to_dict("records"),
            key=lambda row: abs(float(row["test_accuracy"]) - mean_test),
        )
        record = {
            "group": group,
            "strategy": group_df["strategy"].iloc[0],
            "network_type": group_df["network_type"].iloc[0],
            "router_mode": group_df["router_mode"].iloc[0],
            "variant": group_df["variant"].iloc[0],
            "display_label": group_df["display_label"].iloc[0],
            "n_seeds": int(len(group_df)),
            "run_name": rep_row["run_name"],
            "run_dir": rep_row["run_dir"],
            "train_accuracy": _mean(group_df["train_accuracy"].tolist()),
            "train_accuracy_std": _std(group_df["train_accuracy"].tolist()),
            "valid_accuracy": _mean(group_df["valid_accuracy"].tolist()),
            "valid_accuracy_std": _std(group_df["valid_accuracy"].tolist()),
            "test_accuracy": mean_test,
            "test_accuracy_std": _std(tests),
            "test_accuracy_min": float(group_df["test_accuracy"].min()),
            "test_accuracy_max": float(group_df["test_accuracy"].max()),
            "router_mean_max_assignment": float("nan"),
            "router_mean_max_assignment_std": float("nan"),
            "router_assignment_entropy": float("nan"),
            "router_assignment_entropy_std": float("nan"),
        }

        if "router_mean_max_assignment" in group_df.columns:
            assign_df = group_df.dropna(subset=["router_mean_max_assignment"])
            if not assign_df.empty:
                record["router_mean_max_assignment"] = _mean(
                    assign_df["router_mean_max_assignment"].tolist()
                )
                record["router_mean_max_assignment_std"] = _std(
                    assign_df["router_mean_max_assignment"].tolist()
                )
                record["router_assignment_entropy"] = _mean(
                    assign_df["router_assignment_entropy"].tolist()
                )
                record["router_assignment_entropy_std"] = _std(
                    assign_df["router_assignment_entropy"].tolist()
                )

        records.append(record)

    return pd.DataFrame(records).sort_values(
        ["strategy", "router_mode", "network_type", "variant"]
    )


def summarize_runs(
    sweep_dirs: list[Path],
    *,
    raw_csv: Path = RAW_CSV,
    summary_csv: Path = SUMMARY_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_dirs = sweep_dirs or DEFAULT_SWEEP_DIRS
    runs = _collect_runs(sweep_dirs)
    if runs.empty:
        raise RuntimeError("No cue-routing runs were collected from the provided paths.")

    summary = _aggregate_runs(runs)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    return runs, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        action="append",
        type=Path,
        default=[],
        help="Result directory containing config_*/config.json entries.",
    )
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    args = parser.parse_args()

    sweep_dirs = args.sweep_dir or DEFAULT_SWEEP_DIRS
    summarize_runs(sweep_dirs, raw_csv=args.raw_csv, summary_csv=args.summary_csv)
    print(f"Wrote raw run summary to {args.raw_csv}")
    print(f"Wrote aggregated summary to {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
