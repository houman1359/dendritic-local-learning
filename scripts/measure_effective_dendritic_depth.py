#!/usr/bin/env python3
"""Audit learned dendritic couplings and nominal versus effective depth.

This checkpoint-only diagnostic reads coupling parameters directly; it does
not require a data loader or a forward pass.  A coupling stage is called
active when its median transformed conductance exceeds a reporting threshold.
Raw stage statistics and threshold sensitivities are always retained so that
the effective-depth label is not treated as threshold-free.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dendritic_modeling.networks.utils.weight_transforms import (  # noqa: E402
    apply_weight_transform,
)


_COUPLING_RE = re.compile(
    r"^core_network\.layers\.(\d+)\.excitatory_cells\.branch_layers\."
    r"(\d+)\.branches_to_output\.log_weight$"
)


def _nested_get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _checkpoint_path(run_dir: Path, strategy: str) -> Path:
    candidates = (
        [
            run_dir / "main_network" / "local_learning_best_model.pt",
            run_dir / "final_model.pt",
        ]
        if strategy == "local_ca"
        else [
            run_dir / "main_network" / "standard_best_model.pt",
            run_dir / "final_model.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint object at {path}: {type(loaded)}")
    for key in ("state_dict", "model_state_dict"):
        value = loaded.get(key)
        if isinstance(value, dict):
            return value
    return loaded


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().reshape(-1), q).item())


def _depth_manifest(path: Path, sweep_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["sweep_name"].eq(sweep_name)].copy()
    keep = [
        "run_dir",
        "dataset",
        "network_type",
        "strategy",
        "branch_factors",
        "ie_value",
        "seed",
        "test_accuracy",
    ]
    frame = frame[keep].drop_duplicates("run_dir")
    frame["source"] = "depth_scaling"
    return frame


def _morphology_manifest(path: Path, root: Path) -> pd.DataFrame:
    indices = pd.read_csv(path)["config_idx"].drop_duplicates().astype(int)
    rows: list[dict[str, Any]] = []
    for index in indices:
        run_dir = root / f"config_{index}"
        with (run_dir / "config.json").open() as handle:
            config = json.load(handle)
        with (run_dir / "performance" / "final.json").open() as handle:
            performance = json.load(handle)
        rows.append(
            {
                "run_dir": str(run_dir),
                "dataset": "noise_resilience",
                "network_type": _nested_get(config, "model.core.type"),
                "strategy": "local_ca",
                "branch_factors": str(
                    _nested_get(
                        config,
                        "model.core.architecture.excitatory_branch_factors",
                    )
                ),
                "ie_value": _nested_get(
                    config,
                    (
                        "model.core.connectivity."
                        "ie_synapses_per_branch_per_layer"
                    ),
                    [None],
                )[0],
                "seed": _nested_get(config, "experiment.seed", -1),
                "test_accuracy": performance["accuracy"]["test"],
                "source": "morphology_ie",
            }
        )
    return pd.DataFrame(rows)


def _stage_rows(
    manifest: pd.DataFrame,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(manifest.to_dict("records"), start=1):
        run_dir = Path(str(record["run_dir"]))
        print(f"[{index}/{len(manifest)}] {run_dir}")
        with (run_dir / "config.json").open() as handle:
            config = json.load(handle)
        transform = str(
            _nested_get(
                config,
                "model.core.morphology.weight_transform",
                "exp",
            )
        )
        branch_factors = _nested_get(
            config,
            "model.core.architecture.excitatory_branch_factors",
            record.get("branch_factors"),
        )
        state = _state_dict(_checkpoint_path(run_dir, str(record["strategy"])))
        found = 0
        for key, raw in state.items():
            match = _COUPLING_RE.match(key)
            if match is None or not isinstance(raw, torch.Tensor):
                continue
            found += 1
            population = int(match.group(1))
            parent_stage = int(match.group(2))
            weight = apply_weight_transform(raw.detach().float(), transform)
            flat = weight.reshape(-1)
            row: dict[str, Any] = {
                **record,
                "branch_factors": str(branch_factors),
                "weight_transform": transform,
                "checkpoint": str(
                    _checkpoint_path(run_dir, str(record["strategy"]))
                ),
                "population": population,
                "parent_stage": parent_stage,
                "n_couplings": int(flat.numel()),
                "coupling_mean": float(flat.mean().item()),
                "coupling_std": float(flat.std(unbiased=False).item()),
                "coupling_min": float(flat.min().item()),
                "coupling_q05": _quantile(flat, 0.05),
                "coupling_q50": _quantile(flat, 0.50),
                "coupling_q95": _quantile(flat, 0.95),
                "coupling_max": float(flat.max().item()),
                "coupling_rms": float(flat.square().mean().sqrt().item()),
            }
            for threshold in thresholds:
                label = f"{threshold:.0e}".replace("-", "m")
                row[f"fraction_below_{label}"] = float(
                    (flat < threshold).float().mean().item()
                )
            rows.append(row)
        if found == 0:
            raise RuntimeError(f"No dendritic coupling parameters found in {run_dir}")
    return rows


def _run_rows(
    stages: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    run_rows: list[dict[str, Any]] = []
    group_columns = [
        "run_dir",
        "source",
        "dataset",
        "network_type",
        "strategy",
        "branch_factors",
        "ie_value",
        "seed",
        "test_accuracy",
        "population",
    ]
    for keys, group in stages.groupby(group_columns, dropna=False):
        meta = dict(zip(group_columns, keys))
        ordered = group.sort_values("parent_stage")
        active = {
            int(row.parent_stage): bool(row.coupling_q50 >= threshold)
            for row in ordered.itertuples()
        }
        nominal_depth = len(active)
        contiguous = 0
        for stage in sorted(active, reverse=True):
            if not active[stage]:
                break
            contiguous += 1
        run_rows.append(
            {
                **meta,
                "nominal_depth": nominal_depth,
                "active_stage_count": int(sum(active.values())),
                "effective_depth_from_soma": contiguous,
                "all_stages_active": bool(all(active.values())),
                "minimum_stage_median": float(ordered["coupling_q50"].min()),
                "minimum_stage_q05": float(ordered["coupling_q05"].min()),
                "log10_product_stage_medians": float(
                    sum(
                        math.log10(max(float(value), 1e-30))
                        for value in ordered["coupling_q50"]
                    )
                ),
                "activity_threshold": threshold,
            }
        )
    return pd.DataFrame(run_rows)


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in group_columns and column not in {"seed", "population"}
    ]
    return (
        frame.groupby(group_columns, dropna=False)[numeric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .pipe(
            lambda table: table.set_axis(
                [
                    "_".join(str(part) for part in column if str(part))
                    if isinstance(column, tuple)
                    else str(column)
                    for column in table.columns
                ],
                axis=1,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-manifest",
        type=Path,
        default=DRAFT_ROOT / "analysis" / "combined_results.csv",
    )
    parser.add_argument("--depth-sweep-name", default="depth_scaling")
    parser.add_argument(
        "--morphology-manifest",
        type=Path,
        default=DRAFT_ROOT / "figures" / "data" / "morphology_ie_regime_runs.csv",
    )
    parser.add_argument(
        "--morphology-root",
        type=Path,
        default=(
            DRAFT_ROOT
            / "local_sweep_runs"
            / "noise_resilience_morphology_ie_regime_nonnegativeinput_fix_20260409164042"
            / "results"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument("--activity-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DRAFT_ROOT / "figures" / "data" / "effective_dendritic_depth",
    )
    args = parser.parse_args()

    manifest = pd.concat(
        [
            _depth_manifest(args.depth_manifest, args.depth_sweep_name),
            _morphology_manifest(args.morphology_manifest, args.morphology_root),
        ],
        ignore_index=True,
    )
    missing = [
        Path(path)
        for path in manifest["run_dir"].astype(str)
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing[:5]}")

    stages = pd.DataFrame(_stage_rows(manifest, args.thresholds))
    runs = _run_rows(stages, args.activity_threshold)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stages.to_csv(prefix.with_name(prefix.name + "_stages.csv"), index=False)
    runs.to_csv(prefix.with_name(prefix.name + "_runs.csv"), index=False)
    _summary(
        stages,
        [
            "source",
            "dataset",
            "network_type",
            "strategy",
            "branch_factors",
            "ie_value",
            "population",
            "parent_stage",
        ],
    ).to_csv(prefix.with_name(prefix.name + "_stage_summary.csv"), index=False)
    _summary(
        runs,
        [
            "source",
            "dataset",
            "network_type",
            "strategy",
            "branch_factors",
            "ie_value",
            "population",
        ],
    ).to_csv(prefix.with_name(prefix.name + "_run_summary.csv"), index=False)
    print(f"Wrote outputs with prefix {prefix}")


if __name__ == "__main__":
    main()
