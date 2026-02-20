#!/usr/bin/env python3
"""Summarize phase-wise NeurIPS local-learning sweeps from raw result files.

Notes
-----
This script can optionally parse per-epoch performance logs under
`results/config_*/performance/epochs/epoch*.json`. On large sweeps stored on
networked filesystems, that can be very slow (many small files).

By default we only parse the run-level artifacts:
- `results/config_*/config.json`
- `results/config_*/performance/final.json`
- optional `results/config_*/information_analysis/final`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
DEFAULT_OUTPUT_DIR = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/neurips_phase_program"
)

SWEEP_PREFIXES = {
    "phase1_capacity": "sweep_neurips_phase1_capacity_calibration",
    "phase1_cifar": "sweep_neurips_phase1_cifar_sanity",
    "phase2_signal": "sweep_neurips_phase2_local_competence_signal",
    "phase2_morphology": "sweep_neurips_phase2_local_competence_morphology",
    "phase2_three_factor": "sweep_neurips_phase2_local_competence_three_factor",
    "phase2_hsic": "sweep_neurips_phase2_local_competence_hsic",
    "phase2b_gap_pilot": "sweep_neurips_phase2b_gap_closing_pilot",
    "claimA_shunting": "sweep_neurips_phase3_claimA_shunting_regime_strong",
    "claimB_morphology": "sweep_neurips_phase3_claimB_morphology_scaling",
    "claimC_error": "sweep_neurips_phase3_claimC_error_shaping",
    "info_panel": "sweep_neurips_phase3_information_panel",
}


def _latest_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _dig(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _to_compact_string(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    if value is None:
        return ""
    return str(value)


def _infer_sweep_key(sweep_dir: Path) -> str:
    name = sweep_dir.name
    for key, prefix in SWEEP_PREFIXES.items():
        if name.startswith(prefix + "_") or name == prefix:
            return key
    return name


def _extract_mi_metrics(info_blob: dict[str, Any]) -> dict[str, float]:
    if not info_blob:
        return {}

    basic = info_blob.get("basic_mi", {}) if isinstance(info_blob, dict) else {}
    if not isinstance(basic, dict):
        return {}

    mapping = {
        "I(E,I;C)": "mi_E_I_C",
        "I(E;C)": "mi_E_C",
        "I(I;C)": "mi_I_C",
        "I(Vb;C)": "mi_Vb_C",
        "I(Vout;C)": "mi_Vout_C",
    }

    out: dict[str, float] = {}
    for source_key, out_key in mapping.items():
        value = basic.get(source_key)
        if isinstance(value, (int, float)):
            out[out_key] = float(value)
    return out


def _extract_run_record(sweep_dir: Path, run_dir: Path) -> dict[str, Any] | None:
    config_blob = _safe_load_json(run_dir / "config.json")
    perf_blob = _safe_load_json(run_dir / "performance" / "final.json")
    if config_blob is None or perf_blob is None:
        return None

    info_blob = _safe_load_json(run_dir / "information_analysis" / "final")

    accuracy = perf_blob.get("accuracy", {}) if isinstance(perf_blob, dict) else {}
    cat_ll = (
        perf_blob.get("categorical_loglikelihood", {})
        if isinstance(perf_blob, dict)
        else {}
    )

    nparams = None
    nparams_path = run_dir / "nparams"
    if nparams_path.exists():
        try:
            nparams = float(nparams_path.read_text(encoding="utf-8").strip())
        except Exception:
            nparams = None

    layer_sizes = _dig(
        config_blob,
        ["model", "core", "architecture", "excitatory_layer_sizes"],
        [],
    )
    branch_factors = _dig(
        config_blob,
        ["model", "core", "architecture", "excitatory_branch_factors"],
        [],
    )
    ie_values = _dig(
        config_blob,
        ["model", "core", "connectivity", "ie_synapses_per_branch_per_layer"],
        [],
    )
    ee_values = _dig(
        config_blob,
        ["model", "core", "connectivity", "ee_synapses_per_branch_per_layer"],
        [],
    )

    local_cfg = _dig(
        config_blob,
        ["training", "main", "learning_strategy_config"],
        {},
    )
    morphology_cfg = local_cfg.get("morphology_aware", {}) if isinstance(local_cfg, dict) else {}
    three_factor = local_cfg.get("three_factor", {}) if isinstance(local_cfg, dict) else {}
    four_factor = local_cfg.get("four_factor", {}) if isinstance(local_cfg, dict) else {}
    hsic = local_cfg.get("hsic", {}) if isinstance(local_cfg, dict) else {}

    record: dict[str, Any] = {
        "sweep_dir": str(sweep_dir),
        "sweep_name": _infer_sweep_key(sweep_dir),
        "config_id": run_dir.name,
        "dataset": _dig(config_blob, ["data", "dataset_name"], ""),
        "network_type": _dig(config_blob, ["model", "core", "type"], ""),
        "strategy": _dig(config_blob, ["training", "main", "strategy"], ""),
        "layer_sizes": _to_compact_string(layer_sizes),
        "branch_factors": _to_compact_string(branch_factors),
        "ie_value": ie_values[0] if isinstance(ie_values, list) and ie_values else None,
        "ee_value": ee_values[0] if isinstance(ee_values, list) and ee_values else None,
        "seed": _dig(config_blob, ["experiment", "seed"], None),
        "rule_variant": local_cfg.get("rule_variant") if isinstance(local_cfg, dict) else None,
        "error_broadcast_mode": local_cfg.get("error_broadcast_mode") if isinstance(local_cfg, dict) else None,
        "decoder_update_mode": local_cfg.get("decoder_update_mode") if isinstance(local_cfg, dict) else None,
        "rho_mode": four_factor.get("rho_mode") if isinstance(four_factor, dict) else None,
        "use_path_propagation": morphology_cfg.get("use_path_propagation") if isinstance(morphology_cfg, dict) else None,
        "morphology_modulator_mode": morphology_cfg.get("morphology_modulator_mode") if isinstance(morphology_cfg, dict) else None,
        "use_dendritic_normalization": morphology_cfg.get("use_dendritic_normalization") if isinstance(morphology_cfg, dict) else None,
        "use_branch_type_rules": morphology_cfg.get("use_branch_type_rules") if isinstance(morphology_cfg, dict) else None,
        "use_conductance_scaling": three_factor.get("use_conductance_scaling") if isinstance(three_factor, dict) else None,
        "use_driving_force": three_factor.get("use_driving_force") if isinstance(three_factor, dict) else None,
        "hsic_enabled": hsic.get("enabled") if isinstance(hsic, dict) else None,
        "hsic_weight": hsic.get("weight") if isinstance(hsic, dict) else None,
        "train_accuracy": accuracy.get("train"),
        "valid_accuracy": accuracy.get("valid"),
        "test_accuracy": accuracy.get("test"),
        "train_categorical_loglikelihood": cat_ll.get("train"),
        "valid_categorical_loglikelihood": cat_ll.get("valid"),
        "test_categorical_loglikelihood": cat_ll.get("test"),
        "nparams": nparams,
        "run_dir": str(run_dir),
    }

    record.update(_extract_mi_metrics(info_blob or {}))
    return record


def _extract_epoch_rows(sweep_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    epoch_dir = run_dir / "performance" / "epochs"
    if not epoch_dir.exists():
        return rows

    for epoch_file in sorted(epoch_dir.glob("epoch*.json")):
        epoch_blob = _safe_load_json(epoch_file)
        if not epoch_blob:
            continue

        stem = epoch_file.stem
        epoch = None
        if stem.startswith("epoch"):
            try:
                epoch = int(stem.replace("epoch", ""))
            except Exception:
                epoch = None

        accuracy = epoch_blob.get("accuracy", {}) if isinstance(epoch_blob, dict) else {}
        cat_ll = (
            epoch_blob.get("categorical_loglikelihood", {})
            if isinstance(epoch_blob, dict)
            else {}
        )

        rows.append(
            {
                "sweep_dir": str(sweep_dir),
                "sweep_name": _infer_sweep_key(sweep_dir),
                "config_id": run_dir.name,
                "epoch": epoch,
                "train_accuracy": accuracy.get("train"),
                "valid_accuracy": accuracy.get("valid"),
                "test_accuracy": accuracy.get("test"),
                "train_categorical_loglikelihood": cat_ll.get("train"),
                "valid_categorical_loglikelihood": cat_ll.get("valid"),
                "test_categorical_loglikelihood": cat_ll.get("test"),
            }
        )

    return rows


def _load_sweep_results(
    sweep_dir: Path, *, include_epochs: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    results_dir = sweep_dir / "results"
    if not results_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for run_dir in sorted(results_dir.glob("config_*")):
        if not run_dir.is_dir():
            continue
        row = _extract_run_record(sweep_dir, run_dir)
        if row is None:
            continue
        run_records.append(row)
        if include_epochs:
            epoch_records.extend(_extract_epoch_rows(sweep_dir, run_dir))

    return pd.DataFrame(run_records), pd.DataFrame(epoch_records)


def _group_mean_std(data: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=keys)

    grouped = data.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    sort_col = "valid_accuracy_mean" if "valid_accuracy_mean" in grouped.columns else None
    if sort_col:
        grouped = grouped.sort_values(sort_col, ascending=False)
    return grouped


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_\n"

    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else ""
            )

    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for values in frame.fillna("").astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sweep-dir", action="append", default=[])
    parser.add_argument(
        "--include-epochs",
        action="store_true",
        help=(
            "Also parse per-epoch metrics from performance/epochs/*.json "
            "(slow on network filesystems)."
        ),
    )
    args = parser.parse_args()

    selected_dirs: list[Path] = [Path(path_str) for path_str in args.sweep_dir]
    if not selected_dirs:
        for prefix in SWEEP_PREFIXES.values():
            match = _latest_match(args.sweep_root, f"{prefix}_*")
            if match is not None:
                selected_dirs.append(match)

    if not selected_dirs:
        raise RuntimeError("No phase sweep directories found.")

    all_runs: list[pd.DataFrame] = []
    all_epochs: list[pd.DataFrame] = []

    for sweep_dir in selected_dirs:
        runs_df, epochs_df = _load_sweep_results(
            sweep_dir, include_epochs=bool(args.include_epochs)
        )
        if not runs_df.empty:
            all_runs.append(runs_df)
        if not epochs_df.empty:
            all_epochs.append(epochs_df)

    if not all_runs:
        raise RuntimeError("No run-level records could be parsed from selected sweeps.")

    runs = pd.concat(all_runs, ignore_index=True)
    epochs = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs.to_csv(output_dir / "combined_results.csv", index=False)
    if not epochs.empty:
        epochs.to_csv(output_dir / "learning_curves.csv", index=False)

    metrics = [
        "valid_accuracy",
        "test_accuracy",
        "valid_categorical_loglikelihood",
    ]

    phase1 = runs[runs["sweep_name"].isin(["phase1_capacity", "phase1_cifar"])]
    phase1_best = (
        phase1.sort_values("valid_accuracy", ascending=False)
        .groupby(["dataset", "network_type"], as_index=False)
        .first()[
            [
                "dataset",
                "network_type",
                "layer_sizes",
                "branch_factors",
                "ie_value",
                "valid_accuracy",
                "test_accuracy",
                "nparams",
                "sweep_name",
                "config_id",
                "run_dir",
            ]
        ]
        .sort_values(["dataset", "valid_accuracy"], ascending=[True, False])
    )
    phase1_best.to_csv(output_dir / "phase1_best_standard.csv", index=False)

    phase2 = runs[runs["sweep_name"].str.startswith(("phase2_", "phase2b_"))].copy()
    phase2_ranked = phase2.sort_values("valid_accuracy", ascending=False).head(200)
    phase2_ranked.to_csv(output_dir / "phase2_local_competence_ranked.csv", index=False)

    phase2b = runs[runs["sweep_name"] == "phase2b_gap_pilot"].copy()
    phase2b_grouped = _group_mean_std(
        phase2b,
        [
            "dataset",
            "network_type",
            "rule_variant",
            "error_broadcast_mode",
            "decoder_update_mode",
            "hsic_enabled",
            "hsic_weight",
        ],
        metrics,
    )
    phase2b_grouped.to_csv(output_dir / "phase2b_gap_closing.csv", index=False)

    claimA = runs[runs["sweep_name"] == "claimA_shunting"].copy()
    claimA_grouped = _group_mean_std(
        claimA,
        ["dataset", "network_type", "error_broadcast_mode", "ie_value"],
        metrics,
    )
    claimA_grouped.to_csv(output_dir / "claimA_shunting_regime.csv", index=False)

    claimB = runs[runs["sweep_name"] == "claimB_morphology"].copy()
    claimB_grouped = _group_mean_std(
        claimB,
        [
            "dataset",
            "layer_sizes",
            "branch_factors",
            "use_path_propagation",
            "morphology_modulator_mode",
        ],
        metrics,
    )
    claimB_grouped.to_csv(output_dir / "claimB_morphology_scaling.csv", index=False)

    claimC = runs[runs["sweep_name"] == "claimC_error"].copy()
    claimC_grouped = _group_mean_std(
        claimC,
        [
            "dataset",
            "error_broadcast_mode",
            "decoder_update_mode",
            "use_path_propagation",
        ],
        metrics,
    )
    claimC_grouped.to_csv(output_dir / "claimC_error_shaping.csv", index=False)

    info_panel = runs[runs["sweep_name"] == "info_panel"].copy()
    info_metrics = [
        "valid_accuracy",
        "test_accuracy",
        "mi_E_I_C",
        "mi_E_C",
        "mi_I_C",
        "mi_Vb_C",
        "mi_Vout_C",
    ]
    info_metrics = [column for column in info_metrics if column in info_panel.columns]
    info_grouped = _group_mean_std(
        info_panel,
        ["dataset", "network_type", "error_broadcast_mode", "use_path_propagation"],
        info_metrics,
    )
    info_grouped.to_csv(output_dir / "info_panel_metrics.csv", index=False)

    # Effect sizes for quick diagnostics
    effects: list[dict[str, Any]] = []
    if not claimA_grouped.empty:
        base_cols = [
            "dataset",
            "error_broadcast_mode",
            "ie_value",
            "network_type",
            "test_accuracy_mean",
        ]
        if all(column in claimA_grouped.columns for column in base_cols):
            pivot = claimA_grouped[base_cols].pivot_table(
                index=["dataset", "error_broadcast_mode", "ie_value"],
                columns="network_type",
                values="test_accuracy_mean",
            )
            if {
                "dendritic_shunting",
                "dendritic_additive",
            }.issubset(set(pivot.columns)):
                diff = (
                    pivot["dendritic_shunting"] - pivot["dendritic_additive"]
                ).reset_index(name="shunting_minus_additive")
                for row in diff.itertuples(index=False):
                    effects.append(
                        {
                            "effect": "claimA_shunting_minus_additive",
                            "dataset": row.dataset,
                            "error_broadcast_mode": row.error_broadcast_mode,
                            "ie_value": row.ie_value,
                            "value": row.shunting_minus_additive,
                        }
                    )

    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(output_dir / "effect_sizes.csv", index=False)

    summary_md = output_dir / "phase_summary.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# NeurIPS Phase Program Summary\n\n")
        handle.write(f"Total runs parsed: {len(runs)}\\\n\n")
        handle.write("## Phase 1 best standard baselines\n\n")
        handle.write(_to_md(phase1_best.head(40)))
        handle.write("\n\n")
        handle.write("## Phase 2 top local-competence runs\n\n")
        handle.write(_to_md(phase2_ranked.head(40)))
        handle.write("\n\n")
        handle.write("## Phase 2b: gap closing (HSIC x broadcast)\n\n")
        handle.write(_to_md(phase2b_grouped.head(80)))
        handle.write("\n\n")
        handle.write("## Claim A: shunting regime\n\n")
        handle.write(_to_md(claimA_grouped.head(60)))
        handle.write("\n\n")
        handle.write("## Claim B: morphology scaling\n\n")
        handle.write(_to_md(claimB_grouped.head(60)))
        handle.write("\n\n")
        handle.write("## Claim C: local error shaping\n\n")
        handle.write(_to_md(claimC_grouped.head(60)))
        handle.write("\n\n")
        handle.write("## Information panel\n\n")
        handle.write(_to_md(info_grouped.head(40)))
        handle.write("\n\n")
        handle.write("## Effect sizes\n\n")
        handle.write(_to_md(effects_df.head(80)))
        handle.write("\n")

    print(f"Wrote phase summary to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
