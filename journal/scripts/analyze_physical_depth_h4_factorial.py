#!/usr/bin/env python3
"""Audit and analyze the frozen H=4 physical-depth factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_remaining_physical_experiments import (  # noqa: E402
    bootstrap_mean,
    exact_sign_flip_p,
)
from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "physical_depth_h4_runs"
H23_REFERENCE = (
    ROOT
    / "source_data"
    / "remaining_physical_experiments"
    / "seed_outcomes_with_h3_reference.csv"
)
CLEAN_H23_REFERENCE = (
    ROOT
    / "source_data"
    / "physical_depth_clean_source_replication"
    / "seed_outcomes.csv"
)
INITIALIZATION_AUDIT = (
    ROOT / "analysis" / "PHYSICAL_DEPTH_H4_D1_INITIALIZATION_AUDIT_20260819.json"
)
OUTPUT = ROOT / "source_data" / "physical_depth_h4_factorial"
FIGURES = ROOT / "figures" / "generated"
EXPECTED_SOURCE_COMMIT = "a99c3a777f99913e13dfe673a3f3a28bfe3566af"
REPAIR_SOURCE_COMMIT = "e90fb9896daa95df4aad4c0d92acc0cd65bcd750"
REPAIR_TIER_GROUPS = [[0], [1], [2, 3]]

SEEDS = list(range(10400, 10410))
DEPTH_FACTORS = {
    1: "8",
    2: "2x3",
    3: "2x1x2",
    4: "1x1x2x2",
}

# stem: (regime, architecture, mechanism, default credit)
RUN_SPECS = {
    "journal_h4_aligned_serial_shunting_bp": (
        "aligned",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_h4_rewired_tree_serial_shunting_bp": (
        "rewired_tree",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_h4_aligned_grouped_point_shunting_bp": (
        "aligned",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_h4_rewired_tree_grouped_point_shunting_bp": (
        "rewired_tree",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_h4_aligned_serial_shunting_local3f": (
        "aligned",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
    "journal_h4_rewired_tree_serial_shunting_local3f": (
        "rewired_tree",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
    "journal_h4_aligned_serial_additive_bp": (
        "aligned",
        "serial_tree",
        "raw_additive",
        "full_bp",
    ),
}

RESOURCE_COLUMNS = [
    "total_parameters",
    "trainable_parameters",
    "active_synapses",
    "candidate_synapse_slots",
    "persistent_state_scalars",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_run(runs: Path, stem: str) -> Path:
    matches = sorted(runs.glob(f"{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {stem} below {runs}")
    return matches[-1]


def _metric(final: dict[str, Any], name: str, split: str) -> float:
    return float(final[name][split])


def _pathway_spec(config: dict[str, Any], pathway: str) -> dict[str, Any]:
    return config["model"]["core"]["population_network"]["layers"][0][
        "population_defaults"
    ]["structured_connectivity"]["pathways"][pathway]


def collect(
    runs: Path, *, d3_repair_runs: Path | None, allow_incomplete: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    contract_failures: list[str] = []
    source_failures: list[str] = []
    replaced_failed_configs: list[str] = []

    for stem, (regime, architecture, mechanism, credit_default) in RUN_SPECS.items():
        run = latest_run(runs, stem)
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        expected = int(original["sweep_contract"]["expected_config_count"])
        configs = sorted(
            (run / "configs").glob("unified_config_*.yaml"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        manifest_path = run / "frozen_sweep_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source = manifest["source_identity"]["git"]
        if source["commit"] != EXPECTED_SOURCE_COMMIT:
            source_failures.append(f"{stem}/commit/{source['commit']}")
        if source["tracked_worktree_dirty"]:
            source_failures.append(f"{stem}/dirty")
        run_records.append(
            {
                "stem": stem,
                "run_dir": str(run),
                "expected": expected,
                "generated": len(configs),
                "manifest_sha256": sha256(manifest_path),
                "source_commit": source["commit"],
                "source_dirty": bool(source["tracked_worktree_dirty"]),
            }
        )
        if len(configs) != expected:
            missing.append(f"{stem}: generated {len(configs)}/{expected}")

        for config_path in configs:
            index = int(config_path.stem.rsplit("_", 1)[1])
            config = yaml.safe_load(config_path.read_text())
            population = config["model"]["core"]["population_network"]["layers"][0][
                "populations"
            ][0]
            factors = list(population["branch_factors"])
            depth = len(factors)
            result = run / "results" / f"config_{index}"
            final_path = result / "performance" / "final.json"
            resources_path = result / "model_resources.json"
            if not final_path.is_file() or not resources_path.is_file():
                if depth == 3 and d3_repair_runs is not None:
                    replaced_failed_configs.append(f"{stem}/config_{index}")
                    continue
                missing.append(f"{stem}/config_{index}")
                continue

            final = json.loads(final_path.read_text())
            resources = json.loads(resources_path.read_text())
            learning = config["training"]["main"]
            if credit_default == "local_auto":
                broadcast = str(
                    learning["learning_strategy_config"]["error_broadcast_mode"]
                ).lower()
                credit = (
                    "local_path" if broadcast == "path_transport" else "local_shared"
                )
            else:
                credit = credit_default

            expected_ranges = (
                [[0, 16], [16, 32], [32, 48], [48, 64]]
                if regime == "aligned"
                else [[48, 64], [32, 48], [16, 32], [0, 16]]
            )
            data_params = config["data"]["dataset_params"]["hierarchical_gain_load"]
            for pathway in ("ee", "ie"):
                spec = _pathway_spec(config, pathway)
                if spec["inventory_counts"] != [4, 2, 1, 1]:
                    contract_failures.append(f"{stem}/{index}/{pathway}/inventory")
                if spec["feature_ranges"] != expected_ranges:
                    contract_failures.append(f"{stem}/{index}/{pathway}/ranges")
            if int(data_params["n_levels"]) != 4:
                contract_failures.append(f"{stem}/{index}/n_levels")
            if DEPTH_FACTORS.get(depth) != "x".join(map(str, factors)):
                contract_failures.append(f"{stem}/{index}/morphology")

            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result / "train.log", result / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            row: dict[str, Any] = {
                "hierarchy": 4,
                "cohort": stem,
                "run_dir": str(run),
                "config_index": index,
                "seed": int(config["experiment"]["seed"]),
                "regime": regime,
                "architecture": architecture,
                "mechanism": mechanism,
                "credit": credit,
                "depth": depth,
                "branch_factors": "x".join(map(str, factors)),
                "result_origin": "original_factorial",
                "repair_config_index": np.nan,
                "total_parameters": int(resources["total_parameters"]),
                "trainable_parameters": int(resources["trainable_parameters"]),
                "active_synapses": int(resources.get("active_synapses", 0)),
                "candidate_synapse_slots": int(
                    resources.get("candidate_synapse_slots", 0)
                ),
                "persistent_state_scalars": int(
                    resources.get("persistent_state_scalars_per_sample", 0)
                ),
                "config_sha256": sha256(config_path),
                "final_sha256": sha256(final_path),
                "fallback_mentions": int(log_text.count("fallback")),
                "nonfinite_alert": bool(
                    "nan detected" in log_text
                    or "non-finite" in log_text
                    or "nonfinite" in log_text
                ),
            }
            for name in ("accuracy", "auc", "categorical_loglikelihood"):
                for split in ("train", "valid", "test"):
                    row[f"{split}_{name}"] = _metric(final, name, split)
            rows.append(row)

    if d3_repair_runs is not None:
        for stem, (
            regime,
            architecture,
            mechanism,
            credit_default,
        ) in RUN_SPECS.items():
            repair_stem = stem.replace("journal_h4_", "journal_h4_d3repair_", 1)
            run = latest_run(d3_repair_runs, repair_stem)
            original = yaml.safe_load((run / "original_config.yaml").read_text())
            expected = int(original["sweep_contract"]["expected_config_count"])
            configs = sorted(
                (run / "configs").glob("unified_config_*.yaml"),
                key=lambda path: int(path.stem.rsplit("_", 1)[1]),
            )
            manifest_path = run / "frozen_sweep_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            source = manifest["source_identity"]["git"]
            if source["commit"] != REPAIR_SOURCE_COMMIT:
                source_failures.append(f"{repair_stem}/commit/{source['commit']}")
            if source["tracked_worktree_dirty"]:
                source_failures.append(f"{repair_stem}/dirty")
            run_records.append(
                {
                    "stem": repair_stem,
                    "role": "D3 construction repair",
                    "run_dir": str(run),
                    "expected": expected,
                    "generated": len(configs),
                    "manifest_sha256": sha256(manifest_path),
                    "source_commit": source["commit"],
                    "source_dirty": bool(source["tracked_worktree_dirty"]),
                }
            )
            if len(configs) != expected:
                missing.append(f"{repair_stem}: generated {len(configs)}/{expected}")

            original_index_offset = 40 if expected == 20 else 20
            for config_path in configs:
                repair_index = int(config_path.stem.rsplit("_", 1)[1])
                original_index = original_index_offset + repair_index
                result = run / "results" / f"config_{repair_index}"
                final_path = result / "performance" / "final.json"
                resources_path = result / "model_resources.json"
                if not final_path.is_file() or not resources_path.is_file():
                    missing.append(f"{repair_stem}/config_{repair_index}")
                    continue

                config = yaml.safe_load(config_path.read_text())
                final = json.loads(final_path.read_text())
                resources = json.loads(resources_path.read_text())
                population = config["model"]["core"]["population_network"]["layers"][0][
                    "populations"
                ][0]
                factors = list(population["branch_factors"])
                depth = len(factors)
                learning = config["training"]["main"]
                if credit_default == "local_auto":
                    broadcast = str(
                        learning["learning_strategy_config"]["error_broadcast_mode"]
                    ).lower()
                    credit = (
                        "local_path"
                        if broadcast == "path_transport"
                        else "local_shared"
                    )
                else:
                    credit = credit_default

                expected_ranges = (
                    [[0, 16], [16, 32], [32, 48], [48, 64]]
                    if regime == "aligned"
                    else [[48, 64], [32, 48], [16, 32], [0, 16]]
                )
                data_params = config["data"]["dataset_params"]["hierarchical_gain_load"]
                for pathway in ("ee", "ie"):
                    spec = _pathway_spec(config, pathway)
                    if spec["inventory_counts"] != [4, 2, 1, 1]:
                        contract_failures.append(
                            f"{repair_stem}/{repair_index}/{pathway}/inventory"
                        )
                    if spec["feature_ranges"] != expected_ranges:
                        contract_failures.append(
                            f"{repair_stem}/{repair_index}/{pathway}/ranges"
                        )
                    if spec.get("tier_groups") != REPAIR_TIER_GROUPS:
                        contract_failures.append(
                            f"{repair_stem}/{repair_index}/{pathway}/tier_groups"
                        )
                if int(data_params["n_levels"]) != 4:
                    contract_failures.append(f"{repair_stem}/{repair_index}/n_levels")
                if depth != 3 or factors != [2, 1, 2]:
                    contract_failures.append(f"{repair_stem}/{repair_index}/morphology")

                log_text = "\n".join(
                    path.read_text(errors="replace")
                    for path in (
                        result / "train.log",
                        result / "dendritic_modeling.log",
                    )
                    if path.is_file()
                ).lower()
                row: dict[str, Any] = {
                    "hierarchy": 4,
                    "cohort": stem,
                    "run_dir": str(run),
                    "config_index": original_index,
                    "seed": int(config["experiment"]["seed"]),
                    "regime": regime,
                    "architecture": architecture,
                    "mechanism": mechanism,
                    "credit": credit,
                    "depth": depth,
                    "branch_factors": "x".join(map(str, factors)),
                    "result_origin": "D3_construction_repair",
                    "repair_config_index": repair_index,
                    "total_parameters": int(resources["total_parameters"]),
                    "trainable_parameters": int(resources["trainable_parameters"]),
                    "active_synapses": int(resources.get("active_synapses", 0)),
                    "candidate_synapse_slots": int(
                        resources.get("candidate_synapse_slots", 0)
                    ),
                    "persistent_state_scalars": int(
                        resources.get("persistent_state_scalars_per_sample", 0)
                    ),
                    "config_sha256": sha256(config_path),
                    "final_sha256": sha256(final_path),
                    "fallback_mentions": int(log_text.count("fallback")),
                    "nonfinite_alert": bool(
                        "nan detected" in log_text
                        or "non-finite" in log_text
                        or "nonfinite" in log_text
                    ),
                }
                for name in ("accuracy", "auc", "categorical_loglikelihood"):
                    for split in ("train", "valid", "test"):
                        row[f"{split}_{name}"] = _metric(final, name, split)
                rows.append(row)

    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Experiment matrix incomplete ({len(missing)} missing): "
            + ", ".join(missing[:20])
        )
    frame = pd.DataFrame(rows)
    collection = {
        "status": "incomplete" if missing else "complete",
        "expected_row_count": 360,
        "observed_rows": len(frame),
        "missing_count": len(missing),
        "missing_examples": missing[:40],
        "contract_failures": contract_failures[:80],
        "source_failures": source_failures,
        "replaced_failed_config_count": len(replaced_failed_configs),
        "replaced_failed_configs": replaced_failed_configs,
        "runs": run_records,
    }
    return frame, collection


def select(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    part = frame.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if part.seed.duplicated().any():
        raise ValueError(f"Duplicate seed rows for {filters}")
    return part.set_index("seed").test_accuracy.sort_index()


def difference(
    frame: pd.DataFrame,
    left: dict[str, Any],
    right: dict[str, Any],
) -> pd.Series:
    first, second = select(frame, **left).align(select(frame, **right), join="inner")
    if len(first) != 10 or not first.index.equals(second.index):
        raise ValueError(f"Expected ten paired seeds for {left} minus {right}")
    return first - second


def _contrast_row(
    *, name: str, family: str, values: pd.Series, seed: int, detail: str
) -> dict[str, Any]:
    mean, low, high = bootstrap_mean(values.to_numpy(float), seed)
    return {
        "contrast": name,
        "family": family,
        "detail": detail,
        "n_seeds": len(values),
        "mean_accuracy": mean,
        "mean_pp": 100.0 * mean,
        "ci_low_accuracy": low,
        "ci_high_accuracy": high,
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "positive_pairs": int((values > 0).sum()),
        "negative_pairs": int((values < 0).sum()),
        "zero_pairs": int((values == 0).sum()),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(values.to_numpy(float)),
        "positive_claim_gate": bool(low > 0 and int((values > 0).sum()) >= 8),
        "seed_values_pp": ";".join(f"{100 * x:.8f}" for x in values.to_numpy(float)),
    }


def _bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def build_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seed_values: dict[str, pd.Series] = {}

    def add(name: str, family: str, values: pd.Series, detail: str) -> None:
        seed_values[name] = values
        rows.append(
            _contrast_row(
                name=name,
                family=family,
                values=values,
                seed=14000 + len(rows),
                detail=detail,
            )
        )

    serial_bp = {
        "architecture": "serial_tree",
        "mechanism": "shunting",
        "credit": "full_bp",
    }
    grouped_bp = {
        "architecture": "grouped_point",
        "mechanism": "shunting",
        "credit": "full_bp",
    }
    additive_bp = {
        "architecture": "serial_tree",
        "mechanism": "raw_additive",
        "credit": "full_bp",
    }

    for label, base in (
        ("serial_bp", serial_bp),
        ("grouped_bp", grouped_bp),
        (
            "shared_local",
            {
                "architecture": "serial_tree",
                "mechanism": "shunting",
                "credit": "local_shared",
            },
        ),
        (
            "path_local",
            {
                "architecture": "serial_tree",
                "mechanism": "shunting",
                "credit": "local_path",
            },
        ),
    ):
        for regime in ("aligned", "rewired_tree"):
            for shallow in (3, 1):
                add(
                    f"depth__{label}__{regime}__d4_d{shallow}",
                    "primary",
                    difference(
                        frame,
                        dict(**base, regime=regime, depth=4),
                        dict(**base, regime=regime, depth=shallow),
                    ),
                    f"H=4 {regime}: D4 minus D{shallow} for {label}",
                )
        for shallow in (3, 1):
            add(
                f"placement_interaction__{label}__d4_d{shallow}",
                "primary",
                seed_values[f"depth__{label}__aligned__d4_d{shallow}"]
                - seed_values[f"depth__{label}__rewired_tree__d4_d{shallow}"],
                f"H=4 aligned-minus-reversed D4-minus-D{shallow} interaction for {label}",
            )

    for regime in ("aligned", "rewired_tree"):
        for depth in (1, 2, 3, 4):
            add(
                f"serial_minus_grouped__{regime}__d{depth}",
                "architecture",
                difference(
                    frame,
                    dict(**serial_bp, regime=regime, depth=depth),
                    dict(**grouped_bp, regime=regime, depth=depth),
                ),
                f"H=4 {regime}: serial tree minus literal grouped point at D{depth}",
            )
    add(
        "architecture_placement_interaction__d4",
        "primary",
        seed_values["serial_minus_grouped__aligned__d4"]
        - seed_values["serial_minus_grouped__rewired_tree__d4"],
        "H=4 aligned-minus-reversed serial-minus-grouped interaction at D4",
    )

    add(
        "depth__additive_bp__aligned__d4_d3",
        "mechanism",
        difference(
            frame,
            dict(**additive_bp, regime="aligned", depth=4),
            dict(**additive_bp, regime="aligned", depth=3),
        ),
        "H=4 aligned raw-additive BP: D4 minus D3",
    )
    add(
        "shunting_additive_depth_interaction__aligned__d4_d3",
        "primary",
        seed_values["depth__serial_bp__aligned__d4_d3"]
        - seed_values["depth__additive_bp__aligned__d4_d3"],
        "H=4 aligned shunting-minus-additive interaction in D4 minus D3",
    )

    contrasts = pd.DataFrame(rows)
    primary = contrasts.family.eq("primary")
    contrasts.loc[primary, "bh_adjusted_p_primary_family"] = _bh_adjust(
        contrasts.loc[primary, "exact_two_sided_sign_flip_p"]
    )
    seed_frame = pd.DataFrame(seed_values).rename_axis("seed").reset_index()
    return contrasts, seed_frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["hierarchy", "regime", "architecture", "mechanism", "credit", "depth"]
    for index, (key, part) in enumerate(frame.groupby(grouping, sort=True)):
        mean, low, high = bootstrap_mean(
            part.test_accuracy.to_numpy(float), 15100 + index
        )
        rows.append(
            dict(
                zip(grouping, key),
                n_seeds=part.seed.nunique(),
                mean_test_accuracy=mean,
                ci_low=low,
                ci_high=high,
            )
        )
    return pd.DataFrame(rows)


def audit(frame: pd.DataFrame, collection: dict[str, Any]) -> dict[str, Any]:
    finite_columns = [
        column
        for column in frame.columns
        if column.endswith(("_accuracy", "_auc", "_categorical_loglikelihood"))
    ]
    checks: dict[str, Any] = {
        **collection,
        "unique_rows": int(
            frame[["cohort", "config_index"]].drop_duplicates().shape[0]
        ),
        "duplicate_seed_conditions": int(
            frame.duplicated(
                ["regime", "architecture", "mechanism", "credit", "depth", "seed"]
            ).sum()
        ),
        "all_metrics_finite": bool(
            len(frame) > 0 and np.isfinite(frame[finite_columns].to_numpy(float)).all()
        ),
        "fallback_mentions": int(frame.fallback_mentions.sum()) if len(frame) else 0,
        "nonfinite_alerts": int(frame.nonfinite_alert.sum()) if len(frame) else 0,
        "observed_seeds": sorted(map(int, frame.seed.unique())) if len(frame) else [],
    }
    if collection["status"] != "complete":
        checks["status"] = "incomplete"
        return checks

    resource_failures: list[str] = []
    bp = frame[frame.credit.eq("full_bp")]
    for (regime, depth), group in bp.groupby(["regime", "depth"]):
        for column in RESOURCE_COLUMNS:
            if group[column].nunique() != 1:
                resource_failures.append(f"{regime}/D{depth}/{column}")
    for (regime, credit), group in frame[
        frame.architecture.eq("serial_tree") & frame.mechanism.eq("shunting")
    ].groupby(["regime", "credit"]):
        for column in RESOURCE_COLUMNS:
            if group[column].nunique() != 1:
                resource_failures.append(
                    f"{regime}/{credit}/fixed-depth-budget/{column}"
                )

    checks["resource_failures"] = resource_failures
    checks["resource_gate"] = not resource_failures
    checks["expected_seeds"] = checks["observed_seeds"] == SEEDS
    checks["expected_row_count_gate"] = checks["expected_row_count"] == 360
    initialization = json.loads(INITIALIZATION_AUDIT.read_text())
    checks["d1_initialization_audit"] = initialization
    checks["d1_initialization_gate"] = bool(
        initialization.get("status") == "pass"
        and initialization.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and initialization.get("logits_bitwise_equal") is True
        and float(initialization.get("maximum_absolute_logit_difference", np.inf))
        == 0.0
    )
    checks["status"] = (
        "complete_pass"
        if checks["all_metrics_finite"]
        and checks["fallback_mentions"] == 0
        and checks["nonfinite_alerts"] == 0
        and checks["duplicate_seed_conditions"] == 0
        and checks["resource_gate"]
        and checks["expected_seeds"]
        and checks["expected_row_count_gate"]
        and checks["d1_initialization_gate"]
        and not checks["contract_failures"]
        and not checks["source_failures"]
        else "complete_fail"
    )
    return checks


def _line_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    regime: str,
    letter: str,
    title: str,
    legend: bool,
) -> None:
    styles = [
        (
            "serial_tree",
            "shunting",
            "full_bp",
            "serial BP",
            COLORS["shunting"],
            "o",
            "-",
        ),
        (
            "grouped_point",
            "shunting",
            "full_bp",
            "grouped point BP",
            COLORS["point_mlp"],
            "s",
            "--",
        ),
        (
            "serial_tree",
            "shunting",
            "local_shared",
            "shared LocalCA",
            COLORS["local"],
            "D",
            "-.",
        ),
        (
            "serial_tree",
            "shunting",
            "local_path",
            "path LocalCA",
            COLORS["pathway"],
            "^",
            ":",
        ),
    ]
    if regime == "aligned":
        styles.append(
            (
                "serial_tree",
                "raw_additive",
                "full_bp",
                "raw-additive BP",
                COLORS["mute"],
                "v",
                "--",
            )
        )
    for architecture, mechanism, credit, label, color, marker, linestyle in styles:
        part = summary[
            summary.regime.eq(regime)
            & summary.architecture.eq(architecture)
            & summary.mechanism.eq(mechanism)
            & summary.credit.eq(credit)
        ].sort_values("depth")
        if part.empty:
            continue
        y = part.mean_test_accuracy.to_numpy(float)
        low = part.ci_low.to_numpy(float)
        high = part.ci_high.to_numpy(float)
        ax.errorbar(
            part.depth,
            y,
            yerr=np.vstack([y - low, high - y]),
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            markersize=MARKER_MS,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    panel_title(ax, letter, title)
    ax.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax.set_ylabel("test accuracy")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.7, 4.3)
    # Headroom so the topmost mean marker and its error-bar cap clear the
    # axes edge instead of being clipped against the top spine.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.05 * (high - low))
    style_axis(ax, grid="y")
    if legend:
        clean_legend(ax, loc="best", fontsize=PT_LEGEND, handlelength=2.0)
    else:
        ax.text(
            0.03,
            0.965,
            "key as in A",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PT_SMALL,
            color=COLORS["mute"],
        )


def _forest(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    names = [
        "depth__serial_bp__aligned__d4_d3",
        "placement_interaction__serial_bp__d4_d3",
        "serial_minus_grouped__aligned__d4",
        "architecture_placement_interaction__d4",
        "depth__shared_local__aligned__d4_d3",
        "depth__path_local__aligned__d4_d3",
        "shunting_additive_depth_interaction__aligned__d4_d3",
    ]
    labels = [
        "serial BP\nD4$-$D3",
        "depth ×\nplacement",
        "serial$-$point\nat D4",
        "architecture ×\nplacement",
        "shared local\nD4$-$D3",
        "path LocalCA\nD4$-$D3",
        "shunting ×\ndepth",
    ]
    colors = [
        COLORS["shunting"],
        COLORS["bp"],
        COLORS["point_mlp"],
        COLORS["bp"],
        COLORS["local"],
        COLORS["pathway"],
        COLORS["mute"],
    ]
    indexed = contrasts.set_index("contrast")
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color=COLORS["mute"], linewidth=LW_HAIR, zorder=0)
    for yi, name, color in zip(y, names, colors):
        row = indexed.loc[name]
        mean, low, high = map(float, (row.mean_pp, row.ci_low_pp, row.ci_high_pp))
        ax.errorbar(
            mean,
            yi,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=MARKER_MS,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    panel_title(ax, "C", "Frozen primary contrasts")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("paired difference (pp)")
    style_axis(ax, grid="x")


def _cross_hierarchy_panel(ax: plt.Axes, h4: pd.DataFrame) -> None:
    reference_path = (
        CLEAN_H23_REFERENCE if CLEAN_H23_REFERENCE.is_file() else H23_REFERENCE
    )
    reference = pd.read_csv(reference_path)
    reference = reference[
        reference.hierarchy.isin([2, 3])
        & reference.regime.eq("aligned")
        & reference.architecture.eq("serial_tree")
        & reference.credit.eq("full_bp")
    ].copy()
    # H=2/H=3 references are shunting cohorts; normalize the legacy column name.
    if "mechanism" in reference:
        reference = reference[reference.mechanism.eq("shunting")]
    points: list[tuple[int, int, float]] = []
    for hierarchy, part in reference.groupby("hierarchy"):
        means = part.groupby("depth").test_accuracy.mean()
        points.append((int(hierarchy), int(means.idxmax()), float(means.max())))
    part4 = h4[
        h4.regime.eq("aligned")
        & h4.architecture.eq("serial_tree")
        & h4.mechanism.eq("shunting")
        & h4.credit.eq("full_bp")
    ]
    means4 = part4.groupby("depth").test_accuracy.mean()
    points.append((4, int(means4.idxmax()), float(means4.max())))
    points.sort()
    hierarchy = np.asarray([p[0] for p in points])
    optimum = np.asarray([p[1] for p in points])
    ax.plot(
        [1.8, 4.2], [1.8, 4.2], linestyle="--", color=COLORS["mute"], linewidth=LW_HAIR
    )
    ax.plot(
        hierarchy,
        optimum,
        color=COLORS["shunting"],
        marker="o",
        linewidth=LW_DATA,
        markersize=MARKER_MS + 1,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    for h, d, accuracy in points:
        ax.annotate(
            f"{accuracy:.3f}",
            (h, d),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=PT_SMALL,
        )
    panel_title(ax, "D", "Trained optimum across task depth")
    ax.set_xlabel(r"task hierarchy $H$")
    ax.set_ylabel(r"best mean physical depth $D_{\mathrm{p}}^*$")
    ax.set_xticks([2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlim(1.75, 4.25)
    ax.set_ylim(0.75, 4.25)
    style_axis(ax, grid="both")


def make_figure(
    summary: pd.DataFrame, contrasts: pd.DataFrame, frame: pd.DataFrame
) -> None:
    apply_neurips_style()
    fig = plt.figure(figsize=(FIG_W, 6.1))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.175,
        right=0.985,
        top=0.925,
        bottom=0.105,
        wspace=0.52,
        hspace=0.62,
    )
    axes = [
        fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)
    ]
    _line_panel(
        axes[0],
        summary,
        regime="aligned",
        letter="A",
        title="H=4 aligned hierarchy",
        legend=True,
    )
    _line_panel(
        axes[1],
        summary,
        regime="rewired_tree",
        letter="B",
        title="H=4 reversed placement",
        legend=False,
    )
    _forest(axes[2], contrasts)
    _cross_hierarchy_panel(axes[3], frame)
    fig.canvas.draw()
    audit_layout(fig, "fig_physical_depth_h4_factorial")
    audit_text_over_data(fig, "fig_physical_depth_h4_factorial")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "fig_physical_depth_h4_factorial.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_physical_depth_h4_factorial.png", dpi=600)
    plt.close(fig)


def write_report(contrasts: pd.DataFrame, record: dict[str, Any]) -> None:
    indexed = contrasts.set_index("contrast")

    def sentence(name: str) -> str:
        row = indexed.loc[name]
        adjusted = row.bh_adjusted_p_primary_family
        q_text = "n/a" if pd.isna(adjusted) else f"{float(adjusted):.4f}"
        return (
            f"{row.mean_pp:.2f} pp (95% paired-seed bootstrap interval "
            f"{row.ci_low_pp:.2f} to {row.ci_high_pp:.2f}; "
            f"{int(row.positive_pairs)}/10 positive; exact two-sided sign-flip "
            f"P={row.exact_two_sided_sign_flip_p:.4f}; primary-family BH q={q_text}; "
            f"positive gate={'pass' if row.positive_claim_gate else 'fail'})"
        )

    report = f"""# Frozen H=4 physical-depth factorial

Status: **{record["status"]}** ({record["observed_rows"]}/360 fits).

## Depth and placement

- Aligned serial-BP D4 minus D3: {sentence("depth__serial_bp__aligned__d4_d3")}.
- Aligned serial-BP D4 minus D1: {sentence("depth__serial_bp__aligned__d4_d1")}.
- Serial-BP D4-minus-D3 depth-by-placement interaction: {sentence("placement_interaction__serial_bp__d4_d3")}.

## Point-neuron and local-credit controls

- Serial minus literal grouped point at aligned D4: {sentence("serial_minus_grouped__aligned__d4")}.
- Architecture-by-placement interaction at D4: {sentence("architecture_placement_interaction__d4")}.
- Shared-LocalCA aligned D4 minus D3: {sentence("depth__shared_local__aligned__d4_d3")}.
- Path-LocalCA aligned D4 minus D3: {sentence("depth__path_local__aligned__d4_d3")}.

## Mechanism specificity

- Raw-additive aligned D4 minus D3: {sentence("depth__additive_bp__aligned__d4_d3")}.
- Shunting-minus-additive interaction in D4 minus D3: {sentence("shunting_additive_depth_interaction__aligned__d4_d3")}.

All frozen contrasts, including null or reversed outcomes, remain in the source-data tables. A positive directional statement requires an interval excluding zero and at least 8/10 paired differences in the predicted direction. Exact tests are seed-level; primary-family q values use Benjamini--Hochberg correction.
"""
    (OUTPUT / "RESULTS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--d3-repair-runs-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    frame, collection = collect(
        args.runs_root.resolve(),
        d3_repair_runs=(
            args.d3_repair_runs_root.resolve()
            if args.d3_repair_runs_root is not None
            else None
        ),
        allow_incomplete=args.allow_incomplete,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False)
    record = audit(frame, collection)
    (OUTPUT / "audit.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if collection["status"] != "complete":
        print(json.dumps(record, indent=2))
        return

    summary = summarize(frame)
    contrasts, seed_contrasts = build_contrasts(frame)
    summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
    contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    seed_contrasts.to_csv(OUTPUT / "paired_contrasts_by_seed.csv", index=False)
    write_report(contrasts, record)
    make_figure(summary, contrasts, frame)
    print(json.dumps(record, indent=2))
    print((OUTPUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
