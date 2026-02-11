#!/usr/bin/env python3
"""Summarize local-learning pilot sweep outputs into comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SWEEPS = {
    "rule_variant": "sweep_localca_rule_variant_pilot_*",
    "broadcast_mode": "sweep_localca_broadcast_mode_pilot_*",
    "decoder_mode": "sweep_localca_decoder_mode_pilot_*",
    "morphology": "sweep_localca_morphology_core_factors_pilot_*",
    "interaction_focus": "sweep_localca_interaction_focus_pilot_*",
}


def _latest_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _load_processed_csv(sweep_dir: Path) -> pd.DataFrame:
    csv_path = sweep_dir / "plots" / "locallearning_processed_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing processed CSV: {csv_path}")
    data = pd.read_csv(csv_path)
    data["sweep_dir"] = str(sweep_dir)
    return data


def _extract_scalar_from_yaml(config_file: Path, key: str) -> str | None:
    prefix = f"{key}:"
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith(prefix):
                    return stripped.split(":", 1)[1].strip().strip("\"'")
    except FileNotFoundError:
        return None
    return None


def _ensure_columns_from_config(
    data: pd.DataFrame, column_to_yaml_key: dict[str, str]
) -> pd.DataFrame:
    fixed = data.copy()
    for column, yaml_key in column_to_yaml_key.items():
        if column not in fixed.columns:
            fixed[column] = None
        needs_fill = fixed[column].isna() | (fixed[column].astype(str) == "")
        if not needs_fill.any():
            continue
        fixed.loc[needs_fill, column] = fixed.loc[needs_fill, "config_file"].apply(
            lambda path: _extract_scalar_from_yaml(Path(path), yaml_key)
        )
    return fixed


def _add_sweep_name_column(all_data: pd.DataFrame) -> pd.DataFrame:
    data = all_data.copy()
    data["sweep_name"] = "unknown"
    data.loc[data["sweep_dir"].str.contains("rule_variant"), "sweep_name"] = "rule_variant"
    data.loc[data["sweep_dir"].str.contains("broadcast_mode"), "sweep_name"] = "broadcast_mode"
    data.loc[data["sweep_dir"].str.contains("decoder_mode"), "sweep_name"] = "decoder_mode"
    data.loc[data["sweep_dir"].str.contains("morphology_core_factors"), "sweep_name"] = "morphology"
    data.loc[data["sweep_dir"].str.contains("interaction_focus"), "sweep_name"] = "interaction_focus"
    return data


def _safe_group_mean(
    data: pd.DataFrame, cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    return (
        data.groupby(cols, dropna=False)[value_cols]
        .mean()
        .reset_index()
        .sort_values("valid_accuracy", ascending=False)
    )


def _write_markdown_summary(
    output_dir: Path,
    best_per_sweep: pd.DataFrame,
    rule_rank: pd.DataFrame,
    broadcast_rank: pd.DataFrame,
    decoder_rank: pd.DataFrame,
    morph_combo_rank: pd.DataFrame,
    morph_main_effects: pd.DataFrame,
    interaction_rank: pd.DataFrame,
) -> None:
    def _to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No data_\n"
        header = "| " + " | ".join(df.columns.astype(str)) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = []
        for values in df.fillna("").astype(str).itertuples(index=False, name=None):
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows]) + "\n"

    summary_path = output_dir / "pilot_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Local Learning Pilot Sweep Summary\n\n")
        handle.write("## Best setting per sweep\n\n")
        handle.write(_to_md(best_per_sweep))
        handle.write("\n\n")

        if not rule_rank.empty:
            handle.write("## Rule variant ranking\n\n")
            handle.write(_to_md(rule_rank))
            handle.write("\n\n")

        if not broadcast_rank.empty:
            handle.write("## Broadcast mode ranking\n\n")
            handle.write(_to_md(broadcast_rank))
            handle.write("\n\n")

        if not decoder_rank.empty:
            handle.write("## Decoder mode ranking\n\n")
            handle.write(_to_md(decoder_rank))
            handle.write("\n\n")

        if not morph_combo_rank.empty:
            handle.write("## Morphology combination ranking\n\n")
            handle.write(_to_md(morph_combo_rank))
            handle.write("\n\n")

        if not morph_main_effects.empty:
            handle.write("## Morphology main effects\n\n")
            handle.write(_to_md(morph_main_effects))
            handle.write("\n")

        if not interaction_rank.empty:
            handle.write("## Interaction focus ranking\n\n")
            handle.write(_to_md(interaction_rank))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
        ),
        help="Root folder containing sweep run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/pilot_summary"
        ),
        help="Directory for summary tables/reports.",
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        default=[],
        help="Optional explicit sweep directory (can pass multiple times).",
    )
    args = parser.parse_args()

    sweep_dirs: list[Path] = [Path(path_str) for path_str in args.sweep_dir]
    if not sweep_dirs:
        for pattern in DEFAULT_SWEEPS.values():
            match = _latest_match(args.sweep_root, pattern)
            if match is not None:
                sweep_dirs.append(match)

    if not sweep_dirs:
        raise RuntimeError("No sweep directories found.")

    all_frames = [_load_processed_csv(sweep_dir) for sweep_dir in sweep_dirs]
    all_data = pd.concat(all_frames, ignore_index=True)
    all_data = _ensure_columns_from_config(
        all_data,
        {
            "decoder_update_mode": "decoder_update_mode",
            "rule_variant": "rule_variant",
            "error_broadcast_mode": "error_broadcast_mode",
            "use_path_propagation": "use_path_propagation",
            "morphology_modulator_mode": "morphology_modulator_mode",
            "use_dendritic_normalization": "use_dendritic_normalization",
        },
    )
    all_data = _add_sweep_name_column(all_data)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(output_dir / "combined_results.csv", index=False)

    metric_cols = ["valid_accuracy", "test_accuracy", "valid_categorical_loglikelihood"]

    best_per_sweep = (
        all_data.sort_values("valid_accuracy", ascending=False)
        .groupby("sweep_name", as_index=False)
        .first()[
            [
                "sweep_name",
                "config_id",
                "rule_variant",
                "error_broadcast_mode",
                "decoder_update_mode",
                "use_path_propagation",
                "morphology_modulator_mode",
                "use_dendritic_normalization",
                *metric_cols,
                "sweep_dir",
            ]
        ]
    )
    best_per_sweep.to_csv(output_dir / "best_per_sweep.csv", index=False)

    rule_rank = _safe_group_mean(
        all_data[all_data["sweep_name"] == "rule_variant"],
        ["rule_variant"],
        metric_cols,
    )
    broadcast_rank = _safe_group_mean(
        all_data[all_data["sweep_name"] == "broadcast_mode"],
        ["error_broadcast_mode"],
        metric_cols,
    )
    decoder_rank = _safe_group_mean(
        all_data[all_data["sweep_name"] == "decoder_mode"],
        ["decoder_update_mode"],
        metric_cols,
    )

    morph_data = all_data[all_data["sweep_name"] == "morphology"]
    morph_combo_rank = _safe_group_mean(
        morph_data,
        [
            "use_path_propagation",
            "morphology_modulator_mode",
            "use_dendritic_normalization",
        ],
        metric_cols,
    )

    morph_effect_rows = []
    for factor in [
        "use_path_propagation",
        "morphology_modulator_mode",
        "use_dendritic_normalization",
    ]:
        if factor not in morph_data.columns:
            continue
        grouped = _safe_group_mean(morph_data, [factor], metric_cols)
        grouped.insert(0, "factor", factor)
        grouped.rename(columns={factor: "level"}, inplace=True)
        morph_effect_rows.append(grouped)
    morph_main_effects = (
        pd.concat(morph_effect_rows, ignore_index=True) if morph_effect_rows else pd.DataFrame()
    )
    interaction_rank = _safe_group_mean(
        all_data[all_data["sweep_name"] == "interaction_focus"],
        ["error_broadcast_mode", "use_path_propagation", "decoder_update_mode"],
        metric_cols,
    )

    rule_rank.to_csv(output_dir / "rule_variant_ranking.csv", index=False)
    broadcast_rank.to_csv(output_dir / "broadcast_mode_ranking.csv", index=False)
    decoder_rank.to_csv(output_dir / "decoder_mode_ranking.csv", index=False)
    morph_combo_rank.to_csv(output_dir / "morphology_combo_ranking.csv", index=False)
    morph_main_effects.to_csv(output_dir / "morphology_main_effects.csv", index=False)
    interaction_rank.to_csv(output_dir / "interaction_focus_ranking.csv", index=False)

    _write_markdown_summary(
        output_dir,
        best_per_sweep,
        rule_rank,
        broadcast_rank,
        decoder_rank,
        morph_combo_rank,
        morph_main_effects,
        interaction_rank,
    )

    print(f"Wrote pilot summary to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
