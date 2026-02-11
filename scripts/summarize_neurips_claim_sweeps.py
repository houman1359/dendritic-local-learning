#!/usr/bin/env python3
"""Summarize NeurIPS claim-driven local-learning sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SWEEPS = {
    "claim1": "sweep_neurips_claim1_mechanism_mnist_*",
    "claim2": "sweep_neurips_claim2_decoder_locality_multidataset_*",
    "claim3": "sweep_neurips_claim3_shunting_regime_*",
    "claim4": "sweep_neurips_claim4_source_analysis_*",
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


def _add_claim_name_column(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["claim_name"] = "unknown"
    result.loc[result["sweep_dir"].str.contains("claim1_mechanism"), "claim_name"] = (
        "claim1"
    )
    result.loc[
        result["sweep_dir"].str.contains("claim2_decoder_locality"), "claim_name"
    ] = "claim2"
    result.loc[result["sweep_dir"].str.contains("claim3_shunting_regime"), "claim_name"] = (
        "claim3"
    )
    result.loc[result["sweep_dir"].str.contains("claim4_source_analysis"), "claim_name"] = (
        "claim4"
    )
    return result


def _group_mean(data: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=keys + metrics)
    return (
        data.groupby(keys, dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values("valid_accuracy", ascending=False)
    )


def _group_mean_std(
    data: pd.DataFrame, keys: list[str], metrics: list[str]
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=keys)

    grouped = (
        data.groupby(keys, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    sort_col = "valid_accuracy_mean" if "valid_accuracy_mean" in grouped.columns else None
    if sort_col is not None:
        grouped = grouped.sort_values(sort_col, ascending=False)
    return grouped


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_\n"
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for values in df.fillna("").astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/neurips_claims"
        ),
    )
    parser.add_argument("--sweep-dir", action="append", default=[])
    args = parser.parse_args()

    sweep_dirs: list[Path] = [Path(path_str) for path_str in args.sweep_dir]
    if not sweep_dirs:
        for pattern in DEFAULT_SWEEPS.values():
            match = _latest_match(args.sweep_root, pattern)
            if match is not None:
                sweep_dirs.append(match)

    if not sweep_dirs:
        raise RuntimeError("No claim sweep directories found.")

    frames = [_load_processed_csv(path) for path in sweep_dirs]
    all_data = pd.concat(frames, ignore_index=True)
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
    all_data = _add_claim_name_column(all_data)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(output_dir / "combined_results.csv", index=False)

    metrics = ["valid_accuracy", "test_accuracy", "valid_categorical_loglikelihood"]

    best_per_claim = (
        all_data.sort_values("valid_accuracy", ascending=False)
        .groupby("claim_name", as_index=False)
        .first()[
            [
                "claim_name",
                "config_id",
                "dataset",
                "network_type",
                "rule_variant",
                "error_broadcast_mode",
                "decoder_update_mode",
                "use_path_propagation",
                "morphology_modulator_mode",
                "use_dendritic_normalization",
                *metrics,
                "sweep_dir",
            ]
        ]
    )
    best_per_claim.to_csv(output_dir / "best_per_claim.csv", index=False)

    claim1 = all_data[all_data["claim_name"] == "claim1"]
    claim2 = all_data[all_data["claim_name"] == "claim2"]
    claim3 = all_data[all_data["claim_name"] == "claim3"]
    claim4 = all_data[all_data["claim_name"] == "claim4"]

    claim1_interaction = _group_mean(
        claim1,
        ["error_broadcast_mode", "use_path_propagation"],
        metrics,
    )
    claim2_decoder_dataset = _group_mean(
        claim2,
        ["dataset", "decoder_update_mode"],
        metrics,
    )
    claim3_group_keys = ["network_type", "ie_value", "error_broadcast_mode"]
    if "dataset" in claim3.columns:
        claim3_group_keys = ["dataset", *claim3_group_keys]

    claim3_regime = _group_mean(
        claim3,
        claim3_group_keys,
        metrics,
    )

    mi_cols = [
        column
        for column in claim4.columns
        if column.startswith("mi_") and not column.endswith("_std")
    ]
    claim4_metrics = ["valid_accuracy", "test_accuracy"]
    claim4_metrics.extend(mi_cols[:6])
    claim4_metrics = [column for column in claim4_metrics if column in claim4.columns]
    claim4_source = _group_mean_std(
        claim4,
        ["error_broadcast_mode", "use_path_propagation"],
        claim4_metrics,
    )

    claim1_interaction.to_csv(output_dir / "claim1_interaction.csv", index=False)
    claim2_decoder_dataset.to_csv(output_dir / "claim2_decoder_dataset.csv", index=False)
    claim3_regime.to_csv(output_dir / "claim3_regime.csv", index=False)
    claim4_source.to_csv(output_dir / "claim4_source_metrics.csv", index=False)

    summary_path = output_dir / "claim_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# NeurIPS Claim Sweep Summary\n\n")
        handle.write("## Best per claim\n\n")
        handle.write(_to_md(best_per_claim))
        handle.write("\n\n")
        handle.write("## Claim1 interaction (broadcast x path)\n\n")
        handle.write(_to_md(claim1_interaction))
        handle.write("\n\n")
        handle.write("## Claim2 decoder locality by dataset\n\n")
        handle.write(_to_md(claim2_decoder_dataset))
        handle.write("\n\n")
        handle.write("## Claim3 shunting regime table\n\n")
        handle.write(_to_md(claim3_regime))
        handle.write("\n\n")
        handle.write("## Claim4 source metrics (top columns)\n\n")
        handle.write(_to_md(claim4_source))
        handle.write("\n")

    print(f"Wrote claim summary to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
