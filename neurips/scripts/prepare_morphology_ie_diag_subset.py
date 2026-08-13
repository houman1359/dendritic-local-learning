#!/usr/bin/env python3
"""Prepare a representative subset of morphology x inhibition runs for diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
DEFAULT_SWEEP_DIR = (
    DRAFT_DIR
    / "local_sweep_runs"
    / "noise_resilience_morphology_ie_regime_20260402004716"
)
DEFAULT_OUTPUT_DIR = DRAFT_DIR / "analysis" / "morphology_ie_diag_subset"


CELL_SPECS = [
    ("dendritic_shunting", "[4, 4]", 0, "best_depth2_lowI_shunting"),
    ("dendritic_additive", "[4, 4]", 0, "best_depth2_lowI_additive"),
    ("dendritic_shunting", "[3, 3, 3]", 5, "best_depth3_midI_shunting"),
    ("dendritic_additive", "[3, 3, 3]", 5, "best_depth3_midI_additive"),
    ("dendritic_shunting", "[3, 3]", 20, "highI_collapse_shunting"),
    ("dendritic_additive", "[3, 3]", 20, "highI_match_additive"),
    ("dendritic_shunting", "[2, 2]", 40, "very_highI_collapse_shunting"),
    ("dendritic_additive", "[2, 2]", 40, "very_highI_match_additive"),
]


def _load_runs(sweep_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cfg_path in sorted((sweep_dir / "configs").glob("unified_config_*.yaml")):
        idx = cfg_path.stem.split("_")[-1]
        perf_path = sweep_dir / "results" / f"config_{idx}" / "performance" / "final.json"
        if not perf_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        perf = json.loads(perf_path.read_text())
        rows.append(
            {
                "config_idx": int(idx),
                "run_dir": str(sweep_dir / "results" / f"config_{idx}"),
                "network_type": cfg["model"]["core"]["type"],
                "branch_factors": str(cfg["model"]["core"]["architecture"]["excitatory_branch_factors"]),
                "ie": cfg["model"]["core"]["connectivity"]["ie_synapses_per_branch_per_layer"][0],
                "seed": cfg["experiment"]["seed"],
                "valid_acc": perf["accuracy"]["valid"],
                "test_acc": perf["accuracy"]["test"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sweep_dir = DEFAULT_SWEEP_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_sweep = output_dir / "selected_sweep"
    results_dir = subset_sweep / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = _load_runs(sweep_dir)
    selected_rows: list[pd.Series] = []
    for network_type, branch_factors, ie, tag in CELL_SPECS:
        sub = df[
            (df["network_type"] == network_type)
            & (df["branch_factors"] == branch_factors)
            & (df["ie"] == ie)
        ].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["valid_acc", "test_acc"]).reset_index(drop=True)
        row = sub.iloc[len(sub) // 2].copy()
        row["tag"] = tag
        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "selected_runs.csv", index=False)

    for new_idx, row in enumerate(selected.itertuples(index=False)):
        src = Path(row.run_dir)
        dst = results_dir / f"config_{new_idx}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src, target_is_directory=True)

    manifest = {
        "source_sweep": str(sweep_dir),
        "selected_runs_csv": str(output_dir / "selected_runs.csv"),
        "num_selected_runs": len(selected),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(selected)} runs under {subset_sweep}")


if __name__ == "__main__":
    main()
