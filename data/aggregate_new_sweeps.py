#!/usr/bin/env python3
"""Aggregate results from three corrected sweeps (2026-02-25) into CSVs.

Produces:
  data/gradient_fidelity_vs_ie_corrected.csv       (raw, 100 rows)
  data/gradient_fidelity_vs_ie_corrected_summary.csv (grouped, 20 rows)
  data/fashion_mnist_competence.csv                 (raw, 20 rows)
  data/fashion_mnist_competence_summary.csv          (grouped, 4 rows)
  data/verification_seeds.csv                       (raw, 12 rows)
  data/verification_seeds_summary.csv                (grouped, 4 rows)
"""

import json
import sys
from pathlib import Path

import pandas as pd

SWEEP_BASE = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/"
    "LOCAL_LEARNING/sweep_runs"
)

SWEEPS = {
    "gradient_fidelity_vs_ie_corrected": (
        SWEEP_BASE / "sweep_neurips_gradient_fidelity_vs_ie_20260225030528" / "results"
    ),
    "fashion_mnist_competence": (
        SWEEP_BASE / "sweep_neurips_fashion_mnist_competence_20260225030611" / "results"
    ),
    "verification_seeds": (
        SWEEP_BASE / "sweep_neurips_verification_seeds_20260225030457" / "results"
    ),
}

OUTPUT_DIR = Path(__file__).resolve().parent


def load_single_result(config_dir: Path) -> dict | None:
    config_path = config_dir / "config.json"
    perf_path = config_dir / "performance" / "final.json"

    if not config_path.exists() or not perf_path.exists():
        print(f"  SKIP {config_dir.name}: missing files", file=sys.stderr)
        return None

    with open(config_path) as f:
        config = json.load(f)
    with open(perf_path) as f:
        perf = json.load(f)

    core_cfg = config["model"]["core"]
    conn = core_cfg.get("connectivity", {})

    ie_list = conn.get("ie_synapses_per_branch_per_layer", [0])
    ie_synapses = ie_list[0] if ie_list else 0

    strategy = config.get("training", {}).get("main", {}).get("strategy", "unknown")

    return {
        "config_dir": config_dir.name,
        "core_type": core_cfg["type"],
        "ie_synapses": ie_synapses,
        "dataset_name": config["data"]["dataset_name"],
        "seed": config["experiment"]["seed"],
        "strategy": strategy,
        "test_accuracy": perf["accuracy"]["test"],
        "train_accuracy": perf["accuracy"]["train"],
        "valid_accuracy": perf["accuracy"]["valid"],
    }


def aggregate_sweep(name: str, results_dir: Path):
    print(f"\n{'='*70}")
    print(f"Aggregating: {name}")
    print(f"  Source: {results_dir}")

    config_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("config_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    print(f"  Found {len(config_dirs)} config directories")

    rows = []
    for cd in config_dirs:
        result = load_single_result(cd)
        if result is not None:
            rows.append(result)

    if not rows:
        print(f"  ERROR: No results for {name}", file=sys.stderr)
        return

    df_raw = pd.DataFrame(rows)
    raw_path = OUTPUT_DIR / f"{name}.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"  Raw CSV ({len(df_raw)} rows): {raw_path}")

    group_cols = ["core_type", "dataset_name", "ie_synapses", "strategy"]
    # Only group by columns that have variation
    group_cols = [c for c in group_cols if df_raw[c].nunique() > 1 or c in ("core_type", "dataset_name")]

    agg = (
        df_raw.groupby(group_cols)
        .agg(
            n_seeds=("seed", "count"),
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            train_acc_mean=("train_accuracy", "mean"),
            valid_acc_mean=("valid_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    summary_path = OUTPUT_DIR / f"{name}_summary.csv"
    agg.to_csv(summary_path, index=False)
    print(f"  Summary CSV ({len(agg)} rows): {summary_path}")

    # Print summary table
    print(f"\n  {'Core Type':<25s} {'Dataset':<20s} {'IE':>4s} {'Strategy':<12s} {'N':>3s} {'Test Acc':>18s}")
    print(f"  {'-'*90}")
    for _, row in agg.iterrows():
        ie_str = str(int(row["ie_synapses"])) if "ie_synapses" in row.index else "-"
        strat = row.get("strategy", "-")
        test_str = f"{row['test_acc_mean']:.4f} +/- {row['test_acc_std']:.4f}"
        print(f"  {row['core_type']:<25s} {row['dataset_name']:<20s} {ie_str:>4s} {strat:<12s} {int(row['n_seeds']):>3d} {test_str:>18s}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, results_dir in SWEEPS.items():
        if not results_dir.exists():
            print(f"WARNING: {results_dir} not found, skipping {name}", file=sys.stderr)
            continue
        aggregate_sweep(name, results_dir)

    print(f"\n{'='*70}")
    print("All aggregations complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
