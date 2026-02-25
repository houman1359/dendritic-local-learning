"""
Aggregate results from the gradient_fidelity_vs_ie sweep experiment.

Reads config.json and performance/final.json from each config directory,
extracts key parameters, and produces raw + summary CSVs.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/"
    "LOCAL_LEARNING/sweep_runs/"
    "sweep_neurips_gradient_fidelity_vs_ie_20260225004141/results"
)

OUTPUT_DIR = Path(
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/"
    "drafts/dendritic-local-learning/data"
)

RAW_CSV = OUTPUT_DIR / "gradient_fidelity_vs_ie_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "gradient_fidelity_vs_ie_summary.csv"


def load_single_result(config_dir: Path) -> dict | None:
    """Load config and performance from a single config directory."""
    config_path = config_dir / "config.json"
    perf_path = config_dir / "performance" / "final.json"

    if not config_path.exists():
        print(f"  SKIP {config_dir.name}: missing config.json", file=sys.stderr)
        return None
    if not perf_path.exists():
        print(f"  SKIP {config_dir.name}: missing performance/final.json", file=sys.stderr)
        return None

    with open(config_path) as f:
        config = json.load(f)
    with open(perf_path) as f:
        perf = json.load(f)

    core_type = config["model"]["core"]["type"]
    ie_synapses_list = config["model"]["core"]["connectivity"][
        "ie_synapses_per_branch_per_layer"
    ]
    ie_synapses = ie_synapses_list[0]
    dataset_name = config["data"]["dataset_name"]
    seed = config["experiment"]["seed"]

    test_acc = perf["accuracy"]["test"]
    train_acc = perf["accuracy"]["train"]
    valid_acc = perf["accuracy"]["valid"]

    return {
        "config_dir": config_dir.name,
        "core_type": core_type,
        "ie_synapses": ie_synapses,
        "dataset_name": dataset_name,
        "seed": seed,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "valid_accuracy": valid_acc,
    }


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all config directories
    config_dirs = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("config_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    print(f"Found {len(config_dirs)} config directories in:\n  {RESULTS_DIR}\n")

    # Load all results
    rows = []
    skipped = 0
    for config_dir in config_dirs:
        result = load_single_result(config_dir)
        if result is not None:
            rows.append(result)
        else:
            skipped += 1

    if not rows:
        print("ERROR: No results loaded!", file=sys.stderr)
        sys.exit(1)

    df_raw = pd.DataFrame(rows)
    print(f"Loaded {len(df_raw)} results ({skipped} skipped)\n")

    # Save raw data
    df_raw.to_csv(RAW_CSV, index=False)
    print(f"Raw data saved to:\n  {RAW_CSV}\n")

    # --- Aggregated summary ---
    group_cols = ["core_type", "ie_synapses", "dataset_name"]
    agg = (
        df_raw.groupby(group_cols)
        .agg(
            n_seeds=("seed", "count"),
            seeds=("seed", lambda s: sorted(s.tolist())),
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            train_acc_mean=("train_accuracy", "mean"),
            train_acc_std=("train_accuracy", "std"),
            valid_acc_mean=("valid_accuracy", "mean"),
            valid_acc_std=("valid_accuracy", "std"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    # Save summary
    agg.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary saved to:\n  {SUMMARY_CSV}\n")

    # --- Print nice summary table ---
    print("=" * 100)
    print("GRADIENT FIDELITY vs IE SYNAPSES — SUMMARY")
    print("=" * 100)

    for dataset in sorted(df_raw["dataset_name"].unique()):
        subset = agg[agg["dataset_name"] == dataset]
        print(f"\n--- Dataset: {dataset} ---")
        print(
            f"{'Core Type':<25s} {'IE Syn':>7s} {'N':>3s} "
            f"{'Test Acc':>12s} {'Train Acc':>12s} {'Valid Acc':>12s}"
        )
        print("-" * 80)
        for _, row in subset.iterrows():
            test_str = f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}"
            train_str = f"{row['train_acc_mean']:.4f} ± {row['train_acc_std']:.4f}"
            valid_str = f"{row['valid_acc_mean']:.4f} ± {row['valid_acc_std']:.4f}"
            print(
                f"{row['core_type']:<25s} {int(row['ie_synapses']):>7d} {int(row['n_seeds']):>3d} "
                f"{test_str:>12s} {train_str:>12s} {valid_str:>12s}"
            )

    print("\n" + "=" * 100)

    # Also print a pivot-style view: ie_synapses as columns, core_type as rows
    print("\nPIVOT VIEW — Test Accuracy (mean ± std)")
    print("=" * 100)
    for dataset in sorted(df_raw["dataset_name"].unique()):
        subset = agg[agg["dataset_name"] == dataset]
        ie_vals = sorted(subset["ie_synapses"].unique())
        core_types = sorted(subset["core_type"].unique())

        # Header
        header = f"{'Dataset: ' + dataset:<30s}"
        for ie in ie_vals:
            header += f"  IE={str(int(ie)):<14s}"
        print(header)
        print("-" * (30 + 18 * len(ie_vals)))

        for ct in core_types:
            line = f"{ct:<30s}"
            for ie in ie_vals:
                match = subset[(subset["core_type"] == ct) & (subset["ie_synapses"] == ie)]
                if len(match) == 1:
                    r = match.iloc[0]
                    line += f"  {r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f}  "
                else:
                    line += f"  {'N/A':^14s}"
            print(line)
        print()


if __name__ == "__main__":
    main()
