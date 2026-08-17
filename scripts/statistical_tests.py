#!/usr/bin/env python3
"""Compute statistical tests (CIs, p-values, Cohen's d) on sweep results.

Reads the analysis CSV produced by ``summarize_neurips_phase_sweeps.py`` or
directly from sweep directories.  Performs pairwise comparisons between
conditions and outputs a publication-ready summary table.

Outputs:
- ``statistical_tests.csv``: all pairwise comparisons with CI, p, Cohen's d
- ``stats_summary.txt``: human-readable summary

Usage:
    python statistical_tests.py --sweep-root <path> --output-dir <path>
    python statistical_tests.py --csv <existing_summary.csv> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def _dig(data: dict, keys: list[str], default=None):
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_sweep_results(sweep_root: Path) -> pd.DataFrame:
    """Load final test accuracy from all sweep configs."""
    rows = []
    for sweep_dir in sorted(sweep_root.iterdir()):
        if not sweep_dir.is_dir() or not sweep_dir.name.startswith("sweep_"):
            continue
        results_dir = sweep_dir / "results"
        if not results_dir.exists():
            continue

        for config_dir in sorted(results_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            cfg = _safe_json(config_dir / "config.json")
            perf = _safe_json(config_dir / "performance" / "final.json")
            if cfg is None or perf is None:
                continue

            acc = perf.get("accuracy", {})
            local = _dig(cfg, ["training", "main", "learning_strategy_config"], {}) or {}
            arch = _dig(cfg, ["model", "core", "architecture"], {}) or {}

            rows.append({
                "sweep": sweep_dir.name,
                "config_id": config_dir.name,
                "strategy": _dig(cfg, ["training", "main", "strategy"], ""),
                "core_type": _dig(cfg, ["model", "core", "type"], ""),
                "use_shunting": arch.get("use_shunting"),
                "rule_variant": local.get("rule_variant"),
                "error_broadcast_mode": local.get("error_broadcast_mode"),
                "dataset": _dig(cfg, ["data", "dataset_name"], ""),
                "seed": _dig(cfg, ["experiment", "seed"]),
                "test_accuracy": acc.get("test"),
                "valid_accuracy": acc.get("valid"),
                "train_accuracy": acc.get("train"),
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Statistical helpers
# ------------------------------------------------------------------ #
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD) for two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if pooled_var == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def bootstrap_ci(
    x: np.ndarray, confidence: float = 0.95, n_boot: int = 10000
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - confidence) / 2
    return (float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha))))


def mean_ci(x: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using t-distribution."""
    n = len(x)
    m = np.mean(x)
    if n < 2:
        return float(m), np.nan, np.nan
    se = stats.sem(x)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(m), float(m - h), float(m + h)


# ------------------------------------------------------------------ #
# Pairwise comparison
# ------------------------------------------------------------------ #
def pairwise_tests(
    df: pd.DataFrame,
    metric: str = "test_accuracy",
    condition_col: str = "condition",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute pairwise t-tests, CIs, and Cohen's d between conditions."""
    conditions = sorted(df[condition_col].unique())
    records = []

    # Per-condition summary
    for cond in conditions:
        vals = df.loc[df[condition_col] == cond, metric].dropna().values
        m, ci_lo, ci_hi = mean_ci(vals, confidence)
        records.append({
            "comparison": f"{cond} (summary)",
            "condition_a": cond,
            "condition_b": "",
            "n_a": len(vals),
            "n_b": 0,
            "mean_a": m,
            "mean_b": np.nan,
            "ci_lo_a": ci_lo,
            "ci_hi_a": ci_hi,
            "diff": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "p_corrected": np.nan,
            "cohens_d": np.nan,
            "significant": "",
        })

    # Pairwise comparisons
    pairs = list(combinations(conditions, 2))
    p_values = []
    pair_records = []

    for cond_a, cond_b in pairs:
        vals_a = df.loc[df[condition_col] == cond_a, metric].dropna().values
        vals_b = df.loc[df[condition_col] == cond_b, metric].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        d = cohens_d(vals_a, vals_b)
        ma, ci_lo_a, ci_hi_a = mean_ci(vals_a, confidence)
        mb, ci_lo_b, ci_hi_b = mean_ci(vals_b, confidence)

        p_values.append(p_val)
        pair_records.append({
            "comparison": f"{cond_a} vs {cond_b}",
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n_a": len(vals_a),
            "n_b": len(vals_b),
            "mean_a": ma,
            "mean_b": mb,
            "ci_lo_a": ci_lo_a,
            "ci_hi_a": ci_hi_a,
            "diff": ma - mb,
            "t_stat": t_stat,
            "p_value": p_val,
            "p_corrected": np.nan,  # filled below
            "cohens_d": d,
            "significant": "",
        })

    # Holm-Bonferroni correction
    if p_values:
        from scipy.stats import false_discovery_control
        try:
            # scipy >= 1.11
            corrected = false_discovery_control(p_values, method="bh")
        except (ImportError, AttributeError):
            # Manual Holm-Bonferroni
            n_tests = len(p_values)
            sorted_idx = np.argsort(p_values)
            corrected = np.array(p_values, dtype=float)
            for rank, idx in enumerate(sorted_idx):
                corrected[idx] = min(1.0, p_values[idx] * n_tests / (rank + 1))

        for i, rec in enumerate(pair_records):
            rec["p_corrected"] = corrected[i]
            if corrected[i] < 0.001:
                rec["significant"] = "***"
            elif corrected[i] < 0.01:
                rec["significant"] = "**"
            elif corrected[i] < 0.05:
                rec["significant"] = "*"
            else:
                rec["significant"] = "ns"

    records.extend(pair_records)
    return pd.DataFrame(records)


def build_condition_label(row: pd.Series, cols: list[str]) -> str:
    """Build a composite condition label from multiple columns."""
    parts = []
    for c in cols:
        v = row.get(c)
        if v is not None and str(v) not in ("", "nan", "None"):
            parts.append(str(v))
    return "/".join(parts) if parts else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep-root", type=Path, help="Root of sweep_* directories")
    group.add_argument("--csv", type=Path, help="Pre-existing summary CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/stats"))
    parser.add_argument(
        "--condition-cols",
        nargs="+",
        default=["strategy", "rule_variant"],
        help="Columns to combine into condition label",
    )
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--filter-sweep", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = load_sweep_results(args.sweep_root)

    if df.empty:
        print("No data loaded.")
        return

    if args.filter_sweep:
        df = df[df["sweep"].str.contains(args.filter_sweep)]

    # Build condition column
    df["condition"] = df.apply(
        lambda r: build_condition_label(r, args.condition_cols), axis=1
    )

    print(f"Loaded {len(df)} runs across {df['condition'].nunique()} conditions")
    print(f"Conditions: {sorted(df['condition'].unique())}")

    results = pairwise_tests(df, metric=args.metric, condition_col="condition")

    csv_path = args.output_dir / "statistical_tests.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Human-readable summary
    txt_path = args.output_dir / "stats_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Statistical Comparison Summary\n")
        f.write("=" * 60 + "\n\n")

        summaries = results[results["condition_b"] == ""]
        if not summaries.empty:
            f.write("Per-condition summaries:\n")
            for _, row in summaries.iterrows():
                f.write(
                    f"  {row['condition_a']}: "
                    f"mean={row['mean_a']:.4f} "
                    f"[{row['ci_lo_a']:.4f}, {row['ci_hi_a']:.4f}] "
                    f"(n={row['n_a']})\n"
                )
            f.write("\n")

        comparisons = results[results["condition_b"] != ""]
        if not comparisons.empty:
            f.write("Pairwise comparisons:\n")
            for _, row in comparisons.iterrows():
                f.write(
                    f"  {row['comparison']}: "
                    f"diff={row['diff']:+.4f}, "
                    f"t={row['t_stat']:.3f}, "
                    f"p={row['p_value']:.4f} "
                    f"(corrected={row['p_corrected']:.4f} {row['significant']}), "
                    f"d={row['cohens_d']:.3f}\n"
                )

    print(f"Saved summary to {txt_path}")

    # Print key comparisons
    sig = comparisons[comparisons["significant"].isin(["*", "**", "***"])] if not comparisons.empty else pd.DataFrame()
    if not sig.empty:
        print(f"\nSignificant comparisons ({len(sig)}):")
        for _, row in sig.iterrows():
            print(
                f"  {row['comparison']}: "
                f"diff={row['diff']:+.4f}, p={row['p_corrected']:.4f} {row['significant']}, "
                f"d={row['cohens_d']:.3f}"
            )


if __name__ == "__main__":
    main()
