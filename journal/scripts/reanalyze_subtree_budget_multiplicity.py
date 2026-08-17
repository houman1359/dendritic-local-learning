#!/usr/bin/env python3
"""Apply the frozen four-budget BH correction to the ancestry scan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "source_data"
    / "trained_subtree_address_full_factorial"
    / "paired_contrasts.csv"
)
OUTPUT = ROOT / "source_data" / "trained_subtree_address_full_factorial"


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def main() -> None:
    frame = pd.read_csv(SOURCE)
    scan = frame[
        frame.architecture.eq("dendritic_tree")
        & frame.contrast.eq("correct - best_matched_nonanatomical_oracle")
        & frame.endpoint.eq("heldout_accuracy")
    ].sort_values("budget_k").copy()
    if scan.budget_k.tolist() != [1, 2, 4, 8]:
        raise RuntimeError(f"Unexpected budget scan: {scan.budget_k.tolist()}")
    scan["bh_family"] = "four ancestry-route budgets"
    scan["bh_adjusted_p"] = benjamini_hochberg(
        scan.wilcoxon_p_two_sided.to_numpy(float)
    )
    scan.to_csv(
        OUTPUT / "budget_scan_bh_correction.csv", index=False, float_format="%.10g"
    )
    k4 = scan[scan.budget_k.eq(4)].iloc[0]
    payload = {
        "family": "four ancestry-route budgets K=1,2,4,8",
        "method": "Benjamini-Hochberg",
        "n_tests": 4,
        "k4_raw_p": float(k4.wilcoxon_p_two_sided),
        "k4_bh_adjusted_p": float(k4.bh_adjusted_p),
        "k4_reject_at_fdr_0_05": bool(k4.bh_adjusted_p < 0.05),
    }
    (OUTPUT / "budget_scan_bh_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
