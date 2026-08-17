#!/usr/bin/env python3
"""Aggregate every eligible functional-topology scan within target cell."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EXTRACTS = ROOT.parents[1] / "dendritic-credit-routing" / "results" / "microns_functional_partner_responses"
OUT = ROOT / "source_data" / "functional_topology_all_scans"


def bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    rows = []
    for path in sorted(EXTRACTS.glob("target*_automatic_conservative/functional_topology/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        tests = value["tests"]
        rows.append(
            {
                "target_root_id": int(value["target_root_id"]),
                "target_nucleus_id": int(value["target_nucleus_id"]),
                "session": int(value["session"]),
                "scan_idx": int(value["scan_idx"]),
                "n_presynaptic_partners": int(value["n_presynaptic_partners"]),
                "n_pairs": int(value["n_pairs"]),
                "median_repeat_reliability": float(value["median_repeat_reliability"]),
                "shared_path_spearman_r": float(tests["shared_path_fraction"]["spearman_r"]),
                "negative_tree_distance_spearman_r": float(tests["negative_tree_distance"]["spearman_r"]),
                "shared_path_partial_r": float(tests["shared_path_partial_euclidean_and_depth"]["partial_rank_r"]),
                "same_major_branch_delta": float(tests["same_major_branch"]["mean_similarity_delta"]),
                "source_summary": str(path.resolve()),
            }
        )
    scans = pd.DataFrame(rows)
    if len(scans) != 13 or scans.target_root_id.nunique() != 7:
        raise SystemExit(f"expected 13 scans nested in 7 targets, found {len(scans)} and {scans.target_root_id.nunique()}")
    metrics = [
        "shared_path_spearman_r",
        "negative_tree_distance_spearman_r",
        "shared_path_partial_r",
        "same_major_branch_delta",
    ]
    cells = scans.groupby(["target_root_id", "target_nucleus_id"], as_index=False)[metrics].mean()
    summary = {
        "study": "functional_topology_all_eligible_scans",
        "status": "complete",
        "eligibility_rule": "all automatic_conservative target-session-scan cohorts with at least five unique connected imaged presynaptic roots",
        "n_scans": int(len(scans)),
        "n_target_cells": int(cells.target_root_id.nunique()),
        "inference": "scan metrics averaged within target before target bootstrap and paired signed-rank test",
        "metrics": {},
        "scope_boundary": "One MICrONS mouse; automatic coregistration tier; pairwise observations are not inferential replicates.",
    }
    for index, metric in enumerate(metrics):
        values = cells[metric].to_numpy(float)
        test = wilcoxon(values, alternative="two-sided") if not np.allclose(values, 0) else None
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "target_bootstrap_ci95": bootstrap(values, 260_000 + index),
            "positive_targets": int(np.sum(values > 0)),
            "negative_targets": int(np.sum(values < 0)),
            "wilcoxon_two_sided_p": float(test.pvalue) if test is not None else 1.0,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    scans.to_csv(OUT / "scan_metrics.csv", index=False, float_format="%.10g")
    cells.to_csv(OUT / "cell_metrics.csv", index=False, float_format="%.10g")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
