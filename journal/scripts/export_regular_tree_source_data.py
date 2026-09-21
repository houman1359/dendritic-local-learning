#!/usr/bin/env python3
"""Export validated regular-tree source tables into the journal package.

The source tables are the ones used by the arXiv/NeurIPS figure generators.
Inputs are hash-checked before export.  Machine-local result paths are replaced
by stable checkpoint or run identifiers so the journal Source Data contains no
private filesystem names.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "neurips"

SOURCES = {
    "figures/data/feedback_learning_relevance_runs.csv": (
        "565fa75b5f9210717ae54155e77a498ea6fb83608aef5f44bc4846e1676e99bf",
        "source_data/figure2/feedback_learning_relevance_runs.csv",
        "feedback",
    ),
    "figures/data/feedback_learning_relevance_summary.csv": (
        "03ce98320856fa7191a7459516f6d9b7f421bb5c6fc9b5af691e0c0c7adec795",
        "source_data/figure2/feedback_learning_relevance_summary.csv",
        "copy",
    ),
    "figures/data/competence_summary_20260422.csv": (
        "910f02452fa106238696940882e27987f2951d75fca5190ebacaffe02ed268a0",
        "source_data/regular_tree_regimes/competence_summary.csv",
        "copy",
    ),
    "figures/data/gradient_fidelity_vs_ie_summary.csv": (
        "5901c5187d1c027f2a1e3f266d6e5d87e2c60f77c7e772f8543122e84bef95fe",
        "source_data/regular_tree_regimes/inhibition_dose_summary.csv",
        "copy",
    ),
    "analysis/depth_scaling.csv": (
        "8abd538efd42ba314ca87f3c1df4bcc5c6cb0cf8356ed6f96095fb5915410113",
        "source_data/regular_tree_regimes/depth_scaling_summary.csv",
        "copy",
    ),
    "analysis/noise_robustness.csv": (
        "c158876c0641478c89aa4bbad91ff10e847ae20701438ead995cf41d7d403add",
        "source_data/regular_tree_regimes/broadcast_noise_summary.csv",
        "copy",
    ),
    "analysis/core_fair_tuning.csv": (
        "e963b222fff43e7921f1be8d75196cf5f306576fc2afcb5667e0a7520eccbbdf",
        "source_data/regular_tree_regimes/rule_family_summary_source.csv",
        "copy",
    ),
    "figures/data/local_mismatch_recheck_runs.csv": (
        "3e2911aeb66ac6811932afba66ee33311a6854d13a1b1f3c269486cc2287c908",
        "source_data/regular_tree_regimes/error_source_runs.csv",
        "copy",
    ),
    "figures/data/revision_exact_transport_factorial_grouped.csv": (
        "15092918e7cabe98d3eabff4d90b2dd3f3004331fc237e0da0aef732aa2a369b",
        "source_data/regular_tree_regimes/exact_transport_summary.csv",
        "copy",
    ),
    "figures/data/revision_exact_transport_bp_grouped.csv": (
        "b4419337a67c9aa438418c2b15d8eb1d3bd056dd7d4043cf7fecf9e16cf2774f",
        "source_data/regular_tree_regimes/backprop_reference_summary.csv",
        "copy",
    ),
    "figures/data/revision_additive_gain_norm_grouped.csv": (
        "522ba4f6d672a021ea3ce14464673887a65cc005b9b4d4bb118aab68634e41f7",
        "source_data/regular_tree_regimes/additive_controls_summary.csv",
        "copy",
    ),
    "figures/data/revision_reactivation_identity_grouped.csv": (
        "8685717cc082127721271525c8c1b3d92c14344968d610879c54cde635750f2b",
        "source_data/regular_tree_regimes/reactivation_controls_summary.csv",
        "copy",
    ),
    "figures/data/noise_resilience_rank_bridge_summary.csv": (
        "4816a217300a02e44123c887232aaa173365cfc9746693c9bac1534cb6c3ccbd",
        "source_data/regular_tree_regimes/noise_feedback_ladder_summary.csv",
        "copy",
    ),
    "figures/data/cifar10_control_ladder_detailed_results.csv": (
        "d107a05d5566f7b8408c4b9b95a1e7351a7d90247aa163cdd2a20a84ddaab7ba",
        "source_data/regular_tree_regimes/cifar10_control_ladder_runs.csv",
        "cifar",
    ),
    "configs/sweeps/sweep_neurips_localca_core_fair_tuning.yaml": (
        "5db5a5111dfcb43ec5b6713f3dff255f544bfac505486f40e78ccb3542fa5e9b",
        "configs/regular_tree/arxiv_regimes/core_fair_tuning.yaml",
        "config",
    ),
    "configs/sweeps/sweep_neurips_local_mismatch_recheck.yaml": (
        "7dc89b00ddad8873c8921aa1bafff691864238baf38c129fd3f03d31c8f82d4c",
        "configs/regular_tree/arxiv_regimes/local_mismatch_recheck.yaml",
        "config",
    ),
    "configs/sweeps/sweep_neurips_noise_resilience_rank_bridge_activation_corrected.yaml": (
        "fdb3189b8de5b1d785258f42d8e5c575d740e1dbdcc5eb29217b44ee1f8dd3ed",
        "configs/regular_tree/arxiv_regimes/noise_feedback_rank_bridge.yaml",
        "config",
    ),
    "configs/sweeps/sweep_neurips_cifar10_control_ladder_20260427.yaml": (
        "95b85bf4b973bb78b2122be4672511ee05dfd11b81ae32b6f7c336c9701377a1",
        "configs/regular_tree/arxiv_regimes/cifar10_control_ladder.yaml",
        "config",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_feedback(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    if "run_dir" not in frame:
        raise ValueError(f"feedback source lacks run_dir: {source}")
    stable = frame["run_dir"].map(lambda value: Path(str(value)).name)
    frame.insert(
        0,
        "checkpoint_id",
        frame["network_type"].str.replace("dendritic_", "", regex=False) + ":" + stable,
    )
    frame = frame.drop(columns=["run_dir"])
    frame.to_csv(destination, index=False)


def export_cifar(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    frame = frame.drop(columns=["result_dir"], errors="ignore")
    frame.to_csv(destination, index=False)


def export_config(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text, count = re.subn(
        r'^output_dir:\s*["\'].*?["\']\s*$',
        'output_dir: "./results"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"expected one output_dir line in {source}")
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    for relative, (expected, output, mode) in SOURCES.items():
        source = args.source / relative
        destination = ROOT / output
        observed = sha256(source)
        if observed != expected:
            raise ValueError(
                f"source hash changed for {relative}: expected {expected}, observed {observed}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "feedback":
            export_feedback(source, destination)
        elif mode == "cifar":
            export_cifar(source, destination)
        elif mode == "config":
            export_config(source, destination)
        else:
            shutil.copyfile(source, destination)
        print(f"exported {relative} -> {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
