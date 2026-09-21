#!/usr/bin/env python3
"""Exclude legacy signed-input shunting runs from publication-facing analyses.

The synthetic noise task produces signed transfer values.  Historical runs
used ``output_activation: null`` for both additive and positive-conductance
shunting cores.  The additive model accepts signed values; the shunting model
does not.  This audit preserves every historical row, verifies its resolved
configuration, and exports only validity-qualified rows for current figures
and claims.  No outcome value enters the exclusion rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_prospective_mechanism_diagnostics import (
    association_summary,
    feedback_summary,
    paired_feedback_contrasts,
    summarize,
)
from analyze_prospective_followup_results import plot_fixed_budget


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source_data" / "prospective_input_validity"


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    """Return BH-adjusted values for the tests that remain publication-valid."""

    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def is_invalid_signed_shunting(row: pd.Series) -> bool:
    return (
        row["task"] == "noise_resilience"
        and row["core"] == "dendritic_shunting"
    )


def resolved_config(row: pd.Series) -> dict:
    path = ROOT / str(row["run_dir"]) / "results" / f"config_{int(row['config_index'])}" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def annotate(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        config = resolved_config(row)
        activation = config["model"]["core"]["transfer"].get("output_activation")
        invalid = is_invalid_signed_shunting(row)
        if invalid and activation is not None:
            raise RuntimeError(
                f"Expected the historical invalid convention to use null activation: {row['run_dir']} config {row['config_index']}"
            )
        rows.append(
            {
                "cohort": cohort,
                "run_dir": row["run_dir"],
                "config_index": int(row["config_index"]),
                "family": row.get("family", "primary"),
                "task": row["task"],
                "core": row["core"],
                "strategy": row["strategy"],
                "feedback": row["feedback"],
                "depth": int(row["depth"]),
                "seed": int(row["seed"]),
                "transfer_output_activation": activation,
                "input_convention_valid": not invalid,
                "publication_included": (not invalid)
                and not (cohort == "followup" and row.get("family") == "inhibition"),
                "exclusion_reason": (
                    "signed synthetic transfer output is not a valid positive-conductance shunting drive"
                    if invalid
                    else (
                        "inhibitory-dose family excluded because its prespecified cross-core contrast depends on invalid signed-shunting cells"
                        if cohort == "followup" and row.get("family") == "inhibition"
                        else ""
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def central_tables(frame: pd.DataFrame) -> None:
    valid = frame[
        ~(
            frame.task.eq("noise_resilience")
            & frame.core.eq("dendritic_shunting")
        )
    ].copy()
    valid.to_csv(OUT / "central_valid_seed_outcomes.csv", index=False)
    condition = pd.read_csv(ROOT / "source_data/prospective_learning/condition_summary.csv")
    condition = condition[
        ~(
            condition.task.eq("noise_resilience")
            & condition.core.eq("dendritic_shunting")
        )
    ]
    condition.to_csv(OUT / "central_valid_condition_summary.csv", index=False)
    contrast = pd.read_csv(ROOT / "source_data/prospective_learning/paired_contrasts.csv")
    contrast = contrast[
        ~(
            contrast.task.eq("noise_resilience")
            & contrast.core.eq("dendritic_shunting")
        )
    ]
    # Cross-core rows involving the excluded core cannot be publication claims.
    contrast = contrast[~contrast.core.astype(str).str.contains("shunting - additive")]
    contrast.to_csv(OUT / "central_valid_paired_contrasts.csv", index=False)


def followup_tables(frame: pd.DataFrame) -> None:
    valid = frame[
        ~(
            frame.task.eq("noise_resilience")
            & frame.core.eq("dendritic_shunting")
        )
    ].copy()
    publication = valid[~valid.family.eq("inhibition")].copy()
    publication.to_csv(OUT / "followup_publication_seed_outcomes.csv", index=False)

    condition = pd.read_csv(ROOT / "source_data/prospective_followup/condition_summary.csv")
    condition = condition[
        ~(
            condition.task.eq("noise_resilience")
            & condition.core.eq("dendritic_shunting")
        )
        & ~condition.family.eq("inhibition")
    ]
    condition.to_csv(OUT / "followup_publication_condition_summary.csv", index=False)

    contrast = pd.read_csv(ROOT / "source_data/prospective_followup/paired_contrasts.csv")
    invalid_direct = contrast.task.eq("noise_resilience") & contrast.core.eq(
        "dendritic_shunting"
    )
    cross_core = contrast.core.astype(str).str.contains("shunting - additive")
    contrast = contrast[
        ~invalid_direct & ~cross_core & ~contrast.study.eq("inhibition")
    ]
    contrast.to_csv(OUT / "followup_publication_paired_contrasts.csv", index=False)

    routing = pd.read_csv(
        ROOT / "source_data/prospective_routing_control/paired_contrasts.csv"
    )
    routing = routing[
        ~(
            routing.task.eq("noise_resilience")
            & routing.core.eq("dendritic_shunting")
        )
    ].copy()
    routing = routing.drop(columns=["fdr_bh_across_eight"], errors="ignore")
    routing["fdr_bh_across_valid_tests"] = benjamini_hochberg(
        routing["wilcoxon_p_two_sided"]
    )
    routing.to_csv(OUT / "routing_valid_paired_contrasts.csv", index=False)

    spatial = publication[publication.family.eq("spatial")].copy()
    wide = spatial.pivot(
        index=["task", "core", "strategy", "feedback", "seed"],
        columns="topology",
        values="test_accuracy",
    ).reset_index()
    wide["difference"] = wide["spatial"] - wide["random"]
    seed_mean = (
        wide.groupby(["task", "strategy", "feedback", "seed"], as_index=False)
        .difference.mean()
    )
    rows: list[dict[str, object]] = []
    for key, group in seed_mean.groupby(["task", "strategy", "feedback"], sort=True):
        values = group.difference.to_numpy(float)
        rng = np.random.default_rng(71_000 + len(rows))
        means = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
        low, high = np.quantile(means, [0.025, 0.975])
        rows.append(
            {
                "task": key[0],
                "strategy": key[1],
                "feedback": key[2],
                "n_paired_seeds": len(values),
                "cores_averaged_within_seed": int(
                    spatial[spatial.task.eq(key[0])].core.nunique()
                ),
                "mean_difference": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "spatial_valid_task_feedback_effects.csv", index=False)


def mechanism_tables() -> None:
    frame = pd.read_csv(
        ROOT / "source_data/prospective_learning/mechanism_checkpoint_rows.csv"
    )
    valid = frame[
        ~(
            frame.dataset.eq("noise_resilience")
            & frame.network_type.eq("dendritic_shunting")
        )
    ].copy()
    checkpoints = valid[["dataset", "network_type", "depth", "seed"]].drop_duplicates()
    if len(valid) != 2_400 or len(checkpoints) != 120:
        raise RuntimeError("Unexpected validity-qualified mechanism accounting")
    valid.to_csv(OUT / "mechanism_checkpoint_rows_valid.csv", index=False)
    summarize(valid).to_csv(OUT / "mechanism_checkpoint_summary_valid.csv", index=False)
    paired_feedback_contrasts(valid).to_csv(
        OUT / "mechanism_paired_contrasts_valid.csv", index=False
    )
    association_summary(valid).to_csv(
        OUT / "mechanism_association_summary_valid.csv", index=False
    )
    feedback_summary(valid).to_csv(
        OUT / "mechanism_feedback_summary_valid.csv", index=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    central = pd.read_csv(ROOT / "source_data/prospective_learning/seed_outcomes.csv")
    followup = pd.read_csv(ROOT / "source_data/prospective_followup/seed_outcomes.csv")
    audit = pd.concat(
        [annotate(central, "primary"), annotate(followup, "followup")],
        ignore_index=True,
    )
    if len(audit) != 1_840 or int((~audit.input_convention_valid).sum()) != 640:
        raise RuntimeError("Unexpected prospective validity accounting")
    if int(audit.publication_included.sum()) != 1_000:
        raise RuntimeError("Unexpected publication-included accounting")
    audit.to_csv(OUT / "historical_run_validity.csv", index=False)
    central_tables(central)
    followup_tables(followup)
    mechanism_tables()
    summary = {
        "rule": "exclude noise_resilience plus dendritic_shunting when the resolved transfer output activation is null",
        "rule_uses_outcomes": False,
        "historical_runs_audited": 1840,
        "invalid_signed_shunting_runs": 640,
        "valid_primary_runs": 480,
        "valid_publication_followup_runs": 520,
        "excluded_inhibitory_dose_family_runs": 400,
        "scope": "The exclusion concerns an artificial signed-input positive-conductance convention; MNIST shunting, additive synthetic models, active-compartment equilibria and reconstructed-cell perturbations are unaffected.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_fixed_budget(
        pd.read_csv(OUT / "followup_publication_condition_summary.csv"),
        pd.read_csv(OUT / "followup_publication_paired_contrasts.csv"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
