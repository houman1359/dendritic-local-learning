#!/usr/bin/env python3
"""Interior-optimum reanalysis of the credit-operator bound.

Deterministic secondary analysis; no training outcomes are altered or rerun.

The nested-projection ladder (scalar < neuron < ancestry-K < exact) makes the
bound utility U(M) non-monotone whenever admitted noise grows faster than
retained signal.  This script asks whether the *location* of the utility
optimum predicts the location of the observed learning optimum in the two
frozen experiments where bandwidth was varied:

1. Depth phase (50 seeds x task depth H x model depth D, aligned tree):
   predicted D* = argmax_D initial maximum guaranteed decrease, observed
   D* = argmin_D final population loss, per (seed, H) pair.
2. 2,700-fit subtree factorial (20 seeds x K in {1,2,4,8}, dendritic tree):
   the ancestry-minus-best-non-anatomical-control contrast in U versus the
   same contrast in held-out accuracy, per seed; controls are the four
   matched non-anatomical families of the published Fig. 4c comparison.

Before any new quantity is reported, the script reproduces the published
K=4 accuracy contrast (0.0127, 95% CI 0.0059-0.0197, 15/20 seeds) from the
frozen seed outcomes as a pipeline check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEPTH_SEED = ROOT / "source_data" / "credit_phase_theory" / "depth_training_seed.csv"
OPERATOR = ROOT / "source_data" / "credit_phase_existing" / "operator_metrics.csv"
OUTCOMES = (
    ROOT / "source_data" / "trained_subtree_address_full_factorial" / "seed_outcomes.csv"
)
OUTPUT = ROOT / "source_data" / "interior_optimum"

CONTROL_FAMILIES = (
    "depth_interleaved_bins",
    "learned_rank_k_upper_bound",
    "random_rank_k",
    "random_sparse_matched",
)
NON_ORACLE_CONTROLS = (
    "depth_interleaved_bins",
    "random_rank_k",
    "random_sparse_matched",
)
ANCESTRY = "correct_ancestry_subtrees"
BUDGETS = (1, 2, 4, 8)
RNG_SEED = 20260812
N_BOOT = 20000


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def depth_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(DEPTH_SEED)
    df = df[df["method"] == "aligned_tree"].copy()
    df["task_depth"] = df["task_depth"].astype(int)
    df["model_depth"] = df["model_depth"].astype(int)

    records = []
    for (seed, task_depth), grp in df.groupby(["seed", "task_depth"]):
        grp = grp.sort_values("model_depth")
        observed = int(grp.loc[grp["final_population_loss"].idxmin(), "model_depth"])
        predicted = int(
            grp.loc[grp["initial_maximum_guaranteed_decrease"].idxmax(), "model_depth"]
        )
        records.append(
            {
                "seed": seed,
                "task_depth": task_depth,
                "observed_best_depth": observed,
                "predicted_best_depth": predicted,
                "agree": observed == predicted,
                "observed_is_matched": observed == task_depth,
                "predicted_is_matched": predicted == task_depth,
            }
        )
    agreement = pd.DataFrame.from_records(records)

    curves = (
        df.groupby(["task_depth", "model_depth"])
        .agg(
            mean_final_population_loss=("final_population_loss", "mean"),
            mean_utility=("initial_maximum_guaranteed_decrease", "mean"),
            n=("seed", "size"),
        )
        .reset_index()
    )

    total = len(agreement)
    summary = {
        "n_pairs": int(total),
        "agreement_fraction": float(agreement["agree"].mean()),
        "agreement_count": int(agreement["agree"].sum()),
        "observed_matched_fraction": float(agreement["observed_is_matched"].mean()),
        "predicted_matched_fraction": float(agreement["predicted_is_matched"].mean()),
        "group_level_argmax_match": {},
    }
    for task_depth, grp in curves.groupby("task_depth"):
        obs = int(grp.loc[grp["mean_final_population_loss"].idxmin(), "model_depth"])
        pred = int(grp.loc[grp["mean_utility"].idxmax(), "model_depth"])
        summary["group_level_argmax_match"][int(task_depth)] = {
            "observed_best_depth": obs,
            "predicted_best_depth": pred,
        }
    return agreement, curves, summary


def factorial_analysis(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    outcomes = pd.read_csv(OUTCOMES)
    operator = pd.read_csv(OPERATOR)
    outcomes = outcomes[outcomes["architecture"] == "dendritic_tree"]
    operator = operator[operator["architecture"] == "dendritic_tree"]

    def contrast_table(
        df: pd.DataFrame, value: str, controls: tuple[str, ...]
    ) -> pd.DataFrame:
        pivot = df.pivot_table(
            index=["seed", "budget_k"], columns="feedback_family", values=value
        )
        best_control = pivot[list(controls)].max(axis=1)
        out = pd.DataFrame(
            {
                "ancestry": pivot[ANCESTRY],
                "best_control": best_control,
                "contrast": pivot[ANCESTRY] - best_control,
            }
        ).reset_index()
        return out

    acc = contrast_table(outcomes, "heldout_accuracy", CONTROL_FAMILIES).rename(
        columns={"contrast": "accuracy_contrast"}
    )
    util = contrast_table(
        operator, "maximum_guaranteed_decrease", CONTROL_FAMILIES
    ).rename(columns={"contrast": "utility_contrast"})
    util_no = contrast_table(
        operator, "maximum_guaranteed_decrease", NON_ORACLE_CONTROLS
    ).rename(columns={"contrast": "utility_contrast_non_oracle"})
    acc_no = contrast_table(
        outcomes, "heldout_accuracy", NON_ORACLE_CONTROLS
    ).rename(columns={"contrast": "accuracy_contrast_non_oracle"})
    merged = (
        acc[["seed", "budget_k", "accuracy_contrast"]]
        .merge(
            util[["seed", "budget_k", "utility_contrast"]], on=["seed", "budget_k"]
        )
        .merge(
            util_no[["seed", "budget_k", "utility_contrast_non_oracle"]],
            on=["seed", "budget_k"],
        )
        .merge(
            acc_no[["seed", "budget_k", "accuracy_contrast_non_oracle"]],
            on=["seed", "budget_k"],
        )
    )

    # Pipeline check: reproduce the published K=4 accuracy contrast.
    at4 = merged[merged["budget_k"] == 4]["accuracy_contrast"].to_numpy()
    lo, hi = bootstrap_ci(at4, rng)
    check = {
        "published": {"mean": 0.0127, "ci_low": 0.0059, "ci_high": 0.0197, "wins": 15},
        "reproduced": {
            "mean": float(at4.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "wins": int((at4 > 0).sum()),
            "n": int(len(at4)),
        },
    }
    check["passes"] = bool(
        abs(at4.mean() - 0.0127) < 5e-4 and int((at4 > 0).sum()) == 15
    )

    per_k = []
    for k, grp in merged.groupby("budget_k"):
        a = grp["accuracy_contrast"].to_numpy()
        u = grp["utility_contrast"].to_numpy()
        uno = grp["utility_contrast_non_oracle"].to_numpy()
        alo, ahi = bootstrap_ci(a, rng)
        ulo, uhi = bootstrap_ci(u, rng)
        unolo, unohi = bootstrap_ci(uno, rng)
        per_k.append(
            {
                "budget_k": int(k),
                "mean_accuracy_contrast": float(a.mean()),
                "accuracy_ci_low": alo,
                "accuracy_ci_high": ahi,
                "accuracy_positive_seeds": int((a > 0).sum()),
                "mean_utility_contrast": float(u.mean()),
                "utility_ci_low": ulo,
                "utility_ci_high": uhi,
                "utility_positive_seeds": int((u > 0).sum()),
                "mean_utility_contrast_non_oracle": float(uno.mean()),
                "utility_non_oracle_ci_low": unolo,
                "utility_non_oracle_ci_high": unohi,
                "utility_non_oracle_positive_seeds": int((uno > 0).sum()),
                "n_seeds": int(len(a)),
            }
        )
    per_k = pd.DataFrame(per_k)

    per_seed = []
    for seed, grp in merged.groupby("seed"):
        grp = grp.sort_values("budget_k")
        k_obs = int(grp.loc[grp["accuracy_contrast"].idxmax(), "budget_k"])
        k_pred = int(grp.loc[grp["utility_contrast"].idxmax(), "budget_k"])
        per_seed.append(
            {
                "seed": seed,
                "observed_peak_k": k_obs,
                "predicted_peak_k": k_pred,
                "agree": k_obs == k_pred,
            }
        )
    per_seed = pd.DataFrame(per_seed)

    group_peak = {
        "observed_peak_k": int(
            per_k.loc[per_k["mean_accuracy_contrast"].idxmax(), "budget_k"]
        ),
        "predicted_peak_k": int(
            per_k.loc[per_k["mean_utility_contrast"].idxmax(), "budget_k"]
        ),
    }
    summary = {
        "pipeline_check": check,
        "group_peak": group_peak,
        "per_seed_agreement_fraction": float(per_seed["agree"].mean()),
        "per_seed_agreement_count": int(per_seed["agree"].sum()),
        "n_seeds": int(len(per_seed)),
        "control_families": list(CONTROL_FAMILIES),
    }
    return merged, per_k, {"summary": summary, "per_seed": per_seed}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    depth_agreement, depth_curves, depth_summary = depth_analysis()
    merged, per_k, factorial = factorial_analysis(rng)

    depth_agreement.to_csv(OUTPUT / "depth_agreement_seed.csv", index=False)
    depth_curves.to_csv(OUTPUT / "depth_curves.csv", index=False)
    merged.to_csv(OUTPUT / "factorial_contrast_seed.csv", index=False)
    per_k.to_csv(OUTPUT / "factorial_contrast_summary.csv", index=False)
    factorial["per_seed"].to_csv(OUTPUT / "factorial_peak_seed.csv", index=False)

    summary = {
        "depth": depth_summary,
        "factorial": factorial["summary"],
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOT,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Interior-optimum reanalysis",
        "",
        "Deterministic secondary analysis of the frozen depth-phase and",
        "subtree-factorial outputs; see script docstring for definitions.",
        "",
        "## Depth phase (aligned tree)",
        f"- seed x task-depth pairs: {depth_summary['n_pairs']}",
        f"- predicted D* (argmax utility) equals observed D* (argmin loss): "
        f"{depth_summary['agreement_count']}/{depth_summary['n_pairs']} "
        f"({depth_summary['agreement_fraction']:.3f})",
        f"- observed D* = H fraction: {depth_summary['observed_matched_fraction']:.3f}",
        f"- predicted D* = H fraction: {depth_summary['predicted_matched_fraction']:.3f}",
        f"- group-level argmax match per H: {depth_summary['group_level_argmax_match']}",
        "",
        "## Factorial ancestry-minus-best-control contrast",
        f"- pipeline check vs published K=4 contrast: {factorial['summary']['pipeline_check']}",
        f"- group-level peak: {factorial['summary']['group_peak']}",
        f"- per-seed peak agreement: {factorial['summary']['per_seed_agreement_count']}"
        f"/{factorial['summary']['n_seeds']}",
        "",
        "Per-K means:",
        per_k.to_string(index=False),
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
