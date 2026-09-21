#!/usr/bin/env python3
"""Analyze the frozen intermediate alignment dose--response cohort."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    MARKERS,
    MARKER_MS,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "physical_alignment_dose_runs"
REFERENCE = ROOT / "source_data" / "nonlinear_physical_depth_confirmatory" / "seed_outcomes.csv"
OUTPUT = ROOT / "source_data" / "physical_alignment_dose"
FIGURES = ROOT / "figures" / "generated"
EXPECTED_ALPHAS = [0.25, 0.50, 0.75]
ALL_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 50_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    means = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(np.abs(means) >= observed - 1e-15))


def run_directories() -> list[Path]:
    matches = sorted(RUNS.glob("journal_physical_alignment_dose_intermediate_bp_*"))
    if not matches:
        raise FileNotFoundError("No physical alignment-dose run directory")
    return matches


def collect(allow_incomplete: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs = run_directories()
    expected = 90
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, float, int]] = set()
    duplicate_complete: list[str] = []
    run_records: list[dict[str, Any]] = []
    for run in runs:
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        if int(original["sweep_contract"]["expected_config_count"]) != expected:
            raise RuntimeError(f"Unexpected sweep size in {run}")
        run_records.append({
            "run_dir": str(run.relative_to(ROOT)),
            "manifest_sha256": sha256(run / "frozen_sweep_manifest.json"),
        })
        configs = sorted(
            (run / "configs").glob("unified_config_*.yaml"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        for config_path in configs:
            index = int(config_path.stem.rsplit("_", 1)[1])
            result = run / "results" / f"config_{index}"
            final_path = result / "performance" / "final.json"
            resource_path = result / "model_resources.json"
            if not final_path.is_file() or not resource_path.is_file():
                continue
            config = yaml.safe_load(config_path.read_text())
            dataset = config["data"]["dataset_params"]["hierarchical_gain_load"]
            factors = config["model"]["core"]["population_network"]["layers"][0]["populations"][0]["branch_factors"]
            key = (
                int(config["experiment"]["seed"]),
                float(dataset["sensor_alignment_alpha"]),
                len(factors),
            )
            if key in seen:
                duplicate_complete.append("/".join(map(str, key)))
                continue
            seen.add(key)
            final = json.loads(final_path.read_text())
            resources = json.loads(resource_path.read_text())
            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result / "train.log", result / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            rows.append({
                "cohort": "prospective_intermediate",
                "run_dir": str(run.relative_to(ROOT)),
                "config_index": index,
                "seed": key[0],
                "alignment_alpha": key[1],
                "depth": key[2],
                "branch_factors": "x".join(map(str, factors)),
                "test_accuracy": float(final["accuracy"]["test"]),
                "test_auc": float(final["auc"]["test"]),
                "trainable_parameters": int(resources["trainable_parameters"]),
                "active_synapses": int(resources["active_synapses"]),
                "candidate_synapse_slots": int(resources["candidate_synapse_slots"]),
                "persistent_state_scalars": int(resources["persistent_state_scalars_per_sample"]),
                "fallback_mentions": int(log_text.count("fallback")),
                "nonfinite_alert": bool("nan detected" in log_text or "non-finite" in log_text or "nonfinite" in log_text),
                "config_sha256": sha256(config_path),
                "final_sha256": sha256(final_path),
            })
    expected_keys = {
        (seed, alpha, depth)
        for seed in range(10200, 10210)
        for alpha in EXPECTED_ALPHAS
        for depth in (1, 2, 3)
    }
    missing = sorted(expected_keys - seen)
    if missing and not allow_incomplete:
        raise RuntimeError(f"Alignment-dose cohort incomplete: {len(missing)} missing; examples {missing[:12]}")
    frame = pd.DataFrame(rows)
    resources_ok = bool(
        len(frame)
        and all(frame[column].nunique() == 1 for column in (
            "trainable_parameters", "active_synapses", "candidate_synapse_slots", "persistent_state_scalars"
        ))
        and set(frame.trainable_parameters) == {66178}
        and set(frame.active_synapses) == {14336}
        and set(frame.candidate_synapse_slots) == {21760}
        and set(frame.persistent_state_scalars) == {2944}
    )
    audit = {
        "status": "complete" if not missing and len(frame) == expected else "incomplete",
        "expected_new_rows": expected,
        "observed_new_rows": len(frame),
        "missing_count": len(missing),
        "missing_examples": ["/".join(map(str, item)) for item in missing[:30]],
        "duplicate_complete_ignored": duplicate_complete,
        "alphas": sorted(frame.alignment_alpha.unique().tolist()) if len(frame) else [],
        "depths": sorted(frame.depth.unique().tolist()) if len(frame) else [],
        "seeds": sorted(frame.seed.unique().tolist()) if len(frame) else [],
        "finite_metrics": bool(len(frame) and np.isfinite(frame[["test_accuracy", "test_auc"]].to_numpy()).all()),
        "fallback_mentions": int(frame.fallback_mentions.sum()) if len(frame) else 0,
        "nonfinite_alert_rows": int(frame.nonfinite_alert.sum()) if len(frame) else 0,
        "resources_equal_and_frozen": resources_ok,
        "runs": run_records,
    }
    audit["all_gates_pass"] = bool(
        audit["status"] == "complete"
        and audit["alphas"] == EXPECTED_ALPHAS
        and audit["depths"] == [1, 2, 3]
        and audit["seeds"] == list(range(10200, 10210))
        and audit["finite_metrics"]
        and audit["fallback_mentions"] == 0
        and audit["nonfinite_alert_rows"] == 0
        and resources_ok
    )
    return frame, audit


def endpoints() -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE)
    keep = frame[
        frame.mechanism.eq("shunting")
        & frame.method.eq("bp")
        & frame.transport.eq("backpropagation")
        & frame.regime.isin(["zero_alignment", "aligned"])
    ].copy()
    keep["alignment_alpha"] = keep.regime.map({"zero_alignment": 0.0, "aligned": 1.0})
    keep["cohort"] = "known_endpoint"
    return keep[["cohort", "seed", "alignment_alpha", "depth", "branch_factors", "test_accuracy", "test_auc"]]


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([
        endpoints(),
        frame[["cohort", "seed", "alignment_alpha", "depth", "branch_factors", "test_accuracy", "test_auc"]],
    ], ignore_index=True)
    summary_rows = []
    for index, ((alpha, depth), part) in enumerate(combined.groupby(["alignment_alpha", "depth"], sort=True)):
        mean, low, high = bootstrap_mean(part.test_accuracy.to_numpy(), 8_700_000 + index)
        summary_rows.append({
            "alignment_alpha": alpha, "depth": depth, "n_seeds": part.seed.nunique(),
            "mean_test_accuracy": mean, "ci95_low_test_accuracy": low, "ci95_high_test_accuracy": high,
        })
    pivot = combined.pivot(index=["seed", "alignment_alpha"], columns="depth", values="test_accuracy").reset_index()
    pivot["depth_effect_d3_minus_d1"] = pivot[3] - pivot[1]
    effect_rows = []
    for index, (alpha, part) in enumerate(pivot.groupby("alignment_alpha", sort=True)):
        values = part.depth_effect_d3_minus_d1.to_numpy()
        mean, low, high = bootstrap_mean(values, 8_710_000 + index)
        effect_rows.append({
            "estimand": f"depth_effect_alpha_{alpha:.2f}", "alignment_alpha": alpha,
            "n_pairs": len(values), "mean_difference": mean, "ci95_low": low, "ci95_high": high,
            "positive_pairs": int(np.sum(values > 0)), "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
        })
    effect_frame = pd.DataFrame(effect_rows)
    effect_pivot = pivot.pivot(index="seed", columns="alignment_alpha", values="depth_effect_d3_minus_d1")
    contrast_rows = []
    for index, (left, right) in enumerate(zip(ALL_ALPHAS[1:], ALL_ALPHAS[:-1])):
        values = (effect_pivot[left] - effect_pivot[right]).to_numpy()
        mean, low, high = bootstrap_mean(values, 8_720_000 + index)
        contrast_rows.append({
            "estimand": f"adjacent_alpha_{left:.2f}_minus_{right:.2f}", "mean_difference": mean,
            "ci95_low": low, "ci95_high": high, "positive_pairs": int(np.sum(values > 0)),
            "n_pairs": len(values), "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
        })
    values = (effect_pivot[0.75] - effect_pivot[0.25]).to_numpy()
    mean, low, high = bootstrap_mean(values, 8_720_010)
    contrast_rows.append({
        "estimand": "alpha_0.75_minus_0.25", "mean_difference": mean,
        "ci95_low": low, "ci95_high": high, "positive_pairs": int(np.sum(values > 0)),
        "n_pairs": len(values), "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
    })
    x = np.asarray(ALL_ALPHAS)
    slopes = np.asarray([np.polyfit(x, effect_pivot.loc[seed, ALL_ALPHAS].to_numpy(), 1)[0] for seed in effect_pivot.index])
    mean, low, high = bootstrap_mean(slopes, 8_720_011)
    contrast_rows.append({
        "estimand": "within_seed_linear_slope_per_unit_alpha", "mean_difference": mean,
        "ci95_low": low, "ci95_high": high, "positive_pairs": int(np.sum(slopes > 0)),
        "n_pairs": len(slopes), "exact_sign_flip_p_two_sided": exact_sign_flip_p(slopes),
    })
    return combined, pd.DataFrame(summary_rows), pd.concat([effect_frame, pd.DataFrame(contrast_rows)], ignore_index=True, sort=False)


def render(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    apply_neurips_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.50), gridspec_kw={
        "left": 0.115, "right": 0.985, "bottom": 0.16, "top": 0.82, "wspace": 0.62,
    })
    # D2 wears a lighter tint of the depth-family green (violet is
    # reserved figure-wide for the grouped-star family).
    colors = {1: COLORS["mute"], 2: "#6FB58F", 3: COLORS["shunting"]}
    label_y = {}
    for depth, marker in zip((1, 2, 3), MARKERS):
        part = summary[summary.depth.eq(depth)].sort_values("alignment_alpha")
        mean = part.mean_test_accuracy.to_numpy()
        low = part.ci95_low_test_accuracy.to_numpy()
        high = part.ci95_high_test_accuracy.to_numpy()
        axes[0].errorbar(part.alignment_alpha, mean, yerr=np.vstack([mean-low, high-mean]),
                         color=colors[depth], marker=marker, ms=MARKER_MS, lw=LW_DATA,
                         markeredgecolor="white", markeredgewidth=0.5,
                         elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
        label_y[depth] = float(mean[-1])
    # Three series: direct labels at the curve ends instead of a legend.
    label_y[1] -= 0.021  # D1/D2 ends nearly coincide; nudge apart
    label_y[2] += 0.021
    for depth in (1, 2, 3):
        axes[0].text(
            1.06, label_y[depth], f"D{depth}", ha="left", va="center",
            fontsize=PT_LEGEND, color=colors[depth],
        )
    axes[0].set_xlim(-0.07, 1.23)
    axes[0].set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[0].set_ylim(0.44, 1.06)
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_xlabel("sensor alignment $\\alpha$")
    axes[0].set_ylabel("test accuracy")
    panel_title(axes[0], "M", "Alignment dose")
    style_axis(axes[0], grid="y")

    effect = contrasts[contrasts.alignment_alpha.notna()].sort_values("alignment_alpha")
    mean = 100 * effect.mean_difference.to_numpy()
    low = 100 * effect.ci95_low.to_numpy()
    high = 100 * effect.ci95_high.to_numpy()
    axes[1].axhline(0, color=COLORS["mute"], lw=LW_HAIR, zorder=0)
    axes[1].errorbar(effect.alignment_alpha, mean, yerr=np.vstack([mean-low, high-mean]),
                     color=COLORS["shunting"], marker="o", ms=MARKER_MS, lw=LW_DATA,
                     markeredgecolor="white", markeredgewidth=0.5,
                     elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
    axes[1].set_xlim(-0.07, 1.07)
    axes[1].set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[1].set_xlabel("sensor alignment $\\alpha$")
    axes[1].set_ylabel("D3 − D1 (pp)")
    panel_title(axes[1], "N", "Depth benefit")
    style_axis(axes[1], grid="y")

    names = ["alpha_0.75_minus_0.25", "within_seed_linear_slope_per_unit_alpha"]
    rows = contrasts.set_index("estimand").loc[names]
    y = np.array([1, 0])
    mean = 100 * rows.mean_difference.to_numpy()
    low = 100 * rows.ci95_low.to_numpy()
    high = 100 * rows.ci95_high.to_numpy()
    axes[2].axvline(0, color=COLORS["mute"], lw=LW_HAIR)
    axes[2].errorbar(mean, y, xerr=np.vstack([mean-low, high-mean]), fmt="o",
                     color=COLORS["shunting"], ecolor=COLORS["shunting"],
                     markerfacecolor=COLORS["shunting"], markeredgecolor="white",
                     markeredgewidth=0.5, ms=MARKER_MS,
                     elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
    axes[2].set_xlim(-2.5, 30.5)
    axes[2].set_xticks([0, 10, 20, 30])
    axes[2].set_ylim(-0.6, 1.6)
    axes[2].set_yticks(y, [r"$\alpha$ 0.75 − 0.25", "linear slope"])
    axes[2].set_xlabel("depth-benefit change (pp)")
    panel_title(axes[2], "O", "Dose contrasts")
    style_axis(axes[2], grid="x")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_physical_alignment_dose")
    audit_text_over_data(fig, "fig_physical_alignment_dose")
    fig.savefig(
        FIGURES / "fig_physical_alignment_dose.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_physical_alignment_dose.png", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not RUNS.exists():
        # Frozen-source render: raw run directories are not in this checkout;
        # restyle the figure from the audited source-data summaries.
        summary = pd.read_csv(OUTPUT / "condition_summary.csv")
        contrasts = pd.read_csv(OUTPUT / "paired_contrasts.csv")
        render(summary, contrasts)
        print("Rendered fig_physical_alignment_dose from frozen source data "
              "(run directories absent; collection skipped).")
        return
    new, audit = collect(args.allow_incomplete)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTPUT / "new_seed_outcomes.csv", index=False)
    if audit["all_gates_pass"]:
        combined, summary, contrasts = analyze(new)
        combined.to_csv(OUTPUT / "combined_seed_outcomes.csv", index=False)
        summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
        contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
        render(summary, contrasts)
        slope = contrasts.set_index("estimand").loc["within_seed_linear_slope_per_unit_alpha"]
        report = (
            "# Physical-depth alignment dose response\n\n"
            f"Status: complete; all {audit['observed_new_rows']} new fits passed the gates.\n\n"
            f"The within-seed slope of D3-minus-D1 accuracy was {100*slope.mean_difference:.2f} "
            f"percentage points per unit alpha ({100*slope.ci95_low:.2f} to {100*slope.ci95_high:.2f}; "
            f"{int(slope.positive_pairs)}/10 positive seeds).\n"
        )
    else:
        report = f"# Physical-depth alignment dose response\n\nStatus: incomplete ({audit['observed_new_rows']}/{audit['expected_new_rows']}).\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
