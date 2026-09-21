#!/usr/bin/env python3
"""Audit and analyze the fixed-depth task-family by alignment factorial."""

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
    MARKER_MS,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260820/sweep_runs/task_family_alignment"
)
OUTPUT = ROOT / "source_data" / "task_family_alignment"
FIGURES = ROOT / "figures" / "generated"
FAMILIES = ("nested_factor", "flat_factor", "local_ratio")
ARCHITECTURES = ("serial", "grouped_point")
CREDITS = ("bp", "local3f")
ALPHAS = (0.0, 0.5, 1.0)
SEEDS = tuple(range(10500, 10510))
RESOURCE_COLUMNS = (
    "trainable_parameters",
    "active_synapses",
    "candidate_synapse_slots",
    "persistent_state_scalars",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(
    values: np.ndarray, seed: int, draws: int = 50_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    means = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    return float(np.mean(np.abs(means) >= observed - 1e-15))


def run_stem(family: str, architecture: str, credit: str) -> str:
    return f"journal_taskfamily_{family}_{architecture}_{credit}"


def latest_run(runs: Path, stem: str) -> Path:
    matches = sorted(runs.glob(f"{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory matching {stem}_* below {runs}")
    return matches[-1]


def _resources(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "trainable_parameters": int(payload["trainable_parameters"]),
        "active_synapses": int(payload["active_synapses"]),
        "candidate_synapse_slots": int(payload["candidate_synapse_slots"]),
        "persistent_state_scalars": int(
            payload["persistent_state_scalars_per_sample"]
        ),
    }


def collect(
    runs: Path, allow_incomplete: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    contract_failures: list[str] = []
    seen: set[tuple[str, str, str, float, int]] = set()
    for family in FAMILIES:
        for architecture in ARCHITECTURES:
            for credit in CREDITS:
                stem = run_stem(family, architecture, credit)
                try:
                    run = latest_run(runs, stem)
                except FileNotFoundError:
                    if not allow_incomplete:
                        raise
                    missing.extend(f"{stem}/config_{index}" for index in range(30))
                    continue
                results = run / "results"
                run_records.append({"stem": stem, "run_dir": str(run)})
                for index in range(30):
                    result = results / f"config_{index}"
                    config_path = result / "config.json"
                    final_path = result / "performance" / "final.json"
                    resource_path = result / "model_resources.json"
                    if not all(
                        path.is_file()
                        for path in (config_path, final_path, resource_path)
                    ):
                        missing.append(f"{stem}/config_{index}")
                        continue
                    config = json.loads(config_path.read_text())
                    final = json.loads(final_path.read_text())
                    resources = _resources(json.loads(resource_path.read_text()))
                    dataset = config["data"]["dataset_params"][
                        "hierarchical_gain_load"
                    ]
                    population = config["model"]["core"]["population_network"][
                        "layers"
                    ][0]["populations"][0]
                    factors = list(population["branch_factors"])
                    alpha = float(dataset["sensor_alignment_alpha"])
                    seed = int(config["experiment"]["seed"])
                    key = (family, architecture, credit, alpha, seed)
                    if key in seen:
                        contract_failures.append(f"duplicate/{key}")
                        continue
                    seen.add(key)
                    if factors != [2, 1, 2]:
                        contract_failures.append(f"{stem}/{index}/depth")
                    observed_mode = population.get("population", {}).get(
                        "cross_level_mode", "serial"
                    )
                    if architecture == "grouped_point" and observed_mode != "parallel_readout":
                        contract_failures.append(f"{stem}/{index}/grouped_mode")
                    if architecture == "serial" and observed_mode == "parallel_readout":
                        contract_failures.append(f"{stem}/{index}/serial_mode")
                    expected_pair = {
                        "nested_factor": ("factorized_sensors", "hierarchical"),
                        "flat_factor": ("factorized_sensors", "flat"),
                        "local_ratio": ("paired_cumulative", "hierarchical"),
                    }[family]
                    if (dataset["nuisance_layout"], dataset["gain_structure"]) != expected_pair:
                        contract_failures.append(f"{stem}/{index}/family")
                    broadcast = config["training"]["main"].get(
                        "learning_strategy_config", {}
                    ).get("error_broadcast_mode")
                    if credit == "local3f" and broadcast != "path_transport":
                        contract_failures.append(f"{stem}/{index}/transport")

                    log_text = "\n".join(
                        path.read_text(errors="replace")
                        for path in (result / "train.log", result / "dendritic_modeling.log")
                        if path.is_file()
                    ).lower()
                    rows.append(
                        {
                            "family": family,
                            "architecture": architecture,
                            "credit": credit,
                            "alignment_alpha": alpha,
                            "seed": seed,
                            "config_index": index,
                            "test_accuracy": float(final["accuracy"]["test"]),
                            "train_accuracy": float(final["accuracy"]["train"]),
                            "test_auc": float(final["auc"]["test"]),
                            "fallback_mentions": int(log_text.count("fallback")),
                            "nonfinite_alert": bool(
                                "nan detected" in log_text
                                or "non-finite" in log_text
                                or "nonfinite" in log_text
                            ),
                            "config_sha256": sha256(config_path),
                            "final_sha256": sha256(final_path),
                            **resources,
                        }
                    )

    frame = pd.DataFrame(rows)
    expected_keys = {
        (family, architecture, credit, alpha, seed)
        for family in FAMILIES
        for architecture in ARCHITECTURES
        for credit in CREDITS
        for alpha in ALPHAS
        for seed in SEEDS
    }
    missing_keys = sorted(expected_keys - seen)
    missing = sorted(set(missing))
    finite = bool(
        len(frame)
        and np.isfinite(
            frame[["test_accuracy", "train_accuracy", "test_auc"]].to_numpy()
        ).all()
    )
    resource_equal = bool(
        len(frame)
        and all(frame[column].nunique() == 1 for column in RESOURCE_COLUMNS)
    )
    audit = {
        "status": (
            "complete"
            if len(frame) == 360 and not missing and not missing_keys
            else "incomplete"
        ),
        "expected_rows": 360,
        "observed_rows": int(len(frame)),
        "missing_count": len(missing_keys),
        "missing_examples": ["/".join(map(str, key)) for key in missing_keys[:40]],
        "missing_artifact_count": len(missing),
        "contract_failures": contract_failures,
        "families": sorted(frame.family.unique().tolist()) if len(frame) else [],
        "architectures": sorted(frame.architecture.unique().tolist()) if len(frame) else [],
        "credits": sorted(frame.credit.unique().tolist()) if len(frame) else [],
        "alignment_alpha": sorted(frame.alignment_alpha.unique().tolist()) if len(frame) else [],
        "seeds": sorted(frame.seed.unique().tolist()) if len(frame) else [],
        "finite_metrics": finite,
        "fallback_mentions": int(frame.fallback_mentions.sum()) if len(frame) else 0,
        "nonfinite_alert_rows": int(frame.nonfinite_alert.sum()) if len(frame) else 0,
        "resource_equal": resource_equal,
        "resource_values": {
            column: sorted(frame[column].unique().tolist()) if len(frame) else []
            for column in RESOURCE_COLUMNS
        },
        "runs": run_records,
    }
    audit["all_gates_pass"] = bool(
        audit["status"] == "complete"
        and not contract_failures
        and finite
        and audit["fallback_mentions"] == 0
        and audit["nonfinite_alert_rows"] == 0
        and resource_equal
        and audit["families"] == sorted(FAMILIES)
        and audit["architectures"] == sorted(ARCHITECTURES)
        and audit["credits"] == sorted(CREDITS)
        and audit["alignment_alpha"] == list(ALPHAS)
        and audit["seeds"] == list(SEEDS)
    )
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Task-family factorial incomplete: {len(frame)}/360 rows; "
            f"examples {missing[:10]}"
        )
    return frame, audit


def summarize(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    grouped = frame.groupby(
        ["family", "architecture", "credit", "alignment_alpha"], sort=True
    )
    for index, (key, part) in enumerate(grouped):
        mean, low, high = bootstrap_mean(
            part["test_accuracy"].to_numpy(), 8_900_000 + index
        )
        summary_rows.append(
            {
                "family": key[0],
                "architecture": key[1],
                "credit": key[2],
                "alignment_alpha": key[3],
                "n_seeds": int(part.seed.nunique()),
                "mean_test_accuracy": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    wide = frame.pivot(
        index=["family", "credit", "alignment_alpha", "seed"],
        columns="architecture",
        values="test_accuracy",
    ).reset_index()
    wide["serial_minus_grouped"] = wide["serial"] - wide["grouped_point"]
    effect_rows: list[dict[str, Any]] = []
    for index, (key, part) in enumerate(
        wide.groupby(["family", "credit", "alignment_alpha"], sort=True)
    ):
        values = part["serial_minus_grouped"].to_numpy()
        mean, low, high = bootstrap_mean(values, 8_910_000 + index)
        effect_rows.append(
            {
                "estimand": "serial_minus_grouped",
                "family": key[0],
                "credit": key[1],
                "alignment_alpha": key[2],
                "n_pairs": len(values),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int(np.sum(values > 0)),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
            }
        )
    effects = pd.DataFrame(effect_rows)

    pivot = wide.pivot(
        index=["family", "credit", "seed"],
        columns="alignment_alpha",
        values="serial_minus_grouped",
    ).reset_index()
    pivot["alignment_interaction"] = pivot[1.0] - pivot[0.0]
    contrast_rows: list[dict[str, Any]] = []
    for index, (key, part) in enumerate(
        pivot.groupby(["family", "credit"], sort=True)
    ):
        values = part["alignment_interaction"].to_numpy()
        mean, low, high = bootstrap_mean(values, 8_920_000 + index)
        contrast_rows.append(
            {
                "estimand": "alignment_interaction",
                "family": key[0],
                "credit": key[1],
                "comparison": "alpha_1_minus_0",
                "n_pairs": len(values),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int(np.sum(values > 0)),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
            }
        )

    task_wide = pivot.pivot(
        index=["credit", "seed"],
        columns="family",
        values="alignment_interaction",
    ).reset_index()
    for credit in CREDITS:
        part = task_wide[task_wide.credit.eq(credit)]
        for offset, comparison in enumerate(("flat_factor", "local_ratio")):
            values = (
                part["nested_factor"] - part[comparison]
            ).to_numpy(dtype=float)
            mean, low, high = bootstrap_mean(
                values, 8_930_000 + 10 * CREDITS.index(credit) + offset
            )
            contrast_rows.append(
                {
                    "estimand": "task_specificity_interaction",
                    "family": "nested_factor",
                    "credit": credit,
                    "comparison": f"nested_minus_{comparison}",
                    "n_pairs": len(values),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_pairs": int(np.sum(values > 0)),
                    "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
                }
            )

    credit_wide = wide.pivot(
        index=["family", "alignment_alpha", "seed"],
        columns="credit",
        values="serial_minus_grouped",
    ).reset_index()
    for index, (key, part) in enumerate(
        credit_wide.groupby(["family", "alignment_alpha"], sort=True)
    ):
        values = (part["local3f"] - part["bp"]).to_numpy(dtype=float)
        mean, low, high = bootstrap_mean(values, 8_940_000 + index)
        contrast_rows.append(
            {
                "estimand": "credit_interaction",
                "family": key[0],
                "credit": "local3f_minus_bp",
                "comparison": f"alpha_{key[1]:.1f}",
                "n_pairs": len(values),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int(np.sum(values > 0)),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
            }
        )
    return pd.DataFrame(summary_rows), effects, pd.DataFrame(contrast_rows)


def render(effects: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    apply_neurips_style()
    family_labels = {
        "nested_factor": "nested factors",
        "flat_factor": "flat factors",
        "local_ratio": "local ratios",
    }
    family_colors = {
        "nested_factor": COLORS["shunting"],
        "flat_factor": COLORS["point_mlp"],
        "local_ratio": COLORS["mute"],
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 2.55),
        gridspec_kw={
            "left": 0.105,
            "right": 0.985,
            # Leave enough source-panel margin for the journal compositor to
            # replace panel headings without covering axis labels.
            "bottom": 0.23,
            "top": 0.75,
            "wspace": 0.72,
        },
    )
    heading_artists = []
    for axis, credit, letter, title in zip(
        axes[:2],
        CREDITS,
        ("A", "B"),
        ("Exact backpropagation", "Path-transport LocalCA"),
    ):
        axis.axhline(0, color=COLORS["mute"], lw=LW_HAIR, zorder=0)
        for family in FAMILIES:
            part = effects[
                effects.family.eq(family) & effects.credit.eq(credit)
            ].sort_values("alignment_alpha")
            mean = 100 * part.mean_difference.to_numpy()
            low = 100 * part.ci95_low.to_numpy()
            high = 100 * part.ci95_high.to_numpy()
            axis.errorbar(
                part.alignment_alpha,
                mean,
                yerr=np.vstack([mean - low, high - mean]),
                color=family_colors[family],
                marker="o",
                ms=MARKER_MS,
                lw=LW_DATA,
                elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=family_labels[family],
            )
        axis.set_xticks(ALPHAS)
        axis.set_xlabel("sensor alignment $\\alpha$")
        axis.set_ylabel("accuracy difference (pp)")
        heading_artists.append(panel_title(axis, letter, title))
        style_axis(axis, grid="y")
    clean_legend(axes[0], loc="best", fontsize=PT_LEGEND)

    rows = contrasts[contrasts.estimand.eq("alignment_interaction")].copy()
    rows["label"] = rows.apply(
        lambda row: (
            f"{family_labels[row.family].replace(' factors', '').replace('local ratios', 'ratio')}"
            f"/{'BP' if row.credit == 'bp' else 'local'}"
        ),
        axis=1,
    )
    rows = rows.sort_values(["family", "credit"])
    y = np.arange(len(rows))[::-1]
    mean = 100 * rows.mean_difference.to_numpy()
    low = 100 * rows.ci95_low.to_numpy()
    high = 100 * rows.ci95_high.to_numpy()
    colors = [family_colors[value] for value in rows.family]
    axes[2].axvline(0, color=COLORS["mute"], lw=LW_HAIR)
    for idx in range(len(rows)):
        axes[2].errorbar(
            mean[idx],
            y[idx],
            xerr=np.asarray([[mean[idx] - low[idx]], [high[idx] - mean[idx]]]),
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            ms=MARKER_MS,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    axes[2].set_yticks(y, rows.label.tolist(), fontsize=6.6)
    axes[2].set_xlabel("alignment interaction (pp)")
    heading_artists.append(panel_title(axes[2], "C", "Architecture × alignment"))
    style_axis(axes[2], grid="x")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_task_family_alignment")
    audit_text_over_data(fig, "fig_task_family_alignment")
    fig.savefig(
        FIGURES / "fig_task_family_alignment.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_task_family_alignment.png", dpi=600)
    # The compact main-figure compositor supplies its own uniform headings.
    # Preserve a headerless vector source so replacement titles never cover
    # axes, tick labels, or legends.
    for axis in axes:
        for title_artist in (axis.title, axis._left_title, axis._right_title):
            title_artist.set_text("")
    for artist in heading_artists:
        if artist is not None:
            artist.set_visible(False)
    fig.savefig(
        FIGURES / "fig_task_family_alignment_composite.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    frame, audit = collect(args.runs, args.allow_incomplete)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False)
    if audit["all_gates_pass"]:
        summary, effects, contrasts = summarize(frame)
        summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
        effects.to_csv(OUTPUT / "architecture_effects.csv", index=False)
        contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
        render(effects, contrasts)
        primary = contrasts[
            contrasts.estimand.isin(
                ["alignment_interaction", "task_specificity_interaction"]
            )
        ]
        lines = [
            "# Fixed-depth task-family by alignment factorial",
            "",
            "Status: complete; all 360 fits passed the frozen gates.",
            "",
        ]
        for row in primary.itertuples():
            lines.append(
                f"- {row.estimand}; {row.family}; {row.credit}; {row.comparison}: "
                f"{100 * row.mean_difference:.2f} pp "
                f"(95% paired-seed bootstrap CI {100 * row.ci95_low:.2f} to "
                f"{100 * row.ci95_high:.2f}; {row.positive_pairs}/{row.n_pairs} "
                "positive seeds)."
            )
        report = "\n".join(lines) + "\n"
    else:
        report = (
            "# Fixed-depth task-family by alignment factorial\n\n"
            f"Status: incomplete ({audit['observed_rows']}/360 fits).\n"
        )
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
