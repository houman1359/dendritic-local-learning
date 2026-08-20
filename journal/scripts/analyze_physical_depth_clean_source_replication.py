#!/usr/bin/env python3
"""Audit H=2/H=3 clean-source reruns and compare historical outcomes."""

from __future__ import annotations

import argparse
import hashlib
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

from analyze_remaining_physical_experiments import (  # noqa: E402
    bootstrap_mean,
    exact_sign_flip_p,
)
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
OUTPUT = ROOT / "source_data" / "physical_depth_clean_source_replication"
FIGURES = ROOT / "figures" / "generated"
HISTORICAL_REMAINING = (
    ROOT / "source_data" / "remaining_physical_experiments" / "seed_outcomes_new.csv"
)
HISTORICAL_CONFIRMATORY = (
    ROOT / "source_data" / "nonlinear_physical_depth_confirmatory" / "seed_outcomes.csv"
)
SOURCE_COMMIT = "a99c3a777f99913e13dfe673a3f3a28bfe3566af"

# stem: (hierarchy, regime, architecture, mechanism, default credit)
REMAINING_SPECS = {
    "journal_remaining_h3_aligned_grouped_point_bp": (
        3,
        "aligned",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h3_rewired_tree_grouped_point_bp": (
        3,
        "rewired_tree",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h2_aligned_serial_bp": (
        2,
        "aligned",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h2_rewired_tree_serial_bp": (
        2,
        "rewired_tree",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h2_aligned_grouped_point_bp": (
        2,
        "aligned",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h2_rewired_tree_grouped_point_bp": (
        2,
        "rewired_tree",
        "grouped_point",
        "shunting",
        "full_bp",
    ),
    "journal_remaining_h2_aligned_serial_local3f": (
        2,
        "aligned",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
    "journal_remaining_h2_rewired_tree_serial_local3f": (
        2,
        "rewired_tree",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
}

CONFIRMATORY_SPECS = {
    "journal_confirmatory_physical_depth_aligned_additive_bp": (
        3,
        "aligned",
        "serial_tree",
        "raw_additive",
        "full_bp",
    ),
    "journal_confirmatory_physical_depth_aligned_shunting_bp": (
        3,
        "aligned",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_confirmatory_physical_depth_aligned_shunting_local3f": (
        3,
        "aligned",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
    "journal_confirmatory_physical_depth_rewired_tree_shunting_bp": (
        3,
        "rewired_tree",
        "serial_tree",
        "shunting",
        "full_bp",
    ),
    "journal_confirmatory_physical_depth_rewired_tree_shunting_local3f": (
        3,
        "rewired_tree",
        "serial_tree",
        "shunting",
        "local_auto",
    ),
}

RESOURCE_COLUMNS = [
    "total_parameters",
    "trainable_parameters",
    "active_synapses",
    "candidate_synapse_slots",
    "persistent_state_scalars",
]
PAIR_KEYS = [
    "hierarchy",
    "regime",
    "architecture",
    "mechanism",
    "credit",
    "depth",
    "seed",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_run(root: Path, stem: str) -> Path:
    matches = sorted(root.glob(f"{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {stem} below {root}")
    return matches[-1]


def _metric(final: dict[str, Any], name: str, split: str) -> float:
    return float(final[name][split])


def collect_family(
    runs_root: Path,
    specs: dict[str, tuple[int, str, str, str, str]],
    *,
    allow_incomplete: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    missing: list[str] = []
    source_failures: list[str] = []
    for stem, (
        hierarchy,
        regime,
        architecture,
        mechanism,
        default_credit,
    ) in specs.items():
        run = latest_run(runs_root, stem)
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        expected = int(original["sweep_contract"]["expected_config_count"])
        configs = sorted(
            (run / "configs").glob("unified_config_*.yaml"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        manifest_path = run / "frozen_sweep_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source = manifest["source_identity"]["git"]
        if source["commit"] != SOURCE_COMMIT:
            source_failures.append(f"{stem}/commit/{source['commit']}")
        if source["tracked_worktree_dirty"]:
            source_failures.append(f"{stem}/dirty")
        run_records.append(
            {
                "stem": stem,
                "run_dir": str(run),
                "expected": expected,
                "generated": len(configs),
                "manifest_sha256": sha256(manifest_path),
                "source_commit": source["commit"],
                "source_dirty": bool(source["tracked_worktree_dirty"]),
            }
        )
        if len(configs) != expected:
            missing.append(f"{stem}: generated {len(configs)}/{expected}")

        for config_path in configs:
            index = int(config_path.stem.rsplit("_", 1)[1])
            result = run / "results" / f"config_{index}"
            final_path = result / "performance" / "final.json"
            resources_path = result / "model_resources.json"
            if not final_path.is_file() or not resources_path.is_file():
                missing.append(f"{stem}/config_{index}")
                continue
            config = yaml.safe_load(config_path.read_text())
            final = json.loads(final_path.read_text())
            resources = json.loads(resources_path.read_text())
            factors = config["model"]["core"]["population_network"]["layers"][0][
                "populations"
            ][0]["branch_factors"]
            if default_credit == "local_auto":
                broadcast = str(
                    config["training"]["main"]["learning_strategy_config"][
                        "error_broadcast_mode"
                    ]
                ).lower()
                credit = (
                    "local_path" if broadcast == "path_transport" else "local_shared"
                )
            else:
                credit = default_credit
            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result / "train.log", result / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            row: dict[str, Any] = {
                "hierarchy": hierarchy,
                "cohort": stem,
                "run_dir": str(run),
                "config_index": index,
                "seed": int(config["experiment"]["seed"]),
                "regime": regime,
                "architecture": architecture,
                "mechanism": mechanism,
                "credit": credit,
                "depth": len(factors),
                "branch_factors": "x".join(map(str, factors)),
                "total_parameters": int(resources["total_parameters"]),
                "trainable_parameters": int(resources["trainable_parameters"]),
                "active_synapses": int(resources.get("active_synapses", 0)),
                "candidate_synapse_slots": int(
                    resources.get("candidate_synapse_slots", 0)
                ),
                "persistent_state_scalars": int(
                    resources.get("persistent_state_scalars_per_sample", 0)
                ),
                "config_sha256": sha256(config_path),
                "final_sha256": sha256(final_path),
                "fallback_mentions": int(log_text.count("fallback")),
                "nonfinite_alert": bool(
                    "nan detected" in log_text
                    or "non-finite" in log_text
                    or "nonfinite" in log_text
                ),
            }
            for name in ("accuracy", "auc", "categorical_loglikelihood"):
                for split in ("train", "valid", "test"):
                    row[f"{split}_{name}"] = _metric(final, name, split)
            rows.append(row)

    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Clean-source replication incomplete ({len(missing)} missing): "
            + ", ".join(missing[:20])
        )
    return pd.DataFrame(rows), run_records, missing, source_failures


def historical_frame() -> pd.DataFrame:
    remaining = pd.read_csv(HISTORICAL_REMAINING).copy()
    remaining["mechanism"] = "shunting"

    confirmatory = pd.read_csv(HISTORICAL_CONFIRMATORY)
    confirmatory = confirmatory[
        confirmatory.cohort.isin(
            [
                "aligned_additive_bp",
                "aligned_shunting_bp",
                "aligned_shunting_local3f",
                "rewired_tree_shunting_bp",
                "rewired_tree_shunting_local3f",
            ]
        )
    ].copy()
    confirmatory["hierarchy"] = 3
    confirmatory["architecture"] = "serial_tree"
    confirmatory["mechanism"] = confirmatory.mechanism.replace(
        {"additive": "raw_additive"}
    )
    confirmatory["credit"] = np.where(
        confirmatory.method.eq("bp"),
        "full_bp",
        np.where(
            confirmatory.transport.eq("path_transport"),
            "local_path",
            "local_shared",
        ),
    )
    columns = sorted(set(remaining.columns) | set(confirmatory.columns))
    combined = pd.concat(
        [remaining.reindex(columns=columns), confirmatory.reindex(columns=columns)],
        ignore_index=True,
    )
    if len(combined) != 430:
        raise ValueError(f"Expected 430 historical rows, found {len(combined)}")
    return combined


def concordance(clean: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    metrics = ["train_accuracy", "valid_accuracy", "test_accuracy"]
    left = clean[PAIR_KEYS + metrics].copy()
    right = historical[PAIR_KEYS + metrics].copy()
    paired = left.merge(
        right,
        on=PAIR_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_clean", "_historical"),
        indicator=True,
    )
    for metric in metrics:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_clean"] - paired[f"{metric}_historical"]
        )
        paired[f"{metric}_absolute_difference"] = paired[f"{metric}_difference"].abs()
    return paired


def select(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    part = frame.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if part.seed.duplicated().any():
        raise ValueError(f"Duplicate seed rows for {filters}")
    return part.set_index("seed").test_accuracy.sort_index()


def difference(
    frame: pd.DataFrame, left: dict[str, Any], right: dict[str, Any]
) -> pd.Series:
    first, second = select(frame, **left).align(select(frame, **right), join="inner")
    if len(first) != 10 or not first.index.equals(second.index):
        raise ValueError(f"Expected ten paired seeds for {left} minus {right}")
    return first - second


def contrast_row(
    name: str, values: pd.Series, detail: str, seed: int
) -> dict[str, Any]:
    mean, low, high = bootstrap_mean(values.to_numpy(float), seed)
    return {
        "contrast": name,
        "detail": detail,
        "n_seeds": len(values),
        "mean_accuracy": mean,
        "mean_pp": 100 * mean,
        "ci_low_pp": 100 * low,
        "ci_high_pp": 100 * high,
        "positive_pairs": int((values > 0).sum()),
        "negative_pairs": int((values < 0).sum()),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(values.to_numpy(float)),
        "positive_claim_gate": bool(low > 0 and int((values > 0).sum()) >= 8),
    }


def build_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    values_by_name: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, detail: str) -> None:
        values_by_name[name] = values
        rows.append(contrast_row(name, values, detail, 17100 + len(rows)))

    for hierarchy, deep in ((2, 2), (3, 3)):
        for method, architecture, credit in (
            ("serial_bp", "serial_tree", "full_bp"),
            ("grouped_bp", "grouped_point", "full_bp"),
            ("shared_local", "serial_tree", "local_shared"),
            ("path_local", "serial_tree", "local_path"),
        ):
            for regime in ("aligned", "rewired_tree"):
                add(
                    f"h{hierarchy}_depth__{method}__{regime}",
                    difference(
                        frame,
                        {
                            "hierarchy": hierarchy,
                            "regime": regime,
                            "architecture": architecture,
                            "mechanism": "shunting",
                            "credit": credit,
                            "depth": deep,
                        },
                        {
                            "hierarchy": hierarchy,
                            "regime": regime,
                            "architecture": architecture,
                            "mechanism": "shunting",
                            "credit": credit,
                            "depth": 1,
                        },
                    ),
                    f"H={hierarchy} {regime}: D{deep} minus D1 for {method}",
                )
            add(
                f"h{hierarchy}_placement_interaction__{method}",
                values_by_name[f"h{hierarchy}_depth__{method}__aligned"]
                - values_by_name[f"h{hierarchy}_depth__{method}__rewired_tree"],
                f"H={hierarchy} aligned-minus-reversed depth interaction for {method}",
            )
        for regime in ("aligned", "rewired_tree"):
            add(
                f"h{hierarchy}_serial_minus_grouped__{regime}__d{deep}",
                difference(
                    frame,
                    {
                        "hierarchy": hierarchy,
                        "regime": regime,
                        "architecture": "serial_tree",
                        "mechanism": "shunting",
                        "credit": "full_bp",
                        "depth": deep,
                    },
                    {
                        "hierarchy": hierarchy,
                        "regime": regime,
                        "architecture": "grouped_point",
                        "mechanism": "shunting",
                        "credit": "full_bp",
                        "depth": deep,
                    },
                ),
                f"H={hierarchy} {regime}: serial minus grouped point at D{deep}",
            )
        add(
            f"h{hierarchy}_architecture_placement_interaction__d{deep}",
            values_by_name[f"h{hierarchy}_serial_minus_grouped__aligned__d{deep}"]
            - values_by_name[
                f"h{hierarchy}_serial_minus_grouped__rewired_tree__d{deep}"
            ],
            f"H={hierarchy} architecture-by-placement interaction at D{deep}",
        )

    add(
        "h3_additive_depth__aligned",
        difference(
            frame,
            {
                "hierarchy": 3,
                "regime": "aligned",
                "architecture": "serial_tree",
                "mechanism": "raw_additive",
                "credit": "full_bp",
                "depth": 3,
            },
            {
                "hierarchy": 3,
                "regime": "aligned",
                "architecture": "serial_tree",
                "mechanism": "raw_additive",
                "credit": "full_bp",
                "depth": 1,
            },
        ),
        "H=3 aligned raw-additive BP: D3 minus D1",
    )
    add(
        "h3_shunting_additive_depth_interaction__aligned",
        values_by_name["h3_depth__serial_bp__aligned"]
        - values_by_name["h3_additive_depth__aligned"],
        "H=3 aligned shunting-minus-additive D3-minus-D1 interaction",
    )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(values_by_name).rename_axis("seed").reset_index(),
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["hierarchy", "regime", "architecture", "mechanism", "credit", "depth"]
    for index, (key, part) in enumerate(frame.groupby(grouping, sort=True)):
        mean, low, high = bootstrap_mean(
            part.test_accuracy.to_numpy(float), 18100 + index
        )
        rows.append(
            dict(
                zip(grouping, key),
                n_seeds=part.seed.nunique(),
                mean_test_accuracy=mean,
                ci_low=low,
                ci_high=high,
            )
        )
    return pd.DataFrame(rows)


def audit(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    run_records: list[dict[str, Any]],
    missing: list[str],
    source_failures: list[str],
) -> dict[str, Any]:
    finite_columns = [
        column
        for column in frame.columns
        if column.endswith(("_accuracy", "_auc", "_categorical_loglikelihood"))
    ]
    resource_failures: list[str] = []
    for (hierarchy, regime, depth), group in frame[frame.credit.eq("full_bp")].groupby(
        ["hierarchy", "regime", "depth"]
    ):
        for column in RESOURCE_COLUMNS:
            if group[column].nunique() != 1:
                resource_failures.append(f"H{hierarchy}/{regime}/D{depth}/{column}")
    differences = paired.test_accuracy_absolute_difference.dropna().to_numpy(float)
    record = {
        "status": "complete" if not missing else "incomplete",
        "expected_rows": 430,
        "observed_rows": len(frame),
        "missing_count": len(missing),
        "missing_examples": missing[:40],
        "source_commit": SOURCE_COMMIT,
        "source_failures": source_failures,
        "runs": run_records,
        "duplicate_conditions": int(frame.duplicated(PAIR_KEYS).sum()),
        "all_metrics_finite": bool(
            len(frame) > 0 and np.isfinite(frame[finite_columns].to_numpy(float)).all()
        ),
        "fallback_mentions": int(frame.fallback_mentions.sum()) if len(frame) else 0,
        "nonfinite_alerts": int(frame.nonfinite_alert.sum()) if len(frame) else 0,
        "resource_failures": resource_failures,
        "historical_pair_rows": len(paired),
        "historical_unmatched_rows": int((paired._merge != "both").sum()),
        "mean_absolute_test_accuracy_difference": (
            float(differences.mean()) if len(differences) else None
        ),
        "median_absolute_test_accuracy_difference": (
            float(np.median(differences)) if len(differences) else None
        ),
        "maximum_absolute_test_accuracy_difference": (
            float(differences.max()) if len(differences) else None
        ),
        "pairs_within_1e_6": int((differences <= 1e-6).sum()),
        "pairs_within_0_1pp": int((differences <= 0.001).sum()),
    }
    record["all_validity_gates_pass"] = bool(
        record["status"] == "complete"
        and record["observed_rows"] == 430
        and not record["source_failures"]
        and record["duplicate_conditions"] == 0
        and record["all_metrics_finite"]
        and record["fallback_mentions"] == 0
        and record["nonfinite_alerts"] == 0
        and not record["resource_failures"]
        and record["historical_pair_rows"] == 430
        and record["historical_unmatched_rows"] == 0
    )
    return record


def _line_panel(
    ax: plt.Axes, summary: pd.DataFrame, hierarchy: int, regime: str, letter: str
) -> None:
    styles = [
        (
            "serial_tree",
            "shunting",
            "full_bp",
            "serial BP",
            COLORS["shunting"],
            "o",
            "-",
        ),
        (
            "grouped_point",
            "shunting",
            "full_bp",
            "grouped point BP",
            COLORS["point_mlp"],
            "s",
            "--",
        ),
        (
            "serial_tree",
            "shunting",
            "local_shared",
            "shared LocalCA",
            COLORS["local"],
            "D",
            "-.",
        ),
        (
            "serial_tree",
            "shunting",
            "local_path",
            "path LocalCA",
            COLORS["pathway"],
            "^",
            ":",
        ),
    ]
    if hierarchy == 3 and regime == "aligned":
        styles.append(
            (
                "serial_tree",
                "raw_additive",
                "full_bp",
                "raw-additive BP",
                COLORS["mute"],
                "v",
                "--",
            )
        )
    for architecture, mechanism, credit, label, color, marker, linestyle in styles:
        part = summary[
            summary.hierarchy.eq(hierarchy)
            & summary.regime.eq(regime)
            & summary.architecture.eq(architecture)
            & summary.mechanism.eq(mechanism)
            & summary.credit.eq(credit)
        ].sort_values("depth")
        if part.empty:
            continue
        y = part.mean_test_accuracy.to_numpy(float)
        low = part.ci_low.to_numpy(float)
        high = part.ci_high.to_numpy(float)
        ax.errorbar(
            part.depth,
            y,
            yerr=np.vstack([y - low, high - y]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            markersize=MARKER_MS,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
    panel_title(ax, letter, f"Clean source: H={hierarchy} {regime.replace('_', ' ')}")
    ax.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(list(range(1, hierarchy + 1)))
    style_axis(ax, grid="y")
    clean_legend(ax, loc="best", fontsize=PT_LEGEND, handlelength=2.0)


def make_figure(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    apply_neurips_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 5.8),
        gridspec_kw={
            "left": 0.11,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.92,
            "wspace": 0.43,
            "hspace": 0.62,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    both = paired[paired._merge.eq("both")]
    ax_a.scatter(
        both.test_accuracy_historical,
        both.test_accuracy_clean,
        s=10,
        alpha=0.35,
        color=COLORS["shunting"],
        edgecolors="none",
    )
    limits = [
        float(min(both.test_accuracy_historical.min(), both.test_accuracy_clean.min())),
        float(max(both.test_accuracy_historical.max(), both.test_accuracy_clean.max())),
    ]
    ax_a.plot(limits, limits, color=COLORS["mute"], linewidth=LW_HAIR, linestyle="--")
    ax_a.set_xlim(limits)
    ax_a.set_ylim(limits)
    ax_a.set_xlabel("historical test accuracy")
    ax_a.set_ylabel("clean-source test accuracy")
    panel_title(ax_a, "A", "Outcome concordance")
    style_axis(ax_a, grid="both")

    delta_pp = 100 * both.test_accuracy_difference.to_numpy(float)
    ax_b.hist(delta_pp, bins=30, color=COLORS["shunting"], alpha=0.8)
    ax_b.axvline(0, color=COLORS["mute"], linewidth=LW_HAIR)
    ax_b.set_xlabel("clean minus historical (pp)")
    ax_b.set_ylabel("seed--condition pairs")
    panel_title(ax_b, "B", "Paired source sensitivity")
    style_axis(ax_b, grid="y")

    _line_panel(ax_c, summary, 2, "aligned", "C")
    _line_panel(ax_d, summary, 3, "aligned", "D")
    fig.canvas.draw()
    audit_layout(fig, "fig_physical_depth_clean_source_replication")
    audit_text_over_data(fig, "fig_physical_depth_clean_source_replication")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "fig_physical_depth_clean_source_replication.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_physical_depth_clean_source_replication.png", dpi=600)
    plt.close(fig)


def write_report(contrasts: pd.DataFrame, record: dict[str, Any]) -> None:
    indexed = contrasts.set_index("contrast")

    def sentence(name: str) -> str:
        row = indexed.loc[name]
        return (
            f"{row.mean_pp:.2f} pp (95% paired-seed bootstrap interval "
            f"{row.ci_low_pp:.2f} to {row.ci_high_pp:.2f}; "
            f"{int(row.positive_pairs)}/10 positive; exact two-sided sign-flip "
            f"P={row.exact_two_sided_sign_flip_p:.4f}; "
            f"positive gate={'pass' if row.positive_claim_gate else 'fail'})"
        )

    report = f"""# Physical-depth clean-source replication

Status: **{"complete_pass" if record["all_validity_gates_pass"] else record["status"]}** ({record["observed_rows"]}/430 reruns).

## Historical concordance

- Mean absolute test-accuracy change: {100 * record["mean_absolute_test_accuracy_difference"]:.4f} pp.
- Median absolute change: {100 * record["median_absolute_test_accuracy_difference"]:.4f} pp.
- Maximum absolute change: {100 * record["maximum_absolute_test_accuracy_difference"]:.4f} pp.
- Pairs within 0.1 pp: {record["pairs_within_0_1pp"]}/430.

## H=2

- Aligned serial-BP D2 minus D1: {sentence("h2_depth__serial_bp__aligned")}.
- Serial-BP depth-by-placement interaction: {sentence("h2_placement_interaction__serial_bp")}.
- Serial minus grouped point at aligned D2: {sentence("h2_serial_minus_grouped__aligned__d2")}.
- Architecture-by-placement interaction: {sentence("h2_architecture_placement_interaction__d2")}.

## H=3

- Aligned serial-BP D3 minus D1: {sentence("h3_depth__serial_bp__aligned")}.
- Serial-BP depth-by-placement interaction: {sentence("h3_placement_interaction__serial_bp")}.
- Serial minus grouped point at aligned D3: {sentence("h3_serial_minus_grouped__aligned__d3")}.
- Architecture-by-placement interaction: {sentence("h3_architecture_placement_interaction__d3")}.
- Shunting-minus-additive depth interaction: {sentence("h3_shunting_additive_depth_interaction__aligned")}.
"""
    (OUTPUT / "RESULTS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining-runs-root", required=True, type=Path)
    parser.add_argument("--confirmatory-runs-root", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    remaining, remaining_runs, missing_a, source_a = collect_family(
        args.remaining_runs_root.resolve(),
        REMAINING_SPECS,
        allow_incomplete=args.allow_incomplete,
    )
    confirmatory, confirmatory_runs, missing_b, source_b = collect_family(
        args.confirmatory_runs_root.resolve(),
        CONFIRMATORY_SPECS,
        allow_incomplete=args.allow_incomplete,
    )
    clean = pd.concat([remaining, confirmatory], ignore_index=True)
    historical = historical_frame()
    paired = concordance(clean, historical)
    record = audit(
        clean,
        paired,
        remaining_runs + confirmatory_runs,
        missing_a + missing_b,
        source_a + source_b,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not clean.empty:
        clean.to_csv(OUTPUT / "seed_outcomes.csv", index=False)
        paired.to_csv(OUTPUT / "historical_concordance.csv", index=False)
    (OUTPUT / "audit.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if record["status"] != "complete":
        print(json.dumps(record, indent=2))
        return

    summary = summarize(clean)
    contrasts, seed_contrasts = build_contrasts(clean)
    summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
    contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    seed_contrasts.to_csv(OUTPUT / "paired_contrasts_by_seed.csv", index=False)
    write_report(contrasts, record)
    make_figure(summary, paired)
    print(json.dumps(record, indent=2))
    print((OUTPUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
