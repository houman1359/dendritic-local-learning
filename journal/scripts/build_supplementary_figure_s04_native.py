#!/usr/bin/env python3
"""Supplementary Fig. S4 -- the regular-tree boundary tests -- rebuilt as ONE
native full-width canvas in the journal style.

The reduced sheet was first produced by cropping panels C, D and I out of the
archived nine-panel regime asset (still preserved at
``figures/supplementary/figure_S04_panels_A-I.pdf``); that recomposition kept
the marks bit-identical but inherited the NeurIPS type and stroke scales and
carried no native geometry manifest.  This builder redraws the SAME three
panels (same letters A--C as the reduced sheet, same content, same frozen
numbers) natively:

* row 0 -- A the nominal-depth stress sweep and B the broadcast-noise sweep,
  six modules each (solid = local learning, faint dashed = matched
  backpropagation in A; whiskers are the +-1 s.d. the archived generator
  drew);
* row 1 -- C the flattened CIFAR-10 control ladder, six modules, centred.

Every plotted number is the archived one: the aggregations are ported
verbatim from the frozen ``scripts/build_regular_tree_regime_figure.py``
(same CSVs, same filters, same ddof), and the build prints the values it
draws so they can be diffed against the frozen tables.  Only layout, palette
mapping and label wording are native:

* shunting keeps the house green circle and additive the house blue square;
* the CIFAR ladder wears the manuscript-wide condition semantics (amber =
  scalar, control gray = random rank, oracle violet = exact transport,
  red-brown = backpropagation) instead of the archived green/orange ramp, and
  its rotated tick labels become horizontal ones;
* legend boxes become direct labels.

The default build remains the frozen three-panel A--C sheet.  A fourth panel
can be rendered only by explicitly supplying a finalized analysis directory
from ``analyze_cifar10_additive_feedback_ladder_confirmatory.py``.  That path
is deliberately fail-closed: incomplete, convergence-flagged or internally
inconsistent outputs cannot produce a publication asset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_ERR,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    Margins,
    NativeCanvas,
)

ROOT = SCRIPT_DIR.parent
SRC = ROOT / "source_data" / "regular_tree_regimes"
OUT = ROOT / "figures" / "supplementary" / "figure_S04_panels_A-C.pdf"
OUT_EXTENDED = ROOT / "figures" / "supplementary" / "figure_S04_panels_A-D.pdf"

DEPTH_CSV = SRC / "depth_scaling_summary.csv"
NOISE_CSV = SRC / "broadcast_noise_summary.csv"
CIFAR_CSV = SRC / "cifar10_control_ladder_runs.csv"

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]

CORE_MARKER = {"dendritic_shunting": "o", "dendritic_additive": "s"}
CORE_COLOR = {"dendritic_shunting": SHUNT, "dendritic_additive": ADD}


def _series(ax, x, mean, std, *, color, marker, ls="-", alpha=1.0, z=3):
    ax.errorbar(x, mean, yerr=std, color=color, ls=ls, marker=marker,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_DATA,
                elinewidth=LW_ERR, capsize=ERR_CAPSIZE, alpha=alpha,
                zorder=z)


def panel_depth(ax):
    """Nominal dendritic-depth stress; aggregation ported verbatim."""
    frame = pd.read_csv(DEPTH_CSV)
    frame["depth"] = frame["branch_factors"].astype(str).map(
        lambda value: len(value.strip("[]").split(",")))
    for strategy, ls, alpha in (("local_ca", "-", 1.0),
                                ("standard", "--", 0.32)):
        for network in ("dendritic_shunting", "dendritic_additive"):
            sub = frame[frame["strategy"].eq(strategy)
                        & frame["network_type"].eq(network)
                        ].sort_values("depth")
            print(f"  A {network} {strategy}: "
                  f"{list(np.round(sub['test_accuracy_mean'], 5))}")
            _series(ax, sub["depth"], sub["test_accuracy_mean"],
                    sub["test_accuracy_std"], color=CORE_COLOR[network],
                    marker=CORE_MARKER[network], ls=ls, alpha=alpha,
                    z=3 if strategy == "local_ca" else 2)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.62, 4.38)
    ax.set_xlabel("dendritic layers")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.15, 0.98)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    # Direct labels: the two local series separate at depth 4; the faint
    # dashed pair coincides near the top and gets one mute word.
    ax.text(2.5, 0.845, "matched BP", color=MUTE, fontsize=PT_SMALL,
            ha="center", va="top")
    ax.text(1.12, 0.575, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="bottom")
    ax.text(1.12, 0.44, "additive", color=ADD, fontsize=PT_LEGEND,
            ha="left", va="top")
    return ax


def panel_noise(ax):
    """Broadcast-noise stress; aggregation ported verbatim."""
    frame = pd.read_csv(NOISE_CSV)
    for network in ("dendritic_shunting", "dendritic_additive"):
        sub = frame[frame["network_type"].eq(network)
                    ].sort_values("error_noise_sigma")
        print(f"  B {network}: "
              f"{list(np.round(sub['test_accuracy_mean'], 5))}")
        _series(ax, sub["error_noise_sigma"], sub["test_accuracy_mean"],
                sub["test_accuracy_std"], color=CORE_COLOR[network],
                marker=CORE_MARKER[network])
    ax.set_xlabel("broadcast-noise σ")  # noqa: RUF001 -- scientific symbol
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.05, 0.72)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.text(0.52, 0.60, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="bottom")
    ax.text(0.30, 0.30, "additive", color=ADD, fontsize=PT_LEGEND,
            ha="left", va="top")
    return ax


# The ladder keeps the manuscript-wide condition hues: amber scalar, control
# gray for the random rank field, oracle violet for exact transport and the
# red-brown backpropagation reference.
CIFAR_SPECS = (
    ("cifar10_shunting_5f_per_soma_learned_i", "scalar", COLORS["local"]),
    ("cifar10_shunting_5f_low_rank4_learned_i", "rank 4",
     COLORS["point_mlp"]),
    ("cifar10_shunting_5f_path_transport_learned_i", "exact",
     COLORS["oracle"]),
    ("cifar10_shunting_standard_learned_i", "backprop", COLORS["bp"]),
)

CONFIRMATORY_SPECS = (
    ("strict scalar", "scalar", COLORS["scalar"], "o"),
    ("neuron specific", "neuron", COLORS["per_soma"], "s"),
    ("exact path", "exact path", COLORS["oracle"], "^"),
    ("backpropagation", "backprop", COLORS["bp"], "D"),
)
CONFIRMATORY_CONTRASTS = {
    "neuron specific minus strict scalar",
    "exact path minus neuron specific",
    "exact path minus backpropagation",
}
EXPECTED_CONFIRMATORY_SEEDS = set(range(10800, 10820))


def panel_cifar(ax):
    """Flattened CIFAR-10 ladder; per-run dots, mean bar, +-1 s.d."""
    frame = pd.read_csv(CIFAR_CSV)
    for index, (condition, _label, color) in enumerate(CIFAR_SPECS):
        values = frame[frame["condition"].eq(condition)
                       ]["test_accuracy"].to_numpy(float)
        mean = values.mean()
        std = values.std(ddof=1)
        print(f"  C {condition}: mean {mean:.5f} sd {std:.5f} n {values.size}")
        ax.bar(index, mean, color=color, width=0.62, zorder=2)
        ax.errorbar(index, mean, yerr=std, color=INK, lw=LW_ERR,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
        ax.scatter(np.full(values.size, index)
                   + np.linspace(-0.10, 0.10, values.size), values,
                   s=7.0, facecolor="white", edgecolor=color,
                   linewidth=0.55, zorder=4)
    ax.set_xticks(range(len(CIFAR_SPECS)))
    ax.set_xticklabels([label for _, label, _ in CIFAR_SPECS])
    ax.set_xlim(-0.62, len(CIFAR_SPECS) - 0.38)
    ax.set_ylim(0.0, 0.56)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    return ax


def load_confirmatory_analysis(analysis_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load one complete, analyzer-validated 80-run confirmatory package.

    This is stricter than a plotting convenience function.  It verifies the
    analyzer decision record, the four paired 20-seed arms, and the exported
    condition/contrast summaries before any marks are drawn.  The scientific
    promotion gates are intentionally *not* required: a fully valid negative
    result may still be shown as a supplementary boundary result.
    """
    analysis_dir = Path(analysis_dir)
    required = {
        "summary": analysis_dir / "summary.json",
        "outcomes": analysis_dir / "seed_outcomes.csv",
        "conditions": analysis_dir / "condition_summary.csv",
        "contrasts": analysis_dir / "paired_contrasts.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "confirmatory S4D requires the finalized analyzer package; missing: "
            + ", ".join(missing)
        )

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    audit = summary.get("audit", {})
    decision = summary.get("decision", {})
    errors: list[str] = []
    if summary.get("contract_frozen_before_confirmatory_outcomes") is not True:
        errors.append("the frozen-before-outcomes contract flag is absent")
    if audit.get("status") != "complete_and_validated":
        errors.append(f"audit status is {audit.get('status')!r}")
    if audit.get("integrity_valid") is not True:
        errors.append("integrity audit did not pass")
    if audit.get("convergence_valid") is not True:
        errors.append("convergence audit did not pass")
    if audit.get("n_expected") != 80 or audit.get("n_results_complete") != 80:
        errors.append(
            "audit does not record exactly 80/80 complete results "
            f"({audit.get('n_results_complete')!r}/{audit.get('n_expected')!r})"
        )
    if decision.get("audit_passes") is not True:
        errors.append("analyzer decision does not certify the audit")

    outcomes = pd.read_csv(required["outcomes"])
    needed_columns = {"feedback", "seed", "test_accuracy"}
    absent_columns = sorted(needed_columns - set(outcomes.columns))
    if absent_columns:
        errors.append(f"seed_outcomes.csv lacks columns {absent_columns}")
    else:
        feedback_order = [name for name, *_ in CONFIRMATORY_SPECS]
        observed_feedback = set(outcomes["feedback"].astype(str))
        if observed_feedback != set(feedback_order):
            errors.append(
                f"feedback arms are {sorted(observed_feedback)!r}, expected "
                f"{feedback_order!r}"
            )
        if len(outcomes) != 80:
            errors.append(f"seed_outcomes.csv has {len(outcomes)} rather than 80 rows")
        if outcomes.duplicated(["feedback", "seed"]).any():
            errors.append("seed_outcomes.csv contains duplicate feedback/seed rows")
        counts = outcomes.groupby("feedback")["seed"].nunique().to_dict()
        if any(counts.get(name) != 20 for name in feedback_order):
            errors.append(f"paired seed counts are {counts!r}, expected 20 per arm")
        seed_sets = [
            set(outcomes.loc[outcomes["feedback"].eq(name), "seed"])
            for name in feedback_order
        ]
        if seed_sets and any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
            errors.append("the four feedback arms do not share the same seed set")
        if seed_sets and seed_sets[0] != EXPECTED_CONFIRMATORY_SEEDS:
            errors.append("the paired seed set is not the frozen 10800--10819 set")
        values = pd.to_numeric(outcomes["test_accuracy"], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0.0, 1.0).all():
            errors.append("test accuracies are non-finite or outside [0,1]")

    conditions = pd.read_csv(required["conditions"])
    condition_columns = {"feedback", "n_seeds", "mean_test_accuracy"}
    if not condition_columns.issubset(conditions.columns):
        errors.append("condition_summary.csv lacks required summary columns")
    elif needed_columns.issubset(outcomes.columns):
        indexed = conditions.set_index("feedback")
        expected_feedback = {name for name, *_ in CONFIRMATORY_SPECS}
        if set(indexed.index.astype(str)) != expected_feedback:
            errors.append("condition_summary.csv has the wrong feedback arms")
        else:
            for name in expected_feedback:
                observed = outcomes.loc[
                    outcomes["feedback"].eq(name), "test_accuracy"
                ].to_numpy(float)
                if int(indexed.loc[name, "n_seeds"]) != len(observed):
                    errors.append(f"{name}: condition-summary seed count disagrees")
                if not np.isclose(
                    float(indexed.loc[name, "mean_test_accuracy"]),
                    float(observed.mean()),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    errors.append(f"{name}: condition-summary mean disagrees")

    contrasts = pd.read_csv(required["contrasts"])
    if "contrast" not in contrasts.columns:
        errors.append("paired_contrasts.csv lacks the contrast column")
    elif set(contrasts["contrast"].astype(str)) != CONFIRMATORY_CONTRASTS:
        errors.append("paired_contrasts.csv has the wrong contrast inventory")

    if errors:
        raise RuntimeError("confirmatory S4D package failed validation: " + "; ".join(errors))
    outcomes = outcomes.copy()
    outcomes["feedback"] = pd.Categorical(
        outcomes["feedback"],
        categories=[name for name, *_ in CONFIRMATORY_SPECS],
        ordered=True,
    )
    outcomes = outcomes.sort_values(["seed", "feedback"]).reset_index(drop=True)
    return outcomes, summary


def panel_confirmatory_cifar(ax, outcomes: pd.DataFrame):
    """Fresh paired raw-additive CIFAR-10 ladder with mean and 95% CI."""
    wide = outcomes.pivot(index="seed", columns="feedback", values="test_accuracy")
    order = [name for name, *_ in CONFIRMATORY_SPECS]
    wide = wide.loc[:, order]
    x = np.arange(len(order), dtype=float)
    jitters = np.linspace(-0.10, 0.10, len(wide))
    for (_, row), jitter in zip(wide.iterrows(), jitters, strict=True):
        ax.plot(x + jitter, row.to_numpy(float), color=MUTE, alpha=0.18,
                lw=0.55, zorder=1)
    for index, (condition, _label, color, marker) in enumerate(CONFIRMATORY_SPECS):
        values = wide[condition].to_numpy(float)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(values.size))
        half_width = float(stats.t.ppf(0.975, values.size - 1)) * sem
        ax.scatter(np.full(values.size, index) + jitters, values, s=8.0,
                   facecolor="white", edgecolor=color, linewidth=0.55,
                   alpha=0.78, zorder=3)
        ax.errorbar(index, mean, yerr=half_width, color=color, marker=marker,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.45, ms=MARKER_MS + 0.5,
                    lw=LW_DATA, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=5)
        print(f"  D {condition}: mean {mean:.5f} 95CI half-width {half_width:.5f} "
              f"n {values.size}")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, *_ in CONFIRMATORY_SPECS])
    ax.set_xlim(-0.42, len(order) - 0.58)
    upper = max(0.60, float(np.ceil((wide.to_numpy().max() + 0.02) * 20.0) / 20.0))
    ax.set_ylim(0.0, upper)
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    return ax


CANVAS_H_PT = 358.0                     # 518.4 / 358.0 = 1.45 aspect


def build(path=None, *, confirmatory_analysis_dir=None):
    confirmatory = None
    if confirmatory_analysis_dir is not None:
        confirmatory, _summary = load_confirmatory_analysis(
            Path(confirmatory_analysis_dir)
        )
    canvas = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=[1.0, 1.0],
                          hgutter_pt=32.0, vgutter_pt=30.0,
                          margins=Margins(left=46.0, right=12.0, top=14.0,
                                          bottom=26.0))
    ax_a = canvas.panel("depth", 0, 0, 6, grid="y", title="Depth stress")
    ax_b = canvas.panel("noise", 0, 6, 6, grid="y",
                        title="Noisy teaching signal")
    if confirmatory is None:
        ax_c = canvas.panel("cifar", 1, 3, 6, grid="y",
                            title="Harder-data control")
    else:
        ax_c = canvas.panel("cifar", 1, 0, 6, grid="y",
                            title="Shunting feedback ladder")
        ax_d = canvas.panel("cifar_confirmatory", 1, 6, 6, grid="y",
                            title="Raw-additive feedback ladder")
    panel_depth(ax_a)
    panel_noise(ax_b)
    panel_cifar(ax_c)
    if confirmatory is not None:
        panel_confirmatory_cifar(ax_d, confirmatory)
    target = Path(path) if path else (OUT_EXTENDED if confirmatory is not None else OUT)
    asset_name = (
        "figure_S04_panels_A-D" if confirmatory is not None
        else "figure_S04_panels_A-C"
    )
    problems = canvas.save(target, name=asset_name)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--confirmatory-analysis-dir",
        type=Path,
        help=(
            "finalized output directory from the CIFAR-10 confirmatory analyzer; "
            "when supplied, validates 80/80 runs and emits panels A-D"
        ),
    )
    args = parser.parse_args()
    raise SystemExit(
        1
        if build(
            args.output,
            confirmatory_analysis_dir=args.confirmatory_analysis_dir,
        )
        else 0
    )
