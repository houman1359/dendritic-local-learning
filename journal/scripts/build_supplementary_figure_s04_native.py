#!/usr/bin/env python3
"""Supplementary Fig. S4 -- the regular-tree boundary tests -- rebuilt as ONE
native full-width canvas in the journal style.

The reduced sheet was first produced by cropping panels C, D and I out of the
archived nine-panel regime asset (still preserved at
``figures/supplementary/figure_S04_panels_A-I.pdf``); that recomposition kept
the marks bit-identical but inherited the NeurIPS type and stroke scales and
carried no native geometry manifest. This builder redraws the same three
inherited panels and appends the completed feedback ladders natively:

* row 0 -- A the nominal-depth stress sweep and B the broadcast-noise sweep,
  six modules each (solid = local learning, faint dashed = matched
  backpropagation in A; whiskers are the +-1 s.d. the archived generator
  drew);
* row 1 -- C the flattened CIFAR-10 control ladder; D the independent
  raw-additive CIFAR-10 confirmation when its validated package is supplied;
  and E the Fashion-MNIST feedback-resolution replication.

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

The default build contains panels A--D (the three inherited panels and the
Fashion-MNIST replication). The canonical five-panel sheet can be rendered
only by explicitly supplying a finalized analysis directory
from ``analyze_cifar10_additive_feedback_ladder_confirmatory.py``.  Incomplete or internally inconsistent packages are rejected. A complete
convergence-flagged cohort requires an explicit fixed-budget display opt-in;
its original failed inferential decisions and seed-level flags are retained.
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

from journal_style import style_direct_color_labels
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
OUT = ROOT / "figures" / "supplementary" / "figure_S04_panels_A-D.pdf"
OUT_EXTENDED = ROOT / "figures" / "supplementary" / "figure_S04_panels_A-E.pdf"

DEPTH_CSV = SRC / "depth_scaling_summary.csv"
NOISE_CSV = SRC / "broadcast_noise_summary.csv"
CIFAR_CSV = SRC / "cifar10_control_ladder_runs.csv"
FASHION_SRC = ROOT / "source_data" / "fashion_strict_scalar_control"
FASHION_RUNS_CSV = FASHION_SRC / "seed_outcomes.csv"
FASHION_SUMMARY_CSV = FASHION_SRC / "condition_summary.csv"

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]

CORE_MARKER = {"dendritic_shunting": "o", "dendritic_additive": "s"}
CORE_COLOR = {"dendritic_shunting": SHUNT, "dendritic_additive": ADD}

FASHION_ORDER = ("strict scalar", "scalar fallback", "neuron indexed", "exact path")
FASHION_LABELS = ("strict\nscalar", "scalar\nfallback", "per\nneuron", "exact\npath")
FASHION_ARCHITECTURES = ("shunting", "additive")
FASHION_SEEDS = set(range(22600, 22610))


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
    ax.set_xticks([1, 2, 3, 4],["D1\n[9]","D2\n[3,3]","D3\n[3,3,3]","D4\n[3,3,3,3]"])
    ax.set_xlim(0.62, 4.38)
    ax.set_xlabel("physical depth; resources also change")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.15, 0.98)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    # Direct labels: the two local series separate at depth 4; the faint
    # dashed pair coincides near the top and gets one mute word.
    ax.text(2.5, 0.845, "matched BP", color=MUTE, fontsize=PT_SMALL,
            ha="center", va="top")
    ax.text(1.12, 0.575, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="bottom")
    ax.text(1.85, 0.38, "normalized additive", color=ADD, fontsize=PT_LEGEND,
            ha="center", va="top")
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
    ax.text(0.78, 0.34, "normalized additive", color=ADD, fontsize=PT_LEGEND,
            ha="center", va="bottom")
    return ax


# The ladder keeps the manuscript-wide condition hues: amber scalar, control
# gray for the random rank field, oracle violet for exact transport and the
# red-brown backpropagation reference.
# Tick wording matches panel E ("matched-width fallback") for the one
# condition the two panels share; marker shapes match panel D for exact path
# and BP.
CIFAR_SPECS = (
    ("cifar10_shunting_5f_per_soma_learned_i", "matched-\nwidth\nfallback",
     COLORS["local"], "o"),
    ("cifar10_shunting_5f_low_rank4_learned_i", "random\nrank 4",
     COLORS["point_mlp"], "o"),
    ("cifar10_shunting_5f_path_transport_learned_i", "exact\npath",
     COLORS["oracle"], "^"),
    ("cifar10_shunting_standard_learned_i", "BP", COLORS["bp"], "D"),
)
# Panels C and D draw the same quantity (CIFAR-10 test accuracy) and share
# one y window and tick set; every seed of both panels lies inside it
# (2026-09-11 visual review: C ran 0-56% and D 0-60% with the payload in an
# 11 px ribbon).  The window is asserted against the data, never widened
# silently.
CIFAR_Y_WINDOW = (0.245, 0.545)
CIFAR_Y_TICKS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

CONFIRMATORY_SPECS = (
    ("strict scalar", "strict\nscalar", COLORS["scalar"], "o"),
    ("neuron specific", "per\nneuron", COLORS["per_soma"], "s"),
    ("exact path", "exact\npath", COLORS["oracle"], "^"),
    ("backpropagation", "BP", COLORS["bp"], "D"),
)
CONFIRMATORY_CONTRASTS = {
    "neuron specific minus strict scalar",
    "exact path minus neuron specific",
    "exact path minus backpropagation",
}
EXPECTED_CONFIRMATORY_SEEDS = set(range(10800, 10820))


def _assert_in_window(values, where):
    low, high = CIFAR_Y_WINDOW
    if values.min() < low or values.max() > high:
        raise RuntimeError(
            f"{where}: seed accuracies {values.min():.4f}..{values.max():.4f} "
            f"leave the shared CIFAR window {CIFAR_Y_WINDOW}")


def _shared_cifar_axis(ax):
    ax.set_ylim(*CIFAR_Y_WINDOW)
    ax.set_yticks(list(CIFAR_Y_TICKS))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def panel_cifar(ax):
    """Flattened CIFAR-10 ladder: filled mean, +-1 s.d. bar, five open
    seed marks jittered across the category (same values as the archived
    bar chart; the bar fill is gone so the panel can share D's window)."""
    frame = pd.read_csv(CIFAR_CSV)
    for index, (condition, _label, color, marker) in enumerate(CIFAR_SPECS):
        values = frame[frame["condition"].eq(condition)
                       ]["test_accuracy"].to_numpy(float)
        mean = values.mean()
        std = values.std(ddof=1)
        print(f"  C {condition}: mean {mean:.5f} sd {std:.5f} n {values.size}")
        _assert_in_window(values, f"C {condition}")
        # Five seeds in a column to the right of the mean and its bar, so
        # every seed stays countable and none hides behind the mean mark.
        ax.scatter(np.full(values.size, index)
                   + np.linspace(0.08, 0.30, values.size), values,
                   s=8.0, facecolor="white", edgecolor=color,
                   linewidth=0.55, alpha=0.78, zorder=3)
        ax.errorbar(index - 0.12, mean, yerr=std, color=color, marker=marker,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.45, ms=MARKER_MS + 0.5,
                    lw=LW_DATA, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                    zorder=5)
    ax.set_xticks(range(len(CIFAR_SPECS)))
    ax.set_xticklabels([label for _, label, *_ in CIFAR_SPECS], fontsize=PT_SMALL)
    ax.set_xlim(-0.62, len(CIFAR_SPECS) - 0.38)
    _shared_cifar_axis(ax)
    ax.set_ylabel("CIFAR-10 accuracy")
    return ax


def load_confirmatory_analysis(analysis_dir: Path, *, architecture="additive", allow_convergence_flags=False) -> tuple[pd.DataFrame, dict]:
    """Load one complete, analyzer-validated 80-run confirmatory package.

    This is stricter than a plotting convenience function.  It verifies the
    analyzer decision record, the four paired 20-seed arms, and the exported
    condition/contrast summaries before any marks are drawn.  The scientific
    promotion gates are intentionally *not* required: a complete negative result may still be shown. Convergence-flagged results
    additionally require explicit disclosure and consistent seed-level flags.
    """
    expected_seeds = {"additive": set(range(10800, 10820)),
                      "shunting": set(range(22000, 22020))}[architecture]
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
    from cifar_display_validation import validate_protocol_scope
    try:
        validate_protocol_scope(summary)
    except ValueError as exc:
        errors.append(str(exc))
    if audit.get("status") not in {"complete_and_validated", "complete_with_convergence_flags"}:
        errors.append(f"audit status is {audit.get('status')!r}")
    if audit.get("integrity_valid") is not True:
        errors.append("integrity audit did not pass")
    if audit.get("n_expected") != 80 or audit.get("n_results_complete") != 80:
        errors.append(
            "audit does not record exactly 80/80 complete results "
            f"({audit.get('n_results_complete')!r}/{audit.get('n_expected')!r})"
        )

    outcomes = pd.read_csv(required["outcomes"])
    from cifar_display_validation import validate_convergence_disclosure
    try:
        validate_convergence_disclosure(summary, outcomes, allow_convergence_flags=allow_convergence_flags)
    except ValueError as exc:
        errors.append(str(exc))
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
        if seed_sets and seed_sets[0] != expected_seeds:
            errors.append(f"the paired seed set is not the frozen {architecture} set")
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
    """Fresh paired CIFAR-10 ladder with mean and 95% CI."""
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
    _assert_in_window(wide.to_numpy(float), "D")
    _shared_cifar_axis(ax)
    # C, directly to the left, carries the one y-axis title for the pair;
    # D keeps the shared tick labels.
    bp_mean = float(wide.iloc[:, -1].mean())
    ax.axhspan(bp_mean - .01, bp_mean + .01, color=MUTE, alpha=.12, zorder=0)
    # The prespecified +-1 pp BP equivalence band is keyed by one short
    # label on the band's empty left end (the strict-scalar column sits
    # 15 pp below it); the seed count and the interval are in the caption.
    ax.text(-0.36, bp_mean + 0.0125, "BP ±1 pp", ha="left", va="bottom",
            fontsize=PT_SMALL, color=MUTE, zorder=2)
    return ax


def load_fashion_ladder() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and verify the complete paired Fashion-MNIST replication."""
    missing = [
        str(path) for path in (FASHION_RUNS_CSV, FASHION_SUMMARY_CSV)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Fashion-MNIST S4 panel requires the frozen source-data tables; "
            "missing: " + ", ".join(missing)
        )

    outcomes = pd.read_csv(FASHION_RUNS_CSV)
    summary = pd.read_csv(FASHION_SUMMARY_CSV)
    errors: list[str] = []
    outcome_columns = {"architecture", "feedback", "seed", "test_accuracy"}
    summary_columns = {
        "architecture", "feedback", "n_seeds", "mean_accuracy",
        "ci95_low", "ci95_high",
    }
    if not outcome_columns.issubset(outcomes.columns):
        errors.append("seed_outcomes.csv lacks required columns")
    if not summary_columns.issubset(summary.columns):
        errors.append("condition_summary.csv lacks required columns")

    if outcome_columns.issubset(outcomes.columns):
        if len(outcomes) != 80:
            errors.append(f"seed_outcomes.csv has {len(outcomes)} rather than 80 rows")
        if outcomes.duplicated(["architecture", "feedback", "seed"]).any():
            errors.append("seed_outcomes.csv contains duplicate paired outcomes")
        if set(outcomes["architecture"].astype(str)) != set(FASHION_ARCHITECTURES):
            errors.append("seed_outcomes.csv has the wrong architecture inventory")
        if set(outcomes["feedback"].astype(str)) != set(FASHION_ORDER):
            errors.append("seed_outcomes.csv has the wrong feedback inventory")
        for architecture in FASHION_ARCHITECTURES:
            for feedback in FASHION_ORDER:
                seeds = set(outcomes.loc[
                    outcomes["architecture"].eq(architecture)
                    & outcomes["feedback"].eq(feedback), "seed"
                ])
                if seeds != FASHION_SEEDS:
                    errors.append(
                        f"{architecture}/{feedback} does not contain the frozen "
                        "paired seed set 22600--22609"
                    )
        values = pd.to_numeric(outcomes["test_accuracy"], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0.0, 1.0).all():
            errors.append("Fashion-MNIST accuracies are non-finite or outside [0,1]")

    if summary_columns.issubset(summary.columns) and outcome_columns.issubset(outcomes.columns):
        indexed = summary.set_index(["architecture", "feedback"])
        expected = {
            (architecture, feedback)
            for architecture in FASHION_ARCHITECTURES
            for feedback in FASHION_ORDER
        }
        if set(indexed.index) != expected:
            errors.append("condition_summary.csv has the wrong condition inventory")
        else:
            for key in sorted(expected):
                architecture, feedback = key
                values = outcomes.loc[
                    outcomes["architecture"].eq(architecture)
                    & outcomes["feedback"].eq(feedback), "test_accuracy"
                ].to_numpy(float)
                row = indexed.loc[key]
                if int(row["n_seeds"]) != values.size:
                    errors.append(f"{architecture}/{feedback}: summary n disagrees")
                if not np.isclose(
                    float(row["mean_accuracy"]), float(values.mean()),
                    rtol=0.0, atol=1e-12,
                ):
                    errors.append(f"{architecture}/{feedback}: summary mean disagrees")
                if not (
                    float(row["ci95_low"]) <= float(row["mean_accuracy"])
                    <= float(row["ci95_high"])
                ):
                    errors.append(f"{architecture}/{feedback}: invalid 95% interval")

    if errors:
        raise RuntimeError("Fashion-MNIST S4 source-data validation failed: "
                           + "; ".join(errors))
    return outcomes, summary


def panel_fashion(ax, outcomes: pd.DataFrame, summary: pd.DataFrame):
    """Paired Fashion-MNIST feedback ladder with archived 95% intervals."""
    x_base = np.arange(len(FASHION_ORDER), dtype=float)
    offsets = {"shunting": -0.07, "additive": 0.07}
    colors = {"shunting": SHUNT, "additive": ADD}
    markers = {"shunting": "o", "additive": "s"}
    names = {"shunting": "shunting", "additive": "raw additive"}

    for architecture in FASHION_ARCHITECTURES:
        wide = (
            outcomes[outcomes["architecture"].eq(architecture)]
            .pivot(index="seed", columns="feedback", values="test_accuracy")
            .loc[sorted(FASHION_SEEDS), list(FASHION_ORDER)]
        )
        seed_jitter = np.linspace(-0.012, 0.012, len(wide))
        for (_, values), jitter_x in zip(wide.iterrows(), seed_jitter, strict=True):
            ax.plot(
                x_base + offsets[architecture] + jitter_x,
                values.to_numpy(float),
                color=colors[architecture], marker=markers[architecture],
                markerfacecolor="white", markeredgewidth=0.35,
                ms=2.2, lw=0.48, alpha=0.24, zorder=2,
            )

        part = (
            summary[summary["architecture"].eq(architecture)]
            .set_index("feedback").loc[list(FASHION_ORDER)]
        )
        mean = part["mean_accuracy"].to_numpy(float)
        interval = np.vstack([
            mean - part["ci95_low"].to_numpy(float),
            part["ci95_high"].to_numpy(float) - mean,
        ])
        # Filled = cohort mean, small open = seed, as in panels C and D.
        ax.errorbar(
            x_base + offsets[architecture], mean, yerr=interval,
            color=colors[architecture], marker=markers[architecture],
            markerfacecolor=colors[architecture], markeredgecolor="white",
            markeredgewidth=0.45, ms=MARKER_MS + 0.5, lw=LW_DATA,
            elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5,
        )
        print(
            f"  E {names[architecture]}: means "
            f"{list(np.round(mean, 5))} n {len(wide)}"
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(FASHION_LABELS)
    ax.set_xlim(-0.36, 3.36)
    ax.set_ylim(0.80, 0.895)
    ax.set_yticks([0.80, 0.84, 0.88])
    ax.set_ylabel("Fashion-MNIST accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.text(3.28, 0.842, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="right", va="center")
    ax.text(3.28, 0.8245, "raw additive", color=ADD, fontsize=PT_LEGEND,
            ha="right", va="center")
    return ax


CANVAS_H_PT = 358.0                     # 518.4 / 358.0 = 1.45 aspect


def build(path=None, *, confirmatory_analysis_dir=None, shunting_analysis_dir=None, allow_convergence_flags=False):
    confirmatory = None
    if confirmatory_analysis_dir is not None:
        confirmatory, _summary = load_confirmatory_analysis(
            Path(confirmatory_analysis_dir)
        )
    shunting = None
    if shunting_analysis_dir is not None:
        if confirmatory is None:
            raise ValueError("A shunting update requires the additive cohort as well")
        shunting, _ = load_confirmatory_analysis(
            Path(shunting_analysis_dir), architecture="shunting",
            allow_convergence_flags=allow_convergence_flags)
    fashion_outcomes, fashion_summary = load_fashion_ladder()
    # A 38 pt gutter holds every y tick column plus its label outright, so
    # no column lock is carved and the middle panel of row 1 keeps the same
    # width as its neighbours (a 32 pt gutter left it 4.5 pt narrower).
    canvas = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=[1.0, 1.0],
                          hgutter_pt=38.0, vgutter_pt=30.0,
                          margins=Margins(left=46.0, right=12.0, top=14.0,
                                          bottom=26.0))
    # With no lock carved, a y tick column plus its rotated title reaches
    # 28.5 pt left of the axes; one figure-wide 30 pt letter offset keeps
    # every letter the leftmost mark of its own panel, which the supplement
    # paste step relies on to partition the row at each letter.
    canvas.letter_dx = 30.0
    ax_a = canvas.panel("depth", 0, 0, 6, grid="y", title="Depth stress")
    ax_b = canvas.panel("noise", 0, 6, 6, grid="y",
                        title="Noisy teaching signal")
    if confirmatory is None:
        ax_c = canvas.panel("cifar", 1, 0, 6, grid="y",
                            title="Harder-data control")
        ax_e = canvas.panel("fashion", 1, 6, 6, grid="y",
                            title="Fashion-MNIST replication")
    else:
        ax_c = canvas.panel("cifar", 1, 0, 4, grid="y",
                            title="Shunting CIFAR-10")
        ax_d = canvas.panel("cifar_confirmatory", 1, 4, 4, grid="y",
                            title="Raw-additive CIFAR-10")
        ax_e = canvas.panel("fashion", 1, 8, 4, grid="y",
                            title="Fashion-MNIST")
    panel_depth(ax_a)
    panel_noise(ax_b)
    if shunting is None:
        panel_cifar(ax_c)
    else:
        panel_confirmatory_cifar(ax_c, shunting)
        ax_c.set_ylabel("CIFAR-10 accuracy")
    if confirmatory is not None:
        panel_confirmatory_cifar(ax_d, confirmatory)
    panel_fashion(ax_e, fashion_outcomes, fashion_summary)
    target = Path(path) if path else (OUT_EXTENDED if confirmatory is not None else OUT)
    asset_name = (
        "figure_S04_panels_A-E" if confirmatory is not None
        else "figure_S04_panels_A-D"
    )
    style_direct_color_labels(canvas.fig)
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
            "when supplied, validates 80/80 runs and emits panels A-E"
        ),
    )
    parser.add_argument("--shunting-analysis-dir", type=Path,
                        help="Validated 20-seed shunting cohort replacing the historical ladder")
    parser.add_argument("--allow-convergence-flags", action="store_true",
                        help="Display complete fixed-budget data with disclosed convergence flags")
    args = parser.parse_args()
    raise SystemExit(
        1
        if build(
            args.output,
            confirmatory_analysis_dir=args.confirmatory_analysis_dir,
            shunting_analysis_dir=args.shunting_analysis_dir,
            allow_convergence_flags=args.allow_convergence_flags,
        )
        else 0
    )
