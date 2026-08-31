#!/usr/bin/env python3
"""Supplementary Fig. S3 -- rule and feedback controls -- as ONE native canvas.

Rebuilds the inherited NeurIPS-era sheet ``figure_S03_panels_A-D.pdf``
(archived verbatim under ``figures/supplementary/inherited/``) in the journal's
native visual system: one :class:`figure_canvas.NativeCanvas` at exactly
518.4 pt wide, the journal type/stroke tokens, the CVD-validated house palette,
and direct in-panel labels instead of boxed legends.

Panels, letters and every plotted value are FROZEN to the archived generator
``scripts/inherited_neurips/generate_neurips_figures.py``
(``figure_rule_feedback_design``); only layout, spacing, colours and label
wording change:

* **A** -- the three-, four- and five-factor rules in the archived three-seed
  within-shunting scalar-feedback cohort.  Aggregation ported verbatim from
  ``_submitted_matched_rule_values``: the nominal base-LR axis of
  ``core_fair_tuning.csv`` is inert, verified exactly, then collapsed.
* **B** -- somatic teaching error versus the unmatched compartment-local
  quantity, three local-decoder runs per cell of
  ``local_mismatch_recheck_runs.csv`` (mean and ddof=1 s.d., as inherited).
* **C** -- the five-seed exact-transport factorial
  (``source_data/figure2/exact_transport_factorial_summary.csv``) against
  matched backpropagation (``backprop_summary.csv``); the main text cites the
  3F local-decoder bar (97.18%) against the matched-BP reference (97.03%).
* **D** -- the noise-resilience feedback ladder
  (``noise_resilience_rank_bridge_summary.csv``): scalar matched-width,
  path-propagated, random low-rank (K2/K4/K8) and transported-oracle fields.
  The shunting noise-task cells keep the excluded signed-input convention on
  display, exactly as the archived asset does; the caption carries the caveat.

House palette semantics: rule hues keep their dedicated ``rule_3f/4f/5f``
slots; shunting is green and additive blue as in every main figure; the
exact-transport family wears the oracle violet (two lightness steps for the
two decoders); matched backpropagation is the red-brown ``bp`` reference;
the legacy matched-width/scalar-fallback field is amber; the random low-rank
fields are gray controls at three lightness steps.

Emits to the canonical ``figures/supplementary/figure_S03_panels_A-D.pdf``
(the ``inherited/`` copy preserves the original).  Every plotted endpoint is
printed at build time for numeric self-verification against the frozen CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    LW_ERR,
    Margins,
    NativeCanvas,
)

ROOT = SCRIPT_DIR.parent
DATA_INHERITED = ROOT / "source_data" / "inherited_neurips"
DATA_FIG2 = ROOT / "source_data" / "figure2"
TARGET = ROOT / "figures" / "supplementary" / "figure_S03_panels_A-D.pdf"

CORE_FAIR_TUNING_CSV = DATA_INHERITED / "core_fair_tuning.csv"
LOCAL_MISMATCH_RUNS_CSV = DATA_INHERITED / "local_mismatch_recheck_runs.csv"
EXACT_TRANSPORT_CSV = DATA_FIG2 / "exact_transport_factorial_summary.csv"
BACKPROP_CSV = DATA_FIG2 / "backprop_summary.csv"
RANK_BRIDGE_NOISE_CSV = DATA_INHERITED / "noise_resilience_rank_bridge_summary.csv"

INK = COLORS["ink"]
MUTE = COLORS["mute"]


def _lighten(hex_color: str, amount: float) -> str:
    """Blend a hex colour toward white by ``amount`` (0..1)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


ORACLE = COLORS["oracle"]                 # exact transport = the oracle violet
ORACLE_LIGHT = _lighten(ORACLE, 0.45)     # its local-decoder lightness step
GRAY_RAMP = ("#A9A9A9", "#8A8A8A", COLORS["point_mlp"])  # K2, K4, K8 controls

BAR_KW = dict(edgecolor="white", lw=LW_HAIR, zorder=3)
ERR_KW = dict(ecolor=INK, capsize=ERR_CAPSIZE,
              error_kw={"elinewidth": LW_ERR, "capthick": LW_ERR, "zorder": 4})


# ── frozen aggregations (ported verbatim from the archived generator) ─────
def submitted_matched_rule_values():
    """Panel A table: within-shunting matched-rule cohort, duplicates verified.

    Ported verbatim from ``_submitted_matched_rule_values`` in the frozen
    generator: the archived sweep nominally varied the base ``param_groups.lr``
    but every effective parameter-group rate was fixed, so the exported rows
    are identical along that axis; verify exactly, then collapse.
    """
    df = pd.read_csv(CORE_FAIR_TUNING_CSV)
    sub = df[
        (df["dataset"] == "mnist")
        & (df["error_broadcast_mode"] == "per_soma")
        & (df["decoder_update_mode"] == "local")
        & (df["rule_variant"].isin(["3f", "4f", "5f"]))
        & (df["network_type"] == "dendritic_shunting")
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No submitted matched-rule rows found in {CORE_FAIR_TUNING_CSV}")
    data, errors = {}, {}
    for rule in ("3f", "4f", "5f"):
        cell = sub[sub["rule_variant"] == rule]
        if cell.empty:
            raise ValueError(f"Missing matched-rule cell for {rule}")
        means = cell["test_accuracy_mean"].to_numpy(dtype=float)
        stds = cell["test_accuracy_std"].to_numpy(dtype=float)
        if not np.allclose(means, means[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Nominal base-LR duplicates disagree for {rule}: "
                f"{means.tolist()}")
        if not np.allclose(stds, stds[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Nominal base-LR uncertainty duplicates disagree for {rule}: "
                f"{stds.tolist()}")
        key = rule.upper()
        data[key] = 100.0 * float(means[0])
        errors[key] = 100.0 * float(stds[0])
    return data, errors


def error_source_values():
    """Panel B table: three local-decoder runs per (core, broadcast) cell."""
    mismatch = pd.read_csv(LOCAL_MISMATCH_RUNS_CSV)
    out = {}
    for core in ("dendritic_shunting", "dendritic_additive"):
        means, stds = [], []
        for mode in ("per_soma", "local_mismatch"):
            vals = mismatch[
                (mismatch["network_type"] == core)
                & (mismatch["error_broadcast_mode"] == mode)
                & (mismatch["decoder_update_mode"] == "local")
            ]["test_accuracy"].to_numpy(dtype=float)
            if len(vals) != 3:
                raise ValueError(
                    f"Expected three local-decoder runs for {core}, {mode}; "
                    f"found {len(vals)}")
            means.append(100.0 * float(np.mean(vals)))
            stds.append(100.0 * float(np.std(vals, ddof=1)))
        out[core] = (means, stds)
    return out


def exact_transport_values():
    """Panel C table: the 2x2 rule/decoder factorial and the matched-BP mean."""
    exact = pd.read_csv(EXACT_TRANSPORT_CSV)
    bp = pd.read_csv(BACKPROP_CSV).iloc[0]
    cells = {}
    for decoder_mode in ("backprop", "local"):
        means, stds = [], []
        for rule in ("3f", "5f"):
            row = exact[
                (exact["rule_variant"] == rule)
                & (exact["decoder_update_mode"] == decoder_mode)
            ].iloc[0]
            means.append(100.0 * float(row["test_acc_mean"]))
            stds.append(100.0 * float(row["test_acc_std"]))
        cells[decoder_mode] = (means, stds)
    return cells, 100.0 * float(bp["test_acc_mean"])


# Same six ladder cells, same order and same filters as the frozen generator
# (the rank-1 row of the CSV is excluded there too).  Colour is the only
# change: amber for matched-width scalar-fallback feedback, the per-soma
# salmon for the path-propagated field, a gray control ramp for the random
# low-rank fields and the oracle violet for exact transport.
LADDER_ROWS = (
    ("scalar\nfallback", "per_soma", 4, False, COLORS["scalar"]),
    ("fallback\n+ path", "per_soma", 4, True, COLORS["per_soma"]),
    ("rank 2", "low_rank", 2, False, GRAY_RAMP[0]),
    ("rank 4", "low_rank", 4, False, GRAY_RAMP[1]),
    ("rank 8", "low_rank", 8, False, GRAY_RAMP[2]),
    ("oracle", "path_transport", 4, False, ORACLE),
)


def noise_ladder_values():
    """Panel D table: the noise-resilience rank/propagation ladder."""
    rank_noise = pd.read_csv(RANK_BRIDGE_NOISE_CSV)
    vals, errs, labels, colors = [], [], [], []
    for label, mode, rank, path_prop, color in LADDER_ROWS:
        row = rank_noise[
            (rank_noise["broadcast_mode"] == mode)
            & (rank_noise["broadcast_rank"] == rank)
            & (rank_noise["use_path_propagation"] == path_prop)
        ]
        if len(row) == 0:
            continue
        labels.append(label)
        vals.append(float(row.iloc[0]["test_accuracy_mean"]) * 100)
        errs.append(float(row.iloc[0]["test_accuracy_std"]) * 100)
        colors.append(color)
    return labels, vals, errs, colors


# ── panels ────────────────────────────────────────────────────────────────
def panel_rule_family(ax):
    """A -- 3F/4F/5F rules, archived three-seed within-shunting cohort."""
    data, errors = submitted_matched_rule_values()
    rules = ["3F", "4F", "5F"]
    colors = [COLORS["rule_3f"], COLORS["rule_4f"], COLORS["rule_5f"]]
    x = np.arange(len(rules), dtype=float)
    vals = [data[r] for r in rules]
    errs = [errors[r] for r in rules]
    ax.bar(x, vals, 0.62, yerr=errs, color=colors, **BAR_KW, **ERR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(84, 94)
    ax.set_yticks([84, 86, 88, 90, 92, 94])
    ax.set_ylabel("MNIST accuracy (%)")
    print("  [A] rule family (mean, sd):",
          {r: (round(v, 4), round(e, 4)) for r, v, e in zip(rules, vals, errs)})
    return ax


def panel_error_source(ax):
    """B -- somatic teaching error vs the unmatched compartment-local field."""
    table = error_source_values()
    x = np.arange(2, dtype=float)
    bw = 0.30
    specs = [
        ("dendritic_shunting", "shunting", COLORS["shunting"]),
        ("dendritic_additive", "normalized additive", COLORS["additive"]),
    ]
    for j, (core, label, color) in enumerate(specs):
        means, stds = table[core]
        ax.bar(x + (j - 0.5) * bw, means, bw * 0.92, yerr=stds, color=color,
               **BAR_KW, **ERR_KW)
        print(f"  [B] {label} (mean, sd) per (somatic, mismatch):",
              [(round(m, 4), round(s, 4)) for m, s in zip(means, stds)])
    ax.set_xticks(x)
    ax.set_xticklabels(["somatic error", "local mismatch"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("MNIST accuracy (%)")
    # Direct labels in the empty upper-right region replace the boxed legend;
    # the hues match the architecture semantics of every main figure.
    ax.text(0.97, 0.94, "shunting", color=COLORS["shunting"],
            fontsize=PT_LEGEND, ha="right", va="center",
            transform=ax.transAxes)
    ax.text(0.97, 0.85, "normalized additive", color=COLORS["additive"],
            fontsize=PT_LEGEND, ha="right", va="center",
            transform=ax.transAxes)
    return ax


def panel_exact_transport(ax):
    """C -- five-seed exact-transport factorial vs matched backpropagation."""
    cells, bp_mean = exact_transport_values()
    x = np.arange(2, dtype=float)
    bw = 0.28
    specs = [
        ("backprop", "BP decoder", ORACLE),
        ("local", "local decoder", ORACLE_LIGHT),
    ]
    for j, (decoder_mode, label, color) in enumerate(specs):
        means, stds = cells[decoder_mode]
        ax.bar(x + (j - 0.5) * bw, means, bw * 0.92, yerr=stds, color=color,
               **BAR_KW, **ERR_KW)
        print(f"  [C] {label} (mean, sd) per (3F, 5F):",
              [(round(m, 4), round(s, 4)) for m, s in zip(means, stds)])
    ax.axhline(bp_mean, color=COLORS["bp"], lw=LW_REF, ls="--", zorder=2)
    print(f"  [C] matched BP reference: {bp_mean:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(["3F", "5F"])
    ax.set_xlim(-0.62, 2.08)
    ax.set_ylim(96.7, 97.65)
    ax.set_yticks([96.8, 97.0, 97.2, 97.4, 97.6])
    ax.set_ylabel("MNIST accuracy (%)")
    # Direct labels: the two decoder shades in the free band above the bars,
    # the matched-BP reference named at the free right end of its own line.
    ax.text(0.04, 0.955, "BP decoder", color=ORACLE, fontsize=PT_LEGEND,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.04, 0.875, "local decoder", color=ORACLE_LIGHT,
            fontsize=PT_LEGEND, ha="left", va="center", transform=ax.transAxes)
    ax.text(2.02, bp_mean + 0.022, "matched BP", color=COLORS["bp"],
            fontsize=PT_ANNOT, ha="right", va="bottom")
    return ax


def panel_feedback_ladder(ax):
    """D -- the noise-resilience feedback ladder (display convention kept)."""
    labels, vals, errs, colors = noise_ladder_values()
    xpos = np.arange(len(vals), dtype=float)
    ax.bar(xpos, vals, 0.68, yerr=errs, color=colors, **BAR_KW, **ERR_KW)
    print("  [D] ladder (mean, sd):",
          {l: (round(v, 4), round(e, 4))
           for l, v, e in zip(labels, vals, errs)})
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.62, 5.62)
    ax.set_ylim(35, 90)
    ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.set_ylabel("noise-task accuracy (%)")
    return ax


# ── the canvas ────────────────────────────────────────────────────────────
CANVAS_H_PT = 380.0                       # 518.4 / 380 = 1.36, in the band


def build(path: Path | str = TARGET):
    canvas = NativeCanvas(CANVAS_H_PT / 72.0, 2,
                          margins=Margins(left=36.5, right=12.0, top=16.0,
                                          bottom=26.0))
    ax_a = canvas.panel("rule_family", 0, 0, 5,
                        title="Rule family (shunting)", grid="y")
    ax_b = canvas.panel("error_source", 0, 5, 7,
                        title="Error source", grid="y")
    ax_c = canvas.panel("exact_transport", 1, 0, 5,
                        title="Exact-transport factorial", grid="y")
    ax_d = canvas.panel("feedback_ladder", 1, 5, 7,
                        title="Feedback ladder (noise task)", grid="y")

    print(f"Building {Path(path).name}")
    panel_rule_family(ax_a)
    panel_error_source(ax_b)
    panel_exact_transport(ax_c)
    panel_feedback_ladder(ax_d)

    problems = canvas.save(Path(path), name="figure_S03_panels_A-D",
                           png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)
