#!/usr/bin/env python3
"""Render Supplementary Figure S11 from the frozen passive focal matrix."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "focal_selectivity_phase1"
FIGURES = ROOT / "figures" / "generated"


def main() -> None:
    apply_neurips_style()
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    rows = pd.read_csv(SOURCE / "site_outcomes.csv.gz")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 2.95),
        gridspec_kw={
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.20,
            "top": 0.84,
            "wspace": 0.55,
        },
    )
    absolute = contrasts[
        contrasts.dose_scheme.eq("fixed_absolute_ns")
        & contrasts.metric.eq("localization_index")
        & np.isclose(contrasts.background_leak_multiplier, 0.0)
    ]
    colors = {
        300.0: COLORS["shunting"],
        1000.0: COLORS["per_soma"],
        15000.0: COLORS["additive"],
    }
    for rm, part in absolute.groupby("membrane_resistance_ohm_cm2"):
        part = part.sort_values("dose_value")
        ax_a.plot(
            part.dose_value,
            part.mean_shunt_minus_additive,
            color=colors[float(rm)],
            marker="o",
            lw=LW_DATA,
            label=rf"$R_m={rm:g}$",
        )
    ax_a.set_xscale("log")
    ax_a.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_a.set_xlabel("fixed shunt conductance (nS)")
    ax_a.set_ylabel("shunt - additive localization")
    panel_title(ax_a, "A", "Fixed absolute dose")
    style_axis(ax_a)
    clean_legend(ax_a, fontsize=PT_LEGEND - 0.4, loc="best")

    selected = rows[
        rows.perturbation.eq("focal shunt")
        & rows.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(rows.dose_value, 1.0)
    ]
    sites = selected.groupby(
        ["root_id", "focal_segment_id", "membrane_resistance_ohm_cm2", "background_leak_multiplier"],
        as_index=False,
    )[["transport_selectivity", "localization_index"]].mean()
    ax_b.scatter(
        sites.transport_selectivity,
        sites.localization_index,
        s=9,
        color=COLORS["per_soma"],
        alpha=0.35,
        edgecolors="none",
    )
    ax_b.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_b.axvline(1, color=COLORS["mute"], ls=":", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"transport selectivity $S_k$")
    ax_b.set_ylabel("localization index")
    panel_title(ax_b, "B", "Cable selectivity")
    style_axis(ax_b)

    central = summary[
        summary.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(summary.dose_value, 1.0)
        & np.isclose(summary.membrane_resistance_ohm_cm2, 1000.0)
        & np.isclose(summary.background_leak_multiplier, 1.0)
        & summary.perturbation.eq("focal shunt")
    ].iloc[0]
    values = [
        central.mean_descendant_attenuated_fraction,
        central.mean_descendant_enhanced_fraction,
        central.mean_descendant_sign_flip_fraction,
    ]
    ax_c.bar(
        range(3),
        values,
        color=[COLORS["shunting"], COLORS["additive"], COLORS["mute"]],
        width=0.68,
    )
    ax_c.set_xticks(range(3), ["attenuated", "enhanced", "sign flip"], rotation=20, ha="right")
    ax_c.set_ylabel("descendant fraction")
    ax_c.set_ylim(0, 1)
    panel_title(ax_c, "C", "Signed outcomes")
    style_axis(ax_c, grid="y")

    fig.canvas.draw()
    audit_layout(fig, "fig_focal_selectivity_matrix")
    audit_text_over_data(fig, "fig_focal_selectivity_matrix")
    fig.savefig(
        FIGURES / "fig_focal_selectivity_matrix.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_focal_selectivity_matrix.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
