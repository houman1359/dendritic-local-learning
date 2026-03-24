#!/usr/bin/env python3
"""Generate a publication-style appendix figure for weight-distribution sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import apply_neurips_style, COLORS as NEURIPS_COLORS, panel_label
apply_neurips_style()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
DATA_DIR = DRAFT_DIR / "data" / "weight_distributions"
FIGURES_DIR = DRAFT_DIR / "figures"

SONG_2005_SIGMA = 0.9355
MICRONS_EE_SIGMA = 1.140
MICRONS_IE_SIGMA = 0.917

COLORS = {
    "standard_dendritic_shunting": "#18864B",
    "standard_dendritic_additive": "#2D5DA8",
    "local_ca_dendritic_shunting": "#65B57D",
    "local_ca_dendritic_additive": "#82A4DA",
    "dendritic_shunting": "#18864B",
    "dendritic_additive": "#2D5DA8",
}
LABELS = {
    "standard_dendritic_shunting": "BP + shunting",
    "standard_dendritic_additive": "BP + additive",
    "local_ca_dendritic_shunting": "Local + shunting",
    "local_ca_dendritic_additive": "Local + additive",
    "dendritic_shunting": "Shunting",
    "dendritic_additive": "Additive",
}
DPI = 300


def _setup_style() -> None:
    # Unified style already applied at import time; this is a no-op kept for call-site compat.
    pass


def _panel(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.10) -> None:
    panel_label(ax, label, x=x, y=y)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}", dpi=DPI)
    print(f"Saved {stem}.{{png,pdf}}")


def build_figure() -> plt.Figure:
    _setup_style()
    depth_data = _load_json(DATA_DIR / "sigma_vs_depth.json")
    ei_data = _load_json(DATA_DIR / "sigma_vs_ei_synapses.json")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), gridspec_kw={"wspace": 0.38, "hspace": 0.52})

    cond_order = [
        "standard_dendritic_shunting",
        "standard_dendritic_additive",
        "local_ca_dendritic_shunting",
        "local_ca_dendritic_additive",
    ]

    # Panel A: sigma vs depth
    ax = axes[0, 0]
    _panel(ax, "A")
    depth_labels = ["[9]", "[3,3]", "[3,3,3]", "[3,3,3,3]"]
    x_depth = np.arange(len(depth_labels))
    conditions = depth_data["conditions"]
    markers = {
        "standard_dendritic_shunting": "s",
        "standard_dendritic_additive": "s",
        "local_ca_dendritic_shunting": "o",
        "local_ca_dendritic_additive": "o",
    }
    for cond in cond_order:
        means, stds = [], []
        for bf_str in ["[9]", "[3, 3]", "[3, 3, 3]", "[3, 3, 3, 3]"]:
            entry = conditions.get(f"{cond}_{bf_str}", {})
            means.append(entry.get("sigma_exc_mean", np.nan))
            stds.append(entry.get("sigma_exc_std", 0.0))
        ax.errorbar(
            x_depth,
            means,
            yerr=stds,
            marker=markers[cond],
            markersize=4.5,
            capsize=2,
            linewidth=1.3,
            color=COLORS[cond],
            label=LABELS[cond],
        )
    ax.axhline(MICRONS_EE_SIGMA, color="#777777", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(SONG_2005_SIGMA, color="#333333", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.text(3.05, MICRONS_EE_SIGMA + 0.03, "MICrONS E→E", fontsize=5.8, color="#666666")
    ax.text(3.05, SONG_2005_SIGMA + 0.03, "Song 2005", fontsize=5.8, color="#333333")
    ax.set_xticks(x_depth)
    ax.set_xticklabels(depth_labels)
    ax.set_ylabel(r"Excitatory log-normal width $\sigma$")
    ax.set_xlabel("Branch factors (depth)")
    ax.set_title(r"$\sigma$ vs.\ dendritic depth")
    ax.set_ylim(0.65, 2.65)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper left", ncol=2, fontsize=5.7, columnspacing=0.9, handletextpad=0.4)

    # Panel B: sigma vs inhibitory synapses
    ax = axes[0, 1]
    _panel(ax, "B")
    ie_values = np.array(ei_data["dimensions"]["ie_synapses"], dtype=float)
    for core_type in ["dendritic_shunting", "dendritic_additive"]:
        means, stds, lo_band, hi_band = [], [], [], []
        for ie in ie_values:
            entry = ei_data["excitatory_sigma"].get(f"{core_type}_ee40_ie{int(ie)}", {})
            means.append(entry.get("mean", np.nan))
            stds.append(entry.get("std", 0.0))
            lo_entry = ei_data["excitatory_sigma"].get(f"{core_type}_ee20_ie{int(ie)}", {})
            hi_entry = ei_data["excitatory_sigma"].get(f"{core_type}_ee80_ie{int(ie)}", {})
            lo_band.append(lo_entry.get("mean", np.nan))
            hi_band.append(hi_entry.get("mean", np.nan))
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        lo_band = np.array(lo_band, dtype=float)
        hi_band = np.array(hi_band, dtype=float)
        color = COLORS[core_type]
        ax.fill_between(
            ie_values,
            np.nanmin(np.vstack([lo_band, hi_band]), axis=0),
            np.nanmax(np.vstack([lo_band, hi_band]), axis=0),
            color=color,
            alpha=0.10,
        )
        ax.errorbar(
            ie_values,
            means,
            yerr=stds,
            marker="o",
            markersize=4.0,
            capsize=2,
            linewidth=1.4,
            color=color,
            label=LABELS[core_type],
        )
    ax.axhline(MICRONS_EE_SIGMA, color="#777777", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(SONG_2005_SIGMA, color="#333333", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.text(0.02, 0.05, "shaded band: $N_E=20$ to $80$", transform=ax.transAxes, fontsize=5.8, color="#666666")
    ax.set_xlabel(r"Inhibitory synapses per branch $N_I$")
    ax.set_ylabel(r"Excitatory log-normal width $\sigma$")
    ax.set_title(r"$\sigma$ vs.\ inhibitory synapse count")
    ax.set_ylim(0.65, 2.95)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper right")

    # Panel C: main summary across datasets
    ax = axes[1, 0]
    _panel(ax, "C")
    clean_mnist = {
        "standard_dendritic_shunting": (0.943, 0.007),
        "standard_dendritic_additive": (1.304, 0.005),
        "local_ca_dendritic_shunting": (1.215, 0.013),
        "local_ca_dendritic_additive": (1.686, 0.158),
    }
    clean_fmnist = {
        "standard_dendritic_shunting": (0.918, 0.009),
        "standard_dendritic_additive": (1.159, 0.010),
        "local_ca_dendritic_shunting": (1.133, 0.061),
        "local_ca_dendritic_additive": (1.430, 0.155),
    }
    x = np.arange(len(cond_order))
    width = 0.28
    mnist_vals = [clean_mnist[c][0] for c in cond_order]
    mnist_errs = [clean_mnist[c][1] for c in cond_order]
    fmnist_vals = [clean_fmnist[c][0] for c in cond_order]
    fmnist_errs = [clean_fmnist[c][1] for c in cond_order]
    for xpos, cond, m, s in zip(x - width / 2, cond_order, mnist_vals, mnist_errs):
        ax.bar(xpos, m, width, color=COLORS[cond], edgecolor="white", linewidth=0.4, alpha=0.95)
        ax.errorbar(xpos, m, yerr=s, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2)
    for xpos, cond, m, s in zip(x + width / 2, cond_order, fmnist_vals, fmnist_errs):
        ax.bar(xpos, m, width, color=COLORS[cond], edgecolor="white", linewidth=0.4, alpha=0.45)
        ax.errorbar(xpos, m, yerr=s, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2)
    ax.axhline(MICRONS_EE_SIGMA, color="#777777", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(SONG_2005_SIGMA, color="#333333", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in cond_order], rotation=25, ha="right")
    ax.set_ylabel(r"Excitatory log-normal width $\sigma$")
    ax.set_title("Summary across datasets")
    ax.set_ylim(0.65, 2.05)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    legend_handles = [
        plt.Line2D([0], [0], color="#555555", linewidth=6, alpha=0.95, label="MNIST"),
        plt.Line2D([0], [0], color="#555555", linewidth=6, alpha=0.45, label="Fashion-MNIST"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    # Panel D: excitatory vs inhibitory
    ax = axes[1, 1]
    _panel(ax, "D")
    clean_mnist_inh = {
        "standard_dendritic_shunting": (0.874, 0.014),
        "standard_dendritic_additive": (1.215, 0.006),
        "local_ca_dendritic_shunting": (1.708, 0.020),
        "local_ca_dendritic_additive": (1.572, 0.065),
    }
    exc_vals = [clean_mnist[c][0] for c in cond_order]
    exc_errs = [clean_mnist[c][1] for c in cond_order]
    inh_vals = [clean_mnist_inh[c][0] for c in cond_order]
    inh_errs = [clean_mnist_inh[c][1] for c in cond_order]
    x = np.arange(len(cond_order))
    width = 0.34
    ax.bar(x - width / 2, exc_vals, width, color="#D47D6A", edgecolor="white", linewidth=0.4, label="Excitatory")
    ax.bar(x + width / 2, inh_vals, width, color="#4C78A8", edgecolor="white", linewidth=0.4, label="Inhibitory")
    ax.errorbar(x - width / 2, exc_vals, yerr=exc_errs, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2)
    ax.errorbar(x + width / 2, inh_vals, yerr=inh_errs, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2)
    ax.axhline(MICRONS_EE_SIGMA, color="#D47D6A", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(MICRONS_IE_SIGMA, color="#4C78A8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in cond_order], rotation=25, ha="right")
    ax.set_ylabel(r"Log-normal width $\sigma$")
    ax.set_title(r"Excitatory vs inhibitory $\sigma$")
    ax.set_ylim(0.65, 2.05)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper left")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95)
    return fig


def main() -> int:
    fig = build_figure()
    _save(fig, "fig_weight_distributions")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
