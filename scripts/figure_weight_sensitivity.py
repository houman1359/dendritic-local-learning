"""
Generate supplementary figure: sensitivity of weight distribution σ to architecture.

Panels:
  (A) σ vs depth (branch_factors) for 4 conditions
  (B) σ vs ie_synapses for shunting vs additive (ee=40)
  (C) σ bar chart: main 4 conditions + MICrONS + Song 2005 (clean 2×2 sweep)
  (D) E vs I weight σ comparison with MICrONS

Data sources:
  - sigma_vs_depth.json (depth_scaling_v2 sweep, 80 models)
  - sigma_vs_ei_synapses.json (ei_grid_pilot sweep, 168 models)
  - microns_ei_distributions.json (MICrONS connectome)
  - Clean 2×2 sweep results
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Biological references
SONG_2005_SIGMA = 0.9355
MICRONS_EE_SIGMA = 1.140
MICRONS_IE_SIGMA = 0.917
MICRONS_EI_SIGMA = 0.825
MICRONS_II_SIGMA = 0.847

# Style
COLORS = {
    "standard_dendritic_shunting": "#4477AA",
    "standard_dendritic_additive": "#66CCEE",
    "local_ca_dendritic_shunting": "#EE6677",
    "local_ca_dendritic_additive": "#CCBB44",
    "dendritic_shunting": "#228833",
    "dendritic_additive": "#AA3377",
}
LABELS = {
    "standard_dendritic_shunting": "BP + Shunting",
    "standard_dendritic_additive": "BP + Additive",
    "local_ca_dendritic_shunting": "Local + Shunting",
    "local_ca_dendritic_additive": "Local + Additive",
    "dendritic_shunting": "Shunting",
    "dendritic_additive": "Additive",
}

BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "weight_distributions"
)


def load_depth_data():
    path = os.path.join(BASE_DIR, "sigma_vs_depth.json")
    with open(path) as f:
        return json.load(f)


def load_ei_synapse_data():
    path = os.path.join(BASE_DIR, "sigma_vs_ei_synapses.json")
    with open(path) as f:
        return json.load(f)


def load_microns_data():
    path = os.path.join(BASE_DIR, "..", "..", "data", "microns_distributions",
                        "microns_ei_distributions.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    depth_data = load_depth_data()
    ei_data = load_ei_synapse_data()
    microns = load_microns_data()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.07, right=0.96, top=0.93, bottom=0.08)

    # ── Panel A: σ vs depth ──
    ax_a = fig.add_subplot(gs[0, 0])
    depth_labels = ["[9]", "[3,3]", "[3,3,3]", "[3,3,3,3]"]
    x_depth = np.arange(len(depth_labels))

    conditions = depth_data["conditions"]
    for cond_prefix in [
        "standard_dendritic_shunting",
        "standard_dendritic_additive",
        "local_ca_dendritic_shunting",
        "local_ca_dendritic_additive",
    ]:
        means, stds = [], []
        for bf_str in ["[9]", "[3, 3]", "[3, 3, 3]", "[3, 3, 3, 3]"]:
            key = f"{cond_prefix}_{bf_str}"
            if key in conditions:
                means.append(conditions[key]["sigma_exc_mean"])
                stds.append(conditions[key]["sigma_exc_std"])
            else:
                means.append(np.nan)
                stds.append(0)

        color = COLORS.get(cond_prefix, "gray")
        label = LABELS.get(cond_prefix, cond_prefix)
        ax_a.errorbar(x_depth, means, yerr=stds, marker="o", color=color,
                      label=label, capsize=3, linewidth=2, markersize=6)

    # Biological references
    ax_a.axhline(MICRONS_EE_SIGMA, color="gray", ls="--", lw=1.5, alpha=0.7,
                 label=f"MICrONS E→E (σ={MICRONS_EE_SIGMA:.2f})")
    ax_a.axhline(SONG_2005_SIGMA, color="black", ls=":", lw=1.5, alpha=0.7,
                 label=f"Song 2005 (σ={SONG_2005_SIGMA:.2f})")

    ax_a.set_xticks(x_depth)
    ax_a.set_xticklabels(depth_labels, fontsize=10)
    ax_a.set_xlabel("Branch factors (depth)", fontsize=11)
    ax_a.set_ylabel("Log-normal σ (excitatory)", fontsize=11)
    ax_a.set_title("A. Weight σ vs dendritic depth", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.set_ylim(0.5, 2.8)

    # ── Panel B: σ vs ie_synapses (ee=40, from ei_grid) ──
    ax_b = fig.add_subplot(gs[0, 1])
    ie_values = ei_data["dimensions"]["ie_synapses"]

    for core_type in ["dendritic_shunting", "dendritic_additive"]:
        means, stds = [], []
        for ie in ie_values:
            key = f"{core_type}_ee40_ie{ie}"
            entry = ei_data["excitatory_sigma"].get(key, {})
            if entry:
                means.append(entry["mean"])
                stds.append(entry["std"])
            else:
                means.append(np.nan)
                stds.append(0)

        color = COLORS.get(core_type, "gray")
        label = LABELS.get(core_type, core_type)
        ax_b.errorbar(ie_values, means, yerr=stds, marker="s", color=color,
                      label=f"{label} (ee=40)", capsize=3, linewidth=2, markersize=6)

    # Also plot ee=20 and ee=80 as lighter lines
    for ee, alpha in [(20, 0.4), (80, 0.4)]:
        for core_type in ["dendritic_shunting", "dendritic_additive"]:
            means = []
            for ie in ie_values:
                key = f"{core_type}_ee{ee}_ie{ie}"
                entry = ei_data["excitatory_sigma"].get(key, {})
                means.append(entry["mean"] if entry else np.nan)
            color = COLORS.get(core_type, "gray")
            label_ct = LABELS.get(core_type, core_type)
            ax_b.plot(ie_values, means, marker=".", color=color, alpha=alpha,
                      linewidth=1, markersize=4, label=f"{label_ct} (ee={ee})")

    ax_b.axhline(MICRONS_EE_SIGMA, color="gray", ls="--", lw=1.5, alpha=0.7,
                 label=f"MICrONS E→E (σ={MICRONS_EE_SIGMA:.2f})")
    ax_b.axhline(SONG_2005_SIGMA, color="black", ls=":", lw=1.5, alpha=0.7,
                 label=f"Song 2005 (σ={SONG_2005_SIGMA:.2f})")

    ax_b.set_xlabel("IE synapses per branch ($N_I$)", fontsize=11)
    ax_b.set_ylabel("Log-normal σ (excitatory)", fontsize=11)
    ax_b.set_title("B. Weight σ vs inhibitory synapse count", fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax_b.set_ylim(0.5, 3.0)

    # ── Panel C: Main bar chart (clean 2×2 + biology) ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Clean sweep data (from agent results)
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

    cond_order = [
        "standard_dendritic_shunting",
        "standard_dendritic_additive",
        "local_ca_dendritic_shunting",
        "local_ca_dendritic_additive",
    ]

    n_c = len(cond_order)
    x_c = np.arange(n_c)
    width = 0.35

    # MNIST bars
    mnist_means = [clean_mnist[c][0] for c in cond_order]
    mnist_errs = [clean_mnist[c][1] for c in cond_order]
    mnist_colors = [COLORS[c] for c in cond_order]

    bars1 = ax_c.bar(x_c - width/2, mnist_means, width, yerr=mnist_errs,
                     color=mnist_colors, alpha=0.9, capsize=3,
                     edgecolor="black", linewidth=0.5, label="MNIST")

    # Fashion-MNIST bars
    fmnist_means = [clean_fmnist[c][0] for c in cond_order]
    fmnist_errs = [clean_fmnist[c][1] for c in cond_order]

    bars2 = ax_c.bar(x_c + width/2, fmnist_means, width, yerr=fmnist_errs,
                     color=mnist_colors, alpha=0.5, capsize=3,
                     edgecolor="black", linewidth=0.5, hatch="//",
                     label="Fashion-MNIST")

    # Biology reference lines
    ax_c.axhline(MICRONS_EE_SIGMA, color="gray", ls="--", lw=2, alpha=0.7,
                 label=f"MICrONS E→E (σ={MICRONS_EE_SIGMA:.2f})")
    ax_c.axhline(SONG_2005_SIGMA, color="black", ls=":", lw=2, alpha=0.7,
                 label=f"Song 2005 (σ={SONG_2005_SIGMA:.2f})")

    cond_labels = [LABELS[c] for c in cond_order]
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=9)
    ax_c.set_ylabel("Log-normal σ (excitatory)", fontsize=11)
    ax_c.set_title("C. Weight σ: model vs biology", fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.set_ylim(0, 2.0)

    # ── Panel D: E vs I comparison ──
    ax_d = fig.add_subplot(gs[1, 1])

    # Clean sweep E and I sigma (MNIST)
    clean_mnist_inh = {
        "standard_dendritic_shunting": (0.874, 0.014),
        "standard_dendritic_additive": (1.215, 0.006),
        "local_ca_dendritic_shunting": (1.708, 0.020),
        "local_ca_dendritic_additive": (1.572, 0.065),
    }

    x_d = np.arange(n_c)
    width_d = 0.35

    exc_means = [clean_mnist[c][0] for c in cond_order]
    exc_errs = [clean_mnist[c][1] for c in cond_order]
    inh_means = [clean_mnist_inh[c][0] for c in cond_order]
    inh_errs = [clean_mnist_inh[c][1] for c in cond_order]

    ax_d.bar(x_d - width_d/2, exc_means, width_d, yerr=exc_errs,
             color="#EE6677", alpha=0.7, capsize=3, edgecolor="black",
             linewidth=0.5, label="Excitatory")
    ax_d.bar(x_d + width_d/2, inh_means, width_d, yerr=inh_errs,
             color="#4477AA", alpha=0.7, capsize=3, edgecolor="black",
             linewidth=0.5, label="Inhibitory")

    # MICrONS references
    ax_d.axhline(MICRONS_EE_SIGMA, color="#EE6677", ls="--", lw=1.5, alpha=0.6,
                 label=f"MICrONS E→E ({MICRONS_EE_SIGMA:.2f})")
    ax_d.axhline(MICRONS_IE_SIGMA, color="#4477AA", ls="--", lw=1.5, alpha=0.6,
                 label=f"MICrONS I→E ({MICRONS_IE_SIGMA:.2f})")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=9)
    ax_d.set_ylabel("Log-normal σ", fontsize=11)
    ax_d.set_title("D. Excitatory vs inhibitory weight σ", fontsize=12, fontweight="bold")
    ax_d.legend(fontsize=7.5, loc="upper left")
    ax_d.set_ylim(0, 2.0)

    fig.suptitle(
        "Synaptic Weight Distribution Analysis: Sensitivity to Architecture",
        fontsize=14, fontweight="bold",
    )

    out_dir = os.path.join(BASE_DIR, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_weight_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    # Also save to data dir for easy access
    out_path2 = os.path.join(BASE_DIR, "fig_weight_sensitivity.png")
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path2}")

    plt.close()


if __name__ == "__main__":
    main()
