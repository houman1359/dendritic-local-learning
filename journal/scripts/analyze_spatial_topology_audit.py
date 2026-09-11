#!/usr/bin/env python3
"""Audit what differs between spatial and random fixed sparse input maps.

The registered spatial-topology experiment holds the number of active contacts
fixed, but the spatial sampler partitions the image into disjoint leaf regions.
Consequently, its branches can cover more unique input coordinates than rows
sampled independently from the whole image.  This script quantifies that
forward-connectivity difference and places it beside the completed learning
contrast.  It does not reinterpret that contrast as a credit-routing result.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import style_direct_color_labels
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (  # noqa: E402
    _sample_indices_from_mask,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.spatial_morphology import (  # noqa: E402
    sample_spatial_morphology_indices,
)
from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    PT_LEGEND,
    PT_SMALL,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


OUT = ROOT / "source_data" / "spatial_topology_audit"
FIGURES = ROOT / "figures" / "generated"
FOLLOWUP = ROOT / "source_data" / "prospective_followup"
# Published per-seed outcomes of the same input-valid spatial cohort; the
# plotted means are reproduced from it before any seed point is drawn.
SEED_OUTCOMES = (
    ROOT / "source_data" / "prospective_input_validity"
    / "followup_publication_seed_outcomes.csv"
)

# Panel geometry in points: three 160-pt panel columns (the supplement pastes
# three per row at scale 1.0) and one common axes height for every plot panel
# so the pasted rows share a baseline.
FIG_H = 4.55
AX_W = 118.0
AX_H = 96.0
COL_X0 = (42.0, 218.0, 394.0)
ROW_Y_TOP = (38.0, 200.0)

N_OWNERS = 128
BRANCH_FACTORS = (2, 2, 2, 2)
BRANCHES_PER_OWNER = int(np.prod(BRANCH_FACTORS))
INPUT_DIM = 28 * 28
CONTACTS_PER_BRANCH = 21
SEEDS = tuple(range(42, 52))

apply_neurips_style()


def _indices(seed: int, topology: str) -> torch.Tensor:
    if topology == "spatial":
        return sample_spatial_morphology_indices(
            input_shape=(1, 28, 28),
            out_features=N_OWNERS * BRANCHES_PER_OWNER,
            in_features=INPUT_DIM,
            synapses_per_branch=CONTACTS_PER_BRANCH,
            owner_count=N_OWNERS,
            branch_factors=BRANCH_FACTORS,
            level_idx=0,
            split_axes=("height", "width", "height", "width"),
            seed=seed,
            index_dtype="int64",
        ).long()
    if topology == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return _sample_indices_from_mask(
            out_features=N_OWNERS * BRANCHES_PER_OWNER,
            in_features=INPUT_DIM,
            K=CONTACTS_PER_BRANCH,
            connection_mask=None,
            generator=generator,
            index_dtype="int64",
        ).long()
    raise ValueError(topology)


def connectivity_metrics() -> pd.DataFrame:
    owner_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for topology in ("random", "spatial"):
            indices = _indices(seed, topology).reshape(
                N_OWNERS, BRANCHES_PER_OWNER, CONTACTS_PER_BRANCH
            )
            for owner, block in enumerate(indices):
                flat = block.flatten()
                unique = int(torch.unique(flat).numel())
                pairwise_overlap = []
                for first, second in itertools.combinations(range(BRANCHES_PER_OWNER), 2):
                    a = set(block[first].tolist())
                    b = set(block[second].tolist())
                    pairwise_overlap.append(len(a & b) / len(a | b))
                degree = torch.bincount(flat, minlength=INPUT_DIM).numpy()
                owner_rows.append(
                    {
                        "seed": seed,
                        "topology": topology,
                        "owner": owner,
                        "active_contacts": int(flat.numel()),
                        "unique_input_features": unique,
                        "input_coverage_fraction": unique / INPUT_DIM,
                        "repeated_contact_fraction": 1.0 - unique / flat.numel(),
                        "mean_pairwise_branch_jaccard": float(np.mean(pairwise_overlap)),
                        "feature_degree_cv": float(degree.std() / degree.mean()),
                    }
                )
    return pd.DataFrame(owner_rows)


def summarize(owner: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "unique_input_features",
        "input_coverage_fraction",
        "repeated_contact_fraction",
        "mean_pairwise_branch_jaccard",
        "feature_degree_cv",
    ]
    rows = []
    for topology, part in owner.groupby("topology"):
        for metric in metrics:
            values = part[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "topology": topology,
                    "metric": metric,
                    "n_owner_maps": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def performance_effects() -> pd.DataFrame:
    path = FOLLOWUP / "paired_contrasts.csv"
    contrasts = pd.read_csv(path)
    part = contrasts[
        (contrasts.study == "spatial")
        & (contrasts.contrast == "spatial - random")
        & ~(
            contrasts.task.eq("noise_resilience")
            & contrasts.core.eq("dendritic_shunting")
        )
    ].copy()
    if len(part) != 12:
        raise SystemExit(f"Expected 12 input-valid spatial contrasts, found {len(part)}")
    return part


def task_feedback_effects() -> pd.DataFrame:
    """Average the two neuron-model contrasts within each paired seed."""
    runs = pd.read_csv(FOLLOWUP / "seed_outcomes.csv")
    part = runs[runs.run_dir.str.contains("spatial_topology", na=False)].copy()
    part = part[
        ~(
            part.task.eq("noise_resilience")
            & part.core.eq("dendritic_shunting")
        )
    ]
    wide = part.pivot(
        index=["task", "core", "strategy", "feedback", "seed"],
        columns="topology",
        values="test_accuracy",
    ).reset_index()
    if set(wide.columns) < {"random", "spatial"}:
        raise SystemExit("Spatial run table lacks random or spatial conditions")
    wide["difference"] = wide.spatial - wide.random
    paired_seed = (
        wide.groupby(["task", "strategy", "feedback", "seed"], as_index=False)
        .difference.mean()
    )
    rows = []
    for values, group in paired_seed.groupby(
        ["task", "strategy", "feedback"], sort=True
    ):
        observed = group.difference.to_numpy(dtype=float)
        rng = np.random.default_rng(700 + len(rows))
        means = rng.choice(observed, size=(20_000, len(observed)), replace=True).mean(1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        rows.append(
            {
                "task": values[0],
                "strategy": values[1],
                "feedback": values[2],
                "n_paired_seeds": len(observed),
                "cores_averaged_within_seed": int(
                    part[part.task.eq(values[0])].core.nunique()
                ),
                "mean_difference": observed.mean(),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _seed_task_differences() -> pd.DataFrame:
    """Per-seed spatial-minus-random differences behind ``task_feedback_effects``.

    Read from the published seed-outcome table; the MNIST value of a seed is
    the within-seed mean over its two cores, exactly as the summary table
    averages it, and the noise-task value is the raw-additive core alone.
    """
    runs = pd.read_csv(SEED_OUTCOMES)
    part = runs[runs.family.eq("spatial")]
    part = part[~(part.task.eq("noise_resilience") & part.core.eq("dendritic_shunting"))]
    wide = part.pivot(
        index=["task", "core", "strategy", "feedback", "seed"],
        columns="topology",
        values="test_accuracy",
    ).reset_index()
    wide["difference"] = wide.spatial - wide.random
    return wide.groupby(["task", "strategy", "feedback", "seed"], as_index=False).difference.mean()


def _branch_map(seed: int, topology: str) -> np.ndarray:
    """Image with pixel value = branch index + 1 (0 = uncontacted)."""
    example = _indices(seed, topology).reshape(
        N_OWNERS, BRANCHES_PER_OWNER, CONTACTS_PER_BRANCH
    )[0]
    image = np.zeros((28, 28), dtype=float)
    for branch, values in enumerate(example):
        image.flat[values.numpy()] = branch + 1
    return image


# Bayer-ordered lightness levels for the 4 x 4 block partition: horizontally
# and vertically adjacent blocks always differ by at least four of sixteen
# steps, so region boundaries stay visible in one neutral hue.
_BAYER_4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])


def _branch_levels(spatial_image: np.ndarray) -> np.ndarray:
    """Lightness level (0..15) of every branch, from its 7 x 7 block."""
    levels = np.zeros(BRANCHES_PER_OWNER, dtype=int)
    block = 28 // 4
    for branch in range(BRANCHES_PER_OWNER):
        rows, cols = np.nonzero(spatial_image == branch + 1)
        cell_r, cell_c = rows // block, cols // block
        if cell_r.min() != cell_r.max() or cell_c.min() != cell_c.max():
            raise ValueError("Spatial branch is not confined to one 7 x 7 block")
        levels[branch] = _BAYER_4[cell_r[0], cell_c[0]]
    return levels


def plot(owner: pd.DataFrame, performance: pd.DataFrame) -> None:
    from matplotlib.colors import ListedColormap, to_rgb

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    pt = 72.0
    fw, fh = FIG_W * pt, FIG_H * pt

    def axes_pt(x0, y_top, width, height):
        return fig.add_axes([x0 / fw, (fh - y_top - height) / fh, width / fw, height / fh])

    colors = {"random": COLORS["mute"], "spatial": COLORS["highlight"]}
    labels = {"random": "Random", "spatial": "Spatial"}

    # Panel A: one neuron's sixteen branches on the 28 x 28 plane, spatial
    # sampler beside the matched random sampler of the same seed.
    spatial_image = _branch_map(SEEDS[0], "spatial")
    random_image = _branch_map(SEEDS[0], "random")
    levels = _branch_levels(spatial_image)
    dark, light = np.array(to_rgb("#232B36")), np.array(to_rgb("#A8B0BB"))
    ramp = [tuple(dark + (light - dark) * t) for t in np.linspace(0, 1, BRANCHES_PER_OWNER)]
    cmap = ListedColormap(["#F4F6F7", *[ramp[levels[b]] for b in range(BRANCHES_PER_OWNER)]])
    map_w = 62.0
    ax_a = axes_pt(COL_X0[0] - 4, ROW_Y_TOP[0], map_w, map_w)
    ax_a2 = axes_pt(COL_X0[0] - 4 + map_w + 6, ROW_Y_TOP[0], map_w, map_w)
    for ax, image, sub in [(ax_a, spatial_image, "spatial"), (ax_a2, random_image, "random")]:
        ax.imshow(image, cmap=cmap, vmin=-0.5, vmax=BRANCHES_PER_OWNER + 0.5,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(LW_HAIR)
            spine.set_color(COLORS["edge"])
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=PT_SMALL, color=COLORS["ink"])
    for edge in (6.5, 13.5, 20.5):
        ax_a.axhline(edge, color=COLORS["edge"], lw=LW_HAIR)
        ax_a.axvline(edge, color=COLORS["edge"], lw=LW_HAIR)
    panel_title(ax_a, "A", "Branch input maps")

    # Panels B and C: every owner map (128 neurons x 10 seeds per topology).
    for col, letter, metric, title, ylabel in [
        (1, "B", "unique_input_features", "Unique input coverage", "features per neuron"),
        (2, "C", "mean_pairwise_branch_jaccard", "Cross-branch collision", "mean branch Jaccard"),
    ]:
        ax = axes_pt(COL_X0[col], ROW_Y_TOP[0], AX_W, AX_H)
        for x, topology in enumerate(("random", "spatial")):
            values = owner.loc[owner.topology == topology, metric].to_numpy(float)
            rng = np.random.default_rng(100 + x)
            ax.scatter(
                x + rng.normal(0, 0.05, len(values)),
                values,
                s=4,
                alpha=0.10,
                color=colors[topology],
                edgecolors="none",
                zorder=2,
            )
            ax.errorbar(
                x,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                color=colors[topology],
                ms=4.8,
                capsize=ERR_CAPSIZE,
                capthick=LW_ERR,
                lw=LW_ERR,
                zorder=5,
            )
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1], [labels["random"], labels["spatial"]])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("contact map")
        if metric == "unique_input_features":
            ax.set_yticks([260, 280, 300, 320, 340])
            ax.set_ylim(252, 344)
            full = owner.loc[owner.topology == "spatial", metric].to_numpy(float)
            ax.annotate(
                f"{full.mean():.0f}/{full.mean():.0f}" if np.allclose(full, full[0]) else f"{full.mean():.0f}",
                xy=(1, full.mean()), xytext=(0.78, full.mean()),
                textcoords="data", ha="right", va="center",
                fontsize=PT_SMALL, color=COLORS["ink"],
            )
        panel_title(ax, letter, title)
        style_axis(ax)

    # Panel D: paired spatial-minus-random effect, seeds behind the means.
    ax = axes_pt(COL_X0[0], ROW_Y_TOP[1], AX_W, AX_H)
    order = ["backprop", "per_soma", "per_soma_shared", "path_transport"]
    x = np.arange(len(order))
    seeds = _seed_task_differences()
    series = [
        ("mnist", "o", COLORS["ink"], -0.16, "MNIST"),
        ("noise_resilience", "s", COLORS["additive"], 0.16, "Noise task"),
    ]
    for task, marker, color, offset, label in series:
        part = performance[performance.task == task].set_index("feedback").loc[order]
        y = 100 * part.mean_difference.to_numpy()
        rng = np.random.default_rng(11 if task == "mnist" else 12)
        for position, feedback in enumerate(order):
            values = seeds[(seeds.task == task) & (seeds.feedback == feedback)]
            points = 100 * values.difference.to_numpy(float)
            if len(points) != int(part.loc[feedback, "n_paired_seeds"]) or not np.isclose(points.mean(), y[position]):
                raise ValueError(f"Seed table does not reproduce the {task}/{feedback} mean")
            ax.scatter(
                position + offset + rng.normal(0, 0.045, len(points)),
                points,
                s=SEED_MS**2,
                marker=marker,
                color=color,
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack(
                [
                    100 * (part.mean_difference - part.ci95_low),
                    100 * (part.ci95_high - part.mean_difference),
                ]
            ),
            fmt=marker,
            ms=4.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
            capthick=LW_ERR,
            label=label,
            zorder=5,
        )
    ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-2.8, 8.2)
    ax.set_xticks(x, ["BP", "MW\nscalar", "Neuron", "Exact\npath"])
    ax.set_xlabel("feedback rule")
    ax.set_ylabel("spatial - random (pp)")
    panel_title(ax, "D", "Forward effect")
    clean_legend(ax, fontsize=PT_SMALL, loc="lower right",
                 handlelength=0.7, handletextpad=0.3, borderaxespad=0.15)
    style_axis(ax)

    style_direct_color_labels(fig)
    audit_text_over_data(fig, "fig_spatial_topology_audit")
    audit_layout(fig, "fig_spatial_topology_audit")
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_spatial_topology_audit.{suffix}", dpi=350)
    plt.close(fig)


def write_report(summary: pd.DataFrame, performance: pd.DataFrame) -> None:
    indexed = summary.set_index(["topology", "metric"])
    random_unique = indexed.loc[("random", "unique_input_features"), "mean"]
    spatial_unique = indexed.loc[("spatial", "unique_input_features"), "mean"]
    random_collision = indexed.loc[("random", "mean_pairwise_branch_jaccard"), "mean"]
    spatial_collision = indexed.loc[("spatial", "mean_pairwise_branch_jaccard"), "mean"]
    bp = performance[performance.feedback == "backprop"].mean_difference.mean() * 100
    text = f"""# Fixed spatial-topology audit

The spatial and random networks contain the same number of trainable contacts,
but they do not have the same input coverage. Across {len(SEEDS)} topology seeds
and {N_OWNERS} neurons per seed, the random map covers {random_unique:.1f} unique
pixels per neuron on average, whereas the spatial map covers exactly
{spatial_unique:.0f}. Mean pairwise branch overlap is {random_collision:.4f} for
random maps and {spatial_collision:.4f} for spatial maps. This follows from the
registered spatial sampler: its 16 distal branch regions are disjoint and each
contains 21 unique contacts.

The spatial-minus-random accuracy effect is also present under matched
backpropagation (mean {bp:.2f} percentage points across the input-valid
task--core conditions). MNIST averages additive and shunting cores, whereas the
randomly projected noise task uses the additive core only: the historical
signed-input positive-conductance shunting cells are excluded independently of
outcome. The retained experiment therefore demonstrates a forward sparse-
connectivity prior, but it does not isolate task-aligned dendritic credit
routing. We retain it as a boundary/control result and do not use it as
evidence that spatial topology improves local credit assignment specifically.
"""
    (OUT / "report.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    owner = connectivity_metrics()
    summary = summarize(owner)
    condition_performance = performance_effects()
    performance = task_feedback_effects()
    owner.to_csv(OUT / "owner_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    condition_performance.to_csv(OUT / "performance_contrasts.csv", index=False)
    performance.to_csv(OUT / "task_feedback_effects.csv", index=False)
    plot(owner, performance)
    write_report(summary, condition_performance)
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
