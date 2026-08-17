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
from neurips_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_ERR,
    LW_REF,
    PT_LEGEND,
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


def plot(owner: pd.DataFrame, performance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIG_W, 2.45),
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "bottom": 0.25,
            "top": 0.82,
            "wspace": 0.63,
        },
    )
    colors = {"random": COLORS["mute"], "spatial": COLORS["highlight"]}
    labels = {"random": "Random", "spatial": "Spatial"}

    # One owner's contacts, aggregated over the 28 x 28 feature plane.
    ax = axes[0]
    example = _indices(SEEDS[0], "spatial").reshape(
        N_OWNERS, BRANCHES_PER_OWNER, CONTACTS_PER_BRANCH
    )[0]
    image = np.zeros((28, 28), dtype=float)
    for branch, values in enumerate(example):
        image.flat[values.numpy()] = branch + 1
    ax.imshow(image, cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    panel_title(ax, "A", "Disjoint spatial regions")

    for ax, letter, metric, title, ylabel in [
        (
            axes[1],
            "B",
            "unique_input_features",
            "Unique input coverage",
            "features per neuron",
        ),
        (
            axes[2],
            "C",
            "mean_pairwise_branch_jaccard",
            "Cross-branch collision",
            "mean branch Jaccard",
        ),
    ]:
        for x, topology in enumerate(("random", "spatial")):
            values = owner.loc[owner.topology == topology, metric].to_numpy(float)
            rng = np.random.default_rng(100 + x)
            take = rng.choice(len(values), size=min(250, len(values)), replace=False)
            ax.scatter(
                x + rng.normal(0, 0.035, len(take)),
                values[take],
                s=5,
                alpha=0.18,
                color=colors[topology],
                edgecolors="none",
            )
            ax.errorbar(
                x,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                color=colors[topology],
                ms=4.8,
                capsize=ERR_CAPSIZE,
                lw=LW_ERR,
                zorder=5,
            )
        ax.set_xticks([0, 1], [labels["random"], labels["spatial"]])
        ax.set_ylabel(ylabel)
        panel_title(ax, letter, title)
        style_axis(ax)

    ax = axes[3]
    order = ["backprop", "per_soma", "per_soma_shared", "path_transport"]
    x = np.arange(len(order))
    for task, marker, offset in [("mnist", "o", -0.09), ("noise_resilience", "s", 0.09)]:
        part = performance[performance.task == task].set_index("feedback").loc[order]
        y = 100 * part.mean_difference.to_numpy()
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
            color=COLORS["shunting"] if task == "mnist" else COLORS["additive"],
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
            label="MNIST" if task == "mnist" else "Noise task",
        )
    ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax.set_xticks(x, ["BP", "Scalar", "Ancestry", "Exact"], rotation=25, ha="right")
    ax.set_ylabel("spatial - random (pp)")
    panel_title(ax, "D", "Forward effect")
    clean_legend(ax, fontsize=PT_LEGEND, loc="upper right")
    style_axis(ax)

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
