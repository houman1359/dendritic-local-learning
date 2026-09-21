#!/usr/bin/env python3
"""Test a signed credit-routing prediction in published mouse BCI data.

Francioni et al. (Nature, 2026) assigned two intermingled populations of
neurons opposite causal signs in a brain-computer-interface task.  For each
animal, their Extended Data Fig. 13E reports the change in soma-dendrite (SD)
residual between error-reduction and error-increase epochs for P+ and P-
neurons.  A scalar broadcast predicts a common-mode contrast.  A teaching
signal multiplied by the known causal sign of each population predicts an
antisymmetric, signed contrast.

This script reanalyses the authors' public source-data workbook.  It is an
external consistency test of signed vectorized credit, not evidence that the
experimental signal implements the particular conductance mechanism derived
in our model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "external_data" / "animal_learning" / "francioni_2026" / "source_data.xlsx"
DEFAULT_OUTDIR = ROOT / "source_data" / "animal_learning_francioni"
DEFAULT_FIGURE_STEM = ROOT / "figures" / "generated" / "fig_francioni_signed_credit_validation"
EXPECTED_SHA256 = "be1b87481d0f1d1bffc85d8a4cabfd37b7055d640306afab5b3622de0ced8264"

PPLUS = COLORS["shunting"]
PMINUS = COLORS["additive"]
SIGNED = COLORS["highlight"]
COMMON = COLORS["point_mlp"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 50_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(int(n_boot), len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def exact_sign_flip_p_greater(values: np.ndarray) -> float:
    """Exact one-sided random-sign p-value for a positive sample mean."""

    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    signs = np.asarray(
        [[1.0 if (mask >> bit) & 1 else -1.0 for bit in range(len(values))]
         for mask in range(1 << len(values))],
        dtype=float,
    )
    null = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(null >= observed - 1e-15))


def load_source(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    animal = pd.read_excel(workbook, sheet_name="Ext 13E", header=None)
    animal = animal.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    animal.columns = ["pplus_contrast", "pminus_contrast", "signed_separation"]
    animal.insert(0, "animal", np.arange(1, len(animal) + 1, dtype=int))
    recomputed = animal["pplus_contrast"] - animal["pminus_contrast"]
    if not np.allclose(recomputed, animal["signed_separation"], atol=1e-12, rtol=1e-12):
        raise ValueError("Extended Data Fig. 13E signed separation is internally inconsistent")

    neuron = pd.read_excel(workbook, sheet_name="5E", header=None)
    neuron = neuron.apply(pd.to_numeric, errors="coerce")
    # The workbook follows the Fig. 5E caption: columns 4/5 are P+/P- during
    # error reduction; columns 0/1 are P+/P- during error increase.
    records: list[pd.DataFrame] = []
    for population, epoch, column in [
        ("P+", "error increase", 0),
        ("P-", "error increase", 1),
        ("P+", "error reduction", 4),
        ("P-", "error reduction", 5),
    ]:
        values = neuron[column].dropna().to_numpy(dtype=float)
        records.append(
            pd.DataFrame(
                {
                    "population": population,
                    "epoch": epoch,
                    "sd_residual_z": values,
                    "source_row": np.flatnonzero(neuron[column].notna().to_numpy()) + 1,
                }
            )
        )
    return animal, pd.concat(records, ignore_index=True)


def project_modes(animal: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    matrix = animal[["pplus_contrast", "pminus_contrast"]].to_numpy(dtype=float)
    common_basis = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    signed_basis = np.asarray([1.0, -1.0]) / np.sqrt(2.0)
    common = matrix @ common_basis
    signed = matrix @ signed_basis
    total_energy = float(np.sum(matrix * matrix))
    common_energy = float(np.sum(common * common))
    signed_energy = float(np.sum(signed * signed))
    if not np.isclose(common_energy + signed_energy, total_energy, atol=1e-12):
        raise RuntimeError("orthogonal mode energies do not sum to total energy")
    modes = animal[["animal"]].copy()
    modes["common_mode"] = common
    modes["signed_mode"] = signed
    modes["signed_energy_fraction_within_animal"] = (
        signed * signed / (signed * signed + common * common)
    )

    def pooled_signed_fraction(indices: np.ndarray) -> float:
        signed_energy_i = float(np.sum(signed[indices] * signed[indices]))
        common_energy_i = float(np.sum(common[indices] * common[indices]))
        return signed_energy_i / (signed_energy_i + common_energy_i)

    rng = np.random.default_rng(20260810)
    bootstrap_indices = rng.integers(0, len(animal), size=(50_000, len(animal)))
    bootstrap_fractions = np.asarray(
        [pooled_signed_fraction(indices) for indices in bootstrap_indices],
        dtype=float,
    )
    leave_one_out = [
        pooled_signed_fraction(np.delete(np.arange(len(animal)), index))
        for index in range(len(animal))
    ]
    summary = {
        "common_energy_fraction": common_energy / total_energy,
        "signed_energy_fraction": signed_energy / total_energy,
        "signed_to_common_energy_ratio": signed_energy / common_energy,
        "aggregation": "pooled squared projections across the six animal contrast vectors",
        "animal_level_signed_energy_fractions": modes[
            "signed_energy_fraction_within_animal"
        ].tolist(),
        "animal_bootstrap_95_ci": np.quantile(
            bootstrap_fractions, [0.025, 0.975]
        ).tolist(),
        "leave_one_animal_out_signed_energy_fractions": leave_one_out,
    }
    return modes, summary


def build_summary(animal: pd.DataFrame, neuron: pd.DataFrame, mode_summary: dict[str, float]) -> dict:
    output: dict[str, object] = {
        "claim_level": "external consistency test of a signed, neuron-indexed teaching signal",
        "source": {
            "citation": "Francioni et al., Nature 652, 1254-1263 (2026)",
            "doi": "10.1038/s41586-026-10190-7",
            "source_sheet_animal": "Ext 13E",
            "source_sheet_neuron": "5E",
        },
        "n_animals": int(len(animal)),
        "mode_decomposition": mode_summary,
        "animal_level": {},
        "neuron_level_descriptive": {},
        "interpretation": (
            "The known causal sign of the BCI mapping predicts the dominant component of the "
            "population contrast. This supports signed vectorized credit over a common scalar "
            "contrast in this dataset, but does not uniquely identify the conductance mechanism."
        ),
    }
    tests = {
        "pplus_contrast": "greater",
        "pminus_contrast": "less",
        "signed_separation": "greater",
    }
    for idx, (column, alternative) in enumerate(tests.items()):
        values = animal[column].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_ci(values, seed=1200 + idx)
        transformed = -values if alternative == "less" else values
        output["animal_level"][column] = {
            "mean": mean,
            "bootstrap_95_ci": [lo, hi],
            "ordering_positive": int(np.sum(values > 0)),
            "ordering_negative": int(np.sum(values < 0)),
            "wilcoxon_one_sided_p": float(stats.wilcoxon(values, alternative=alternative).pvalue),
            "exact_sign_flip_one_sided_p": exact_sign_flip_p_greater(transformed),
        }

    grouped = neuron.groupby(["population", "epoch"], sort=True)["sd_residual_z"]
    for (population, epoch), values in grouped:
        key = f"{population}_{epoch}".replace("+", "plus").replace("-", "minus").replace(" ", "_")
        output["neuron_level_descriptive"][key] = {
            "n_neurons": int(len(values)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
        }
    return output


def schematic_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Known causal signs specify credit")
    ax.text(0.50, 0.89, "BCI error", ha="center", va="center", fontsize=PT_ANNOT, fontweight="bold")
    for x, label, sign, color in [(0.25, "P+", "+", PPLUS), (0.75, "P-", "-", PMINUS)]:
        ax.add_patch(Circle((x, 0.53), 0.105, fc="white", ec=color, lw=LW_DATA))
        ax.text(x, 0.53, label, ha="center", va="center", fontsize=PT_ANNOT, fontweight="bold", color=color)
        ax.add_patch(
            FancyArrowPatch(
                (0.48 if x < 0.5 else 0.52, 0.82),
                (x + (0.04 if x < 0.5 else -0.04), 0.65),
                arrowstyle="-|>", mutation_scale=9, lw=LW_REF, color=color,
            )
        )
        ax.text(x, 0.33, f"causal sign {sign}", ha="center", va="center", fontsize=PT_SMALL, color=color)
    ax.text(0.50, 0.10, "prediction: opposite dendritic contrasts", ha="center", va="center", fontsize=PT_SMALL)


def make_figure(animal: pd.DataFrame, neuron: pd.DataFrame, modes: pd.DataFrame, mode_summary: dict, stem: Path) -> None:
    apply_neurips_style()
    fig = plt.figure(figsize=(FIG_W, 4.55))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.98, bottom=0.10, top=0.92, wspace=0.42, hspace=0.70)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    schematic_panel(axes[0])

    ax = axes[1]
    panel_title(ax, "B", "Sign inversion across animals")
    for row in animal.itertuples(index=False):
        ax.plot([0, 1], [row.pplus_contrast, row.pminus_contrast], color=COLORS["mute"], lw=LW_HAIR, alpha=0.55, zorder=1)
        ax.scatter(0, row.pplus_contrast, s=24, color=PPLUS, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(1, row.pminus_contrast, s=24, color=PMINUS, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axhline(0, color=COLORS["edge"], lw=LW_REF, ls="--", zorder=0)
    ax.set_xticks([0, 1], ["P+", "P-"])
    ax.set_ylabel("SD residual\n(reduction - increase)")
    ax.text(0.03, 0.04, "6/6 signed separations > 0", transform=ax.transAxes, fontsize=PT_SMALL)
    style_axis(ax, grid="y")

    ax = axes[2]
    panel_title(ax, "C", "Most contrast energy is in the signed mode")
    fractions = [mode_summary["common_energy_fraction"], mode_summary["signed_energy_fraction"]]
    ax.bar([0, 1], fractions, color=[COMMON, SIGNED], edgecolor=COLORS["edge"], linewidth=LW_EDGE, width=0.63)
    for x, value in enumerate(fractions):
        ax.text(x, value + 0.035, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=PT_ANNOT, fontweight="bold")
    ax.set_xticks([0, 1], ["common\n(scalar)", "signed\n(P+ / P-)"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of contrast energy")
    style_axis(ax, grid="y")

    ax = axes[3]
    panel_title(ax, "D", "Neuron-level sign reversal")
    order = [
        ("P+", "error increase"),
        ("P+", "error reduction"),
        ("P-", "error increase"),
        ("P-", "error reduction"),
    ]
    labels = ["P+\nincrease", "P+\nreduction", "P-\nincrease", "P-\nreduction"]
    colors = [PPLUS, PPLUS, PMINUS, PMINUS]
    for idx, ((population, epoch), color) in enumerate(zip(order, colors)):
        values = neuron.loc[(neuron.population == population) & (neuron.epoch == epoch), "sd_residual_z"].to_numpy(dtype=float)
        parts = ax.violinplot(values, positions=[idx], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        ax.errorbar(idx, mean, yerr=sem, fmt="o", ms=4.2, color=color, mec="white", mew=0.45, lw=LW_REF, capsize=ERR_CAPSIZE, zorder=4)
        ax.text(idx, ax.get_ylim()[0] + 0.05, f"n={len(values)}", ha="center", va="bottom", fontsize=PT_SMALL, color=COLORS["mute"])
    ax.axhline(0, color=COLORS["edge"], lw=LW_REF, ls="--", zorder=0)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("SD residual (z-score)")
    style_axis(ax, grid="y")

    fig.canvas.draw()
    layout = audit_layout(fig, stem.name)
    overlap = audit_text_over_data(fig, stem.name)
    if layout or overlap:
        print(f"layout audit: {len(layout)} layout and {len(overlap)} text/data warnings")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--allow-unverified-input", action="store_true")
    args = parser.parse_args()

    observed_hash = file_sha256(args.workbook)
    if observed_hash != EXPECTED_SHA256 and not args.allow_unverified_input:
        raise ValueError(f"source workbook SHA-256 mismatch: {observed_hash}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    animal, neuron = load_source(args.workbook)
    modes, mode_summary = project_modes(animal)
    summary = build_summary(animal, neuron, mode_summary)
    summary["source"]["sha256"] = observed_hash

    animal.to_csv(args.outdir / "animal_signed_contrasts.csv", index=False)
    neuron.to_csv(args.outdir / "neuron_sd_residual_distributions.csv", index=False)
    modes.to_csv(args.outdir / "animal_common_signed_modes.csv", index=False)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# External signed-credit test",
        "",
        "Francioni et al. assigned P+ and P- neurons opposite causal signs in a mouse BCI task.",
        "The analysis was specified as a common-mode versus signed-mode comparison.",
        "",
        f"- Animals: {len(animal)}.",
        f"- Signed separation was positive in {int((animal.signed_separation > 0).sum())}/{len(animal)} animals.",
        f"- Mean signed separation: {animal.signed_separation.mean():.3f} SD-residual z units.",
        f"- Exact one-sided random-sign permutation p: {summary['animal_level']['signed_separation']['exact_sign_flip_one_sided_p']:.4f}.",
        f"- Signed mode captured {100 * mode_summary['signed_energy_fraction']:.1f}% of the two-population contrast energy; the common mode captured {100 * mode_summary['common_energy_fraction']:.1f}%.",
        f"- Animal-bootstrap 95% interval for pooled signed-mode energy: {100 * mode_summary['animal_bootstrap_95_ci'][0]:.1f}% to {100 * mode_summary['animal_bootstrap_95_ci'][1]:.1f}%.",
        "",
        "Interpretation: the known causal sign of each population predicts the dominant dendritic contrast. This is external support for signed vectorized credit, not a unique identification of the conductance mechanism.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    make_figure(animal, neuron, modes, mode_summary, args.figure_stem)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
