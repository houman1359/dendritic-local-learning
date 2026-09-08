#!/usr/bin/env python3
"""Label the passive operating regime without regenerating historical SI assets."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
J = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_figure as previous
sys.path.insert(0, str(J / "scripts/credit_first_figures"))
from focused_provenance import publish
from figure_canvas import NativeCanvas, Margins
from journal_style import style_direct_color_labels


def display_rows():
    old = previous.old
    rows = []
    category = pd.read_csv(J / "source_data/figure4/category_effects.csv")
    selected = category[np.isclose(category.dose, 1.)]
    cells = selected.groupby(["perturbation", "root_id", "category"], as_index=False).median_abs_log_gradient_change.mean()
    for perturbation, group in cells.groupby("perturbation"):
        for index, relation in enumerate(old.CATEGORIES):
            part = group[group.category.eq(relation)]
            assert len(part) == 8
            mean, low, high = old.mean_ci(part.median_abs_log_gradient_change.to_numpy(float), seed=1610+index)
            rows.append(dict(panel="B", record="cohort_mean", perturbation=perturbation, category=relation,
                             mean=mean, ci95_low=low, ci95_high=high, n_cells=8, regime="normalized passive"))
            rows.extend(dict(panel="B", record="cell_value", **r) for r in part.to_dict("records"))
    shapley = pd.read_csv(J / "source_data/focal_decomposition/cell_shapley.csv")
    shapley = shapley[shapley.estimand.eq("full_shunt_minus_matched_additive")].sort_values("root_id")
    assert len(shapley) == 8
    for index, column in enumerate(["driving_force_only_localization", "full_shunt_localization"]):
        mean, low, high = old.mean_ci(shapley[column].to_numpy(float), seed=1710+index)
        rows.append(dict(panel="C", record="cohort_mean", condition=column, mean=mean, ci95_low=low, ci95_high=high, n_cells=8))
        rows.extend(dict(panel="C", record="cell_value", condition=column, root_id=r.root_id, value=getattr(r, column)) for r in shapley.itertuples())
    signed = pd.read_csv(previous.OUT / "signed_calibration/signed_cohort_summary.csv")
    signed = signed[signed.perturbation.eq("focal shunt")]
    assert len(signed) == 8
    rows.extend(dict(panel="D", record="cohort_mean", **r) for r in signed.to_dict("records"))
    physical = pd.read_csv(J / "source_data/physical_cable_sensitivity/cell_primary_contrasts.csv")
    ratio = pd.read_csv(J / "source_data/physical_cable_sensitivity/cell_electrotonic_ratios.csv")
    ratio = ratio.merge(physical[["cohort", "regime", "root_id"]].drop_duplicates(), on=["cohort", "regime", "root_id"], validate="one_to_one")
    median = ratio.groupby(["cohort", "regime"], as_index=False).median_axial_to_leak_ratio.median()
    physical = physical.merge(median, on=["cohort", "regime"], validate="many_to_one")
    for cohort, _, _, _ in old.COHORTS:
        subset = physical[physical.cohort.eq(cohort)]
        if cohort == "original_eight": subset = subset[subset.regime.str.startswith("Ra150_")]
        for index, ((regime, x), part) in enumerate(subset.groupby(["regime", "median_axial_to_leak_ratio"])):
            mean, low, high = old.mean_ci(part.difference.to_numpy(float), seed=1740+index)
            rows.append(dict(panel="E", record="cohort_mean", cohort=cohort, regime=regime,
                             median_axial_to_leak_ratio=x, mean=mean, ci95_low=low, ci95_high=high, n_cells=len(part)))
            rows.extend(dict(panel="E", record="cell_value", **r) for r in part.to_dict("records"))
    return rows


def build(emit_main=False):
    canvas = NativeCanvas(490/72, 3, row_weights=[110, 110, 120], hgutter_pt=37, vgutter_pt=44,
                          margins=Margins(left=49, right=12, top=23, bottom=35))
    a = canvas.panel("A", 0, 0, 6, schematic=True, title="Shunting acts on an ancestry partition")
    b = canvas.panel("B", 0, 6, 6)
    c = canvas.panel("C", 1, 0, 6, grid="y")
    d = canvas.panel("D", 1, 6, 6, grid="y")
    e = canvas.panel("E", 2, 0, 12)
    previous.ancestry(a)
    previous.old.panel_tree_relation(b)
    b.set_title("Normalized passive regime")
    previous.old.panel_factor_freeze(c)
    previous.signed_calibration(d)
    previous.old.panel_electrotonic(e)
    e.set_title("Electrical state determines the localization contrast")
    style_direct_color_labels(canvas.fig)
    output = J / "figures/components/focused_main_08.pdf"
    findings = canvas.save(output, name="focused_main_08", dpi=180)
    plt.close(canvas.fig)
    sources = [J/"source_data"/folder/name for folder, name in [
        ("figure4", "category_effects.csv"), ("figure4", "focal_localization.csv"),
        ("focal_decomposition", "cell_shapley.csv"),
        ("physical_cable_sensitivity", "cell_primary_contrasts.csv"),
        ("physical_cable_sensitivity", "cell_electrotonic_ratios.csv"),
        ("shunt_ancestry_gain/signed_calibration", "signed_cohort_summary.csv"),
        ("shunt_ancestry_gain/signed_calibration", "signed_cell_effects.csv"),
        ("shunt_ancestry_gain/signed_calibration", "signed_category_effects.csv"),
        ("shunt_ancestry_gain/signed_calibration", "validation.json")]]
    builders = [Path(__file__), Path(previous.__file__), Path(previous.old.__file__),
                J/"scripts/build_journal_figures.py", J/"scripts/figure_canvas.py", J/"scripts/journal_style.py"]
    panels = {"A": "Unchanged exact passive ancestry-partition gain schematic; local driving forces may vary.",
              "B": "Unchanged unit-dose category effects under the permissive normalized passive model; regime now named directly on the panel.",
              "C": "Unchanged adjoint substitution comparison, holding post-shunt voltage.",
              "D": "Unchanged signed descendant/off-route log gradient changes at the two physical endpoints; negative means attenuation.",
              "E": "Unchanged full-width physical-calibration contrast against current injection; localization is distinct from absolute attenuation."}
    publish(8, output, display_rows(), sources, builders, panels, emit_main=emit_main,
            layout_findings=findings, notes="Text-only panel clarification; exact archived plotting functions, rows, estimands and resampling seeds retained. No conversion to arbitrary biological credit units. Historical native and supplementary assets are unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-main", action="store_true")
    build(parser.parse_args().emit_main)
