#!/usr/bin/env python3
"""Recover signed physical-cable responses discarded by the original wrapper.

This is a deterministic replay, not a new fitted experiment. The frozen focal
site identities, electrical calibration, dose and soma-restoring control
are unchanged. Every original absolute endpoint must reproduce before signed
effects may be used. The two membrane resistances are existing endpoints of
the Figure 7E sweep, chosen to expose amplitude separately from localization.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from analyze_physical_cable_sensitivity import physical_conductance_system
from run_focal_shunting_credit_perturbation import (
    exact_e_gradient, solve_with_soma_clamp, relation_indices, category_record,
)

OUT = JOURNAL / "source_data/shunt_ancestry_gain/signed_calibration"
REGIMES = {"Ra150_Rm300": 300.0, "Ra150_Rm15000": 15000.0}
COHORTS = {
    "original_eight": "source_data/figure3/segment_metrics.csv",
    "v661_disjoint": "source_data/microns_v661_replication/routing/segment_metrics.csv",
}
REFERENCE = JOURNAL / "source_data/physical_cable_sensitivity/focal_localization.csv.gz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_cell(root_id, segments, rm, frozen_sites):
    electrical, matrix0, rhs0, soma, parents, children = physical_conductance_system(
        segments, .35, .35, 1.0, -.2,
        axial_resistivity_ohm_cm=150.0, membrane_resistance_ohm_cm2=rm)
    lookup = {int(s): i for i, s in enumerate(electrical.segment_id)}
    voltage0 = np.linalg.solve(matrix0, rhs0)
    target = float(voltage0[soma] - 1.0)
    gradient0 = exact_e_gradient(matrix0, voltage0, soma, target, 1.0)
    e_segments = electrical.loc[electrical.E_size.gt(0), "segment_id"].astype(int).tolist()
    e_indices = np.array([lookup[s] for s in e_segments])
    e_weights = electrical.loc[e_indices, "E_size"].to_numpy(float)
    lengths = electrical.set_index("segment_id").path_length_um.astype(float).to_dict()
    categories, focal = [], []
    max_clamp_error = 0.0
    for site in frozen_sites:
        k = lookup[site]
        relations = relation_indices(site, e_segments, lengths, parents, children)
        assert all(len(relations[key]) >= 3 for key in ["descendant", "depth-matched unrelated"])
        delta = float(electrical.loc[k, ["g_leak", "g_e", "g_i"]].sum())
        for perturbation in ["matched additive", "focal shunt"]:
            matrix, rhs = matrix0.copy(), rhs0.copy()
            if perturbation == "focal shunt":
                matrix[k, k] += delta
                rhs[k] += -.2 * delta
            else:
                rhs[k] += delta * (-.2 - voltage0[k])
            voltage, current, error = solve_with_soma_clamp(matrix, rhs, soma, voltage0[soma])
            max_clamp_error = max(max_clamp_error, float(error))
            gradient = exact_e_gradient(matrix, voltage, soma, target, 1.0)
            base, changed = gradient0[e_indices], gradient[e_indices]
            epsilon = 1e-15 * max(float(np.max(np.abs(base))), 1.0)
            signed = np.log((np.abs(changed) + epsilon) / (np.abs(base) + epsilon))
            records = {}
            for relation in ["descendant", "depth-matched unrelated"]:
                row = category_record(root_id, site, 1.0, perturbation, relation,
                                      relations[relation], np.abs(signed), signed,
                                      np.signbit(changed) != np.signbit(base), e_weights)
                categories.append(row)
                records[relation] = row
            desc = records["descendant"]["median_abs_log_gradient_change"]
            other = records["depth-matched unrelated"]["median_abs_log_gradient_change"]
            focal.append(dict(root_id=int(root_id), focal_segment_id=int(site), dose=1.0,
                              perturbation=perturbation, delta_conductance=delta,
                              soma_clamp_current=float(current), localization_index=desc-other,
                              descendant_change=desc, matched_unrelated_change=other))
    return categories, focal, dict(root_id=int(root_id), n_selected_focal_sites=len(frozen_sites),
                                    maximum_soma_clamp_error=max_clamp_error,
                                    selected_focal_segments=frozen_sites)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.regime.isin(REGIMES)].copy()
    categories, replay, checks = [], [], []
    for cohort, filename in COHORTS.items():
        segments = pd.read_csv(JOURNAL / filename)
        for regime, rm in REGIMES.items():
            eligible = set(reference.loc[(reference.cohort == cohort) & (reference.regime == regime), "root_id"])
            for root_id, group in segments.groupby("root_id", sort=True):
                if root_id not in eligible:
                    continue
                frozen_sites = sorted(reference.loc[(reference.cohort == cohort) &
                    (reference.regime == regime) & (reference.root_id == root_id), "focal_segment_id"].unique().astype(int).tolist())
                category, focal, validation = replay_cell(int(root_id), group.copy(), rm, frozen_sites)
                for target, rows in [(categories, category), (replay, focal)]:
                    target.extend(dict(cohort=cohort, regime=regime, **row) for row in rows)
                checks.append(dict(cohort=cohort, regime=regime, **validation))
            print(f"Replayed {cohort}/{regime}: {len(eligible)} cells", flush=True)
    replay = pd.DataFrame(replay)
    keys = ["cohort", "regime", "root_id", "focal_segment_id", "dose", "perturbation"]
    paired = reference.merge(replay, on=keys, how="outer", validate="one_to_one", indicator=True,
                             suffixes=("_frozen", "_replayed"))
    assert paired._merge.eq("both").all(), "Original focal-site coverage changed"
    errors = {}
    for column in ["localization_index", "descendant_change", "matched_unrelated_change",
                   "delta_conductance", "soma_clamp_current"]:
        errors[column] = float(np.max(np.abs(paired[column + "_frozen"] - paired[column + "_replayed"])))
        np.testing.assert_allclose(paired[column + "_frozen"], paired[column + "_replayed"],
                                   rtol=2e-10, atol=2e-11, err_msg=column)
    category = pd.DataFrame(categories)
    category = category[category.category.isin(["descendant", "depth-matched unrelated"])].copy()
    category.to_csv(OUT / "signed_category_effects.csv", index=False)
    replay.to_csv(OUT / "replayed_focal_localization.csv", index=False)
    cell = category.groupby(["cohort", "regime", "root_id", "perturbation", "category"], as_index=False).agg(
        signed_log_change=("median_signed_log_gradient_change", "mean"),
        absolute_log_change=("median_abs_log_gradient_change", "mean"),
        sign_flip_fraction=("gradient_sign_flip_fraction", "mean"),
        n_focal_sites=("focal_segment_id", "size"))
    cell.to_csv(OUT / "signed_cell_effects.csv", index=False)
    summary = []
    for i, (keys, group) in enumerate(cell.groupby(["cohort", "regime", "perturbation", "category"], sort=True)):
        values = group.signed_log_change.to_numpy()
        means = np.random.default_rng(2026090700 + i).choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
        lo, hi = np.quantile(means, [.025, .975])
        summary.append(dict(zip(["cohort", "regime", "perturbation", "category"], keys),
                            mean_signed_log_change=float(values.mean()), ci95_low=float(lo), ci95_high=float(hi),
                            n_cells=len(values), mean_absolute_log_change=float(group.absolute_log_change.mean()),
                            maximum_sign_flip_fraction=float(group.sign_flip_fraction.max())))
    pd.DataFrame(summary).to_csv(OUT / "signed_cohort_summary.csv", index=False)
    sources = [Path(__file__), HERE.parent / "analyze_physical_cable_sensitivity.py",
               JOURNAL / "code/reconstructed_tree/run_focal_shunting_credit_perturbation.py",
               JOURNAL / "code/reconstructed_tree/analyze_microns_morphology_credit.py",
               REFERENCE, *(JOURNAL / filename for filename in COHORTS.values())]
    record = dict(status="passed", analysis="deterministic signed-response replay of two frozen physical-calibration endpoints",
                  site_selection="Exact focal_segment_id rows from the archived focal_localization table; no sites resampled",
                  dose=1.0, e_scale=0.35, i_scale=0.35,
                  excitatory_reversal=1.0, inhibitory_reversal=-0.2,
                  max_focal_sites=16, minimum_sites_per_relation=3,
                  signed_metric="median_i log[(abs(gamma_i_prime)+epsilon)/(abs(gamma_i)+epsilon)]; negative means attenuation of gradient magnitude",
                  epsilon="1e-15 * max(max_i abs(gamma_i), 1), separately within each cell baseline",
                  summary_unit="site medians averaged over focal sites within cell, then equal-weighted cell means",
                  bootstrap="20000 whole-cell resamples within each cohort/calibration/perturbation/relation; deterministic seeds 2026090700 + sorted row index",
                  paired_focal_rows=len(paired), cell_regimes=len(checks), maximum_absolute_replay_errors=errors,
                  source_sha256={str(path.relative_to(JOURNAL)): sha(path) for path in sources},
                  output_sha256={path.name: sha(path) for path in sorted(OUT.glob("*.csv"))},
                  validation=checks,
                  interpretation="Signed log gain separates common attenuation from descendant selectivity; resistances and synapse-to-conductance scales remain imposed calibrations.")
    (OUT / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ["status", "paired_focal_rows", "cell_regimes", "maximum_absolute_replay_errors"]}, indent=2))


if __name__ == "__main__":
    main()
