#!/usr/bin/env python3
"""Post-review decomposition of retained passive fields and surrogate variation.

This is descriptive reanalysis of existing arbors and perturbation settings.
No new fit, outcome selection, statistical family or task field is introduced.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[1]
OUT = J / "source_data/passive_field_diagnostics"
sys.path.insert(0, str(J / "code/reconstructed_tree"))
from run_focal_shunting_credit_perturbation import conductance_system, solve_with_soma_clamp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), J / "code/reconstructed_tree/run_focal_shunting_credit_perturbation.py",
               J / "source_data/figure3/segment_metrics.csv"]
    segments = pd.read_csv(sources[-1])
    cells = []
    for rid, seg in segments.groupby("root_id"):
        electrical, G, b, root, parents, _ = conductance_system(seg, .35, .35, 1., -.2)
        V = np.linalg.solve(G, b)
        R = np.linalg.inv(G)
        q = R[:, root]
        ids = electrical.segment_id.astype(int).tolist()
        index = {s: i for i, s in enumerate(ids)}
        eids = electrical.loc[electrical.E_size > 0, "segment_id"].astype(int).tolist()
        inds = np.array([index[i] for i in eids])
        weights = electrical.loc[electrical.E_size > 0, "E_size"].to_numpy()
        cachepath = J / "source_data/anatomy_commonmode/original8/cells" / f"operator_{rid}.npz"
        sources.append(cachepath)
        cache = np.load(cachepath)
        assert np.array_equal(cache["e_sites"], eids)
        total = drive_energy = adj_energy = resid_energy = adj_resid = cross = 0.
        sum_adj = sum_drive = sum_weight = 0.
        ncol = 0
        source_difference = 0.
        for k in electrical.loc[electrical.I_size > 0, "segment_id"].astype(int):
            ki = index[k]
            dose = .001
            delta = dose * sum(float(electrical.iloc[ki][n]) for n in ["g_leak", "g_e", "g_i"])
            Gp = G.copy(); Gp[ki, ki] += delta
            bp = b.copy(); bp[ki] -= .2 * delta
            Vp, *_ = solve_with_soma_clamp(Gp, bp, root, V[root])
            qp = q - delta * q[ki] / (1 + delta * R[ki, ki]) * R[:, ki]
            adj = np.log(qp[inds] / q[inds]) / dose
            drive = np.log((1 - Vp[inds]) / (1 - V[inds])) / dose
            full = adj + drive
            assert int(cache["i_sites"][ncol]) == k
            source_difference = max(source_difference, float(np.max(abs(full - cache["response"][:, ncol]))))
            ancestors = []
            c = k
            while True:
                ancestors.append(c)
                if c not in parents:
                    break
                c = parents[c]
            labels = []
            for e in eids:
                c = e
                while c not in ancestors:
                    c = parents[c]
                labels.append(c)
            labels = np.array(labels)
            pred = np.zeros(len(inds)); pred_adj = np.zeros(len(inds))
            for label in np.unique(labels):
                mask = labels == label
                pred[mask] = np.average(full[mask], weights=weights[mask])
                pred_adj[mask] = np.average(adj[mask], weights=weights[mask])
            total += np.sum(weights * full**2)
            adj_energy += np.sum(weights * adj**2)
            drive_energy += np.sum(weights * drive**2)
            resid_energy += np.sum(weights * (full - pred)**2)
            adj_resid += np.sum(weights * (adj - pred_adj)**2)
            cross += np.sum(weights * adj * drive)
            sum_adj += np.sum(weights * adj)
            sum_drive += np.sum(weights * drive)
            sum_weight += weights.sum()
            ncol += 1
        closure = abs(total - (adj_energy + drive_energy + 2 * cross)) / total
        assert closure < 1e-12 and source_difference < 1e-8
        covariance = cross - sum_adj * sum_drive / sum_weight
        centered_norm = np.sqrt((adj_energy - sum_adj**2 / sum_weight) *
                                (drive_energy - sum_drive**2 / sum_weight))
        cells.append(dict(root_id=int(rid), n_e_sites=len(eids), n_focal_sites=ncol,
            adjoint_energy_over_full=adj_energy / total,
            driving_force_energy_over_full=drive_energy / total,
            twice_cross_energy_over_full=2 * cross / total,
            adjoint_driving_weighted_cosine=cross / np.sqrt(adj_energy * drive_energy),
            adjoint_driving_weighted_correlation=covariance / centered_norm,
            full_partition_capture=1 - resid_energy / total,
            adjoint_partition_residual_over_full=adj_resid / total,
            energy_identity_relative_error=closure,
            max_difference_from_retained_operator=source_difference))
    field = pd.DataFrame(cells)
    field.to_csv(OUT / "passive_field_decomposition.csv", index=False)

    rows = []
    for path in sorted((J / "source_data/anatomy_commonmode/v661/cells").glob("rows_*.csv.gz")):
        sources.append(path)
        d = pd.read_csv(path)
        d = d[d.channels.eq(8)]
        actual = float(d[d.method.eq("common + ancestry")].total_capture.iloc[0])
        surrogate = d[d.method.eq("common + surrogate ancestry")].total_capture.to_numpy()
        assert len(surrogate) == 200
        rows.append(dict(root_id=int(d.root_id.iloc[0]), ancestry_capture=actual,
            surrogate_mean=float(surrogate.mean()), n_surrogates=len(surrogate),
            fraction_surrogates_match_or_exceed=float(np.mean(surrogate >= actual - 1e-12)),
            fraction_surrogates_strictly_exceed=float(np.mean(surrogate > actual + 1e-12))))
    heterogeneity = pd.DataFrame(rows)
    assert len(heterogeneity) == 47
    heterogeneity.to_csv(OUT / "surrogate_cell_heterogeneity.csv", index=False)
    report = dict(scope="Descriptive post-review reanalysis of the original passive perturbation family; no new independent biological evidence.",
        partition_definition="Complete focal-site ancestry partition differs for each shunt; not the budgeted seven-route dictionary.",
        field_definition="Finite-difference log excitatory-conductance gradient change divided by .001 dose; fixed normalized reciprocal passive tree, soma voltage restored.",
        units="Energy ratios use squared response magnitudes, not amplitudes. Weighted correlation is centered within each cell across its E-site/focal-site entries.",
        n_original_cells=len(field), n_disjoint_cells=len(heterogeneity),
        full_partition_capture_range=[float(field.full_partition_capture.min()), float(field.full_partition_capture.max())],
        driving_force_energy_over_full_range=[float(field.driving_force_energy_over_full.min()), float(field.driving_force_energy_over_full.max())],
        weighted_correlation_range=[float(field.adjoint_driving_weighted_correlation.min()), float(field.adjoint_driving_weighted_correlation.max())],
        cells_majority_surrogates_match_or_exceed=int((heterogeneity.fraction_surrogates_match_or_exceed > .5).sum()),
        numerical_tie_tolerance=1e-12,
        sources_sha256={str(p.relative_to(J)): sha(p) for p in sources})
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "README.md").write_text("# Passive-field and surrogate diagnostics\n\n"
        "This post-review reanalysis separates adjoint and driving-force terms in the original "
        "eight-cell focal-field construction. Large opposing terms can cancel while the full field "
        "remains nearly ancestry-partition constant; a small outside-partition residual does not "
        "mean the driving-force contribution is small. Energy ratios, centered weighted correlations "
        "and numerical reconstruction checks are supplied for every cell. The second table retains "
        "all47 actual-versus-200-surrogate distributions atK8. These are diagnostics of existing "
        "arbors and modeled fields, not new biological samples. Inputs and normalization are in "
        "report.json; regenerate with scripts/passive_field_diagnostics.py.\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources_sha256"}, indent=2))


if __name__ == "__main__":
    main()
