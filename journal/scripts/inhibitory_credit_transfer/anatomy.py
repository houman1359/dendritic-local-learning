"""Transport/voltage separation on existing real arbors, using modeled fields.

No measured error or learning is inferred. Source arbors are reused without
new outcome-dependent sampling; MICrONS cohorts share one animal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def changes(base, changed, indices):
    tiny = 1e-15 * max(float(np.max(np.abs(base))), 1.)
    log = np.log((np.abs(changed[indices])+tiny)/(np.abs(base[indices])+tiny))
    return float(np.median(log)), float(np.median(np.abs(log)))


def analyze(root, cohort):
    sys.path.insert(0, str(root/'anatomy_code/scripts'))
    sys.path.insert(0, str(root/'anatomy_code/code/reconstructed_tree'))
    from analyze_physical_cable_sensitivity import physical_conductance_system
    from run_focal_shunting_credit_perturbation import choose_focal_sites, stable_rng
    from run import save_new, sha
    protocol = json.loads((root/'protocol.json').read_text())
    source = root/'inputs'/f'{cohort}.csv'
    assert sha(source) == protocol['input_sha256'][str(source.relative_to(root))]
    for relative, expected in protocol['source_sha256'].items():
        if relative.startswith(('study/', 'anatomy_code/')):
            assert sha(root/relative) == expected
    segments = pd.read_csv(source)
    rows, exclusions = [], []
    if 'qc_included' in segments:
        qc = segments.qc_included.astype(str).str.lower().eq('true')
        for cell_id in segments.loc[~qc, 'root_id'].unique():
            exclusions.append(dict(root_id=int(cell_id), reason='Inherited anatomical cohort QC exclusion'))
        segments = segments.loc[qc].copy()
    for cell_id, cell in segments.groupby('root_id', sort=True):
        for rm in protocol['anatomy']['membrane_resistance']:
            electrical, matrix, rhs, soma, parents, children = physical_conductance_system(
                cell, .35, .35, 1., -.2, axial_resistivity_ohm_cm=150., membrane_resistance_ohm_cm2=rm)
            solver = splu(csc_matrix(matrix))
            voltage = solver.solve(rhs)
            source_soma = np.zeros(len(rhs)); source_soma[soma] = 1.
            q = solver.solve(source_soma)
            e_ids = electrical.loc[electrical.E_size.gt(0), 'segment_id'].astype(int).tolist()
            ids = electrical.segment_id.astype(int).tolist()
            index = {v:k for k,v in enumerate(ids)}
            e_indices = np.array([index[v] for v in e_ids])
            sites, relations = choose_focal_sites(electrical, e_ids, parents, children,
                                                  stable_rng(9182026, int(cell_id), 73), 16, 3)
            if not sites:
                exclusions.append(dict(root_id=int(cell_id), rm=rm, reason='No site with >=3 descendant and depth-matched comparison contacts'))
            for site in sites:
                k = index[site]
                unit = np.zeros(len(rhs)); unit[k] = 1.
                column = solver.solve(unit)
                rin = column[k]
                local = electrical.loc[k, ['g_leak', 'g_e', 'g_i']].sum()
                descendant = e_indices[relations[site]['descendant']]
                comparison = e_indices[relations[site]['depth-matched unrelated']]
                for normalized_dose in protocol['anatomy']['doses']:
                    eta = normalized_dose / rin
                    qnew = q - eta * column * q[k] / (1 + eta * rin)
                    vfree = voltage + eta * (-.2-voltage[k]) * column / (1 + eta * rin)
                    vnew = vfree + (voltage[soma]-vfree[soma]) / qnew[soma] * qnew
                    vcurrent = voltage + eta * (-.2-voltage[k]) * column
                    vcurrent += (voltage[soma]-vcurrent[soma]) / q[soma] * q
                    clamp_error = abs(vnew[soma]-voltage[soma])
                    assert clamp_error < 1e-9
                    baseline = q * (1-voltage)
                    fields = dict(full_shunt=qnew*(1-vnew), transport_only=qnew*(1-voltage),
                                  driving_force_only=q*(1-vnew), matched_current=q*(1-vcurrent))
                    # Test the coarse resistance-gate approximation on unit-error
                    # transport, weighted by baseline sensitivity (not contacts).
                    coarse = q.copy()
                    coarse[descendant] /= 1 + normalized_dose
                    transport_nmse = float(np.sum((coarse[e_indices]-qnew[e_indices])**2)/max(np.sum(qnew[e_indices]**2), 1e-30))
                    for mode, field in fields.items():
                        ds, da = changes(baseline, field, descendant)
                        cs, ca = changes(baseline, field, comparison)
                        rows.append(dict(cohort=cohort, animal='pinky' if cohort=='pinky' else 'microns',
                                         root_id=int(cell_id), focal_segment_id=site, rm=rm,
                                         input_normalized_dose=normalized_dose, dose_ns=float(eta),
                                         dose_relative_local=float(eta/local), input_resistance=float(rin), mode=mode,
                                         signed_descendant_log_change=ds, signed_comparison_log_change=cs,
                                         signed_localization=ds-cs, magnitude_localization=da-ca,
                                         soma_restore_error=float(clamp_error), coarse_transport_nmse=transport_nmse,
                                         n_descendants=len(descendant), n_comparison=len(comparison)))
        print(cohort, int(cell_id), len(rows), flush=True)
    out = root/'anatomy_results'
    frame = pd.DataFrame(rows)
    path = out/f'{cohort}_sites.csv'
    with path.open('x') as handle:
        frame.to_csv(handle, index=False)
    keys = ['cohort', 'animal', 'root_id', 'rm', 'input_normalized_dose', 'mode']
    metrics = ['signed_localization', 'magnitude_localization', 'coarse_transport_nmse', 'dose_relative_local']
    cellmeans = frame.groupby(keys, as_index=False)[metrics].mean()
    with (out/f'{cohort}_cells.csv').open('x') as handle:
        cellmeans.to_csv(handle, index=False)
    save_new(out/f'{cohort}_audit.json', dict(protocol_sha256=sha(root/'protocol.json'),
             source_sha256=sha(source), cells=int(frame.root_id.nunique()), site_rows=len(frame),
             exclusions=exclusions, max_soma_restore_error=float(frame.soma_restore_error.max()),
             interpretation='Modeled interventions on observed arbors; not measured inhibition or plasticity',
             dose_warning='Input-resistance-normalized doses can require large local conductances; relative-local doses are retained'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cohort', choices=['microns_pilot', 'microns_replication', 'pinky'], required=True)
    a = p.parse_args()
    analyze(a.root, a.cohort)
