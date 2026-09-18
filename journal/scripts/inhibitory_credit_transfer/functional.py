"""Exploratory measured-response screen of anatomically defined inhibitory domains.

Split-repeat covariance avoids squaring the same measurement noise. These are
sensory responses, not gradients, errors, contexts or inhibitory recordings.
Report target-level descriptive associations, not independent-pair P values.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from run import save_new, sha


def cross_covariance(left, right):
    left = left-left.mean(0)
    right = right-right.mean(0)
    scale = np.sqrt((left.var(0, ddof=1)+right.var(0, ddof=1))/2).clip(1e-12)
    left, right = left/scale, right/scale
    return (left.T@right + right.T@left)/(2*(len(left)-1))


def partial_rank(x, y, covariates):
    design = np.column_stack([np.ones(len(x)), *[rankdata(c) for c in covariates.T]])
    xx, yy = rankdata(x), rankdata(y)
    xx -= design@np.linalg.lstsq(design, xx, rcond=None)[0]
    yy -= design@np.linalg.lstsq(design, yy, rcond=None)[0]
    norm = np.linalg.norm(xx)*np.linalg.norm(yy)
    return float(xx@yy/norm) if norm > 1e-12 else None


def analyze(root):
    protocol = json.loads((root/'protocol.json').read_text())
    for relative, expected in protocol['input_sha256'].items():
        if relative.startswith('inputs/functional') or relative == 'inputs/microns_pilot.csv':
            assert sha(root/relative) == expected
    source = root/'inputs/functional'
    scans = pd.read_csv(source/'scan_index.csv')
    segments = pd.read_csv(root/'inputs/microns_pilot.csv')
    rows, exclusions = [], []
    for scan in scans.itertuples(index=False):
        target = int(scan.target_root_id)
        cell = segments[segments.root_id.eq(target)]
        parents = dict(zip(cell.segment_id.astype(int), cell.parent_segment_id.astype(int)))
        contacts = pd.read_csv(source/scan.scan/'functional_contacts.csv')
        response = np.load(source/scan.scan/'observed_partner_responses.npz', allow_pickle=False)
        assert np.array_equal(response['partner_root_ids'], contacts.pre_pt_root_id.to_numpy(np.int64))
        covariance = cross_covariance(response['half_left'], response['half_right'])
        chains = []
        for segment in contacts.segment_id:
            chain = set()
            cursor = int(segment)
            while cursor >= 0 and cursor not in chain:
                chain.add(cursor); cursor = parents.get(cursor, -1)
            chains.append(chain)
        eligible = 0
        for site in cell[cell.I_size.gt(0)].itertuples(index=False):
            mask = np.array([int(site.segment_id) in c for c in chains])
            inside, outside = np.flatnonzero(mask), np.flatnonzero(~mask)
            if min(len(inside), len(outside)) < 3:
                continue
            within = covariance[np.ix_(inside, inside)]
            within = within[np.triu_indices(len(inside), k=1)]
            between = covariance[np.ix_(inside, outside)]
            rows.append(dict(target_root_id=target, scan=scan.scan, segment_id=int(site.segment_id),
                             inhibitory_size=float(site.I_size), excitatory_size=float(site.E_size),
                             path_um=float(site.path_length_um), n_inside=len(inside), n_outside=len(outside),
                             within_covariance=float(within.mean()), between_covariance=float(between.mean()),
                             within_minus_between=float(within.mean()-between.mean()),
                             median_repeat_reliability=float(contacts.repeat_reliability.median())))
            eligible += 1
        if not eligible:
            exclusions.append(dict(scan=scan.scan, reason='Fewer than three mapped partners on one side of every inhibitory domain'))
    frame = pd.DataFrame(rows)
    if frame.empty:
        save_new(root/'functional_results/audit.json', dict(status='No eligible domains', exclusions=exclusions))
        return
    # Equal weighting over available scans for each anatomical site; no scan
    # counted as another target, and no site/pair counted as another animal.
    site = frame.groupby(['target_root_id', 'segment_id'], as_index=False).mean(numeric_only=True)
    targets = []
    for target, part in site.groupby('target_root_id'):
        valid = len(part) >= 8
        effect = partial_rank(part.inhibitory_size.to_numpy(), part.within_minus_between.to_numpy(),
                              part[['path_um', 'n_inside', 'n_outside', 'excitatory_size']].to_numpy()) if valid else None
        targets.append(dict(target_root_id=int(target), n_domains=len(part),
                            partial_rank_inhibitory_load_vs_segregation=effect,
                            mean_within_minus_between=float(part.within_minus_between.mean()),
                            status='descriptive' if valid else 'too few domains for adjusted association'))
    out = root/'functional_results'
    for name, table in [('scan_domains', frame), ('site_domains', site), ('targets', pd.DataFrame(targets))]:
        with (out/f'{name}.csv').open('x') as handle:
            table.to_csv(handle, index=False)
    save_new(out/'audit.json', dict(protocol_sha256=sha(root/'protocol.json'),
                                   targets=len(targets), animals=1, exclusions=exclusions,
                                   interpretation='Descriptive anatomical-functional association; no measured teaching signals or causal inhibition',
                                   statistical_scope='Nested and overlapping domains; no independent-site/pair tests; no claim of animal-population replication',
                                   targets_summary=targets))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    analyze(p.parse_args().root)
