"""Summarize every completed arm; incomplete cohorts are explicitly incomplete."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run import sha, save_new


def collect(root, output):
    protocol = json.loads((root/'protocol.json').read_text())
    expected = []
    for job in protocol['jobs']:
        for rule, rate in itertools.product(['exact', 'broadcast', 'resistance', 'swapped', 'uniform_rms'], protocol['rates']):
            expected.append((job['seed'], job['somata'], job['streams'], job['task'], job['mode'], rule, rate))
    expected = set(expected)
    rows, diagnostic_rows, found = [], [], set()
    for path in sorted((root/'results').glob('*.json')):
        result = json.loads(path.read_text())
        assert result['protocol_sha256'] == sha(root/'protocol.json'), path
        key = tuple(result[k] for k in ['seed', 'somata', 'streams', 'task', 'forward', 'rule', 'rate'])
        assert key in expected and key not in found, key
        found.add(key)
        checkpoint = root/'checkpoints'/path.with_suffix('.pt').name
        assert sha(checkpoint) == result['checkpoint_sha256'], path
        row = {k:v for k,v in result.items() if not isinstance(v, (dict, list))}
        row.update({f'ood_severity_{k}':v for k,v in result['generalization_nmse'].items()})
        rows.append(row)
        for state in ['initial', 'selected']:
            for diagnostic in result[f'diagnostic_{state}']:
                diagnostic_rows.append(dict(seed=result['seed'], somata=result['somata'], streams=result['streams'],
                    task=result['task'], forward=result['forward'], trained_rule=result['rule'], rate=result['rate'],
                    state=state, **diagnostic))
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(output/'all_endpoints.csv', index=False)
        pd.DataFrame(diagnostic_rows).to_csv(output/'common_state_diagnostics.csv', index=False)
        keys = ['somata', 'streams', 'task', 'forward', 'rule', 'rate']
        summary = frame.groupby(keys, as_index=False).agg(n=('seed', 'nunique'), test_nmse=('test_nmse', 'mean'),
                   validation_nmse=('validation_nmse', 'mean'), mean_selected_step=('best_step', 'mean'),
                   bound_runs=('bound_steps', lambda x: int((x>0).sum())))
        summary.to_csv(output/'all_rates_summary.csv', index=False)
        if found == expected:
            # Development only: rate choice is fixed across seeds and uses validation.
            best = summary.sort_values('validation_nmse').drop_duplicates(keys[:-1])
            best.to_csv(output/'development_selected_rates.csv', index=False)
            selected = frame.merge(best[keys], on=keys, validate='many_to_one')
            selected.to_csv(output/'selected_rate_endpoints.csv', index=False)
            contrasts = []
            for condition, part in selected.groupby(['somata', 'streams', 'task', 'forward']):
                wide = part.pivot(index='seed', columns='rule', values='test_nmse')
                for left in ['broadcast', 'uniform_rms', 'swapped', 'exact']:
                    difference = wide[left]-wide.resistance
                    contrasts.append(dict(zip(['somata', 'streams', 'task', 'forward'], condition),
                                          left=left, right='resistance', n=len(difference),
                                          mean_difference=float(difference.mean()), positive_seeds=int((difference>0).sum())))
            pd.DataFrame(contrasts).to_csv(output/'development_paired_contrasts.csv', index=False)
    save_new(output/'status.json', dict(phase=protocol['phase'], completed=len(found), expected=len(expected),
             complete=found==expected, missing=[list(x) for x in sorted(expected-found)],
             inference='Development summaries are descriptive; no confirmatory P values or equivalence claims',
             protocol_sha256=sha(root/'protocol.json'), report_source_sha256=sha(__file__)))
    print(json.dumps(dict(completed=len(found), expected=len(expected), complete=found==expected)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    collect(a.root, a.output)
