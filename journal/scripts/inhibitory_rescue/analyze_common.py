"""Verify and join common-rate fits with reused primary-cohort trajectories."""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import interval, replay
from run import sha


def analyze(root, primary, workers):
    protocol = json.loads((root / 'fresh_protocol.json').read_text())
    assert sha(primary / 'fresh_protocol.json') == protocol['primary_protocol_sha256']
    for relative, digest in protocol['source_sha256'].items():
        assert sha(root / relative) == digest, relative
    assert len(list((root / 'fresh/results').glob('*.json'))) == len(protocol['jobs'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(replay, [(root, job) for job in protocol['jobs']]))
    columns = ['seed', 'rule', 'rate', 'test_nmse', 'selected_step', 'bounds']
    extra = pd.DataFrame([{**{key: r[key] for key in columns}, 'ood_3.0': r['ood_nmse']['3.0']}
                          for _, r, _, _ in verified])
    original = pd.read_csv(primary / 'reporting_v1/inhibitory_rescue_endpoints.csv')
    original = original[(original.optimizer == 'adam') & (original.bound == 9)
                        & (original.rate == protocol['common_rate'])]
    joined = pd.concat([extra, original[extra.columns]], ignore_index=True).sort_values(['seed', 'rule'])
    assert len(joined) == 120 and joined.seed.nunique() == 20
    assert not joined.duplicated(['seed', 'rule']).any()
    rng = np.random.default_rng(9202027)
    contrasts = []
    for metric in ['test_nmse', 'ood_3.0']:
        wide = joined.pivot(index='seed', columns='rule', values=metric)
        for left in ['resistance', 'shuffled_derivative', 'broadcast', 'exact']:
            difference = wide[left] - wide['derivative']
            contrasts.append(dict(metric=metric, left=left, right='derivative',
                                  positive=int((difference > 0).sum()), **interval(difference, rng)))
    contrasts = pd.DataFrame(contrasts)
    out = root / 'reporting_v1'
    out.mkdir(exist_ok=False)
    joined.to_csv(out / 'inhibitory_rescue_common_endpoints.csv', index=False)
    contrasts.to_csv(out / 'inhibitory_rescue_common_contrasts.csv', index=False)
    meta = dict(protocol=protocol, primary_endpoints_sha256=sha(primary / 'reporting_v1/inhibitory_rescue_endpoints.csv'),
                analysis_source_sha256=sha(__file__), maximum_replay_error=max(r[2] for r in verified),
                results_sha256={key: digest for key, _, _, digest in verified},
                outputs={p.name: sha(p) for p in out.glob('*.csv')})
    with (out / 'inhibitory_rescue_common_provenance.json').open('x') as handle:
        json.dump(meta, handle, indent=2)
    print(joined.groupby('rule')[['test_nmse', 'ood_3.0']].mean().to_string())
    print(contrasts.to_string(index=False))
    print('Maximum replay error:', meta['maximum_replay_error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    analyze(args.root, args.primary, args.workers)
