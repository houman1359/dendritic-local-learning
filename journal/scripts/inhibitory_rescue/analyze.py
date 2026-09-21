"""Replay every fresh endpoint, then report every rule and sensitivity arm."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rescue import RescueNet, dataset
from run import evaluate, sha


def replay(item):
    root, job = item
    torch.set_num_threads(1)
    key = f"s{job['seed']}_{job['optimizer']}_b{job['bound']}_{job['rule']}_r{job['rate']:g}"
    path = root / 'fresh/results' / f'{key}.json'
    r = json.loads(path.read_text())
    checkpoint = root / 'fresh/checkpoints' / f'{key}.pt'
    assert r['protocol_sha256'] == sha(root / 'fresh_protocol.json')
    assert r['checkpoint_sha256'] == sha(checkpoint)
    assert all(r[k] == v for k, v in job.items())
    net = RescueNet(job['seed']).double()
    states = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net.load_state_dict(states['selected'])
    observed = dict(test_nmse=evaluate(net, dataset(job['seed'], 'test', 4096, 'interaction')),
                    validation_nmse=evaluate(net, dataset(job['seed'], 'validation', 1024, 'interaction')))
    errors = [abs(observed[k] - r[k]) for k in observed]
    for s in [1., 2., 3.]:
        errors.append(abs(evaluate(net, dataset(job['seed'], 'ood', 4096, 'interaction', s)) - r['ood_nmse'][str(s)]))
    assert max(errors) < 1e-10, (key, errors)
    net.load_state_dict(states['endpoint'])
    error = abs(evaluate(net, dataset(job['seed'], 'validation', 1024, 'interaction')) - r['endpoint_validation_nmse'])
    assert error < 1e-10, (key, error)
    del net, states
    gc.collect()
    return key, r, max(errors + [error]), sha(path)


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci_low=float(low), ci_high=float(high), n=len(values))


def signflip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.sum())
    count = 0
    for start in range(0, 1 << len(values), 65536):
        patterns = np.arange(start, min(start + 65536, 1 << len(values)), dtype=np.uint64)
        signs = 2 * ((patterns[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float) - 1
        count += int((np.abs(signs @ values) >= observed - 1e-12).sum())
    return count / (1 << len(values))


def analyze(root, workers):
    protocol = json.loads((root / 'fresh_protocol.json').read_text())
    for relative, expected in protocol['source_sha256'].items():
        assert sha(root / relative) == expected, relative
    assert len(list((root / 'fresh/results').glob('*.json'))) == len(protocol['jobs']) == 320
    assert not set(protocol['development_seeds']) & set(protocol['fresh_seeds'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        replayed = list(pool.map(replay, [(root, job) for job in protocol['jobs']]))
    rows, diagnostics, hashes = [], [], {}
    for key, result, error, digest in replayed:
        hashes[key] = digest
        row = {k: result[k] for k in ['seed', 'rule', 'optimizer', 'bound', 'rate', 'selected_step',
                                      'test_nmse', 'validation_nmse', 'endpoint_validation_nmse',
                                      'bounds', 'clips', 'first_bound_step', 'selected_bound_fraction']}
        row.update({f'ood_{s}': value for s, value in result['ood_nmse'].items()})
        rows.append(row)
        for state in ['initial', 'selected']:
            for d in result[f'{state}_diagnostics']:
                diagnostics.append(dict(seed=result['seed'], optimizer=result['optimizer'], bound=result['bound'],
                                        trained_rule=result['rule'], state=state, **d))
    endpoints = pd.DataFrame(rows)
    rng = np.random.default_rng(9202026)
    summaries = []
    for (optimizer, bound, rule), group in endpoints.groupby(['optimizer', 'bound', 'rule']):
        assert set(group.seed) == set(protocol['fresh_seeds'])
        for metric in ['test_nmse', 'ood_1.0', 'ood_2.0', 'ood_3.0']:
            summaries.append(dict(optimizer=optimizer, bound=bound, rule=rule, metric=metric,
                                  bound_runs=int((group.bounds > 0).sum()), **interval(group[metric], rng)))
    contrasts = []
    for (optimizer, bound), group in endpoints.groupby(['optimizer', 'bound']):
        for metric in ['test_nmse', 'ood_3.0']:
            wide = group.pivot(index='seed', columns='rule', values=metric).sort_index()
            for left in ['resistance', 'shuffled_derivative', 'broadcast', 'exact']:
                if left not in wide:
                    continue
                difference = wide[left] - wide['derivative']
                primary = optimizer == 'adam' and bound == 9 and metric == 'test_nmse' and left in ['resistance', 'shuffled_derivative']
                contrasts.append(dict(optimizer=optimizer, bound=bound, metric=metric, left=left, right='derivative',
                                      primary=primary, positive=int((difference > 0).sum()),
                                      signflip_p=signflip(difference) if primary else np.nan,
                                      **interval(difference, rng)))
    contrasts = pd.DataFrame(contrasts)
    primary_indices = contrasts[contrasts.primary].sort_values('signflip_p').index
    running = 0.
    for rank, index in enumerate(primary_indices):
        running = max(running, min(1., (len(primary_indices) - rank) * contrasts.loc[index, 'signflip_p']))
        contrasts.loc[index, 'holm_p'] = running
    directory = root / 'reporting_v1'
    directory.mkdir(exist_ok=False)
    tables = dict(endpoints=endpoints, summary=pd.DataFrame(summaries), contrasts=contrasts,
                  diagnostics=pd.DataFrame(diagnostics))
    for name, frame in tables.items():
        frame.to_csv(directory / f'inhibitory_rescue_{name}.csv', index=False)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
                    maximum_replay_error=max(x[2] for x in replayed), results_sha256=hashes,
                    analysis_source_sha256=sha(__file__),
                    outputs={name: sha(directory / f'inhibitory_rescue_{name}.csv') for name in tables})
    with (directory / 'inhibitory_rescue_provenance.json').open('x') as handle:
        json.dump(metadata, handle, indent=2)
    print(tables['summary'].query("metric == 'test_nmse'").to_string(index=False))
    print(contrasts[contrasts.primary].to_string(index=False))
    print('Maximum replay discrepancy:', metadata['maximum_replay_error'])
    print('Saved:', directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    analyze(args.root, args.workers)
